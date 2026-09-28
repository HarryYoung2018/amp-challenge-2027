"""Producer-independent verifier for categorical-diffusion post-hoc bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FROZEN_CONFIG_SHA256 = "d230c740ff5c3636a790eea2262bb09129c7197119c6c8a6f165366d68c226e4"
EXPECTED_FILES = frozenset(
    {
        "SHA256SUMS",
        "adapter-contract.toml",
        "attempts.jsonl",
        "controller-composition.jsonl",
        "manifest.json",
        "pool-context.json",
        "posterior-snapshot.json",
        "ranking.jsonl",
        "resolved-pool.jsonl",
        "scores.jsonl",
    }
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_RAW_OUTPUT_CHARACTERS = 256
MAX_FILE_BYTES = {
    "SHA256SUMS": 4 * 1024,
    "adapter-contract.toml": 64 * 1024,
    "attempts.jsonl": 32 * 1024 * 1024,
    "controller-composition.jsonl": 2 * 1024 * 1024,
    "manifest.json": 256 * 1024,
    "pool-context.json": 4 * 1024 * 1024,
    "posterior-snapshot.json": 64 * 1024,
    "ranking.jsonl": 8 * 1024 * 1024,
    "resolved-pool.jsonl": 8 * 1024 * 1024,
    "scores.jsonl": 8 * 1024 * 1024,
}
MAX_JSONL_LINE_BYTES = {
    "attempts.jsonl": 2 * 1024,
    "controller-composition.jsonl": 512,
    "ranking.jsonl": 1024,
    "resolved-pool.jsonl": 1024,
    "scores.jsonl": 1024,
}
MAX_JSONL_ROWS = {
    "attempts.jsonl": 16_384,
    "controller-composition.jsonl": 448,
    "ranking.jsonl": 2_048,
    "resolved-pool.jsonl": 2_048,
    "scores.jsonl": 2_048,
}
MAX_EXCLUSION_IDENTITIES = 16_384
MAX_SIGNED_63 = (1 << 63) - 1
MAX_PATH_CHARACTERS = 4_096
MAX_RAW_OUTPUT_UTF8_BYTES = 256
MAX_BUNDLE_FILE_NAME_CHARACTERS = 64


class PosthocVerificationError(ValueError):
    """Raised when a purported engineering bundle cannot be reconstructed."""


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Structural verification only; external identities remain unauthenticated."""

    bundle_sha256: str
    attempt_count: int
    resolved_candidate_count: int
    ranked_candidate_count: int
    method_seat_count: int
    controller_seat_count: int
    organizer_reference_present: bool = False
    execution_authorized: bool = False
    oracle_calls_authorized: bool = False
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False
    structural_validation_only: bool = True
    external_inputs_authenticated: bool = False
    common_reserve_authority_accepted: bool = False
    charged_call_evidence_present: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.bundle_sha256) is not str
            or len(self.bundle_sha256) != 64
            or _SHA_RE.fullmatch(self.bundle_sha256) is None
        ):
            raise PosthocVerificationError("verification receipt bundle identity is invalid")
        exact_counts = (
            (self.attempt_count, 16_384),
            (self.method_seat_count, 392),
            (self.controller_seat_count, 448),
        )
        if any(type(value) is not int or value != expected for value, expected in exact_counts):
            raise PosthocVerificationError("verification receipt exact counts are invalid")
        if (
            type(self.resolved_candidate_count) is not int
            or not 392 <= self.resolved_candidate_count <= 2_048
            or type(self.ranked_candidate_count) is not int
            or self.ranked_candidate_count != self.resolved_candidate_count
        ):
            raise PosthocVerificationError("verification receipt candidate counts are invalid")
        for value in (
            self.organizer_reference_present,
            self.execution_authorized,
            self.oracle_calls_authorized,
            self.scientific_evidence_accepted,
            self.production_input_eligible,
            self.external_inputs_authenticated,
            self.common_reserve_authority_accepted,
            self.charged_call_evidence_present,
        ):
            if value is not False:
                raise PosthocVerificationError(
                    "verification receipt authority flags must remain false"
                )
        if self.structural_validation_only is not True:
            raise PosthocVerificationError("verification receipt must remain structural-only")


@dataclass(frozen=True, slots=True)
class _HeldPayload:
    """One captured payload whose descriptor remains held through verification."""

    name: str
    file_descriptor: int
    initial_stat: os.stat_result
    payload: bytes


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise PosthocVerificationError(f"nonfinite JSON number: {value}")


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PosthocVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(payload: bytes, name: str) -> Any:
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PosthocVerificationError(f"invalid JSON in {name}") from exc
    if _canonical(value) != payload:
        raise PosthocVerificationError(f"noncanonical JSON in {name}")
    return value


def _jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    line_cap = MAX_JSONL_LINE_BYTES[name]
    row_cap = MAX_JSONL_ROWS[name]
    result: list[dict[str, Any]] = []
    offset = 0
    while offset < len(payload):
        newline = payload.find(b"\n", offset, min(len(payload), offset + line_cap))
        if newline < 0:
            raise PosthocVerificationError(f"{name} line exceeds cap or lacks final newline")
        number = len(result) + 1
        if number > row_cap:
            raise PosthocVerificationError(f"{name} exceeds its row cap")
        line = payload[offset : newline + 1]
        value = _json(line, f"{name}:{number}")
        if type(value) is not dict:
            raise PosthocVerificationError(f"{name}:{number} must be an object")
        result.append(value)
        offset = newline + 1
    return result


def _canonical(value: object) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PosthocVerificationError("value is not canonicalizable") from exc
    return (encoded + "\n").encode()


def _require_sha(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or _SHA_RE.fullmatch(value) is None:
        raise PosthocVerificationError(f"{field} is not a lowercase SHA-256")
    return value


def _require_exact_keys(row: Mapping[str, Any], keys: set[str], where: str) -> None:
    if len(row) != len(keys):
        raise PosthocVerificationError(f"{where} field count differs")
    actual = set(row)
    if actual != keys:
        raise PosthocVerificationError(f"{where} fields differ: {sorted(actual ^ keys)}")


def _canonical_sequence(raw: object) -> tuple[str, str] | None:
    if not _raw_output_within_caps(raw):
        return None
    sequence = "".join(raw.split()).upper()
    if not 8 <= len(sequence) <= 50 or not set(sequence) <= _ALPHABET:
        return None
    return sequence, hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _raw_output_within_caps(raw: object) -> bool:
    if type(raw) is not str or len(raw) > MAX_RAW_OUTPUT_CHARACTERS:
        return False
    try:
        return len(raw.encode("utf-8")) <= MAX_RAW_OUTPUT_UTF8_BYTES
    except UnicodeEncodeError:
        return False


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
    try:
        midpoint = float(first) * 0.5 + float(second) * 0.5
    except (OverflowError, ValueError) as exc:
        raise PosthocVerificationError("derived mean utility is nonfinite") from exc
    if not math.isfinite(midpoint):
        raise PosthocVerificationError("derived mean utility is nonfinite")
    return midpoint


def _verify_checksums(payloads: Mapping[str, bytes]) -> None:
    expected = "".join(
        f"{_sha(payloads[name])}  {name}\n" for name in sorted(EXPECTED_FILES - {"SHA256SUMS"})
    ).encode("ascii")
    if payloads["SHA256SUMS"] != expected:
        raise PosthocVerificationError("SHA256SUMS mismatch")


def _verify_contract(payload: bytes) -> None:
    if _sha(payload) != FROZEN_CONFIG_SHA256:
        raise PosthocVerificationError("adapter contract hash mismatch")
    raw = tomllib.loads(payload.decode())
    if (
        raw.get("accepted_policy_sets") != []
        or raw.get("identity", {}).get("accepted_policy_count") != 0
    ):
        raise PosthocVerificationError("accepted policy inventory is not empty")
    flags = (
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
    if any(raw.get(field) is not False for field in flags):
        raise PosthocVerificationError("adapter contract contains authorization")
    compliance = raw.get("compliance")
    generation = raw.get("generation")
    resources = raw.get("resources")
    if (
        raw.get("charged_call_output") != "unauthenticated_structural_claim_only"
        or type(compliance) is not dict
        or compliance.get("organizer_reference_role")
        != "unauthenticated_post_generation_structural_claim_only"
        or compliance.get("compliance_claim_may_veto_later_shadow_or_production") is not False
        or compliance.get("maximum_reason_characters") != 256
        or compliance.get("maximum_claim_bytes") != 2 * 1024 * 1024
        or type(generation) is not dict
        or generation.get("maximum_raw_output_utf8_bytes") != MAX_RAW_OUTPUT_UTF8_BYTES
        or generation.get("maximum_attempt_jsonl_line_bytes")
        != MAX_JSONL_LINE_BYTES["attempts.jsonl"]
        or type(resources) is not dict
        or resources.get("maximum_adapter_config_bytes") != 64 * 1024
        or resources.get("maximum_path_characters") != MAX_PATH_CHARACTERS
        or resources.get("maximum_nonnegative_signed63") != MAX_SIGNED_63
    ):
        raise PosthocVerificationError("unauthenticated structural-claim boundary changed")


def _verify_attempts_and_pool(
    attempts: list[dict[str, Any]], slots: list[dict[str, Any]], context: dict[str, Any]
) -> tuple[str, int]:
    if len(attempts) != 16_384 or len(slots) != 2_048:
        raise PosthocVerificationError("grid must contain 16,384 attempts and 2,048 slots")
    attempt_keys = {
        "slot",
        "attempt",
        "source_ordinal",
        "request_sha256",
        "raw_sequence",
        "policy_member_index",
        "policy_member_sha256",
        "sample_seed",
        "sample_counter",
        "trajectory_receipt_sha256",
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
    }
    slot_keys = {"slot", "selected_source_ordinal", "sequence", "sequence_id", "rejection"}
    context_keys = {
        "request_sha256",
        "initial64_snapshot_sha256",
        "exact_training_overlap_ids",
        "accepted_training_homology_ids",
        "pool_seal_sha256",
    }
    _require_exact_keys(context, context_keys, "pool context")
    request_sha = _require_sha(context["request_sha256"], "request_sha256")
    _require_sha(context["initial64_snapshot_sha256"], "initial64_snapshot_sha256")
    training = context["exact_training_overlap_ids"]
    homology = context["accepted_training_homology_ids"]
    if (
        type(training) is not list
        or type(homology) is not list
        or len(training) > MAX_EXCLUSION_IDENTITIES
        or len(homology) > MAX_EXCLUSION_IDENTITIES
    ):
        raise PosthocVerificationError("exclusion identities exceed their bounded list contract")
    validated_sets: list[set[str]] = []
    for field, values in (("training", training), ("homology", homology)):
        validated: set[str] = set()
        for value in values:
            validated.add(_require_sha(value, f"{field} identity"))
        if values != sorted(validated):
            raise PosthocVerificationError("exclusion identities must be sorted unique lists")
        validated_sets.append(validated)
    training_set, homology_set = validated_sets
    excluded = training_set | homology_set
    policy_members: dict[int, str] = {}
    trajectory_receipts: set[str] = set()
    sample_seed = attempts[0].get("sample_seed")
    for ordinal, row in enumerate(attempts):
        _require_exact_keys(row, attempt_keys, f"attempt {ordinal}")
        if (
            type(row["slot"]) is not int
            or type(row["attempt"]) is not int
            or type(row["source_ordinal"]) is not int
            or row["slot"] != ordinal // 8
            or row["attempt"] != ordinal % 8
            or row["source_ordinal"] != ordinal
            or row["request_sha256"] != request_sha
            or type(row["policy_member_index"]) is not int
            or row["policy_member_index"] != ordinal % 10
            or type(row["sample_seed"]) is not int
            or not 0 <= row["sample_seed"] <= MAX_SIGNED_63
            or row["sample_seed"] != sample_seed
            or type(row["sample_counter"]) is not int
            or row["sample_counter"] != ordinal
            or not _raw_output_within_caps(row["raw_sequence"])
        ):
            raise PosthocVerificationError("attempt order, ordinal, or request was changed")
        member_sha = _require_sha(row["policy_member_sha256"], "policy member identity")
        if policy_members.setdefault(row["policy_member_index"], member_sha) != member_sha:
            raise PosthocVerificationError("policy member identity changed within schedule")
        receipt_sha = _require_sha(row["trajectory_receipt_sha256"], "trajectory receipt")
        if receipt_sha in trajectory_receipts:
            raise PosthocVerificationError("trajectory receipt is not unique per output")
        trajectory_receipts.add(receipt_sha)
        for flag in (
            "execution_authorized",
            "oracle_calls_authorized",
            "scientific_evidence_accepted",
            "production_input_eligible",
        ):
            if row[flag] is not False:
                raise PosthocVerificationError("attempt contains false authorization")
    if set(policy_members) != set(range(10)) or len(set(policy_members.values())) != 10:
        raise PosthocVerificationError("grid does not bind ten distinct policy members")
    accepted: set[str] = set()
    recomputed: list[dict[str, Any]] = []
    for slot in range(2_048):
        winner: tuple[int, str, str] | None = None
        last_reason = "attempt_budget_exhausted"
        for row in attempts[slot * 8 : (slot + 1) * 8]:
            parsed = _canonical_sequence(row["raw_sequence"])
            if parsed is None:
                last_reason = "canonical_support_rejection"
                continue
            sequence, sequence_id = parsed
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
            winner = (row["source_ordinal"], sequence, sequence_id)
            accepted.add(sequence_id)
            break
        recomputed.append(
            {
                "slot": slot,
                "selected_source_ordinal": None if winner is None else winner[0],
                "sequence": None if winner is None else winner[1],
                "sequence_id": None if winner is None else winner[2],
                "rejection": last_reason if winner is None else None,
            }
        )
        _require_exact_keys(slots[slot], slot_keys, f"resolved slot {slot}")
        if type(slots[slot]["slot"]) is not int or (
            slots[slot]["selected_source_ordinal"] is not None
            and type(slots[slot]["selected_source_ordinal"]) is not int
        ):
            raise PosthocVerificationError("resolved slot contains non-exact integer fields")
    if slots != recomputed:
        raise PosthocVerificationError("resolved pool does not reconstruct from fixed attempts")
    core = {
        "request_sha256": request_sha,
        "initial64_snapshot_sha256": context["initial64_snapshot_sha256"],
        "attempts": attempts,
        "slots": slots,
        "exact_training_overlap_ids": training,
        "accepted_training_homology_ids": homology,
    }
    pool_seal = _sha(_canonical(core))
    if context["pool_seal_sha256"] != pool_seal:
        raise PosthocVerificationError("pool seal mismatch")
    return pool_seal, len(accepted)


def _verify_snapshot(snapshot: dict[str, Any], context: dict[str, Any]) -> tuple[str, float]:
    keys = {
        "snapshot_sha256",
        "initial_archive_sha256",
        "posterior_model_sha256",
        "response_count",
        "adaptive_response_count",
        "joint_chance_threshold",
        "covariance_dimension",
        "covariance_rank",
        "affine_support_validated",
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
    }
    _require_exact_keys(snapshot, keys, "posterior snapshot")
    for field in ("snapshot_sha256", "initial_archive_sha256", "posterior_model_sha256"):
        _require_sha(snapshot[field], field)
    if snapshot["snapshot_sha256"] != context["initial64_snapshot_sha256"]:
        raise PosthocVerificationError("snapshot identity does not match generation seal")
    if (
        type(snapshot["response_count"]) is not int
        or type(snapshot["adaptive_response_count"]) is not int
        or snapshot["response_count"] != 64
        or snapshot["adaptive_response_count"] != 0
    ):
        raise PosthocVerificationError("posterior snapshot includes adaptive responses")
    threshold = snapshot["joint_chance_threshold"]
    if not _is_finite_number(threshold) or not 0 <= threshold <= 1:
        raise PosthocVerificationError("invalid chance threshold")
    dimension, rank = snapshot["covariance_dimension"], snapshot["covariance_rank"]
    if (
        type(dimension) is not int
        or type(rank) is not int
        or not 1 <= dimension <= MAX_SIGNED_63
        or not 0 <= rank <= MAX_SIGNED_63
        or rank > dimension
    ):
        raise PosthocVerificationError("invalid covariance shape/rank")
    if snapshot["affine_support_validated"] is not True:
        raise PosthocVerificationError(
            "degenerate covariance lacks a structural affine-support claim"
        )
    for flag in (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
    ):
        if snapshot[flag] is not False:
            raise PosthocVerificationError("snapshot contains false authorization")
    return snapshot["snapshot_sha256"], float(threshold)


def _verify_ranking(
    scores: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    snapshot_sha: str,
    threshold: float,
    pool_seal: str,
    method_stream: object,
) -> tuple[str, int]:
    score_keys = {
        "sequence_id",
        "source_ordinal",
        "snapshot_sha256",
        "gram_positive_mean",
        "gram_negative_mean",
        "joint_chance_probability",
    }
    rank_keys = {
        "rank",
        "sequence_id",
        "source_ordinal",
        "chance_feasible",
        "mean_utility",
        "joint_chance_probability",
    }
    candidates = {row["sequence_id"]: row for row in slots if row["sequence_id"] is not None}
    if len(scores) != len(candidates):
        raise PosthocVerificationError("score count does not match resolved pool")
    seen: set[str] = set()
    for index, score in enumerate(scores):
        _require_exact_keys(score, score_keys, f"score {index}")
        sid = _require_sha(score["sequence_id"], "score sequence_id")
        if (
            type(score["source_ordinal"]) is not int
            or sid not in candidates
            or sid in seen
            or score["source_ordinal"] != candidates[sid]["selected_source_ordinal"]
        ):
            raise PosthocVerificationError("score identity/source ordinal was changed")
        if score["snapshot_sha256"] != snapshot_sha:
            raise PosthocVerificationError("score snapshot mismatch")
        values = [
            score[k]
            for k in ("gram_positive_mean", "gram_negative_mean", "joint_chance_probability")
        ]
        if not all(_is_finite_number(value) for value in values):
            raise PosthocVerificationError("nonfinite or nonnumeric score")
        if not 0 <= score["joint_chance_probability"] <= 1:
            raise PosthocVerificationError("invalid joint chance probability")
        seen.add(sid)
    if scores != sorted(scores, key=lambda row: (row["source_ordinal"], row["sequence_id"])):
        raise PosthocVerificationError("score records are not in canonical source order")
    ordered = sorted(
        scores,
        key=lambda row: (
            -(row["joint_chance_probability"] >= threshold),
            -_finite_midpoint(row["gram_positive_mean"], row["gram_negative_mean"]),
            row["sequence_id"],
            row["source_ordinal"],
        ),
    )
    expected_ranked = [
        {
            "rank": index,
            "sequence_id": row["sequence_id"],
            "source_ordinal": row["source_ordinal"],
            "chance_feasible": row["joint_chance_probability"] >= threshold,
            "mean_utility": _finite_midpoint(row["gram_positive_mean"], row["gram_negative_mean"]),
            "joint_chance_probability": row["joint_chance_probability"],
        }
        for index, row in enumerate(ordered, start=1)
    ]
    for index, row in enumerate(ranked):
        _require_exact_keys(row, rank_keys, f"ranking row {index}")
        if (
            type(row["rank"]) is not int
            or type(row["source_ordinal"]) is not int
            or type(row["chance_feasible"]) is not bool
            or type(row["mean_utility"]) not in (int, float)
            or type(row["joint_chance_probability"]) not in (int, float)
            or not _is_finite_number(row["mean_utility"])
            or not _is_finite_number(row["joint_chance_probability"])
        ):
            raise PosthocVerificationError("ranking row contains non-exact or nonfinite fields")
    if _canonical(ranked) != _canonical(expected_ranked):
        raise PosthocVerificationError("ranking was reordered or score-derived fields were changed")
    expected_stream = [row["sequence_id"] for row in ranked if row["chance_feasible"]][:392]
    if len(expected_stream) != 392 or method_stream != expected_stream:
        raise PosthocVerificationError("method-seat stream mismatch")
    ranking_core = {
        "pool_seal_sha256": pool_seal,
        "snapshot_sha256": snapshot_sha,
        "scores": scores,
        "ranked": ranked,
        "method_stream": expected_stream,
    }
    return _sha(_canonical(ranking_core)), len(ranked)


def _verify_controller(rows: list[dict[str, Any]], method_stream: Sequence[str]) -> None:
    keys = {"wave", "seat", "source", "sequence_id"}
    if len(rows) != 448:
        raise PosthocVerificationError("controller must contain 28 waves of 16 seats")
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        _require_exact_keys(row, keys, f"controller seat {ordinal}")
        wave, seat = divmod(ordinal, 16)
        expected_source = "categorical_diffusion_posthoc" if seat < 14 else "common_reserve"
        if (
            type(row["wave"]) is not int
            or type(row["seat"]) is not int
            or row["wave"] != wave
            or row["seat"] != seat
            or row["source"] != expected_source
        ):
            raise PosthocVerificationError("14+2 controller seat mapping was changed")
        sid = _require_sha(row["sequence_id"], "controller sequence_id")
        if sid in seen:
            raise PosthocVerificationError("duplicate controller identity")
        seen.add(sid)
        if seat < 14 and sid != method_stream[wave * 14 + seat]:
            raise PosthocVerificationError("controller method seat is not the immutable stream")


def verify_bundle(payloads: Mapping[str, bytes]) -> VerificationReceipt:
    """Recompute structural records without authenticating any external identity."""

    if type(payloads) is not dict:
        raise PosthocVerificationError("bundle must be an exact built-in dict")
    if len(payloads) != len(EXPECTED_FILES):
        raise PosthocVerificationError("bundle file count differs")
    for name in payloads:
        if type(name) is not str or len(name) > MAX_BUNDLE_FILE_NAME_CHARACTERS:
            raise PosthocVerificationError("bundle file name violates its exact bounded contract")
    actual_files = set(payloads)
    if actual_files != EXPECTED_FILES:
        raise PosthocVerificationError("bundle file names differ")
    total_bytes = 0
    for name, payload in payloads.items():
        if type(payload) is not bytes or len(payload) > MAX_FILE_BYTES[name]:
            raise PosthocVerificationError(f"{name} exceeds its byte cap or is not exact bytes")
        total_bytes += len(payload)
    if total_bytes > MAX_BUNDLE_BYTES:
        raise PosthocVerificationError("bundle exceeds total byte cap")
    _verify_checksums(payloads)
    _verify_contract(payloads["adapter-contract.toml"])
    context = _json(payloads["pool-context.json"], "pool-context.json")
    manifest = _json(payloads["manifest.json"], "manifest.json")
    snapshot = _json(payloads["posterior-snapshot.json"], "posterior-snapshot.json")
    if (
        not isinstance(context, dict)
        or not isinstance(manifest, dict)
        or not isinstance(snapshot, dict)
    ):
        raise PosthocVerificationError("singleton payloads must be JSON objects")
    attempts = _jsonl(payloads["attempts.jsonl"], "attempts.jsonl")
    slots = _jsonl(payloads["resolved-pool.jsonl"], "resolved-pool.jsonl")
    scores = _jsonl(payloads["scores.jsonl"], "scores.jsonl")
    ranked = _jsonl(payloads["ranking.jsonl"], "ranking.jsonl")
    controller = _jsonl(payloads["controller-composition.jsonl"], "controller-composition.jsonl")
    manifest_keys = {
        "schema_version",
        "artifact",
        "status",
        "pool_seal_sha256",
        "ranking_seal_sha256",
        "method_stream",
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
        "organizer_reference_present",
        "evidence_class",
        "external_request_policy_trajectory_and_snapshot_authenticated",
        "common_reserve_authority_accepted",
        "charged_call_evidence_present",
        "payload_sha256",
    }
    _require_exact_keys(manifest, manifest_keys, "manifest")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["artifact"] != "categorical_diffusion_posthoc_engineering_bundle_v1"
        or manifest["status"] != "blocked_engineering_only"
        or manifest["evidence_class"] != "structural_only_external_inputs_unauthenticated"
    ):
        raise PosthocVerificationError("manifest identity/status mismatch")
    for flag in (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
        "organizer_reference_present",
        "external_request_policy_trajectory_and_snapshot_authenticated",
        "common_reserve_authority_accepted",
        "charged_call_evidence_present",
    ):
        if manifest[flag] is not False:
            raise PosthocVerificationError("manifest contains authorization or reference leakage")
    expected_hashes = {
        name: _sha(payloads[name])
        for name in sorted(EXPECTED_FILES - {"manifest.json", "SHA256SUMS"})
    }
    if manifest["payload_sha256"] != expected_hashes:
        raise PosthocVerificationError("manifest payload hashes mismatch")
    pool_seal, candidate_count = _verify_attempts_and_pool(attempts, slots, context)
    snapshot_sha, threshold = _verify_snapshot(snapshot, context)
    ranking_seal, ranking_count = _verify_ranking(
        scores, ranked, slots, snapshot_sha, threshold, pool_seal, manifest["method_stream"]
    )
    if manifest["pool_seal_sha256"] != pool_seal or manifest["ranking_seal_sha256"] != ranking_seal:
        raise PosthocVerificationError("manifest pool/ranking seals mismatch")
    _verify_controller(controller, manifest["method_stream"])
    bundle_sha = _sha(
        "".join(f"{name}:{_sha(payloads[name])}\n" for name in sorted(payloads)).encode()
    )
    return VerificationReceipt(bundle_sha, len(attempts), candidate_count, ranking_count, 392, 448)


_STABLE_STAT_FIELDS = (
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


def _same_stat(first: os.stat_result, second: os.stat_result) -> bool:
    return all(getattr(first, field) == getattr(second, field) for field in _STABLE_STAT_FIELDS)


def _capture_payload(directory_fd: int, name: str, cap: int) -> _HeldPayload:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise PosthocVerificationError("safe no-follow nonblocking opens are unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PosthocVerificationError(f"cannot safely open {name}") from exc
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PosthocVerificationError(f"{name} must be a single-link regular file")
        if before.st_size > cap:
            raise PosthocVerificationError(f"{name} exceeds its byte cap")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > cap:
                raise PosthocVerificationError(f"{name} grew beyond its byte cap")
        after = os.fstat(file_fd)
        try:
            entry_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PosthocVerificationError(f"{name} changed during bounded read") from exc
        if (
            total != before.st_size
            or not _same_stat(before, after)
            or not _same_stat(after, entry_after)
        ):
            raise PosthocVerificationError(f"{name} changed during bounded read")
        held = _HeldPayload(name, file_fd, before, b"".join(chunks))
        _recheck_payload(directory_fd, held, phase="bounded read")
        return held
    except BaseException:
        os.close(file_fd)
        raise


def _scan_exact_inventory(directory_fd: int) -> None:
    names: set[str] = set()
    try:
        iterator = os.scandir(directory_fd)
    except OSError as exc:
        raise PosthocVerificationError("cannot enumerate bundle directory") from exc
    with iterator:
        for entry in iterator:
            if len(names) >= len(EXPECTED_FILES):
                raise PosthocVerificationError("directory contains too many entries")
            if type(entry.name) is not str:
                raise PosthocVerificationError("directory entry name is not text")
            names.add(entry.name)
    if names != EXPECTED_FILES:
        raise PosthocVerificationError(f"directory files differ: {sorted(names ^ EXPECTED_FILES)}")


def _recheck_payload(
    directory_fd: int,
    held: _HeldPayload,
    *,
    phase: str = "structural verification",
) -> None:
    """Compare fresh held-descriptor bytes between two complete identity checks."""

    message = f"{held.name} changed during {phase}"

    def check_identity() -> None:
        descriptor_after = os.fstat(held.file_descriptor)
        entry_after = os.stat(held.name, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_stat(held.initial_stat, descriptor_after) or not _same_stat(
            descriptor_after, entry_after
        ):
            raise PosthocVerificationError(message)

    try:
        check_identity()
        os.lseek(held.file_descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(held.payload):
            chunk = os.read(held.file_descriptor, min(1024 * 1024, len(held.payload) - offset))
            if not chunk or chunk != held.payload[offset : offset + len(chunk)]:
                raise PosthocVerificationError(message)
            offset += len(chunk)
        if os.read(held.file_descriptor, 1):
            raise PosthocVerificationError(message)
        check_identity()
    except OSError as exc:
        raise PosthocVerificationError(message) from exc


def _bounded_exact_path(path: Path) -> str:
    if type(path) is not type(Path()):
        raise PosthocVerificationError("bundle path must have the exact platform Path type")
    raw_path = os.fspath(path)
    if (
        type(raw_path) is not str
        or not raw_path
        or len(raw_path) > MAX_PATH_CHARACTERS
        or "\x00" in raw_path
    ):
        raise PosthocVerificationError("bundle path exceeds its bounded text contract")
    return raw_path


def verify_directory(path: Path) -> VerificationReceipt:
    """Use descriptor-relative, no-follow, bounded reads for structural verification."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise PosthocVerificationError("safe no-follow nonblocking opens are unavailable")
    raw_path = _bounded_exact_path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        directory_fd = os.open(raw_path, flags)
    except OSError as exc:
        raise PosthocVerificationError("bundle path must be a safe directory") from exc
    held_payloads: list[_HeldPayload] = []
    try:
        directory_before = os.fstat(directory_fd)
        _scan_exact_inventory(directory_fd)
        payloads: dict[str, bytes] = {}
        total_bytes = 0
        for name in sorted(EXPECTED_FILES):
            held = _capture_payload(directory_fd, name, MAX_FILE_BYTES[name])
            held_payloads.append(held)
            total_bytes += len(held.payload)
            if total_bytes > MAX_BUNDLE_BYTES:
                raise PosthocVerificationError("bundle exceeds total byte cap")
            payloads[name] = held.payload
        receipt = verify_bundle(payloads)
        for held in held_payloads:
            _recheck_payload(directory_fd, held)
        _scan_exact_inventory(directory_fd)
        if not _same_stat(directory_before, os.fstat(directory_fd)):
            raise PosthocVerificationError("bundle directory changed during verification")
        return receipt
    finally:
        for held in reversed(held_payloads):
            os.close(held.file_descriptor)
        os.close(directory_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    receipt = verify_directory(args.bundle)
    print(json.dumps({name: getattr(receipt, name) for name in receipt.__slots__}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

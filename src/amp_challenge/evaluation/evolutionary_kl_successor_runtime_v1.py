"""Blocked producer scaffold for the evolutionary/KL successor screen.

The objects in this module describe schedules, resource accounting, digest
slots, and deterministic stop handling.  They contain no model, oracle, query
selection, filesystem publication, or job-dispatch implementation.  Every
authority is deliberately non-authorizing and every campaign artifact starts
in a missing, execution-blocking state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from amp_challenge.evaluation.evolutionary_kl_successor_protocol_v2 import (
    CONFIGURATION_IDS_V2,
    FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
    FULL_METHOD_ID_V2,
    METHOD_IDS_V2,
    SCREEN_SEEDS_V2,
    SuccessorProtocolV2,
)

FROZEN_SUCCESSOR_RUNTIME_V1_SHA256 = (
    "d0a34eb1adb14608a5fafb771f85c988552ce532be7274f71666f82af5ae251a"
)
SUCCESSOR_RUNTIME_V1_RELATIVE_PATH = Path(
    "configs/search/evolutionary_kl_successor_runtime_v1.toml"
)
BLOCKING_DIGEST = "missing_execution_blocking"
ENGINEERING_EVIDENCE_CLASS = "producer_scaffold_only_not_scientific_evidence"
CALL_CHECKPOINTS = tuple(range(64, 513, 16))
WALL_CHECKPOINT_MINUTES = (0, 15, 30, 45, 60, 75, 90, 105, 120)
HARD_STOP_REASONS = (
    "nonfinite_or_psd_support_failure",
    "forbidden_support_or_namespace_overlap",
    "unsealed_oracle_response",
    "kg_tie_or_numerical_instability",
    "exhausted_deterministic_proposal_stream",
    "artifact_or_receipt_digest_mismatch",
    "kl_constraint_violation",
    "replay_integrity_or_propensity_failure",
    "policy_version_lag_exceeded",
    "validity_drop_or_terminal_eligibility_failure",
)
RUN_DIGEST_ROLES = (
    "adapter",
    "asset_manifest",
    "query_ledger",
    "query_identity_uniqueness_receipt",
    "timing_ledger",
    "common_initial_copy_seal",
    "terminal_record",
)
RUN_EVIDENCE_ROLES = (
    "query_ledger",
    "query_identity_uniqueness_receipt",
    "timing_ledger",
    "common_initial_copy_seal",
    "progress_chain_receipt",
    "oracle_request_seal_manifest",
    "oracle_response_seal_manifest",
    "resource_usage_receipt",
    "terminal_record",
    "primary_evidence",
    "secondary_evidence",
)
SHARED_ARTIFACT_ROLES = (
    "primary_source_registry",
    "namespace_receipt",
    "generator_checkpoint_manifest",
    "oracle_bundle_manifest",
    "adaptive_posterior_manifest",
    "generated_sequence_feature_manifest",
    "laplacian_feature_manifest",
    "joint_kg_preflight_receipt",
    "external_timing_issuer_receipt",
    "common_initial_schedule",
    "common_reserve_schedule",
    "common_initial_source_response_seal",
    "common_initial_latency_receipt",
    "common_initial_copy_manifest",
    "training_exclusion_inventory",
    "organizer_veto_inventory",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_ARTIFACT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,255}\Z")
_RUN_ID_RE = re.compile(r"screen\.[a-z0-9][a-z0-9_.-]{0,127}\.seed-[0-9]+\Z")


class SuccessorRuntimeV1Error(ValueError):
    """Raised when a successor runtime object fails closed."""


class SuccessorExecutionUnavailable(RuntimeError):
    """Raised for every attempt to use this scaffold as execution authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorRuntimeV1Error(message)


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a full lowercase SHA-256",
    )
    return value  # type: ignore[return-value]


def _exact_false(value: object, *, label: str) -> None:
    _require(type(value) is bool and value is False, f"{label} must remain exact false")


def _exact_int(value: object, expected: int, *, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} differs")


def _table(value: object, *, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a TOML table")
    return value  # type: ignore[return-value]


def _array(value: object, *, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a TOML array")
    return value  # type: ignore[return-value]


def _canonical_json_bytes(document: object) -> bytes:
    try:
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise SuccessorRuntimeV1Error("document is not finite canonical JSON") from error


def _document_sha256(domain: bytes, document: object) -> str:
    _require(type(domain) is bytes and domain.endswith(b"\0"), "hash domain is invalid")
    return hashlib.sha256(domain + _canonical_json_bytes(document)).hexdigest()


def _read_regular_file_no_follow(path: Path) -> bytes:
    _require(isinstance(path, Path), "runtime path must be pathlib.Path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SuccessorRuntimeV1Error("runtime path is unavailable or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "runtime path is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    _require(bool(payload), "runtime config is empty")
    return payload


@dataclass(frozen=True, slots=True)
class DigestSlot:
    """One explicitly pinned or explicitly missing artifact digest."""

    role: str
    status: Literal["pinned", "missing_execution_blocking"]
    sha256: str | None

    def __post_init__(self) -> None:
        _require(
            type(self.role) is str and _ID_RE.fullmatch(self.role) is not None,
            "digest role invalid",
        )
        _require(
            type(self.status) is str and self.status in {"pinned", "missing_execution_blocking"},
            "digest status invalid",
        )
        if self.status == "pinned":
            _sha256(self.sha256, label=f"{self.role} digest")
        else:
            _require(self.sha256 is None, f"{self.role} missing digest must be null")

    @classmethod
    def from_config(cls, role: str, value: object) -> DigestSlot:
        _require(type(value) is str, f"{role} config digest must be text")
        if value == BLOCKING_DIGEST:
            return cls(role=role, status="missing_execution_blocking", sha256=None)
        return cls(role=role, status="pinned", sha256=_sha256(value, label=f"{role} config"))

    @property
    def is_blocking(self) -> bool:
        self.__post_init__()
        return self.status == "missing_execution_blocking"

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {"role": self.role, "sha256": self.sha256, "status": self.status}


@dataclass(frozen=True, slots=True)
class AuthenticatedArtifact:
    """Artifact bytes whose claimed identity is verified at construction.

    A lowercase digest-shaped string alone is never evidence.  The bytes stay
    attached to this in-memory receipt so every downstream reconstruction can
    re-hash them before trusting the binding.
    """

    artifact_id: str
    role: str
    payload: bytes
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require(
            type(self.artifact_id) is str
            and _ARTIFACT_ID_RE.fullmatch(self.artifact_id) is not None,
            "authenticated artifact ID invalid",
        )
        _require(
            type(self.role) is str and _ID_RE.fullmatch(self.role) is not None,
            "authenticated artifact role invalid",
        )
        _require(
            type(self.payload) is bytes and 0 < len(self.payload) <= 16 * 1024 * 1024,
            "authenticated artifact payload must be non-empty bounded bytes",
        )
        observed = hashlib.sha256(self.payload).hexdigest()
        if self.sha256 is None:
            object.__setattr__(self, "sha256", observed)
        else:
            _sha256(self.sha256, label=f"{self.artifact_id} claimed digest")
            _require(self.sha256 == observed, "authenticated artifact digest differs from bytes")

    @classmethod
    def from_bytes(cls, artifact_id: str, role: str, payload: bytes) -> AuthenticatedArtifact:
        return cls(artifact_id=artifact_id, role=role, payload=payload)

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact_id": self.artifact_id,
            "byte_length": len(self.payload),
            "role": self.role,
            "sha256": self.sha256,
        }

    def digest_slot(self) -> DigestSlot:
        self.__post_init__()
        assert self.sha256 is not None
        return DigestSlot(role=self.role, status="pinned", sha256=self.sha256)


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentPin:
    """Byte pins and process identity required by every validation/runtime job."""

    runner: str
    uv_version: str
    python_implementation: str
    python_version: str
    uv_lock_path: str
    uv_lock_sha256: str
    project_manifest_path: str
    project_manifest_sha256: str
    pythonhashseed: str
    locale: str
    timezone: str
    pytest_plugin_autoload: Literal[False]
    python_user_site: Literal[False]

    def __post_init__(self) -> None:
        _require(
            self.runner == "/home/yonghan.yang/.local/bin/uv run --locked --no-sync",
            "runtime runner differs",
        )
        _require(self.uv_version == "0.9.9", "runtime uv version differs")
        _require(self.python_implementation == "CPython", "Python implementation differs")
        _require(self.python_version == "3.11.14", "Python version differs")
        _require(self.uv_lock_path == "uv.lock", "uv lock path differs")
        _sha256(self.uv_lock_sha256, label="uv lock")
        _require(
            self.project_manifest_path == "pyproject.toml",
            "project manifest path differs",
        )
        _sha256(self.project_manifest_sha256, label="project manifest")
        _require(self.pythonhashseed == "0", "PYTHONHASHSEED differs")
        _require(self.locale == "C.UTF-8", "runtime locale differs")
        _require(self.timezone == "UTC", "runtime timezone differs")
        _exact_false(self.pytest_plugin_autoload, label="pytest plugin autoload")
        _exact_false(self.python_user_site, label="Python user site")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "locale": self.locale,
            "project_manifest_path": self.project_manifest_path,
            "project_manifest_sha256": self.project_manifest_sha256,
            "pytest_plugin_autoload": self.pytest_plugin_autoload,
            "python_implementation": self.python_implementation,
            "python_user_site": self.python_user_site,
            "python_version": self.python_version,
            "pythonhashseed": self.pythonhashseed,
            "runner": self.runner,
            "timezone": self.timezone,
            "uv_lock_path": self.uv_lock_path,
            "uv_lock_sha256": self.uv_lock_sha256,
            "uv_version": self.uv_version,
        }


@dataclass(frozen=True, slots=True)
class ConfigurationBinding:
    """One method or ablation slot and its unavailable implementation pins."""

    configuration_id: str
    kind: Literal["method", "ablation"]
    base_method: str
    adapter: DigestSlot
    asset_manifest: DigestSlot

    def __post_init__(self) -> None:
        _require(
            type(self.configuration_id) is str and self.configuration_id in CONFIGURATION_IDS_V2,
            "configuration ID is not frozen",
        )
        expected_kind = "method" if self.configuration_id in METHOD_IDS_V2 else "ablation"
        _require(
            type(self.kind) is str and self.kind == expected_kind, "configuration kind differs"
        )
        expected_base = "none" if self.kind == "method" else FULL_METHOD_ID_V2
        _require(
            type(self.base_method) is str and self.base_method == expected_base,
            "configuration base method differs",
        )
        _require(type(self.adapter) is DigestSlot, "adapter slot type differs")
        _require(type(self.asset_manifest) is DigestSlot, "asset slot type differs")
        self.adapter.__post_init__()
        self.asset_manifest.__post_init__()
        _require(self.adapter.role == "adapter", "adapter digest role differs")
        _require(self.asset_manifest.role == "asset_manifest", "asset digest role differs")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "adapter": self.adapter.document(),
            "asset_manifest": self.asset_manifest.document(),
            "base_method": self.base_method,
            "configuration_id": self.configuration_id,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class PhysicalCommonInitialAccounting:
    """Additive physical accounting for the shared initial responses."""

    physical_seed_blocks: int
    physical_calls_per_seed: int
    physical_calls: int
    logical_copies_per_seed: int
    logical_charges: int
    wall_seconds_per_seed: int
    additive_a100_hours: float
    method_run_a100_hours: float
    proposed_total_a100_hours: float
    maximum_adaptive_physical_calls: int
    maximum_total_physical_calls: int
    fallback_or_recompute_allowed: Literal[False] = False
    unallocated_resource_pool_allowed: Literal[False] = False
    resource_ceiling_reconciliation_accepted: Literal[True] = True

    def __post_init__(self) -> None:
        for value, expected, label in (
            (self.physical_seed_blocks, 5, "physical seed blocks"),
            (self.physical_calls_per_seed, 64, "physical calls per seed"),
            (self.physical_calls, 320, "physical initial calls"),
            (self.logical_copies_per_seed, 13, "logical copies per seed"),
            (self.logical_charges, 4160, "logical initial charges"),
            (self.wall_seconds_per_seed, 900, "common-initial wall seconds"),
            (self.maximum_adaptive_physical_calls, 29120, "maximum adaptive physical calls"),
            (self.maximum_total_physical_calls, 29440, "maximum total physical calls"),
        ):
            _exact_int(value, expected, label=label)
        for value, expected, label in (
            (self.additive_a100_hours, 1.25, "additive A100 hours"),
            (self.method_run_a100_hours, 130.0, "method-run A100 hours"),
            (self.proposed_total_a100_hours, 131.25, "proposed total A100 hours"),
        ):
            _require(type(value) is float and value == expected, f"{label} differs")
        _exact_false(self.fallback_or_recompute_allowed, label="common-initial fallback/recompute")
        _exact_false(self.unallocated_resource_pool_allowed, label="unallocated resource pool")
        _require(
            type(self.resource_ceiling_reconciliation_accepted) is bool
            and self.resource_ceiling_reconciliation_accepted is True,
            "resource-ceiling reconciliation must remain exact true",
        )
        _require(
            self.physical_calls == self.physical_seed_blocks * self.physical_calls_per_seed,
            "physical initial-call arithmetic differs",
        )
        _require(
            self.logical_charges
            == self.physical_seed_blocks
            * self.logical_copies_per_seed
            * self.physical_calls_per_seed,
            "logical initial-charge arithmetic differs",
        )
        _require(
            self.additive_a100_hours
            == self.physical_seed_blocks * self.wall_seconds_per_seed / 3600,
            "additive A100-hour arithmetic differs",
        )
        _require(
            self.proposed_total_a100_hours == self.method_run_a100_hours + self.additive_a100_hours,
            "proposed total A100-hour arithmetic differs",
        )
        _require(
            self.maximum_total_physical_calls
            == self.physical_calls + self.maximum_adaptive_physical_calls,
            "maximum physical-call arithmetic differs",
        )

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "additive_a100_hours": self.additive_a100_hours,
            "fallback_or_recompute_allowed": self.fallback_or_recompute_allowed,
            "logical_charges": self.logical_charges,
            "logical_copies_per_seed": self.logical_copies_per_seed,
            "maximum_adaptive_physical_calls": self.maximum_adaptive_physical_calls,
            "maximum_total_physical_calls": self.maximum_total_physical_calls,
            "method_run_a100_hours": self.method_run_a100_hours,
            "physical_calls": self.physical_calls,
            "physical_calls_per_seed": self.physical_calls_per_seed,
            "physical_seed_blocks": self.physical_seed_blocks,
            "proposed_total_a100_hours": self.proposed_total_a100_hours,
            "resource_ceiling_reconciliation_accepted": (
                self.resource_ceiling_reconciliation_accepted
            ),
            "unallocated_resource_pool_allowed": self.unallocated_resource_pool_allowed,
            "wall_seconds_per_seed": self.wall_seconds_per_seed,
        }


@dataclass(frozen=True, slots=True)
class SuccessorRuntimeRegistryV1:
    """Exact 13-by-5 blocked runtime registry."""

    runtime_sha256: str
    protocol_sha256: str
    configurations: tuple[ConfigurationBinding, ...]
    shared_artifacts: tuple[DigestSlot, ...]
    physical_accounting: PhysicalCommonInitialAccounting
    runtime_environment: RuntimeEnvironmentPin
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False
    scientific_evidence_accepted: Literal[False] = False
    automatic_production_eligible: Literal[False] = False
    biological_superiority_claim_allowed: Literal[False] = False
    hidden_confirmation_available: Literal[False] = False
    independent_verification_complete: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.runtime_sha256) is str
            and self.runtime_sha256 == FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "runtime registry digest differs",
        )
        _require(
            type(self.protocol_sha256) is str
            and self.protocol_sha256 == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "runtime protocol digest differs",
        )
        _require(
            type(self.configurations) is tuple
            and all(type(item) is ConfigurationBinding for item in self.configurations),
            "configuration bindings must be exact tuple members",
        )
        _require(
            tuple(item.configuration_id for item in self.configurations) == CONFIGURATION_IDS_V2,
            "configuration registry order or membership differs",
        )
        for item in self.configurations:
            item.__post_init__()
        _require(
            type(self.shared_artifacts) is tuple
            and all(type(item) is DigestSlot for item in self.shared_artifacts),
            "shared artifacts must be exact tuple members",
        )
        _require(
            tuple(item.role for item in self.shared_artifacts) == SHARED_ARTIFACT_ROLES,
            "shared artifact roles differ",
        )
        for item in self.shared_artifacts:
            item.__post_init__()
        _require(
            type(self.physical_accounting) is PhysicalCommonInitialAccounting,
            "physical accounting type differs",
        )
        self.physical_accounting.__post_init__()
        _require(
            type(self.runtime_environment) is RuntimeEnvironmentPin,
            "runtime environment type differs",
        )
        self.runtime_environment.__post_init__()
        for label, value in (
            ("execution authorization", self.execution_authorized),
            ("oracle-call authorization", self.oracle_calls_authorized),
            ("scientific-evidence acceptance", self.scientific_evidence_accepted),
            ("automatic-production eligibility", self.automatic_production_eligible),
            ("biological-superiority claim", self.biological_superiority_claim_allowed),
            ("hidden-confirmation availability", self.hidden_confirmation_available),
            ("independent verification", self.independent_verification_complete),
        ):
            _exact_false(value, label=label)

    @property
    def blockers(self) -> tuple[str, ...]:
        self.__post_init__()
        result = [f"shared:{slot.role}" for slot in self.shared_artifacts if slot.is_blocking]
        for binding in self.configurations:
            if binding.adapter.is_blocking:
                result.append(f"{binding.configuration_id}:adapter")
            if binding.asset_manifest.is_blocking:
                result.append(f"{binding.configuration_id}:asset_manifest")
        result.extend(
            (
                "execution_authorized=false",
                "oracle_calls_authorized=false",
                "independent_verification_complete=false",
                "hidden_confirmation_available=false",
            )
        )
        return tuple(result)

    def configuration_map(self) -> dict[str, ConfigurationBinding]:
        self.__post_init__()
        return {item.configuration_id: item for item in self.configurations}


def load_successor_runtime_registry_v1(
    path: Path,
    protocol: SuccessorProtocolV2,
) -> SuccessorRuntimeRegistryV1:
    """Load the exact runtime overlay while retaining every authority flag false."""

    _require(type(protocol) is SuccessorProtocolV2, "protocol must have exact successor-v2 type")
    protocol.__post_init__()
    payload = _read_regular_file_no_follow(path)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    _require(
        observed_sha256 == FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
        "successor runtime byte digest differs",
    )
    try:
        parsed = tomllib.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SuccessorRuntimeV1Error("successor runtime is not strict UTF-8 TOML") from error
    _require(type(parsed) is dict, "runtime root must be a TOML table")
    _exact_int(parsed.get("schema_version"), 1, label="runtime schema version")
    _require(
        parsed.get("artifact") == "evolutionary_kl_successor_runtime_v1"
        and type(parsed.get("artifact")) is str,
        "runtime artifact differs",
    )
    _require(
        parsed.get("status") == "blocked_evidence_bound_transition_validator"
        and type(parsed.get("status")) is str,
        "runtime status differs",
    )
    _require(
        parsed.get("protocol_sha256") == protocol.protocol_sha256,
        "runtime protocol pin differs",
    )
    for key in (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "automatic_production_eligible",
        "biological_superiority_claim_allowed",
        "hidden_confirmation_available",
        "independent_verification_complete",
    ):
        _exact_false(parsed.get(key), label=f"runtime {key}")
    _require(tuple(parsed.get("screen_seeds", ())) == SCREEN_SEEDS_V2, "runtime seeds differ")
    _require(parsed.get("full_method_id") == FULL_METHOD_ID_V2, "runtime full method differs")
    _require(parsed.get("blocking_digest_value") == BLOCKING_DIGEST, "blocking marker differs")

    logical = _table(parsed.get("logical_schedule"), label="logical schedule")
    for key, expected in (
        ("configuration_count", 13),
        ("seed_count", 5),
        ("run_count", 65),
        ("common_initial_calls_per_run", 64),
        ("adaptive_batch_count", 28),
        ("calls_per_adaptive_batch", 16),
        ("method_seats_per_adaptive_batch", 14),
        ("reserve_seats_per_adaptive_batch", 2),
        ("adaptive_calls_per_run", 448),
        ("total_calls_per_run", 512),
        ("scientific_wall_seconds_per_run", 7200),
        ("logical_initial_charges", 4160),
        ("logical_adaptive_charges", 29120),
        ("logical_total_charges", 33280),
    ):
        _exact_int(logical.get(key), expected, label=f"logical {key}")
    for key, expected in (
        ("failed_submission_is_charged", True),
        ("failed_submission_replacement_allowed", False),
        ("duplicate_query_identity_within_run_allowed", False),
    ):
        _require(
            type(logical.get(key)) is bool and logical[key] is expected, f"logical {key} differs"
        )
    for key in (
        "within_run_all_512_query_identity_uniqueness_receipt_required",
        "cross_arm_common_initial_identity_reuse_within_seed_required",
    ):
        _require(type(logical.get(key)) is bool and logical[key] is True, f"logical {key} differs")
    _exact_false(
        logical.get("common_initial_adaptive_or_reserve_identity_overlap_allowed"),
        label="within-run common/adaptive identity overlap",
    )

    visibility = _table(parsed.get("visibility_and_clock"), label="visibility and clock")
    _require(
        visibility.get("unsubmitted_proposal_and_tree_evaluation_source")
        == "method_posterior_only",
        "unsubmitted-candidate scoring source differs",
    )
    _exact_false(
        visibility.get("hidden_surrogate_truth_may_score_unsubmitted_candidates"),
        label="hidden surrogate scoring of unsubmitted candidates",
    )
    for key in (
        "hidden_surrogate_truth_visible_only_after_charged_submission",
        "method_specific_model_loading_inside_scientific_clock",
        "generation_search_and_proposal_inside_scientific_clock",
        "posterior_acquisition_and_update_inside_scientific_clock",
        "endpoint_distillation_inside_scientific_clock",
        "adaptive_oracle_queue_and_poll_inside_scientific_clock",
    ):
        _require(
            type(visibility.get(key)) is bool and visibility[key] is True,
            f"visibility/clock {key} differs",
        )

    statistics = _table(parsed.get("statistics"), label="statistics")
    _require(
        statistics.get("exact_sign_test_zero_difference_policy")
        == "discard_ties_and_use_n_equal_to_non_tied_pairs",
        "runtime exact-sign tie policy differs",
    )
    _require(
        type(statistics.get("exact_sign_test_all_ties_p_value")) is float
        and statistics["exact_sign_test_all_ties_p_value"] == 1.0,
        "runtime exact-sign all-ties value differs",
    )

    stopping = _table(parsed.get("stopping"), label="stopping")
    for key in (
        "stop_at_call_or_wall_budget_whichever_first",
        "discard_unsealed_partial_batch",
        "carry_last_authenticated_sealed_incumbent",
        "algorithmic_failures_retained",
        "resume_only_from_last_authenticated_round",
        "cumulative_progress_chain_required",
        "terminal_progress_receipt_required",
        "unique_call_receipt_required",
        "oracle_request_and_response_seals_required",
        "resource_usage_receipt_required",
    ):
        _require(
            type(stopping.get(key)) is bool and stopping[key] is True, f"stopping {key} differs"
        )
    _exact_false(stopping.get("resumed_clock_may_reset"), label="resumed scientific clock reset")
    _exact_int(stopping.get("result_driven_reruns"), 0, label="result-driven reruns")
    _exact_int(
        stopping.get("infrastructure_reruns_before_first_oracle_response"),
        1,
        label="pre-oracle infrastructure reruns",
    )
    for reason in HARD_STOP_REASONS:
        if reason in {
            "exhausted_deterministic_proposal_stream",
            "artifact_or_receipt_digest_mismatch",
        }:
            continue
        _require(stopping.get(reason) == "hard_stop", f"hard-stop mapping {reason} differs")

    source_provenance = _table(parsed.get("source_provenance"), label="source provenance")
    _require(
        source_provenance.get("primary_source_registry_role") == "provenance_only_non_authorizing",
        "primary-source registry role differs",
    )
    _exact_false(
        source_provenance.get(
            "primary_source_registry_may_substitute_for_adapter_or_asset_authority"
        ),
        label="source-registry substitution authority",
    )
    _require(
        type(source_provenance.get("arcadiamp_tr2d2_mp2d_arms_are_adaptations_not_reproductions"))
        is bool
        and source_provenance["arcadiamp_tr2d2_mp2d_arms_are_adaptations_not_reproductions"]
        is True,
        "style-arm adaptation boundary differs",
    )
    for key in (
        "tr2d2_tree_selection_and_pareto_retention_propensities_are_explicit",
        "mp2d_numeric_rollout_cap_is_frozen",
    ):
        _require(
            type(source_provenance.get(key)) is bool and source_provenance[key] is True,
            f"source provenance {key} differs",
        )
    _exact_false(
        source_provenance.get("tr2d2_rn_or_exact_offpolicy_claim_allowed_without_propensities"),
        label="source provenance exact off-policy claim without propensities",
    )

    raw_environment = _table(parsed.get("runtime_environment"), label="runtime environment")
    runtime_environment = RuntimeEnvironmentPin(
        runner=raw_environment.get("runner"),  # type: ignore[arg-type]
        uv_version=raw_environment.get("uv_version"),  # type: ignore[arg-type]
        python_implementation=raw_environment.get("python_implementation"),  # type: ignore[arg-type]
        python_version=raw_environment.get("python_version"),  # type: ignore[arg-type]
        uv_lock_path=raw_environment.get("uv_lock_path"),  # type: ignore[arg-type]
        uv_lock_sha256=raw_environment.get("uv_lock_sha256"),  # type: ignore[arg-type]
        project_manifest_path=raw_environment.get("project_manifest_path"),  # type: ignore[arg-type]
        project_manifest_sha256=raw_environment.get("project_manifest_sha256"),  # type: ignore[arg-type]
        pythonhashseed=raw_environment.get("pythonhashseed"),  # type: ignore[arg-type]
        locale=raw_environment.get("locale"),  # type: ignore[arg-type]
        timezone=raw_environment.get("timezone"),  # type: ignore[arg-type]
        pytest_plugin_autoload=raw_environment.get("pytest_plugin_autoload"),  # type: ignore[arg-type]
        python_user_site=raw_environment.get("python_user_site"),  # type: ignore[arg-type]
    )
    repository_root = path.parent.parent.parent
    for relative, expected, label in (
        (
            runtime_environment.uv_lock_path,
            runtime_environment.uv_lock_sha256,
            "runtime uv lock",
        ),
        (
            runtime_environment.project_manifest_path,
            runtime_environment.project_manifest_sha256,
            "runtime project manifest",
        ),
    ):
        candidate = repository_root / relative
        observed = hashlib.sha256(_read_regular_file_no_follow(candidate)).hexdigest()
        _require(observed == expected, f"{label} bytes differ")

    common = _table(parsed.get("physical_common_initial"), label="physical common initial")
    envelope = _table(parsed.get("physical_screen_envelope"), label="physical screen envelope")
    _require(
        common.get("policy")
        == "evaluate_once_per_seed_then_share_byte_identical_authenticated_responses",
        "common-initial sharing policy differs",
    )
    for key in (
        "outside_each_arm_scientific_clock",
        "inside_proposed_additive_screen_resource_envelope",
        "latency_receipt_required",
        "timing_ledger_required",
        "source_response_seal_required",
        "per_arm_copy_seal_required",
    ):
        _require(type(common.get(key)) is bool and common[key] is True, f"common {key} differs")
    for key in ("fallback_or_recompute_allowed", "unallocated_resource_pool_allowed"):
        _exact_false(common.get(key), label=f"common {key}")
    _require(
        common.get("slurm_account") == "bio"
        and common.get("partition") == "gpumid"
        and common.get("gpu_type") == "A100",
        "common-initial Slurm placement differs",
    )
    for key, expected in (
        ("nodes_per_seed", 1),
        ("gpus_per_seed", 1),
        ("cpus_per_seed", 8),
        ("host_memory_gib_per_seed", 32),
    ):
        _exact_int(common.get(key), expected, label=f"common {key}")

    reconciliation = _table(
        parsed.get("resource_ceiling_reconciliation"),
        label="resource ceiling reconciliation",
    )
    _require(
        reconciliation.get("parent_protocol_field") == "resources.screen_a100_hour_ceiling"
        and reconciliation.get("parent_value_derivation")
        == "130_method_run_a100_hours_plus_1.25_shared_initial_a100_hours",
        "parent resource-ceiling identity or derivation differs",
    )
    _require(
        type(reconciliation.get("parent_protocol_value")) is float
        and reconciliation["parent_protocol_value"] == 131.25,
        "parent resource-ceiling value differs",
    )
    _require(
        type(reconciliation.get("runtime_proposed_additive_total_a100_hours")) is float
        and reconciliation["runtime_proposed_additive_total_a100_hours"] == 131.25,
        "proposed additive resource ceiling differs",
    )
    for key in ("parent_arm_clock_starts_after_common_initial_responses",):
        _require(
            type(reconciliation.get(key)) is bool and reconciliation[key] is True,
            f"resource reconciliation {key} differs",
        )
    for key in (
        "parent_protocol_modified_or_reinterpreted",
        "gap_audit_identifies_shared_initial_as_unaccounted_additive_resource",
        "execution_requires_reviewed_protocol_clarification_or_correction",
    ):
        _exact_false(reconciliation.get(key), label=f"resource reconciliation {key}")
    _require(
        type(reconciliation.get("resource_ceiling_reconciliation_accepted")) is bool
        and reconciliation["resource_ceiling_reconciliation_accepted"] is True,
        "resource reconciliation acceptance differs",
    )

    accounting = PhysicalCommonInitialAccounting(
        physical_seed_blocks=common.get("seed_block_productions"),  # type: ignore[arg-type]
        physical_calls_per_seed=common.get("physical_unique_calls_per_seed"),  # type: ignore[arg-type]
        physical_calls=common.get("physical_unique_calls"),  # type: ignore[arg-type]
        logical_copies_per_seed=common.get("logical_arm_copies_per_seed"),  # type: ignore[arg-type]
        logical_charges=common.get("logical_charges"),  # type: ignore[arg-type]
        wall_seconds_per_seed=common.get("wall_seconds_per_seed"),  # type: ignore[arg-type]
        additive_a100_hours=common.get("additive_a100_hours"),  # type: ignore[arg-type]
        method_run_a100_hours=envelope.get("method_run_a100_hours"),  # type: ignore[arg-type]
        proposed_total_a100_hours=envelope.get("proposed_total_a100_hours"),  # type: ignore[arg-type]
        maximum_adaptive_physical_calls=envelope.get("maximum_adaptive_physical_unique_calls"),  # type: ignore[arg-type]
        maximum_total_physical_calls=envelope.get("maximum_total_physical_unique_calls"),  # type: ignore[arg-type]
        fallback_or_recompute_allowed=common.get("fallback_or_recompute_allowed"),  # type: ignore[arg-type]
        unallocated_resource_pool_allowed=common.get("unallocated_resource_pool_allowed"),  # type: ignore[arg-type]
        resource_ceiling_reconciliation_accepted=reconciliation.get(
            "resource_ceiling_reconciliation_accepted"
        ),  # type: ignore[arg-type]
    )

    raw_shared = _table(parsed.get("required_shared_artifacts"), label="shared artifacts")
    _require(
        tuple(raw_shared) == tuple(f"{role}_sha256" for role in SHARED_ARTIFACT_ROLES),
        "shared artifact registry order or membership differs",
    )
    shared = tuple(
        DigestSlot.from_config(role, raw_shared[f"{role}_sha256"]) for role in SHARED_ARTIFACT_ROLES
    )
    raw_configurations = _array(parsed.get("configurations"), label="configurations")
    bindings: list[ConfigurationBinding] = []
    for index, raw in enumerate(raw_configurations):
        item = _table(raw, label=f"configuration {index}")
        _require(
            set(item) == {"id", "kind", "base_method", "adapter_sha256", "asset_manifest_sha256"},
            f"configuration {index} keys differ",
        )
        bindings.append(
            ConfigurationBinding(
                configuration_id=item.get("id"),  # type: ignore[arg-type]
                kind=item.get("kind"),  # type: ignore[arg-type]
                base_method=item.get("base_method"),  # type: ignore[arg-type]
                adapter=DigestSlot.from_config("adapter", item.get("adapter_sha256")),
                asset_manifest=DigestSlot.from_config(
                    "asset_manifest", item.get("asset_manifest_sha256")
                ),
            )
        )

    return SuccessorRuntimeRegistryV1(
        runtime_sha256=observed_sha256,
        protocol_sha256=protocol.protocol_sha256,
        configurations=tuple(bindings),
        shared_artifacts=shared,
        physical_accounting=accounting,
        runtime_environment=runtime_environment,
    )


@dataclass(frozen=True, slots=True)
class LogicalSeat:
    """One logical charged position, without a query identity."""

    charged_call_position: int
    phase: Literal["common_initial", "adaptive"]
    batch_index: int | None
    seat_index: int
    source: Literal["common_initial", "method", "prefrozen_common_reserve"]

    def __post_init__(self) -> None:
        _require(
            type(self.charged_call_position) is int and 1 <= self.charged_call_position <= 512,
            "charged call position invalid",
        )
        _require(
            type(self.phase) is str and self.phase in {"common_initial", "adaptive"},
            "logical phase invalid",
        )
        _require(type(self.seat_index) is int, "logical seat index must be exact integer")
        if self.phase == "common_initial":
            _require(self.batch_index is None, "common-initial seat cannot have a batch")
            _require(1 <= self.seat_index <= 64, "common-initial seat index invalid")
            _require(self.source == "common_initial", "common-initial source differs")
            _require(
                self.charged_call_position == self.seat_index,
                "common-initial charged position differs",
            )
        else:
            _require(
                type(self.batch_index) is int and 1 <= self.batch_index <= 28,
                "adaptive batch index invalid",
            )
            _require(1 <= self.seat_index <= 16, "adaptive seat index invalid")
            expected_source = "method" if self.seat_index <= 14 else "prefrozen_common_reserve"
            _require(self.source == expected_source, "adaptive seat source differs")
            expected_position = 64 + (self.batch_index - 1) * 16 + self.seat_index
            _require(
                self.charged_call_position == expected_position,
                "adaptive charged position differs",
            )

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "batch_index": self.batch_index,
            "charged_call_position": self.charged_call_position,
            "phase": self.phase,
            "seat_index": self.seat_index,
            "source": self.source,
        }


@lru_cache(maxsize=1)
def build_logical_schedule() -> tuple[LogicalSeat, ...]:
    """Build the exact 64 + 28 x (14 + 2) logical topology."""

    seats = [
        LogicalSeat(
            charged_call_position=position,
            phase="common_initial",
            batch_index=None,
            seat_index=position,
            source="common_initial",
        )
        for position in range(1, 65)
    ]
    for batch_index in range(1, 29):
        for seat_index in range(1, 17):
            seats.append(
                LogicalSeat(
                    charged_call_position=64 + (batch_index - 1) * 16 + seat_index,
                    phase="adaptive",
                    batch_index=batch_index,
                    seat_index=seat_index,
                    source="method" if seat_index <= 14 else "prefrozen_common_reserve",
                )
            )
    result = tuple(seats)
    _require(len(result) == 512, "logical schedule size differs")
    _require(
        tuple(seat.charged_call_position for seat in result) == tuple(range(1, 513)),
        "logical charged positions are not contiguous",
    )
    return result


@dataclass(frozen=True, slots=True)
class ScreenRunPlan:
    """One blocked public-screen run plan."""

    run_id: str
    configuration_id: str
    seed: int
    logical_schedule: tuple[LogicalSeat, ...]
    topology_sha256: str
    artifact_digests: tuple[DigestSlot, ...]
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False

    def __post_init__(self) -> None:
        expected_run_id = f"screen.{self.configuration_id}.seed-{self.seed}"
        _require(
            type(self.run_id) is str
            and _RUN_ID_RE.fullmatch(self.run_id) is not None
            and self.run_id == expected_run_id,
            "run ID differs from frozen slot",
        )
        _require(
            type(self.configuration_id) is str and self.configuration_id in CONFIGURATION_IDS_V2,
            "run configuration is not frozen",
        )
        _require(type(self.seed) is int and self.seed in SCREEN_SEEDS_V2, "run seed is not frozen")
        _require(
            type(self.logical_schedule) is tuple
            and all(type(seat) is LogicalSeat for seat in self.logical_schedule),
            "logical schedule members differ",
        )
        for seat in self.logical_schedule:
            seat.__post_init__()
        _require(self.logical_schedule == build_logical_schedule(), "logical topology differs")
        expected_topology = _document_sha256(
            b"amp/evolutionary-kl/successor-v2/logical-topology/v1\0",
            {
                "configuration_id": self.configuration_id,
                "run_id": self.run_id,
                "seats": [seat.document() for seat in self.logical_schedule],
                "seed": self.seed,
            },
        )
        _require(
            type(self.topology_sha256) is str and self.topology_sha256 == expected_topology,
            "logical topology digest differs",
        )
        _require(
            type(self.artifact_digests) is tuple
            and all(type(slot) is DigestSlot for slot in self.artifact_digests),
            "run artifact slots must be exact tuple members",
        )
        _require(
            tuple(slot.role for slot in self.artifact_digests) == RUN_DIGEST_ROLES,
            "run artifact digest roles differ",
        )
        for slot in self.artifact_digests:
            slot.__post_init__()
        _exact_false(self.execution_authorized, label="run execution authorization")
        _exact_false(self.oracle_calls_authorized, label="run oracle-call authorization")

    @property
    def plan_sha256(self) -> str:
        self.__post_init__()
        return _document_sha256(
            b"amp/evolutionary-kl/successor-v2/run-plan/v1\0",
            self.document(),
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        self.__post_init__()
        missing = tuple(slot.role for slot in self.artifact_digests if slot.is_blocking)
        return (*missing, "execution_authorized=false", "oracle_calls_authorized=false")

    def document(self) -> dict[str, object]:
        _exact_false(self.execution_authorized, label="run execution authorization")
        _exact_false(self.oracle_calls_authorized, label="run oracle-call authorization")
        return {
            "artifact": "evolutionary_kl_successor_screen_run_plan_v1",
            "artifact_digests": [slot.document() for slot in self.artifact_digests],
            "authorization": {
                "execution_authorized": False,
                "oracle_calls_authorized": False,
            },
            "configuration_id": self.configuration_id,
            "evidence_class": ENGINEERING_EVIDENCE_CLASS,
            "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "run_id": self.run_id,
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "schema_version": 1,
            "seed": self.seed,
            "topology_sha256": self.topology_sha256,
        }


def build_screen_run_plans(
    registry: SuccessorRuntimeRegistryV1,
) -> tuple[ScreenRunPlan, ...]:
    """Build all 13 x 5 public plans; no plan contains query identities."""

    _require(type(registry) is SuccessorRuntimeRegistryV1, "registry exact type differs")
    registry.__post_init__()
    return _build_screen_run_plans_from_bindings(registry.configurations)


def _build_screen_run_plans_from_bindings(
    bindings: tuple[ConfigurationBinding, ...],
) -> tuple[ScreenRunPlan, ...]:
    _require(
        type(bindings) is tuple
        and tuple(binding.configuration_id for binding in bindings) == CONFIGURATION_IDS_V2,
        "screen-plan binding inventory differs",
    )
    for binding in bindings:
        _require(type(binding) is ConfigurationBinding, "screen-plan binding type differs")
        binding.__post_init__()
    schedule = build_logical_schedule()
    plans: list[ScreenRunPlan] = []
    for binding in bindings:
        for seed in SCREEN_SEEDS_V2:
            run_id = f"screen.{binding.configuration_id}.seed-{seed}"
            topology_sha256 = _document_sha256(
                b"amp/evolutionary-kl/successor-v2/logical-topology/v1\0",
                {
                    "configuration_id": binding.configuration_id,
                    "run_id": run_id,
                    "seats": [seat.document() for seat in schedule],
                    "seed": seed,
                },
            )
            plans.append(
                ScreenRunPlan(
                    run_id=run_id,
                    configuration_id=binding.configuration_id,
                    seed=seed,
                    logical_schedule=schedule,
                    topology_sha256=topology_sha256,
                    artifact_digests=(
                        binding.adapter,
                        binding.asset_manifest,
                        DigestSlot("query_ledger", "missing_execution_blocking", None),
                        DigestSlot(
                            "query_identity_uniqueness_receipt",
                            "missing_execution_blocking",
                            None,
                        ),
                        DigestSlot("timing_ledger", "missing_execution_blocking", None),
                        DigestSlot(
                            "common_initial_copy_seal",
                            "missing_execution_blocking",
                            None,
                        ),
                        DigestSlot("terminal_record", "missing_execution_blocking", None),
                    ),
                )
            )
    result = tuple(plans)
    _require(len(result) == 65, "screen plan count differs")
    _require(len({plan.run_id for plan in result}) == 65, "screen plan IDs are not unique")
    _require(
        len({plan.plan_sha256 for plan in result}) == 65,
        "screen plan digests are not unique",
    )
    return result


@dataclass(frozen=True, slots=True)
class NonAuthorizingCampaignAuthority:
    """Path-free evidence identity that remains incapable of authorizing work."""

    protocol_sha256: str
    runtime_sha256: str
    source_commit: str
    source_tree_sha256: str
    environment_lock_sha256: str
    plans: tuple[ScreenRunPlan, ...]
    shared_artifacts: tuple[DigestSlot, ...]
    independent_verifier: DigestSlot
    authenticated_artifacts: tuple[AuthenticatedArtifact, ...] = ()
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False
    scientific_evidence_accepted: Literal[False] = False
    automatic_production_eligible: Literal[False] = False
    biological_superiority_claim_allowed: Literal[False] = False
    hidden_confirmation_available: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.protocol_sha256) is str
            and self.protocol_sha256 == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "authority protocol digest differs",
        )
        _require(
            type(self.runtime_sha256) is str
            and self.runtime_sha256 == FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "authority runtime digest differs",
        )
        _require(
            type(self.source_commit) is str
            and _GIT_SHA1_RE.fullmatch(self.source_commit) is not None,
            "authority source commit must be full lowercase Git SHA-1",
        )
        _sha256(self.source_tree_sha256, label="authority source tree")
        _sha256(self.environment_lock_sha256, label="authority environment lock")
        _require(
            type(self.plans) is tuple and all(type(plan) is ScreenRunPlan for plan in self.plans),
            "authority plans must be exact tuple members",
        )
        expected_slots = tuple(
            (configuration_id, seed)
            for configuration_id in CONFIGURATION_IDS_V2
            for seed in SCREEN_SEEDS_V2
        )
        _require(
            tuple((plan.configuration_id, plan.seed) for plan in self.plans) == expected_slots,
            "authority screen inventory differs",
        )
        for plan in self.plans:
            plan.__post_init__()
        _require(
            type(self.shared_artifacts) is tuple
            and all(type(slot) is DigestSlot for slot in self.shared_artifacts),
            "authority shared artifacts must be exact digest slots",
        )
        _require(
            tuple(slot.role for slot in self.shared_artifacts) == SHARED_ARTIFACT_ROLES,
            "authority shared artifact inventory differs",
        )
        for slot in self.shared_artifacts:
            slot.__post_init__()
        _require(type(self.independent_verifier) is DigestSlot, "verifier slot type differs")
        self.independent_verifier.__post_init__()
        _require(
            self.independent_verifier.role == "independent_verifier",
            "independent verifier role differs",
        )
        _require(
            type(self.authenticated_artifacts) is tuple
            and all(
                type(artifact) is AuthenticatedArtifact for artifact in self.authenticated_artifacts
            ),
            "authority authenticated artifacts must be exact tuple members",
        )
        authenticated: dict[str, AuthenticatedArtifact] = {}
        for artifact in self.authenticated_artifacts:
            artifact.__post_init__()
            _require(
                artifact.artifact_id not in authenticated,
                "authority duplicates an authenticated artifact ID",
            )
            authenticated[artifact.artifact_id] = artifact
        _require(
            len({artifact.sha256 for artifact in self.authenticated_artifacts})
            == len(self.authenticated_artifacts),
            "authority aliases authenticated artifact bytes across roles",
        )
        for slot in self.shared_artifacts:
            artifact = authenticated.get(f"shared:{slot.role}")
            if slot.status == "pinned":
                _require(artifact is not None, f"pinned shared {slot.role} lacks artifact bytes")
                assert artifact is not None
                _require(
                    artifact.role == slot.role and artifact.sha256 == slot.sha256,
                    f"pinned shared {slot.role} differs from artifact bytes",
                )
            else:
                _require(artifact is None, f"blocking shared {slot.role} has unbound bytes")
        for plan in self.plans:
            for role in ("adapter", "asset_manifest"):
                slot = next(item for item in plan.artifact_digests if item.role == role)
                artifact = authenticated.get(f"config:{plan.configuration_id}:{role}")
                if slot.status == "pinned":
                    _require(
                        artifact is not None,
                        f"pinned {plan.configuration_id} {role} lacks artifact bytes",
                    )
                    assert artifact is not None
                    _require(
                        artifact.role == role and artifact.sha256 == slot.sha256,
                        f"pinned {plan.configuration_id} {role} differs from artifact bytes",
                    )
                else:
                    _require(
                        artifact is None,
                        f"blocking {plan.configuration_id} {role} has unbound bytes",
                    )
        verifier_artifact = authenticated.get("independent:verifier")
        if self.independent_verifier.status == "pinned":
            _require(verifier_artifact is not None, "pinned verifier lacks artifact bytes")
            assert verifier_artifact is not None
            _require(
                verifier_artifact.role == "independent_verifier"
                and verifier_artifact.sha256 == self.independent_verifier.sha256,
                "independent verifier differs from artifact bytes",
            )
        else:
            _require(verifier_artifact is None, "blocking verifier has unbound bytes")
        for label, value in (
            ("execution authorization", self.execution_authorized),
            ("oracle-call authorization", self.oracle_calls_authorized),
            ("scientific-evidence acceptance", self.scientific_evidence_accepted),
            ("automatic-production eligibility", self.automatic_production_eligible),
            ("biological-superiority claim", self.biological_superiority_claim_allowed),
            ("hidden-confirmation availability", self.hidden_confirmation_available),
        ):
            _exact_false(value, label=f"authority {label}")

    @property
    def authority_sha256(self) -> str:
        self.__post_init__()
        return _document_sha256(
            b"amp/evolutionary-kl/successor-v2/non-authorizing-authority/v1\0",
            self.document(),
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        self.__post_init__()
        result = [f"{plan.run_id}:{role}" for plan in self.plans for role in plan.blockers]
        result.extend(f"shared:{slot.role}" for slot in self.shared_artifacts if slot.is_blocking)
        if self.independent_verifier.is_blocking:
            result.append("independent_verifier")
        result.extend(
            (
                "execution_authorized=false",
                "oracle_calls_authorized=false",
                "scientific_evidence_accepted=false",
                "automatic_production_eligible=false",
                "hidden_confirmation_available=false",
            )
        )
        return tuple(result)

    def document(self) -> dict[str, object]:
        for label, value in (
            ("execution authorization", self.execution_authorized),
            ("oracle-call authorization", self.oracle_calls_authorized),
            ("scientific-evidence acceptance", self.scientific_evidence_accepted),
            ("automatic-production eligibility", self.automatic_production_eligible),
            ("biological-superiority claim", self.biological_superiority_claim_allowed),
            ("hidden-confirmation availability", self.hidden_confirmation_available),
        ):
            _exact_false(value, label=f"authority {label}")
        return {
            "artifact": "evolutionary_kl_successor_non_authorizing_campaign_authority_v1",
            "authorization": {
                "automatic_production_eligible": False,
                "biological_superiority_claim_allowed": False,
                "execution_authorized": False,
                "hidden_confirmation_available": False,
                "oracle_calls_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "environment_lock_sha256": self.environment_lock_sha256,
            "evidence_class": ENGINEERING_EVIDENCE_CLASS,
            "authenticated_artifacts": [
                artifact.document() for artifact in self.authenticated_artifacts
            ],
            "independent_verifier": self.independent_verifier.document(),
            "plan_sha256s": [plan.plan_sha256 for plan in self.plans],
            "shared_artifacts": [slot.document() for slot in self.shared_artifacts],
            "primary_source_registry_role": "provenance_only_non_authorizing",
            "primary_source_registry_may_substitute_for_adapter_or_asset_authority": False,
            "protocol_sha256": self.protocol_sha256,
            "runtime_sha256": self.runtime_sha256,
            "schema_version": 1,
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "status": "blocked_non_authorizing",
        }


def build_non_authorizing_authority(
    registry: SuccessorRuntimeRegistryV1,
    *,
    source_commit: str,
    source_tree_sha256: str,
    environment_lock_sha256: str,
) -> NonAuthorizingCampaignAuthority:
    """Bind source identity to all plans without granting any capability."""

    _require(type(registry) is SuccessorRuntimeRegistryV1, "registry exact type differs")
    registry.__post_init__()
    _require(
        environment_lock_sha256 == registry.runtime_environment.uv_lock_sha256,
        "authority environment lock differs from verified runtime bytes",
    )
    return NonAuthorizingCampaignAuthority(
        protocol_sha256=registry.protocol_sha256,
        runtime_sha256=registry.runtime_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        environment_lock_sha256=environment_lock_sha256,
        plans=build_screen_run_plans(registry),
        shared_artifacts=registry.shared_artifacts,
        independent_verifier=DigestSlot("independent_verifier", "missing_execution_blocking", None),
    )


def materialize_evidence_bound_authority(
    registry: SuccessorRuntimeRegistryV1,
    *,
    source_commit: str,
    source_tree_sha256: str,
    shared_artifacts: tuple[AuthenticatedArtifact, ...],
    configuration_artifacts: tuple[tuple[str, AuthenticatedArtifact, AuthenticatedArtifact], ...],
    independent_verifier: AuthenticatedArtifact,
) -> NonAuthorizingCampaignAuthority:
    """Bind actual bytes for every prerequisite without granting execution.

    This is an evidence-construction primitive, not an acceptance primitive.
    Even a complete result remains descriptive because all authorization bits
    are structurally false.
    """

    _require(type(registry) is SuccessorRuntimeRegistryV1, "registry exact type differs")
    registry.__post_init__()
    _require(
        tuple(artifact.artifact_id for artifact in shared_artifacts)
        == tuple(f"shared:{role}" for role in SHARED_ARTIFACT_ROLES),
        "evidence-bound shared artifact order or membership differs",
    )
    for artifact, role in zip(shared_artifacts, SHARED_ARTIFACT_ROLES, strict=True):
        artifact.__post_init__()
        _require(artifact.role == role, f"shared artifact role {role} differs")
    _require(
        tuple(row[0] for row in configuration_artifacts) == CONFIGURATION_IDS_V2,
        "evidence-bound configuration artifact inventory differs",
    )
    bindings: list[ConfigurationBinding] = []
    authenticated = list(shared_artifacts)
    registry_by_id = registry.configuration_map()
    for configuration_id, adapter, asset_manifest in configuration_artifacts:
        adapter.__post_init__()
        asset_manifest.__post_init__()
        _require(
            adapter.artifact_id == f"config:{configuration_id}:adapter"
            and adapter.role == "adapter",
            f"{configuration_id} adapter identity differs",
        )
        _require(
            asset_manifest.artifact_id == f"config:{configuration_id}:asset_manifest"
            and asset_manifest.role == "asset_manifest",
            f"{configuration_id} asset-manifest identity differs",
        )
        template = registry_by_id[configuration_id]
        bindings.append(
            ConfigurationBinding(
                configuration_id=configuration_id,
                kind=template.kind,
                base_method=template.base_method,
                adapter=adapter.digest_slot(),
                asset_manifest=asset_manifest.digest_slot(),
            )
        )
        authenticated.extend((adapter, asset_manifest))
    independent_verifier.__post_init__()
    _require(
        independent_verifier.artifact_id == "independent:verifier"
        and independent_verifier.role == "independent_verifier",
        "independent verifier artifact identity differs",
    )
    authenticated.append(independent_verifier)
    return NonAuthorizingCampaignAuthority(
        protocol_sha256=registry.protocol_sha256,
        runtime_sha256=registry.runtime_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        environment_lock_sha256=registry.runtime_environment.uv_lock_sha256,
        plans=_build_screen_run_plans_from_bindings(tuple(bindings)),
        shared_artifacts=tuple(artifact.digest_slot() for artifact in shared_artifacts),
        independent_verifier=independent_verifier.digest_slot(),
        authenticated_artifacts=tuple(authenticated),
    )


ProgressStatus = Literal["running", "completed", "budget_stopped", "hard_stopped"]
StopTrigger = Literal["none", "call_budget", "wall_budget", "hard_stop"]
SCIENTIFIC_LIMIT_NS = 7_200_000_000_000


@dataclass(frozen=True, slots=True)
class RunSafetyState:
    """Frozen hard-stop predicates evaluated at every progress transition."""

    numerical_support_ok: bool = True
    namespace_and_support_ok: bool = True
    oracle_responses_sealed: bool = True
    kg_numerics_stable: bool = True
    proposal_stream_available: bool = True
    artifact_receipts_match: bool = True
    kl_constraints_ok: bool = True
    replay_integrity_and_propensities_ok: bool = True
    policy_version_lag_ok: bool = True
    validity_and_terminal_eligibility_ok: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.numerical_support_ok,
            self.namespace_and_support_ok,
            self.oracle_responses_sealed,
            self.kg_numerics_stable,
            self.proposal_stream_available,
            self.artifact_receipts_match,
            self.kl_constraints_ok,
            self.replay_integrity_and_propensities_ok,
            self.policy_version_lag_ok,
            self.validity_and_terminal_eligibility_ok,
        ):
            _require(type(value) is bool, "safety predicate must be exact Boolean")

    @property
    def first_failure(self) -> str | None:
        self.__post_init__()
        checks = (
            (self.numerical_support_ok, "nonfinite_or_psd_support_failure"),
            (self.namespace_and_support_ok, "forbidden_support_or_namespace_overlap"),
            (self.oracle_responses_sealed, "unsealed_oracle_response"),
            (self.kg_numerics_stable, "kg_tie_or_numerical_instability"),
            (self.proposal_stream_available, "exhausted_deterministic_proposal_stream"),
            (self.artifact_receipts_match, "artifact_or_receipt_digest_mismatch"),
            (self.kl_constraints_ok, "kl_constraint_violation"),
            (
                self.replay_integrity_and_propensities_ok,
                "replay_integrity_or_propensity_failure",
            ),
            (self.policy_version_lag_ok, "policy_version_lag_exceeded"),
            (
                self.validity_and_terminal_eligibility_ok,
                "validity_drop_or_terminal_eligibility_failure",
            ),
        )
        return next((reason for passed, reason in checks if not passed), None)

    def document(self) -> dict[str, bool]:
        self.__post_init__()
        return {
            "artifact_receipts_match": self.artifact_receipts_match,
            "kg_numerics_stable": self.kg_numerics_stable,
            "kl_constraints_ok": self.kl_constraints_ok,
            "namespace_and_support_ok": self.namespace_and_support_ok,
            "numerical_support_ok": self.numerical_support_ok,
            "oracle_responses_sealed": self.oracle_responses_sealed,
            "policy_version_lag_ok": self.policy_version_lag_ok,
            "proposal_stream_available": self.proposal_stream_available,
            "replay_integrity_and_propensities_ok": (self.replay_integrity_and_propensities_ok),
            "validity_and_terminal_eligibility_ok": (self.validity_and_terminal_eligibility_ok),
        }


@dataclass(frozen=True, slots=True)
class RunProgressSnapshot:
    """One self-hashed cumulative checkpoint in a predecessor chain."""

    run_id: str
    run_plan_sha256: str
    authority_sha256: str
    ordinal: int
    predecessor_sha256: str
    charged_calls: int
    last_authenticated_sealed_call: int
    partial_batch_index: int | None
    partial_batch_submitted_calls: int
    segment_id: str
    segment_elapsed_scientific_ns: int
    segment_elapsed_wall_ns: int
    cumulative_elapsed_scientific_ns: int
    cumulative_elapsed_wall_ns: int
    status: ProgressStatus
    stop_trigger: StopTrigger
    hard_stop_reason: str | None
    query_ledger_sha256: str
    query_identity_uniqueness_receipt_sha256: str
    timing_ledger_sha256: str
    common_initial_copy_seal_sha256: str
    oracle_request_seal_manifest_sha256: str
    oracle_response_seal_manifest_sha256: str
    resource_usage_receipt_sha256: str
    safety_state: RunSafetyState
    snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        _require(
            type(self.run_id) is str and _RUN_ID_RE.fullmatch(self.run_id) is not None,
            "progress run ID invalid",
        )
        _sha256(self.run_plan_sha256, label="progress run plan")
        _sha256(self.authority_sha256, label="progress authority")
        _require(type(self.ordinal) is int and self.ordinal >= 0, "progress ordinal invalid")
        _sha256(self.predecessor_sha256, label="progress predecessor")
        _require(
            type(self.charged_calls) is int and 64 <= self.charged_calls <= 512,
            "progress charged-call count invalid",
        )
        _require(
            type(self.last_authenticated_sealed_call) is int
            and self.last_authenticated_sealed_call in CALL_CHECKPOINTS
            and self.last_authenticated_sealed_call <= self.charged_calls,
            "last authenticated sealed call invalid",
        )
        partial = self.charged_calls - self.last_authenticated_sealed_call
        _require(partial <= 16, "more than one unsealed partial batch is present")
        _require(
            type(self.partial_batch_submitted_calls) is int
            and self.partial_batch_submitted_calls == partial,
            "partial-batch submitted count differs from charged denominator",
        )
        expected_partial_batch = (
            None if partial == 0 else (self.last_authenticated_sealed_call - 64) // 16 + 1
        )
        _require(
            (self.partial_batch_index is None or type(self.partial_batch_index) is int)
            and self.partial_batch_index == expected_partial_batch,
            "partial-batch index is ambiguous or differs",
        )
        _require(
            type(self.segment_id) is str and _ID_RE.fullmatch(self.segment_id) is not None,
            "progress segment ID invalid",
        )
        for value, label in (
            (self.segment_elapsed_scientific_ns, "segment scientific time"),
            (self.segment_elapsed_wall_ns, "segment wall time"),
            (self.cumulative_elapsed_scientific_ns, "cumulative scientific time"),
            (self.cumulative_elapsed_wall_ns, "cumulative wall time"),
        ):
            _require(type(value) is int and value >= 0, f"{label} invalid")
        _require(
            self.cumulative_elapsed_scientific_ns <= SCIENTIFIC_LIMIT_NS
            and self.cumulative_elapsed_wall_ns <= SCIENTIFIC_LIMIT_NS,
            "progress exceeds the cumulative scientific or wall-time ceiling",
        )
        _require(
            type(self.status) is str
            and self.status in {"running", "completed", "budget_stopped", "hard_stopped"},
            "progress status invalid",
        )
        _require(
            type(self.stop_trigger) is str
            and self.stop_trigger in {"none", "call_budget", "wall_budget", "hard_stop"},
            "progress stop trigger invalid",
        )
        _require(
            self.hard_stop_reason is None
            or (type(self.hard_stop_reason) is str and self.hard_stop_reason in HARD_STOP_REASONS),
            "hard-stop reason invalid",
        )
        for label, value in (
            ("query ledger", self.query_ledger_sha256),
            (
                "query-identity uniqueness receipt",
                self.query_identity_uniqueness_receipt_sha256,
            ),
            ("timing ledger", self.timing_ledger_sha256),
            ("common-initial copy seal", self.common_initial_copy_seal_sha256),
            ("oracle request seals", self.oracle_request_seal_manifest_sha256),
            ("oracle response seals", self.oracle_response_seal_manifest_sha256),
            ("resource usage receipt", self.resource_usage_receipt_sha256),
        ):
            _sha256(value, label=f"progress {label}")
        _require(type(self.safety_state) is RunSafetyState, "progress safety state type differs")
        self.safety_state.__post_init__()
        if self.status == "completed":
            _require(
                self.charged_calls == 512
                and self.last_authenticated_sealed_call == 512
                and partial == 0
                and self.stop_trigger == "call_budget"
                and self.hard_stop_reason is None,
                "completed progress is incomplete",
            )
            _require(
                max(
                    self.cumulative_elapsed_scientific_ns,
                    self.cumulative_elapsed_wall_ns,
                )
                < SCIENTIFIC_LIMIT_NS,
                "completed progress reached or exceeded the wall-time ceiling",
            )
            _require(self.safety_state.first_failure is None, "completed progress violates safety")
        elif self.status == "budget_stopped":
            _require(self.hard_stop_reason is None, "budget stop cannot name a hard-stop reason")
            _require(
                self.stop_trigger == "wall_budget"
                and max(
                    self.cumulative_elapsed_scientific_ns,
                    self.cumulative_elapsed_wall_ns,
                )
                == SCIENTIFIC_LIMIT_NS,
                "budget stop did not land exactly on the cumulative wall-time ceiling",
            )
            _require(self.safety_state.first_failure is None, "budget stop violates safety")
        elif self.status == "hard_stopped":
            _require(
                self.stop_trigger == "hard_stop"
                and self.hard_stop_reason is not None
                and self.hard_stop_reason == self.safety_state.first_failure,
                "hard stop must name the first failed frozen safety predicate",
            )
        else:
            _require(
                self.stop_trigger == "none" and self.hard_stop_reason is None,
                "running progress cannot name a stop trigger or reason",
            )
            _require(
                self.charged_calls < 512
                and self.cumulative_elapsed_scientific_ns < SCIENTIFIC_LIMIT_NS
                and self.cumulative_elapsed_wall_ns < SCIENTIFIC_LIMIT_NS,
                "running progress already reached a frozen budget",
            )
            _require(
                partial == 0,
                "running checkpoints must be sealed and unambiguously resumable",
            )
            _require(self.safety_state.first_failure is None, "running progress violates safety")
        expected_digest = _document_sha256(
            b"amp/evolutionary-kl/successor-v2/run-progress/v2\0",
            self.unsigned_document(),
        )
        if self.snapshot_sha256 is None:
            object.__setattr__(self, "snapshot_sha256", expected_digest)
        else:
            _sha256(self.snapshot_sha256, label="progress snapshot")
            _require(
                self.snapshot_sha256 == expected_digest, "progress snapshot self-digest differs"
            )

    def unsigned_document(self) -> dict[str, object]:
        return {
            "artifact": "evolutionary_kl_successor_run_progress_v2",
            "authority_sha256": self.authority_sha256,
            "charged_calls": self.charged_calls,
            "cumulative_elapsed_scientific_ns": self.cumulative_elapsed_scientific_ns,
            "cumulative_elapsed_wall_ns": self.cumulative_elapsed_wall_ns,
            "common_initial_copy_seal_sha256": self.common_initial_copy_seal_sha256,
            "hard_stop_reason": self.hard_stop_reason,
            "last_authenticated_sealed_call": self.last_authenticated_sealed_call,
            "oracle_request_seal_manifest_sha256": self.oracle_request_seal_manifest_sha256,
            "oracle_response_seal_manifest_sha256": self.oracle_response_seal_manifest_sha256,
            "ordinal": self.ordinal,
            "partial_batch_index": self.partial_batch_index,
            "partial_batch_submitted_calls": self.partial_batch_submitted_calls,
            "predecessor_sha256": self.predecessor_sha256,
            "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "query_identity_uniqueness_receipt_sha256": (
                self.query_identity_uniqueness_receipt_sha256
            ),
            "query_ledger_sha256": self.query_ledger_sha256,
            "resource_usage_receipt_sha256": self.resource_usage_receipt_sha256,
            "run_id": self.run_id,
            "run_plan_sha256": self.run_plan_sha256,
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "safety_state": self.safety_state.document(),
            "schema_version": 2,
            "segment_elapsed_scientific_ns": self.segment_elapsed_scientific_ns,
            "segment_elapsed_wall_ns": self.segment_elapsed_wall_ns,
            "segment_id": self.segment_id,
            "status": self.status,
            "stop_trigger": self.stop_trigger,
            "timing_ledger_sha256": self.timing_ledger_sha256,
        }

    def document(self) -> dict[str, object]:
        self.__post_init__()
        result = self.unsigned_document()
        result["snapshot_sha256"] = self.snapshot_sha256
        return result


def progress_genesis_sha256(plan: ScreenRunPlan, authority_sha256: str) -> str:
    plan.__post_init__()
    _sha256(authority_sha256, label="progress genesis authority")
    return _document_sha256(
        b"amp/evolutionary-kl/successor-v2/progress-genesis/v2\0",
        {
            "authority_sha256": authority_sha256,
            "run_id": plan.run_id,
            "run_plan_sha256": plan.plan_sha256,
        },
    )


def validate_progress_chain(
    plan: ScreenRunPlan,
    authority_sha256: str,
    snapshots: tuple[RunProgressSnapshot, ...],
) -> RunProgressSnapshot:
    """Validate the complete cumulative chain and return its terminal head."""

    plan.__post_init__()
    _sha256(authority_sha256, label="progress-chain authority")
    _require(
        type(snapshots) is tuple
        and bool(snapshots)
        and all(type(snapshot) is RunProgressSnapshot for snapshot in snapshots),
        "progress chain must be a non-empty exact snapshot tuple",
    )
    seen_segments: set[str] = set()
    previous: RunProgressSnapshot | None = None
    plan_sha256 = plan.plan_sha256
    for ordinal, snapshot in enumerate(snapshots):
        snapshot.__post_init__()
        _require(snapshot.run_id == plan.run_id, "progress chain run ID differs")
        _require(snapshot.run_plan_sha256 == plan_sha256, "progress chain plan differs")
        _require(snapshot.authority_sha256 == authority_sha256, "progress chain authority differs")
        _require(snapshot.ordinal == ordinal, "progress chain ordinal is not contiguous")
        _require(snapshot.segment_id not in seen_segments, "progress chain reuses a segment ID")
        seen_segments.add(snapshot.segment_id)
        if previous is None:
            _require(
                snapshot.predecessor_sha256 == progress_genesis_sha256(plan, authority_sha256),
                "progress chain genesis predecessor differs",
            )
            _require(
                snapshot.charged_calls == 64
                and snapshot.last_authenticated_sealed_call == 64
                and snapshot.partial_batch_submitted_calls == 0
                and snapshot.status == "running"
                and snapshot.segment_elapsed_scientific_ns == 0
                and snapshot.segment_elapsed_wall_ns == 0
                and snapshot.cumulative_elapsed_scientific_ns == 0
                and snapshot.cumulative_elapsed_wall_ns == 0,
                "progress chain genesis is not the sealed common-initial checkpoint",
            )
        else:
            _require(previous.status == "running", "progress chain continues after terminal status")
            _require(
                snapshot.segment_elapsed_scientific_ns > 0 and snapshot.segment_elapsed_wall_ns > 0,
                "post-genesis progress segment must advance both cumulative clocks",
            )
            _require(
                snapshot.predecessor_sha256 == previous.snapshot_sha256,
                "progress chain predecessor digest differs",
            )
            _require(
                snapshot.cumulative_elapsed_scientific_ns
                == previous.cumulative_elapsed_scientific_ns
                + snapshot.segment_elapsed_scientific_ns,
                "cumulative scientific time resets or differs from segment sum",
            )
            _require(
                snapshot.cumulative_elapsed_wall_ns
                == previous.cumulative_elapsed_wall_ns + snapshot.segment_elapsed_wall_ns,
                "cumulative wall time resets or differs from segment sum",
            )
            _require(
                0 <= snapshot.charged_calls - previous.charged_calls <= 16,
                "progress transition skips or reverses a charged batch",
            )
            _require(
                previous.last_authenticated_sealed_call
                <= snapshot.last_authenticated_sealed_call
                <= previous.last_authenticated_sealed_call + 16,
                "progress transition reverses or skips a sealed checkpoint",
            )
            if snapshot.status == "running":
                _require(
                    snapshot.charged_calls - previous.charged_calls == 16,
                    "running transition must seal exactly one complete adaptive batch",
                )
            advanced_calls = snapshot.charged_calls > previous.charged_calls
            if advanced_calls:
                for field in (
                    "query_ledger_sha256",
                    "query_identity_uniqueness_receipt_sha256",
                    "oracle_request_seal_manifest_sha256",
                    "oracle_response_seal_manifest_sha256",
                ):
                    _require(
                        getattr(snapshot, field) != getattr(previous, field),
                        f"progress {field} did not advance with charged calls",
                    )
            _require(
                snapshot.common_initial_copy_seal_sha256
                == previous.common_initial_copy_seal_sha256,
                "common-initial copy seal changed across progress segments",
            )
            _require(
                snapshot.timing_ledger_sha256 != previous.timing_ledger_sha256
                and snapshot.resource_usage_receipt_sha256
                != previous.resource_usage_receipt_sha256,
                "timing/resource evidence did not advance across a segment",
            )
        previous = snapshot
    assert previous is not None
    return previous


def progress_chain_receipt_bytes(
    plan: ScreenRunPlan,
    authority_sha256: str,
    snapshots: tuple[RunProgressSnapshot, ...],
) -> bytes:
    head = validate_progress_chain(plan, authority_sha256, snapshots)
    return _canonical_json_bytes(
        {
            "artifact": "evolutionary_kl_successor_progress_chain_receipt_v2",
            "authority_sha256": authority_sha256,
            "head_sha256": head.snapshot_sha256,
            "run_id": plan.run_id,
            "run_plan_sha256": plan.plan_sha256,
            "schema_version": 2,
            "snapshots": [snapshot.document() for snapshot in snapshots],
        }
    )


@dataclass(frozen=True, slots=True)
class ControlDirective:
    """Deterministic structural action for one external progress snapshot."""

    run_id: str
    status: ProgressStatus
    stop_trigger: StopTrigger
    hard_stop_reason: str | None
    must_stop: bool
    last_authenticated_sealed_call: int
    partial_batch_submitted_calls: int
    discard_unsealed_calls: int
    carry_from_call: int | None
    next_batch_index: int | None
    source_snapshot_sha256: str
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.run_id) is str and _RUN_ID_RE.fullmatch(self.run_id) is not None,
            "directive run ID invalid",
        )
        _require(
            type(self.status) is str
            and self.status in {"running", "completed", "budget_stopped", "hard_stopped"},
            "directive status invalid",
        )
        _require(
            self.stop_trigger in {"none", "call_budget", "wall_budget", "hard_stop"},
            "directive stop trigger invalid",
        )
        _require(
            self.hard_stop_reason is None or self.hard_stop_reason in HARD_STOP_REASONS,
            "directive hard-stop reason invalid",
        )
        _require(type(self.must_stop) is bool, "directive stop flag must be exact Boolean")
        _require(
            type(self.last_authenticated_sealed_call) is int
            and self.last_authenticated_sealed_call in CALL_CHECKPOINTS,
            "directive sealed checkpoint invalid",
        )
        _require(
            type(self.partial_batch_submitted_calls) is int
            and 0 <= self.partial_batch_submitted_calls <= 16
            and self.last_authenticated_sealed_call + self.partial_batch_submitted_calls <= 512,
            "directive partial-batch count invalid",
        )
        _require(
            type(self.discard_unsealed_calls) is int and 0 <= self.discard_unsealed_calls <= 16,
            "discarded unsealed count invalid",
        )
        _require(
            self.carry_from_call is None
            or (type(self.carry_from_call) is int and self.carry_from_call in CALL_CHECKPOINTS),
            "directive carry checkpoint invalid",
        )
        _require(
            self.next_batch_index is None
            or (type(self.next_batch_index) is int and 1 <= self.next_batch_index <= 28),
            "directive next batch invalid",
        )
        _sha256(self.source_snapshot_sha256, label="directive source snapshot")
        if self.must_stop:
            _require(self.status != "running", "running directive cannot stop")
            _require(self.next_batch_index is None, "stopped directive cannot schedule a batch")
            _require(
                self.carry_from_call == self.last_authenticated_sealed_call,
                "stopped directive must carry its exact sealed checkpoint",
            )
            _require(
                self.discard_unsealed_calls == self.partial_batch_submitted_calls,
                "stopped directive discard count differs from partial batch",
            )
            if self.status == "completed":
                _require(
                    self.stop_trigger == "call_budget"
                    and self.hard_stop_reason is None
                    and self.last_authenticated_sealed_call == 512
                    and self.partial_batch_submitted_calls == 0,
                    "completed directive contradicts call-budget completion",
                )
            elif self.status == "budget_stopped":
                _require(
                    self.stop_trigger == "wall_budget" and self.hard_stop_reason is None,
                    "budget directive contradicts wall stop",
                )
            else:
                _require(
                    self.stop_trigger == "hard_stop" and self.hard_stop_reason is not None,
                    "hard-stop directive lacks its frozen reason",
                )
        else:
            _require(
                self.status == "running"
                and self.stop_trigger == "none"
                and self.hard_stop_reason is None,
                "only an unstopped running directive may continue",
            )
            _require(
                self.carry_from_call is None
                and self.discard_unsealed_calls == 0
                and self.partial_batch_submitted_calls == 0,
                "running directive cannot carry or discard terminal values",
            )
            expected_batch = (self.last_authenticated_sealed_call - 64) // 16 + 1
            _require(
                self.next_batch_index == expected_batch,
                "running directive next batch contradicts sealed checkpoint",
            )
        _exact_false(self.execution_authorized, label="directive execution authorization")
        _exact_false(self.oracle_calls_authorized, label="directive oracle-call authorization")


@dataclass(frozen=True, slots=True)
class SuccessorScreenController:
    """Pure validator/controller shell; all side-effecting capabilities fail."""

    authority: NonAuthorizingCampaignAuthority
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False
    model_execution_available: Literal[False] = False
    q14_selector_available: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.authority) is NonAuthorizingCampaignAuthority,
            "controller authority type differs",
        )
        self.authority.__post_init__()
        for label, value in (
            ("execution authorization", self.execution_authorized),
            ("oracle-call authorization", self.oracle_calls_authorized),
            ("model execution", self.model_execution_available),
            ("q14 selector", self.q14_selector_available),
        ):
            _exact_false(value, label=f"controller {label}")

    def plans(self) -> tuple[ScreenRunPlan, ...]:
        self.__post_init__()
        return self.authority.plans

    def _plan_for_run(self, run_id: str) -> ScreenRunPlan:
        _require(type(run_id) is str, "run ID must be exact text")
        matches = tuple(plan for plan in self.authority.plans if plan.run_id == run_id)
        _require(len(matches) == 1, "run ID is not one exact screen slot")
        return matches[0]

    def structural_directive(
        self,
        snapshots: tuple[RunProgressSnapshot, ...],
    ) -> ControlDirective:
        """Validate a complete chain and derive stop/carry semantics without I/O."""

        self.__post_init__()
        _require(type(snapshots) is tuple and bool(snapshots), "progress chain is empty")
        snapshot = snapshots[-1]
        _require(type(snapshot) is RunProgressSnapshot, "progress snapshot exact type differs")
        plan = self._plan_for_run(snapshot.run_id)
        validate_progress_chain(plan, self.authority.authority_sha256, snapshots)
        must_stop = snapshot.status != "running"
        discarded = (
            snapshot.charged_calls - snapshot.last_authenticated_sealed_call if must_stop else 0
        )
        next_batch = None
        if not must_stop:
            next_batch = (snapshot.last_authenticated_sealed_call - 64) // 16 + 1
            _require(next_batch <= 28, "running progress has no adaptive batch remaining")
        return ControlDirective(
            run_id=snapshot.run_id,
            status=snapshot.status,
            stop_trigger=snapshot.stop_trigger,
            hard_stop_reason=snapshot.hard_stop_reason,
            must_stop=must_stop,
            last_authenticated_sealed_call=snapshot.last_authenticated_sealed_call,
            partial_batch_submitted_calls=snapshot.partial_batch_submitted_calls,
            discard_unsealed_calls=discarded,
            carry_from_call=snapshot.last_authenticated_sealed_call if must_stop else None,
            next_batch_index=next_batch,
            source_snapshot_sha256=snapshot.snapshot_sha256,  # type: ignore[arg-type]
        )

    def require_execution_authority(self) -> None:
        self.__post_init__()
        raise SuccessorExecutionUnavailable(
            "successor runtime v1 is non-authorizing and cannot dispatch execution"
        )

    def dispatch_run(self, *_: object, **__: object) -> None:
        self.require_execution_authority()

    def submit_oracle_query(self, *_: object, **__: object) -> None:
        self.require_execution_authority()

    def accept_oracle_response(self, *_: object, **__: object) -> None:
        self.require_execution_authority()

    def hidden_confirmation_plans(self) -> tuple[()]:
        self.__post_init__()
        raise SuccessorExecutionUnavailable(
            "hidden confirmation seeds are unavailable and have no runtime path"
        )


def carry_forward_fixed_checkpoints(
    sealed: tuple[tuple[int, float], ...],
    *,
    checkpoints: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    """Carry the last authenticated value without concealing the stop status."""

    _require(type(sealed) is tuple and bool(sealed), "sealed checkpoint series is empty")
    _require(
        type(checkpoints) is tuple
        and bool(checkpoints)
        and all(type(position) is int for position in checkpoints),
        "checkpoint grid invalid",
    )
    _require(
        tuple(sorted(set(checkpoints))) == checkpoints,
        "checkpoint grid must be strictly increasing",
    )
    parsed: list[tuple[int, float]] = []
    for index, row in enumerate(sealed):
        _require(
            type(row) is tuple and len(row) == 2,
            f"sealed checkpoint {index} must be exact pair",
        )
        position, value = row
        _require(
            type(position) is int and position in checkpoints,
            f"sealed checkpoint {index} position invalid",
        )
        _require(
            type(value) is float and math.isfinite(value),
            f"sealed checkpoint {index} value must be finite float",
        )
        parsed.append((position, value))
    positions = tuple(position for position, _ in parsed)
    _require(
        positions == checkpoints[: len(positions)],
        "sealed checkpoints must be a contiguous prefix",
    )
    final_value = parsed[-1][1]
    return (*tuple(parsed), *((position, final_value) for position in checkpoints[len(parsed) :]))


def carry_forward_call_checkpoints(
    sealed: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    """Carry a call-based incumbent over the exact 64..512 grid."""

    return carry_forward_fixed_checkpoints(sealed, checkpoints=CALL_CHECKPOINTS)


def carry_forward_wall_checkpoints(
    sealed: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    """Carry a wall-time incumbent over the exact 0..120 minute grid."""

    return carry_forward_fixed_checkpoints(sealed, checkpoints=WALL_CHECKPOINT_MINUTES)

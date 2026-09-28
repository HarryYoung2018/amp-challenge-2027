"""All-arm producer reporting for the blocked successor public screen.

The reducer covers exactly eight methods and five ablations over the five
published development seeds.  It retains missing, nonfinite, hard-stop, and
algorithmic-failure slots as explicit gate failures.  The output is producer
evidence only: no hidden-confirmation route, independent verification,
scientific acceptance, or production authorization exists here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from amp_challenge.evaluation.evolutionary_kl_successor_protocol_v2 import (
    ABLATION_IDS_V2,
    CONFIGURATION_IDS_V2,
    FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
    FULL_METHOD_ID_V2,
    METHOD_IDS_V2,
    SCREEN_SEEDS_V2,
    SuccessorProtocolV2,
)
from amp_challenge.evaluation.evolutionary_kl_successor_runtime_v1 import (
    ENGINEERING_EVIDENCE_CLASS,
    FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
    RUN_EVIDENCE_ROLES,
    SHARED_ARTIFACT_ROLES,
    AuthenticatedArtifact,
    NonAuthorizingCampaignAuthority,
    RunProgressSnapshot,
    ScreenRunPlan,
    SuccessorExecutionUnavailable,
    SuccessorRuntimeRegistryV1,
    _canonical_json_bytes,
    _document_sha256,
    progress_chain_receipt_bytes,
    validate_progress_chain,
)

BOOTSTRAP_ALGORITHM = "sha256_counter_rejection_uint64_be_seed_block_v1"
BOOTSTRAP_INDEX_DOMAIN = b"amp/evolutionary-kl/successor-v2/bootstrap-index/v1\0"
BOOTSTRAP_INVENTORY_DOMAIN = b"amp/evolutionary-kl/successor-v2/bootstrap-inventory/v1\0"
OUTCOME_ARTIFACT = "evolutionary_kl_successor_screen_run_outcome_v1"
INVENTORY_ARTIFACT = "evolutionary_kl_successor_screen_outcome_inventory_v1"
REPORT_ARTIFACT = "evolutionary_kl_successor_all_arm_screen_report_v1"

OutcomeStatus = Literal[
    "complete",
    "budget_stopped",
    "hard_stopped",
    "algorithmic_failure",
]
InventoryStatus = Literal[
    "complete",
    "budget_stopped",
    "hard_stopped",
    "algorithmic_failure",
    "missing",
    "nonfinite",
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FAILURE_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}\Z")


class SuccessorReportingV1Error(ValueError):
    """Raised when screen reporting input fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorReportingV1Error(message)


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be full lowercase SHA-256",
    )
    return value  # type: ignore[return-value]


def _slot(configuration_id: object, seed: object) -> tuple[str, int]:
    _require(
        type(configuration_id) is str and configuration_id in CONFIGURATION_IDS_V2,
        "configuration ID is not a frozen screen configuration",
    )
    _require(type(seed) is int and seed in SCREEN_SEEDS_V2, "seed is not a frozen screen seed")
    return configuration_id, seed


def _run_id(configuration_id: str, seed: int) -> str:
    return f"screen.{configuration_id}.seed-{seed}"


def _finite_primary(value: object, *, label: str) -> float:
    _require(
        type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0,
        f"{label} must be finite float in [0, 1]",
    )
    return value


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


def _type7(values: tuple[float, ...], probability: float) -> float:
    _require(bool(values), "type-7 values are empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _sign_p(successes: int, pairs: int) -> tuple[float, float]:
    upper = math.fsum(math.comb(pairs, count) for count in range(successes, pairs + 1)) / 2**pairs
    lower = math.fsum(math.comb(pairs, count) for count in range(successes + 1)) / 2**pairs
    return upper, min(1.0, 2.0 * min(lower, upper))


@lru_cache(maxsize=1)
def _bootstrap_draws() -> tuple[tuple[int, ...], ...]:
    blocks = len(SCREEN_SEEDS_V2)
    samples = 10_000
    seed = 20_260_909
    limit = 1 << 64
    rejection_limit = limit - limit % blocks
    result: list[tuple[int, ...]] = []
    for replicate in range(samples):
        draw: list[int] = []
        for position in range(blocks):
            nonce = 0
            while True:
                payload = (
                    BOOTSTRAP_INDEX_DOMAIN
                    + seed.to_bytes(8, "big")
                    + replicate.to_bytes(8, "big")
                    + position.to_bytes(8, "big")
                    + nonce.to_bytes(8, "big")
                )
                candidate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
                if candidate < rejection_limit:
                    draw.append(candidate % blocks)
                    break
                nonce += 1
        result.append(tuple(draw))
    return tuple(result)


def _bootstrap_inventory_sha256(draws: tuple[tuple[int, ...], ...]) -> str:
    payload = bytes(index for draw in draws for index in draw)
    header = (
        BOOTSTRAP_INVENTORY_DOMAIN
        + (20_260_909).to_bytes(8, "big")
        + len(draws).to_bytes(8, "big")
        + len(SCREEN_SEEDS_V2).to_bytes(8, "big")
    )
    return hashlib.sha256(header + payload).hexdigest()


@lru_cache(maxsize=512)
def _bootstrap_interval(effects: tuple[float, ...]) -> tuple[float, float]:
    _require(
        len(effects) == len(SCREEN_SEEDS_V2)
        and all(type(effect) is float and math.isfinite(effect) for effect in effects),
        "bootstrap effects differ from exact finite five-pair vector",
    )
    values = tuple(_mean(tuple(effects[index] for index in draw)) for draw in _bootstrap_draws())
    return _type7(values, 0.025), _type7(values, 0.975)


@dataclass(frozen=True, slots=True)
class ScreenRunOutcome:
    """One producer outcome carrying byte-authenticated run evidence."""

    configuration_id: str
    seed: int
    run_id: str
    status: OutcomeStatus
    normalized_primary_auc: float | None
    failure_code: str | None
    authority_sha256: str
    run_plan_sha256: str
    adapter_sha256: str
    asset_manifest_sha256: str
    shared_artifact_sha256s: tuple[tuple[str, str], ...]
    progress_chain: tuple[RunProgressSnapshot, ...]
    evidence_artifacts: tuple[AuthenticatedArtifact, ...]
    outcome_sha256: str | None
    scientific_evidence_accepted: Literal[False] = False
    production_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        configuration_id, seed = _slot(self.configuration_id, self.seed)
        _require(
            type(self.run_id) is str and self.run_id == _run_id(configuration_id, seed),
            "outcome run ID differs from slot",
        )
        _require(
            type(self.status) is str
            and self.status
            in {"complete", "budget_stopped", "hard_stopped", "algorithmic_failure"},
            "outcome status invalid",
        )
        if self.status in {"complete", "budget_stopped"}:
            _finite_primary(self.normalized_primary_auc, label="normalized primary AUC")
            _require(self.failure_code is None, "successful outcome cannot have failure code")
        else:
            _require(
                self.normalized_primary_auc is None
                or (
                    type(self.normalized_primary_auc) is float
                    and math.isfinite(self.normalized_primary_auc)
                    and 0.0 <= self.normalized_primary_auc <= 1.0
                ),
                "failed outcome primary AUC is invalid",
            )
            _require(
                type(self.failure_code) is str
                and _FAILURE_RE.fullmatch(self.failure_code) is not None,
                "failed outcome must retain a normalized failure code",
            )
        for label, value in (
            ("authority", self.authority_sha256),
            ("run plan", self.run_plan_sha256),
            ("adapter", self.adapter_sha256),
            ("asset manifest", self.asset_manifest_sha256),
        ):
            _sha256(value, label=label)
        _require(
            type(self.shared_artifact_sha256s) is tuple
            and tuple(role for role, _ in self.shared_artifact_sha256s) == SHARED_ARTIFACT_ROLES,
            "outcome shared artifact inventory differs",
        )
        for role, digest in self.shared_artifact_sha256s:
            _require(type(role) is str, "outcome shared artifact role invalid")
            _sha256(digest, label=f"outcome shared {role}")
        _require(
            type(self.progress_chain) is tuple
            and bool(self.progress_chain)
            and all(type(item) is RunProgressSnapshot for item in self.progress_chain),
            "outcome progress chain must be a non-empty exact tuple",
        )
        for snapshot in self.progress_chain:
            snapshot.__post_init__()
            _require(snapshot.run_id == self.run_id, "outcome progress run ID differs")
            _require(
                snapshot.run_plan_sha256 == self.run_plan_sha256,
                "outcome progress plan differs",
            )
            _require(
                snapshot.authority_sha256 == self.authority_sha256,
                "outcome progress authority differs",
            )
        head = self.progress_chain[-1]
        expected_progress_status = {
            "complete": "completed",
            "algorithmic_failure": "hard_stopped",
        }.get(self.status, self.status)
        _require(
            head.status == expected_progress_status,
            "outcome status differs from terminal progress",
        )
        _require(
            type(self.evidence_artifacts) is tuple
            and tuple(artifact.role for artifact in self.evidence_artifacts) == RUN_EVIDENCE_ROLES,
            "outcome run evidence inventory differs",
        )
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for artifact in self.evidence_artifacts:
            _require(type(artifact) is AuthenticatedArtifact, "run evidence type differs")
            artifact.__post_init__()
            _require(
                artifact.artifact_id == f"run:{self.run_id}:{artifact.role}",
                "run evidence artifact ID differs",
            )
            _require(artifact.artifact_id not in seen_ids, "run evidence ID is duplicated")
            _require(artifact.sha256 not in seen_digests, "run evidence payload is aliased")
            seen_ids.add(artifact.artifact_id)
            assert artifact.sha256 is not None
            seen_digests.add(artifact.sha256)
        _require(
            type(self.scientific_evidence_accepted) is bool
            and self.scientific_evidence_accepted is False,
            "outcome cannot accept scientific evidence",
        )
        _require(
            type(self.production_eligible) is bool and self.production_eligible is False,
            "outcome cannot become production eligible",
        )
        expected = _document_sha256(
            b"amp/evolutionary-kl/successor-v2/screen-outcome/v1\0",
            self.unsigned_document(),
        )
        if self.outcome_sha256 is None:
            object.__setattr__(self, "outcome_sha256", expected)
        else:
            _sha256(self.outcome_sha256, label="outcome")
            _require(self.outcome_sha256 == expected, "outcome self-digest differs")

    @property
    def gate_usable(self) -> bool:
        self.__post_init__()
        return self.status in {"complete", "budget_stopped"}

    def unsigned_document(self) -> dict[str, object]:
        return {
            "adapter_sha256": self.adapter_sha256,
            "artifact": OUTCOME_ARTIFACT,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "authority_sha256": self.authority_sha256,
            "configuration_id": self.configuration_id,
            "evidence_class": ENGINEERING_EVIDENCE_CLASS,
            "evidence_artifacts": [artifact.document() for artifact in self.evidence_artifacts],
            "failure_code": self.failure_code,
            "normalized_primary_auc": self.normalized_primary_auc,
            "production_eligible": False,
            "progress_head_sha256": self.progress_chain[-1].snapshot_sha256,
            "protocol_sha256": self.protocol_sha256,
            "run_id": self.run_id,
            "run_plan_sha256": self.run_plan_sha256,
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "schema_version": 1,
            "scientific_evidence_accepted": False,
            "seed": self.seed,
            "shared_artifact_sha256s": dict(self.shared_artifact_sha256s),
            "status": self.status,
        }

    @property
    def protocol_sha256(self) -> str:
        return FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256

    def document_bytes(self) -> bytes:
        self.__post_init__()
        document = self.unsigned_document()
        document["outcome_sha256"] = self.outcome_sha256
        return _canonical_json_bytes(document)


def build_bound_run_evidence_artifact(
    *,
    role: str,
    run_id: str,
    authority_sha256: str,
    run_plan_sha256: str,
    content: dict[str, object],
) -> AuthenticatedArtifact:
    """Build canonical run evidence; callers cannot supply a detached digest."""

    _require(role in RUN_EVIDENCE_ROLES, "run evidence role is not frozen")
    _require(role != "progress_chain_receipt", "use the progress-chain receipt bytes directly")
    _require(type(content) is dict, "run evidence content must be an exact dict")
    _sha256(authority_sha256, label="run evidence authority")
    _sha256(run_plan_sha256, label="run evidence plan")
    payload = _canonical_json_bytes(
        {
            "artifact": f"evolutionary_kl_successor_{role}_v2",
            "authority_sha256": authority_sha256,
            "content": content,
            "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "role": role,
            "run_id": run_id,
            "run_plan_sha256": run_plan_sha256,
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "schema_version": 2,
        }
    )
    return AuthenticatedArtifact.from_bytes(f"run:{run_id}:{role}", role, payload)


def build_progress_chain_evidence_artifact(
    *,
    run_plan: ScreenRunPlan,
    authority_sha256: str,
    progress_chain: tuple[RunProgressSnapshot, ...],
) -> AuthenticatedArtifact:
    payload = progress_chain_receipt_bytes(run_plan, authority_sha256, progress_chain)
    return AuthenticatedArtifact.from_bytes(
        f"run:{run_plan.run_id}:progress_chain_receipt",
        "progress_chain_receipt",
        payload,
    )


def _parse_bound_run_artifact(
    artifact: AuthenticatedArtifact,
    *,
    run_id: str,
    authority_sha256: str,
    run_plan_sha256: str,
) -> dict[str, object]:
    try:
        document = json.loads(artifact.payload.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuccessorReportingV1Error("run evidence is not strict canonical JSON") from error
    _require(type(document) is dict, "run evidence root must be an exact object")
    _require(
        set(document)
        == {
            "artifact",
            "authority_sha256",
            "content",
            "protocol_sha256",
            "role",
            "run_id",
            "run_plan_sha256",
            "runtime_sha256",
            "schema_version",
        },
        "run evidence schema differs",
    )
    _require(_canonical_json_bytes(document) == artifact.payload, "run evidence is not canonical")
    _require(
        document["artifact"] == f"evolutionary_kl_successor_{artifact.role}_v2",
        "run evidence artifact differs",
    )
    _require(document["authority_sha256"] == authority_sha256, "run evidence authority differs")
    _require(
        document["protocol_sha256"] == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
        "run evidence protocol differs",
    )
    _require(document["role"] == artifact.role, "run evidence role differs")
    _require(document["run_id"] == run_id, "run evidence run ID differs")
    _require(document["run_plan_sha256"] == run_plan_sha256, "run evidence plan differs")
    _require(
        document["runtime_sha256"] == FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
        "run evidence runtime differs",
    )
    _require(
        type(document["schema_version"]) is int and document["schema_version"] == 2,
        "run evidence schema version differs",
    )
    content = document["content"]
    _require(type(content) is dict, "run evidence content must be an exact object")
    return content  # type: ignore[return-value]


def _validate_run_evidence_artifacts(
    *,
    run_id: str,
    authority: NonAuthorizingCampaignAuthority,
    run_plan: ScreenRunPlan,
    progress_chain: tuple[RunProgressSnapshot, ...],
    evidence_artifacts: tuple[AuthenticatedArtifact, ...],
    status: OutcomeStatus,
    normalized_primary_auc: float | None,
    failure_code: str | None,
) -> None:
    authority_sha256 = authority.authority_sha256
    run_plan_sha256 = run_plan.plan_sha256
    head = validate_progress_chain(run_plan, authority_sha256, progress_chain)
    _require(
        type(evidence_artifacts) is tuple
        and tuple(artifact.role for artifact in evidence_artifacts) == RUN_EVIDENCE_ROLES,
        "run evidence role inventory differs",
    )
    artifacts = {artifact.role: artifact for artifact in evidence_artifacts}
    for artifact in evidence_artifacts:
        _require(type(artifact) is AuthenticatedArtifact, "run evidence exact type differs")
        artifact.__post_init__()
        _require(
            artifact.artifact_id == f"run:{run_id}:{artifact.role}",
            "run evidence artifact identity differs",
        )
    expected_progress = progress_chain_receipt_bytes(
        run_plan,
        authority_sha256,
        progress_chain,
    )
    _require(
        artifacts["progress_chain_receipt"].payload == expected_progress,
        "progress-chain receipt bytes differ from the complete predecessor chain",
    )
    expected_head_digests = {
        "query_ledger": head.query_ledger_sha256,
        "query_identity_uniqueness_receipt": (head.query_identity_uniqueness_receipt_sha256),
        "timing_ledger": head.timing_ledger_sha256,
        "common_initial_copy_seal": head.common_initial_copy_seal_sha256,
        "oracle_request_seal_manifest": head.oracle_request_seal_manifest_sha256,
        "oracle_response_seal_manifest": head.oracle_response_seal_manifest_sha256,
        "resource_usage_receipt": head.resource_usage_receipt_sha256,
    }
    for role, digest in expected_head_digests.items():
        _require(
            artifacts[role].sha256 == digest,
            f"terminal progress {role} digest differs from authenticated bytes",
        )
    content = {
        role: _parse_bound_run_artifact(
            artifact,
            run_id=run_id,
            authority_sha256=authority_sha256,
            run_plan_sha256=run_plan_sha256,
        )
        for role, artifact in artifacts.items()
        if role != "progress_chain_receipt"
    }
    _require(
        content["query_ledger"]
        == {"charged_calls": head.charged_calls, "entry_count": head.charged_calls},
        "query-ledger receipt count differs",
    )
    _require(
        content["query_identity_uniqueness_receipt"]
        == {"all_unique": True, "identity_count": head.charged_calls},
        "query-identity uniqueness receipt differs",
    )
    _require(
        content["timing_ledger"]
        == {
            "cumulative_elapsed_scientific_ns": head.cumulative_elapsed_scientific_ns,
            "cumulative_elapsed_wall_ns": head.cumulative_elapsed_wall_ns,
            "segment_count": len(progress_chain),
        },
        "timing-ledger receipt differs from progress chain",
    )
    shared = {slot.role: slot.sha256 for slot in authority.shared_artifacts}
    _require(
        content["common_initial_copy_seal"]
        == {
            "copied_initial_calls": 64,
            "source_response_seal_sha256": shared["common_initial_source_response_seal"],
        },
        "common-initial copy seal differs from shared source response",
    )
    _require(
        content["oracle_request_seal_manifest"] == {"sealed_request_count": head.charged_calls},
        "oracle request-seal count differs",
    )
    _require(
        content["oracle_response_seal_manifest"]
        == {"sealed_response_count": head.last_authenticated_sealed_call},
        "oracle response-seal count differs",
    )
    resource = content["resource_usage_receipt"]
    _require(
        set(resource)
        == {
            "a100_hours",
            "charged_calls",
            "cumulative_elapsed_scientific_ns",
            "cumulative_elapsed_wall_ns",
            "output_gib",
            "peak_gpu_memory_gib",
        },
        "resource-usage receipt schema differs",
    )
    _require(resource["charged_calls"] == head.charged_calls, "resource charged calls differ")
    _require(
        resource["cumulative_elapsed_scientific_ns"] == head.cumulative_elapsed_scientific_ns
        and resource["cumulative_elapsed_wall_ns"] == head.cumulative_elapsed_wall_ns,
        "resource cumulative clocks differ",
    )
    for key, ceiling in (
        ("a100_hours", 2.0),
        ("peak_gpu_memory_gib", 16.0),
        ("output_gib", 5.0),
    ):
        value = resource[key]
        _require(
            type(value) is float and math.isfinite(value) and 0.0 <= value <= ceiling,
            f"resource {key} exceeds its frozen ceiling",
        )
    _require(
        content["terminal_record"]
        == {
            "failure_code": failure_code,
            "last_authenticated_sealed_call": head.last_authenticated_sealed_call,
            "normalized_primary_auc": normalized_primary_auc,
            "status": status,
        },
        "terminal record differs from outcome",
    )
    _require(
        content["primary_evidence"] == {"normalized_primary_auc": normalized_primary_auc},
        "primary evidence differs from outcome",
    )
    _require(
        content["secondary_evidence"] == {"required_metric_count": 13, "retained": True},
        "secondary evidence receipt differs",
    )


def build_screen_run_outcome(
    *,
    configuration_id: str,
    seed: int,
    status: OutcomeStatus,
    normalized_primary_auc: float | None,
    failure_code: str | None,
    authority: NonAuthorizingCampaignAuthority,
    run_plan: ScreenRunPlan,
    progress_chain: tuple[RunProgressSnapshot, ...],
    evidence_artifacts: tuple[AuthenticatedArtifact, ...],
) -> ScreenRunOutcome:
    """Build a self-hashed producer outcome from actual receipt bytes."""

    run_id = _run_id(configuration_id, seed)
    authority.__post_init__()
    run_plan.__post_init__()
    _require(
        all(not slot.is_blocking for slot in authority.shared_artifacts)
        and not authority.independent_verifier.is_blocking,
        "blocked campaign authority cannot build an evidence-bearing outcome",
    )
    _require(
        run_plan.run_id == run_id
        and run_plan.configuration_id == configuration_id
        and run_plan.seed == seed,
        "outcome run plan differs from requested slot",
    )
    head = validate_progress_chain(run_plan, authority.authority_sha256, progress_chain)
    expected_progress_status = {
        "complete": "completed",
        "algorithmic_failure": "hard_stopped",
    }.get(status, status)
    _require(head.status == expected_progress_status, "outcome status differs from progress head")
    slots = {slot.role: slot for slot in run_plan.artifact_digests}
    _require(
        not slots["adapter"].is_blocking and not slots["asset_manifest"].is_blocking,
        "outcome run plan has blocking adapter or asset evidence",
    )
    _validate_run_evidence_artifacts(
        run_id=run_id,
        authority=authority,
        run_plan=run_plan,
        progress_chain=progress_chain,
        evidence_artifacts=evidence_artifacts,
        status=status,
        normalized_primary_auc=normalized_primary_auc,
        failure_code=failure_code,
    )
    values = {
        "configuration_id": configuration_id,
        "seed": seed,
        "run_id": run_id,
        "status": status,
        "normalized_primary_auc": normalized_primary_auc,
        "failure_code": failure_code,
        "authority_sha256": authority.authority_sha256,
        "run_plan_sha256": run_plan.plan_sha256,
        "adapter_sha256": slots["adapter"].sha256,
        "asset_manifest_sha256": slots["asset_manifest"].sha256,
        "shared_artifact_sha256s": tuple(
            (slot.role, slot.sha256) for slot in authority.shared_artifacts
        ),
        "progress_chain": progress_chain,
        "evidence_artifacts": evidence_artifacts,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }
    return ScreenRunOutcome(**values, outcome_sha256=None)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ScreenOutcomeInventoryRow:
    """One required slot, including explicit absence and nonfinite failure."""

    configuration_id: str
    seed: int
    run_id: str
    status: InventoryStatus
    outcome_sha256: str | None

    def __post_init__(self) -> None:
        configuration_id, seed = _slot(self.configuration_id, self.seed)
        _require(
            type(self.run_id) is str and self.run_id == _run_id(configuration_id, seed),
            "inventory run ID differs from slot",
        )
        _require(
            type(self.status) is str
            and self.status
            in {
                "complete",
                "budget_stopped",
                "hard_stopped",
                "algorithmic_failure",
                "missing",
                "nonfinite",
            },
            "inventory status invalid",
        )
        if self.status in {"missing", "nonfinite"}:
            _require(
                self.outcome_sha256 is None,
                "missing or nonfinite inventory slot cannot claim an outcome digest",
            )
        else:
            _sha256(self.outcome_sha256, label="inventory outcome")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "configuration_id": self.configuration_id,
            "outcome_sha256": self.outcome_sha256,
            "run_id": self.run_id,
            "seed": self.seed,
            "status": self.status,
        }


def _ordered_inventory_rows(
    rows: tuple[ScreenOutcomeInventoryRow, ...],
) -> tuple[ScreenOutcomeInventoryRow, ...]:
    _require(
        type(rows) is tuple and all(type(row) is ScreenOutcomeInventoryRow for row in rows),
        "inventory rows must be exact tuple members",
    )
    for row in rows:
        row.__post_init__()
    by_slot = {(row.configuration_id, row.seed): row for row in rows}
    _require(len(by_slot) == len(rows), "inventory duplicates a screen slot")
    expected = tuple(
        (configuration_id, seed)
        for configuration_id in CONFIGURATION_IDS_V2
        for seed in SCREEN_SEEDS_V2
    )
    _require(set(by_slot) == set(expected), "inventory does not cover exact 13-by-5 screen")
    return tuple(by_slot[slot] for slot in expected)


def screen_outcome_inventory_sha256(
    rows: tuple[ScreenOutcomeInventoryRow, ...],
) -> str:
    """Hash the exact 65-slot inventory, including failures and absences."""

    ordered = _ordered_inventory_rows(rows)
    return _document_sha256(
        b"amp/evolutionary-kl/successor-v2/screen-outcome-inventory/v1\0",
        {
            "artifact": INVENTORY_ARTIFACT,
            "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "rows": [row.document() for row in ordered],
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "schema_version": 1,
        },
    )


def build_screen_outcome_inventory(
    outcomes: tuple[ScreenRunOutcome, ...],
    *,
    absent_status_by_slot: dict[tuple[str, int], Literal["missing", "nonfinite"]],
) -> tuple[ScreenOutcomeInventoryRow, ...]:
    """Build a complete inventory only when every absent slot is classified."""

    _require(
        type(outcomes) is tuple and all(type(item) is ScreenRunOutcome for item in outcomes),
        "outcomes must be exact tuple members",
    )
    _require(type(absent_status_by_slot) is dict, "absent-status map must be exact dict")
    outcome_by_slot: dict[tuple[str, int], ScreenRunOutcome] = {}
    for outcome in outcomes:
        outcome.__post_init__()
        slot = (outcome.configuration_id, outcome.seed)
        _require(slot not in outcome_by_slot, "outcomes duplicate a screen slot")
        outcome_by_slot[slot] = outcome
    for slot, status in absent_status_by_slot.items():
        _require(
            type(slot) is tuple and len(slot) == 2,
            "absent-status map key must be an exact screen-slot pair",
        )
        _slot(slot[0], slot[1])
        _require(
            type(status) is str and status in {"missing", "nonfinite"},
            "absent slot status invalid",
        )
        _require(slot not in outcome_by_slot, "present outcome also classified absent")
    expected = tuple(
        (configuration_id, seed)
        for configuration_id in CONFIGURATION_IDS_V2
        for seed in SCREEN_SEEDS_V2
    )
    _require(
        set(outcome_by_slot) | set(absent_status_by_slot) == set(expected),
        "every absent screen slot must be explicitly classified",
    )
    rows: list[ScreenOutcomeInventoryRow] = []
    for configuration_id, seed in expected:
        slot = (configuration_id, seed)
        outcome = outcome_by_slot.get(slot)
        if outcome is None:
            rows.append(
                ScreenOutcomeInventoryRow(
                    configuration_id=configuration_id,
                    seed=seed,
                    run_id=_run_id(configuration_id, seed),
                    status=absent_status_by_slot[slot],
                    outcome_sha256=None,
                )
            )
        else:
            rows.append(
                ScreenOutcomeInventoryRow(
                    configuration_id=configuration_id,
                    seed=seed,
                    run_id=outcome.run_id,
                    status=outcome.status,
                    outcome_sha256=outcome.outcome_sha256,
                )
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class PairedScreenComparison:
    """Full-versus-one exact five-seed comparison."""

    comparator_id: str
    comparator_kind: Literal["method", "ablation"]
    seeds: tuple[int, ...]
    full_values: tuple[float | None, ...]
    comparator_values: tuple[float | None, ...]
    paired_effects: tuple[float | None, ...]
    unavailable_seeds: tuple[int, ...]
    tied_seeds: tuple[int, ...]
    statistics_available: bool
    positive_pairs: int | None
    sign_test_pairs: int | None
    exact_one_sided_sign_p: float | None
    exact_two_sided_sign_p: float | None
    paired_effect_mean: float | None
    paired_effect_median: float | None
    bootstrap_95_type7_lower: float | None
    bootstrap_95_type7_upper: float | None
    comparison_gate_pass: bool

    def __post_init__(self) -> None:
        _require(
            type(self.comparator_id) is str
            and self.comparator_id in (*METHOD_IDS_V2[:-1], *ABLATION_IDS_V2),
            "comparison target differs",
        )
        expected_kind = "method" if self.comparator_id in METHOD_IDS_V2 else "ablation"
        _require(
            type(self.comparator_kind) is str and self.comparator_kind == expected_kind,
            "comparison kind differs",
        )
        _require(
            type(self.seeds) is tuple and self.seeds == SCREEN_SEEDS_V2,
            "paired seed assignment differs",
        )
        for values, label in (
            (self.full_values, "full values"),
            (self.comparator_values, "comparator values"),
            (self.paired_effects, "paired effects"),
        ):
            _require(type(values) is tuple and len(values) == 5, f"{label} length differs")
            _require(
                all(
                    value is None or (type(value) is float and math.isfinite(value))
                    for value in values
                ),
                f"{label} contain invalid values",
            )
        _require(
            type(self.unavailable_seeds) is tuple
            and all(
                type(seed) is int and seed in SCREEN_SEEDS_V2 for seed in self.unavailable_seeds
            ),
            "unavailable seed list invalid",
        )
        _require(
            type(self.tied_seeds) is tuple
            and all(type(seed) is int and seed in SCREEN_SEEDS_V2 for seed in self.tied_seeds),
            "tied seed list invalid",
        )
        _require(type(self.statistics_available) is bool, "statistics flag must be Boolean")
        _require(type(self.comparison_gate_pass) is bool, "comparison gate must be Boolean")
        expected_effects = tuple(
            None if left is None or right is None else left - right
            for left, right in zip(self.full_values, self.comparator_values, strict=True)
        )
        _require(self.paired_effects == expected_effects, "paired effects differ from values")
        expected_unavailable = tuple(
            seed
            for seed, effect in zip(SCREEN_SEEDS_V2, expected_effects, strict=True)
            if effect is None
        )
        expected_ties = tuple(
            seed
            for seed, effect in zip(SCREEN_SEEDS_V2, expected_effects, strict=True)
            if effect == 0.0
        )
        _require(
            self.unavailable_seeds == expected_unavailable,
            "unavailable seeds differ from paired values",
        )
        _require(self.tied_seeds == expected_ties, "tied seeds differ from paired values")
        _require(
            self.statistics_available is (not expected_unavailable),
            "statistics availability differs from paired values",
        )
        numeric_fields = (
            self.exact_one_sided_sign_p,
            self.exact_two_sided_sign_p,
            self.paired_effect_mean,
            self.paired_effect_median,
            self.bootstrap_95_type7_lower,
            self.bootstrap_95_type7_upper,
        )
        if self.statistics_available:
            _require(not self.unavailable_seeds, "available statistics cannot omit seeds")
            _require(
                type(self.positive_pairs) is int and 0 <= self.positive_pairs <= 5,
                "positive-pair count invalid",
            )
            _require(
                type(self.sign_test_pairs) is int and 0 <= self.sign_test_pairs <= 5,
                "sign-test pair count invalid",
            )
            _require(
                all(type(value) is float and math.isfinite(value) for value in numeric_fields),
                "available statistics contain missing or nonfinite values",
            )
            finite_effects = tuple(effect for effect in expected_effects if effect is not None)
            expected_positives = sum(effect > 0.0 for effect in finite_effects)
            expected_sign_pairs = sum(effect != 0.0 for effect in finite_effects)
            expected_one_sided, expected_two_sided = _sign_p(
                expected_positives,
                expected_sign_pairs,
            )
            expected_bootstrap_lower, expected_bootstrap_upper = _bootstrap_interval(finite_effects)
            expected_median = _median(finite_effects)
            expected_gate = not expected_ties
            if self.comparator_kind == "ablation":
                expected_gate = (
                    expected_gate and expected_positives >= 4 and expected_median >= 0.05
                )
            _require(self.positive_pairs == expected_positives, "positive-pair count differs")
            _require(self.sign_test_pairs == expected_sign_pairs, "sign-test pair count differs")
            _require(
                self.exact_one_sided_sign_p == expected_one_sided
                and self.exact_two_sided_sign_p == expected_two_sided,
                "exact sign probabilities differ",
            )
            _require(
                self.paired_effect_mean == _mean(finite_effects)
                and self.paired_effect_median == expected_median,
                "paired effect summaries differ",
            )
            _require(
                self.bootstrap_95_type7_lower == expected_bootstrap_lower
                and self.bootstrap_95_type7_upper == expected_bootstrap_upper,
                "bootstrap interval differs",
            )
            _require(self.comparison_gate_pass is expected_gate, "comparison gate differs")
        else:
            _require(self.positive_pairs is None, "unavailable statistics have positive count")
            _require(self.sign_test_pairs is None, "unavailable statistics have sign-test count")
            _require(
                all(value is None for value in numeric_fields),
                "unavailable statistics must retain null summaries",
            )
            _require(not self.comparison_gate_pass, "unavailable comparison cannot pass")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "bootstrap_95_type7_lower": self.bootstrap_95_type7_lower,
            "bootstrap_95_type7_upper": self.bootstrap_95_type7_upper,
            "comparator_id": self.comparator_id,
            "comparator_kind": self.comparator_kind,
            "comparator_values": list(self.comparator_values),
            "comparison_gate_pass": self.comparison_gate_pass,
            "exact_one_sided_sign_p": self.exact_one_sided_sign_p,
            "exact_two_sided_sign_p": self.exact_two_sided_sign_p,
            "full_values": list(self.full_values),
            "paired_effect_mean": self.paired_effect_mean,
            "paired_effect_median": self.paired_effect_median,
            "paired_effects": list(self.paired_effects),
            "positive_pairs": self.positive_pairs,
            "sign_test_pairs": self.sign_test_pairs,
            "seeds": list(self.seeds),
            "statistics_available": self.statistics_available,
            "tied_seeds": list(self.tied_seeds),
            "unavailable_seeds": list(self.unavailable_seeds),
        }


def _paired_comparison(
    comparator_id: str,
    values: dict[tuple[str, int], float | None],
    draws: tuple[tuple[int, ...], ...],
) -> PairedScreenComparison:
    full = tuple(values[(FULL_METHOD_ID_V2, seed)] for seed in SCREEN_SEEDS_V2)
    comparator = tuple(values[(comparator_id, seed)] for seed in SCREEN_SEEDS_V2)
    effects = tuple(
        None if left is None or right is None else left - right
        for left, right in zip(full, comparator, strict=True)
    )
    unavailable = tuple(
        seed for seed, effect in zip(SCREEN_SEEDS_V2, effects, strict=True) if effect is None
    )
    ties = tuple(
        seed for seed, effect in zip(SCREEN_SEEDS_V2, effects, strict=True) if effect == 0.0
    )
    kind: Literal["method", "ablation"] = "method" if comparator_id in METHOD_IDS_V2 else "ablation"
    if unavailable:
        return PairedScreenComparison(
            comparator_id=comparator_id,
            comparator_kind=kind,
            seeds=SCREEN_SEEDS_V2,
            full_values=full,
            comparator_values=comparator,
            paired_effects=effects,
            unavailable_seeds=unavailable,
            tied_seeds=ties,
            statistics_available=False,
            positive_pairs=None,
            sign_test_pairs=None,
            exact_one_sided_sign_p=None,
            exact_two_sided_sign_p=None,
            paired_effect_mean=None,
            paired_effect_median=None,
            bootstrap_95_type7_lower=None,
            bootstrap_95_type7_upper=None,
            comparison_gate_pass=False,
        )
    finite_effects = tuple(effect for effect in effects if effect is not None)
    _require(len(finite_effects) == 5, "complete paired effect vector differs")
    positives = sum(effect > 0.0 for effect in finite_effects)
    sign_pairs = sum(effect != 0.0 for effect in finite_effects)
    one_sided, two_sided = _sign_p(positives, sign_pairs)
    _require(draws is _bootstrap_draws(), "bootstrap draw inventory object differs")
    bootstrap_lower, bootstrap_upper = _bootstrap_interval(finite_effects)
    paired_median = _median(finite_effects)
    gate_pass = not ties
    if kind == "ablation":
        gate_pass = gate_pass and positives >= 4 and paired_median >= 0.05
    return PairedScreenComparison(
        comparator_id=comparator_id,
        comparator_kind=kind,
        seeds=SCREEN_SEEDS_V2,
        full_values=full,
        comparator_values=comparator,
        paired_effects=effects,
        unavailable_seeds=(),
        tied_seeds=ties,
        statistics_available=True,
        positive_pairs=positives,
        sign_test_pairs=sign_pairs,
        exact_one_sided_sign_p=one_sided,
        exact_two_sided_sign_p=two_sided,
        paired_effect_mean=_mean(finite_effects),
        paired_effect_median=paired_median,
        bootstrap_95_type7_lower=bootstrap_lower,
        bootstrap_95_type7_upper=bootstrap_upper,
        comparison_gate_pass=gate_pass,
    )


@dataclass(frozen=True, slots=True)
class AllArmScreenReport:
    """Self-hashed descriptive report over the exact public screen."""

    authority_sha256: str
    outcome_inventory_sha256: str
    inventory_rows: tuple[ScreenOutcomeInventoryRow, ...]
    bootstrap_draw_inventory_sha256: str
    comparisons: tuple[PairedScreenComparison, ...]
    method_means: tuple[tuple[str, float | None], ...]
    failure_status_counts: tuple[tuple[str, int], ...]
    method_mean_rank_gate_pass: bool
    all_pair_integrity_gates_pass: bool
    all_ablation_gates_pass: bool
    producer_metric_gate_pass: bool
    authority_gate_pass: bool
    screen_gate_pass: bool
    report_sha256: str | None
    screen_is_descriptive_only: Literal[True] = True
    independent_verification_complete: Literal[False] = False
    scientific_evidence_accepted: Literal[False] = False
    comparative_claim_allowed: Literal[False] = False
    automatic_production_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="report authority")
        _sha256(self.outcome_inventory_sha256, label="report outcome inventory")
        _sha256(self.bootstrap_draw_inventory_sha256, label="bootstrap draw inventory")
        ordered_rows = _ordered_inventory_rows(self.inventory_rows)
        _require(ordered_rows == self.inventory_rows, "report inventory row order differs")
        _require(
            self.outcome_inventory_sha256 == screen_outcome_inventory_sha256(ordered_rows),
            "report outcome inventory digest differs",
        )
        _require(
            self.bootstrap_draw_inventory_sha256 == _bootstrap_inventory_sha256(_bootstrap_draws()),
            "report bootstrap draw inventory differs",
        )
        _require(
            type(self.comparisons) is tuple
            and all(type(item) is PairedScreenComparison for item in self.comparisons),
            "report comparisons must be exact tuple members",
        )
        expected_comparators = (*METHOD_IDS_V2[:-1], *ABLATION_IDS_V2)
        _require(
            tuple(item.comparator_id for item in self.comparisons) == expected_comparators,
            "report comparator inventory differs",
        )
        for item in self.comparisons:
            item.__post_init__()
        _require(
            type(self.method_means) is tuple
            and tuple(method_id for method_id, _ in self.method_means) == METHOD_IDS_V2,
            "report method means differ",
        )
        for _, value in self.method_means:
            _require(
                value is None or (type(value) is float and math.isfinite(value)),
                "report method mean invalid",
            )
        _require(
            type(self.failure_status_counts) is tuple
            and tuple(status for status, _ in self.failure_status_counts)
            == ("missing", "nonfinite", "hard_stopped", "algorithmic_failure"),
            "report failure status inventory differs",
        )
        _require(
            all(type(count) is int and count >= 0 for _, count in self.failure_status_counts),
            "report failure status count invalid",
        )
        expected_failure_counts = tuple(
            (status, sum(row.status == status for row in ordered_rows))
            for status in ("missing", "nonfinite", "hard_stopped", "algorithmic_failure")
        )
        _require(
            self.failure_status_counts == expected_failure_counts,
            "report failure counts differ from inventory",
        )
        for label, value in (
            ("method mean rank gate", self.method_mean_rank_gate_pass),
            ("pair integrity gates", self.all_pair_integrity_gates_pass),
            ("ablation gates", self.all_ablation_gates_pass),
            ("producer metric gate", self.producer_metric_gate_pass),
            ("authority gate", self.authority_gate_pass),
            ("screen gate", self.screen_gate_pass),
        ):
            _require(type(value) is bool, f"{label} must be exact Boolean")
        full_vectors = {comparison.full_values for comparison in self.comparisons}
        _require(len(full_vectors) == 1, "report full-method vectors differ across comparisons")
        full_values = next(iter(full_vectors))
        values_by_method = {
            FULL_METHOD_ID_V2: full_values,
            **{
                comparison.comparator_id: comparison.comparator_values
                for comparison in self.comparisons
                if comparison.comparator_kind == "method"
            },
        }
        expected_method_means = tuple(
            (
                method_id,
                (
                    _mean(
                        tuple(value for value in values_by_method[method_id] if value is not None)
                    )
                    if all(value is not None for value in values_by_method[method_id])
                    else None
                ),
            )
            for method_id in METHOD_IDS_V2
        )
        _require(self.method_means == expected_method_means, "method means differ from pairs")
        means = dict(expected_method_means)
        full_mean = means[FULL_METHOD_ID_V2]
        comparison_means = tuple(means[method_id] for method_id in METHOD_IDS_V2[:-1])
        expected_method_gate = (
            full_mean is not None
            and all(value is not None for value in comparison_means)
            and all(full_mean > value for value in comparison_means if value is not None)
        )
        expected_pair_gate = all(
            comparison.statistics_available and not comparison.tied_seeds
            for comparison in self.comparisons
        )
        expected_ablation_gate = all(
            comparison.comparison_gate_pass
            for comparison in self.comparisons
            if comparison.comparator_kind == "ablation"
        )
        no_failures = all(count == 0 for _, count in expected_failure_counts)
        _require(
            self.method_mean_rank_gate_pass is expected_method_gate,
            "method mean rank gate differs",
        )
        _require(
            self.all_pair_integrity_gates_pass is expected_pair_gate,
            "pair integrity gate differs",
        )
        _require(
            self.all_ablation_gates_pass is expected_ablation_gate,
            "ablation gate differs",
        )
        _require(
            self.producer_metric_gate_pass
            is (
                expected_method_gate
                and expected_pair_gate
                and expected_ablation_gate
                and no_failures
            ),
            "producer metric gate differs",
        )
        _require(
            self.authority_gate_pass is False,
            "non-authorizing producer report cannot pass its authority gate",
        )
        _require(
            self.screen_gate_pass is (self.producer_metric_gate_pass and self.authority_gate_pass),
            "screen gate differs",
        )
        _require(
            type(self.screen_is_descriptive_only) is bool
            and self.screen_is_descriptive_only is True,
            "screen must remain descriptive only",
        )
        for label, value in (
            ("independent verification", self.independent_verification_complete),
            ("scientific evidence acceptance", self.scientific_evidence_accepted),
            ("comparative claim", self.comparative_claim_allowed),
            ("automatic production eligibility", self.automatic_production_eligible),
        ):
            _require(type(value) is bool and value is False, f"{label} must remain false")
        expected_digest = _document_sha256(
            b"amp/evolutionary-kl/successor-v2/all-arm-screen-report/v1\0",
            self.unsigned_document(),
        )
        if self.report_sha256 is None:
            object.__setattr__(self, "report_sha256", expected_digest)
        else:
            _sha256(self.report_sha256, label="report")
            _require(self.report_sha256 == expected_digest, "report self-digest differs")

    def unsigned_document(self) -> dict[str, object]:
        return {
            "all_ablation_gates_pass": self.all_ablation_gates_pass,
            "all_pair_integrity_gates_pass": self.all_pair_integrity_gates_pass,
            "artifact": REPORT_ARTIFACT,
            "authority_sha256": self.authority_sha256,
            "automatic_production_eligible": False,
            "authority_gate_pass": self.authority_gate_pass,
            "bootstrap": {
                "algorithm": BOOTSTRAP_ALGORITHM,
                "draw_inventory_sha256": self.bootstrap_draw_inventory_sha256,
                "interval": "two_sided_95_percentile_type7",
                "samples": 10_000,
                "seed": 20_260_909,
            },
            "comparative_claim_allowed": False,
            "comparisons": [item.document() for item in self.comparisons],
            "evidence_class": ENGINEERING_EVIDENCE_CLASS,
            "failure_status_counts": dict(self.failure_status_counts),
            "independent_verification_complete": False,
            "inventory_rows": [row.document() for row in self.inventory_rows],
            "method_mean_rank_gate_pass": self.method_mean_rank_gate_pass,
            "method_means": [
                {"mean": value, "method_id": method_id} for method_id, value in self.method_means
            ],
            "outcome_inventory_sha256": self.outcome_inventory_sha256,
            "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "producer_metric_gate_pass": self.producer_metric_gate_pass,
            "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
            "schema_version": 1,
            "scientific_evidence_accepted": False,
            "screen_gate_pass": self.screen_gate_pass,
            "screen_is_descriptive_only": True,
        }

    def document_bytes(self) -> bytes:
        self.__post_init__()
        document = self.unsigned_document()
        document["report_sha256"] = self.report_sha256
        return _canonical_json_bytes(document)


def _build_report(
    *,
    authority_sha256: str,
    outcome_inventory_sha256: str,
    inventory_rows: tuple[ScreenOutcomeInventoryRow, ...],
    bootstrap_draw_inventory_sha256: str,
    comparisons: tuple[PairedScreenComparison, ...],
    method_means: tuple[tuple[str, float | None], ...],
    failure_status_counts: tuple[tuple[str, int], ...],
    method_mean_rank_gate_pass: bool,
    all_pair_integrity_gates_pass: bool,
    all_ablation_gates_pass: bool,
    producer_metric_gate_pass: bool,
    authority_gate_pass: bool,
    screen_gate_pass: bool,
) -> AllArmScreenReport:
    values = {
        "authority_sha256": authority_sha256,
        "outcome_inventory_sha256": outcome_inventory_sha256,
        "inventory_rows": inventory_rows,
        "bootstrap_draw_inventory_sha256": bootstrap_draw_inventory_sha256,
        "comparisons": comparisons,
        "method_means": method_means,
        "failure_status_counts": failure_status_counts,
        "method_mean_rank_gate_pass": method_mean_rank_gate_pass,
        "all_pair_integrity_gates_pass": all_pair_integrity_gates_pass,
        "all_ablation_gates_pass": all_ablation_gates_pass,
        "producer_metric_gate_pass": producer_metric_gate_pass,
        "authority_gate_pass": authority_gate_pass,
        "screen_gate_pass": screen_gate_pass,
        "screen_is_descriptive_only": True,
        "independent_verification_complete": False,
        "scientific_evidence_accepted": False,
        "comparative_claim_allowed": False,
        "automatic_production_eligible": False,
    }
    return AllArmScreenReport(**values, report_sha256=None)  # type: ignore[arg-type]


def reduce_all_arm_successor_screen(
    protocol: SuccessorProtocolV2,
    registry: SuccessorRuntimeRegistryV1,
    authority: NonAuthorizingCampaignAuthority,
    *,
    outcomes: tuple[ScreenRunOutcome, ...],
    outcome_inventory: tuple[ScreenOutcomeInventoryRow, ...],
    expected_outcome_inventory_sha256: str,
) -> AllArmScreenReport:
    """Produce a descriptive 13-by-5 report without accepting its evidence."""

    _require(type(protocol) is SuccessorProtocolV2, "report protocol exact type differs")
    protocol.__post_init__()
    _require(type(registry) is SuccessorRuntimeRegistryV1, "report registry exact type differs")
    registry.__post_init__()
    _require(
        type(authority) is NonAuthorizingCampaignAuthority,
        "report authority exact type differs",
    )
    authority.__post_init__()
    _require(
        authority.protocol_sha256 == protocol.protocol_sha256
        and authority.runtime_sha256 == registry.runtime_sha256,
        "report authority pins differ",
    )
    expected_inventory_digest = _sha256(
        expected_outcome_inventory_sha256,
        label="externally expected outcome inventory",
    )
    observed_inventory_digest = screen_outcome_inventory_sha256(outcome_inventory)
    _require(
        observed_inventory_digest == expected_inventory_digest,
        "outcome inventory digest differs from external expectation",
    )
    ordered_rows = _ordered_inventory_rows(outcome_inventory)
    _require(
        type(outcomes) is tuple and all(type(item) is ScreenRunOutcome for item in outcomes),
        "report outcomes must be exact tuple members",
    )
    outcome_by_slot: dict[tuple[str, int], ScreenRunOutcome] = {}
    authority_sha256 = authority.authority_sha256
    plan_by_slot = {(plan.configuration_id, plan.seed): plan for plan in authority.plans}
    _require(
        all(not slot.is_blocking for slot in authority.shared_artifacts),
        "blocked shared-artifact authority cannot admit outcomes",
    )
    _require(
        not authority.independent_verifier.is_blocking,
        "missing independent-verifier bytes cannot admit outcomes",
    )
    expected_shared = tuple((slot.role, slot.sha256) for slot in authority.shared_artifacts)
    for outcome in outcomes:
        outcome.__post_init__()
        slot = (outcome.configuration_id, outcome.seed)
        _require(slot not in outcome_by_slot, "report outcomes duplicate a screen slot")
        _require(
            outcome.authority_sha256 == authority_sha256,
            "outcome authority digest differs",
        )
        _require(
            outcome.run_plan_sha256 == plan_by_slot[slot].plan_sha256,
            "outcome run-plan digest differs",
        )
        plan = plan_by_slot[slot]
        plan_slots = {item.role: item for item in plan.artifact_digests}
        _require(
            outcome.adapter_sha256 == plan_slots["adapter"].sha256
            and outcome.asset_manifest_sha256 == plan_slots["asset_manifest"].sha256,
            "outcome adapter or asset bytes differ from authority-bound plan",
        )
        _require(
            outcome.shared_artifact_sha256s == expected_shared,
            "outcome shared artifacts differ from authority",
        )
        _validate_run_evidence_artifacts(
            run_id=outcome.run_id,
            authority=authority,
            run_plan=plan,
            progress_chain=outcome.progress_chain,
            evidence_artifacts=outcome.evidence_artifacts,
            status=outcome.status,
            normalized_primary_auc=outcome.normalized_primary_auc,
            failure_code=outcome.failure_code,
        )
        outcome_by_slot[slot] = outcome

    expected_present_slots = {
        (row.configuration_id, row.seed)
        for row in ordered_rows
        if row.status not in {"missing", "nonfinite"}
    }
    _require(
        set(outcome_by_slot) == expected_present_slots,
        "outcome set differs from inventory-present slots",
    )
    for row in ordered_rows:
        slot = (row.configuration_id, row.seed)
        outcome = outcome_by_slot.get(slot)
        if outcome is not None:
            _require(row.status == outcome.status, "inventory outcome status differs")
            _require(
                row.outcome_sha256 == outcome.outcome_sha256,
                "inventory outcome digest differs",
            )

    outcome_digests = tuple(outcome.outcome_sha256 for outcome in outcomes)
    _require(
        len(outcome_digests) == len(set(outcome_digests)),
        "outcome_sha256 must be unique across present screen slots",
    )
    for evidence_role in RUN_EVIDENCE_ROLES:
        digests = tuple(
            next(
                artifact.sha256
                for artifact in outcome.evidence_artifacts
                if artifact.role == evidence_role
            )
            for outcome in outcomes
        )
        _require(
            len(digests) == len(set(digests)),
            f"{evidence_role} evidence must be unique across present screen slots",
        )
    for configuration_id in CONFIGURATION_IDS_V2:
        config_outcomes = tuple(
            outcome for outcome in outcomes if outcome.configuration_id == configuration_id
        )
        for digest_field in ("adapter_sha256", "asset_manifest_sha256"):
            digests = {getattr(outcome, digest_field) for outcome in config_outcomes}
            _require(
                len(digests) <= 1,
                f"{configuration_id} changes {digest_field} across seeds",
            )

    values: dict[tuple[str, int], float | None] = {}
    for configuration_id in CONFIGURATION_IDS_V2:
        for seed in SCREEN_SEEDS_V2:
            outcome = outcome_by_slot.get((configuration_id, seed))
            values[(configuration_id, seed)] = (
                outcome.normalized_primary_auc
                if outcome is not None and outcome.gate_usable
                else None
            )

    draws = _bootstrap_draws()
    bootstrap_digest = _bootstrap_inventory_sha256(draws)
    comparator_ids = (*METHOD_IDS_V2[:-1], *ABLATION_IDS_V2)
    comparisons = tuple(
        _paired_comparison(comparator_id, values, draws) for comparator_id in comparator_ids
    )

    method_means_list: list[tuple[str, float | None]] = []
    for method_id in METHOD_IDS_V2:
        method_values = tuple(values[(method_id, seed)] for seed in SCREEN_SEEDS_V2)
        method_mean = (
            _mean(tuple(value for value in method_values if value is not None))
            if all(value is not None for value in method_values)
            else None
        )
        method_means_list.append((method_id, method_mean))
    method_means = tuple(method_means_list)
    means_by_method = dict(method_means)
    full_mean = means_by_method[FULL_METHOD_ID_V2]
    comparator_means = tuple(means_by_method[method_id] for method_id in METHOD_IDS_V2[:-1])
    method_rank_gate = (
        full_mean is not None
        and all(value is not None for value in comparator_means)
        and all(full_mean > value for value in comparator_means if value is not None)
    )
    pair_integrity_gate = all(
        comparison.statistics_available and not comparison.tied_seeds for comparison in comparisons
    )
    ablation_gate = all(
        comparison.comparison_gate_pass
        for comparison in comparisons
        if comparison.comparator_kind == "ablation"
    )
    failure_status_counts = tuple(
        (
            status,
            sum(row.status == status for row in ordered_rows),
        )
        for status in ("missing", "nonfinite", "hard_stopped", "algorithmic_failure")
    )
    no_retained_failures = all(count == 0 for _, count in failure_status_counts)
    producer_metric_gate = (
        method_rank_gate and pair_integrity_gate and ablation_gate and no_retained_failures
    )
    authority_gate = False
    screen_gate = producer_metric_gate and authority_gate
    return _build_report(
        authority_sha256=authority_sha256,
        outcome_inventory_sha256=observed_inventory_digest,
        inventory_rows=ordered_rows,
        bootstrap_draw_inventory_sha256=bootstrap_digest,
        comparisons=comparisons,
        method_means=method_means,
        failure_status_counts=failure_status_counts,
        method_mean_rank_gate_pass=method_rank_gate,
        all_pair_integrity_gates_pass=pair_integrity_gate,
        all_ablation_gates_pass=ablation_gate,
        producer_metric_gate_pass=producer_metric_gate,
        authority_gate_pass=authority_gate,
        screen_gate_pass=screen_gate,
    )


def reduce_hidden_confirmation(*_: object, **__: object) -> None:
    """Fail because hidden confirmation has no committed seed or execution path."""

    raise SuccessorExecutionUnavailable(
        "hidden confirmation is distinct and unavailable in successor runtime v1"
    )

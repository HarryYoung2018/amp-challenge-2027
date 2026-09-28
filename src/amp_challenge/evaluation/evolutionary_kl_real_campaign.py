"""Fail-closed registry and asset preflight for a future real AMP campaign.

The registry distinguishes reusable repository mechanics from complete method
adapters. This module authenticates small content-addressed asset manifests and
externally pinned trusted receipts. It never trains a model, opens an oracle,
evaluates a peptide, or authorizes the blocked research protocol.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    ABLATION_IDS,
    CONFIRMATION_METHOD_IDS,
    FROZEN_PROTOCOL_SHA256,
    METHOD_IDS,
    EvolutionaryKLProtocol,
    load_evolutionary_kl_protocol,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)

FROZEN_REAL_CAMPAIGN_REGISTRY_SHA256 = (
    "059a3b30770028a1d95d71e766df99937059dd60bd9216f212f2b83a2ec01da3"
)

REGISTRY_ARTIFACT = "evolutionary_kl_real_campaign_adapter_registry_v1"
REGISTRY_STATUS = "preflight_only_blocked_on_real_campaign_adapters_and_assets"
INVENTORY_STATUS = "frozen_preflight_input_not_execution_authority"
Phase = Literal["screen", "confirmation"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_PROVIDER_RE = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*:"
    r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*\Z"
)
_CONTENT_KINDS = frozenset({"canonical_json_manifest_v1", "canonical_sequence_id_jsonl_v1"})
_PHASES = frozenset({"screen", "confirmation"})
_REQUIRED_ASSET_IDS = frozenset(
    {
        "generator_training_sequence_inventory",
        "oracle_training_sequence_inventory",
        "screen_initial_and_reserve_sequence_inventory",
        "confirmation_initial_and_reserve_sequence_inventory",
        "base_fold_native_policy_checkpoints",
        "de_novo_checkpoint_aggregation_contract",
        "training_homology_exclusion_contract",
        "generator_oracle_training_provenance_separation_receipt",
        "organizer_reference_set_receipt",
        "hidden_oracle_contract",
        "oracle_objective_constraint_semantics",
        "oracle_missing_censoring_replicate_semantics",
        "common_initial_oracle_resource_accounting_receipt",
        "calibrated_joint_posterior_bundle",
        "generated_sequence_esm_contact_bundle",
        "persistent_search_ledger_contract",
        "terminal_evaluation_contract",
        "independent_verifier_bundle",
        "manifest_hashed_telemetry_contract",
        "external_trusted_receipt_issuer_contract",
        "hidden_confirmation_seed_reveal_contract",
        "sequestered_confirmation_oracle_access_control",
        "tuned_peptide_ga_adapter_bundle",
        "categorical_diffusion_posthoc_adapter_bundle",
        "diffusion_reward_kl_no_search_adapter_bundle",
        "arcadiamp_style_iterative_d3pm_adapter_bundle",
        "tr2d2_style_tree_offpolicy_adapter_bundle",
        "mp2d_style_inference_search_adapter_bundle",
        "ga_endpoint_distillation_no_kg_adapter_bundle",
        "counterfactual_softkg_evolutionary_diffusion_adapter_bundle",
    }
)
_REUSE_ASSET_IDS = frozenset(
    {
        "categorical_diffusion_corpus_v1",
        "native_diffusion_v1_development_projection",
        "fold4_blind_candidate_pool_v1",
        "candidate_activity_scoring_v1",
        "candidate_activity_mean_ledger_v1",
        "candidate_diversity_novelty_v1",
        "union_activity_replay_v1",
        "known_sequence_esm320_v1",
    }
)
_EXTERNAL_SOURCE_IDS = ("arcadiamp", "tr2d2", "mp2d", "equiformer_v3")
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "asset_id",
        "status",
        "evidence_class",
        "origin_kind",
        "independent_audit_accepted",
        "production_input_eligible",
        "payload_sha256",
        "sequence_namespace",
        "visibility_scope",
        "protocol_sha256",
        "registry_sha256",
        "verifier_git_commit",
    }
)


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Matched per-run resource declaration copied from the frozen protocol."""

    profile: str
    slurm_account: str
    cpu_partition: str
    gpu_partition: str
    gpu_type: str
    nodes_per_run: int
    gpus_per_run: int
    cpus_per_run: int
    host_memory_gib: int
    max_peak_gpu_memory_gib: int
    scientific_wall_seconds: int
    outer_allowance_seconds: int
    unique_oracle_calls_per_run: int
    oracle_batch_size_cap: int
    array_concurrency_cap: int


@dataclass(frozen=True, slots=True)
class BatchingProfile:
    """Frozen upper bounds for every material batch axis."""

    rollout_batch_size_cap: int
    proposal_batch_size_cap: int
    surrogate_batch_size_cap: int
    kg_candidate_chunk_size_cap: int
    kg_fantasy_chunk_size_cap: int
    replay_sequence_batch_cap: int
    replay_token_batch_cap: int
    gradient_accumulation_steps: int


@dataclass(frozen=True, slots=True)
class AssetPolicy:
    """Trust and filesystem requirements for controller-supplied assets."""

    inventory_artifact: str
    trusted_receipt_artifact: str
    accepted_status: str
    accepted_evidence_class: str
    allowed_origin_kinds: tuple[str, ...]
    forbidden_origin_tokens: tuple[str, ...]
    require_external_inventory_sha256: bool
    require_external_payload_sha256: bool
    require_external_trusted_receipt_sha256: bool
    require_independent_audit_acceptance: bool
    require_production_input_eligibility: bool
    require_read_only_single_link_regular_files: bool
    reject_symbolic_link_chain: bool
    reject_exact_sequence_overlap_across_namespaces: bool
    maximum_asset_manifest_bytes: int


@dataclass(frozen=True, slots=True)
class AssetRequirement:
    """One content-addressed manifest required before a campaign phase."""

    asset_id: str
    required_phases: tuple[Phase, ...]
    content_kind: str
    sequence_namespace: str
    visibility_scope: str
    required_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReuseAsset:
    """Existing authenticated evidence that remains campaign-ineligible."""

    asset_id: str
    payload_sha256: str
    current_evidence_scope: str
    campaign_input_eligible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ExternalSourcePin:
    """Research candidate pin, not an authenticated runtime source snapshot."""

    source_id: str
    repository: str
    commit: str
    license_candidate: str
    source_authentication_status: str
    adapter_status: str
    reproduction_claim_allowed: bool


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """A real provider, a reusable core, or an explicitly missing adapter."""

    capability_id: str
    status: str
    providers: tuple[str, ...]
    scientific_campaign_ready: bool


@dataclass(frozen=True, slots=True)
class MethodAdapterSpec:
    """Typed declaration for one of the eight frozen method arms."""

    method_id: str
    implementation_status: str
    adapter_providers: tuple[str, ...]
    reusable_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    resource_profile: str
    scientific_campaign_ready: bool


@dataclass(frozen=True, slots=True)
class AblationAdapterSpec:
    """Typed declaration for one of the five frozen component ablations."""

    ablation_id: str
    base_method: str
    disabled_capability: str
    implementation_status: str
    adapter_providers: tuple[str, ...]
    resource_profile: str
    scientific_campaign_ready: bool


@dataclass(frozen=True, slots=True)
class RealCampaignRegistry:
    """Immutable, path-free description of the current integration boundary."""

    sha256: str
    artifact: str
    status: str
    protocol_sha256: str
    execution_authorized: bool
    oracle_calls_authorized: bool
    scientific_evidence_accepted: bool
    automatic_production_eligible: bool
    biological_superiority_claim_allowed: bool
    resources: ResourceProfile
    batching: BatchingProfile
    asset_policy: AssetPolicy
    required_assets: tuple[AssetRequirement, ...]
    reuse_assets: tuple[ReuseAsset, ...]
    external_sources: tuple[ExternalSourcePin, ...]
    capabilities: tuple[CapabilitySpec, ...]
    methods: tuple[MethodAdapterSpec, ...]
    ablations: tuple[AblationAdapterSpec, ...]

    @property
    def configuration_ids(self) -> tuple[str, ...]:
        return tuple(method.method_id for method in self.methods) + tuple(
            ablation.ablation_id for ablation in self.ablations
        )

    @property
    def required_asset_by_id(self) -> Mapping[str, AssetRequirement]:
        return {item.asset_id: item for item in self.required_assets}

    @property
    def capability_by_id(self) -> Mapping[str, CapabilitySpec]:
        return {item.capability_id: item for item in self.capabilities}


@dataclass(frozen=True, slots=True)
class ExpectedAssetDigests:
    """Controller-authoritative hashes supplied outside an asset inventory."""

    payload_sha256: str
    trusted_receipt_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.payload_sha256, label="expected asset payload SHA-256")
        _sha256(
            self.trusted_receipt_sha256,
            label="expected asset trusted receipt SHA-256",
        )
        if self.payload_sha256 == self.trusted_receipt_sha256:
            raise ValueError("asset payload and trusted receipt SHA-256 must differ")


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """Path-free identity of one asset authenticated by the preflight."""

    asset_id: str
    payload_sha256: str
    trusted_receipt_sha256: str
    content_kind: str
    sequence_namespace: str
    byte_count: int
    sequence_count: int | None


@dataclass(frozen=True, slots=True)
class ArmReadiness:
    """Explicit reasons that one declared arm is not executable."""

    configuration_id: str
    implementation_status: str
    adapter_providers: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    blockers: tuple[str, ...]
    scientific_campaign_ready: bool


@dataclass(frozen=True, slots=True)
class RealCampaignPreflight:
    """Path-free blocked result; it is not an execution capability."""

    registry_sha256: str
    protocol_sha256: str
    phase: Phase
    inventory_sha256: str
    resolved_assets: tuple[ResolvedAsset, ...]
    missing_assets: tuple[str, ...]
    arms: tuple[ArmReadiness, ...]
    blockers: tuple[str, ...]
    execution_authorized: bool
    oracle_calls_authorized: bool
    scientific_evidence_accepted: bool
    automatic_production_eligible: bool
    biological_superiority_claim_allowed: bool

    def as_document(self) -> Mapping[str, object]:
        """Return a deterministic path-free engineering report."""

        return {
            "schema_version": 1,
            "artifact": "evolutionary_kl_real_campaign_preflight_v1",
            "status": "blocked",
            "registry_sha256": self.registry_sha256,
            "protocol_sha256": self.protocol_sha256,
            "phase": self.phase,
            "inventory_sha256": self.inventory_sha256,
            "resolved_assets": [
                {
                    "asset_id": asset.asset_id,
                    "payload_sha256": asset.payload_sha256,
                    "trusted_receipt_sha256": asset.trusted_receipt_sha256,
                    "content_kind": asset.content_kind,
                    "sequence_namespace": asset.sequence_namespace,
                    "byte_count": asset.byte_count,
                    "sequence_count": asset.sequence_count,
                }
                for asset in self.resolved_assets
            ],
            "missing_assets": list(self.missing_assets),
            "arms": [
                {
                    "configuration_id": arm.configuration_id,
                    "implementation_status": arm.implementation_status,
                    "adapter_providers": list(arm.adapter_providers),
                    "missing_capabilities": list(arm.missing_capabilities),
                    "blockers": list(arm.blockers),
                    "scientific_campaign_ready": arm.scientific_campaign_ready,
                }
                for arm in self.arms
            ],
            "blockers": list(self.blockers),
            "execution_authorized": self.execution_authorized,
            "oracle_calls_authorized": self.oracle_calls_authorized,
            "scientific_evidence_accepted": self.scientific_evidence_accepted,
            "automatic_production_eligible": self.automatic_production_eligible,
            "biological_superiority_claim_allowed": self.biological_superiority_claim_allowed,
        }

    def document_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_document())


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int, int, int]
    directory_chain: tuple[tuple[int, int, int, int, int], ...]
    allowed_root: _AllowedRoot
    maximum_bytes: int


@dataclass(frozen=True, slots=True)
class _AllowedRoot:
    path: Path
    directory_chain: tuple[tuple[int, int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _AuthenticatedRead:
    payload: bytes
    identity: tuple[int, int, int, int, int, int, int, int]
    directory_chain: tuple[tuple[int, int, int, int, int], ...]


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{name} keys changed; missing={missing}, extra={extra}")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _table_array(value: object, *, name: str) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or not value:
        raise TypeError(f"{name} must be a non-empty array of tables")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}[{index}] must be a table")
        result.append(item)
    return tuple(result)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be an integer at least {minimum}")
    return value


def _string_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list or (not value and not allow_empty):
        raise TypeError(f"{name} must be a {'possibly empty ' if allow_empty else ''}array")
    values = tuple(_string(item, name=f"{name} item") for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _identifier(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ValueError(f"{name} is not a canonical identifier")
    return result


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise ValueError(f"{label} must not be an all-zero placeholder")
    return value


def _load_config_bytes(path: str | Path) -> tuple[bytes, str]:
    candidate = _absolute_lexical_path(path, label="registry config")
    descriptor, _ = _open_file_beneath_absolute_path(
        candidate,
        label="registry config",
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("registry config must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise ValueError("registry config changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _resource_profile(value: object) -> ResourceProfile:
    keys = frozenset(
        {
            "profile",
            "slurm_account",
            "cpu_partition",
            "gpu_partition",
            "gpu_type",
            "nodes_per_run",
            "gpus_per_run",
            "cpus_per_run",
            "host_memory_gib",
            "max_peak_gpu_memory_gib",
            "scientific_wall_seconds",
            "outer_allowance_seconds",
            "unique_oracle_calls_per_run",
            "oracle_batch_size_cap",
            "array_concurrency_cap",
        }
    )
    table = _exact_mapping(value, name="resources", keys=keys)
    return ResourceProfile(
        profile=_string(table["profile"], name="resource profile"),
        slurm_account=_string(table["slurm_account"], name="Slurm account"),
        cpu_partition=_string(table["cpu_partition"], name="CPU partition"),
        gpu_partition=_string(table["gpu_partition"], name="GPU partition"),
        gpu_type=_string(table["gpu_type"], name="GPU type"),
        nodes_per_run=_integer(table["nodes_per_run"], name="nodes per run"),
        gpus_per_run=_integer(table["gpus_per_run"], name="GPUs per run"),
        cpus_per_run=_integer(table["cpus_per_run"], name="CPUs per run"),
        host_memory_gib=_integer(table["host_memory_gib"], name="host memory GiB"),
        max_peak_gpu_memory_gib=_integer(
            table["max_peak_gpu_memory_gib"],
            name="maximum peak GPU memory GiB",
        ),
        scientific_wall_seconds=_integer(
            table["scientific_wall_seconds"],
            name="scientific wall seconds",
        ),
        outer_allowance_seconds=_integer(
            table["outer_allowance_seconds"],
            name="outer allowance seconds",
        ),
        unique_oracle_calls_per_run=_integer(
            table["unique_oracle_calls_per_run"],
            name="unique oracle calls per run",
        ),
        oracle_batch_size_cap=_integer(
            table["oracle_batch_size_cap"],
            name="oracle batch size cap",
        ),
        array_concurrency_cap=_integer(
            table["array_concurrency_cap"],
            name="array concurrency cap",
        ),
    )


def _batching_profile(value: object) -> BatchingProfile:
    keys = frozenset(
        {
            "rollout_batch_size_cap",
            "proposal_batch_size_cap",
            "surrogate_batch_size_cap",
            "kg_candidate_chunk_size_cap",
            "kg_fantasy_chunk_size_cap",
            "replay_sequence_batch_cap",
            "replay_token_batch_cap",
            "gradient_accumulation_steps",
        }
    )
    table = _exact_mapping(value, name="batching", keys=keys)
    return BatchingProfile(
        **{key: _integer(table[key], name=key.replace("_", " ")) for key in sorted(keys)}
    )


def _asset_policy(value: object) -> AssetPolicy:
    keys = frozenset(
        {
            "inventory_artifact",
            "trusted_receipt_artifact",
            "accepted_status",
            "accepted_evidence_class",
            "allowed_origin_kinds",
            "forbidden_origin_tokens",
            "require_external_inventory_sha256",
            "require_external_payload_sha256",
            "require_external_trusted_receipt_sha256",
            "require_independent_audit_acceptance",
            "require_production_input_eligibility",
            "require_read_only_single_link_regular_files",
            "reject_symbolic_link_chain",
            "reject_exact_sequence_overlap_across_namespaces",
            "maximum_asset_manifest_bytes",
        }
    )
    table = _exact_mapping(value, name="asset_policy", keys=keys)
    boolean_fields = (
        "require_external_inventory_sha256",
        "require_external_payload_sha256",
        "require_external_trusted_receipt_sha256",
        "require_independent_audit_acceptance",
        "require_production_input_eligibility",
        "require_read_only_single_link_regular_files",
        "reject_symbolic_link_chain",
        "reject_exact_sequence_overlap_across_namespaces",
    )
    booleans = {key: _boolean(table[key], name=key.replace("_", " ")) for key in boolean_fields}
    if not all(booleans.values()):
        raise ValueError("all real-campaign asset trust controls must remain enabled")
    allowed = _string_tuple(table["allowed_origin_kinds"], name="allowed origin kinds")
    forbidden = _string_tuple(
        table["forbidden_origin_tokens"],
        name="forbidden origin tokens",
    )
    if any(token != token.lower() for token in (*allowed, *forbidden)):
        raise ValueError("asset origin policy values must be lowercase")
    return AssetPolicy(
        inventory_artifact=_identifier(
            table["inventory_artifact"],
            name="inventory artifact",
        ),
        trusted_receipt_artifact=_identifier(
            table["trusted_receipt_artifact"],
            name="trusted receipt artifact",
        ),
        accepted_status=_identifier(table["accepted_status"], name="accepted status"),
        accepted_evidence_class=_identifier(
            table["accepted_evidence_class"],
            name="accepted evidence class",
        ),
        allowed_origin_kinds=allowed,
        forbidden_origin_tokens=forbidden,
        maximum_asset_manifest_bytes=_integer(
            table["maximum_asset_manifest_bytes"],
            name="maximum asset manifest bytes",
        ),
        **booleans,
    )


def _asset_requirements(value: object) -> tuple[AssetRequirement, ...]:
    keys = frozenset(
        {
            "id",
            "required_phases",
            "content_kind",
            "sequence_namespace",
            "visibility_scope",
            "required_by",
        }
    )
    result: list[AssetRequirement] = []
    for index, item in enumerate(_table_array(value, name="required_assets")):
        table = _exact_mapping(item, name=f"required_assets[{index}]", keys=keys)
        phases = _string_tuple(
            table["required_phases"],
            name=f"required_assets[{index}].required_phases",
        )
        if not set(phases).issubset(_PHASES):
            raise ValueError("required asset contains an unknown campaign phase")
        content_kind = _string(
            table["content_kind"],
            name=f"required_assets[{index}].content_kind",
        )
        if content_kind not in _CONTENT_KINDS:
            raise ValueError("required asset contains an unknown content kind")
        namespace = _identifier(
            table["sequence_namespace"],
            name=f"required_assets[{index}].sequence_namespace",
        )
        if (content_kind == "canonical_sequence_id_jsonl_v1") != (namespace != "none"):
            raise ValueError("sequence asset kind and namespace are inconsistent")
        result.append(
            AssetRequirement(
                asset_id=_identifier(table["id"], name=f"required_assets[{index}].id"),
                required_phases=tuple(phases),  # type: ignore[arg-type]
                content_kind=content_kind,
                sequence_namespace=namespace,
                visibility_scope=_identifier(
                    table["visibility_scope"],
                    name=f"required_assets[{index}].visibility_scope",
                ),
                required_by=_string_tuple(
                    table["required_by"],
                    name=f"required_assets[{index}].required_by",
                ),
            )
        )
    if {item.asset_id for item in result} != _REQUIRED_ASSET_IDS:
        raise ValueError("real-campaign required asset inventory changed")
    sequence_namespaces = [
        item.sequence_namespace for item in result if item.sequence_namespace != "none"
    ]
    if len(set(sequence_namespaces)) != len(sequence_namespaces):
        raise ValueError("sequence namespaces must be unique across required assets")
    return tuple(result)


def _reuse_assets(value: object) -> tuple[ReuseAsset, ...]:
    keys = frozenset(
        {
            "id",
            "payload_sha256",
            "current_evidence_scope",
            "campaign_input_eligible",
            "reason",
        }
    )
    result: list[ReuseAsset] = []
    for index, item in enumerate(_table_array(value, name="reuse_assets")):
        table = _exact_mapping(item, name=f"reuse_assets[{index}]", keys=keys)
        eligible = _boolean(
            table["campaign_input_eligible"],
            name=f"reuse_assets[{index}].campaign_input_eligible",
        )
        if eligible:
            raise ValueError("existing reuse evidence must remain campaign-ineligible")
        result.append(
            ReuseAsset(
                asset_id=_identifier(table["id"], name=f"reuse_assets[{index}].id"),
                payload_sha256=_sha256(
                    table["payload_sha256"],
                    label=f"reuse_assets[{index}] payload SHA-256",
                ),
                current_evidence_scope=_identifier(
                    table["current_evidence_scope"],
                    name=f"reuse_assets[{index}].current_evidence_scope",
                ),
                campaign_input_eligible=eligible,
                reason=_identifier(
                    table["reason"],
                    name=f"reuse_assets[{index}].reason",
                ),
            )
        )
    if {item.asset_id for item in result} != _REUSE_ASSET_IDS:
        raise ValueError("existing reuse asset census changed")
    return tuple(result)


def _external_sources(value: object) -> tuple[ExternalSourcePin, ...]:
    keys = frozenset(
        {
            "id",
            "repository",
            "commit",
            "license_candidate",
            "source_authentication_status",
            "adapter_status",
            "reproduction_claim_allowed",
        }
    )
    result: list[ExternalSourcePin] = []
    for index, item in enumerate(_table_array(value, name="external_sources")):
        table = _exact_mapping(item, name=f"external_sources[{index}]", keys=keys)
        source_id = _identifier(table["id"], name=f"external_sources[{index}].id")
        commit = _string(table["commit"], name=f"external_sources[{index}].commit")
        if source_id != "mp2d" and _GIT_SHA_RE.fullmatch(commit) is None:
            raise ValueError(f"{source_id} must retain a full candidate Git pin")
        if source_id == "mp2d" and commit != "missing_execution_blocking":
            raise ValueError("MP2D must remain unpinned until official code is identified")
        source = ExternalSourcePin(
            source_id=source_id,
            repository=_string(
                table["repository"],
                name=f"external_sources[{index}].repository",
            ),
            commit=commit,
            license_candidate=_string(
                table["license_candidate"],
                name=f"external_sources[{index}].license_candidate",
            ),
            source_authentication_status=_identifier(
                table["source_authentication_status"],
                name=f"external_sources[{index}].source_authentication_status",
            ),
            adapter_status=_identifier(
                table["adapter_status"],
                name=f"external_sources[{index}].adapter_status",
            ),
            reproduction_claim_allowed=_boolean(
                table["reproduction_claim_allowed"],
                name=f"external_sources[{index}].reproduction_claim_allowed",
            ),
        )
        if source.reproduction_claim_allowed:
            raise ValueError("candidate source pins cannot authorize a reproduction claim")
        if not source.source_authentication_status.startswith("missing_") and source_id != "mp2d":
            raise ValueError("candidate source snapshots must remain independently unauthenticated")
        result.append(source)
    if tuple(item.source_id for item in result) != _EXTERNAL_SOURCE_IDS:
        raise ValueError("external source candidate pins changed")
    return tuple(result)


def _capabilities(value: object) -> tuple[CapabilitySpec, ...]:
    keys = frozenset({"id", "status", "providers", "scientific_campaign_ready"})
    result: list[CapabilitySpec] = []
    for index, item in enumerate(_table_array(value, name="capabilities")):
        table = _exact_mapping(item, name=f"capabilities[{index}]", keys=keys)
        status_value = _identifier(
            table["status"],
            name=f"capabilities[{index}].status",
        )
        providers = _string_tuple(
            table["providers"],
            name=f"capabilities[{index}].providers",
            allow_empty=True,
        )
        for provider in providers:
            if _PROVIDER_RE.fullmatch(provider) is None:
                raise ValueError(f"capability provider is not canonical: {provider}")
        missing = status_value.startswith("missing_")
        if missing == bool(providers):
            raise ValueError(
                "missing capabilities need no providers; reusable cores need providers"
            )
        ready = _boolean(
            table["scientific_campaign_ready"],
            name=f"capabilities[{index}].scientific_campaign_ready",
        )
        if ready:
            raise ValueError("no capability is accepted as real-campaign-ready in registry v1")
        result.append(
            CapabilitySpec(
                capability_id=_identifier(
                    table["id"],
                    name=f"capabilities[{index}].id",
                ),
                status=status_value,
                providers=providers,
                scientific_campaign_ready=ready,
            )
        )
    identifiers = [item.capability_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("capability identifiers must be unique")
    return tuple(result)


def _methods(
    value: object,
    *,
    capabilities: Mapping[str, CapabilitySpec],
    resource_profile: str,
) -> tuple[MethodAdapterSpec, ...]:
    keys = frozenset(
        {
            "id",
            "implementation_status",
            "adapter_providers",
            "reusable_capabilities",
            "missing_capabilities",
            "resource_profile",
            "scientific_campaign_ready",
        }
    )
    result: list[MethodAdapterSpec] = []
    for index, item in enumerate(_table_array(value, name="methods")):
        table = _exact_mapping(item, name=f"methods[{index}]", keys=keys)
        providers = _string_tuple(
            table["adapter_providers"],
            name=f"methods[{index}].adapter_providers",
            allow_empty=True,
        )
        reusable = _string_tuple(
            table["reusable_capabilities"],
            name=f"methods[{index}].reusable_capabilities",
        )
        missing = _string_tuple(
            table["missing_capabilities"],
            name=f"methods[{index}].missing_capabilities",
        )
        if set(reusable) & set(missing):
            raise ValueError("method reusable and missing capabilities overlap")
        if not set((*reusable, *missing)).issubset(capabilities):
            raise ValueError("method refers to an unknown capability")
        if any(capabilities[item].status.startswith("missing_") for item in reusable):
            raise ValueError("method marks a missing capability as reusable")
        if any(not capabilities[item].status.startswith("missing_") for item in missing):
            raise ValueError("method missing capability is not explicitly missing")
        status_value = _identifier(
            table["implementation_status"],
            name=f"methods[{index}].implementation_status",
        )
        ready = _boolean(
            table["scientific_campaign_ready"],
            name=f"methods[{index}].scientific_campaign_ready",
        )
        if (
            not status_value.startswith("missing_")
            or providers
            or ready
            or table["resource_profile"] != resource_profile
        ):
            raise ValueError(
                "method adapter must remain explicitly missing under matched resources"
            )
        result.append(
            MethodAdapterSpec(
                method_id=_identifier(table["id"], name=f"methods[{index}].id"),
                implementation_status=status_value,
                adapter_providers=providers,
                reusable_capabilities=reusable,
                missing_capabilities=missing,
                resource_profile=resource_profile,
                scientific_campaign_ready=ready,
            )
        )
    if tuple(item.method_id for item in result) != METHOD_IDS:
        raise ValueError("registry methods differ from the frozen eight arms")
    return tuple(result)


def _ablations(
    value: object,
    *,
    capabilities: Mapping[str, CapabilitySpec],
    resource_profile: str,
) -> tuple[AblationAdapterSpec, ...]:
    keys = frozenset(
        {
            "id",
            "base_method",
            "disabled_capability",
            "implementation_status",
            "adapter_providers",
            "resource_profile",
            "scientific_campaign_ready",
        }
    )
    result: list[AblationAdapterSpec] = []
    for index, item in enumerate(_table_array(value, name="ablations")):
        table = _exact_mapping(item, name=f"ablations[{index}]", keys=keys)
        providers = _string_tuple(
            table["adapter_providers"],
            name=f"ablations[{index}].adapter_providers",
            allow_empty=True,
        )
        status_value = _identifier(
            table["implementation_status"],
            name=f"ablations[{index}].implementation_status",
        )
        disabled = _identifier(
            table["disabled_capability"],
            name=f"ablations[{index}].disabled_capability",
        )
        ready = _boolean(
            table["scientific_campaign_ready"],
            name=f"ablations[{index}].scientific_campaign_ready",
        )
        if disabled not in capabilities:
            raise ValueError("ablation disables an unknown capability")
        if (
            table["base_method"] != "counterfactual_softkg_evolutionary_diffusion"
            or not status_value.startswith("missing_")
            or providers
            or ready
            or table["resource_profile"] != resource_profile
        ):
            raise ValueError("ablation adapter must remain an explicit missing composition")
        result.append(
            AblationAdapterSpec(
                ablation_id=_identifier(table["id"], name=f"ablations[{index}].id"),
                base_method="counterfactual_softkg_evolutionary_diffusion",
                disabled_capability=disabled,
                implementation_status=status_value,
                adapter_providers=providers,
                resource_profile=resource_profile,
                scientific_campaign_ready=ready,
            )
        )
    if tuple(item.ablation_id for item in result) != ABLATION_IDS:
        raise ValueError("registry ablations differ from the frozen five arms")
    if len({item.disabled_capability for item in result}) != len(result):
        raise ValueError("each ablation must disable one distinct component")
    return tuple(result)


def load_real_campaign_registry(path: str | Path) -> RealCampaignRegistry:
    """Load only the exact frozen, execution-disabled adapter registry."""

    payload, observed_sha256 = _load_config_bytes(path)
    if observed_sha256 != FROZEN_REAL_CAMPAIGN_REGISTRY_SHA256:
        raise ValueError("real-campaign registry SHA-256 differs from the frozen value")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("real-campaign registry is not canonical UTF-8 TOML") from error
    top_keys = frozenset(
        {
            "schema_version",
            "artifact",
            "status",
            "decision_date",
            "protocol_sha256",
            "execution_authorized",
            "oracle_calls_authorized",
            "scientific_evidence_accepted",
            "automatic_production_eligible",
            "biological_superiority_claim_allowed",
            "resources",
            "batching",
            "asset_policy",
            "required_assets",
            "reuse_assets",
            "external_sources",
            "capabilities",
            "methods",
            "ablations",
        }
    )
    root = _exact_mapping(raw, name="registry", keys=top_keys)
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise ValueError("real-campaign registry schema version must be one")
    if root["artifact"] != REGISTRY_ARTIFACT or root["status"] != REGISTRY_STATUS:
        raise ValueError("real-campaign registry identity or status changed")
    if root["decision_date"] != "2026-09-08":
        raise ValueError("real-campaign registry decision date changed")
    protocol_sha256 = _sha256(
        root["protocol_sha256"],
        label="registry protocol SHA-256",
    )
    if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError("registry is not bound to the frozen successor protocol")
    false_fields = (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "automatic_production_eligible",
        "biological_superiority_claim_allowed",
    )
    for field in false_fields:
        if _boolean(root[field], name=field.replace("_", " ")):
            raise ValueError(f"{field} must remain false")
    resources = _resource_profile(root["resources"])
    batching = _batching_profile(root["batching"])
    policy = _asset_policy(root["asset_policy"])
    requirements = _asset_requirements(root["required_assets"])
    reuse = _reuse_assets(root["reuse_assets"])
    external = _external_sources(root["external_sources"])
    capabilities = _capabilities(root["capabilities"])
    capability_mapping = {item.capability_id: item for item in capabilities}
    methods = _methods(
        root["methods"],
        capabilities=capability_mapping,
        resource_profile=resources.profile,
    )
    ablations = _ablations(
        root["ablations"],
        capabilities=capability_mapping,
        resource_profile=resources.profile,
    )
    return RealCampaignRegistry(
        sha256=observed_sha256,
        artifact=REGISTRY_ARTIFACT,
        status=REGISTRY_STATUS,
        protocol_sha256=protocol_sha256,
        execution_authorized=False,
        oracle_calls_authorized=False,
        scientific_evidence_accepted=False,
        automatic_production_eligible=False,
        biological_superiority_claim_allowed=False,
        resources=resources,
        batching=batching,
        asset_policy=policy,
        required_assets=requirements,
        reuse_assets=reuse,
        external_sources=external,
        capabilities=capabilities,
        methods=methods,
        ablations=ablations,
    )


def validate_registry_against_protocol(
    registry: RealCampaignRegistry,
    protocol: EvolutionaryKLProtocol,
) -> None:
    """Reject drift from the frozen campaign resources, arms, or blockers."""

    if type(registry) is not RealCampaignRegistry:
        raise TypeError("registry must be an exact RealCampaignRegistry")
    if type(protocol) is not EvolutionaryKLProtocol:
        raise TypeError("protocol must be an exact EvolutionaryKLProtocol")
    if registry.sha256 != FROZEN_REAL_CAMPAIGN_REGISTRY_SHA256:
        raise ValueError("registry object is not bound to the frozen registry bytes")
    if registry.protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError("registry object is not bound to the frozen protocol")
    if registry.configuration_ids != (*METHOD_IDS, *ABLATION_IDS):
        raise ValueError("registry configuration identifiers changed")
    limits = protocol.resource_limits
    resources = registry.resources
    expected_resources = (
        "matched_a100_v1",
        limits.slurm_account,
        limits.cpu_partition,
        limits.gpu_partition,
        limits.gpu_type,
        limits.nodes_per_run,
        limits.gpus_per_gpu_run,
        limits.cpus_per_run,
        limits.host_memory_gib,
        limits.max_peak_gpu_memory_gib,
        limits.scientific_wall_seconds,
        limits.outer_allowance_seconds,
        protocol.total_unique_calls,
        protocol.batching_limits.oracle_batch_size_cap,
        limits.array_concurrency_cap,
    )
    observed_resources = (
        resources.profile,
        resources.slurm_account,
        resources.cpu_partition,
        resources.gpu_partition,
        resources.gpu_type,
        resources.nodes_per_run,
        resources.gpus_per_run,
        resources.cpus_per_run,
        resources.host_memory_gib,
        resources.max_peak_gpu_memory_gib,
        resources.scientific_wall_seconds,
        resources.outer_allowance_seconds,
        resources.unique_oracle_calls_per_run,
        resources.oracle_batch_size_cap,
        resources.array_concurrency_cap,
    )
    if observed_resources != expected_resources:
        raise ValueError("registry resource profile differs from the frozen protocol")
    batching = registry.batching
    protocol_batching = protocol.batching_limits
    expected_batching = (
        protocol_batching.rollout_batch_size_cap,
        protocol_batching.proposal_batch_size_cap,
        protocol_batching.surrogate_batch_size_cap,
        protocol_batching.kg_candidate_chunk_size_cap,
        protocol_batching.kg_fantasy_chunk_size_cap,
        protocol_batching.replay_sequence_batch_cap,
        protocol_batching.replay_token_batch_cap,
        protocol_batching.gradient_accumulation_steps,
    )
    observed_batching = (
        batching.rollout_batch_size_cap,
        batching.proposal_batch_size_cap,
        batching.surrogate_batch_size_cap,
        batching.kg_candidate_chunk_size_cap,
        batching.kg_fantasy_chunk_size_cap,
        batching.replay_sequence_batch_cap,
        batching.replay_token_batch_cap,
        batching.gradient_accumulation_steps,
    )
    if observed_batching != expected_batching:
        raise ValueError("registry batching profile differs from the frozen protocol")
    if (
        registry.execution_authorized
        or registry.oracle_calls_authorized
        or registry.scientific_evidence_accepted
        or registry.automatic_production_eligible
        or registry.biological_superiority_claim_allowed
        or protocol.execution_authorized
        or protocol.automatic_production_eligible
        or protocol.biological_superiority_claim_allowed
    ):
        raise ValueError("registry and protocol must remain execution- and claim-disabled")
    if not protocol.execution_blockers:
        raise ValueError("frozen protocol unexpectedly has no execution blockers")


def validate_registry_provider_bindings(registry: RealCampaignRegistry) -> None:
    """Resolve declared reusable providers without accepting any method adapter."""

    if type(registry) is not RealCampaignRegistry:
        raise TypeError("registry must be an exact RealCampaignRegistry")
    seen: set[str] = set()
    for capability in registry.capabilities:
        for reference in capability.providers:
            if reference in seen:
                raise ValueError(f"provider is ambiguously reused: {reference}")
            seen.add(reference)
            module_name, qualname = reference.split(":", 1)
            module = importlib.import_module(module_name)
            provider: object = module
            for component in qualname.split("."):
                if component.startswith("_"):
                    raise ValueError(f"provider traverses a private symbol: {reference}")
                if not hasattr(provider, component):
                    raise ValueError(f"provider does not resolve: {reference}")
                provider = getattr(provider, component)
            if not callable(provider):
                raise ValueError(f"provider is not callable: {reference}")
    if any(method.adapter_providers for method in registry.methods) or any(
        ablation.adapter_providers for ablation in registry.ablations
    ):
        raise ValueError("registry v1 must not claim an implemented arm adapter")


def _absolute_lexical_path(path: str | Path, *, label: str) -> Path:
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise TypeError(f"{label} must be a filesystem path") from error
    if type(raw) is not str:
        raise TypeError(f"{label} must be a text filesystem path")
    candidate = Path(os.path.abspath(raw))
    if not candidate.is_absolute() or candidate.anchor != os.sep:
        raise ValueError(f"{label} must resolve to a canonical absolute POSIX path")
    return candidate


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _file_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("descriptor-relative no-follow directory opens are unavailable")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("descriptor-relative no-follow file opens are unavailable")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _open_directory_chain(
    path: Path,
    *,
    label: str,
    expected_prefix: tuple[tuple[int, int, int, int, int], ...] | None = None,
) -> tuple[int, tuple[tuple[int, int, int, int, int], ...]]:
    """Open an absolute directory one component at a time without following links."""

    flags = _directory_open_flags()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as error:
        raise ValueError(f"{label} filesystem anchor cannot be opened safely") from error
    try:
        chain = [_directory_identity(os.fstat(descriptor))]
    except BaseException:
        os.close(descriptor)
        raise

    def assert_expected_prefix() -> None:
        if expected_prefix is None:
            return
        compared = min(len(chain), len(expected_prefix))
        if tuple(chain[:compared]) != expected_prefix[:compared]:
            raise ValueError(f"{label} approved root changed before the leaf was opened")

    current = Path(os.sep)
    try:
        assert_expected_prefix()
        for component in path.parts[1:]:
            current = current / component
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{label} must not traverse a symbolic link, missing entry, "
                    f"or non-directory component: {current}"
                ) from error
            try:
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"{label} component is not a directory: {current}")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            chain.append(_directory_identity(metadata))
            assert_expected_prefix()
        if expected_prefix is not None and len(chain) < len(expected_prefix):
            raise ValueError(f"{label} is above its controller-approved root")
        return descriptor, tuple(chain)
    except BaseException:
        os.close(descriptor)
        raise


def _open_file_beneath_absolute_path(
    path: Path,
    *,
    label: str,
    expected_root: _AllowedRoot | None = None,
) -> tuple[int, tuple[tuple[int, int, int, int, int], ...]]:
    """Open a leaf through pinned parent descriptors, never through its full path."""

    if len(path.parts) < 2:
        raise ValueError(f"{label} must identify a file below the filesystem anchor")
    parent = Path(path.anchor).joinpath(*path.parts[1:-1])
    parent_descriptor, directory_chain = _open_directory_chain(
        parent,
        label=label,
        expected_prefix=(None if expected_root is None else expected_root.directory_chain),
    )
    try:
        try:
            descriptor = os.open(
                path.parts[-1],
                _file_open_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise ValueError(f"{label} leaf must exist and must not be a symbolic link") from error
        return descriptor, directory_chain
    finally:
        os.close(parent_descriptor)


def _assert_allowed_root_unchanged(root: _AllowedRoot, *, label: str) -> None:
    descriptor, current_chain = _open_directory_chain(root.path, label=label)
    os.close(descriptor)
    if current_chain != root.directory_chain:
        raise ValueError(f"{label} changed after it was authenticated")


def _normalize_allowed_roots(roots: Sequence[str | Path]) -> tuple[_AllowedRoot, ...]:
    if isinstance(roots, str | bytes | Path) or not roots:
        raise TypeError("allowed asset roots must be a non-empty sequence")
    result: list[_AllowedRoot] = []
    for index, raw in enumerate(roots):
        label = f"allowed asset root {index}"
        path = _absolute_lexical_path(raw, label=label)
        if any(item.path == path for item in result):
            raise ValueError("allowed asset roots must be unique")
        descriptor, directory_chain = _open_directory_chain(path, label=label)
        os.close(descriptor)
        root = _AllowedRoot(path=path, directory_chain=directory_chain)
        _assert_allowed_root_unchanged(root, label=label)
        result.append(root)
    return tuple(result)


def _select_allowed_root(
    path: Path, roots: tuple[_AllowedRoot, ...], *, label: str
) -> _AllowedRoot:
    matches = tuple(root for root in roots if path == root.path or path.is_relative_to(root.path))
    if not matches:
        raise ValueError(f"{label} escapes the controller-approved roots")
    return max(matches, key=lambda root: len(root.path.parts))


def _read_authenticated_path(
    path: Path,
    *,
    label: str,
    root: _AllowedRoot,
    maximum_bytes: int,
) -> _AuthenticatedRead:
    descriptor, directory_chain = _open_file_beneath_absolute_path(
        path,
        label=label,
        expected_root=root,
    )
    try:
        root_depth = len(root.directory_chain)
        if (
            len(directory_chain) < root_depth
            or directory_chain[:root_depth] != root.directory_chain
        ):
            raise ValueError(f"{label} approved root changed before the leaf was read")
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_uid != os.geteuid():
            raise ValueError(f"{label} must be owned by the campaign account")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link")
        if mode & 0o222 or mode & 0o111 or not mode & 0o400:
            raise ValueError(f"{label} must be read-only and non-executable")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} size is outside the preflight manifest bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} was truncated while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _file_identity(before)
    if identity != _file_identity(after):
        raise ValueError(f"{label} changed while its pinned descriptor was read")
    return _AuthenticatedRead(
        payload=b"".join(chunks),
        identity=identity,
        directory_chain=directory_chain,
    )


def _snapshot_immutable_file(
    path: str | Path,
    *,
    label: str,
    roots: tuple[_AllowedRoot, ...],
    maximum_bytes: int,
) -> _Snapshot:
    candidate = _absolute_lexical_path(path, label=label)
    root = _select_allowed_root(candidate, roots, label=label)
    authenticated = _read_authenticated_path(
        candidate,
        label=label,
        root=root,
        maximum_bytes=maximum_bytes,
    )
    repeated = _read_authenticated_path(
        candidate,
        label=label,
        root=root,
        maximum_bytes=maximum_bytes,
    )
    if repeated != authenticated:
        raise ValueError(f"{label} changed while it was authenticated")
    return _Snapshot(
        path=candidate,
        payload=authenticated.payload,
        sha256=hashlib.sha256(authenticated.payload).hexdigest(),
        identity=authenticated.identity,
        directory_chain=authenticated.directory_chain,
        allowed_root=root,
        maximum_bytes=maximum_bytes,
    )


def _assert_snapshot_unchanged(snapshot: _Snapshot, *, label: str) -> None:
    current = _read_authenticated_path(
        snapshot.path,
        label=label,
        root=snapshot.allowed_root,
        maximum_bytes=snapshot.maximum_bytes,
    )
    if (
        current.identity != snapshot.identity
        or current.directory_chain != snapshot.directory_chain
        or current.payload != snapshot.payload
        or hashlib.sha256(current.payload).hexdigest() != snapshot.sha256
    ):
        raise ValueError(f"{label} changed after authentication")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains forbidden non-finite constant: {value}")


def _json_value(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _canonical_json_object(payload: bytes, *, label: str) -> Mapping[str, object]:
    value = _json_value(payload, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _forbidden_origin_text(value: object, *, policy: AssetPolicy, label: str) -> str:
    text = _string(value, name=label)
    lowered = text.lower()
    if any(token in lowered for token in policy.forbidden_origin_tokens):
        raise ValueError(f"{label} contains a forbidden non-real origin token")
    return text


def _sequence_ids(payload: bytes, *, label: str) -> tuple[str, ...]:
    if not payload.endswith(b"\n") or payload.startswith(b"\n"):
        raise ValueError(f"{label} must end in exactly one nonempty JSONL record")
    raw_lines = payload.splitlines()
    rows: list[Mapping[str, object]] = []
    identifiers: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        value = _json_value(raw_line, label=f"{label} row {index}")
        if not isinstance(value, Mapping) or set(value) != {"sequence_id"}:
            raise ValueError(f"{label} row {index} must contain only sequence_id")
        sequence_id = _sha256(
            value["sequence_id"],
            label=f"{label} row {index} sequence ID",
        )
        rows.append({"sequence_id": sequence_id})
        identifiers.append(sequence_id)
    values = tuple(identifiers)
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"{label} sequence IDs must be nonempty, sorted, and unique")
    if canonical_jsonl_bytes(rows) != payload:
        raise ValueError(f"{label} is not canonical JSONL")
    return values


def _validate_generic_manifest(
    payload: bytes,
    *,
    requirement: AssetRequirement,
    registry: RealCampaignRegistry,
) -> None:
    asset_id = requirement.asset_id
    manifest = _canonical_json_object(payload, label=f"{asset_id} payload manifest")
    required = {
        "schema_version",
        "artifact",
        "asset_id",
        "content_kind",
        "status",
        "evidence_class",
        "protocol_sha256",
        "registry_sha256",
        "bindings",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise ValueError(
            f"{asset_id} payload manifest schema differs; missing={missing}, extra={extra}"
        )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError(f"{asset_id} payload manifest schema version is invalid")
    artifact = _forbidden_origin_text(
        manifest["artifact"],
        policy=registry.asset_policy,
        label=f"{asset_id} payload artifact",
    )
    if artifact != f"{asset_id}_v1" or manifest["asset_id"] != asset_id:
        raise ValueError(f"{asset_id} payload manifest identity differs")
    if manifest["content_kind"] != requirement.content_kind:
        raise ValueError(f"{asset_id} payload manifest content kind differs")
    if manifest["status"] != registry.asset_policy.accepted_status:
        raise ValueError(f"{asset_id} payload is not accepted for real-campaign input")
    if manifest["evidence_class"] != registry.asset_policy.accepted_evidence_class:
        raise ValueError(f"{asset_id} payload has a non-production evidence class")
    if manifest["protocol_sha256"] != registry.protocol_sha256:
        raise ValueError(f"{asset_id} payload is bound to a different protocol")
    if manifest["registry_sha256"] != registry.sha256:
        raise ValueError(f"{asset_id} payload is bound to a different registry")
    bindings = manifest["bindings"]
    if type(bindings) is not dict or not bindings or len(bindings) > 128:
        raise ValueError(f"{asset_id} payload manifest requires substantive bindings")
    binding_digests: list[str] = []
    for key, value in bindings.items():
        _identifier(key, name=f"{asset_id} payload binding key")
        binding_digests.append(_sha256(value, label=f"{asset_id} payload binding {key}"))
    if len(set(binding_digests)) != len(binding_digests):
        raise ValueError(f"{asset_id} payload manifest reuses a binding digest")
    if registry.protocol_sha256 in binding_digests or registry.sha256 in binding_digests:
        raise ValueError(f"{asset_id} payload manifest substitutes metadata for content")


def _validate_trusted_receipt(
    payload: bytes,
    *,
    requirement: AssetRequirement,
    payload_sha256: str,
    registry: RealCampaignRegistry,
) -> None:
    receipt = _canonical_json_object(payload, label=f"{requirement.asset_id} trusted receipt")
    if set(receipt) != _RECEIPT_KEYS:
        missing = sorted(_RECEIPT_KEYS - set(receipt))
        extra = sorted(set(receipt) - _RECEIPT_KEYS)
        raise ValueError(
            f"{requirement.asset_id} trusted receipt keys changed; missing={missing}, extra={extra}"
        )
    policy = registry.asset_policy
    expected: Mapping[str, object] = {
        "schema_version": 1,
        "artifact": policy.trusted_receipt_artifact,
        "asset_id": requirement.asset_id,
        "status": policy.accepted_status,
        "evidence_class": policy.accepted_evidence_class,
        "independent_audit_accepted": True,
        "production_input_eligible": True,
        "payload_sha256": payload_sha256,
        "sequence_namespace": requirement.sequence_namespace,
        "visibility_scope": requirement.visibility_scope,
        "protocol_sha256": registry.protocol_sha256,
        "registry_sha256": registry.sha256,
    }
    for key, value in expected.items():
        if receipt[key] != value or type(receipt[key]) is not type(value):
            raise ValueError(f"{requirement.asset_id} trusted receipt field changed: {key}")
    origin = _forbidden_origin_text(
        receipt["origin_kind"],
        policy=policy,
        label=f"{requirement.asset_id} origin kind",
    )
    if origin not in policy.allowed_origin_kinds:
        raise ValueError(f"{requirement.asset_id} origin kind is not allowed")
    commit = receipt["verifier_git_commit"]
    if type(commit) is not str or _GIT_SHA_RE.fullmatch(commit) is None or commit == "0" * 40:
        raise ValueError(f"{requirement.asset_id} verifier Git commit is invalid")


def _inventory_entries(value: object) -> tuple[Mapping[str, str], ...]:
    if type(value) is not list:
        raise TypeError("asset inventory assets must be an array")
    result: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        table = _exact_mapping(
            item,
            name=f"inventory assets[{index}]",
            keys=frozenset({"asset_id", "payload_path", "trusted_receipt_path"}),
        )
        asset_id = _identifier(table["asset_id"], name=f"inventory assets[{index}].asset_id")
        if asset_id in seen:
            raise ValueError(f"asset inventory repeats asset ID: {asset_id}")
        seen.add(asset_id)
        payload_path = _string(
            table["payload_path"],
            name=f"inventory assets[{index}].payload_path",
        )
        receipt_path = _string(
            table["trusted_receipt_path"],
            name=f"inventory assets[{index}].trusted_receipt_path",
        )
        if not Path(payload_path).is_absolute() or not Path(receipt_path).is_absolute():
            raise ValueError("asset inventory paths must be absolute")
        if payload_path == receipt_path:
            raise ValueError("asset payload and trusted receipt paths must differ")
        result.append(
            {
                "asset_id": asset_id,
                "payload_path": payload_path,
                "trusted_receipt_path": receipt_path,
            }
        )
    if tuple(item["asset_id"] for item in result) != tuple(
        sorted(item["asset_id"] for item in result)
    ):
        raise ValueError("asset inventory entries must be sorted by asset ID")
    return tuple(result)


def _phase_requirements(
    registry: RealCampaignRegistry,
    phase: Phase,
) -> tuple[AssetRequirement, ...]:
    return tuple(item for item in registry.required_assets if phase in item.required_phases)


def _ablation_missing_capabilities(
    registry: RealCampaignRegistry,
    ablation: AblationAdapterSpec,
) -> tuple[str, ...]:
    base = next(method for method in registry.methods if method.method_id == ablation.base_method)
    return tuple(
        capability
        for capability in base.missing_capabilities
        if capability != ablation.disabled_capability
    )


def _arm_readiness(registry: RealCampaignRegistry) -> tuple[ArmReadiness, ...]:
    arms: list[ArmReadiness] = []
    for method in registry.methods:
        blockers = (
            f"implementation:{method.implementation_status}",
            *(f"missing_capability:{item}" for item in method.missing_capabilities),
        )
        arms.append(
            ArmReadiness(
                configuration_id=method.method_id,
                implementation_status=method.implementation_status,
                adapter_providers=method.adapter_providers,
                missing_capabilities=method.missing_capabilities,
                blockers=blockers,
                scientific_campaign_ready=False,
            )
        )
    for ablation in registry.ablations:
        missing = _ablation_missing_capabilities(registry, ablation)
        blockers = (
            f"implementation:{ablation.implementation_status}",
            *(f"missing_capability:{item}" for item in missing),
        )
        arms.append(
            ArmReadiness(
                configuration_id=ablation.ablation_id,
                implementation_status=ablation.implementation_status,
                adapter_providers=ablation.adapter_providers,
                missing_capabilities=missing,
                blockers=blockers,
                scientific_campaign_ready=False,
            )
        )
    return tuple(arms)


def preflight_real_campaign(
    *,
    registry_path: str | Path,
    protocol_path: str | Path,
    inventory_path: str | Path,
    expected_inventory_sha256: str,
    expected_asset_digests: Mapping[str, ExpectedAssetDigests],
    allowed_asset_roots: Sequence[str | Path],
    phase: Phase,
) -> RealCampaignPreflight:
    """Authenticate candidate inputs and return an always-blocked readiness report.

    Expected hashes are caller-supplied trust anchors. Asset-internal digests
    never authorize themselves. Missing phase assets are reported as blockers;
    malformed, unaccepted, aliased, mutable, path-escaping, or unexpected
    assets are rejected.
    """

    if type(phase) is not str or phase not in _PHASES:
        raise ValueError("phase must be exactly screen or confirmation")
    registry = load_real_campaign_registry(registry_path)
    protocol = load_evolutionary_kl_protocol(protocol_path)
    validate_registry_against_protocol(registry, protocol)
    validate_registry_provider_bindings(registry)
    roots = _normalize_allowed_roots(allowed_asset_roots)
    inventory_expected = _sha256(
        expected_inventory_sha256,
        label="expected asset inventory SHA-256",
    )
    inventory_snapshot = _snapshot_immutable_file(
        inventory_path,
        label="asset inventory",
        roots=roots,
        maximum_bytes=registry.asset_policy.maximum_asset_manifest_bytes,
    )
    if inventory_snapshot.sha256 != inventory_expected:
        raise ValueError("asset inventory differs from the external expected SHA-256")
    inventory = _canonical_json_object(
        inventory_snapshot.payload,
        label="asset inventory",
    )
    inventory_keys = frozenset(
        {
            "schema_version",
            "artifact",
            "status",
            "registry_sha256",
            "protocol_sha256",
            "phase",
            "assets",
        }
    )
    if set(inventory) != inventory_keys:
        missing = sorted(inventory_keys - set(inventory))
        extra = sorted(set(inventory) - inventory_keys)
        raise ValueError(f"asset inventory keys changed; missing={missing}, extra={extra}")
    exact_inventory_values: Mapping[str, object] = {
        "schema_version": 1,
        "artifact": registry.asset_policy.inventory_artifact,
        "status": INVENTORY_STATUS,
        "registry_sha256": registry.sha256,
        "protocol_sha256": registry.protocol_sha256,
        "phase": phase,
    }
    for key, expected in exact_inventory_values.items():
        if inventory[key] != expected or type(inventory[key]) is not type(expected):
            raise ValueError(f"asset inventory field changed: {key}")
    entries = _inventory_entries(inventory["assets"])
    requirements = _phase_requirements(registry, phase)
    requirement_by_id = {item.asset_id: item for item in requirements}
    observed_ids = {item["asset_id"] for item in entries}
    unexpected = sorted(observed_ids - set(requirement_by_id))
    if unexpected:
        raise ValueError(f"asset inventory contains out-of-phase or unknown assets: {unexpected}")
    if not isinstance(expected_asset_digests, Mapping):
        raise TypeError("expected asset digests must be a mapping")
    if any(type(key) is not str for key in expected_asset_digests):
        raise TypeError("expected asset digest keys must be strings")
    if set(expected_asset_digests) != observed_ids:
        missing = sorted(observed_ids - set(expected_asset_digests))
        extra = sorted(set(expected_asset_digests) - observed_ids)
        raise ValueError(
            f"external asset digest keys differ from inventory; missing={missing}, extra={extra}"
        )
    expectations: dict[str, ExpectedAssetDigests] = {}
    for asset_id, expectation in expected_asset_digests.items():
        if type(expectation) is not ExpectedAssetDigests:
            raise TypeError(f"expected digests for {asset_id} must be ExpectedAssetDigests")
        expectations[asset_id] = expectation
    if len({item.payload_sha256 for item in expectations.values()}) != len(expectations):
        raise ValueError("external expectations reuse a payload digest across asset IDs")
    if len({item.trusted_receipt_sha256 for item in expectations.values()}) != len(expectations):
        raise ValueError("external expectations reuse a receipt digest across asset IDs")

    resolved: list[ResolvedAsset] = []
    snapshots: list[tuple[_Snapshot, str]] = [(inventory_snapshot, "asset inventory")]
    occupied_inodes = {inventory_snapshot.identity[:2]}
    sequence_sets: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        asset_id = entry["asset_id"]
        requirement = requirement_by_id[asset_id]
        expected = expectations[asset_id]
        payload_snapshot = _snapshot_immutable_file(
            entry["payload_path"],
            label=f"{asset_id} payload",
            roots=roots,
            maximum_bytes=registry.asset_policy.maximum_asset_manifest_bytes,
        )
        receipt_snapshot = _snapshot_immutable_file(
            entry["trusted_receipt_path"],
            label=f"{asset_id} trusted receipt",
            roots=roots,
            maximum_bytes=registry.asset_policy.maximum_asset_manifest_bytes,
        )
        for snapshot, label in (
            (payload_snapshot, f"{asset_id} payload"),
            (receipt_snapshot, f"{asset_id} trusted receipt"),
        ):
            inode = snapshot.identity[:2]
            if inode in occupied_inodes:
                raise ValueError(f"{label} aliases an already authenticated file")
            occupied_inodes.add(inode)
            snapshots.append((snapshot, label))
        if payload_snapshot.sha256 != expected.payload_sha256:
            raise ValueError(f"{asset_id} payload differs from its external expected digest")
        if receipt_snapshot.sha256 != expected.trusted_receipt_sha256:
            raise ValueError(
                f"{asset_id} trusted receipt differs from its external expected digest"
            )
        sequence_count: int | None = None
        semantically_resolved = False
        if requirement.content_kind == "canonical_sequence_id_jsonl_v1":
            identifiers = _sequence_ids(
                payload_snapshot.payload,
                label=f"{asset_id} sequence inventory",
            )
            sequence_sets[requirement.sequence_namespace] = identifiers
            sequence_count = len(identifiers)
            semantically_resolved = True
        else:
            _validate_generic_manifest(
                payload_snapshot.payload,
                requirement=requirement,
                registry=registry,
            )
        _validate_trusted_receipt(
            receipt_snapshot.payload,
            requirement=requirement,
            payload_sha256=payload_snapshot.sha256,
            registry=registry,
        )
        if semantically_resolved:
            resolved.append(
                ResolvedAsset(
                    asset_id=asset_id,
                    payload_sha256=payload_snapshot.sha256,
                    trusted_receipt_sha256=receipt_snapshot.sha256,
                    content_kind=requirement.content_kind,
                    sequence_namespace=requirement.sequence_namespace,
                    byte_count=len(payload_snapshot.payload),
                    sequence_count=sequence_count,
                )
            )
    namespaces = sorted(sequence_sets)
    for left_index, left_namespace in enumerate(namespaces):
        left = set(sequence_sets[left_namespace])
        for right_namespace in namespaces[left_index + 1 :]:
            overlap = left.intersection(sequence_sets[right_namespace])
            if overlap:
                first = min(overlap)
                raise ValueError(
                    "exact sequence leakage across namespaces "
                    f"{left_namespace}/{right_namespace}: {first}"
                )
    for snapshot, label in snapshots:
        _assert_snapshot_unchanged(snapshot, label=label)
    for index, root in enumerate(roots):
        _assert_allowed_root_unchanged(root, label=f"allowed asset root {index}")

    resolved.sort(key=lambda item: item.asset_id)
    resolved_ids = {item.asset_id for item in resolved}
    missing_assets = tuple(sorted(set(requirement_by_id) - resolved_ids))
    arms = _arm_readiness(registry)
    blockers = [
        "registry:execution_authorized_false",
        "registry:oracle_calls_authorized_false",
        *(f"protocol:{item}" for item in protocol.execution_blockers),
        *(f"asset:missing:{item}" for item in missing_assets),
        *(f"arm:{arm.configuration_id}:{arm.implementation_status}" for arm in arms),
    ]
    if phase == "confirmation":
        blockers.extend(f"protocol:{item}" for item in protocol.confirmatory_claim_blockers)
        if (
            tuple(
                method.method_id
                for method in registry.methods
                if method.method_id in CONFIRMATION_METHOD_IDS
            )
            != CONFIRMATION_METHOD_IDS
        ):
            raise ValueError("confirmation method registry order changed")
    if len(set(blockers)) != len(blockers):
        raise ValueError("preflight blockers must be unique")
    return RealCampaignPreflight(
        registry_sha256=registry.sha256,
        protocol_sha256=registry.protocol_sha256,
        phase=phase,
        inventory_sha256=inventory_snapshot.sha256,
        resolved_assets=tuple(resolved),
        missing_assets=missing_assets,
        arms=arms,
        blockers=tuple(blockers),
        execution_authorized=False,
        oracle_calls_authorized=False,
        scientific_evidence_accepted=False,
        automatic_production_eligible=False,
        biological_superiority_claim_allowed=False,
    )


__all__ = [
    "FROZEN_REAL_CAMPAIGN_REGISTRY_SHA256",
    "INVENTORY_STATUS",
    "AblationAdapterSpec",
    "ArmReadiness",
    "AssetPolicy",
    "AssetRequirement",
    "BatchingProfile",
    "CapabilitySpec",
    "ExpectedAssetDigests",
    "ExternalSourcePin",
    "MethodAdapterSpec",
    "RealCampaignPreflight",
    "RealCampaignRegistry",
    "ResolvedAsset",
    "ResourceProfile",
    "ReuseAsset",
    "load_real_campaign_registry",
    "preflight_real_campaign",
    "validate_registry_against_protocol",
    "validate_registry_provider_bindings",
]

"""Sealed, label-isolated preflight for retrospective fixed-pool activity replay.

This adapter deliberately stops before a scientific campaign.  It authenticates
the accepted Gate-1 source through the sequential-v2 trusted-stage capability,
authenticates both accepted ESM twins and their independent receipt, and emits a
deterministic label-free census.  The immutable v1 config cannot authorize an
oracle reveal, a search run, a de-novo claim, or production promotion.
Publication and verification require a controller-authoritative stage digest and
the exact prepare-leaf seal inventory; artifact-internal values never authorize
themselves.

The ESM filesystem readers below are intentionally reused from
``union_oracle_ledger``.  They are the existing downstream consumer that checks
the accepted publication/semantic manifests, exact matrix/index identities,
receipt scope, symlink-free locations, and time-of-check/time-of-use identity.
This module never invokes that ledger's outcome-bearing replay builder.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.evaluation.sequential_v2_protocol import (
    FOLDS,
    ordered_rotations,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    PREPARE_ROLE,
    AuthenticatedLeafCapsule,
    StageManifestCapability,
    prepare_capability_from_capsule,
    verify_stage_manifest_capability,
)
from amp_challenge.evaluation.sequential_v2_staging import (
    ACCEPTED_GATE1_SOURCE_CONTRACT,
    PrepareCapability,
)
from amp_challenge.evaluation.union_oracle_ledger import (
    EmbeddingEvidenceConfig as _ExistingEmbeddingEvidenceConfig,
)
from amp_challenge.evaluation.union_oracle_ledger import (
    _assert_unchanged as _assert_embedding_snapshot_unchanged,
)
from amp_challenge.evaluation.union_oracle_ledger import (
    _read_embedding_index as _read_existing_embedding_index,
)
from amp_challenge.evaluation.union_oracle_ledger import (
    _verify_embedding_inputs as _verify_existing_embedding_inputs,
)
from amp_challenge.sequences import canonical_sequence_id

FROZEN_FIXED_POOL_ADAPTER_CONFIG_SHA256 = (
    "858d8a6cb81c31e16de00a8d8329e9cf3391b850aa0f5ff0b38099abd338c7a1"
)

SCHEMA_VERSION = 1
ADAPTER_ARTIFACT = "evolutionary_kl_fixed_pool_adapter_v1"
PREFLIGHT_ARTIFACT = "evolutionary_kl_fixed_pool_preflight_v1"
EVIDENCE_CLASS = "retrospective_real_label_fixed_pool_activity_only"
BLOCKED_STATUS = "preflight_only_blocked_on_authenticated_filtered_census"
PREFLIGHT_STATUS = "authenticated_filtered_pool_census_preflight_only"
TENTATIVE_UNIQUE_SEQUENCE_BUDGET = 96
EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256 = (
    "e5db5f6bbb17c6af789bcd85abbf2e54ad0f7d4479b40994ade90fd498763807"
)
OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
EXPECTED_CONDITIONAL_SUCCESSOR_CLAIMS = (
    "authenticated_sequence_filtered_pool_census_after_controller_bound_production_and_independent_audit",
    "retrospective_within_pool_activity_label_efficiency_after_separate_preregistration_and_oracle_protocol",
)
EXPECTED_CONDITIONAL_SUCCESSOR_CLAIM_PREREQUISITES = (
    "controller_authoritative_stage_digest_and_exact_prepare_leaf_inventory",
    "producer_independent_excluded_node_preflight_reconstruction_accepted",
    "separate_preregistration_before_any_outcome_reveal",
    "append_only_logical_oracle_call_ledger_and_matched_wall_time_budget",
    "terminal_outer_commitment_and_abstention_contract",
)
EXPECTED_FORBIDDEN_CLAIMS = (
    "de_novo_generation_quality",
    "diffusion_proposal_operator_performance",
    "trajectory_or_endpoint_kl_effectiveness",
    "endpoint_distillation_effectiveness",
    "toxicity_selectivity_or_safety",
    "biological_superiority",
    "generator_mixture_or_production_promotion",
)
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
EXPECTED_ROTATION_COUNT = 20
EXPECTED_OUTER_INFERENCE_UNITS = 5
EXPECTED_ROTATIONS_PER_OUTER_UNIT = 4

PREFLIGHT_PAYLOAD_PATHS = (
    "candidates.jsonl",
    "excluded-candidates.jsonl",
    "fold-census.jsonl",
    "inference-units.jsonl",
    "manifest.json",
    "rotations.jsonl",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "evidence_class",
        "decision_date",
        "execution_authorized",
        "oracle_reveal_authorized",
        "automatic_production_eligible",
        "independent_audit_accepted",
        "scientific_evidence_accepted",
        "activity_only",
        "de_novo_claim_allowed",
        "generator_or_search_claim_allowed",
        "biological_superiority_claim_allowed",
        "rotation_count",
        "outer_inference_units",
        "outer_folds",
        "objectives",
        "aggregation",
        "tentative_unique_sequence_budget_to_check",
        "tentative_unique_sequence_budget_status",
        "sequence_filter",
        "representations",
        "gate1",
        "esm",
        "provenance",
        "claims",
    }
)
_SEQUENCE_FILTER_KEYS = frozenset(
    {
        "alphabet",
        "min_length",
        "max_length",
        "canonical_sequence_function",
        "free_termini_and_modification_compliance_established",
        "chemical_metadata_status",
    }
)
_REPRESENTATION_KEYS = frozenset(
    {
        "descriptor_enabled",
        "descriptor_status",
        "accepted_esm_feature_count",
        "accepted_esm_use",
        "esm_pretraining_membership_independence_established",
        "spectral_contact_enabled",
        "spectral_contact_status",
    }
)
_GATE1_KEYS = frozenset(
    {
        "producer_job_id",
        "audit_job_id",
        "git_commit",
        "publication_top_sha256",
        "semantic_top_sha256",
        "examples_sha256",
        "folds_sha256",
        "independent_receipt_sha256",
        "expected_contexts",
        "expected_sequences",
        "expected_support_sequences",
        "expected_support_by_fold",
        "expected_support_ids_sha256",
        "expected_support_esm_id_row_pairs_sha256",
    }
)
_ESM_KEYS = frozenset(
    {
        "producer_job_id",
        "audit_job_id",
        "git_commit",
        "publication_top_sha256",
        "semantic_top_sha256",
        "index_sha256",
        "manifest_sha256",
        "matrix_sha256",
        "tensor_data_sha256",
        "sequence_ids_sha256",
        "independent_receipt_sha256",
        "records",
        "dimensions",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "gate1_twins_required",
        "esm_twins_required",
        "independent_receipts_required",
        "stage_global_seal_required_at_runtime",
        "controller_stage_and_prepare_authority_required_at_publish_and_verify",
        "prepare_role",
        "pool_outcome_role",
        "preflight_publication_is_path_free",
        "preflight_publication_requires_no_replace_phase_seal",
    }
)
_CLAIMS_KEYS = frozenset(
    {"conditional_successor", "conditional_successor_prerequisites", "forbidden"}
)

_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "rotation_id",
        "outer_fold",
        "pool_fold",
        "base_folds",
        "sequence_id",
        "sequence",
        "length",
        "esm_row_index",
        "prepare_leaf_seal_sha256",
        "sequence_filter_eligible",
        "chemical_compliance_established",
        "evidence_class",
    }
)
_ROTATION_FIELDS = frozenset(
    {
        "schema_version",
        "rotation_id",
        "outer_fold",
        "pool_fold",
        "base_folds",
        "prepare_role",
        "prepare_leaf_seal_sha256",
        "unfiltered_support_count",
        "filtered_candidate_count",
        "excluded_candidate_count",
        "filtered_candidate_ids_sha256",
        "filtered_candidate_esm_id_row_pairs_sha256",
    }
)
_FOLD_CENSUS_FIELDS = frozenset(
    {
        "schema_version",
        "pool_fold",
        "unfiltered_support_count",
        "filtered_candidate_count",
        "excluded_candidate_count",
        "filtered_candidate_ids_sha256",
        "filtered_candidate_esm_id_row_pairs_sha256",
        "tentative_96_unique_sequence_budget_census_sufficient",
    }
)
_INFERENCE_UNIT_FIELDS = frozenset(
    {
        "schema_version",
        "inference_unit",
        "outer_fold",
        "rotation_count",
        "rotation_ids",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "evidence_class",
        "activity_only",
        "config_sha256",
        "execution_authorized",
        "oracle_reveal_authorized",
        "automatic_production_eligible",
        "independent_audit_accepted",
        "scientific_evidence_accepted",
        "de_novo_claim_allowed",
        "generator_or_search_claim_allowed",
        "biological_superiority_claim_allowed",
        "accepted_gate1",
        "accepted_esm",
        "stage",
        "protocol",
        "sequence_filter",
        "representations",
        "census",
        "budget_gate",
        "objectives",
        "aggregation",
        "conditional_successor_claims",
        "conditional_successor_claim_prerequisites",
        "forbidden_claims",
        "payload_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class AcceptedEsmContract:
    """Exact accepted identities for the known-sequence ESM feature matrix."""

    producer_job_id: int
    audit_job_id: int
    git_commit: str
    publication_top_sha256: str
    semantic_top_sha256: str
    index_sha256: str
    manifest_sha256: str
    matrix_sha256: str
    tensor_data_sha256: str
    sequence_ids_sha256: str
    independent_receipt_sha256: str
    records: int
    dimensions: int

    def __post_init__(self) -> None:
        if (
            type(self.producer_job_id) is not int
            or self.producer_job_id != 223330
            or type(self.audit_job_id) is not int
            or self.audit_job_id != 223334
        ):
            raise ValueError("ESM contract must name accepted producer 223330/audit 223334")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise ValueError("ESM contract git commit must be a full lowercase Git SHA")
        digests = (
            self.publication_top_sha256,
            self.semantic_top_sha256,
            self.index_sha256,
            self.manifest_sha256,
            self.matrix_sha256,
            self.tensor_data_sha256,
            self.sequence_ids_sha256,
            self.independent_receipt_sha256,
        )
        if any(_SHA256_RE.fullmatch(value) is None for value in digests):
            raise ValueError("ESM contract contains an invalid SHA-256")
        if (
            type(self.records) is not int
            or self.records != 952
            or type(self.dimensions) is not int
            or self.dimensions != 320
        ):
            raise ValueError("ESM contract must describe the accepted 952 x 320 matrix")

    def document(self) -> dict[str, object]:
        return {
            "producer_job_id": self.producer_job_id,
            "audit_job_id": self.audit_job_id,
            "git_commit": self.git_commit,
            "publication_top_sha256": self.publication_top_sha256,
            "semantic_top_sha256": self.semantic_top_sha256,
            "index_sha256": self.index_sha256,
            "manifest_sha256": self.manifest_sha256,
            "matrix_sha256": self.matrix_sha256,
            "tensor_data_sha256": self.tensor_data_sha256,
            "sequence_ids_sha256": self.sequence_ids_sha256,
            "independent_receipt_sha256": self.independent_receipt_sha256,
            "records": self.records,
            "dimensions": self.dimensions,
        }


ACCEPTED_ESM_CONTRACT = AcceptedEsmContract(
    producer_job_id=223330,
    audit_job_id=223334,
    git_commit="753652df2cf9ca9f7e98dc53f3a1884d1f0a75f6",
    publication_top_sha256=("3f6ed4bb54554fa5793e545c67553b48860e954d29ffeb4998460a54883aa603"),
    semantic_top_sha256=("938b7cb80d47f6f68b4d357019d3b7a1cdf9d7e3bf12c98a40af3a950585fa1e"),
    index_sha256="3d72d71f729c31984c2d69aad1c3f70104a3abdf0f0ee47afbdb99326a282ebe",
    manifest_sha256=("1b3ed56beebbec67ebb99110bc5b710a205dd86fab88b3f66552684fb942cc6f"),
    matrix_sha256="1a5846487921c2e7498693918c12da57f354de7db1b79912c29e723984d05f8c",
    tensor_data_sha256=("91ab5a36dbc02222ddf62dbecced3263c357e1ecd65a6420162946c575558c58"),
    sequence_ids_sha256=("45a704812a51d51876b8e32e86c8890db5b7d6aa52de1b1f35634e797acd3f03"),
    independent_receipt_sha256=("a5465d69a81e56e13d0b850b71de1df91e6e780260045d09c69b01680b5e94dc"),
    records=952,
    dimensions=320,
)


@dataclass(frozen=True, slots=True)
class FixedPoolAdapterConfig:
    """Frozen, non-authorizing adapter configuration."""

    path: Path
    sha256: str
    esm: AcceptedEsmContract
    conditional_successor_claims: tuple[str, ...]
    conditional_successor_claim_prerequisites: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    execution_authorized: bool = False
    oracle_reveal_authorized: bool = False
    automatic_production_eligible: bool = False
    independent_audit_accepted: bool = False
    scientific_evidence_accepted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("fixed-pool adapter config path must be a Path")
        if type(self.sha256) is not str or self.sha256 != FROZEN_FIXED_POOL_ADAPTER_CONFIG_SHA256:
            raise ValueError("adapter config must carry the exact frozen SHA-256")
        if type(self.esm) is not AcceptedEsmContract or self.esm != ACCEPTED_ESM_CONTRACT:
            raise ValueError("adapter config must carry the exact accepted ESM contract")
        if self.conditional_successor_claims != EXPECTED_CONDITIONAL_SUCCESSOR_CLAIMS:
            raise ValueError(
                "adapter config conditional successor claims differ from the frozen contract"
            )
        if (
            self.conditional_successor_claim_prerequisites
            != EXPECTED_CONDITIONAL_SUCCESSOR_CLAIM_PREREQUISITES
        ):
            raise ValueError(
                "adapter config conditional successor claim prerequisites differ from the "
                "frozen contract"
            )
        if self.forbidden_claims != EXPECTED_FORBIDDEN_CLAIMS:
            raise ValueError("adapter config forbidden claims differ from the frozen contract")
        authority_flags = (
            self.execution_authorized,
            self.oracle_reveal_authorized,
            self.automatic_production_eligible,
            self.independent_audit_accepted,
            self.scientific_evidence_accepted,
        )
        if any(type(value) is not bool for value in authority_flags) or any(authority_flags):
            raise ValueError(
                "fixed-pool adapter v1 cannot authorize execution, audit acceptance, "
                "scientific evidence, or production"
            )

    def require_oracle_reveal_authority(self) -> None:
        """Reject before any outcome-vault materialization under blocked v1."""

        raise RuntimeError(
            "fixed-pool adapter v1 is preflight-only; no pool or outer outcome vault "
            "may be opened until a new independently reviewed protocol version freezes "
            "the filtered census, budget, commitment barrier, and oracle ledger"
        )


def _require_frozen_config(value: object) -> FixedPoolAdapterConfig:
    if type(value) is not FixedPoolAdapterConfig:
        raise TypeError("fixed-pool adapter requires an exact frozen config capability")
    value.__post_init__()
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingIndexRow:
    row_index: int
    sequence_id: str
    sequence: str

    def __post_init__(self) -> None:
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int):
            raise TypeError("embedding row_index must be an integer")
        if self.row_index < 0:
            raise ValueError("embedding row_index cannot be negative")
        if _SHA256_RE.fullmatch(self.sequence_id) is None:
            raise ValueError("embedding sequence_id must be a lowercase SHA-256")
        if canonical_sequence_id(self.sequence) != self.sequence_id:
            raise ValueError("embedding sequence ID does not match its canonical sequence")


@dataclass(frozen=True, slots=True)
class AuthenticatedEmbeddingIndex:
    """Path-free exact index from two identical accepted ESM twins."""

    rows: tuple[EmbeddingIndexRow, ...]
    publication_top_sha256: str
    semantic_top_sha256: str
    index_sha256: str
    matrix_sha256: str
    independent_receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("authenticated ESM index cannot be empty")
        if tuple(row.row_index for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("authenticated ESM index row ordinals are not contiguous")
        sequence_ids = tuple(row.sequence_id for row in self.rows)
        if sequence_ids != tuple(sorted(set(sequence_ids))):
            raise ValueError("authenticated ESM index IDs are not sorted and unique")
        digests = (
            self.publication_top_sha256,
            self.semantic_top_sha256,
            self.index_sha256,
            self.matrix_sha256,
            self.independent_receipt_sha256,
        )
        if any(_SHA256_RE.fullmatch(value) is None for value in digests):
            raise ValueError("authenticated ESM evidence contains an invalid digest")


@dataclass(frozen=True, slots=True)
class AuthenticatedFixedPoolInputs:
    """Rootless source capabilities after all filesystem authentication."""

    stage_global_seal_sha256: str
    stage_source_anchors_sha256: str
    prepare_capabilities: tuple[PrepareCapability, ...]
    prepare_leaf_seals: tuple[tuple[str, str], ...]
    embeddings: AuthenticatedEmbeddingIndex

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.stage_global_seal_sha256) is None:
            raise ValueError("stage-global seal must be a lowercase SHA-256")
        if _SHA256_RE.fullmatch(self.stage_source_anchors_sha256) is None:
            raise ValueError("stage source-anchor digest must be a lowercase SHA-256")
        rotations = ordered_rotations()
        if len(self.prepare_capabilities) != len(rotations):
            raise ValueError("fixed-pool inputs require exactly twenty prepare capabilities")
        if any(
            type(capability) is not PrepareCapability or capability.spec != rotation
            for capability, rotation in zip(self.prepare_capabilities, rotations, strict=True)
        ):
            raise ValueError("prepare capabilities differ from canonical rotation order")
        expected_ids = tuple(rotation.rotation_id for rotation in rotations)
        if tuple(rotation_id for rotation_id, _seal in self.prepare_leaf_seals) != expected_ids:
            raise ValueError("prepare leaf seals differ from canonical rotation order")
        if any(_SHA256_RE.fullmatch(seal) is None for _, seal in self.prepare_leaf_seals):
            raise ValueError("prepare leaf seal inventory contains an invalid SHA-256")


@dataclass(frozen=True, slots=True)
class FixedPoolPreflightBundle:
    """Deterministic path-free payloads and predecessor identities."""

    payloads: tuple[tuple[str, bytes], ...]
    predecessor_seals: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if tuple(path for path, _payload in self.payloads) != PREFLIGHT_PAYLOAD_PATHS:
            raise ValueError("preflight payload inventory or order is not exact")
        if any(type(payload) is not bytes for _path, payload in self.payloads):
            raise TypeError("preflight payloads must be immutable bytes")
        if self.predecessor_seals != tuple(sorted(self.predecessor_seals)):
            raise ValueError("preflight predecessor seals are not canonical")
        if len(dict(self.predecessor_seals)) != len(self.predecessor_seals):
            raise ValueError("preflight predecessor seal names are duplicated")
        if any(_SHA256_RE.fullmatch(digest) is None for _name, digest in self.predecessor_seals):
            raise ValueError("preflight predecessor inventory contains an invalid SHA-256")

    @property
    def payload_map(self) -> Mapping[str, bytes]:
        return dict(self.payloads)


@dataclass(frozen=True, slots=True)
class FixedPoolPreflightCapability:
    """Structurally verified immutable bytes, not source authority by itself.

    Consumers must pass the retained seal back through
    :func:`verify_fixed_pool_preflight_capability` with the controller-authoritative
    stage digest and prepare-leaf inventory.  Decoded dictionaries are deliberately
    not retained on this object.
    """

    seal: PhaseSeal

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("fixed-pool preflight capability must contain an exact PhaseSeal")
        verify_phase_capability(
            self.seal,
            expected_artifact=PREFLIGHT_ARTIFACT,
            expected_payload_paths=PREFLIGHT_PAYLOAD_PATHS,
            expected_seal_sha256=self.seal.seal_sha256,
        )


def _exact_table(value: object, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a TOML table")
    result = cast(dict[str, Any], value)
    if set(result) != keys:
        raise ValueError(
            f"{label} keys differ from the frozen schema: "
            f"missing={sorted(keys - set(result))}, extra={sorted(set(result) - keys)}"
        )
    return result


def _is_recursively_type_exact(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        observed_mapping = cast(dict[object, object], value)
        expected_mapping = cast(dict[object, object], expected)
        return set(observed_mapping) == set(expected_mapping) and all(
            _is_recursively_type_exact(observed_mapping[key], expected_item)
            for key, expected_item in expected_mapping.items()
        )
    if type(expected) is list:
        observed_list = cast(list[object], value)
        expected_list = cast(list[object], expected)
        return len(observed_list) == len(expected_list) and all(
            _is_recursively_type_exact(observed_item, expected_item)
            for observed_item, expected_item in zip(observed_list, expected_list, strict=True)
        )
    if type(expected) is tuple:
        observed_tuple = cast(tuple[object, ...], value)
        expected_tuple = cast(tuple[object, ...], expected)
        return len(observed_tuple) == len(expected_tuple) and all(
            _is_recursively_type_exact(observed_item, expected_item)
            for observed_item, expected_item in zip(observed_tuple, expected_tuple, strict=True)
        )
    return value == expected


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if not _is_recursively_type_exact(value, expected):
        raise ValueError(f"{label} differs from the frozen fixed-pool v1 contract")


def _require_string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or not value or any(type(item) is not str for item in value):
        raise ValueError(f"{label} must be a nonempty string array")
    return tuple(cast(list[str], value))


def _gate1_config_document() -> dict[str, object]:
    contract = ACCEPTED_GATE1_SOURCE_CONTRACT
    hashes = contract.artifact_hashes
    return {
        "producer_job_id": contract.producer_job_id,
        "audit_job_id": contract.audit_job_id,
        "git_commit": contract.git_commit,
        "publication_top_sha256": contract.publication_top_sha256,
        "semantic_top_sha256": contract.semantic_top_sha256,
        "examples_sha256": hashes["examples.jsonl"],
        "folds_sha256": hashes["folds.json"],
        "independent_receipt_sha256": contract.independent_receipt_sha256,
        "expected_contexts": contract.expected_contexts,
        "expected_sequences": contract.expected_sequences,
        "expected_support_sequences": contract.expected_support_sequences,
        "expected_support_by_fold": list(contract.expected_support_by_fold),
        "expected_support_ids_sha256": contract.expected_support_ids_sha256,
        "expected_support_esm_id_row_pairs_sha256": (EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256),
    }


def load_fixed_pool_adapter_config(path: str | Path) -> FixedPoolAdapterConfig:
    """Load exact frozen bytes; this function cannot activate the v1 bridge."""

    config_path = Path(path)
    payload = config_path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != FROZEN_FIXED_POOL_ADAPTER_CONFIG_SHA256:
        raise ValueError(
            f"fixed-pool adapter v1 bytes differ from the frozen SHA-256: {observed_sha256}"
        )
    try:
        raw_value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("fixed-pool adapter config is not valid UTF-8 TOML") from error
    raw = _exact_table(raw_value, keys=_TOP_LEVEL_KEYS, label="fixed-pool adapter root")
    expected_scalars: Mapping[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ADAPTER_ARTIFACT,
        "status": BLOCKED_STATUS,
        "evidence_class": EVIDENCE_CLASS,
        "decision_date": "2026-09-08",
        "execution_authorized": False,
        "oracle_reveal_authorized": False,
        "automatic_production_eligible": False,
        "independent_audit_accepted": False,
        "scientific_evidence_accepted": False,
        "activity_only": True,
        "de_novo_claim_allowed": False,
        "generator_or_search_claim_allowed": False,
        "biological_superiority_claim_allowed": False,
        "rotation_count": EXPECTED_ROTATION_COUNT,
        "outer_inference_units": EXPECTED_OUTER_INFERENCE_UNITS,
        "outer_folds": list(FOLDS),
        "objectives": list(OBJECTIVES),
        "aggregation": ("equal_context_weight_binary_activity_mean_source_observations_audit_only"),
        "tentative_unique_sequence_budget_to_check": TENTATIVE_UNIQUE_SEQUENCE_BUDGET,
        "tentative_unique_sequence_budget_status": (
            "not_assumed_and_not_authorized_until_authenticated_filtered_census_passes"
        ),
    }
    for field, expected in expected_scalars.items():
        _require_exact(raw[field], expected, label=field)

    sequence_filter = _exact_table(
        raw["sequence_filter"], keys=_SEQUENCE_FILTER_KEYS, label="sequence_filter"
    )
    expected_filter: Mapping[str, object] = {
        "alphabet": ALPHABET,
        "min_length": MIN_LENGTH,
        "max_length": MAX_LENGTH,
        "canonical_sequence_function": "amp_challenge.sequences.canonicalize_sequence",
        "free_termini_and_modification_compliance_established": False,
        "chemical_metadata_status": "not_available_in_gate1_union",
    }
    for field, expected in expected_filter.items():
        _require_exact(sequence_filter[field], expected, label=f"sequence_filter.{field}")

    representations = _exact_table(
        raw["representations"], keys=_REPRESENTATION_KEYS, label="representations"
    )
    expected_representations: Mapping[str, object] = {
        "descriptor_enabled": False,
        "descriptor_status": (
            "blocked_missing_frozen_descriptor_schema_and_authenticated_producer"
        ),
        "accepted_esm_feature_count": 320,
        "accepted_esm_use": "known_sequence_representation_input_only",
        "esm_pretraining_membership_independence_established": False,
        "spectral_contact_enabled": False,
        "spectral_contact_status": "blocked_missing_accepted_contact_matrix_producer",
    }
    for field, expected in expected_representations.items():
        _require_exact(representations[field], expected, label=f"representations.{field}")

    gate1 = _exact_table(raw["gate1"], keys=_GATE1_KEYS, label="gate1")
    _require_exact(dict(gate1), _gate1_config_document(), label="gate1 accepted evidence")
    esm = _exact_table(raw["esm"], keys=_ESM_KEYS, label="esm")
    _require_exact(dict(esm), ACCEPTED_ESM_CONTRACT.document(), label="ESM accepted evidence")

    provenance = _exact_table(raw["provenance"], keys=_PROVENANCE_KEYS, label="provenance")
    expected_provenance: Mapping[str, object] = {
        "gate1_twins_required": True,
        "esm_twins_required": True,
        "independent_receipts_required": True,
        "stage_global_seal_required_at_runtime": True,
        "controller_stage_and_prepare_authority_required_at_publish_and_verify": True,
        "prepare_role": PREPARE_ROLE,
        "pool_outcome_role": "pool-outcome-vault",
        "preflight_publication_is_path_free": True,
        "preflight_publication_requires_no_replace_phase_seal": True,
    }
    for field, expected in expected_provenance.items():
        _require_exact(provenance[field], expected, label=f"provenance.{field}")

    claims = _exact_table(raw["claims"], keys=_CLAIMS_KEYS, label="claims")
    conditional_successor = _require_string_tuple(
        claims["conditional_successor"], label="claims.conditional_successor"
    )
    conditional_prerequisites = _require_string_tuple(
        claims["conditional_successor_prerequisites"],
        label="claims.conditional_successor_prerequisites",
    )
    forbidden = _require_string_tuple(claims["forbidden"], label="claims.forbidden")
    _require_exact(
        conditional_successor,
        EXPECTED_CONDITIONAL_SUCCESSOR_CLAIMS,
        label="claims.conditional_successor",
    )
    _require_exact(
        conditional_prerequisites,
        EXPECTED_CONDITIONAL_SUCCESSOR_CLAIM_PREREQUISITES,
        label="claims.conditional_successor_prerequisites",
    )
    _require_exact(forbidden, EXPECTED_FORBIDDEN_CLAIMS, label="claims.forbidden")

    return FixedPoolAdapterConfig(
        path=config_path,
        sha256=observed_sha256,
        esm=ACCEPTED_ESM_CONTRACT,
        conditional_successor_claims=conditional_successor,
        conditional_successor_claim_prerequisites=conditional_prerequisites,
        forbidden_claims=forbidden,
        independent_audit_accepted=False,
        scientific_evidence_accepted=False,
    )


def _id_stream_sha256(values: Sequence[str]) -> str:
    if any(type(value) is not str or _SHA256_RE.fullmatch(value) is None for value in values):
        raise ValueError("ID stream must contain lowercase SHA-256 strings")
    return sha256_bytes("".join(f"{value}\n" for value in values).encode("ascii"))


def _id_row_pair_stream_sha256(values: Sequence[tuple[str, int]]) -> str:
    if not values:
        raise ValueError("ESM ID/row pair stream cannot be empty")
    ids = tuple(sequence_id for sequence_id, _row_index in values)
    row_indices = tuple(row_index for _sequence_id, row_index in values)
    _id_stream_sha256(ids)
    if (
        ids != tuple(sorted(set(ids)))
        or any(type(row_index) is not int or row_index < 0 for row_index in row_indices)
        or len(set(row_indices)) != len(row_indices)
        or any(left >= right for left, right in pairwise(row_indices))
    ):
        raise ValueError(
            "ESM ID/row pairs must have unique ascending IDs and strictly ascending rows"
        )
    return sha256_bytes(
        "".join(f"{sequence_id}\t{row_index}\n" for sequence_id, row_index in values).encode(
            "ascii"
        )
    )


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must have exactly one final LF")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} duplicates JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not a canonical compact JSON object")
    return cast(dict[str, Any], value)


def _strict_jsonl(
    payload: bytes,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if not payload:
        if allow_empty:
            return ()
        raise ValueError(f"{label} cannot be empty")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be canonically LF framed")
    return tuple(
        _strict_json_object(line, label=f"{label} row {index}")
        for index, line in enumerate(payload.splitlines(keepends=True), start=1)
    )


def _exact_document(
    value: Mapping[str, Any], *, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} does not have the exact fixed-pool schema")
    return value


def _source_predecessors() -> tuple[tuple[str, str], ...]:
    contract = ACCEPTED_GATE1_SOURCE_CONTRACT
    return tuple(
        sorted(
            {
                "source/gate1-independent-receipt": contract.independent_receipt_sha256,
                "source/gate1-publication-top": contract.publication_top_sha256,
                "source/gate1-semantic-top": contract.semantic_top_sha256,
            }.items()
        )
    )


def _validate_stage_gate1_provenance(stage: StageManifestCapability) -> None:
    payload = stage.seal.read_payload_bytes("source-anchors.json")
    if sha256_bytes(payload) != stage.source_anchors_sha256:
        raise ValueError("stage source-anchor payload differs from its authenticated digest")
    anchors = _strict_json_object(payload, label="stage Gate-1 source anchors")
    evidence = anchors.get("gate1_authentication_evidence")
    files = evidence.get("files") if type(evidence) is dict else None
    contract = ACCEPTED_GATE1_SOURCE_CONTRACT
    if (
        anchors.get("publication_top_sha256") != contract.publication_top_sha256
        or anchors.get("semantic_top_sha256") != contract.semantic_top_sha256
        or anchors.get("independent_receipt_sha256") != contract.independent_receipt_sha256
        or type(evidence) is not dict
        or evidence.get("producer_job_id") != contract.producer_job_id
        or evidence.get("audit_job_id") != contract.audit_job_id
        or evidence.get("twins_byte_identical") is not True
        or evidence.get("independent_receipt_sha256") != contract.independent_receipt_sha256
        or type(files) is not dict
    ):
        raise ValueError("stage source anchors do not identify the accepted Gate-1 publication")
    expected_files = {
        "SHA256SUMS": contract.publication_top_sha256,
        "gate1/SHA256SUMS": contract.semantic_top_sha256,
        **{f"gate1/{name}": digest for name, digest in contract.artifact_sha256},
    }
    if any(
        type(files.get(logical)) is not dict or files[logical].get("sha256") != digest
        for logical, digest in expected_files.items()
    ):
        raise ValueError("stage source anchors do not bind every accepted Gate-1 artifact")


def _preflight_predecessors(
    *, stage_global_seal_sha256: str, config: FixedPoolAdapterConfig
) -> tuple[tuple[str, str], ...]:
    if _SHA256_RE.fullmatch(stage_global_seal_sha256) is None:
        raise ValueError("stage-global predecessor must be a lowercase SHA-256")
    gate1 = ACCEPTED_GATE1_SOURCE_CONTRACT
    gate1_hashes = gate1.artifact_hashes
    esm = config.esm
    return tuple(
        sorted(
            {
                "source/esm-index": esm.index_sha256,
                "source/esm-independent-receipt": esm.independent_receipt_sha256,
                "source/esm-manifest": esm.manifest_sha256,
                "source/esm-matrix": esm.matrix_sha256,
                "source/esm-publication-top": esm.publication_top_sha256,
                "source/esm-semantic-top": esm.semantic_top_sha256,
                "source/esm-sequence-ids": esm.sequence_ids_sha256,
                "source/esm-tensor-data": esm.tensor_data_sha256,
                "source/gate1-examples": gate1_hashes["examples.jsonl"],
                "source/gate1-folds": gate1_hashes["folds.json"],
                "source/gate1-independent-receipt": gate1.independent_receipt_sha256,
                "source/gate1-publication-top": gate1.publication_top_sha256,
                "source/gate1-semantic-top": gate1.semantic_top_sha256,
                "stage/global": stage_global_seal_sha256,
            }.items()
        )
    )


def _controller_stage_authority(
    *,
    expected_stage_global_seal_sha256: str,
    expected_prepare_leaf_seals: dict[str, str],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Snapshot the controller-authoritative stage identity into exact immutable values."""

    if (
        type(expected_stage_global_seal_sha256) is not str
        or _SHA256_RE.fullmatch(expected_stage_global_seal_sha256) is None
    ):
        raise ValueError("controller-authoritative stage-global seal must be a lowercase SHA-256")
    if type(expected_prepare_leaf_seals) is not dict:
        raise TypeError("controller prepare-leaf authority must be an exact dictionary")
    expected_ids = tuple(rotation.rotation_id for rotation in ordered_rotations())
    if set(expected_prepare_leaf_seals) != set(expected_ids) or len(
        expected_prepare_leaf_seals
    ) != len(expected_ids):
        raise ValueError(
            "controller prepare-leaf authority must contain every canonical rotation exactly once"
        )
    frozen = tuple(
        (rotation_id, expected_prepare_leaf_seals[rotation_id]) for rotation_id in expected_ids
    )
    if any(
        type(rotation_id) is not str or type(seal) is not str or _SHA256_RE.fullmatch(seal) is None
        for rotation_id, seal in frozen
    ):
        raise ValueError("controller prepare-leaf authority contains an invalid seal")
    return expected_stage_global_seal_sha256, frozen


def _existing_embedding_config(config: FixedPoolAdapterConfig) -> Any:
    """Build the narrow shape required by the accepted downstream verifier."""

    esm = config.esm
    evidence = _ExistingEmbeddingEvidenceConfig(
        producer_job_id=esm.producer_job_id,
        audit_job_id=esm.audit_job_id,
        git_commit=esm.git_commit,
        publication_top_sha256=esm.publication_top_sha256,
        semantic_top_sha256=esm.semantic_top_sha256,
        index_sha256=esm.index_sha256,
        manifest_sha256=esm.manifest_sha256,
        matrix_sha256=esm.matrix_sha256,
        tensor_data_sha256=esm.tensor_data_sha256,
        sequence_ids_sha256=esm.sequence_ids_sha256,
        independent_receipt_sha256=esm.independent_receipt_sha256,
        records=esm.records,
        dimensions=esm.dimensions,
    )
    return SimpleNamespace(embeddings=evidence)


def authenticate_accepted_esm_index(
    twin_roots: Sequence[str | Path],
    independent_receipt_path: str | Path,
    *,
    config: FixedPoolAdapterConfig,
) -> AuthenticatedEmbeddingIndex:
    """Authenticate both accepted ESM twins and retain only their label-free index."""

    config = _require_frozen_config(config)

    if isinstance(twin_roots, str | bytes) or len(twin_roots) != 2:
        raise ValueError("ESM authentication requires exactly two ordered twin roots")
    requested = tuple(Path(path) for path in twin_roots)
    if tuple(path.name for path in requested) != ("0", "1"):
        raise ValueError("ESM twin roots must be ordered as producer slots 0 then 1")
    if requested[0].parent != requested[1].parent:
        raise ValueError("ESM twin roots must share the accepted producer-job parent")
    verifier_config = _existing_embedding_config(config)
    zero_index, _zero_matrix, zero_receipt, zero_snapshots = _verify_existing_embedding_inputs(
        requested[0], independent_receipt_path, config=verifier_config
    )
    one_index, _one_matrix, one_receipt, one_snapshots = _verify_existing_embedding_inputs(
        requested[1], independent_receipt_path, config=verifier_config
    )
    if len(zero_snapshots) != len(one_snapshots) or any(
        left.payload != right.payload
        for left, right in zip(zero_snapshots, one_snapshots, strict=True)
    ):
        raise ValueError("accepted ESM producer twins are not byte-identical")
    zero_rows = _read_existing_embedding_index(zero_index.payload, config=verifier_config)
    one_rows = _read_existing_embedding_index(one_index.payload, config=verifier_config)
    if tuple((row.row_index, row.sequence_id, row.sequence) for row in zero_rows) != tuple(
        (row.row_index, row.sequence_id, row.sequence) for row in one_rows
    ):
        raise ValueError("accepted ESM producer twins have different index semantics")
    for index, snapshots in enumerate((zero_snapshots, one_snapshots)):
        for snapshot in snapshots:
            _assert_embedding_snapshot_unchanged(
                snapshot, label=f"accepted ESM twin {index} authenticated input"
            )
    if zero_receipt.payload != one_receipt.payload:
        raise ValueError("accepted ESM receipt changed between twin authentication passes")
    return AuthenticatedEmbeddingIndex(
        rows=tuple(
            EmbeddingIndexRow(
                row_index=row.row_index,
                sequence_id=row.sequence_id,
                sequence=row.sequence,
            )
            for row in zero_rows
        ),
        publication_top_sha256=config.esm.publication_top_sha256,
        semantic_top_sha256=config.esm.semantic_top_sha256,
        index_sha256=zero_index.sha256,
        matrix_sha256=config.esm.matrix_sha256,
        independent_receipt_sha256=zero_receipt.sha256,
    )


def authenticate_fixed_pool_inputs(
    *,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    prepare_leaf_seals: Mapping[str, PhaseSeal],
    embedding_twin_roots: Sequence[str | Path],
    embedding_independent_receipt: str | Path,
    config: FixedPoolAdapterConfig,
) -> AuthenticatedFixedPoolInputs:
    """Authenticate the exact label-free role capabilities needed by preflight."""

    config = _require_frozen_config(config)

    if type(stage_global_seal) is not PhaseSeal:
        raise TypeError("stage-global input must be an exact PhaseSeal")
    if _SHA256_RE.fullmatch(expected_stage_global_seal_sha256) is None:
        raise ValueError("expected stage-global seal must be a lowercase SHA-256")
    stage = verify_stage_manifest_capability(
        stage_global_seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    if type(stage) is not StageManifestCapability:
        raise TypeError("stage authentication returned an unexpected capability type")
    if stage.source_predecessors != _source_predecessors():
        raise ValueError("trusted stage does not bind the exact accepted Gate-1 evidence")
    _validate_stage_gate1_provenance(stage)
    rotations = ordered_rotations()
    expected_rotation_ids = tuple(rotation.rotation_id for rotation in rotations)
    if set(prepare_leaf_seals) != set(expected_rotation_ids) or len(prepare_leaf_seals) != len(
        expected_rotation_ids
    ):
        raise ValueError("prepare leaf inventory must contain each canonical rotation exactly once")
    capabilities: list[PrepareCapability] = []
    leaf_seals: list[tuple[str, str]] = []
    for rotation in rotations:
        supplied = prepare_leaf_seals[rotation.rotation_id]
        if type(supplied) is not PhaseSeal:
            raise TypeError("every prepare leaf input must be an exact PhaseSeal")
        entry = stage.leaf(spec=rotation, role=PREPARE_ROLE)
        capsule = AuthenticatedLeafCapsule(
            entry=entry,
            seal=supplied,
            source_anchors_sha256=stage.source_anchors_sha256,
            source_predecessors=tuple(sorted(stage.source_predecessors)),
        )
        capability = prepare_capability_from_capsule(capsule)
        if type(capability) is not PrepareCapability or capability.spec != rotation:
            raise ValueError("decoded prepare leaf differs from its canonical rotation")
        capabilities.append(capability)
        leaf_seals.append((rotation.rotation_id, entry.leaf_seal_sha256))
    embeddings = authenticate_accepted_esm_index(
        embedding_twin_roots,
        embedding_independent_receipt,
        config=config,
    )
    return AuthenticatedFixedPoolInputs(
        stage_global_seal_sha256=stage.global_seal_sha256,
        stage_source_anchors_sha256=stage.source_anchors_sha256,
        prepare_capabilities=tuple(capabilities),
        prepare_leaf_seals=tuple(leaf_seals),
        embeddings=embeddings,
    )


def _sequence_filter_reason(sequence: str) -> str | None:
    if type(sequence) is not str or not sequence:
        return "empty_or_non_string_sequence"
    if sequence != sequence.upper() or any(character.isspace() for character in sequence):
        return "noncanonical_sequence_text"
    if len(sequence) < MIN_LENGTH:
        return "length_below_8"
    if len(sequence) > MAX_LENGTH:
        return "length_above_50"
    if not set(sequence).issubset(set(ALPHABET)):
        return "nonstandard_amino_acid"
    return None


def _candidate_sequence_map(capability: PrepareCapability) -> Mapping[str, str]:
    sequence_by_id: dict[str, str] = {}
    for row in capability.acquisition_metadata:
        if row.fold != capability.spec.pool_fold:
            raise ValueError("prepare capability contains metadata outside its pool fold")
        previous = sequence_by_id.setdefault(row.sequence_id, row.sequence)
        if previous != row.sequence:
            raise ValueError("one pool sequence ID maps to multiple sequence strings")
    support = capability.acquisition_support_sequence_ids
    if support != tuple(sorted(set(support))) or not support:
        raise ValueError("pool support sequence IDs must be nonempty, sorted, and unique")
    if not set(support).issubset(sequence_by_id):
        raise ValueError("pool support sequence IDs are absent from label-free metadata")
    return sequence_by_id


def _build_inference_unit_documents() -> tuple[dict[str, object], ...]:
    documents: list[dict[str, object]] = []
    rotations = ordered_rotations()
    for outer_fold in FOLDS:
        rotation_ids = tuple(
            rotation.rotation_id for rotation in rotations if rotation.outer_fold == outer_fold
        )
        if len(rotation_ids) != EXPECTED_ROTATIONS_PER_OUTER_UNIT:
            raise AssertionError("one outer fold does not own exactly four rotations")
        documents.append(
            {
                "schema_version": SCHEMA_VERSION,
                "inference_unit": f"outer-fold-{outer_fold}",
                "outer_fold": outer_fold,
                "rotation_count": len(rotation_ids),
                "rotation_ids": list(rotation_ids),
            }
        )
    if len(documents) != EXPECTED_OUTER_INFERENCE_UNITS:
        raise AssertionError("fixed-pool protocol does not contain exactly five inference units")
    return tuple(documents)


def _derive_fixed_pool_preflight(
    inputs: AuthenticatedFixedPoolInputs,
    *,
    config: FixedPoolAdapterConfig,
) -> FixedPoolPreflightBundle:
    """Derive a deterministic label-free census from authenticated capabilities."""

    if type(inputs) is not AuthenticatedFixedPoolInputs:
        raise TypeError("preflight derivation requires exact authenticated fixed-pool inputs")
    config = _require_frozen_config(config)
    if "".join(STANDARD_AMINO_ACIDS) != ALPHABET:
        raise RuntimeError("repository standard amino-acid alphabet changed")
    embedding_ids = tuple(row.sequence_id for row in inputs.embeddings.rows)
    observed_embedding_evidence = (
        inputs.embeddings.publication_top_sha256,
        inputs.embeddings.semantic_top_sha256,
        inputs.embeddings.index_sha256,
        inputs.embeddings.matrix_sha256,
        inputs.embeddings.independent_receipt_sha256,
    )
    expected_embedding_evidence = (
        config.esm.publication_top_sha256,
        config.esm.semantic_top_sha256,
        config.esm.index_sha256,
        config.esm.matrix_sha256,
        config.esm.independent_receipt_sha256,
    )
    if (
        len(embedding_ids) != config.esm.records
        or _id_stream_sha256(embedding_ids) != config.esm.sequence_ids_sha256
        or observed_embedding_evidence != expected_embedding_evidence
    ):
        raise ValueError("ESM index capability differs from the exact accepted evidence")
    embedding_by_id = {row.sequence_id: row for row in inputs.embeddings.rows}
    if len(embedding_by_id) != len(inputs.embeddings.rows):
        raise ValueError("authenticated ESM index contains duplicate sequence IDs")
    leaf_seal_by_rotation = dict(inputs.prepare_leaf_seals)

    candidate_documents: list[dict[str, object]] = []
    rotation_documents: list[dict[str, object]] = []
    fold_views: dict[int, tuple[tuple[str, str, int], ...]] = {}
    unfiltered_ids_by_fold: dict[int, tuple[str, ...]] = {}

    for rotation, capability in zip(ordered_rotations(), inputs.prepare_capabilities, strict=True):
        if capability.spec != rotation:
            raise ValueError("prepare capabilities are not in canonical rotation order")
        expected_unfiltered_count = ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_by_fold[
            rotation.pool_fold
        ]
        support = capability.acquisition_support_sequence_ids
        if len(support) != expected_unfiltered_count:
            raise ValueError("pool support census differs from accepted Gate-1")
        sequence_by_id = _candidate_sequence_map(capability)
        leaf_seal = leaf_seal_by_rotation[rotation.rotation_id]
        filtered: list[tuple[str, str, int]] = []
        for sequence_id in support:
            sequence = sequence_by_id[sequence_id]
            reason = _sequence_filter_reason(sequence)
            if reason is not None:
                raise ValueError(
                    "authenticated fixed-pool v1 metadata violates its already-enforced "
                    f"sequence contract: {reason}; a new adapter version is required"
                )
            if canonical_sequence_id(sequence) != sequence_id:
                raise ValueError("pool metadata sequence identity changed after filtering")
            embedding = embedding_by_id.get(sequence_id)
            if embedding is None or embedding.sequence != sequence:
                raise ValueError("filtered Gate-1 candidate lacks an exact accepted ESM index join")
            filtered.append((sequence_id, sequence, embedding.row_index))
            candidate_documents.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "rotation_id": rotation.rotation_id,
                    "outer_fold": rotation.outer_fold,
                    "pool_fold": rotation.pool_fold,
                    "base_folds": list(rotation.base_folds),
                    "sequence_id": sequence_id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "esm_row_index": embedding.row_index,
                    "prepare_leaf_seal_sha256": leaf_seal,
                    "sequence_filter_eligible": True,
                    "chemical_compliance_established": False,
                    "evidence_class": EVIDENCE_CLASS,
                }
            )
        filtered_view = tuple(filtered)
        existing_view = fold_views.setdefault(rotation.pool_fold, filtered_view)
        existing_unfiltered = unfiltered_ids_by_fold.setdefault(rotation.pool_fold, support)
        if existing_view != filtered_view or existing_unfiltered != support:
            raise ValueError("rotations with the same pool fold expose different candidate views")
        filtered_ids = tuple(item[0] for item in filtered_view)
        filtered_id_row_pairs = tuple((item[0], item[2]) for item in filtered_view)
        rotation_documents.append(
            {
                "schema_version": SCHEMA_VERSION,
                "rotation_id": rotation.rotation_id,
                "outer_fold": rotation.outer_fold,
                "pool_fold": rotation.pool_fold,
                "base_folds": list(rotation.base_folds),
                "prepare_role": PREPARE_ROLE,
                "prepare_leaf_seal_sha256": leaf_seal,
                "unfiltered_support_count": len(support),
                "filtered_candidate_count": len(filtered_ids),
                "excluded_candidate_count": 0,
                "filtered_candidate_ids_sha256": _id_stream_sha256(filtered_ids),
                "filtered_candidate_esm_id_row_pairs_sha256": (
                    _id_row_pair_stream_sha256(filtered_id_row_pairs)
                ),
            }
        )

    if set(fold_views) != set(FOLDS):
        raise ValueError("preflight does not cover all five pool folds")
    all_unfiltered_ids = tuple(
        sorted(sequence_id for fold in FOLDS for sequence_id in unfiltered_ids_by_fold[fold])
    )
    if (
        len(all_unfiltered_ids) != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_sequences
        or len(set(all_unfiltered_ids)) != len(all_unfiltered_ids)
        or _id_stream_sha256(all_unfiltered_ids)
        != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_ids_sha256
    ):
        raise ValueError("five-fold unfiltered support union differs from accepted Gate-1")
    all_filtered_records = tuple(sorted(record for fold in FOLDS for record in fold_views[fold]))
    all_filtered_id_row_pairs = tuple(
        (sequence_id, row_index) for sequence_id, _sequence, row_index in all_filtered_records
    )
    if (
        _id_row_pair_stream_sha256(all_filtered_id_row_pairs)
        != EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256
    ):
        raise ValueError(
            "filtered support sequence-to-ESM-row mapping differs from the frozen "
            "accepted Gate-1/ESM join"
        )

    fold_census_documents: list[dict[str, object]] = []
    for fold in FOLDS:
        filtered_ids = tuple(item[0] for item in fold_views[fold])
        filtered_id_row_pairs = tuple((item[0], item[2]) for item in fold_views[fold])
        fold_census_documents.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pool_fold": fold,
                "unfiltered_support_count": len(unfiltered_ids_by_fold[fold]),
                "filtered_candidate_count": len(filtered_ids),
                "excluded_candidate_count": 0,
                "filtered_candidate_ids_sha256": _id_stream_sha256(filtered_ids),
                "filtered_candidate_esm_id_row_pairs_sha256": (
                    _id_row_pair_stream_sha256(filtered_id_row_pairs)
                ),
                "tentative_96_unique_sequence_budget_census_sufficient": (
                    len(filtered_ids) >= TENTATIVE_UNIQUE_SEQUENCE_BUDGET
                ),
            }
        )
    filtered_counts = tuple(
        cast(int, document["filtered_candidate_count"]) for document in fold_census_documents
    )
    minimum_filtered = min(filtered_counts)
    census_sufficient = minimum_filtered >= TENTATIVE_UNIQUE_SEQUENCE_BUDGET
    budget_status = (
        "census_sufficient_but_execution_not_authorized"
        if census_sufficient
        else "census_insufficient_budget_must_be_lowered_before_any_outcome_access"
    )
    inference_unit_documents = _build_inference_unit_documents()

    candidates_payload = canonical_jsonl_bytes(candidate_documents)
    excluded_payload = b""
    fold_census_payload = canonical_jsonl_bytes(fold_census_documents)
    inference_units_payload = canonical_jsonl_bytes(inference_unit_documents)
    rotations_payload = canonical_jsonl_bytes(rotation_documents)
    data_payloads = {
        "candidates.jsonl": candidates_payload,
        "excluded-candidates.jsonl": excluded_payload,
        "fold-census.jsonl": fold_census_payload,
        "inference-units.jsonl": inference_units_payload,
        "rotations.jsonl": rotations_payload,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PREFLIGHT_ARTIFACT,
        "status": PREFLIGHT_STATUS,
        "evidence_class": EVIDENCE_CLASS,
        "activity_only": True,
        "config_sha256": config.sha256,
        "execution_authorized": False,
        "oracle_reveal_authorized": False,
        "automatic_production_eligible": False,
        "independent_audit_accepted": False,
        "scientific_evidence_accepted": False,
        "de_novo_claim_allowed": False,
        "generator_or_search_claim_allowed": False,
        "biological_superiority_claim_allowed": False,
        "accepted_gate1": _gate1_config_document(),
        "accepted_esm": config.esm.document(),
        "stage": {
            "global_seal_sha256": inputs.stage_global_seal_sha256,
            "source_anchors_sha256": inputs.stage_source_anchors_sha256,
            "prepare_role": PREPARE_ROLE,
            "prepare_leaf_seals": dict(inputs.prepare_leaf_seals),
        },
        "protocol": {
            "rotation_count": len(rotation_documents),
            "outer_inference_unit_count": len(inference_unit_documents),
            "inference_unit": "outer_fold",
            "rotations_are_repeated_measures_not_independent_units": True,
        },
        "sequence_filter": {
            "alphabet": ALPHABET,
            "min_length": MIN_LENGTH,
            "max_length": MAX_LENGTH,
            "free_termini_and_modification_compliance_established": False,
        },
        "representations": {
            "descriptor_enabled": False,
            "descriptor_status": (
                "blocked_missing_frozen_descriptor_schema_and_authenticated_producer"
            ),
            "accepted_esm_feature_count": config.esm.dimensions,
            "accepted_esm_use": "known_sequence_representation_input_only",
            "esm_pretraining_membership_independence_established": False,
            "spectral_contact_enabled": False,
            "spectral_contact_status": "blocked_missing_accepted_contact_matrix_producer",
        },
        "census": {
            "unfiltered_unique_support_sequences": len(all_unfiltered_ids),
            "filtered_unique_sequences": sum(filtered_counts),
            "filtered_candidate_rotation_associations": len(candidate_documents),
            "excluded_candidate_rotation_associations": 0,
            "filtered_candidates_by_pool_fold": list(filtered_counts),
            "minimum_filtered_pool_count": minimum_filtered,
            "filtered_support_esm_id_row_pairs_sha256": (EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256),
        },
        "budget_gate": {
            "tentative_unique_sequence_budget_to_check": (TENTATIVE_UNIQUE_SEQUENCE_BUDGET),
            "authenticated_filtered_census_sufficient": census_sufficient,
            "status": budget_status,
            "budget_assumed": False,
            "budget_authorized": False,
        },
        "objectives": list(OBJECTIVES),
        "aggregation": ("equal_context_weight_binary_activity_mean_source_observations_audit_only"),
        "conditional_successor_claims": list(config.conditional_successor_claims),
        "conditional_successor_claim_prerequisites": list(
            config.conditional_successor_claim_prerequisites
        ),
        "forbidden_claims": list(config.forbidden_claims),
        "payload_sha256": {
            path: sha256_bytes(payload) for path, payload in sorted(data_payloads.items())
        },
    }
    payloads = {
        **data_payloads,
        "manifest.json": canonical_json_bytes(manifest),
    }
    ordered_payloads = tuple((path, payloads[path]) for path in PREFLIGHT_PAYLOAD_PATHS)
    return FixedPoolPreflightBundle(
        payloads=ordered_payloads,
        predecessor_seals=_preflight_predecessors(
            stage_global_seal_sha256=inputs.stage_global_seal_sha256,
            config=config,
        ),
    )


def build_fixed_pool_preflight(
    *,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    prepare_leaf_seals: Mapping[str, PhaseSeal],
    embedding_twin_roots: Sequence[str | Path],
    embedding_independent_receipt: str | Path,
    config: FixedPoolAdapterConfig,
) -> FixedPoolPreflightBundle:
    """Authenticate all frozen inputs, then derive the label-free preflight."""

    inputs = authenticate_fixed_pool_inputs(
        stage_global_seal=stage_global_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        prepare_leaf_seals=prepare_leaf_seals,
        embedding_twin_roots=embedding_twin_roots,
        embedding_independent_receipt=embedding_independent_receipt,
        config=config,
    )
    return _derive_fixed_pool_preflight(inputs, config=config)


def _validate_preflight_payloads(
    payloads: Mapping[str, bytes],
    predecessor_seals: Mapping[str, str],
    *,
    config: FixedPoolAdapterConfig,
    expected_stage_global_seal_sha256: str,
    expected_prepare_leaf_seals: dict[str, str],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    config = _require_frozen_config(config)
    controller_stage, controller_prepare_seals = _controller_stage_authority(
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_leaf_seals=expected_prepare_leaf_seals,
    )
    if set(payloads) != set(PREFLIGHT_PAYLOAD_PATHS):
        raise ValueError("preflight capability has the wrong payload inventory")
    manifest = _exact_document(
        _strict_json_object(payloads["manifest.json"], label="fixed-pool manifest"),
        fields=_MANIFEST_FIELDS,
        label="fixed-pool manifest",
    )
    expected_top: Mapping[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PREFLIGHT_ARTIFACT,
        "status": PREFLIGHT_STATUS,
        "evidence_class": EVIDENCE_CLASS,
        "activity_only": True,
        "config_sha256": config.sha256,
        "execution_authorized": False,
        "oracle_reveal_authorized": False,
        "automatic_production_eligible": False,
        "independent_audit_accepted": False,
        "scientific_evidence_accepted": False,
        "de_novo_claim_allowed": False,
        "generator_or_search_claim_allowed": False,
        "biological_superiority_claim_allowed": False,
        "objectives": list(OBJECTIVES),
        "aggregation": ("equal_context_weight_binary_activity_mean_source_observations_audit_only"),
        "conditional_successor_claims": list(config.conditional_successor_claims),
        "conditional_successor_claim_prerequisites": list(
            config.conditional_successor_claim_prerequisites
        ),
        "forbidden_claims": list(config.forbidden_claims),
    }
    for field, expected in expected_top.items():
        _require_exact(manifest[field], expected, label=f"manifest.{field}")
    _require_exact(
        manifest["accepted_gate1"],
        _gate1_config_document(),
        label="manifest.accepted_gate1",
    )
    _require_exact(manifest["accepted_esm"], config.esm.document(), label="manifest.accepted_esm")

    stage = _exact_document(
        cast(Mapping[str, Any], manifest["stage"]),
        fields=frozenset(
            {
                "global_seal_sha256",
                "source_anchors_sha256",
                "prepare_role",
                "prepare_leaf_seals",
            }
        ),
        label="manifest.stage",
    )
    stage_global = stage["global_seal_sha256"]
    source_anchors = stage["source_anchors_sha256"]
    if (
        type(stage_global) is not str
        or _SHA256_RE.fullmatch(stage_global) is None
        or stage_global != controller_stage
        or type(source_anchors) is not str
        or _SHA256_RE.fullmatch(source_anchors) is None
        or stage["prepare_role"] != PREPARE_ROLE
    ):
        raise ValueError("manifest stage identity is invalid")
    prepare_seals = stage["prepare_leaf_seals"]
    expected_rotation_ids = tuple(rotation.rotation_id for rotation in ordered_rotations())
    if (
        type(prepare_seals) is not dict
        or tuple(prepare_seals) != expected_rotation_ids
        or tuple(prepare_seals.items()) != controller_prepare_seals
        or any(
            type(value) is not str or _SHA256_RE.fullmatch(value) is None
            for value in prepare_seals.values()
        )
    ):
        raise ValueError("manifest prepare-leaf inventory is not exact")
    expected_predecessors = dict(
        _preflight_predecessors(
            stage_global_seal_sha256=controller_stage,
            config=config,
        )
    )
    if dict(predecessor_seals) != expected_predecessors:
        raise ValueError("preflight predecessor seals differ from accepted inputs")

    payload_hashes = manifest["payload_sha256"]
    data_paths = tuple(path for path in PREFLIGHT_PAYLOAD_PATHS if path != "manifest.json")
    if (
        type(payload_hashes) is not dict
        or set(payload_hashes) != set(data_paths)
        or any(
            type(payload_hashes[path]) is not str
            or payload_hashes[path] != sha256_bytes(payloads[path])
            for path in data_paths
        )
    ):
        raise ValueError("manifest payload hashes do not bind all preflight data")

    protocol = _exact_document(
        cast(Mapping[str, Any], manifest["protocol"]),
        fields=frozenset(
            {
                "rotation_count",
                "outer_inference_unit_count",
                "inference_unit",
                "rotations_are_repeated_measures_not_independent_units",
            }
        ),
        label="manifest.protocol",
    )
    expected_protocol = {
        "rotation_count": EXPECTED_ROTATION_COUNT,
        "outer_inference_unit_count": EXPECTED_OUTER_INFERENCE_UNITS,
        "inference_unit": "outer_fold",
        "rotations_are_repeated_measures_not_independent_units": True,
    }
    _require_exact(dict(protocol), expected_protocol, label="manifest.protocol")
    expected_filter = {
        "alphabet": ALPHABET,
        "min_length": MIN_LENGTH,
        "max_length": MAX_LENGTH,
        "free_termini_and_modification_compliance_established": False,
    }
    _require_exact(manifest["sequence_filter"], expected_filter, label="manifest.sequence_filter")
    expected_representations = {
        "descriptor_enabled": False,
        "descriptor_status": (
            "blocked_missing_frozen_descriptor_schema_and_authenticated_producer"
        ),
        "accepted_esm_feature_count": config.esm.dimensions,
        "accepted_esm_use": "known_sequence_representation_input_only",
        "esm_pretraining_membership_independence_established": False,
        "spectral_contact_enabled": False,
        "spectral_contact_status": "blocked_missing_accepted_contact_matrix_producer",
    }
    _require_exact(
        manifest["representations"],
        expected_representations,
        label="manifest.representations",
    )

    rotation_rows = _strict_jsonl(payloads["rotations.jsonl"], label="rotation census")
    if len(rotation_rows) != EXPECTED_ROTATION_COUNT:
        raise ValueError("rotation census must contain exactly twenty rows")
    rotation_by_id: dict[str, Mapping[str, Any]] = {}
    for row, spec in zip(rotation_rows, ordered_rotations(), strict=True):
        document = _exact_document(row, fields=_ROTATION_FIELDS, label="rotation census row")
        structural = {
            "schema_version": SCHEMA_VERSION,
            "rotation_id": spec.rotation_id,
            "outer_fold": spec.outer_fold,
            "pool_fold": spec.pool_fold,
            "base_folds": list(spec.base_folds),
            "prepare_role": PREPARE_ROLE,
            "prepare_leaf_seal_sha256": prepare_seals[spec.rotation_id],
        }
        for field, expected in structural.items():
            _require_exact(document[field], expected, label=f"rotation.{field}")
        for field in (
            "unfiltered_support_count",
            "filtered_candidate_count",
            "excluded_candidate_count",
        ):
            if isinstance(document[field], bool) or not isinstance(document[field], int):
                raise ValueError(f"rotation.{field} must be an integer")
        if (
            document["unfiltered_support_count"]
            != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_by_fold[spec.pool_fold]
            or cast(int, document["filtered_candidate_count"]) < 0
            or cast(int, document["excluded_candidate_count"]) < 0
            or cast(int, document["filtered_candidate_count"])
            + cast(int, document["excluded_candidate_count"])
            != document["unfiltered_support_count"]
            or type(document["filtered_candidate_ids_sha256"]) is not str
            or _SHA256_RE.fullmatch(document["filtered_candidate_ids_sha256"]) is None
            or type(document["filtered_candidate_esm_id_row_pairs_sha256"]) is not str
            or _SHA256_RE.fullmatch(document["filtered_candidate_esm_id_row_pairs_sha256"]) is None
        ):
            raise ValueError("rotation candidate census is invalid")
        rotation_by_id[spec.rotation_id] = document

    candidates = _strict_jsonl(payloads["candidates.jsonl"], label="filtered candidates")
    candidate_ids_by_rotation: dict[str, list[str]] = defaultdict(list)
    candidate_records_by_rotation: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    previous_candidate_key: tuple[int, str] | None = None
    rotation_ordinal = {
        rotation.rotation_id: index for index, rotation in enumerate(ordered_rotations())
    }
    for row in candidates:
        document = _exact_document(row, fields=_CANDIDATE_FIELDS, label="candidate row")
        rotation_id = document["rotation_id"]
        if type(rotation_id) is not str or rotation_id not in rotation_by_id:
            raise ValueError("candidate row names an unknown rotation")
        rotation = ordered_rotations()[rotation_ordinal[rotation_id]]
        sequence_id = document["sequence_id"]
        sequence = document["sequence"]
        if type(sequence_id) is not str or type(sequence) is not str:
            raise ValueError("candidate sequence identity fields must be strings")
        key = (rotation_ordinal[rotation_id], sequence_id)
        if previous_candidate_key is not None and key <= previous_candidate_key:
            raise ValueError("candidate rows are not in canonical rotation/ID order")
        previous_candidate_key = key
        expected_values: Mapping[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "outer_fold": rotation.outer_fold,
            "pool_fold": rotation.pool_fold,
            "base_folds": list(rotation.base_folds),
            "length": len(sequence),
            "prepare_leaf_seal_sha256": prepare_seals[rotation_id],
            "sequence_filter_eligible": True,
            "chemical_compliance_established": False,
            "evidence_class": EVIDENCE_CLASS,
        }
        for field, expected in expected_values.items():
            _require_exact(document[field], expected, label=f"candidate.{field}")
        if (
            _sequence_filter_reason(sequence) is not None
            or canonical_sequence_id(sequence) != sequence_id
            or isinstance(document["esm_row_index"], bool)
            or not isinstance(document["esm_row_index"], int)
            or not 0 <= document["esm_row_index"] < config.esm.records
        ):
            raise ValueError("candidate row violates sequence or ESM index constraints")
        candidate_ids_by_rotation[rotation_id].append(sequence_id)
        candidate_records_by_rotation[rotation_id].append(
            (sequence_id, sequence, cast(int, document["esm_row_index"]))
        )

    exclusions = _strict_jsonl(
        payloads["excluded-candidates.jsonl"],
        label="excluded candidates",
        allow_empty=True,
    )
    if exclusions:
        raise ValueError(
            "fixed-pool adapter v1 requires an empty exclusion payload; its authenticated "
            "source already enforces the exact sequence contract"
        )
    exclusion_ids_by_rotation: dict[str, list[str]] = defaultdict(list)

    fold_candidate_ids: dict[int, tuple[str, ...]] = {}
    fold_candidate_records: dict[int, tuple[tuple[str, str, int], ...]] = {}
    fold_exclusion_ids: dict[int, tuple[str, ...]] = {}
    fold_unfiltered_ids: dict[int, tuple[str, ...]] = {}
    for spec in ordered_rotations():
        rotation = rotation_by_id[spec.rotation_id]
        candidate_ids = tuple(candidate_ids_by_rotation[spec.rotation_id])
        candidate_records = tuple(candidate_records_by_rotation[spec.rotation_id])
        candidate_id_row_pairs = tuple(
            (sequence_id, row_index) for sequence_id, _sequence, row_index in candidate_records
        )
        exclusion_ids = tuple(exclusion_ids_by_rotation[spec.rotation_id])
        unfiltered_ids = tuple(sorted((*candidate_ids, *exclusion_ids)))
        if (
            candidate_ids != tuple(sorted(set(candidate_ids)))
            or exclusion_ids != tuple(sorted(set(exclusion_ids)))
            or set(candidate_ids) & set(exclusion_ids)
            or len(candidate_ids) != rotation["filtered_candidate_count"]
            or len(exclusion_ids) != rotation["excluded_candidate_count"]
            or _id_stream_sha256(candidate_ids) != rotation["filtered_candidate_ids_sha256"]
            or _id_row_pair_stream_sha256(candidate_id_row_pairs)
            != rotation["filtered_candidate_esm_id_row_pairs_sha256"]
            or len(unfiltered_ids)
            != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_by_fold[spec.pool_fold]
            or len(set(unfiltered_ids)) != len(unfiltered_ids)
        ):
            raise ValueError("candidate payloads differ from their rotation census")
        previous_candidates = fold_candidate_ids.setdefault(spec.pool_fold, candidate_ids)
        previous_records = fold_candidate_records.setdefault(spec.pool_fold, candidate_records)
        previous_exclusions = fold_exclusion_ids.setdefault(spec.pool_fold, exclusion_ids)
        previous_unfiltered = fold_unfiltered_ids.setdefault(spec.pool_fold, unfiltered_ids)
        if (
            previous_candidates != candidate_ids
            or previous_records != candidate_records
            or previous_exclusions != exclusion_ids
            or previous_unfiltered != unfiltered_ids
        ):
            raise ValueError("same-fold rotations have inconsistent filtered candidate views")

    all_unfiltered_ids = tuple(
        sorted(sequence_id for fold in FOLDS for sequence_id in fold_unfiltered_ids[fold])
    )
    if (
        len(all_unfiltered_ids) != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_sequences
        or len(set(all_unfiltered_ids)) != len(all_unfiltered_ids)
        or _id_stream_sha256(all_unfiltered_ids)
        != ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_ids_sha256
    ):
        raise ValueError(
            "candidate and exclusion union differs from the exact accepted Gate-1 support set"
        )
    all_candidate_records = tuple(
        sorted(record for fold in FOLDS for record in fold_candidate_records[fold])
    )
    all_candidate_id_row_pairs = tuple(
        (sequence_id, row_index) for sequence_id, _sequence, row_index in all_candidate_records
    )
    if (
        _id_row_pair_stream_sha256(all_candidate_id_row_pairs)
        != EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256
    ):
        raise ValueError(
            "candidate sequence-to-ESM-row mapping differs from the frozen accepted join"
        )

    fold_rows = _strict_jsonl(payloads["fold-census.jsonl"], label="fold census")
    if len(fold_rows) != len(FOLDS):
        raise ValueError("fold census must contain exactly five rows")
    filtered_counts: list[int] = []
    for row, fold in zip(fold_rows, FOLDS, strict=True):
        document = _exact_document(row, fields=_FOLD_CENSUS_FIELDS, label="fold census row")
        candidates_for_fold = fold_candidate_ids[fold]
        candidate_records_for_fold = fold_candidate_records[fold]
        candidate_id_row_pairs_for_fold = tuple(
            (sequence_id, row_index)
            for sequence_id, _sequence, row_index in candidate_records_for_fold
        )
        exclusions_for_fold = fold_exclusion_ids[fold]
        expected = {
            "schema_version": SCHEMA_VERSION,
            "pool_fold": fold,
            "unfiltered_support_count": (
                ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_by_fold[fold]
            ),
            "filtered_candidate_count": len(candidates_for_fold),
            "excluded_candidate_count": len(exclusions_for_fold),
            "filtered_candidate_ids_sha256": _id_stream_sha256(candidates_for_fold),
            "filtered_candidate_esm_id_row_pairs_sha256": (
                _id_row_pair_stream_sha256(candidate_id_row_pairs_for_fold)
            ),
            "tentative_96_unique_sequence_budget_census_sufficient": (
                len(candidates_for_fold) >= TENTATIVE_UNIQUE_SEQUENCE_BUDGET
            ),
        }
        _require_exact(dict(document), expected, label=f"fold census {fold}")
        filtered_counts.append(len(candidates_for_fold))

    inference_rows = _strict_jsonl(payloads["inference-units.jsonl"], label="inference units")
    expected_inference = _build_inference_unit_documents()
    if len(inference_rows) != len(expected_inference):
        raise ValueError("inference-unit census changed")
    for row, expected in zip(inference_rows, expected_inference, strict=True):
        _exact_document(row, fields=_INFERENCE_UNIT_FIELDS, label="inference-unit row")
        _require_exact(dict(row), expected, label="inference-unit row")

    census = manifest["census"]
    minimum_filtered = min(filtered_counts)
    expected_census = {
        "unfiltered_unique_support_sequences": (
            ACCEPTED_GATE1_SOURCE_CONTRACT.expected_support_sequences
        ),
        "filtered_unique_sequences": sum(filtered_counts),
        "filtered_candidate_rotation_associations": len(candidates),
        "excluded_candidate_rotation_associations": len(exclusions),
        "filtered_candidates_by_pool_fold": filtered_counts,
        "minimum_filtered_pool_count": minimum_filtered,
        "filtered_support_esm_id_row_pairs_sha256": (EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256),
    }
    _require_exact(census, expected_census, label="manifest.census")
    sufficient = minimum_filtered >= TENTATIVE_UNIQUE_SEQUENCE_BUDGET
    expected_budget_gate = {
        "tentative_unique_sequence_budget_to_check": (TENTATIVE_UNIQUE_SEQUENCE_BUDGET),
        "authenticated_filtered_census_sufficient": sufficient,
        "status": (
            "census_sufficient_but_execution_not_authorized"
            if sufficient
            else "census_insufficient_budget_must_be_lowered_before_any_outcome_access"
        ),
        "budget_assumed": False,
        "budget_authorized": False,
    }
    _require_exact(manifest["budget_gate"], expected_budget_gate, label="manifest.budget_gate")
    return manifest, candidates


def publish_fixed_pool_preflight(
    destination: str | Path,
    *,
    bundle: FixedPoolPreflightBundle,
    config: FixedPoolAdapterConfig,
    expected_stage_global_seal_sha256: str,
    expected_prepare_leaf_seals: dict[str, str],
) -> FixedPoolPreflightCapability:
    """Publish under controller stage authority, then verify the no-replace phase."""

    if type(bundle) is not FixedPoolPreflightBundle:
        raise TypeError("preflight publisher requires an exact FixedPoolPreflightBundle")
    config = _require_frozen_config(config)
    controller_stage, controller_prepare_pairs = _controller_stage_authority(
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_leaf_seals=expected_prepare_leaf_seals,
    )
    controller_prepare = dict(controller_prepare_pairs)
    payloads = bundle.payload_map
    predecessors = dict(bundle.predecessor_seals)
    _validate_preflight_payloads(
        payloads,
        predecessors,
        config=config,
        expected_stage_global_seal_sha256=controller_stage,
        expected_prepare_leaf_seals=controller_prepare,
    )
    seal = publish_phase(
        destination,
        artifact=PREFLIGHT_ARTIFACT,
        payloads=payloads,
        predecessor_seals=predecessors,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "evidence_class": EVIDENCE_CLASS,
            "config_sha256": config.sha256,
            "status": PREFLIGHT_STATUS,
            "activity_only": True,
            "execution_authorized": False,
            "oracle_reveal_authorized": False,
            "automatic_production_eligible": False,
            "independent_audit_accepted": False,
            "scientific_evidence_accepted": False,
        },
    )
    return verify_fixed_pool_preflight_capability(
        seal,
        config=config,
        expected_stage_global_seal_sha256=controller_stage,
        expected_prepare_leaf_seals=controller_prepare,
    )


def verify_fixed_pool_preflight_capability(
    seal: PhaseSeal,
    *,
    config: FixedPoolAdapterConfig,
    expected_stage_global_seal_sha256: str,
    expected_prepare_leaf_seals: dict[str, str],
) -> FixedPoolPreflightCapability:
    """Reconstruct all semantics under explicit controller stage authority."""

    config = _require_frozen_config(config)
    controller_stage, controller_prepare_pairs = _controller_stage_authority(
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_leaf_seals=expected_prepare_leaf_seals,
    )
    controller_prepare = dict(controller_prepare_pairs)

    verified = verify_phase_capability(
        seal,
        expected_artifact=PREFLIGHT_ARTIFACT,
        expected_payload_paths=PREFLIGHT_PAYLOAD_PATHS,
    )
    metadata = _strict_json_object(verified.metadata_json, label="preflight phase metadata")
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "config_sha256": config.sha256,
        "status": PREFLIGHT_STATUS,
        "activity_only": True,
        "execution_authorized": False,
        "oracle_reveal_authorized": False,
        "automatic_production_eligible": False,
        "independent_audit_accepted": False,
        "scientific_evidence_accepted": False,
    }
    _require_exact(metadata, expected_metadata, label="preflight phase metadata")
    payloads = {path: verified.read_payload_bytes(path) for path in PREFLIGHT_PAYLOAD_PATHS}
    _validate_preflight_payloads(
        payloads,
        dict(verified.predecessor_seals),
        config=config,
        expected_stage_global_seal_sha256=controller_stage,
        expected_prepare_leaf_seals=controller_prepare,
    )
    verify_phase_capability(
        verified,
        expected_artifact=PREFLIGHT_ARTIFACT,
        expected_payload_paths=PREFLIGHT_PAYLOAD_PATHS,
        expected_predecessor_seals=dict(
            _preflight_predecessors(
                stage_global_seal_sha256=controller_stage,
                config=config,
            )
        ),
        expected_seal_sha256=verified.seal_sha256,
    )
    return FixedPoolPreflightCapability(seal=verified)


def require_fixed_pool_oracle_reveal_authority(config: FixedPoolAdapterConfig) -> None:
    """Public fail-closed oracle bridge; v1 rejects before any vault can open."""

    config = _require_frozen_config(config)
    config.require_oracle_reveal_authority()


__all__ = [
    "ACCEPTED_ESM_CONTRACT",
    "ADAPTER_ARTIFACT",
    "BLOCKED_STATUS",
    "EVIDENCE_CLASS",
    "EXPECTED_CONDITIONAL_SUCCESSOR_CLAIMS",
    "EXPECTED_CONDITIONAL_SUCCESSOR_CLAIM_PREREQUISITES",
    "EXPECTED_FORBIDDEN_CLAIMS",
    "EXPECTED_SUPPORT_ESM_ID_ROW_PAIRS_SHA256",
    "FROZEN_FIXED_POOL_ADAPTER_CONFIG_SHA256",
    "PREFLIGHT_ARTIFACT",
    "PREFLIGHT_PAYLOAD_PATHS",
    "TENTATIVE_UNIQUE_SEQUENCE_BUDGET",
    "AcceptedEsmContract",
    "AuthenticatedEmbeddingIndex",
    "AuthenticatedFixedPoolInputs",
    "EmbeddingIndexRow",
    "FixedPoolAdapterConfig",
    "FixedPoolPreflightBundle",
    "FixedPoolPreflightCapability",
    "authenticate_accepted_esm_index",
    "authenticate_fixed_pool_inputs",
    "build_fixed_pool_preflight",
    "load_fixed_pool_adapter_config",
    "publish_fixed_pool_preflight",
    "require_fixed_pool_oracle_reveal_authority",
    "verify_fixed_pool_preflight_capability",
]

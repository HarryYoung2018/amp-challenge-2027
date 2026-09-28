"""Authenticated, pathless prepare transport for sequential-v2.

The numerical prepare routine returns immutable Python values. This module is
the publication boundary around that routine. It authenticates the rootless
protocol and stage capabilities, publishes four isolated leaves per rotation,
then releases only a payload-free attestation.  The outcome-free campaign
publisher accepts exactly twenty such supervised worker results and never
receives any prepare-leaf capability.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.acquisition.sequential_v2_selector import (
    OBJECTIVES,
    PoolCandidate,
    RandomPoolCandidate,
)
from amp_challenge.evaluation.sequential_v2_prepare import (
    PoolComponent,
    PoolPrediction,
    PreparedRotation,
    id_stream_sha256,
    prepare_rotation,
)
from amp_challenge.evaluation.sequential_v2_primitives import (
    FEATURE_NAMES,
    LOGISTIC_L2,
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_PRIOR_STRENGTH,
    LOGISTIC_TOLERANCE,
    PROBABILITY_CLIP,
    TARGETS,
    ContextRow,
    DescriptorLogisticState,
    DiversityTransform,
    NoveltyEvidence,
    fit_descriptor_logistic,
    fit_diversity_transform,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_CONTEXTS_BY_FOLD,
    EXPECTED_PHYSICAL_POOL_VIEW_ROWS,
    EXPECTED_POOL_CANDIDATES,
    EXPECTED_PREDICTION_POOL_VIEW_ROWS,
    EXPECTED_RANDOM_POOL_VIEW_ROWS,
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_run_by_track_id,
    protocol_census,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_select import (
    RotationViewProvenance,
    ordered_id_stream_sha256,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    GLOBAL_PAYLOAD_PATHS,
    LEAF_ARTIFACTS,
    LEAF_CAPABILITY_ARTIFACT,
    LEAF_DATA_PAYLOAD_PATHS,
    LEAF_PAYLOAD_PATHS,
    PREPARE_ROLE,
    STAGE_ARTIFACT,
    AuthenticatedLeafCapsule,
    IdStreamBinding,
    StageLeafIndex,
    leaf_relative_path,
    prepare_capability_from_capsule,
)
from amp_challenge.evaluation.sequential_v2_staging import LabelFreeContext, PrepareCapability

SCHEMA_VERSION = 1

PROTOCOL_ARTIFACT = "sequential_v2_protocol_v1"
PREPARE_EVIDENCE_ARTIFACT = "sequential_v2_prepare_evidence_v1"
BASE_UPDATE_ARTIFACT = "sequential_v2_prepare_base_update_v1"
BASE_UPDATE_CAPABILITY_ARTIFACT = "sequential_v2_prepare_base_update_capability_v1"
PREDICTION_VIEW_ARTIFACT = "sequential_v2_prepare_prediction_view_v1"
PREDICTION_VIEW_SUMMARY_ARTIFACT = "sequential_v2_prepare_prediction_view_summary_v1"
RANDOM_MINIMAL_VIEW_ARTIFACT = "sequential_v2_prepare_random_minimal_view_v1"
RANDOM_MINIMAL_VIEW_SUMMARY_ARTIFACT = "sequential_v2_prepare_random_minimal_view_summary_v1"
PREPARE_EVIDENCE_SUMMARY_ARTIFACT = "sequential_v2_prepare_evidence_summary_v1"
PREPARE_CAMPAIGN_ARTIFACT = "sequential_v2_prepare_campaign_barrier_v1"
PREPARE_ROTATION_ATTESTATION_ARTIFACT = "sequential_v2_prepare_rotation_attestation_v1"

EVIDENCE_ROLE = "evidence"
BASE_UPDATE_ROLE = "base_update"
PREDICTION_VIEW_ROLE = "prediction_view"
RANDOM_MINIMAL_VIEW_ROLE = "random_minimal_view"
ROLE_ORDER = (
    EVIDENCE_ROLE,
    BASE_UPDATE_ROLE,
    PREDICTION_VIEW_ROLE,
    RANDOM_MINIMAL_VIEW_ROLE,
)

_PATH_SUFFIX_BY_ROLE: Mapping[str, str] = {
    EVIDENCE_ROLE: "evidence",
    BASE_UPDATE_ROLE: "base-update",
    PREDICTION_VIEW_ROLE: "prediction-view",
    RANDOM_MINIMAL_VIEW_ROLE: "random-minimal-view",
}
_ARTIFACT_BY_ROLE: Mapping[str, str] = {
    EVIDENCE_ROLE: PREPARE_EVIDENCE_ARTIFACT,
    BASE_UPDATE_ROLE: BASE_UPDATE_ARTIFACT,
    PREDICTION_VIEW_ROLE: PREDICTION_VIEW_ARTIFACT,
    RANDOM_MINIMAL_VIEW_ROLE: RANDOM_MINIMAL_VIEW_ARTIFACT,
}

PROTOCOL_PAYLOAD_PATHS = (
    "policy-runs.jsonl",
    "protocol-census.json",
    "rotations.jsonl",
)
BASE_UPDATE_PAYLOAD_PATHS = (
    "base-contexts.jsonl",
    "base-model.json",
    "capability.json",
    "diversity-transform.json",
    "rotation.json",
)
PREDICTION_VIEW_PAYLOAD_PATHS = ("candidates.jsonl", "view-summary.json")
RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS = ("candidates.jsonl", "view-summary.json")
EVIDENCE_PAYLOAD_PATHS = (
    "pool-components.jsonl",
    "pool-novelty.jsonl",
    "pool-predictions.jsonl",
    "prepare-summary.json",
    "rotation.json",
)
CAMPAIGN_PAYLOAD_PATHS = ("prepare-index.jsonl", "prepare-summary.json")

_PAYLOAD_PATHS_BY_ROLE: Mapping[str, tuple[str, ...]] = {
    EVIDENCE_ROLE: EVIDENCE_PAYLOAD_PATHS,
    BASE_UPDATE_ROLE: BASE_UPDATE_PAYLOAD_PATHS,
    PREDICTION_VIEW_ROLE: PREDICTION_VIEW_PAYLOAD_PATHS,
    RANDOM_MINIMAL_VIEW_ROLE: RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS,
}
_PREDICTION_FIELDS = (
    "rotation_id",
    "sequence_id",
    "sequence",
    "objective_probabilities",
    "features",
    "novelty",
    "diversity_component_id",
    "eligible",
)
_RANDOM_MINIMAL_FIELDS = (
    "rotation_id",
    "sequence_id",
    "sequence",
    "diversity_component_id",
    "eligible",
)
_BASE_UPDATE_DATA_PAYLOAD_PATHS = (
    "base-contexts.jsonl",
    "base-model.json",
    "diversity-transform.json",
    "rotation.json",
)
_PHASES = frozenset(
    {"protocol", "prepare", "select", "reveal", "update", "outer-select", "finalize"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _exact_fields(
    value: object,
    fields: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _strict_json(payload: bytes, *, label: str) -> object:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be LF-terminated canonical JSON")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains invalid constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical compact JSON")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json(payload, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} must be nonempty canonical JSON Lines")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line, label=f"{label} row {index}")
        if type(value) is not dict:
            raise ValueError(f"{label} row {index} must be a JSON object")
        rows.append(value)
    return tuple(rows)


def _canonical_hex(value: object, *, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical binary64 hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not a binary64 hex string") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{label} is not canonical finite binary64 hex")
    return parsed


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{label} must be an exact string array")
    return tuple(value)


def _scope_id(value: object) -> str:
    if type(value) is not str:
        raise ValueError("publication scope_id must be canonical text")
    if value == "global":
        return value
    try:
        rotation_by_id(value)
    except ValueError:
        try:
            policy_run_by_track_id(value)
        except ValueError as error:
            raise ValueError(
                "publication scope_id is not global, a rotation, or a track"
            ) from error
    return value


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    """Require the exact registered rotation and built-in fold scalar types."""

    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    canonical = rotation_by_id(value.rotation_id)
    if canonical.outer_fold != value.outer_fold or canonical.pool_fold != value.pool_fold:
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


@dataclass(frozen=True, slots=True)
class SequentialV2PublicationIdentity:
    """Exact code/config/lock identity shared by executable phase receipts."""

    git_commit: str
    code_manifest_sha256: str
    config_sha256: str
    lock_sha256: str

    def __post_init__(self) -> None:
        if type(self.git_commit) is not str or _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("publication git_commit must be forty lowercase hex characters")
        _sha256(self.code_manifest_sha256, label="publication code manifest")
        _sha256(self.config_sha256, label="publication config")
        _sha256(self.lock_sha256, label="publication lock")

    def metadata(self, *, phase: str, scope_id: str) -> dict[str, object]:
        """Return exactly the frozen seven-field receipt metadata object."""

        if type(phase) is not str or phase not in _PHASES:
            raise ValueError(
                "publication phase is outside "
                "protocol/prepare/select/reveal/update/outer-select/finalize"
            )
        scope = _scope_id(scope_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "scope_id": scope,
            "git_commit": self.git_commit,
            "code_manifest_sha256": self.code_manifest_sha256,
            "config_sha256": self.config_sha256,
            "lock_sha256": self.lock_sha256,
        }

    def verify_metadata(self, payload: bytes, *, phase: str, scope_id: str) -> None:
        """Require canonical receipt metadata to equal this exact identity."""

        expected = canonical_json_bytes(self.metadata(phase=phase, scope_id=scope_id))
        if type(payload) is not bytes or payload != expected:
            raise ValueError(
                "phase receipt metadata differs from the expected publication identity"
            )


def descriptor_logistic_state_from_document(value: object) -> DescriptorLogisticState:
    """Strict inverse of :meth:`DescriptorLogisticState.document`."""

    document = _exact_fields(
        value,
        {
            "schema_version",
            "artifact",
            "converged",
            "feature_names",
            "feature_mean_hex",
            "feature_scale_hex",
            "strains",
            "coefficient_hex",
            "constant_probability_hex",
            "iterations",
            "training_contexts",
            "positive_contexts",
            "hyperparameters",
        },
        label="descriptor-logistic state",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["artifact"]) is not str
        or document["artifact"] != "sequential_v2_pooled_descriptor_logistic_state"
        or document["converged"] is not True
        or type(document["feature_names"]) is not list
        or document["feature_names"] != list(FEATURE_NAMES)
        or any(type(item) is not str for item in document["feature_names"])
    ):
        raise ValueError("descriptor-logistic identity or feature contract changed")
    hyperparameters = _exact_fields(
        document["hyperparameters"],
        {
            "l2_hex",
            "max_iterations",
            "prior_strength_hex",
            "probability_clip_hex",
            "tolerance_hex",
        },
        label="descriptor-logistic hyperparameters",
    )
    expected_hyperparameters = {
        "l2_hex": LOGISTIC_L2.hex(),
        "max_iterations": LOGISTIC_MAX_ITERATIONS,
        "prior_strength_hex": LOGISTIC_PRIOR_STRENGTH.hex(),
        "probability_clip_hex": PROBABILITY_CLIP.hex(),
        "tolerance_hex": LOGISTIC_TOLERANCE.hex(),
    }
    if set(hyperparameters) != set(expected_hyperparameters) or any(
        type(hyperparameters[key]) is not type(expected_hyperparameters[key])
        or hyperparameters[key] != expected_hyperparameters[key]
        for key in expected_hyperparameters
    ):
        raise ValueError("descriptor-logistic hyperparameters differ from the frozen contract")

    means = document["feature_mean_hex"]
    scales = document["feature_scale_hex"]
    if type(means) is not list or type(scales) is not list:
        raise ValueError("descriptor-logistic feature state must be arrays")
    feature_mean = tuple(
        _canonical_hex(item, label=f"descriptor feature mean {index}")
        for index, item in enumerate(means)
    )
    feature_scale = tuple(
        _canonical_hex(item, label=f"descriptor feature scale {index}")
        for index, item in enumerate(scales)
    )
    strains = _string_tuple(document["strains"], label="descriptor strains")
    if (
        not strains
        or strains != tuple(sorted(set(strains)))
        or any(strain not in TARGETS for strain in strains)
    ):
        raise ValueError("descriptor strains must be ascending unique frozen targets")
    raw_coefficient = document["coefficient_hex"]
    if raw_coefficient is None:
        coefficient = None
    elif type(raw_coefficient) is list:
        coefficient = tuple(
            _canonical_hex(item, label=f"descriptor coefficient {index}")
            for index, item in enumerate(raw_coefficient)
        )
    else:
        raise ValueError("descriptor coefficients must be null or an exact array")
    raw_constant = document["constant_probability_hex"]
    constant_probability = (
        None
        if raw_constant is None
        else _canonical_hex(raw_constant, label="descriptor constant probability")
    )
    state = DescriptorLogisticState(
        strains=strains,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficient=coefficient,
        constant_probability=constant_probability,
        iterations=_exact_int(document["iterations"], label="descriptor iterations"),
        training_contexts=_exact_int(
            document["training_contexts"], label="descriptor training contexts", minimum=1
        ),
        positive_contexts=_exact_int(
            document["positive_contexts"], label="descriptor positive contexts"
        ),
        converged=True,
    )
    if canonical_json_bytes(state.document()) != canonical_json_bytes(document):
        raise ValueError("descriptor-logistic state does not reconstruct exact document bytes")
    return state


def descriptor_logistic_state_from_bytes(payload: bytes) -> DescriptorLogisticState:
    """Decode one canonical descriptor-state payload without accepting aliases."""

    return descriptor_logistic_state_from_document(
        _strict_json_object(payload, label="descriptor-logistic state")
    )


def diversity_transform_from_document(
    value: object,
    *,
    fit_sequence_ids: Sequence[str],
) -> DiversityTransform:
    """Strictly decode a transform, using authenticated IDs hidden behind its digest."""

    document = _exact_fields(
        value,
        {
            "schema_version",
            "artifact",
            "feature_names",
            "fit_sequence_count",
            "fit_sequence_ids_sha256",
            "mean_hex",
            "scale_hex",
            "scale_floor_rule",
            "row_normalization",
        },
        label="diversity transform",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["artifact"]) is not str
        or document["artifact"] != "sequential_v2_fold_local_diversity_transform"
        or type(document["feature_names"]) is not list
        or document["feature_names"] != list(FEATURE_NAMES)
        or any(type(item) is not str for item in document["feature_names"])
        or type(document["scale_floor_rule"]) is not str
        or document["scale_floor_rule"] != "population_sd_below_1e-12_replaced_by_one"
        or type(document["row_normalization"]) is not str
        or document["row_normalization"] != "euclidean_l2_zero_vector_stays_zero"
    ):
        raise ValueError("diversity transform identity or frozen recipe changed")
    ids = tuple(fit_sequence_ids)
    if (
        not ids
        or ids != tuple(sorted(set(ids)))
        or any(type(item) is not str or _SHA256.fullmatch(item) is None for item in ids)
    ):
        raise ValueError("diversity transform fit IDs must be sorted unique SHA-256 values")
    if _exact_int(
        document["fit_sequence_count"],
        label="diversity fit sequence count",
        minimum=1,
    ) != len(ids) or _sha256(
        document["fit_sequence_ids_sha256"], label="diversity fit sequence IDs"
    ) != id_stream_sha256(ids):
        raise ValueError("diversity transform fit-ID evidence differs from its capability")
    means = document["mean_hex"]
    scales = document["scale_hex"]
    if type(means) is not list or type(scales) is not list:
        raise ValueError("diversity transform mean and scale must be arrays")
    transform = DiversityTransform(
        mean=tuple(
            _canonical_hex(item, label=f"diversity mean {index}")
            for index, item in enumerate(means)
        ),
        scale=tuple(
            _canonical_hex(item, label=f"diversity scale {index}")
            for index, item in enumerate(scales)
        ),
        fit_sequence_ids=ids,
    )
    if canonical_json_bytes(transform.document()) != canonical_json_bytes(document):
        raise ValueError("diversity transform does not reconstruct exact document bytes")
    return transform


def diversity_transform_from_bytes_and_ids(
    payload: bytes,
    fit_sequence_ids: Sequence[str],
) -> DiversityTransform:
    """Decode canonical transform-state bytes with authenticated fit IDs."""

    return diversity_transform_from_document(
        _strict_json_object(payload, label="diversity transform"),
        fit_sequence_ids=fit_sequence_ids,
    )


def _context_document(row: ContextRow) -> dict[str, object]:
    if type(row) is not ContextRow:
        raise TypeError("prepare base contexts must contain exact ContextRow values")
    return {
        "example_id": row.example_id,
        "assay_context_id": row.assay_context_id,
        "sequence_id": row.sequence_id,
        "sequence": row.sequence,
        "target": row.target,
        "gram": row.gram,
        "fold": row.fold,
        "label": row.label,
        "source_observations": row.source_observations,
    }


def _context_from_document(value: object, *, label: str) -> ContextRow:
    document = _exact_fields(
        value,
        {
            "example_id",
            "assay_context_id",
            "sequence_id",
            "sequence",
            "target",
            "gram",
            "fold",
            "label",
            "source_observations",
        },
        label=label,
    )
    for field in (
        "example_id",
        "assay_context_id",
        "sequence_id",
        "sequence",
        "target",
        "gram",
    ):
        if type(document[field]) is not str:
            raise ValueError(f"{label} {field} must be exact text")
    example_id = _sha256(document["example_id"], label=f"{label} example ID")
    assay_context_id = _sha256(
        document["assay_context_id"],
        label=f"{label} assay-context ID",
    )
    if assay_context_id != example_id:
        raise ValueError(f"{label} assay-context ID must equal its example ID")
    row = ContextRow(
        example_id=example_id,
        assay_context_id=assay_context_id,
        sequence_id=document["sequence_id"],
        sequence=document["sequence"],
        target=document["target"],
        gram=document["gram"],
        fold=_exact_int(document["fold"], label=f"{label} fold"),
        label=_exact_int(document["label"], label=f"{label} label"),
        source_observations=_exact_int(
            document["source_observations"],
            label=f"{label} source observations",
            minimum=1,
        ),
    )
    if canonical_json_bytes(_context_document(row)) != canonical_json_bytes(document):
        raise ValueError(f"{label} does not reconstruct exactly")
    return row


def _base_context_payload_identity(
    contexts: tuple[ContextRow, ...],
    *,
    label: str,
) -> bytes:
    if type(contexts) is not tuple or any(type(row) is not ContextRow for row in contexts):
        raise TypeError(f"{label} must be an exact ContextRow tuple")
    for row in contexts:
        if any(
            type(getattr(row, field)) is not str
            for field in (
                "example_id",
                "assay_context_id",
                "sequence_id",
                "sequence",
                "target",
                "gram",
            )
        ) or any(
            type(getattr(row, field)) is not int
            for field in ("fold", "label", "source_observations")
        ):
            raise TypeError(f"{label} fields must use exact scalar types")
    return canonical_jsonl_bytes(_context_document(row) for row in contexts)


def _require_exact_prepare_capability(capability: PrepareCapability) -> None:
    """Reject subclass dispatch anywhere in an externally supplied source view."""

    if type(capability) is not PrepareCapability:
        raise TypeError("prepare source must be an exact PrepareCapability")
    _require_frozen_rotation(capability.spec, label="prepare source rotation")
    _base_context_payload_identity(
        capability.base_contexts,
        label="prepare source base contexts",
    )
    if type(capability.acquisition_metadata) is not tuple or any(
        type(row) is not LabelFreeContext for row in capability.acquisition_metadata
    ):
        raise TypeError(
            "prepare source acquisition metadata must be an exact LabelFreeContext tuple"
        )
    for row in capability.acquisition_metadata:
        if (
            any(
                type(getattr(row, field)) is not str
                for field in (
                    "example_id",
                    "assay_context_id",
                    "sequence_id",
                    "sequence",
                    "target",
                    "gram",
                )
            )
            or type(row.fold) is not int
        ):
            raise TypeError("prepare source acquisition metadata fields must use exact types")
    for values, label in (
        (capability.acquisition_support_sequence_ids, "support sequence IDs"),
        (capability.allowed_base_example_ids, "allowed base example IDs"),
        (
            capability.allowed_acquisition_metadata_example_ids,
            "allowed acquisition metadata example IDs",
        ),
    ):
        if type(values) is not tuple or any(type(item) is not str for item in values):
            raise TypeError(f"prepare source {label} must be an exact string tuple")


def _base_sequence_evidence(
    contexts: Sequence[ContextRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_id: dict[str, str] = {}
    for row in contexts:
        prior = by_id.setdefault(row.sequence_id, row.sequence)
        if prior != row.sequence:
            raise ValueError("base capability maps one sequence ID to multiple sequences")
    ids = tuple(sorted(by_id))
    return ids, tuple(by_id[sequence_id] for sequence_id in ids)


def _prepare_relative_path(spec: RotationSpec, role: str) -> str:
    _require_frozen_rotation(spec, label="prepare leaf rotation")
    try:
        suffix = _PATH_SUFFIX_BY_ROLE[role]
    except KeyError as error:
        raise ValueError(f"unknown prepare leaf role {role!r}") from error
    return f"prepare/rotations/{spec.rotation_id}/{suffix}"


@dataclass(frozen=True, slots=True)
class ProtocolCapability:
    """Rootless exact protocol phase used by every later worker."""

    seal: PhaseSeal

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("protocol capability must contain a rootless PhaseSeal")


@dataclass(frozen=True, slots=True)
class PrepareArtifactCapabilities:
    """Four rootless phase capabilities held only inside one prepare worker."""

    spec: RotationSpec
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    source_prepare_leaf_seal_sha256: str
    evidence: PhaseSeal
    base_update: PhaseSeal
    prediction_view: PhaseSeal
    random_minimal_view: PhaseSeal
    view_provenance: RotationViewProvenance

    def __post_init__(self) -> None:
        _require_frozen_rotation(self.spec, label="prepare artifacts rotation")
        _sha256(self.protocol_seal_sha256, label="prepare protocol seal")
        _sha256(self.stage_global_seal_sha256, label="prepare stage-global seal")
        _sha256(self.source_prepare_leaf_seal_sha256, label="prepare source leaf seal")
        if any(
            type(seal) is not PhaseSeal
            for seal in (
                self.evidence,
                self.base_update,
                self.prediction_view,
                self.random_minimal_view,
            )
        ):
            raise TypeError("prepare outputs must be rootless PhaseSeal capabilities")
        if type(self.view_provenance) is not RotationViewProvenance:
            raise TypeError("prepare outputs require typed view provenance")
        _require_frozen_rotation(
            self.view_provenance.spec,
            label="prepare view provenance rotation",
        )
        if self.view_provenance.spec != self.spec:
            raise ValueError("prepare view provenance has the wrong rotation")

    def seal_for_role(self, role: str) -> PhaseSeal:
        mapping = {
            EVIDENCE_ROLE: self.evidence,
            BASE_UPDATE_ROLE: self.base_update,
            PREDICTION_VIEW_ROLE: self.prediction_view,
            RANDOM_MINIMAL_VIEW_ROLE: self.random_minimal_view,
        }
        try:
            return mapping[role]
        except KeyError as error:
            raise ValueError(f"unknown prepare artifact role {role!r}") from error


@dataclass(frozen=True, slots=True)
class PrepareLeafAttestation:
    """Payload-free digest record for one semantically verified prepare leaf."""

    role: str
    relative_path: str
    artifact: str
    phase_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    base_model_state_sha256: str | None
    candidate_count: int | None
    candidate_ids_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in ROLE_ORDER:
            raise ValueError("prepare leaf attestation role is invalid")
        if (
            type(self.relative_path) is not str
            or type(self.artifact) is not str
            or self.artifact != _ARTIFACT_BY_ROLE[self.role]
        ):
            raise ValueError("prepare leaf attestation identity is invalid")
        _sha256(self.phase_seal_sha256, label="prepare leaf attestation seal")
        expected_paths = tuple(sorted(_PAYLOAD_PATHS_BY_ROLE[self.role]))
        if (
            type(self.payload_sha256) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                for item in self.payload_sha256
            )
            or tuple(path for path, _digest in self.payload_sha256) != expected_paths
            or len(dict(self.payload_sha256)) != len(self.payload_sha256)
        ):
            raise ValueError("prepare leaf attestation payload inventory is invalid")
        for path, digest in self.payload_sha256:
            _sha256(digest, label=f"prepare leaf attestation payload {path}")
        if self.role == BASE_UPDATE_ROLE:
            if self.candidate_count is not None or self.candidate_ids_sha256 is not None:
                raise ValueError("base-update attestation cannot claim candidate provenance")
            _sha256(
                self.base_model_state_sha256,
                label="base-update attestation model-state digest",
            )
        elif self.base_model_state_sha256 is not None:
            raise ValueError("non-base prepare attestation cannot claim a base model state")
        elif (
            type(self.candidate_count) is not int
            or self.candidate_count < 1
            or _sha256(
                self.candidate_ids_sha256,
                label="prepare leaf attestation candidate IDs",
            )
            != self.candidate_ids_sha256
        ):
            raise ValueError("prepare leaf attestation lacks candidate provenance")

    def document(self) -> dict[str, object]:
        """Return the canonical label-free campaign-index row body."""

        return {
            "schema_version": SCHEMA_VERSION,
            "leaf_role": _PATH_SUFFIX_BY_ROLE[self.role],
            "relative_path": self.relative_path,
            "artifact": self.artifact,
            "phase_seal_sha256": self.phase_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "base_model_state_sha256": self.base_model_state_sha256,
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrepareRotationAttestation:
    """Safe result of one supervised, semantically verifying prepare worker.

    This is a procedural attestation, not a standalone authenticated
    capability.  Its authority is the supervisor-controlled fresh-exec result
    channel from the worker that exclusively held this rotation's source and
    four output capabilities.  It deliberately retains only public identities,
    censuses, and digests; the campaign process must never receive the omitted
    ``PhaseSeal`` or payload objects.
    """

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    source_prepare_leaf_seal_sha256: str
    base_context_association_count: int
    leaves: tuple[PrepareLeafAttestation, ...]
    view_provenance: RotationViewProvenance

    def __post_init__(self) -> None:
        _require_frozen_rotation(self.spec, label="prepare rotation attestation rotation")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("prepare rotation attestation requires an exact publication identity")
        _sha256(self.protocol_seal_sha256, label="prepare attestation protocol seal")
        _sha256(self.stage_global_seal_sha256, label="prepare attestation stage-global seal")
        _sha256(
            self.source_prepare_leaf_seal_sha256,
            label="prepare attestation source leaf seal",
        )
        expected_base_contexts = sum(
            EXPECTED_CONTEXTS_BY_FOLD[fold] for fold in self.spec.base_folds
        )
        if (
            type(self.base_context_association_count) is not int
            or self.base_context_association_count != expected_base_contexts
        ):
            raise ValueError("prepare attestation base-context census changed")
        if (
            type(self.leaves) is not tuple
            or len(self.leaves) != len(ROLE_ORDER)
            or any(type(leaf) is not PrepareLeafAttestation for leaf in self.leaves)
            or tuple(leaf.role for leaf in self.leaves) != ROLE_ORDER
        ):
            raise ValueError("prepare attestation requires four exact leaves in frozen order")
        if any(
            leaf.relative_path != _prepare_relative_path(self.spec, leaf.role)
            for leaf in self.leaves
        ):
            raise ValueError("prepare attestation leaf path differs from its rotation")
        if len({leaf.phase_seal_sha256 for leaf in self.leaves}) != len(self.leaves):
            raise ValueError("prepare attestation leaf seals must be distinct")
        if (
            type(self.view_provenance) is not RotationViewProvenance
            or self.view_provenance.spec != self.spec
            or type(self.view_provenance.candidate_count) is not int
            or any(
                type(value) is not str
                for value in (
                    self.view_provenance.candidate_ids_sha256,
                    self.view_provenance.prediction_view_seal_sha256,
                    self.view_provenance.prediction_view_payload_sha256,
                    self.view_provenance.random_view_seal_sha256,
                    self.view_provenance.random_view_payload_sha256,
                )
            )
        ):
            raise ValueError("prepare attestation view provenance has the wrong rotation")
        _require_frozen_rotation(
            self.view_provenance.spec,
            label="prepare attestation view provenance rotation",
        )
        expected_candidate_count = EXPECTED_SUPPORT_BY_FOLD[self.spec.pool_fold]
        if self.view_provenance.candidate_count != expected_candidate_count:
            raise ValueError("prepare attestation candidate census changed")
        for role in (EVIDENCE_ROLE, PREDICTION_VIEW_ROLE, RANDOM_MINIMAL_VIEW_ROLE):
            leaf = self.leaf_for_role(role)
            if (
                leaf.candidate_count != self.view_provenance.candidate_count
                or leaf.candidate_ids_sha256 != self.view_provenance.candidate_ids_sha256
            ):
                raise ValueError("prepare attestation candidate identities disagree")
        prediction = self.leaf_for_role(PREDICTION_VIEW_ROLE)
        random_minimal = self.leaf_for_role(RANDOM_MINIMAL_VIEW_ROLE)
        if (
            prediction.phase_seal_sha256 != self.view_provenance.prediction_view_seal_sha256
            or dict(prediction.payload_sha256)["candidates.jsonl"]
            != self.view_provenance.prediction_view_payload_sha256
            or random_minimal.phase_seal_sha256 != self.view_provenance.random_view_seal_sha256
            or dict(random_minimal.payload_sha256)["candidates.jsonl"]
            != self.view_provenance.random_view_payload_sha256
        ):
            raise ValueError("prepare attestation leaf and view identities disagree")

    def leaf_for_role(self, role: str) -> PrepareLeafAttestation:
        """Return one digest-only leaf record from the exact four-leaf tuple."""

        if type(role) is not str or role not in ROLE_ORDER:
            raise ValueError(f"unknown prepare attestation role {role!r}")
        matches = tuple(leaf for leaf in self.leaves if leaf.role == role)
        if len(matches) != 1:
            raise ValueError("prepare attestation lacks one exact leaf role")
        return matches[0]

    def document(self) -> dict[str, object]:
        """Serialize the payload-free worker result for its trusted IPC channel."""

        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": PREPARE_ROTATION_ATTESTATION_ARTIFACT,
            "rotation": self.spec.document(),
            "publication_identity": {
                "git_commit": self.publication_identity.git_commit,
                "code_manifest_sha256": self.publication_identity.code_manifest_sha256,
                "config_sha256": self.publication_identity.config_sha256,
                "lock_sha256": self.publication_identity.lock_sha256,
            },
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "stage_global_seal_sha256": self.stage_global_seal_sha256,
            "source_prepare_leaf_seal_sha256": self.source_prepare_leaf_seal_sha256,
            "base_context_association_count": self.base_context_association_count,
            "leaves": [leaf.document() for leaf in self.leaves],
            "view_provenance": {
                "candidate_count": self.view_provenance.candidate_count,
                "candidate_ids_sha256": self.view_provenance.candidate_ids_sha256,
                "prediction_view_seal_sha256": (self.view_provenance.prediction_view_seal_sha256),
                "prediction_view_payload_sha256": (
                    self.view_provenance.prediction_view_payload_sha256
                ),
                "random_view_seal_sha256": self.view_provenance.random_view_seal_sha256,
                "random_view_payload_sha256": (self.view_provenance.random_view_payload_sha256),
            },
        }

    def canonical_bytes(self) -> bytes:
        """Encode the attestation as canonical LF-terminated IPC JSON."""

        return canonical_json_bytes(self.document())


def prepare_rotation_attestation_from_document(
    value: object,
) -> PrepareRotationAttestation:
    """Strictly decode one label-free fresh-exec worker result."""

    document = _exact_fields(
        value,
        {
            "schema_version",
            "artifact",
            "rotation",
            "publication_identity",
            "protocol_seal_sha256",
            "stage_global_seal_sha256",
            "source_prepare_leaf_seal_sha256",
            "base_context_association_count",
            "leaves",
            "view_provenance",
        },
        label="prepare rotation attestation",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["artifact"]) is not str
        or document["artifact"] != PREPARE_ROTATION_ATTESTATION_ARTIFACT
    ):
        raise ValueError("prepare rotation attestation identity changed")
    rotation = _exact_fields(
        document["rotation"],
        {
            "schema_version",
            "rotation_id",
            "outer_fold",
            "acquisition_pool_fold",
            "base_folds",
        },
        label="prepare attestation rotation",
    )
    if (
        type(rotation["schema_version"]) is not int
        or type(rotation["rotation_id"]) is not str
        or type(rotation["base_folds"]) is not list
        or any(type(fold) is not int for fold in rotation["base_folds"])
    ):
        raise ValueError("prepare attestation rotation uses non-exact field types")
    spec = RotationSpec(
        outer_fold=_exact_int(rotation["outer_fold"], label="attestation outer fold"),
        pool_fold=_exact_int(
            rotation["acquisition_pool_fold"],
            label="attestation acquisition pool fold",
        ),
    )
    if canonical_json_bytes(rotation) != canonical_json_bytes(spec.document()):
        raise ValueError("prepare attestation rotation document changed")
    identity_document = _exact_fields(
        document["publication_identity"],
        {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"},
        label="prepare attestation publication identity",
    )
    if any(type(identity_document[key]) is not str for key in identity_document):
        raise ValueError("prepare attestation publication identity must contain exact text")
    identity = SequentialV2PublicationIdentity(
        git_commit=identity_document["git_commit"],
        code_manifest_sha256=identity_document["code_manifest_sha256"],
        config_sha256=identity_document["config_sha256"],
        lock_sha256=identity_document["lock_sha256"],
    )
    raw_leaves = document["leaves"]
    if type(raw_leaves) is not list or len(raw_leaves) != len(ROLE_ORDER):
        raise ValueError("prepare rotation attestation requires four leaf records")
    leaves: list[PrepareLeafAttestation] = []
    for index, (raw_leaf, role) in enumerate(zip(raw_leaves, ROLE_ORDER, strict=True)):
        leaf = _exact_fields(
            raw_leaf,
            {
                "schema_version",
                "leaf_role",
                "relative_path",
                "artifact",
                "phase_seal_sha256",
                "payload_sha256",
                "base_model_state_sha256",
                "candidate_count",
                "candidate_ids_sha256",
            },
            label=f"prepare attestation leaf {index}",
        )
        payloads = _exact_fields(
            leaf["payload_sha256"],
            set(_PAYLOAD_PATHS_BY_ROLE[role]),
            label=f"prepare attestation leaf payload map {index}",
        )
        if (
            type(leaf["schema_version"]) is not int
            or leaf["schema_version"] != SCHEMA_VERSION
            or type(leaf["leaf_role"]) is not str
            or leaf["leaf_role"] != _PATH_SUFFIX_BY_ROLE[role]
            or any(
                type(path) is not str or type(digest) is not str
                for path, digest in payloads.items()
            )
        ):
            raise ValueError("prepare attestation leaf identity or payload map changed")
        candidate_count = leaf["candidate_count"]
        if candidate_count is not None:
            candidate_count = _exact_int(
                candidate_count,
                label=f"prepare attestation candidate count {index}",
                minimum=1,
            )
        candidate_ids_sha256 = leaf["candidate_ids_sha256"]
        if candidate_ids_sha256 is not None and type(candidate_ids_sha256) is not str:
            raise ValueError("prepare attestation candidate digest must be text or null")
        base_model_state_sha256 = leaf["base_model_state_sha256"]
        if base_model_state_sha256 is not None and type(base_model_state_sha256) is not str:
            raise ValueError("prepare attestation model-state digest must be text or null")
        leaves.append(
            PrepareLeafAttestation(
                role=role,
                relative_path=leaf["relative_path"],
                artifact=leaf["artifact"],
                phase_seal_sha256=leaf["phase_seal_sha256"],
                payload_sha256=tuple(sorted(payloads.items())),
                base_model_state_sha256=base_model_state_sha256,
                candidate_count=candidate_count,
                candidate_ids_sha256=candidate_ids_sha256,
            )
        )
    provenance_document = _exact_fields(
        document["view_provenance"],
        {
            "candidate_count",
            "candidate_ids_sha256",
            "prediction_view_seal_sha256",
            "prediction_view_payload_sha256",
            "random_view_seal_sha256",
            "random_view_payload_sha256",
        },
        label="prepare attestation view provenance",
    )
    if any(
        type(provenance_document[key]) is not str
        for key in provenance_document
        if key != "candidate_count"
    ):
        raise ValueError("prepare attestation view digests must be exact text")
    provenance = RotationViewProvenance(
        spec=spec,
        candidate_count=_exact_int(
            provenance_document["candidate_count"],
            label="prepare attestation view candidate count",
            minimum=1,
        ),
        candidate_ids_sha256=provenance_document["candidate_ids_sha256"],
        prediction_view_seal_sha256=provenance_document["prediction_view_seal_sha256"],
        prediction_view_payload_sha256=provenance_document["prediction_view_payload_sha256"],
        random_view_seal_sha256=provenance_document["random_view_seal_sha256"],
        random_view_payload_sha256=provenance_document["random_view_payload_sha256"],
    )
    attestation = PrepareRotationAttestation(
        spec=spec,
        publication_identity=identity,
        protocol_seal_sha256=document["protocol_seal_sha256"],
        stage_global_seal_sha256=document["stage_global_seal_sha256"],
        source_prepare_leaf_seal_sha256=document["source_prepare_leaf_seal_sha256"],
        base_context_association_count=_exact_int(
            document["base_context_association_count"],
            label="prepare attestation base-context count",
            minimum=1,
        ),
        leaves=tuple(leaves),
        view_provenance=provenance,
    )
    if canonical_json_bytes(attestation.document()) != canonical_json_bytes(document):
        raise ValueError("prepare rotation attestation does not reconstruct exact document")
    return attestation


def prepare_rotation_attestation_from_bytes(payload: bytes) -> PrepareRotationAttestation:
    """Decode one canonical payload-free attestation from trusted IPC bytes."""

    return prepare_rotation_attestation_from_document(
        _strict_json_object(payload, label="prepare rotation attestation")
    )


@dataclass(frozen=True, slots=True)
class BaseUpdateCapability:
    """Base labels and accepted fitted states, with pool row/candidate metadata removed."""

    spec: RotationSpec
    contexts: tuple[ContextRow, ...]
    allowed_example_ids: tuple[str, ...]
    base_model: DescriptorLogisticState
    diversity_transform: DiversityTransform
    base_update_seal_sha256: str

    def __post_init__(self) -> None:
        _require_frozen_rotation(self.spec, label="base-update capability rotation")
        _sha256(self.base_update_seal_sha256, label="base-update phase seal")
        if type(self.contexts) is not tuple or any(
            type(row) is not ContextRow for row in self.contexts
        ):
            raise TypeError("base-update contexts must be an immutable ContextRow tuple")
        ids = tuple(row.example_id for row in self.contexts)
        if (
            not ids
            or ids != tuple(sorted(set(ids)))
            or self.allowed_example_ids != ids
            or any(row.fold not in self.spec.base_folds for row in self.contexts)
            or {row.fold for row in self.contexts} != set(self.spec.base_folds)
        ):
            raise ValueError("base-update capability differs from the exact three base folds")
        base_sequence_ids, base_sequences = _base_sequence_evidence(self.contexts)
        if fit_descriptor_logistic(self.contexts) != self.base_model:
            raise ValueError("base-update model differs from an exact base-context refit")
        if (
            self.diversity_transform.fit_sequence_ids != base_sequence_ids
            or fit_diversity_transform(base_sequences) != self.diversity_transform
        ):
            raise ValueError("base-update transform differs from exact base sequences")


@dataclass(frozen=True, slots=True)
class DecodedPrepareArtifacts:
    """Semantically reconstructed result of four authenticated capabilities."""

    capabilities: PrepareArtifactCapabilities
    prepared: PreparedRotation
    base_update: BaseUpdateCapability


@dataclass(frozen=True, slots=True)
class PrepareCampaignCapability:
    """Least-authority, label-free global prepare marker."""

    seal: PhaseSeal
    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("prepare campaign barrier must be a rootless PhaseSeal")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("prepare campaign barrier requires a publication identity")
        _decode_safe_campaign(self.seal, publication_identity=self.publication_identity)

    @property
    def protocol_seal_sha256(self) -> str:
        """Return the sole authenticated protocol predecessor digest."""

        _rows, _summary, protocol = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
        )
        return protocol

    def leaf_seal_sha256(self, *, spec: RotationSpec, role: str) -> str:
        """Look up one indexed prepare-leaf digest without exposing sibling bytes."""

        row = _campaign_index_row(self, spec=spec, role=role)
        return row["phase_seal_sha256"]

    def leaf_payload_sha256(self, *, spec: RotationSpec, role: str, path: str) -> str:
        """Look up one indexed payload digest without exposing any payload bytes."""

        row = _campaign_index_row(self, spec=spec, role=role)
        if type(path) is not str or path not in _PAYLOAD_PATHS_BY_ROLE[role]:
            raise ValueError("prepare index payload lookup path is invalid for its role")
        return _sha256(
            row["payload_sha256"][path],
            label="prepare index payload lookup digest",
        )

    def base_model_state_sha256(self, *, spec: RotationSpec) -> str:
        """Return the canonical fitted-state digest for one base-update leaf."""

        row = _campaign_index_row(self, spec=spec, role=BASE_UPDATE_ROLE)
        return _sha256(
            row["base_model_state_sha256"],
            label="prepare index base model-state digest",
        )

    def evidence_seal_sha256(self, *, spec: RotationSpec) -> str:
        """Return one rotation's evidence digest, never its evidence bytes."""

        return self.leaf_seal_sha256(spec=spec, role=EVIDENCE_ROLE)

    def rotation_view_provenance(self, *, spec: RotationSpec) -> RotationViewProvenance:
        """Derive the two view identities solely from the authenticated global index."""

        prediction = _campaign_index_row(self, spec=spec, role=PREDICTION_VIEW_ROLE)
        random_minimal = _campaign_index_row(
            self,
            spec=spec,
            role=RANDOM_MINIMAL_VIEW_ROLE,
        )
        if (
            prediction["candidate_count"] != random_minimal["candidate_count"]
            or prediction["candidate_ids_sha256"] != random_minimal["candidate_ids_sha256"]
        ):
            raise ValueError("campaign index candidate views do not share one ID stream")
        return RotationViewProvenance(
            spec=spec,
            candidate_count=prediction["candidate_count"],
            candidate_ids_sha256=prediction["candidate_ids_sha256"],
            prediction_view_seal_sha256=prediction["phase_seal_sha256"],
            prediction_view_payload_sha256=prediction["payload_sha256"]["candidates.jsonl"],
            random_view_seal_sha256=random_minimal["phase_seal_sha256"],
            random_view_payload_sha256=random_minimal["payload_sha256"]["candidates.jsonl"],
        )


def _protocol_payloads() -> dict[str, bytes]:
    return {
        "rotations.jsonl": canonical_jsonl_bytes(spec.document() for spec in ordered_rotations()),
        "policy-runs.jsonl": canonical_jsonl_bytes(run.document() for run in ordered_policy_runs()),
        "protocol-census.json": canonical_json_bytes(protocol_census()),
    }


def publish_protocol_capability(
    destination: str | Path,
    *,
    publication_identity: SequentialV2PublicationIdentity,
) -> ProtocolCapability:
    """Publish the standalone frozen protocol and return no filesystem handle."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("protocol publisher requires a publication identity")
    seal = publish_phase(
        destination,
        artifact=PROTOCOL_ARTIFACT,
        payloads=_protocol_payloads(),
        predecessor_seals={},
        metadata=publication_identity.metadata(phase="protocol", scope_id="global"),
    )
    return verify_protocol_capability(seal, publication_identity=publication_identity)


def verify_protocol_capability(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
) -> ProtocolCapability:
    """Authenticate protocol bytes against the frozen graph and run identity."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("protocol verifier requires a publication identity")
    verified = verify_phase_capability(
        seal,
        expected_artifact=PROTOCOL_ARTIFACT,
        expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
        expected_predecessor_seals={},
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="protocol",
        scope_id="global",
    )
    for path, expected in _protocol_payloads().items():
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"protocol payload differs from the frozen graph: {path}")
    return ProtocolCapability(verified)


def _stage_index_from_source_leaf(seal: PhaseSeal, *, spec: RotationSpec) -> StageLeafIndex:
    _require_frozen_rotation(spec, label="stage source rotation")
    verified = verify_phase_capability(
        seal,
        expected_artifact=LEAF_ARTIFACTS[PREPARE_ROLE],
        expected_payload_paths=LEAF_PAYLOAD_PATHS[PREPARE_ROLE],
    )
    capability = _exact_fields(
        _strict_json_object(
            verified.read_payload_bytes("capability.json"),
            label="stage prepare capability",
        ),
        {
            "schema_version",
            "artifact",
            "rotation_id",
            "role",
            "outer_fold",
            "acquisition_pool_fold",
            "base_folds",
            "data_payload_paths",
            "row_counts",
            "id_streams",
        },
        label="stage prepare capability",
    )
    if (
        type(capability["schema_version"]) is not int
        or capability["schema_version"] != SCHEMA_VERSION
        or type(capability["artifact"]) is not str
        or capability["artifact"] != LEAF_CAPABILITY_ARTIFACT
        or type(capability["rotation_id"]) is not str
        or capability["rotation_id"] != spec.rotation_id
        or type(capability["role"]) is not str
        or capability["role"] != PREPARE_ROLE
        or type(capability["outer_fold"]) is not int
        or capability["outer_fold"] != spec.outer_fold
        or type(capability["acquisition_pool_fold"]) is not int
        or capability["acquisition_pool_fold"] != spec.pool_fold
        or type(capability["base_folds"]) is not list
        or capability["base_folds"] != list(spec.base_folds)
        or any(type(item) is not int for item in capability["base_folds"])
        or type(capability["data_payload_paths"]) is not list
        or capability["data_payload_paths"] != list(LEAF_DATA_PAYLOAD_PATHS[PREPARE_ROLE])
    ):
        raise ValueError("stage prepare capability identity differs from its rotation")
    row_counts_raw = _exact_fields(
        capability["row_counts"],
        set(LEAF_DATA_PAYLOAD_PATHS[PREPARE_ROLE]),
        label="stage prepare row counts",
    )
    row_counts = tuple(
        (
            path,
            _exact_int(
                row_counts_raw[path],
                label=f"stage prepare row count {path}",
                minimum=1,
            ),
        )
        for path in LEAF_DATA_PAYLOAD_PATHS[PREPARE_ROLE]
    )
    expected_streams = (
        "base_example_ids",
        "pool_metadata_example_ids",
        "pool_support_sequence_ids",
    )
    streams_raw = _exact_fields(
        capability["id_streams"],
        set(expected_streams),
        label="stage prepare ID streams",
    )
    streams: list[tuple[str, IdStreamBinding]] = []
    for name in expected_streams:
        binding = _exact_fields(
            streams_raw[name],
            {"count", "sha256"},
            label=f"stage prepare ID stream {name}",
        )
        streams.append(
            (
                name,
                IdStreamBinding(
                    count=_exact_int(
                        binding["count"],
                        label=f"stage prepare ID stream count {name}",
                        minimum=1,
                    ),
                    sha256=_sha256(binding["sha256"], label=f"stage prepare ID stream {name}"),
                ),
            )
        )
    return StageLeafIndex(
        spec=spec,
        role=PREPARE_ROLE,
        relative_path=leaf_relative_path(spec, PREPARE_ROLE),
        leaf_artifact=LEAF_ARTIFACTS[PREPARE_ROLE],
        leaf_seal_sha256=verified.seal_sha256,
        payload_paths=LEAF_PAYLOAD_PATHS[PREPARE_ROLE],
        row_counts=row_counts,
        id_streams=tuple(streams),
    )


def _prepare_capability_from_source_leaf(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
) -> tuple[PrepareCapability, StageLeafIndex]:
    entry = _stage_index_from_source_leaf(seal, spec=spec)
    metadata = _exact_fields(
        _strict_json_object(seal.metadata_json, label="stage prepare metadata"),
        {"schema_version", "rotation_id", "role", "source_anchors_sha256"},
        label="stage prepare metadata",
    )
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
        or type(metadata["rotation_id"]) is not str
        or metadata["rotation_id"] != spec.rotation_id
        or type(metadata["role"]) is not str
        or metadata["role"] != PREPARE_ROLE
    ):
        raise ValueError("stage prepare metadata differs from its rotation")
    anchors = _sha256(metadata["source_anchors_sha256"], label="stage source anchors")
    capsule = AuthenticatedLeafCapsule(
        entry=entry,
        seal=seal,
        source_anchors_sha256=anchors,
        source_predecessors=seal.predecessor_seals,
    )
    return prepare_capability_from_capsule(capsule), entry


def _verify_stage_global_binding(
    stage_global: PhaseSeal,
    *,
    entry: StageLeafIndex,
    expected_stage_global_seal_sha256: str,
) -> None:
    verified = verify_phase_capability(
        stage_global,
        expected_artifact=STAGE_ARTIFACT,
        expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
        expected_seal_sha256=_sha256(
            expected_stage_global_seal_sha256,
            label="expected stage-global seal",
        ),
    )
    rows = _strict_jsonl(
        verified.read_payload_bytes("capability-index.jsonl"),
        label="stage capability index",
    )
    expected = canonical_json_bytes(entry.document())
    if sum(canonical_json_bytes(row) == expected for row in rows) != 1:
        raise ValueError("stage-global index does not bind one exact prepare source leaf")
    if dict(verified.predecessor_seals).get(entry.relative_path) != entry.leaf_seal_sha256:
        raise ValueError("stage-global predecessor closure does not bind the prepare source leaf")


def _authenticate_prepare_inputs(
    *,
    spec: RotationSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    source_prepare_leaf_seal: PhaseSeal,
) -> PrepareCapability:
    _require_frozen_rotation(spec, label="prepare worker rotation")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("prepare worker requires an authenticated ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    verify_phase_capability(
        stage_global_seal,
        expected_artifact=STAGE_ARTIFACT,
        expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
        expected_seal_sha256=_sha256(
            expected_stage_global_seal_sha256,
            label="expected stage-global seal",
        ),
    )
    source, entry = _prepare_capability_from_source_leaf(
        source_prepare_leaf_seal,
        spec=spec,
    )
    _require_exact_prepare_capability(source)
    _verify_stage_global_binding(
        stage_global_seal,
        entry=entry,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    if protocol.seal.seal_sha256 != protocol_capability.seal.seal_sha256:
        raise AssertionError("protocol verification changed its identity")
    return source


def _common_predecessors(
    spec: RotationSpec,
    *,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    source_prepare_leaf_seal_sha256: str,
) -> dict[str, str]:
    return {
        "protocol/SHA256SUMS": _sha256(protocol_seal_sha256, label="prepare protocol seal"),
        "stage/global/SHA256SUMS": _sha256(
            stage_global_seal_sha256, label="prepare stage-global seal"
        ),
        f"stage/rotations/{spec.rotation_id}/prepare-capability/SHA256SUMS": _sha256(
            source_prepare_leaf_seal_sha256,
            label="prepare source leaf seal",
        ),
    }


def _evidence_predecessors(capabilities: PrepareArtifactCapabilities) -> dict[str, str]:
    result = _common_predecessors(
        capabilities.spec,
        protocol_seal_sha256=capabilities.protocol_seal_sha256,
        stage_global_seal_sha256=capabilities.stage_global_seal_sha256,
        source_prepare_leaf_seal_sha256=capabilities.source_prepare_leaf_seal_sha256,
    )
    for role in (BASE_UPDATE_ROLE, PREDICTION_VIEW_ROLE, RANDOM_MINIMAL_VIEW_ROLE):
        result[f"{_prepare_relative_path(capabilities.spec, role)}/SHA256SUMS"] = (
            capabilities.seal_for_role(role).seal_sha256
        )
    return result


def _decode_model_wrapper(
    payload: bytes,
    *,
    spec: RotationSpec,
) -> tuple[DescriptorLogisticState, Mapping[str, Any]]:
    document = _exact_fields(
        _strict_json_object(payload, label="prepare base-model wrapper"),
        {
            "schema_version",
            "rotation_id",
            "base_folds",
            "training_example_ids_sha256",
            "training_sequence_ids_sha256",
            "training_example_count",
            "training_sequence_count",
            "model_state",
        },
        label="prepare base-model wrapper",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["rotation_id"] != spec.rotation_id
        or type(document["base_folds"]) is not list
        or document["base_folds"] != list(spec.base_folds)
        or any(type(item) is not int for item in document["base_folds"])
    ):
        raise ValueError("prepare base-model wrapper has the wrong rotation")
    _sha256(document["training_example_ids_sha256"], label="base training example IDs")
    _sha256(document["training_sequence_ids_sha256"], label="base training sequence IDs")
    _exact_int(
        document["training_example_count"],
        label="base training example count",
        minimum=1,
    )
    _exact_int(
        document["training_sequence_count"],
        label="base training sequence count",
        minimum=1,
    )
    return descriptor_logistic_state_from_document(document["model_state"]), document


def _decode_transform_wrapper(
    payload: bytes,
    *,
    spec: RotationSpec,
    base_sequence_ids: Sequence[str],
) -> DiversityTransform:
    document = _exact_fields(
        _strict_json_object(payload, label="prepare transform wrapper"),
        {
            "schema_version",
            "rotation_id",
            "base_folds",
            "training_sequence_ids_sha256",
            "transform_state",
        },
        label="prepare transform wrapper",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["rotation_id"] != spec.rotation_id
        or type(document["base_folds"]) is not list
        or document["base_folds"] != list(spec.base_folds)
        or any(type(item) is not int for item in document["base_folds"])
        or _sha256(
            document["training_sequence_ids_sha256"],
            label="transform training sequence IDs",
        )
        != id_stream_sha256(tuple(base_sequence_ids))
    ):
        raise ValueError("prepare transform wrapper differs from its base capability")
    return diversity_transform_from_document(
        document["transform_state"],
        fit_sequence_ids=base_sequence_ids,
    )


def _base_capability_document(prepared: PreparedRotation) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": BASE_UPDATE_CAPABILITY_ARTIFACT,
        "rotation_id": prepared.spec.rotation_id,
        "outer_fold": prepared.spec.outer_fold,
        "acquisition_pool_fold": prepared.spec.pool_fold,
        "base_folds": list(prepared.spec.base_folds),
        "base_example_count": len(prepared.base_example_ids),
        "base_example_ids_sha256": id_stream_sha256(prepared.base_example_ids),
        "base_sequence_count": len(prepared.base_sequence_ids),
        "base_sequence_ids_sha256": id_stream_sha256(prepared.base_sequence_ids),
        "data_payload_paths": list(_BASE_UPDATE_DATA_PAYLOAD_PATHS),
    }


def _prepared_rotation_payload_identity(
    prepared: PreparedRotation,
) -> tuple[tuple[str, bytes], ...]:
    """Return every canonical emitted payload determined by a prepare result.

    Equality of frozen dataclasses is not an exact binary64 comparison because
    IEEE ``-0.0 == 0.0``.  These are the actual canonical bytes emitted across
    the four prepare leaves.  Hex-encoded model/evidence fields therefore keep
    their sign bit, while canonical JSON numeric spellings do the same for the
    prediction-bearing candidate view.
    """

    if type(prepared) is not PreparedRotation:
        raise TypeError("prepare payload identity requires an exact PreparedRotation")
    _require_frozen_rotation(prepared.spec, label="prepared rotation identity")
    if (
        type(prepared.base_model) is not DescriptorLogisticState
        or type(prepared.diversity_transform) is not DiversityTransform
    ):
        raise TypeError("prepared rotation identity contains a spoofed state object")
    exact_tuple_members = (
        (prepared.pool_predictions, PoolPrediction, "pool predictions"),
        (prepared.pool_novelty, NoveltyEvidence, "pool novelty"),
        (prepared.pool_components, PoolComponent, "pool components"),
        (prepared.pool_candidates, PoolCandidate, "pool candidates"),
        (
            prepared.random_pool_candidates,
            RandomPoolCandidate,
            "random pool candidates",
        ),
    )
    for values, expected_type, label in exact_tuple_members:
        if type(values) is not tuple or any(type(item) is not expected_type for item in values):
            raise TypeError(f"prepared rotation {label} must contain exact typed values")
    for values, label in (
        (prepared.base_example_ids, "base example IDs"),
        (prepared.base_sequence_ids, "base sequence IDs"),
        (prepared.base_sequences, "base sequences"),
    ):
        if type(values) is not tuple or any(type(item) is not str for item in values):
            raise TypeError(f"prepared rotation {label} must be an exact string tuple")

    candidate_ids = tuple(item.sequence_id for item in prepared.pool_candidates)
    prediction_payload = canonical_jsonl_bytes(prepared.candidate_documents())
    random_payload = canonical_jsonl_bytes(prepared.random_candidate_documents())
    payloads = {
        "base-update/base-model.json": canonical_json_bytes(prepared.model_document()),
        "base-update/capability.json": canonical_json_bytes(_base_capability_document(prepared)),
        "base-update/diversity-transform.json": canonical_json_bytes(prepared.transform_document()),
        "evidence/pool-components.jsonl": canonical_jsonl_bytes(prepared.component_documents()),
        "evidence/pool-novelty.jsonl": canonical_jsonl_bytes(prepared.novelty_documents()),
        "evidence/pool-predictions.jsonl": canonical_jsonl_bytes(prepared.prediction_documents()),
        "prediction-view/candidates.jsonl": prediction_payload,
        "prediction-view/view-summary.json": canonical_json_bytes(
            _view_summary_document(
                spec=prepared.spec,
                view_kind="prediction",
                artifact=PREDICTION_VIEW_SUMMARY_ARTIFACT,
                candidate_ids=candidate_ids,
                candidate_payload=prediction_payload,
                field_names=_PREDICTION_FIELDS,
            )
        ),
        "random-minimal-view/candidates.jsonl": random_payload,
        "random-minimal-view/view-summary.json": canonical_json_bytes(
            _view_summary_document(
                spec=prepared.spec,
                view_kind="random_minimal",
                artifact=RANDOM_MINIMAL_VIEW_SUMMARY_ARTIFACT,
                candidate_ids=candidate_ids,
                candidate_payload=random_payload,
                field_names=_RANDOM_MINIMAL_FIELDS,
            )
        ),
        "rotation.json": canonical_json_bytes(prepared.spec.document()),
    }
    return tuple(sorted(payloads.items()))


def _view_summary_document(
    *,
    spec: RotationSpec,
    view_kind: str,
    artifact: str,
    candidate_ids: Sequence[str],
    candidate_payload: bytes,
    field_names: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "rotation_id": spec.rotation_id,
        "view_kind": view_kind,
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": ordered_id_stream_sha256(candidate_ids),
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "field_names": list(field_names),
    }


def _prediction_from_document(value: object, *, spec: RotationSpec) -> PoolPrediction:
    document = _exact_fields(
        value,
        {
            "schema_version",
            "rotation_id",
            "sequence_id",
            "target_probabilities_hex",
            "objective_probabilities_hex",
        },
        label="pool prediction",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["rotation_id"] != spec.rotation_id
    ):
        raise ValueError("pool prediction identity differs from its rotation")
    sequence_id = _sha256(document["sequence_id"], label="pool prediction sequence ID")
    targets = _exact_fields(
        document["target_probabilities_hex"],
        set(TARGETS),
        label="pool target probabilities",
    )
    objectives = _exact_fields(
        document["objective_probabilities_hex"],
        set(OBJECTIVES),
        label="pool objective probabilities",
    )
    prediction = PoolPrediction(
        sequence_id=sequence_id,
        target_probabilities=tuple(
            _canonical_hex(targets[target], label=f"pool probability {target}")
            for target in TARGETS
        ),
        objective_probabilities=tuple(
            _canonical_hex(objectives[name], label=f"pool objective {name}") for name in OBJECTIVES
        ),
    )
    if canonical_json_bytes(prediction.document(rotation_id=spec.rotation_id)) != (
        canonical_json_bytes(document)
    ):
        raise ValueError("pool prediction does not reconstruct exact document bytes")
    return prediction


def _novelty_from_document(value: object, *, spec: RotationSpec) -> NoveltyEvidence:
    document = _exact_fields(
        value,
        {
            "schema_version",
            "rotation_id",
            "sequence_id",
            "max_similarity_hex",
            "nearest_training_sequence_id",
            "novelty_hex",
        },
        label="pool novelty evidence",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["rotation_id"] != spec.rotation_id
    ):
        raise ValueError("pool novelty identity differs from its rotation")
    sequence_id = _sha256(document["sequence_id"], label="pool novelty sequence ID")
    nearest = _sha256(
        document["nearest_training_sequence_id"],
        label="pool novelty nearest base sequence ID",
    )
    maximum = _canonical_hex(document["max_similarity_hex"], label="pool maximum similarity")
    novelty = _canonical_hex(document["novelty_hex"], label="pool novelty")
    if not 0.0 <= maximum <= 1.0 or not 0.0 <= novelty <= 1.0:
        raise ValueError("pool novelty values must lie in the closed unit interval")
    evidence = NoveltyEvidence(sequence_id, maximum, nearest, novelty)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "rotation_id": spec.rotation_id,
        "sequence_id": evidence.sequence_id,
        "max_similarity_hex": evidence.max_similarity.hex(),
        "nearest_training_sequence_id": evidence.nearest_training_sequence_id,
        "novelty_hex": evidence.novelty.hex(),
    }
    if canonical_json_bytes(expected) != canonical_json_bytes(document):
        raise ValueError("pool novelty does not reconstruct exact document bytes")
    return evidence


def _component_from_document(value: object, *, spec: RotationSpec) -> PoolComponent:
    document = _exact_fields(
        value,
        {
            "schema_version",
            "rotation_id",
            "role",
            "fold",
            "diversity_component_id",
            "sequence_ids",
        },
        label="pool component",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["rotation_id"] != spec.rotation_id
        or type(document["role"]) is not str
        or document["role"] != "pool"
        or type(document["fold"]) is not int
        or document["fold"] != spec.pool_fold
        or type(document["diversity_component_id"]) is not str
    ):
        raise ValueError("pool component identity differs from its role or rotation")
    component = PoolComponent(
        component_id=document["diversity_component_id"],
        sequence_ids=_string_tuple(document["sequence_ids"], label="pool component members"),
    )
    if canonical_json_bytes(component.document(spec=spec)) != canonical_json_bytes(document):
        raise ValueError("pool component does not reconstruct exact document bytes")
    return component


def _decode_base_update(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    predecessors: Mapping[str, str],
    publication_identity: SequentialV2PublicationIdentity,
) -> BaseUpdateCapability:
    verified = verify_phase_capability(
        seal,
        expected_artifact=BASE_UPDATE_ARTIFACT,
        expected_payload_paths=BASE_UPDATE_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="prepare",
        scope_id=spec.rotation_id,
    )
    if verified.read_payload_bytes("rotation.json") != canonical_json_bytes(spec.document()):
        raise ValueError("base-update rotation document differs from its scope")
    contexts = tuple(
        _context_from_document(row, label=f"base-update context {index}")
        for index, row in enumerate(
            _strict_jsonl(
                verified.read_payload_bytes("base-contexts.jsonl"),
                label="base-update contexts",
            )
        )
    )
    example_ids = tuple(row.example_id for row in contexts)
    if example_ids != tuple(sorted(set(example_ids))):
        raise ValueError("base-update contexts must use ascending unique example IDs")
    base_sequence_ids, _base_sequences = _base_sequence_evidence(contexts)
    model, wrapper = _decode_model_wrapper(
        verified.read_payload_bytes("base-model.json"),
        spec=spec,
    )
    if (
        wrapper["training_example_count"] != len(example_ids)
        or wrapper["training_example_ids_sha256"] != id_stream_sha256(example_ids)
        or wrapper["training_sequence_count"] != len(base_sequence_ids)
        or wrapper["training_sequence_ids_sha256"] != id_stream_sha256(base_sequence_ids)
    ):
        raise ValueError("base-model wrapper census differs from base-update contexts")
    transform = _decode_transform_wrapper(
        verified.read_payload_bytes("diversity-transform.json"),
        spec=spec,
        base_sequence_ids=base_sequence_ids,
    )
    expected_capability = {
        "schema_version": SCHEMA_VERSION,
        "artifact": BASE_UPDATE_CAPABILITY_ARTIFACT,
        "rotation_id": spec.rotation_id,
        "outer_fold": spec.outer_fold,
        "acquisition_pool_fold": spec.pool_fold,
        "base_folds": list(spec.base_folds),
        "base_example_count": len(example_ids),
        "base_example_ids_sha256": id_stream_sha256(example_ids),
        "base_sequence_count": len(base_sequence_ids),
        "base_sequence_ids_sha256": id_stream_sha256(base_sequence_ids),
        "data_payload_paths": list(_BASE_UPDATE_DATA_PAYLOAD_PATHS),
    }
    actual_capability = _strict_json_object(
        verified.read_payload_bytes("capability.json"),
        label="base-update capability",
    )
    if canonical_json_bytes(actual_capability) != canonical_json_bytes(expected_capability):
        raise ValueError("base-update capability differs from authenticated payloads")
    return BaseUpdateCapability(
        spec=spec,
        contexts=contexts,
        allowed_example_ids=example_ids,
        base_model=model,
        diversity_transform=transform,
        base_update_seal_sha256=verified.seal_sha256,
    )


def _decode_view(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    predecessors: Mapping[str, str],
    publication_identity: SequentialV2PublicationIdentity,
    prediction: bool,
) -> tuple[tuple[PoolCandidate, ...] | tuple[RandomPoolCandidate, ...], Mapping[str, Any]]:
    artifact = PREDICTION_VIEW_ARTIFACT if prediction else RANDOM_MINIMAL_VIEW_ARTIFACT
    payload_paths = (
        PREDICTION_VIEW_PAYLOAD_PATHS if prediction else RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=artifact,
        expected_payload_paths=payload_paths,
        expected_predecessor_seals=predecessors,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="prepare",
        scope_id=spec.rotation_id,
    )
    candidate_payload = verified.read_payload_bytes("candidates.jsonl")
    rows = _strict_jsonl(
        candidate_payload,
        label="prediction pool view" if prediction else "random-minimal pool view",
    )
    candidates = (
        tuple(PoolCandidate.from_mapping(row) for row in rows)
        if prediction
        else tuple(RandomPoolCandidate.from_mapping(row) for row in rows)
    )
    candidate_ids = tuple(item.sequence_id for item in candidates)
    if (
        len(candidates) != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]
        or candidate_ids != tuple(sorted(set(candidate_ids)))
        or any(
            item.rotation_id != spec.rotation_id or item.eligible is not True for item in candidates
        )
    ):
        raise ValueError("prepare candidate view differs from exact fold support")
    expected_summary = _view_summary_document(
        spec=spec,
        view_kind="prediction" if prediction else "random_minimal",
        artifact=(
            PREDICTION_VIEW_SUMMARY_ARTIFACT if prediction else RANDOM_MINIMAL_VIEW_SUMMARY_ARTIFACT
        ),
        candidate_ids=candidate_ids,
        candidate_payload=candidate_payload,
        field_names=_PREDICTION_FIELDS if prediction else _RANDOM_MINIMAL_FIELDS,
    )
    summary = _strict_json_object(
        verified.read_payload_bytes("view-summary.json"),
        label="prepare view summary",
    )
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise ValueError("prepare view summary differs from authenticated candidates")
    return candidates, summary


def _evidence_summary_document(
    prepared: PreparedRotation,
    *,
    base_update: PhaseSeal,
    prediction_view: PhaseSeal,
    random_minimal_view: PhaseSeal,
) -> dict[str, object]:
    candidate_ids = tuple(item.sequence_id for item in prepared.pool_candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": PREPARE_EVIDENCE_SUMMARY_ARTIFACT,
        "rotation_id": prepared.spec.rotation_id,
        "outer_fold": prepared.spec.outer_fold,
        "acquisition_pool_fold": prepared.spec.pool_fold,
        "base_folds": list(prepared.spec.base_folds),
        "base_example_count": len(prepared.base_example_ids),
        "base_example_ids_sha256": id_stream_sha256(prepared.base_example_ids),
        "base_sequence_count": len(prepared.base_sequence_ids),
        "base_sequence_ids_sha256": id_stream_sha256(prepared.base_sequence_ids),
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": ordered_id_stream_sha256(candidate_ids),
        "pool_prediction_count": len(prepared.pool_predictions),
        "pool_novelty_count": len(prepared.pool_novelty),
        "pool_component_count": len(prepared.pool_components),
        "base_update_seal_sha256": base_update.seal_sha256,
        "base_update_model_payload_sha256": dict(base_update.payload_sha256)["base-model.json"],
        "base_update_transform_payload_sha256": dict(base_update.payload_sha256)[
            "diversity-transform.json"
        ],
        "prediction_view_seal_sha256": prediction_view.seal_sha256,
        "prediction_view_payload_sha256": dict(prediction_view.payload_sha256)["candidates.jsonl"],
        "random_minimal_view_seal_sha256": random_minimal_view.seal_sha256,
        "random_minimal_view_payload_sha256": dict(random_minimal_view.payload_sha256)[
            "candidates.jsonl"
        ],
    }


def _leaf_attestation(
    capabilities: PrepareArtifactCapabilities,
    *,
    role: str,
    base_model_state_sha256: str,
) -> PrepareLeafAttestation:
    seal = capabilities.seal_for_role(role)
    if role == BASE_UPDATE_ROLE:
        model_state_digest: str | None = _sha256(
            base_model_state_sha256,
            label="prepare base model-state digest",
        )
        candidate_count: int | None = None
        candidate_ids_sha256: str | None = None
    else:
        model_state_digest = None
        candidate_count = capabilities.view_provenance.candidate_count
        candidate_ids_sha256 = capabilities.view_provenance.candidate_ids_sha256
    return PrepareLeafAttestation(
        role=role,
        relative_path=_prepare_relative_path(capabilities.spec, role),
        artifact=_ARTIFACT_BY_ROLE[role],
        phase_seal_sha256=seal.seal_sha256,
        payload_sha256=seal.payload_sha256,
        base_model_state_sha256=model_state_digest,
        candidate_count=candidate_count,
        candidate_ids_sha256=candidate_ids_sha256,
    )


def _rotation_attestation(
    decoded: DecodedPrepareArtifacts,
    *,
    publication_identity: SequentialV2PublicationIdentity,
) -> PrepareRotationAttestation:
    capabilities = decoded.capabilities
    base_model_state_sha256 = sha256_bytes(
        canonical_json_bytes(decoded.base_update.base_model.document())
    )
    return PrepareRotationAttestation(
        spec=capabilities.spec,
        publication_identity=publication_identity,
        protocol_seal_sha256=capabilities.protocol_seal_sha256,
        stage_global_seal_sha256=capabilities.stage_global_seal_sha256,
        source_prepare_leaf_seal_sha256=(capabilities.source_prepare_leaf_seal_sha256),
        base_context_association_count=len(decoded.base_update.contexts),
        leaves=tuple(
            _leaf_attestation(
                capabilities,
                role=role,
                base_model_state_sha256=base_model_state_sha256,
            )
            for role in ROLE_ORDER
        ),
        view_provenance=capabilities.view_provenance,
    )


def publish_prepared_rotation_artifacts(
    destination: str | Path,
    *,
    prepared: PreparedRotation,
    capability: PrepareCapability,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    source_prepare_leaf_seal: PhaseSeal,
) -> PrepareRotationAttestation:
    """Publish four leaves and return only their payload-free worker attestation."""

    if type(prepared) is not PreparedRotation:
        raise TypeError("prepare artifact publisher requires a PreparedRotation")
    _require_frozen_rotation(prepared.spec, label="prepare artifact publisher rotation")
    if type(capability) is not PrepareCapability:
        raise TypeError("prepare artifact publisher requires the original PrepareCapability")
    _require_exact_prepare_capability(capability)
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("prepare artifact publisher requires a publication identity")
    authenticated_source = _authenticate_prepare_inputs(
        spec=prepared.spec,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_global_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        source_prepare_leaf_seal=source_prepare_leaf_seal,
    )
    if authenticated_source != capability:
        raise ValueError("supplied prepare capability differs from its authenticated source leaf")
    if _prepared_rotation_payload_identity(
        prepare_rotation(capability)
    ) != _prepared_rotation_payload_identity(prepared):
        raise ValueError("prepared result differs from the exact source capability computation")

    root = Path(destination)
    if not root.is_dir():
        raise ValueError("prepare rotation destination must be an existing directory")
    spec = prepared.spec
    common = _common_predecessors(
        spec,
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        stage_global_seal_sha256=stage_global_seal.seal_sha256,
        source_prepare_leaf_seal_sha256=source_prepare_leaf_seal.seal_sha256,
    )
    metadata = publication_identity.metadata(phase="prepare", scope_id=spec.rotation_id)
    rotation_payload = canonical_json_bytes(spec.document())
    base_update = publish_phase(
        root / _PATH_SUFFIX_BY_ROLE[BASE_UPDATE_ROLE],
        artifact=BASE_UPDATE_ARTIFACT,
        payloads={
            "base-contexts.jsonl": canonical_jsonl_bytes(
                _context_document(row) for row in capability.base_contexts
            ),
            "base-model.json": canonical_json_bytes(prepared.model_document()),
            "capability.json": canonical_json_bytes(_base_capability_document(prepared)),
            "diversity-transform.json": canonical_json_bytes(prepared.transform_document()),
            "rotation.json": rotation_payload,
        },
        predecessor_seals=common,
        metadata=metadata,
    )

    candidate_ids = tuple(item.sequence_id for item in prepared.pool_candidates)
    prediction_payload = canonical_jsonl_bytes(prepared.candidate_documents())
    prediction_view = publish_phase(
        root / _PATH_SUFFIX_BY_ROLE[PREDICTION_VIEW_ROLE],
        artifact=PREDICTION_VIEW_ARTIFACT,
        payloads={
            "candidates.jsonl": prediction_payload,
            "view-summary.json": canonical_json_bytes(
                _view_summary_document(
                    spec=spec,
                    view_kind="prediction",
                    artifact=PREDICTION_VIEW_SUMMARY_ARTIFACT,
                    candidate_ids=candidate_ids,
                    candidate_payload=prediction_payload,
                    field_names=_PREDICTION_FIELDS,
                )
            ),
        },
        predecessor_seals=common,
        metadata=metadata,
    )
    random_payload = canonical_jsonl_bytes(prepared.random_candidate_documents())
    random_minimal_view = publish_phase(
        root / _PATH_SUFFIX_BY_ROLE[RANDOM_MINIMAL_VIEW_ROLE],
        artifact=RANDOM_MINIMAL_VIEW_ARTIFACT,
        payloads={
            "candidates.jsonl": random_payload,
            "view-summary.json": canonical_json_bytes(
                _view_summary_document(
                    spec=spec,
                    view_kind="random_minimal",
                    artifact=RANDOM_MINIMAL_VIEW_SUMMARY_ARTIFACT,
                    candidate_ids=candidate_ids,
                    candidate_payload=random_payload,
                    field_names=_RANDOM_MINIMAL_FIELDS,
                )
            ),
        },
        predecessor_seals=common,
        metadata=metadata,
    )
    if prediction_view.seal_sha256 == random_minimal_view.seal_sha256:
        raise RuntimeError("prediction and random-minimal view seals must be distinct")

    provenance = RotationViewProvenance(
        spec=spec,
        candidate_count=len(candidate_ids),
        candidate_ids_sha256=ordered_id_stream_sha256(candidate_ids),
        prediction_view_seal_sha256=prediction_view.seal_sha256,
        prediction_view_payload_sha256=dict(prediction_view.payload_sha256)["candidates.jsonl"],
        random_view_seal_sha256=random_minimal_view.seal_sha256,
        random_view_payload_sha256=dict(random_minimal_view.payload_sha256)["candidates.jsonl"],
    )
    # Evidence is intentionally last: it binds the other three leaves directly.
    provisional = PrepareArtifactCapabilities(
        spec=spec,
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        stage_global_seal_sha256=stage_global_seal.seal_sha256,
        source_prepare_leaf_seal_sha256=source_prepare_leaf_seal.seal_sha256,
        evidence=base_update,
        base_update=base_update,
        prediction_view=prediction_view,
        random_minimal_view=random_minimal_view,
        view_provenance=provenance,
    )
    evidence = publish_phase(
        root / _PATH_SUFFIX_BY_ROLE[EVIDENCE_ROLE],
        artifact=PREPARE_EVIDENCE_ARTIFACT,
        payloads={
            "pool-components.jsonl": canonical_jsonl_bytes(prepared.component_documents()),
            "pool-novelty.jsonl": canonical_jsonl_bytes(prepared.novelty_documents()),
            "pool-predictions.jsonl": canonical_jsonl_bytes(prepared.prediction_documents()),
            "prepare-summary.json": canonical_json_bytes(
                _evidence_summary_document(
                    prepared,
                    base_update=base_update,
                    prediction_view=prediction_view,
                    random_minimal_view=random_minimal_view,
                )
            ),
            "rotation.json": rotation_payload,
        },
        predecessor_seals=_evidence_predecessors(provisional),
        metadata=metadata,
    )
    result = PrepareArtifactCapabilities(
        spec=spec,
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        stage_global_seal_sha256=stage_global_seal.seal_sha256,
        source_prepare_leaf_seal_sha256=source_prepare_leaf_seal.seal_sha256,
        evidence=evidence,
        base_update=base_update,
        prediction_view=prediction_view,
        random_minimal_view=random_minimal_view,
        view_provenance=provenance,
    )
    attestation = verify_prepared_rotation_artifacts(
        result,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_global_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        source_prepare_leaf_seal=source_prepare_leaf_seal,
    )
    if attestation.spec != prepared.spec:
        raise RuntimeError("published prepare attestation changed rotation identity")
    return attestation


def _decode_prepared_rotation_artifacts(
    capabilities: PrepareArtifactCapabilities,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    source_prepare_leaf_seal: PhaseSeal,
) -> DecodedPrepareArtifacts:
    """Privately reconstruct one rotation while its payloads remain isolated."""

    if type(capabilities) is not PrepareArtifactCapabilities:
        raise TypeError("prepare verification requires PrepareArtifactCapabilities")
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("prepare verifier requires a publication identity")
    spec = capabilities.spec
    source = _authenticate_prepare_inputs(
        spec=spec,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_global_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        source_prepare_leaf_seal=source_prepare_leaf_seal,
    )
    if (
        capabilities.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
        or capabilities.stage_global_seal_sha256 != stage_global_seal.seal_sha256
        or capabilities.stage_global_seal_sha256 != expected_stage_global_seal_sha256
        or capabilities.source_prepare_leaf_seal_sha256 != source_prepare_leaf_seal.seal_sha256
    ):
        raise ValueError("prepare capability predecessors differ from expected rootless seals")
    common = _common_predecessors(
        spec,
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        stage_global_seal_sha256=stage_global_seal.seal_sha256,
        source_prepare_leaf_seal_sha256=source_prepare_leaf_seal.seal_sha256,
    )
    base_update = _decode_base_update(
        capabilities.base_update,
        spec=spec,
        predecessors=common,
        publication_identity=publication_identity,
    )
    if _base_context_payload_identity(
        base_update.contexts,
        label="base-update contexts",
    ) != _base_context_payload_identity(
        source.base_contexts,
        label="authenticated source base contexts",
    ):
        raise ValueError("base-update contexts differ from the authenticated source leaf")
    prediction_raw, _prediction_summary = _decode_view(
        capabilities.prediction_view,
        spec=spec,
        predecessors=common,
        publication_identity=publication_identity,
        prediction=True,
    )
    random_raw, _random_summary = _decode_view(
        capabilities.random_minimal_view,
        spec=spec,
        predecessors=common,
        publication_identity=publication_identity,
        prediction=False,
    )
    pool_candidates = tuple(item for item in prediction_raw if type(item) is PoolCandidate)
    random_candidates = tuple(item for item in random_raw if type(item) is RandomPoolCandidate)
    if len(pool_candidates) != len(prediction_raw) or len(random_candidates) != len(random_raw):
        raise AssertionError("prepare view decoder returned a mixed candidate type")
    candidate_ids = tuple(item.sequence_id for item in pool_candidates)
    if tuple(item.sequence_id for item in random_candidates) != candidate_ids:
        raise ValueError("prediction and random-minimal candidate ID streams differ")
    if capabilities.prediction_view.seal_sha256 == capabilities.random_minimal_view.seal_sha256:
        raise ValueError("prediction and random-minimal view phase seals must be distinct")

    evidence = verify_phase_capability(
        capabilities.evidence,
        expected_artifact=PREPARE_EVIDENCE_ARTIFACT,
        expected_payload_paths=EVIDENCE_PAYLOAD_PATHS,
        expected_predecessor_seals=_evidence_predecessors(capabilities),
    )
    publication_identity.verify_metadata(
        evidence.metadata_json,
        phase="prepare",
        scope_id=spec.rotation_id,
    )
    if evidence.read_payload_bytes("rotation.json") != canonical_json_bytes(spec.document()):
        raise ValueError("prepare evidence rotation document differs from its scope")
    predictions = tuple(
        _prediction_from_document(row, spec=spec)
        for row in _strict_jsonl(
            evidence.read_payload_bytes("pool-predictions.jsonl"),
            label="pool prediction evidence",
        )
    )
    novelty = tuple(
        _novelty_from_document(row, spec=spec)
        for row in _strict_jsonl(
            evidence.read_payload_bytes("pool-novelty.jsonl"),
            label="pool novelty evidence",
        )
    )
    components = tuple(
        _component_from_document(row, spec=spec)
        for row in _strict_jsonl(
            evidence.read_payload_bytes("pool-components.jsonl"),
            label="pool component evidence",
        )
    )
    base_sequence_ids, base_sequences = _base_sequence_evidence(base_update.contexts)
    prepared = PreparedRotation(
        spec=spec,
        base_model=base_update.base_model,
        diversity_transform=base_update.diversity_transform,
        base_example_ids=base_update.allowed_example_ids,
        base_sequence_ids=base_sequence_ids,
        base_sequences=base_sequences,
        pool_predictions=predictions,
        pool_novelty=novelty,
        pool_components=components,
        pool_candidates=pool_candidates,
        random_pool_candidates=random_candidates,
    )
    if _prepared_rotation_payload_identity(prepared) != _prepared_rotation_payload_identity(
        prepare_rotation(source)
    ):
        raise ValueError("prepare leaves differ from the authenticated source computation")
    expected_summary = _evidence_summary_document(
        prepared,
        base_update=capabilities.base_update,
        prediction_view=capabilities.prediction_view,
        random_minimal_view=capabilities.random_minimal_view,
    )
    actual_summary = _strict_json_object(
        evidence.read_payload_bytes("prepare-summary.json"),
        label="prepare evidence summary",
    )
    if canonical_json_bytes(actual_summary) != canonical_json_bytes(expected_summary):
        raise ValueError("prepare evidence summary differs from reconstructed artifacts")
    expected_provenance = RotationViewProvenance(
        spec=spec,
        candidate_count=len(candidate_ids),
        candidate_ids_sha256=ordered_id_stream_sha256(candidate_ids),
        prediction_view_seal_sha256=capabilities.prediction_view.seal_sha256,
        prediction_view_payload_sha256=dict(capabilities.prediction_view.payload_sha256)[
            "candidates.jsonl"
        ],
        random_view_seal_sha256=capabilities.random_minimal_view.seal_sha256,
        random_view_payload_sha256=dict(capabilities.random_minimal_view.payload_sha256)[
            "candidates.jsonl"
        ],
    )
    if capabilities.view_provenance != expected_provenance:
        raise ValueError("prepare view provenance differs from authenticated candidate views")
    return DecodedPrepareArtifacts(capabilities, prepared, base_update)


def verify_prepared_rotation_artifacts(
    capabilities: PrepareArtifactCapabilities,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    source_prepare_leaf_seal: PhaseSeal,
) -> PrepareRotationAttestation:
    """Verify one isolated rotation and release only a payload-free attestation.

    The caller executing this function is the rotation worker and may hold its
    one label-bearing source leaf and four output leaves.  Its supervised
    fresh-exec result channel must transmit only the returned attestation.
    """

    decoded = _decode_prepared_rotation_artifacts(
        capabilities,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_global_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        source_prepare_leaf_seal=source_prepare_leaf_seal,
    )
    return _rotation_attestation(decoded, publication_identity=publication_identity)


def _prepare_index_document(
    item: PrepareRotationAttestation,
    *,
    role: str,
) -> dict[str, object]:
    return {"rotation_id": item.spec.rotation_id, **item.leaf_for_role(role).document()}


def _campaign_predecessors(
    attestations: Sequence[PrepareRotationAttestation],
    *,
    protocol_seal_sha256: str,
) -> dict[str, str]:
    result = {
        "protocol/SHA256SUMS": _sha256(
            protocol_seal_sha256,
            label="prepare campaign protocol seal",
        )
    }
    for item in attestations:
        for role in ROLE_ORDER:
            leaf = item.leaf_for_role(role)
            result[f"{leaf.relative_path}/SHA256SUMS"] = leaf.phase_seal_sha256
    return result


def _campaign_payloads(
    attestations: Sequence[PrepareRotationAttestation],
) -> dict[str, bytes]:
    index_payload = canonical_jsonl_bytes(
        _prepare_index_document(item, role=role) for item in attestations for role in ROLE_ORDER
    )
    logical_candidates = sum(item.view_provenance.candidate_count for item in attestations)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PREPARE_CAMPAIGN_ARTIFACT,
        "rotation_count": len(attestations),
        "prepare_leaf_count": len(attestations) * len(ROLE_ORDER),
        "evidence_leaf_count": len(attestations),
        "base_update_leaf_count": len(attestations),
        "prediction_view_leaf_count": len(attestations),
        "random_minimal_view_leaf_count": len(attestations),
        "base_model_count": len(attestations),
        "diversity_transform_count": len(attestations),
        "base_context_association_count": sum(
            item.base_context_association_count for item in attestations
        ),
        "logical_candidate_count": logical_candidates,
        "prediction_view_row_count": logical_candidates,
        "random_minimal_view_row_count": logical_candidates,
        "physical_view_row_count": 2 * logical_candidates,
        "prepare_index_sha256": sha256_bytes(index_payload),
    }
    expected = {
        "rotation_count": EXPECTED_ROTATIONS,
        "prepare_leaf_count": EXPECTED_ROTATIONS * len(ROLE_ORDER),
        "evidence_leaf_count": EXPECTED_ROTATIONS,
        "base_update_leaf_count": EXPECTED_ROTATIONS,
        "prediction_view_leaf_count": EXPECTED_ROTATIONS,
        "random_minimal_view_leaf_count": EXPECTED_ROTATIONS,
        "base_model_count": EXPECTED_ROTATIONS,
        "diversity_transform_count": EXPECTED_ROTATIONS,
        "base_context_association_count": 29_904,
        "logical_candidate_count": EXPECTED_POOL_CANDIDATES,
        "prediction_view_row_count": EXPECTED_PREDICTION_POOL_VIEW_ROWS,
        "random_minimal_view_row_count": EXPECTED_RANDOM_POOL_VIEW_ROWS,
        "physical_view_row_count": EXPECTED_PHYSICAL_POOL_VIEW_ROWS,
    }
    for name, value in expected.items():
        if summary[name] != value:
            raise ValueError(f"prepare campaign census changed for {name}")
    return {
        "prepare-index.jsonl": index_payload,
        "prepare-summary.json": canonical_json_bytes(summary),
    }


def _decode_safe_campaign(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any], str]:
    """Verify the label-free campaign closure without receiving any leaf bytes."""

    verified = verify_phase_capability(
        seal,
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="prepare",
        scope_id="global",
    )
    index_payload = verified.read_payload_bytes("prepare-index.jsonl")
    rows = _strict_jsonl(index_payload, label="prepare campaign index")
    expected_positions = tuple((spec, role) for spec in ordered_rotations() for role in ROLE_ORDER)
    if len(rows) != len(expected_positions):
        raise ValueError("prepare campaign index must contain exactly eighty rows")
    normalized: list[Mapping[str, Any]] = []
    candidate_digests: dict[str, str] = {}
    for index, (raw, (spec, role)) in enumerate(zip(rows, expected_positions, strict=True)):
        row = _exact_fields(
            raw,
            {
                "schema_version",
                "rotation_id",
                "leaf_role",
                "relative_path",
                "artifact",
                "phase_seal_sha256",
                "payload_sha256",
                "base_model_state_sha256",
                "candidate_count",
                "candidate_ids_sha256",
            },
            label=f"prepare campaign index row {index}",
        )
        payloads = _exact_fields(
            row["payload_sha256"],
            set(_PAYLOAD_PATHS_BY_ROLE[role]),
            label=f"prepare campaign payload map {index}",
        )
        if any(
            type(path) is not str
            or _sha256(digest, label=f"prepare campaign payload {path}") != digest
            for path, digest in payloads.items()
        ):
            raise ValueError("prepare campaign payload map contains an invalid binding")
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != SCHEMA_VERSION
            or type(row["rotation_id"]) is not str
            or row["rotation_id"] != spec.rotation_id
            or type(row["leaf_role"]) is not str
            or row["leaf_role"] != _PATH_SUFFIX_BY_ROLE[role]
            or type(row["relative_path"]) is not str
            or row["relative_path"] != _prepare_relative_path(spec, role)
            or type(row["artifact"]) is not str
            or row["artifact"] != _ARTIFACT_BY_ROLE[role]
        ):
            raise ValueError("prepare campaign index identity or order changed")
        _sha256(row["phase_seal_sha256"], label="prepare campaign leaf seal")
        if role == BASE_UPDATE_ROLE:
            if row["candidate_count"] is not None or row["candidate_ids_sha256"] is not None:
                raise ValueError("base-update index row must not claim a candidate view")
            _sha256(
                row["base_model_state_sha256"],
                label="prepare campaign base model-state digest",
            )
        else:
            if row["base_model_state_sha256"] is not None:
                raise ValueError("non-base prepare index row cannot claim a base model state")
            if (
                type(row["candidate_count"]) is not int
                or row["candidate_count"] != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]
            ):
                raise ValueError("prepare campaign candidate census differs from fold support")
            digest = _sha256(
                row["candidate_ids_sha256"],
                label="prepare campaign candidate IDs",
            )
            prior = candidate_digests.setdefault(spec.rotation_id, digest)
            if prior != digest:
                raise ValueError("prepare campaign rotation candidate-ID digests disagree")
        normalized.append(row)

    predecessors = dict(verified.predecessor_seals)
    protocol_key = "protocol/SHA256SUMS"
    if protocol_key not in predecessors:
        raise ValueError("prepare campaign lacks its exact protocol predecessor")
    protocol = _sha256(predecessors[protocol_key], label="prepare campaign protocol seal")
    expected_predecessors = {protocol_key: protocol}
    for row in normalized:
        expected_predecessors[f"{row['relative_path']}/SHA256SUMS"] = row["phase_seal_sha256"]
    if tuple(sorted(expected_predecessors.items())) != verified.predecessor_seals:
        raise ValueError("prepare campaign index and predecessor closure disagree")

    logical = sum(
        row["candidate_count"]
        for row in normalized
        if row["leaf_role"] == _PATH_SUFFIX_BY_ROLE[EVIDENCE_ROLE]
    )
    expected_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PREPARE_CAMPAIGN_ARTIFACT,
        "rotation_count": EXPECTED_ROTATIONS,
        "prepare_leaf_count": EXPECTED_ROTATIONS * len(ROLE_ORDER),
        "evidence_leaf_count": EXPECTED_ROTATIONS,
        "base_update_leaf_count": EXPECTED_ROTATIONS,
        "prediction_view_leaf_count": EXPECTED_ROTATIONS,
        "random_minimal_view_leaf_count": EXPECTED_ROTATIONS,
        "base_model_count": EXPECTED_ROTATIONS,
        "diversity_transform_count": EXPECTED_ROTATIONS,
        "base_context_association_count": 29_904,
        "logical_candidate_count": logical,
        "prediction_view_row_count": sum(
            row["candidate_count"]
            for row in normalized
            if row["leaf_role"] == _PATH_SUFFIX_BY_ROLE[PREDICTION_VIEW_ROLE]
        ),
        "random_minimal_view_row_count": sum(
            row["candidate_count"]
            for row in normalized
            if row["leaf_role"] == _PATH_SUFFIX_BY_ROLE[RANDOM_MINIMAL_VIEW_ROLE]
        ),
        "physical_view_row_count": sum(
            row["candidate_count"]
            for row in normalized
            if row["leaf_role"]
            in {
                _PATH_SUFFIX_BY_ROLE[PREDICTION_VIEW_ROLE],
                _PATH_SUFFIX_BY_ROLE[RANDOM_MINIMAL_VIEW_ROLE],
            }
        ),
        "prepare_index_sha256": sha256_bytes(index_payload),
    }
    summary = _strict_json_object(
        verified.read_payload_bytes("prepare-summary.json"),
        label="prepare campaign summary",
    )
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise ValueError("prepare campaign summary differs from its exact index census")
    if (
        logical != EXPECTED_POOL_CANDIDATES
        or expected_summary["prediction_view_row_count"] != EXPECTED_PREDICTION_POOL_VIEW_ROWS
        or expected_summary["random_minimal_view_row_count"] != EXPECTED_RANDOM_POOL_VIEW_ROWS
        or expected_summary["physical_view_row_count"] != EXPECTED_PHYSICAL_POOL_VIEW_ROWS
    ):
        raise ValueError("prepare campaign global candidate census changed")
    return tuple(normalized), summary, protocol


def _campaign_index_row(
    capability: PrepareCampaignCapability,
    *,
    spec: RotationSpec,
    role: str,
) -> Mapping[str, Any]:
    if type(capability) is not PrepareCampaignCapability:
        raise TypeError("prepare index lookup requires a campaign capability")
    _require_frozen_rotation(spec, label="prepare index lookup rotation")
    if role not in ROLE_ORDER:
        raise ValueError("prepare index lookup role is invalid")
    rows, _summary, _protocol = _decode_safe_campaign(
        capability.seal,
        publication_identity=capability.publication_identity,
    )
    matches = tuple(
        row
        for row in rows
        if row["rotation_id"] == spec.rotation_id and row["leaf_role"] == _PATH_SUFFIX_BY_ROLE[role]
    )
    if len(matches) != 1:
        raise ValueError("prepare campaign index lacks one exact requested leaf")
    return matches[0]


def verify_prepare_campaign_capability(
    capability: PrepareCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    expected_campaign_seal_sha256: str,
    expected_protocol_seal_sha256: str,
) -> PrepareCampaignCapability:
    """Verify a stripped downstream campaign capability against external authority."""

    if type(capability) is not PrepareCampaignCapability:
        raise TypeError("prepare campaign verification requires a campaign capability")
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("prepare campaign verification requires a publication identity")
    if capability.publication_identity != publication_identity:
        raise ValueError("prepare campaign capability has the wrong publication identity")
    verify_phase_capability(
        capability.seal,
        expected_seal_sha256=_sha256(
            expected_campaign_seal_sha256,
            label="expected prepare campaign seal",
        ),
    )
    _rows, _summary, protocol = _decode_safe_campaign(
        capability.seal,
        publication_identity=publication_identity,
    )
    if protocol != _sha256(
        expected_protocol_seal_sha256,
        label="expected prepare campaign protocol seal",
    ):
        raise ValueError("prepare campaign binds the wrong protocol seal")
    return capability


def _worker_leaf_predecessors(
    campaign: PrepareCampaignCapability,
    *,
    spec: RotationSpec,
    seal: PhaseSeal,
) -> Mapping[str, str]:
    verified = verify_phase_capability(seal)
    expected_keys = {
        "protocol/SHA256SUMS",
        "stage/global/SHA256SUMS",
        f"stage/rotations/{spec.rotation_id}/prepare-capability/SHA256SUMS",
    }
    predecessors = dict(verified.predecessor_seals)
    if set(predecessors) != expected_keys:
        raise ValueError("prepare worker leaf has the wrong predecessor inventory")
    if predecessors["protocol/SHA256SUMS"] != campaign.protocol_seal_sha256:
        raise ValueError("prepare worker leaf binds the wrong protocol")
    return predecessors


def _authorize_worker_campaign(
    campaign: PrepareCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_campaign_seal_sha256: str,
) -> PrepareCampaignCapability:
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("prepare leaf consumer requires an authenticated ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    return verify_prepare_campaign_capability(
        campaign,
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=expected_campaign_seal_sha256,
        expected_protocol_seal_sha256=protocol.seal.seal_sha256,
    )


def prediction_view_from_campaign(
    campaign: PrepareCampaignCapability,
    *,
    spec: RotationSpec,
    prediction_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_campaign_seal_sha256: str,
) -> tuple[PoolCandidate, ...]:
    """Verify and decode exactly one prediction-view leaf for a selector."""

    authorized = _authorize_worker_campaign(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_campaign_seal_sha256,
    )
    expected = authorized.leaf_seal_sha256(spec=spec, role=PREDICTION_VIEW_ROLE)
    verify_phase_capability(prediction_view_seal, expected_seal_sha256=expected)
    candidates, _summary = _decode_view(
        prediction_view_seal,
        spec=spec,
        predecessors=_worker_leaf_predecessors(
            authorized,
            spec=spec,
            seal=prediction_view_seal,
        ),
        publication_identity=publication_identity,
        prediction=True,
    )
    if not all(type(item) is PoolCandidate for item in candidates):
        raise AssertionError("prediction-view decoder returned the wrong candidate type")
    return tuple(item for item in candidates if type(item) is PoolCandidate)


def random_minimal_view_from_campaign(
    campaign: PrepareCampaignCapability,
    *,
    spec: RotationSpec,
    random_minimal_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_campaign_seal_sha256: str,
) -> tuple[RandomPoolCandidate, ...]:
    """Verify and decode exactly one minimal leaf for a random/ceiling selector."""

    authorized = _authorize_worker_campaign(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_campaign_seal_sha256,
    )
    expected = authorized.leaf_seal_sha256(spec=spec, role=RANDOM_MINIMAL_VIEW_ROLE)
    verify_phase_capability(random_minimal_view_seal, expected_seal_sha256=expected)
    candidates, _summary = _decode_view(
        random_minimal_view_seal,
        spec=spec,
        predecessors=_worker_leaf_predecessors(
            authorized,
            spec=spec,
            seal=random_minimal_view_seal,
        ),
        publication_identity=publication_identity,
        prediction=False,
    )
    if not all(type(item) is RandomPoolCandidate for item in candidates):
        raise AssertionError("random-minimal decoder returned the wrong candidate type")
    return tuple(item for item in candidates if type(item) is RandomPoolCandidate)


def base_update_from_campaign(
    campaign: PrepareCampaignCapability,
    *,
    spec: RotationSpec,
    base_update_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_campaign_seal_sha256: str,
) -> BaseUpdateCapability:
    """Verify and decode exactly one base-state leaf for a future update worker."""

    authorized = _authorize_worker_campaign(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_campaign_seal_sha256,
    )
    expected = authorized.leaf_seal_sha256(spec=spec, role=BASE_UPDATE_ROLE)
    verify_phase_capability(base_update_seal, expected_seal_sha256=expected)
    return _decode_base_update(
        base_update_seal,
        spec=spec,
        predecessors=_worker_leaf_predecessors(
            authorized,
            spec=spec,
            seal=base_update_seal,
        ),
        publication_identity=publication_identity,
    )


def _validate_campaign_attestations(
    attestations: tuple[PrepareRotationAttestation, ...],
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
) -> tuple[tuple[PrepareRotationAttestation, ...], ProtocolCapability]:
    """Validate digest-only worker results without opening any prepare leaf."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("prepare campaign requires an exact publication identity")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("prepare campaign requires an authenticated ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    if type(attestations) is not tuple:
        raise TypeError("prepare campaign attestations must be one exact immutable tuple")
    items = attestations
    specs = ordered_rotations()
    if (
        len(items) != EXPECTED_ROTATIONS
        or any(type(item) is not PrepareRotationAttestation for item in items)
        or tuple(item.spec for item in items) != specs
    ):
        raise ValueError("prepare campaign must contain twenty exact attestations in order")
    for item in items:
        if item.publication_identity != publication_identity:
            raise ValueError("prepare attestation has the wrong publication identity")
        if item.protocol_seal_sha256 != protocol.seal.seal_sha256:
            raise ValueError("prepare attestation binds the wrong authenticated protocol")
        if prepare_rotation_attestation_from_document(item.document()) != item:
            raise ValueError("prepare attestation changed during strict reconstruction")
    if len({item.stage_global_seal_sha256 for item in items}) != 1:
        raise ValueError("prepare attestations do not share one anchored stage-global seal")
    if len({item.source_prepare_leaf_seal_sha256 for item in items}) != EXPECTED_ROTATIONS:
        raise ValueError("prepare attestations do not bind twenty distinct source leaves")
    leaf_seals = tuple(
        item.leaf_for_role(role).phase_seal_sha256 for item in items for role in ROLE_ORDER
    )
    if len(set(leaf_seals)) != EXPECTED_ROTATIONS * len(ROLE_ORDER):
        raise ValueError("prepare attestations do not bind eighty distinct output leaves")
    return items, protocol


def publish_prepare_campaign_barrier(
    destination: str | Path,
    *,
    attestations: tuple[PrepareRotationAttestation, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
) -> PrepareCampaignCapability:
    """Publish the 81-predecessor marker from twenty safe worker results.

    Semantic leaf verification happened inside the isolated rotation workers.
    This outcome-free process accepts only their supervisor-authorized,
    payload-free attestations and the authenticated protocol capability; it
    never receives a prepare/source leaf capability.
    """

    items, protocol = _validate_campaign_attestations(
        attestations,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
    )
    payloads = _campaign_payloads(items)
    seal = publish_phase(
        destination,
        artifact=PREPARE_CAMPAIGN_ARTIFACT,
        payloads=payloads,
        predecessor_seals=_campaign_predecessors(
            items,
            protocol_seal_sha256=protocol.seal.seal_sha256,
        ),
        metadata=publication_identity.metadata(phase="prepare", scope_id="global"),
    )
    return verify_prepare_campaign_barrier(
        seal,
        attestations=items,
        publication_identity=publication_identity,
        protocol_capability=protocol,
    )


def verify_prepare_campaign_barrier(
    seal: PhaseSeal,
    *,
    attestations: tuple[PrepareRotationAttestation, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
) -> PrepareCampaignCapability:
    """Authenticate the global marker solely against safe worker attestations."""

    items, protocol = _validate_campaign_attestations(
        attestations,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
    )
    expected_payloads = _campaign_payloads(items)
    verified = verify_phase_capability(
        seal,
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=_campaign_predecessors(
            items,
            protocol_seal_sha256=protocol.seal.seal_sha256,
        ),
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="prepare",
        scope_id="global",
    )
    for path, expected in expected_payloads.items():
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"prepare campaign payload differs from reconstructed graph: {path}")
    return PrepareCampaignCapability(verified, publication_identity)


__all__ = [
    "BASE_UPDATE_ARTIFACT",
    "BASE_UPDATE_CAPABILITY_ARTIFACT",
    "BASE_UPDATE_PAYLOAD_PATHS",
    "CAMPAIGN_PAYLOAD_PATHS",
    "EVIDENCE_PAYLOAD_PATHS",
    "PREDICTION_VIEW_ARTIFACT",
    "PREDICTION_VIEW_PAYLOAD_PATHS",
    "PREDICTION_VIEW_SUMMARY_ARTIFACT",
    "PREPARE_CAMPAIGN_ARTIFACT",
    "PREPARE_EVIDENCE_ARTIFACT",
    "PREPARE_EVIDENCE_SUMMARY_ARTIFACT",
    "PREPARE_ROTATION_ATTESTATION_ARTIFACT",
    "PROTOCOL_ARTIFACT",
    "PROTOCOL_PAYLOAD_PATHS",
    "RANDOM_MINIMAL_VIEW_ARTIFACT",
    "RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS",
    "RANDOM_MINIMAL_VIEW_SUMMARY_ARTIFACT",
    "BaseUpdateCapability",
    "PrepareArtifactCapabilities",
    "PrepareCampaignCapability",
    "PrepareLeafAttestation",
    "PrepareRotationAttestation",
    "ProtocolCapability",
    "SequentialV2PublicationIdentity",
    "base_update_from_campaign",
    "descriptor_logistic_state_from_bytes",
    "descriptor_logistic_state_from_document",
    "diversity_transform_from_bytes_and_ids",
    "diversity_transform_from_document",
    "prediction_view_from_campaign",
    "prepare_rotation_attestation_from_bytes",
    "prepare_rotation_attestation_from_document",
    "publish_prepare_campaign_barrier",
    "publish_prepared_rotation_artifacts",
    "publish_protocol_capability",
    "random_minimal_view_from_campaign",
    "verify_prepare_campaign_barrier",
    "verify_prepare_campaign_capability",
    "verify_prepared_rotation_artifacts",
    "verify_protocol_capability",
]

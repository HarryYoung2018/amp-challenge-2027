"""Outcome-free global barrier for the sealed sequential-v2 update graph.

State, component, and projection workers return payload-free attestations over
680 immutable leaves.  This module validates those attestations against the
four authenticated upstream globals and a separately supplied, controller-
authoritative leaf-digest sequence.  Only then can it publish ``update/global``.

The resulting capability carries captured bytes for the two label-free global
payloads only.  It contains no model, prediction, candidate, component,
sequence, context, or outcome payload from an update leaf.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.acquisition.sequential_v2_selector import OBJECTIVES, OuterMeanCandidate
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_primitives import TARGETS, ContextRow
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_CONTEXTS_BY_FOLD,
    EXPECTED_POLICY_RUNS,
    EXPECTED_REFITS,
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_run_by_track_id,
    policy_runs_for_rotation,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    RevealCampaignCapability,
    verify_reveal_campaign_capability,
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
    OUTER_METADATA_ROLE,
    StageManifestCapability,
    verify_stage_manifest_capability,
)
from amp_challenge.evaluation.sequential_v2_staging import (
    LabelFreeContext,
    OuterOutcomeVault,
    label_free_contexts,
)
from amp_challenge.evaluation.sequential_v2_update import (
    OUTER_COMPONENT_PAYLOAD_PATHS,
    OUTER_COMPONENTS_ARTIFACT,
    OUTER_EVIDENCE_ARTIFACT,
    OUTER_EVIDENCE_PAYLOAD_PATHS,
    OUTER_VIEW_ARTIFACT,
    OUTER_VIEW_PAYLOAD_PATHS,
    UPDATE_CAMPAIGN_ARTIFACT,
    UPDATE_CAMPAIGN_PAYLOAD_PATHS,
    UPDATE_STATE_ARTIFACT,
    UPDATE_STATE_PAYLOAD_PATHS,
    OuterContextPrediction,
    OuterSequencePrediction,
    id_stream_sha256,
)

SCHEMA_VERSION = 1
EXPECTED_UPDATE_LEAVES = 680
EXPECTED_UPDATE_PREDECESSORS = 684
EXPECTED_BASE_CONTEXT_ASSOCIATIONS = 328_944
EXPECTED_COMPONENT_MEMBERSHIPS = 2_600
EXPECTED_OUTER_CONTEXT_PREDICTIONS = 109_648
EXPECTED_OUTER_SEQUENCE_PREDICTIONS = 28_600
EXPECTED_OUTER_TARGET_PROBABILITIES = 200_200
EXPECTED_OUTER_OBJECTIVE_PROBABILITIES = 85_800

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INDEX_ROLES = ("state", "outer_components", "outer_view", "outer_evidence")
_OUTER_VIEW_FIELDS = (
    "rotation_id",
    "sequence_id",
    "sequence",
    "objective_probabilities",
    "diversity_component_id",
    "eligible",
)
_OUTER_EVIDENCE_SUMMARY_ARTIFACT = "sequential_v2_update_outer_evidence_summary_v1"
_OUTER_CONTEXT_PREDICTION_FIELDS = {
    "schema_version",
    "track_id",
    "rotation_id",
    "example_id",
    "sequence_id",
    "target",
    "gram",
    "fold",
    "probability_hex",
}
_OUTER_SEQUENCE_PREDICTION_FIELDS = {
    "schema_version",
    "track_id",
    "rotation_id",
    "sequence_id",
    "target_probabilities_hex",
    "objective_probabilities_hex",
}


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _canonical_hex(value: object, *, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be canonical binary64 hexadecimal text")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be canonical binary64 hexadecimal text") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{label} must be canonical finite binary64 hexadecimal text")
    return parsed


def _exact_object(
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


def _identity_document(value: SequentialV2PublicationIdentity) -> dict[str, object]:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError("update campaign requires an exact publication identity")
    return {
        "git_commit": value.git_commit,
        "code_manifest_sha256": value.code_manifest_sha256,
        "config_sha256": value.config_sha256,
        "lock_sha256": value.lock_sha256,
    }


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if value != rotation_by_id(value.rotation_id):
        raise ValueError(f"{label} differs from the frozen registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if value != policy_run_by_track_id(value.track_id):
        raise ValueError(f"{label} differs from the frozen registry")
    return value


def _rotation_from_document(value: object) -> RotationSpec:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "rotation_id",
            "outer_fold",
            "acquisition_pool_fold",
            "base_folds",
        },
        label="update campaign rotation",
    )
    rotation_id = raw["rotation_id"]
    if type(rotation_id) is not str:
        raise ValueError("update campaign rotation ID must be text")
    spec = rotation_by_id(rotation_id)
    if canonical_json_bytes(raw) != canonical_json_bytes(spec.document()):
        raise ValueError("update campaign rotation differs from the frozen registry")
    return spec


def _run_from_document(value: object) -> PolicyRunSpec:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "track_id",
            "rotation_id",
            "policy",
            "seed",
            "selection_kind",
            "expected_pool_selection_count",
            "expected_outer_selection_count",
            "refit",
        },
        label="update campaign policy run",
    )
    track_id = raw["track_id"]
    if type(track_id) is not str:
        raise ValueError("update campaign track ID must be text")
    run = policy_run_by_track_id(track_id)
    if canonical_json_bytes(raw) != canonical_json_bytes(run.document()):
        raise ValueError("update campaign run differs from the frozen registry")
    return run


def _payload_digests(
    value: object,
    *,
    paths: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    raw = _exact_object(value, set(paths), label=label)
    result = tuple((path, _sha256(raw[path], label=f"{label} {path}")) for path in paths)
    if tuple(sorted(paths)) != paths:
        raise AssertionError("frozen update payload path order is not canonical")
    return result


def _require_payload_digest_tuple(
    value: object,
    *,
    paths: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in value
    ):
        raise ValueError(f"{label} must be an exact immutable digest map")
    if len(dict(value)) != len(value):
        raise ValueError(f"{label} contains a duplicate payload path")
    expected = _payload_digests(dict(value), paths=paths, label=label)
    if value != expected:
        raise ValueError(f"{label} path order or inventory changed")
    return value


def _state_relative_path(run: PolicyRunSpec) -> str:
    return f"update/tracks/{_require_frozen_run(run, label='state index run').track_id}/state"


def _component_relative_path(spec: RotationSpec) -> str:
    return (
        f"update/rotations/"
        f"{_require_frozen_rotation(spec, label='component index rotation').rotation_id}"
        "/outer-components"
    )


def _view_relative_path(run: PolicyRunSpec) -> str:
    return f"update/tracks/{_require_frozen_run(run, label='view index run').track_id}/outer-view"


def _evidence_relative_path(run: PolicyRunSpec) -> str:
    return (
        f"update/tracks/"
        f"{_require_frozen_run(run, label='evidence index run').track_id}"
        "/outer-evidence"
    )


@dataclass(frozen=True, slots=True)
class UpdateStateIndexRow:
    """One label-free state-leaf identity in the global update index."""

    run: PolicyRunSpec
    leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    base_context_count: int
    base_example_ids_sha256: str
    revealed_context_count: int
    revealed_example_ids_sha256: str
    training_context_count: int
    training_example_ids_sha256: str
    refit: bool

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="update state index run")
        _sha256(self.leaf_seal_sha256, label="update state index leaf seal")
        _require_payload_digest_tuple(
            self.payload_sha256,
            paths=UPDATE_STATE_PAYLOAD_PATHS,
            label="update state index payload digests",
        )
        expected_base = sum(EXPECTED_CONTEXTS_BY_FOLD[fold] for fold in run.rotation.base_folds)
        if type(self.base_context_count) is not int or self.base_context_count != expected_base:
            raise ValueError("update state index base-context census changed")
        _sha256(self.base_example_ids_sha256, label="update state index base IDs")
        if type(self.revealed_context_count) is not int or self.revealed_context_count < 0:
            raise ValueError("update state index reveal census must be nonnegative")
        _sha256(self.revealed_example_ids_sha256, label="update state index reveal IDs")
        if (
            type(self.training_context_count) is not int
            or self.training_context_count != self.base_context_count + self.revealed_context_count
        ):
            raise ValueError("update state index training census is not base plus reveal")
        _sha256(self.training_example_ids_sha256, label="update state index training IDs")
        if type(self.refit) is not bool or self.refit != run.refit:
            raise ValueError("update state index refit flag differs from protocol")
        if run.policy == NO_QUERY:
            if (
                self.revealed_context_count != 0
                or self.revealed_example_ids_sha256 != id_stream_sha256((), allow_empty=True)
                or self.training_example_ids_sha256 != self.base_example_ids_sha256
            ):
                raise ValueError("no-query update state index does not preserve its base IDs")
        elif self.revealed_context_count <= 0:
            raise ValueError("refitted update state index has no revealed context")

    @property
    def relative_path(self) -> str:
        return _state_relative_path(self.run)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "state",
            "run": self.run.document(),
            "relative_path": self.relative_path,
            "leaf_artifact": UPDATE_STATE_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "base_context_count": self.base_context_count,
            "base_example_ids_sha256": self.base_example_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "training_context_count": self.training_context_count,
            "training_example_ids_sha256": self.training_example_ids_sha256,
            "refit": self.refit,
        }


@dataclass(frozen=True, slots=True)
class UpdateComponentIndexRow:
    """One shared rotation component-leaf identity in the update index."""

    spec: RotationSpec
    leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    candidate_count: int
    candidate_ids_sha256: str
    component_count: int
    component_membership_count: int

    def __post_init__(self) -> None:
        spec = _require_frozen_rotation(self.spec, label="update component index rotation")
        _sha256(self.leaf_seal_sha256, label="update component index leaf seal")
        _require_payload_digest_tuple(
            self.payload_sha256,
            paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            label="update component index payload digests",
        )
        expected = EXPECTED_SUPPORT_BY_FOLD[spec.outer_fold]
        if type(self.candidate_count) is not int or self.candidate_count != expected:
            raise ValueError("update component index candidate census changed")
        _sha256(self.candidate_ids_sha256, label="update component index candidate IDs")
        if type(self.component_count) is not int or self.component_count < 1:
            raise ValueError("update component index must contain at least one component")
        if (
            type(self.component_membership_count) is not int
            or self.component_membership_count != expected
        ):
            raise ValueError("update component index membership census changed")

    @property
    def relative_path(self) -> str:
        return _component_relative_path(self.spec)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_components",
            "rotation": self.spec.document(),
            "relative_path": self.relative_path,
            "leaf_artifact": OUTER_COMPONENTS_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "component_count": self.component_count,
            "component_membership_count": self.component_membership_count,
        }


@dataclass(frozen=True, slots=True)
class UpdateOuterViewIndexRow:
    """One selector-only outer-view leaf identity in the update index."""

    run: PolicyRunSpec
    leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    state_leaf_seal_sha256: str
    outer_component_leaf_seal_sha256: str
    candidate_count: int
    candidate_ids_sha256: str

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="update outer-view index run")
        _sha256(self.leaf_seal_sha256, label="update outer-view index leaf seal")
        _require_payload_digest_tuple(
            self.payload_sha256,
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="update outer-view index payload digests",
        )
        _sha256(self.state_leaf_seal_sha256, label="update outer-view state seal")
        _sha256(
            self.outer_component_leaf_seal_sha256,
            label="update outer-view component seal",
        )
        expected = EXPECTED_SUPPORT_BY_FOLD[run.rotation.outer_fold]
        if type(self.candidate_count) is not int or self.candidate_count != expected:
            raise ValueError("update outer-view candidate census changed")
        _sha256(self.candidate_ids_sha256, label="update outer-view candidate IDs")

    @property
    def relative_path(self) -> str:
        return _view_relative_path(self.run)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_view",
            "run": self.run.document(),
            "relative_path": self.relative_path,
            "leaf_artifact": OUTER_VIEW_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "outer_component_leaf_seal_sha256": (self.outer_component_leaf_seal_sha256),
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class UpdateOuterEvidenceIndexRow:
    """One audit-only outer-evidence leaf identity in the update index."""

    run: PolicyRunSpec
    leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    state_leaf_seal_sha256: str
    outer_component_leaf_seal_sha256: str
    outer_view_leaf_seal_sha256: str
    outer_context_count: int
    outer_example_ids_sha256: str
    outer_sequence_prediction_count: int
    outer_candidate_ids_sha256: str
    outer_component_count: int

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="update outer-evidence index run")
        _sha256(self.leaf_seal_sha256, label="update outer-evidence index leaf seal")
        _require_payload_digest_tuple(
            self.payload_sha256,
            paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            label="update outer-evidence index payload digests",
        )
        _sha256(self.state_leaf_seal_sha256, label="update outer-evidence state seal")
        _sha256(
            self.outer_component_leaf_seal_sha256,
            label="update outer-evidence component seal",
        )
        _sha256(self.outer_view_leaf_seal_sha256, label="update outer-evidence view seal")
        expected_contexts = EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold]
        expected_candidates = EXPECTED_SUPPORT_BY_FOLD[run.rotation.outer_fold]
        if type(self.outer_context_count) is not int or self.outer_context_count != (
            expected_contexts
        ):
            raise ValueError("update outer-evidence context census changed")
        _sha256(self.outer_example_ids_sha256, label="update outer-evidence example IDs")
        if (
            type(self.outer_sequence_prediction_count) is not int
            or self.outer_sequence_prediction_count != expected_candidates
        ):
            raise ValueError("update outer-evidence sequence census changed")
        _sha256(
            self.outer_candidate_ids_sha256,
            label="update outer-evidence candidate IDs",
        )
        if type(self.outer_component_count) is not int or self.outer_component_count < 1:
            raise ValueError("update outer-evidence must bind at least one component")

    @property
    def relative_path(self) -> str:
        return _evidence_relative_path(self.run)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_evidence",
            "run": self.run.document(),
            "relative_path": self.relative_path,
            "leaf_artifact": OUTER_EVIDENCE_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "outer_component_leaf_seal_sha256": (self.outer_component_leaf_seal_sha256),
            "outer_view_leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "outer_context_count": self.outer_context_count,
            "outer_example_ids_sha256": self.outer_example_ids_sha256,
            "outer_sequence_prediction_count": self.outer_sequence_prediction_count,
            "outer_candidate_ids_sha256": self.outer_candidate_ids_sha256,
            "outer_component_count": self.outer_component_count,
        }


UpdateIndexRow = (
    UpdateStateIndexRow
    | UpdateComponentIndexRow
    | UpdateOuterViewIndexRow
    | UpdateOuterEvidenceIndexRow
)


def _state_index_from_document(value: object) -> UpdateStateIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "index_role",
            "run",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "payload_sha256",
            "base_context_count",
            "base_example_ids_sha256",
            "revealed_context_count",
            "revealed_example_ids_sha256",
            "training_context_count",
            "training_example_ids_sha256",
            "refit",
        },
        label="update state index row",
    )
    run = _run_from_document(raw["run"])
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["index_role"]) is not str
        or raw["index_role"] != "state"
        or type(raw["relative_path"]) is not str
        or raw["relative_path"] != _state_relative_path(run)
        or type(raw["leaf_artifact"]) is not str
        or raw["leaf_artifact"] != UPDATE_STATE_ARTIFACT
        or type(raw["refit"]) is not bool
    ):
        raise ValueError("update state index identity changed")
    row = UpdateStateIndexRow(
        run=run,
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="state index leaf"),
        payload_sha256=_payload_digests(
            raw["payload_sha256"],
            paths=UPDATE_STATE_PAYLOAD_PATHS,
            label="state index payload digests",
        ),
        base_context_count=_exact_int(
            raw["base_context_count"], label="state index base contexts", minimum=1
        ),
        base_example_ids_sha256=_sha256(
            raw["base_example_ids_sha256"], label="state index base IDs"
        ),
        revealed_context_count=_exact_int(
            raw["revealed_context_count"], label="state index reveal contexts"
        ),
        revealed_example_ids_sha256=_sha256(
            raw["revealed_example_ids_sha256"], label="state index reveal IDs"
        ),
        training_context_count=_exact_int(
            raw["training_context_count"],
            label="state index training contexts",
            minimum=1,
        ),
        training_example_ids_sha256=_sha256(
            raw["training_example_ids_sha256"], label="state index training IDs"
        ),
        refit=raw["refit"],
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("update state index row does not round-trip exactly")
    return row


def _component_index_from_document(value: object) -> UpdateComponentIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "index_role",
            "rotation",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "payload_sha256",
            "candidate_count",
            "candidate_ids_sha256",
            "component_count",
            "component_membership_count",
        },
        label="update component index row",
    )
    spec = _rotation_from_document(raw["rotation"])
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["index_role"]) is not str
        or raw["index_role"] != "outer_components"
        or type(raw["relative_path"]) is not str
        or raw["relative_path"] != _component_relative_path(spec)
        or type(raw["leaf_artifact"]) is not str
        or raw["leaf_artifact"] != OUTER_COMPONENTS_ARTIFACT
    ):
        raise ValueError("update component index identity changed")
    row = UpdateComponentIndexRow(
        spec=spec,
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="component index leaf"),
        payload_sha256=_payload_digests(
            raw["payload_sha256"],
            paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            label="component index payload digests",
        ),
        candidate_count=_exact_int(
            raw["candidate_count"], label="component index candidates", minimum=1
        ),
        candidate_ids_sha256=_sha256(
            raw["candidate_ids_sha256"], label="component index candidate IDs"
        ),
        component_count=_exact_int(
            raw["component_count"], label="component index components", minimum=1
        ),
        component_membership_count=_exact_int(
            raw["component_membership_count"],
            label="component index memberships",
            minimum=1,
        ),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("update component index row does not round-trip exactly")
    return row


def _outer_view_index_from_document(value: object) -> UpdateOuterViewIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "index_role",
            "run",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "payload_sha256",
            "state_leaf_seal_sha256",
            "outer_component_leaf_seal_sha256",
            "candidate_count",
            "candidate_ids_sha256",
        },
        label="update outer-view index row",
    )
    run = _run_from_document(raw["run"])
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["index_role"]) is not str
        or raw["index_role"] != "outer_view"
        or type(raw["relative_path"]) is not str
        or raw["relative_path"] != _view_relative_path(run)
        or type(raw["leaf_artifact"]) is not str
        or raw["leaf_artifact"] != OUTER_VIEW_ARTIFACT
    ):
        raise ValueError("update outer-view index identity changed")
    row = UpdateOuterViewIndexRow(
        run=run,
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="outer-view index leaf"),
        payload_sha256=_payload_digests(
            raw["payload_sha256"],
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer-view index payload digests",
        ),
        state_leaf_seal_sha256=_sha256(
            raw["state_leaf_seal_sha256"], label="outer-view state leaf"
        ),
        outer_component_leaf_seal_sha256=_sha256(
            raw["outer_component_leaf_seal_sha256"],
            label="outer-view component leaf",
        ),
        candidate_count=_exact_int(
            raw["candidate_count"], label="outer-view candidates", minimum=1
        ),
        candidate_ids_sha256=_sha256(raw["candidate_ids_sha256"], label="outer-view candidate IDs"),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("update outer-view index row does not round-trip exactly")
    return row


def _outer_evidence_index_from_document(value: object) -> UpdateOuterEvidenceIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "index_role",
            "run",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "payload_sha256",
            "state_leaf_seal_sha256",
            "outer_component_leaf_seal_sha256",
            "outer_view_leaf_seal_sha256",
            "outer_context_count",
            "outer_example_ids_sha256",
            "outer_sequence_prediction_count",
            "outer_candidate_ids_sha256",
            "outer_component_count",
        },
        label="update outer-evidence index row",
    )
    run = _run_from_document(raw["run"])
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["index_role"]) is not str
        or raw["index_role"] != "outer_evidence"
        or type(raw["relative_path"]) is not str
        or raw["relative_path"] != _evidence_relative_path(run)
        or type(raw["leaf_artifact"]) is not str
        or raw["leaf_artifact"] != OUTER_EVIDENCE_ARTIFACT
    ):
        raise ValueError("update outer-evidence index identity changed")
    row = UpdateOuterEvidenceIndexRow(
        run=run,
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="outer-evidence index leaf"),
        payload_sha256=_payload_digests(
            raw["payload_sha256"],
            paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            label="outer-evidence index payload digests",
        ),
        state_leaf_seal_sha256=_sha256(
            raw["state_leaf_seal_sha256"], label="outer-evidence state leaf"
        ),
        outer_component_leaf_seal_sha256=_sha256(
            raw["outer_component_leaf_seal_sha256"],
            label="outer-evidence component leaf",
        ),
        outer_view_leaf_seal_sha256=_sha256(
            raw["outer_view_leaf_seal_sha256"], label="outer-evidence view leaf"
        ),
        outer_context_count=_exact_int(
            raw["outer_context_count"], label="outer-evidence contexts", minimum=1
        ),
        outer_example_ids_sha256=_sha256(
            raw["outer_example_ids_sha256"], label="outer-evidence example IDs"
        ),
        outer_sequence_prediction_count=_exact_int(
            raw["outer_sequence_prediction_count"],
            label="outer-evidence sequences",
            minimum=1,
        ),
        outer_candidate_ids_sha256=_sha256(
            raw["outer_candidate_ids_sha256"],
            label="outer-evidence candidate IDs",
        ),
        outer_component_count=_exact_int(
            raw["outer_component_count"],
            label="outer-evidence components",
            minimum=1,
        ),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("update outer-evidence index row does not round-trip exactly")
    return row


def _decode_index(payload: bytes) -> tuple[UpdateIndexRow, ...]:
    raw_rows = _strict_jsonl(payload, label="update campaign index")
    if len(raw_rows) != EXPECTED_UPDATE_LEAVES:
        raise ValueError("update campaign index must contain exactly 680 rows")
    expected_roles = (
        *("state" for _ in range(EXPECTED_POLICY_RUNS)),
        *("outer_components" for _ in range(EXPECTED_ROTATIONS)),
        *("outer_view" for _ in range(EXPECTED_POLICY_RUNS)),
        *("outer_evidence" for _ in range(EXPECTED_POLICY_RUNS)),
    )
    rows: list[UpdateIndexRow] = []
    parsers = {
        "state": _state_index_from_document,
        "outer_components": _component_index_from_document,
        "outer_view": _outer_view_index_from_document,
        "outer_evidence": _outer_evidence_index_from_document,
    }
    for index, (raw, expected_role) in enumerate(zip(raw_rows, expected_roles, strict=True)):
        role = raw.get("index_role")
        if type(role) is not str or role != expected_role or role not in parsers:
            raise ValueError(f"update campaign index role/order changed at row {index}")
        rows.append(parsers[role](raw))
    result = tuple(rows)
    states = tuple(item for item in result if type(item) is UpdateStateIndexRow)
    components = tuple(item for item in result if type(item) is UpdateComponentIndexRow)
    views = tuple(item for item in result if type(item) is UpdateOuterViewIndexRow)
    evidence = tuple(item for item in result if type(item) is UpdateOuterEvidenceIndexRow)
    if (
        tuple(item.run for item in states) != ordered_policy_runs()
        or tuple(item.spec for item in components) != ordered_rotations()
        or tuple(item.run for item in views) != ordered_policy_runs()
        or tuple(item.run for item in evidence) != ordered_policy_runs()
    ):
        raise ValueError("update campaign index differs from frozen run/rotation order")
    return result


def _partition_index(
    rows: tuple[UpdateIndexRow, ...],
) -> tuple[
    tuple[UpdateStateIndexRow, ...],
    tuple[UpdateComponentIndexRow, ...],
    tuple[UpdateOuterViewIndexRow, ...],
    tuple[UpdateOuterEvidenceIndexRow, ...],
]:
    if type(rows) is not tuple or len(rows) != EXPECTED_UPDATE_LEAVES:
        raise ValueError("update campaign requires one exact 680-row index tuple")
    states = tuple(item for item in rows if type(item) is UpdateStateIndexRow)
    components = tuple(item for item in rows if type(item) is UpdateComponentIndexRow)
    views = tuple(item for item in rows if type(item) is UpdateOuterViewIndexRow)
    evidence = tuple(item for item in rows if type(item) is UpdateOuterEvidenceIndexRow)
    if rows != (*states, *components, *views, *evidence) or (
        len(states),
        len(components),
        len(views),
        len(evidence),
    ) != (EXPECTED_POLICY_RUNS, EXPECTED_ROTATIONS, EXPECTED_POLICY_RUNS, EXPECTED_POLICY_RUNS):
        raise ValueError("update campaign index role partitions changed")
    return states, components, views, evidence


def _validate_index_cross_links(rows: tuple[UpdateIndexRow, ...]) -> None:
    states, components, views, evidence = _partition_index(rows)
    state_by_run = {item.run: item for item in states}
    component_by_rotation = {item.spec: item for item in components}
    view_by_run = {item.run: item for item in views}
    if (
        len(state_by_run) != EXPECTED_POLICY_RUNS
        or len(component_by_rotation) != EXPECTED_ROTATIONS
        or len(view_by_run) != EXPECTED_POLICY_RUNS
    ):
        raise ValueError("update campaign index contains duplicate logical identities")
    leaf_digests = tuple(item.leaf_seal_sha256 for item in rows)
    if len(set(leaf_digests)) != EXPECTED_UPDATE_LEAVES:
        raise ValueError("all 680 update leaf seals must be distinct")
    outer_ids_by_rotation: dict[RotationSpec, str] = {}
    for view, audit in zip(views, evidence, strict=True):
        state = state_by_run[view.run]
        component = component_by_rotation[view.run.rotation]
        if (
            audit.run != view.run
            or view.state_leaf_seal_sha256 != state.leaf_seal_sha256
            or view.outer_component_leaf_seal_sha256 != component.leaf_seal_sha256
            or view.candidate_count != component.candidate_count
            or view.candidate_ids_sha256 != component.candidate_ids_sha256
            or audit.state_leaf_seal_sha256 != state.leaf_seal_sha256
            or audit.outer_component_leaf_seal_sha256 != component.leaf_seal_sha256
            or audit.outer_view_leaf_seal_sha256 != view.leaf_seal_sha256
            or audit.outer_sequence_prediction_count != view.candidate_count
            or audit.outer_candidate_ids_sha256 != view.candidate_ids_sha256
            or audit.outer_component_count != component.component_count
        ):
            raise ValueError("update view/evidence index links are inconsistent")
        prior = outer_ids_by_rotation.setdefault(
            view.run.rotation,
            audit.outer_example_ids_sha256,
        )
        if prior != audit.outer_example_ids_sha256:
            raise ValueError("tracks in one rotation bind different outer example IDs")
    if set(outer_ids_by_rotation) != set(ordered_rotations()):
        raise ValueError("update campaign lacks an outer example-ID stream per rotation")


def _campaign_summary_document(
    rows: tuple[UpdateIndexRow, ...],
    *,
    update_index_sha256: str,
) -> dict[str, object]:
    _validate_index_cross_links(rows)
    states, components, views, evidence = _partition_index(rows)
    base_count = sum(item.base_context_count for item in states)
    revealed_count = sum(item.revealed_context_count for item in states)
    training_count = sum(item.training_context_count for item in states)
    component_count = sum(item.component_count for item in components)
    membership_count = sum(item.component_membership_count for item in components)
    outer_context_count = sum(item.outer_context_count for item in evidence)
    sequence_count = sum(item.outer_sequence_prediction_count for item in evidence)
    candidate_count = sum(item.candidate_count for item in views)
    refits = sum(item.refit for item in states)
    if (
        base_count != EXPECTED_BASE_CONTEXT_ASSOCIATIONS
        or training_count != base_count + revealed_count
        or refits != EXPECTED_REFITS
        or len(states) - refits != EXPECTED_ROTATIONS
        or component_count <= 0
        or membership_count != EXPECTED_COMPONENT_MEMBERSHIPS
        or outer_context_count != EXPECTED_OUTER_CONTEXT_PREDICTIONS
        or sequence_count != EXPECTED_OUTER_SEQUENCE_PREDICTIONS
        or candidate_count != EXPECTED_OUTER_SEQUENCE_PREDICTIONS
        or sequence_count * 7 != EXPECTED_OUTER_TARGET_PROBABILITIES
        or sequence_count * 3 != EXPECTED_OUTER_OBJECTIVE_PROBABILITIES
    ):
        raise ValueError("update campaign census differs from the frozen graph")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": UPDATE_CAMPAIGN_ARTIFACT,
        "rotation_count": EXPECTED_ROTATIONS,
        "track_count": EXPECTED_POLICY_RUNS,
        "update_leaf_count": EXPECTED_UPDATE_LEAVES,
        "state_leaf_count": len(states),
        "outer_component_leaf_count": len(components),
        "outer_view_leaf_count": len(views),
        "outer_evidence_leaf_count": len(evidence),
        "updated_model_count": len(states),
        "refit_count": refits,
        "no_refit_count": len(states) - refits,
        "base_context_association_count": base_count,
        "revealed_context_association_count": revealed_count,
        "training_context_association_count": training_count,
        "training_example_id_row_count": training_count,
        "outer_component_count": component_count,
        "outer_component_membership_association_count": membership_count,
        "outer_context_prediction_count": outer_context_count,
        "outer_sequence_prediction_count": sequence_count,
        "outer_candidate_count": candidate_count,
        "outer_target_probability_scalar_count": sequence_count * 7,
        "outer_objective_probability_scalar_count": sequence_count * 3,
        "update_index_sha256": _sha256(
            update_index_sha256,
            label="update campaign index digest",
        ),
    }


def _protocol_predecessor() -> str:
    return "protocol/SHA256SUMS"


def _stage_predecessor() -> str:
    return "stage/global/SHA256SUMS"


def _prepare_predecessor() -> str:
    return "prepare/global/SHA256SUMS"


def _reveal_predecessor() -> str:
    return "reveal/global/SHA256SUMS"


def _campaign_predecessors(
    rows: tuple[UpdateIndexRow, ...],
    *,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    prepare_global_seal_sha256: str,
    reveal_global_seal_sha256: str,
) -> dict[str, str]:
    _validate_index_cross_links(rows)
    result = {
        _protocol_predecessor(): _sha256(
            protocol_seal_sha256,
            label="update campaign protocol seal",
        ),
        _stage_predecessor(): _sha256(
            stage_global_seal_sha256,
            label="update campaign stage-global seal",
        ),
        _prepare_predecessor(): _sha256(
            prepare_global_seal_sha256,
            label="update campaign prepare-global seal",
        ),
        _reveal_predecessor(): _sha256(
            reveal_global_seal_sha256,
            label="update campaign reveal-global seal",
        ),
        **{f"{item.relative_path}/SHA256SUMS": item.leaf_seal_sha256 for item in rows},
    }
    if len(result) != EXPECTED_UPDATE_PREDECESSORS:
        raise ValueError("update campaign predecessor inventory must contain exactly 684 keys")
    return result


def _campaign_payloads(rows: tuple[UpdateIndexRow, ...]) -> dict[str, bytes]:
    _validate_index_cross_links(rows)
    index_payload = canonical_jsonl_bytes(item.document() for item in rows)
    return {
        "update-index.jsonl": index_payload,
        "update-summary.json": canonical_json_bytes(
            _campaign_summary_document(
                rows,
                update_index_sha256=sha256_bytes(index_payload),
            )
        ),
    }


def _decode_safe_campaign(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[tuple[UpdateIndexRow, ...], str, str, str, str]:
    if type(seal) is not PhaseSeal:
        raise TypeError("update campaign requires an exact rootless PhaseSeal")
    _identity_document(publication_identity)
    expected = _sha256(expected_seal_sha256, label="expected update campaign seal")
    preliminary = verify_phase_capability(
        seal,
        expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected,
    )
    rows = _decode_index(preliminary.read_payload_bytes("update-index.jsonl"))
    anchors = dict(preliminary.predecessor_seals)
    protocol = _sha256(
        anchors.get(_protocol_predecessor()),
        label="update campaign protocol predecessor",
    )
    stage = _sha256(
        anchors.get(_stage_predecessor()),
        label="update campaign stage predecessor",
    )
    prepare = _sha256(
        anchors.get(_prepare_predecessor()),
        label="update campaign prepare predecessor",
    )
    reveal = _sha256(
        anchors.get(_reveal_predecessor()),
        label="update campaign reveal predecessor",
    )
    verified = verify_phase_capability(
        preliminary,
        expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=protocol,
            stage_global_seal_sha256=stage,
            prepare_global_seal_sha256=prepare,
            reveal_global_seal_sha256=reveal,
        ),
        expected_seal_sha256=expected,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id="global",
    )
    index_payload = verified.read_payload_bytes("update-index.jsonl")
    expected_summary = _campaign_summary_document(
        rows,
        update_index_sha256=sha256_bytes(index_payload),
    )
    summary_payload = verified.read_payload_bytes("update-summary.json")
    _strict_json_object(summary_payload, label="update campaign summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("update campaign summary differs from its exact index census")
    return rows, protocol, stage, prepare, reveal


@dataclass(frozen=True, slots=True)
class UpdateCampaignCapability:
    """Rootless label-free global update barrier, not authority by itself."""

    seal: PhaseSeal
    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("update campaign capability requires an exact PhaseSeal")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("update campaign capability requires an exact publication identity")
        _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )

    @property
    def anchor_digests(self) -> tuple[tuple[str, str], ...]:
        _rows, protocol, stage, prepare, reveal = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )
        return (
            (_protocol_predecessor(), protocol),
            (_stage_predecessor(), stage),
            (_prepare_predecessor(), prepare),
            (_reveal_predecessor(), reveal),
        )

    def index_rows(self) -> tuple[UpdateIndexRow, ...]:
        rows, _protocol, _stage, _prepare, _reveal = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )
        return rows

    def outer_view_row(self, *, run: PolicyRunSpec) -> UpdateOuterViewIndexRow:
        frozen = _require_frozen_run(run, label="update campaign view lookup run")
        rows = self.index_rows()
        matches = tuple(
            item for item in rows if type(item) is UpdateOuterViewIndexRow and item.run == frozen
        )
        if len(matches) != 1:
            raise ValueError("update campaign lacks one exact requested outer view")
        return matches[0]

    def outer_evidence_row(self, *, run: PolicyRunSpec) -> UpdateOuterEvidenceIndexRow:
        frozen = _require_frozen_run(run, label="update campaign evidence lookup run")
        rows = self.index_rows()
        matches = tuple(
            item
            for item in rows
            if type(item) is UpdateOuterEvidenceIndexRow and item.run == frozen
        )
        if len(matches) != 1:
            raise ValueError("update campaign lacks one exact requested outer evidence leaf")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _AuthenticatedUpdateGlobals:
    protocol: ProtocolCapability
    stage: StageManifestCapability
    prepare: PrepareCampaignCapability
    reveal: RevealCampaignCapability


def _authenticate_upstream_globals(
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    selection_barrier: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> _AuthenticatedUpdateGlobals:
    _identity_document(publication_identity)
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("update campaign requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    stage_digest = _sha256(
        expected_stage_global_seal_sha256,
        label="expected update campaign stage-global seal",
    )
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("update campaign requires an exact StageManifestCapability")
    stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_digest,
    )
    if type(prepare_campaign) is not PrepareCampaignCapability:
        raise TypeError("update campaign requires an exact PrepareCampaignCapability")
    prepare = verify_prepare_campaign_capability(
        prepare_campaign,
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=_sha256(
            expected_prepare_campaign_seal_sha256,
            label="expected update campaign prepare-global seal",
        ),
        expected_protocol_seal_sha256=protocol.seal.seal_sha256,
    )
    if type(reveal_campaign) is not RevealCampaignCapability:
        raise TypeError("update campaign requires an exact RevealCampaignCapability")
    if type(selection_barrier) is not PhaseSeal:
        raise TypeError("update campaign requires an exact select-global PhaseSeal")
    reveal = verify_reveal_campaign_capability(
        reveal_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=prepare.seal.seal_sha256,
        expected_stage_global_seal_sha256=stage.seal.seal_sha256,
        expected_selection_barrier_seal_sha256=_sha256(
            expected_selection_barrier_seal_sha256,
            label="expected update campaign select-global seal",
        ),
        expected_reveal_campaign_seal_sha256=_sha256(
            expected_reveal_campaign_seal_sha256,
            label="expected update campaign reveal-global seal",
        ),
    )
    return _AuthenticatedUpdateGlobals(protocol, stage, prepare, reveal)


def _validate_expected_leaf_digests(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != EXPECTED_UPDATE_LEAVES:
        raise ValueError("controller update leaf digest sequence must contain exactly 680 items")
    result = tuple(
        _sha256(item, label=f"controller update leaf digest {index}")
        for index, item in enumerate(value)
    )
    if len(set(result)) != EXPECTED_UPDATE_LEAVES:
        raise ValueError("controller update leaf digest sequence must be entirely distinct")
    return result


def _validate_attestations(
    *,
    state_attestations: tuple[object, ...],
    component_attestations: tuple[object, ...],
    projection_attestations: tuple[object, ...],
    expected_update_leaf_seal_sha256s: tuple[str, ...],
    globals_: _AuthenticatedUpdateGlobals,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[UpdateIndexRow, ...]:
    from amp_challenge.evaluation.sequential_v2_update_outer import (
        OuterComponentAttestation,
        OuterProjectionAttestation,
        outer_component_attestation_from_document,
        outer_projection_attestation_from_document,
    )
    from amp_challenge.evaluation.sequential_v2_update_state import (
        UpdateStateAttestation,
        update_state_attestation_from_document,
    )

    if (
        type(state_attestations) is not tuple
        or type(component_attestations) is not tuple
        or type(projection_attestations) is not tuple
    ):
        raise TypeError("update campaign attestations must be exact immutable tuples")
    runs = ordered_policy_runs()
    rotations = ordered_rotations()
    if (
        len(state_attestations) != EXPECTED_POLICY_RUNS
        or any(type(item) is not UpdateStateAttestation for item in state_attestations)
        or tuple(item.run for item in state_attestations) != runs
        or len(component_attestations) != EXPECTED_ROTATIONS
        or any(type(item) is not OuterComponentAttestation for item in component_attestations)
        or tuple(item.spec for item in component_attestations) != rotations
        or len(projection_attestations) != EXPECTED_POLICY_RUNS
        or any(type(item) is not OuterProjectionAttestation for item in projection_attestations)
        or tuple(item.run for item in projection_attestations) != runs
    ):
        raise ValueError("update campaign attestation order or census changed")
    expected = _validate_expected_leaf_digests(expected_update_leaf_seal_sha256s)
    expected_states = expected[:EXPECTED_POLICY_RUNS]
    component_stop = EXPECTED_POLICY_RUNS + EXPECTED_ROTATIONS
    expected_components = expected[EXPECTED_POLICY_RUNS:component_stop]
    view_stop = component_stop + EXPECTED_POLICY_RUNS
    expected_views = expected[component_stop:view_stop]
    expected_evidence = expected[view_stop:]

    state_rows: list[UpdateStateIndexRow] = []
    state_by_run: dict[PolicyRunSpec, object] = {}
    base_identity_by_rotation: dict[RotationSpec, tuple[object, ...]] = {}
    for index, (item, expected_leaf) in enumerate(
        zip(state_attestations, expected_states, strict=True)
    ):
        reconstructed = update_state_attestation_from_document(item.document())
        if reconstructed != item:
            raise ValueError(f"update state attestation {index} changed on reconstruction")
        reveal_row = globals_.reveal.index_row(run=item.run)
        expected_base_leaf = globals_.prepare.leaf_seal_sha256(
            spec=item.run.rotation,
            role="base_update",
        )
        expected_base_contexts_payload = globals_.prepare.leaf_payload_sha256(
            spec=item.run.rotation,
            role="base_update",
            path="base-contexts.jsonl",
        )
        expected_base_model_payload = globals_.prepare.leaf_payload_sha256(
            spec=item.run.rotation,
            role="base_update",
            path="base-model.json",
        )
        expected_base_model_state = globals_.prepare.base_model_state_sha256(spec=item.run.rotation)
        if (
            item.publication_identity != publication_identity
            or item.protocol_seal_sha256 != globals_.protocol.seal.seal_sha256
            or item.prepare_global_seal_sha256 != globals_.prepare.seal.seal_sha256
            or item.reveal_global_seal_sha256 != globals_.reveal.seal.seal_sha256
            or item.base_update_leaf_seal_sha256 != expected_base_leaf
            or item.base_contexts_payload_sha256 != expected_base_contexts_payload
            or item.base_model_payload_sha256 != expected_base_model_payload
            or item.base_model_state_sha256 != expected_base_model_state
            or item.reveal_leaf_seal_sha256 != reveal_row.leaf_seal_sha256
            or item.revealed_context_count != reveal_row.revealed_context_count
            or item.revealed_example_ids_sha256 != reveal_row.revealed_example_ids_sha256
            or item.state_leaf_seal_sha256 != expected_leaf
        ):
            raise ValueError("update state attestation differs from global/controller authority")
        base_identity = (
            item.base_update_leaf_seal_sha256,
            item.base_contexts_payload_sha256,
            item.base_model_payload_sha256,
            item.base_context_count,
            item.base_example_ids_sha256,
        )
        prior = base_identity_by_rotation.setdefault(item.run.rotation, base_identity)
        if prior != base_identity:
            raise ValueError("state tracks in one rotation bind different base identities")
        row = _state_index_from_document(item.index_document())
        state_rows.append(row)
        state_by_run[item.run] = item
    if len(state_by_run) != EXPECTED_POLICY_RUNS:
        raise ValueError("update campaign state attestation identities are not unique")

    component_rows: list[UpdateComponentIndexRow] = []
    component_by_rotation: dict[RotationSpec, object] = {}
    for index, (item, expected_leaf) in enumerate(
        zip(component_attestations, expected_components, strict=True)
    ):
        reconstructed = outer_component_attestation_from_document(item.document())
        if reconstructed != item:
            raise ValueError(f"outer component attestation {index} changed on reconstruction")
        metadata_entry = globals_.stage.leaf(spec=item.spec, role=OUTER_METADATA_ROLE)
        expected_state_seals = tuple(
            state_by_run[run].state_leaf_seal_sha256 for run in policy_runs_for_rotation(item.spec)
        )
        if (
            item.publication_identity != publication_identity
            or item.protocol_seal_sha256 != globals_.protocol.seal.seal_sha256
            or item.stage_global_seal_sha256 != globals_.stage.seal.seal_sha256
            or item.outer_metadata_leaf_seal_sha256 != metadata_entry.leaf_seal_sha256
            or item.state_leaf_seal_sha256s != expected_state_seals
            or item.component_leaf_seal_sha256 != expected_leaf
            or item.candidate_count != EXPECTED_SUPPORT_BY_FOLD[item.spec.outer_fold]
            or item.candidate_ids_sha256
            != metadata_entry.id_stream_map["outer_support_sequence_ids"].sha256
        ):
            raise ValueError("outer component attestation differs from global/controller authority")
        row = _component_index_from_document(item.component_index_document())
        component_rows.append(row)
        component_by_rotation[item.spec] = item
    if len(component_by_rotation) != EXPECTED_ROTATIONS:
        raise ValueError("update campaign component identities are not unique")

    view_rows: list[UpdateOuterViewIndexRow] = []
    evidence_rows: list[UpdateOuterEvidenceIndexRow] = []
    for index, (item, expected_view, expected_audit) in enumerate(
        zip(
            projection_attestations,
            expected_views,
            expected_evidence,
            strict=True,
        )
    ):
        reconstructed = outer_projection_attestation_from_document(item.document())
        if reconstructed != item:
            raise ValueError(f"outer projection attestation {index} changed on reconstruction")
        state = state_by_run[item.run]
        component = component_by_rotation[item.run.rotation]
        metadata_entry = globals_.stage.leaf(
            spec=item.run.rotation,
            role=OUTER_METADATA_ROLE,
        )
        if (
            item.publication_identity != publication_identity
            or item.protocol_seal_sha256 != globals_.protocol.seal.seal_sha256
            or item.stage_global_seal_sha256 != globals_.stage.seal.seal_sha256
            or item.outer_metadata_leaf_seal_sha256 != metadata_entry.leaf_seal_sha256
            or item.state_leaf_seal_sha256 != state.state_leaf_seal_sha256
            or item.updated_model_payload_sha256 != dict(state.payload_sha256)["updated-model.json"]
            or item.outer_component_leaf_seal_sha256 != component.component_leaf_seal_sha256
            or item.outer_components_payload_sha256
            != dict(component.payload_sha256)["outer-components.jsonl"]
            or item.outer_view_leaf_seal_sha256 != expected_view
            or item.outer_evidence_leaf_seal_sha256 != expected_audit
            or item.outer_context_count != EXPECTED_CONTEXTS_BY_FOLD[item.run.rotation.outer_fold]
            or item.outer_example_ids_sha256
            != metadata_entry.id_stream_map["outer_metadata_example_ids"].sha256
            or item.outer_candidate_count != component.candidate_count
            or item.outer_candidate_ids_sha256 != component.candidate_ids_sha256
            or item.outer_component_count != component.component_count
            or item.outer_component_membership_count != component.component_membership_count
        ):
            raise ValueError(
                "outer projection attestation differs from global/controller authority"
            )
        view_rows.append(_outer_view_index_from_document(item.view_index_document()))
        evidence_rows.append(_outer_evidence_index_from_document(item.evidence_index_document()))
    rows: tuple[UpdateIndexRow, ...] = (
        *state_rows,
        *component_rows,
        *view_rows,
        *evidence_rows,
    )
    actual = tuple(item.leaf_seal_sha256 for item in rows)
    if actual != expected:
        raise ValueError("update attestation leaves differ from controller digest order")
    _campaign_summary_document(rows, update_index_sha256="0" * 64)
    return rows


def publish_update_campaign_barrier(
    destination: str | Path,
    *,
    state_attestations: tuple[object, ...],
    component_attestations: tuple[object, ...],
    projection_attestations: tuple[object, ...],
    expected_update_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    selection_barrier: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> UpdateCampaignCapability:
    """Publish the 684-predecessor update barrier from payload-free IPC records."""

    globals_ = _authenticate_upstream_globals(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        selection_barrier=selection_barrier,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    rows = _validate_attestations(
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
        globals_=globals_,
        publication_identity=publication_identity,
    )
    payloads = _campaign_payloads(rows)
    seal = publish_phase(
        destination,
        artifact=UPDATE_CAMPAIGN_ARTIFACT,
        payloads=payloads,
        predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
            stage_global_seal_sha256=globals_.stage.seal.seal_sha256,
            prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
            reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
        ),
        metadata=publication_identity.metadata(phase="update", scope_id="global"),
    )
    return verify_update_campaign_barrier(
        seal,
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=globals_.protocol,
        stage_manifest_capability=globals_.stage,
        prepare_campaign=globals_.prepare,
        reveal_campaign=globals_.reveal,
        selection_barrier=selection_barrier,
        expected_stage_global_seal_sha256=globals_.stage.seal.seal_sha256,
        expected_prepare_campaign_seal_sha256=globals_.prepare.seal.seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=globals_.reveal.seal.seal_sha256,
        expected_update_campaign_seal_sha256=seal.seal_sha256,
    )


def verify_update_campaign_barrier(
    seal: PhaseSeal,
    *,
    state_attestations: tuple[object, ...],
    component_attestations: tuple[object, ...],
    projection_attestations: tuple[object, ...],
    expected_update_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    selection_barrier: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> UpdateCampaignCapability:
    """Reconstruct and authenticate update/global from its safe worker results."""

    globals_ = _authenticate_upstream_globals(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        selection_barrier=selection_barrier,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    rows = _validate_attestations(
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
        globals_=globals_,
        publication_identity=publication_identity,
    )
    expected_payloads = _campaign_payloads(rows)
    verified = verify_phase_capability(
        seal,
        expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
            stage_global_seal_sha256=globals_.stage.seal.seal_sha256,
            prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
            reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
        ),
        expected_seal_sha256=_sha256(
            expected_update_campaign_seal_sha256,
            label="controller-authoritative update campaign seal",
        ),
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id="global",
    )
    for path, expected in expected_payloads.items():
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"update campaign payload differs from attestations: {path}")
    return UpdateCampaignCapability(verified, publication_identity)


def update_campaign_capability_from_seal(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> UpdateCampaignCapability:
    """Derive the rootless global capability under controller anchor digests."""

    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("update campaign derivation requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    rows, actual_protocol, actual_stage, actual_prepare, actual_reveal = _decode_safe_campaign(
        seal,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_update_campaign_seal_sha256,
    )
    if len(rows) != EXPECTED_UPDATE_LEAVES:
        raise AssertionError("authenticated update campaign row census changed")
    if (
        actual_protocol != protocol.seal.seal_sha256
        or actual_stage
        != _sha256(
            expected_stage_global_seal_sha256,
            label="expected update capability stage-global seal",
        )
        or actual_prepare
        != _sha256(
            expected_prepare_campaign_seal_sha256,
            label="expected update capability prepare-global seal",
        )
        or actual_reveal
        != _sha256(
            expected_reveal_campaign_seal_sha256,
            label="expected update capability reveal-global seal",
        )
    ):
        raise ValueError("update campaign binds the wrong upstream global authorities")
    return UpdateCampaignCapability(seal, publication_identity)


def verify_update_campaign_capability(
    capability: UpdateCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> UpdateCampaignCapability:
    """Freshly rederive update/global before granting outer-select authority."""

    if type(capability) is not UpdateCampaignCapability:
        raise TypeError("update campaign verification requires an UpdateCampaignCapability")
    if capability.publication_identity != publication_identity:
        raise ValueError("update campaign capability has the wrong publication identity")
    verified = update_campaign_capability_from_seal(
        capability.seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    if capability.seal.seal_sha256 != verified.seal.seal_sha256:
        raise ValueError("update campaign capability differs from authoritative derivation")
    return verified


def _outer_view_summary(
    value: object,
    *,
    run: PolicyRunSpec,
    row: UpdateOuterViewIndexRow,
) -> tuple[str, str]:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "view_kind",
            "state_leaf_seal_sha256",
            "outer_component_leaf_seal_sha256",
            "outer_metadata_leaf_seal_sha256",
            "candidate_count",
            "candidate_ids_sha256",
            "candidate_payload_sha256",
            "field_names",
        },
        label="authenticated outer-view summary",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["artifact"]) is not str
        or raw["artifact"] != "sequential_v2_update_outer_view_summary_v1"
        or _run_from_document(raw["run"]) != run
        or type(raw["view_kind"]) is not str
        or raw["view_kind"] != "outer_mean"
        or _sha256(raw["state_leaf_seal_sha256"], label="outer-view summary state")
        != row.state_leaf_seal_sha256
        or _sha256(
            raw["outer_component_leaf_seal_sha256"],
            label="outer-view summary component",
        )
        != row.outer_component_leaf_seal_sha256
        or _exact_int(
            raw["candidate_count"],
            label="outer-view summary candidates",
            minimum=1,
        )
        != row.candidate_count
        or _sha256(raw["candidate_ids_sha256"], label="outer-view summary IDs")
        != row.candidate_ids_sha256
        or _sha256(
            raw["candidate_payload_sha256"],
            label="outer-view summary candidate payload",
        )
        != dict(row.payload_sha256)["candidates.jsonl"]
        or type(raw["field_names"]) is not list
        or tuple(raw["field_names"]) != _OUTER_VIEW_FIELDS
        or any(type(item) is not str for item in raw["field_names"])
    ):
        raise ValueError("outer-view summary differs from its authenticated index row")
    metadata_leaf = _sha256(
        raw["outer_metadata_leaf_seal_sha256"],
        label="outer-view summary metadata leaf",
    )
    return metadata_leaf, row.outer_component_leaf_seal_sha256


def _outer_evidence_summary(
    value: object,
    *,
    run: PolicyRunSpec,
    row: UpdateOuterEvidenceIndexRow,
    state_row: UpdateStateIndexRow,
    component_row: UpdateComponentIndexRow,
    view_row: UpdateOuterViewIndexRow,
    stage_global_seal_sha256: str,
) -> str:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "state_leaf_seal_sha256",
            "updated_model_payload_sha256",
            "stage_global_seal_sha256",
            "outer_metadata_leaf_seal_sha256",
            "outer_component_leaf_seal_sha256",
            "outer_components_payload_sha256",
            "outer_view_leaf_seal_sha256",
            "outer_view_candidates_payload_sha256",
            "outer_context_count",
            "outer_example_ids_sha256",
            "outer_context_predictions_payload_sha256",
            "outer_sequence_prediction_count",
            "outer_candidate_ids_sha256",
            "outer_sequence_predictions_payload_sha256",
            "outer_component_count",
            "outer_component_membership_count",
        },
        label="authenticated outer-evidence summary",
    )
    metadata_leaf = _sha256(
        raw["outer_metadata_leaf_seal_sha256"],
        label="outer-evidence summary metadata leaf",
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": _OUTER_EVIDENCE_SUMMARY_ARTIFACT,
        "run": run.document(),
        "state_leaf_seal_sha256": row.state_leaf_seal_sha256,
        "updated_model_payload_sha256": dict(state_row.payload_sha256)["updated-model.json"],
        "stage_global_seal_sha256": stage_global_seal_sha256,
        "outer_metadata_leaf_seal_sha256": metadata_leaf,
        "outer_component_leaf_seal_sha256": row.outer_component_leaf_seal_sha256,
        "outer_components_payload_sha256": dict(component_row.payload_sha256)[
            "outer-components.jsonl"
        ],
        "outer_view_leaf_seal_sha256": row.outer_view_leaf_seal_sha256,
        "outer_view_candidates_payload_sha256": dict(view_row.payload_sha256)["candidates.jsonl"],
        "outer_context_count": row.outer_context_count,
        "outer_example_ids_sha256": row.outer_example_ids_sha256,
        "outer_context_predictions_payload_sha256": dict(row.payload_sha256)[
            "outer-context-predictions.jsonl"
        ],
        "outer_sequence_prediction_count": row.outer_sequence_prediction_count,
        "outer_candidate_ids_sha256": row.outer_candidate_ids_sha256,
        "outer_sequence_predictions_payload_sha256": dict(row.payload_sha256)[
            "outer-sequence-predictions.jsonl"
        ],
        "outer_component_count": row.outer_component_count,
        "outer_component_membership_count": component_row.component_membership_count,
    }
    if canonical_json_bytes(raw) != canonical_json_bytes(expected):
        raise ValueError("outer-evidence summary differs from its authenticated index rows")
    return metadata_leaf


def _outer_vault_contexts(
    value: object,
    *,
    run: PolicyRunSpec,
    row: UpdateOuterEvidenceIndexRow,
) -> tuple[LabelFreeContext, ...]:
    if type(value) is not OuterOutcomeVault:
        raise TypeError("outer-evidence materialization requires an exact OuterOutcomeVault")
    if _require_frozen_rotation(value.spec, label="outer outcome vault rotation") != run.rotation:
        raise ValueError("outer outcome vault belongs to the wrong rotation")
    if type(value.contexts) is not tuple or any(
        type(item) is not ContextRow for item in value.contexts
    ):
        raise TypeError("outer outcome vault contexts must be exact immutable ContextRow values")
    if type(value.allowed_example_ids) is not tuple or any(
        type(item) is not str for item in value.allowed_example_ids
    ):
        raise TypeError("outer outcome vault allowed IDs must be an exact immutable text tuple")
    identifiers = tuple(item.example_id for item in value.contexts)
    if (
        len(identifiers) != row.outer_context_count
        or identifiers != tuple(sorted(set(identifiers)))
        or value.allowed_example_ids != identifiers
        or id_stream_sha256(identifiers) != row.outer_example_ids_sha256
        or any(item.fold != run.rotation.outer_fold for item in value.contexts)
    ):
        raise ValueError("outer outcome vault differs from update-global prediction identity")
    contexts = label_free_contexts(value.contexts)
    if tuple(item.example_id for item in contexts) != identifiers:
        raise AssertionError("outer outcome vault label stripping changed context order")
    return contexts


def _decode_authenticated_outer_context_predictions(
    payload: bytes,
    *,
    run: PolicyRunSpec,
    contexts: tuple[LabelFreeContext, ...],
) -> tuple[OuterContextPrediction, ...]:
    rows = _strict_jsonl(payload, label="authenticated outer-context predictions")
    if len(rows) != len(contexts):
        raise ValueError("outer-context prediction census differs from outcome authority")
    predictions: list[OuterContextPrediction] = []
    for index, (raw, context) in enumerate(zip(rows, contexts, strict=True)):
        item = _exact_object(
            raw,
            _OUTER_CONTEXT_PREDICTION_FIELDS,
            label=f"authenticated outer-context prediction {index}",
        )
        if (
            type(item["schema_version"]) is not int
            or item["schema_version"] != SCHEMA_VERSION
            or type(item["track_id"]) is not str
            or item["track_id"] != run.track_id
            or type(item["rotation_id"]) is not str
            or item["rotation_id"] != run.rotation.rotation_id
            or type(item["example_id"]) is not str
            or item["example_id"] != context.example_id
            or type(item["sequence_id"]) is not str
            or item["sequence_id"] != context.sequence_id
            or type(item["target"]) is not str
            or item["target"] != context.target
            or type(item["gram"]) is not str
            or item["gram"] != context.gram
            or type(item["fold"]) is not int
            or item["fold"] != context.fold
        ):
            raise ValueError("outer-context prediction differs from outcome-authority keys")
        prediction = OuterContextPrediction(
            run=run,
            context=context,
            probability=_canonical_hex(
                item["probability_hex"],
                label=f"authenticated outer-context prediction {index} probability",
            ),
        )
        if canonical_json_bytes(prediction.document()) != canonical_json_bytes(item):
            raise ValueError("outer-context prediction does not round-trip exactly")
        predictions.append(prediction)
    result = tuple(predictions)
    if payload != canonical_jsonl_bytes(item.document() for item in result):
        raise ValueError("outer-context prediction payload is not exact canonical evidence")
    return result


def _validate_authenticated_outer_sequence_predictions(
    payload: bytes,
    *,
    run: PolicyRunSpec,
    row: UpdateOuterEvidenceIndexRow,
) -> None:
    rows = _strict_jsonl(payload, label="authenticated outer-sequence predictions")
    predictions: list[OuterSequencePrediction] = []
    for index, raw in enumerate(rows):
        item = _exact_object(
            raw,
            _OUTER_SEQUENCE_PREDICTION_FIELDS,
            label=f"authenticated outer-sequence prediction {index}",
        )
        if (
            type(item["schema_version"]) is not int
            or item["schema_version"] != SCHEMA_VERSION
            or type(item["track_id"]) is not str
            or item["track_id"] != run.track_id
            or type(item["rotation_id"]) is not str
            or item["rotation_id"] != run.rotation.rotation_id
            or type(item["sequence_id"]) is not str
        ):
            raise ValueError("outer-sequence prediction has the wrong frozen identity")
        targets = _exact_object(
            item["target_probabilities_hex"],
            set(TARGETS),
            label=f"authenticated outer-sequence prediction {index} targets",
        )
        objectives = _exact_object(
            item["objective_probabilities_hex"],
            set(OBJECTIVES),
            label=f"authenticated outer-sequence prediction {index} objectives",
        )
        prediction = OuterSequencePrediction(
            run=run,
            sequence_id=item["sequence_id"],
            target_probabilities=tuple(
                _canonical_hex(
                    targets[target],
                    label=(f"authenticated outer-sequence prediction {index} target {target}"),
                )
                for target in TARGETS
            ),
            objective_probabilities=tuple(
                _canonical_hex(
                    objectives[objective],
                    label=(
                        f"authenticated outer-sequence prediction {index} objective {objective}"
                    ),
                )
                for objective in OBJECTIVES
            ),
        )
        if canonical_json_bytes(prediction.document()) != canonical_json_bytes(item):
            raise ValueError("outer-sequence prediction does not round-trip exactly")
        predictions.append(prediction)
    result = tuple(predictions)
    identifiers = tuple(item.sequence_id for item in result)
    if (
        len(result) != row.outer_sequence_prediction_count
        or identifiers != tuple(sorted(set(identifiers)))
        or id_stream_sha256(identifiers) != row.outer_candidate_ids_sha256
        or payload != canonical_jsonl_bytes(item.document() for item in result)
    ):
        raise ValueError("outer-sequence evidence differs from update-global authority")


def outer_context_predictions_from_campaign(
    campaign: UpdateCampaignCapability,
    *,
    run: PolicyRunSpec,
    outer_evidence_seal: PhaseSeal,
    outer_outcome_vault: OuterOutcomeVault,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> tuple[OuterContextPrediction, ...]:
    """Open one indexed context-evidence leaf without model or metadata leaves.

    ``outer_outcome_vault`` is expected to have already been authenticated under
    stage/global by the finalization custodian.  It is used here only to bind
    each label-free prediction row to the exact ordered context keys; labels
    are neither copied into nor returned by this capability bridge.
    """

    frozen = _require_frozen_run(run, label="outer-evidence campaign lookup run")
    authorized = verify_update_campaign_capability(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    evidence_row = authorized.outer_evidence_row(run=frozen)
    rows = authorized.index_rows()
    state_matches = tuple(
        item for item in rows if type(item) is UpdateStateIndexRow and item.run == frozen
    )
    component_matches = tuple(
        item
        for item in rows
        if type(item) is UpdateComponentIndexRow and item.spec == frozen.rotation
    )
    view_matches = tuple(
        item for item in rows if type(item) is UpdateOuterViewIndexRow and item.run == frozen
    )
    if len(state_matches) != 1 or len(component_matches) != 1 or len(view_matches) != 1:
        raise ValueError("update campaign lacks exact evidence dependency index rows")
    state_row = state_matches[0]
    component_row = component_matches[0]
    view_row = view_matches[0]
    contexts = _outer_vault_contexts(
        outer_outcome_vault,
        run=frozen,
        row=evidence_row,
    )
    if type(outer_evidence_seal) is not PhaseSeal:
        raise TypeError("outer-evidence materialization requires an exact PhaseSeal")
    verified = verify_phase_capability(
        outer_evidence_seal,
        expected_artifact=OUTER_EVIDENCE_ARTIFACT,
        expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
        expected_seal_sha256=evidence_row.leaf_seal_sha256,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=frozen.track_id,
    )
    if verified.payload_sha256 != evidence_row.payload_sha256:
        raise ValueError("outer-evidence payload digests differ from update-global index")

    # The indexed envelope and payload digests are authenticated before the
    # summary is opened.  That summary carries the intentionally omitted
    # outer-metadata leaf digest needed to reconstruct all six predecessors;
    # prediction payloads remain unread until the predecessor map is exact.
    summary_payload = verified.read_payload_bytes("outer-evidence-summary.json")
    anchors = dict(authorized.anchor_digests)
    metadata_leaf = _outer_evidence_summary(
        _strict_json_object(summary_payload, label="authenticated outer-evidence summary"),
        run=frozen,
        row=evidence_row,
        state_row=state_row,
        component_row=component_row,
        view_row=view_row,
        stage_global_seal_sha256=anchors[_stage_predecessor()],
    )
    expected_predecessors = {
        _protocol_predecessor(): anchors[_protocol_predecessor()],
        _stage_predecessor(): anchors[_stage_predecessor()],
        f"stage/rotations/{frozen.rotation.rotation_id}/outer-metadata/SHA256SUMS": (metadata_leaf),
        f"{_component_relative_path(frozen.rotation)}/SHA256SUMS": (component_row.leaf_seal_sha256),
        f"{_state_relative_path(frozen)}/SHA256SUMS": state_row.leaf_seal_sha256,
        f"{_view_relative_path(frozen)}/SHA256SUMS": view_row.leaf_seal_sha256,
    }
    verify_phase_capability(
        verified,
        expected_artifact=OUTER_EVIDENCE_ARTIFACT,
        expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
        expected_predecessor_seals=expected_predecessors,
        expected_seal_sha256=evidence_row.leaf_seal_sha256,
    )
    context_payload = verified.read_payload_bytes("outer-context-predictions.jsonl")
    sequence_payload = verified.read_payload_bytes("outer-sequence-predictions.jsonl")
    predictions = _decode_authenticated_outer_context_predictions(
        context_payload,
        run=frozen,
        contexts=contexts,
    )
    _validate_authenticated_outer_sequence_predictions(
        sequence_payload,
        run=frozen,
        row=evidence_row,
    )
    return predictions


def outer_view_from_campaign(
    campaign: UpdateCampaignCapability,
    *,
    run: PolicyRunSpec,
    outer_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> tuple[OuterMeanCandidate, ...]:
    """Open exactly one indexed selector view after authenticating update/global."""

    frozen = _require_frozen_run(run, label="outer-view campaign lookup run")
    authorized = verify_update_campaign_capability(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    row = authorized.outer_view_row(run=frozen)
    if type(outer_view_seal) is not PhaseSeal:
        raise TypeError("outer-view materialization requires an exact PhaseSeal")
    verified = verify_phase_capability(
        outer_view_seal,
        expected_artifact=OUTER_VIEW_ARTIFACT,
        expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
        expected_seal_sha256=row.leaf_seal_sha256,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=frozen.track_id,
    )
    if verified.payload_sha256 != row.payload_sha256:
        raise ValueError("outer-view payload digests differ from update-global index")

    # The controller-indexed leaf digest, artifact, inventory, metadata, and
    # payload digests are checked before either view payload is decoded.  The
    # authenticated summary then supplies the deliberately omitted stage-role
    # leaf digest needed to recheck the semantic predecessor map; candidate
    # bytes remain unread until that map verifies.
    summary_payload = verified.read_payload_bytes("view-summary.json")
    metadata_leaf, component_leaf = _outer_view_summary(
        _strict_json_object(summary_payload, label="authenticated outer-view summary"),
        run=frozen,
        row=row,
    )
    anchors = dict(authorized.anchor_digests)
    expected_predecessors = {
        _protocol_predecessor(): anchors[_protocol_predecessor()],
        _stage_predecessor(): anchors[_stage_predecessor()],
        f"stage/rotations/{frozen.rotation.rotation_id}/outer-metadata/SHA256SUMS": (metadata_leaf),
        f"{_component_relative_path(frozen.rotation)}/SHA256SUMS": component_leaf,
        f"{_state_relative_path(frozen)}/SHA256SUMS": row.state_leaf_seal_sha256,
    }
    verify_phase_capability(
        verified,
        expected_artifact=OUTER_VIEW_ARTIFACT,
        expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
        expected_predecessor_seals=expected_predecessors,
        expected_seal_sha256=row.leaf_seal_sha256,
    )
    candidates_payload = verified.read_payload_bytes("candidates.jsonl")
    raw_candidates = _strict_jsonl(
        candidates_payload,
        label="authenticated outer-view candidates",
    )
    candidates = tuple(OuterMeanCandidate.from_mapping(item) for item in raw_candidates)
    identifiers = tuple(item.sequence_id for item in candidates)
    if (
        len(candidates) != row.candidate_count
        or identifiers != tuple(sorted(set(identifiers)))
        or id_stream_sha256(identifiers) != row.candidate_ids_sha256
        or sha256_bytes(candidates_payload) != dict(row.payload_sha256)["candidates.jsonl"]
        or any(
            item.rotation_id != frozen.rotation.rotation_id or item.eligible is not True
            for item in candidates
        )
    ):
        raise ValueError("outer-view candidates differ from update-global authority")
    return candidates


__all__ = [
    "EXPECTED_UPDATE_LEAVES",
    "EXPECTED_UPDATE_PREDECESSORS",
    "UpdateCampaignCapability",
    "UpdateComponentIndexRow",
    "UpdateOuterEvidenceIndexRow",
    "UpdateOuterViewIndexRow",
    "UpdateStateIndexRow",
    "outer_context_predictions_from_campaign",
    "outer_view_from_campaign",
    "publish_update_campaign_barrier",
    "update_campaign_capability_from_seal",
    "verify_update_campaign_barrier",
    "verify_update_campaign_capability",
]

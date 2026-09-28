"""Sealed, authority-gated update-state leaves for sequential v2.

One state worker may see exactly one authenticated base-update leaf and one
authenticated selected-reveal leaf.  It publishes only the resulting model,
an ordered training-membership ledger, and a digest-rich summary.  Projection
workers later recover a raw-label-free :class:`UpdateStateCapability` only
after re-anchoring the state leaf through every upstream global authority and
an independently supplied controller digest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    BASE_UPDATE_ROLE,
    BaseUpdateCapability,
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    base_update_from_campaign,
    descriptor_logistic_state_from_document,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_CONTEXTS_BY_FOLD,
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    policy_run_by_track_id,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    RevealCampaignCapability,
    RevealIndexRow,
    SelectedPoolRevealCapability,
    pool_reveal_from_campaign,
    verify_reveal_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    RECEIPT_NAME,
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    checksum_manifest_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_stage import StageManifestCapability
from amp_challenge.evaluation.sequential_v2_update import (
    UPDATE_STATE_ARTIFACT,
    UPDATE_STATE_PAYLOAD_PATHS,
    PolicyUpdateState,
    UpdateStateCapability,
    fit_policy_update_state,
    id_stream_sha256,
)

SCHEMA_VERSION = 1
UPDATE_STATE_SUMMARY_ARTIFACT = "sequential_v2_update_state_summary_v1"
UPDATE_STATE_ATTESTATION_ARTIFACT = "sequential_v2_update_state_attestation_v1"

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


def _exact_object(
    value: object,
    fields: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} keys must be exact text")
    return value


def _strict_json(payload: bytes, *, label: str) -> object:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be LF-terminated canonical JSON")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate object key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
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


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    if value != rotation_by_id(value.rotation_id):
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if type(value.policy) is not str or (value.seed is not None and type(value.seed) is not int):
        raise TypeError(f"{label} policy and seed must use exact scalar types")
    if value != policy_run_by_track_id(value.track_id):
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _run_from_document(value: object) -> PolicyRunSpec:
    document = _exact_object(
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
        label="update-state policy run",
    )
    if type(document["track_id"]) is not str:
        raise ValueError("update-state track ID must be exact text")
    run = policy_run_by_track_id(document["track_id"])
    if canonical_json_bytes(document) != canonical_json_bytes(run.document()):
        raise ValueError("update-state policy run differs from the frozen registry")
    return run


def _identity_document(identity: SequentialV2PublicationIdentity) -> dict[str, object]:
    if type(identity) is not SequentialV2PublicationIdentity:
        raise TypeError("update-state boundary requires an exact publication identity")
    return {
        "git_commit": identity.git_commit,
        "code_manifest_sha256": identity.code_manifest_sha256,
        "config_sha256": identity.config_sha256,
        "lock_sha256": identity.lock_sha256,
    }


def _identity_from_document(value: object) -> SequentialV2PublicationIdentity:
    document = _exact_object(
        value,
        {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"},
        label="update-state publication identity",
    )
    if any(type(document[key]) is not str for key in document):
        raise ValueError("update-state publication identity fields must be exact text")
    if _GIT_COMMIT.fullmatch(document["git_commit"]) is None:
        raise ValueError("update-state publication git commit is invalid")
    return SequentialV2PublicationIdentity(
        git_commit=document["git_commit"],
        code_manifest_sha256=document["code_manifest_sha256"],
        config_sha256=document["config_sha256"],
        lock_sha256=document["lock_sha256"],
    )


def update_state_relative_path(run: PolicyRunSpec) -> str:
    frozen = _require_frozen_run(run, label="update-state path run")
    return f"update/tracks/{frozen.track_id}/state"


def _protocol_predecessor() -> str:
    return "protocol/SHA256SUMS"


def _prepare_predecessor() -> str:
    return "prepare/global/SHA256SUMS"


def _base_predecessor(run: PolicyRunSpec) -> str:
    return f"prepare/rotations/{run.rotation.rotation_id}/base-update/SHA256SUMS"


def _reveal_predecessor() -> str:
    return "reveal/global/SHA256SUMS"


def _reveal_leaf_predecessor(run: PolicyRunSpec) -> str:
    return f"reveal/tracks/{run.track_id}/SHA256SUMS"


def _state_predecessors(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    prepare_global_seal_sha256: str,
    base_update_leaf_seal_sha256: str,
    reveal_global_seal_sha256: str,
    reveal_leaf_seal_sha256: str,
) -> dict[str, str]:
    frozen = _require_frozen_run(run, label="update-state predecessor run")
    result = {
        _protocol_predecessor(): _sha256(
            protocol_seal_sha256,
            label="update-state protocol seal",
        ),
        _prepare_predecessor(): _sha256(
            prepare_global_seal_sha256,
            label="update-state prepare-global seal",
        ),
        _base_predecessor(frozen): _sha256(
            base_update_leaf_seal_sha256,
            label="update-state base-update leaf seal",
        ),
        _reveal_predecessor(): _sha256(
            reveal_global_seal_sha256,
            label="update-state reveal-global seal",
        ),
        _reveal_leaf_predecessor(frozen): _sha256(
            reveal_leaf_seal_sha256,
            label="update-state reveal leaf seal",
        ),
    }
    if len(result) != 5:
        raise AssertionError("update-state predecessor census changed")
    return result


def _payload_digest(seal: PhaseSeal, path: str, *, label: str) -> str:
    if type(seal) is not PhaseSeal:
        raise TypeError(f"{label} requires an exact PhaseSeal")
    matches = tuple(digest for current, digest in seal.payload_sha256 if current == path)
    if len(matches) != 1:
        raise ValueError(f"{label} lacks one exact payload digest for {path}")
    return _sha256(matches[0], label=f"{label} payload {path}")


def _validate_payload_digests(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in value
    ):
        raise ValueError("update-state payload digests must be an exact immutable map")
    result = value
    if tuple(path for path, _digest in result) != UPDATE_STATE_PAYLOAD_PATHS or len(
        dict(result)
    ) != len(result):
        raise ValueError("update-state payload digest inventory changed")
    for path, digest in result:
        _sha256(digest, label=f"update-state payload {path}")
    return result


def _training_id_payload(training_ids: tuple[str, ...]) -> bytes:
    return canonical_jsonl_bytes({"example_id": value} for value in training_ids)


def _training_ids_from_payload(payload: bytes) -> tuple[str, ...]:
    rows = _strict_jsonl(payload, label="update-state training example IDs")
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        exact = _exact_object(
            row,
            {"example_id"},
            label=f"update-state training ID row {index}",
        )
        identifiers.append(_sha256(exact["example_id"], label=f"training example ID {index}"))
    result = tuple(identifiers)
    if result != tuple(sorted(set(result))):
        raise ValueError("update-state training example IDs must be ascending and unique")
    return result


def _summary_document(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    prepare_global_seal_sha256: str,
    base_update_leaf_seal_sha256: str,
    base_contexts_payload_sha256: str,
    base_model_payload_sha256: str,
    base_model_state_sha256: str,
    reveal_global_seal_sha256: str,
    reveal_leaf_seal_sha256: str,
    reveal_contexts_payload_sha256: str,
    base_context_count: int,
    base_example_ids_sha256: str,
    revealed_context_count: int,
    revealed_example_ids_sha256: str,
    training_context_count: int,
    training_example_ids_sha256: str,
    training_example_ids_payload_sha256: str,
    refit: bool,
    updated_model_payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": UPDATE_STATE_SUMMARY_ARTIFACT,
        "run": _require_frozen_run(run, label="update-state summary run").document(),
        "protocol_seal_sha256": protocol_seal_sha256,
        "prepare_global_seal_sha256": prepare_global_seal_sha256,
        "base_update_leaf_seal_sha256": base_update_leaf_seal_sha256,
        "base_contexts_payload_sha256": base_contexts_payload_sha256,
        "base_model_payload_sha256": base_model_payload_sha256,
        "base_model_state_sha256": base_model_state_sha256,
        "reveal_global_seal_sha256": reveal_global_seal_sha256,
        "reveal_leaf_seal_sha256": reveal_leaf_seal_sha256,
        "reveal_contexts_payload_sha256": reveal_contexts_payload_sha256,
        "base_context_count": base_context_count,
        "base_example_ids_sha256": base_example_ids_sha256,
        "revealed_context_count": revealed_context_count,
        "revealed_example_ids_sha256": revealed_example_ids_sha256,
        "training_context_count": training_context_count,
        "training_example_ids_sha256": training_example_ids_sha256,
        "training_example_ids_payload_sha256": training_example_ids_payload_sha256,
        "refit": refit,
        "updated_model_payload_sha256": updated_model_payload_sha256,
    }


@dataclass(frozen=True, slots=True)
class UpdateStateAttestation:
    """Canonical payload-free result from one isolated state worker."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    prepare_global_seal_sha256: str
    base_update_leaf_seal_sha256: str
    base_contexts_payload_sha256: str
    base_model_payload_sha256: str
    base_model_state_sha256: str
    reveal_global_seal_sha256: str
    reveal_leaf_seal_sha256: str
    reveal_contexts_payload_sha256: str
    state_leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    base_context_count: int
    base_example_ids_sha256: str
    revealed_context_count: int
    revealed_example_ids_sha256: str
    training_context_count: int
    training_example_ids_sha256: str
    refit: bool

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="update-state attestation run")
        _identity_document(self.publication_identity)
        for label, value in (
            ("protocol seal", self.protocol_seal_sha256),
            ("prepare-global seal", self.prepare_global_seal_sha256),
            ("base-update leaf seal", self.base_update_leaf_seal_sha256),
            ("base contexts payload", self.base_contexts_payload_sha256),
            ("base model payload", self.base_model_payload_sha256),
            ("base model state", self.base_model_state_sha256),
            ("reveal-global seal", self.reveal_global_seal_sha256),
            ("reveal leaf seal", self.reveal_leaf_seal_sha256),
            ("reveal contexts payload", self.reveal_contexts_payload_sha256),
            ("state leaf seal", self.state_leaf_seal_sha256),
            ("base example IDs", self.base_example_ids_sha256),
            ("revealed example IDs", self.revealed_example_ids_sha256),
            ("training example IDs", self.training_example_ids_sha256),
        ):
            _sha256(value, label=f"update-state attestation {label}")
        payloads = dict(_validate_payload_digests(self.payload_sha256))
        base_count = _exact_int(
            self.base_context_count,
            label="update-state attestation base context count",
            minimum=1,
        )
        expected_base_count = sum(
            EXPECTED_CONTEXTS_BY_FOLD[fold] for fold in run.rotation.base_folds
        )
        if base_count != expected_base_count:
            raise ValueError("update-state base context count differs from frozen base folds")
        revealed_count = _exact_int(
            self.revealed_context_count,
            label="update-state attestation revealed context count",
        )
        training_count = _exact_int(
            self.training_context_count,
            label="update-state attestation training context count",
            minimum=1,
        )
        if training_count != base_count + revealed_count:
            raise ValueError("update-state training count must equal base plus revealed")
        if type(self.refit) is not bool or self.refit != run.refit:
            raise ValueError("update-state refit flag differs from the frozen run")
        empty_digest = id_stream_sha256((), allow_empty=True)
        if run.policy == NO_QUERY:
            if (
                revealed_count != 0
                or self.revealed_example_ids_sha256 != empty_digest
                or self.reveal_contexts_payload_sha256 != sha256_bytes(b"")
                or self.training_example_ids_sha256 != self.base_example_ids_sha256
                or payloads["updated-model.json"] != self.base_model_state_sha256
            ):
                raise ValueError(
                    "no-query update-state attestation must preserve the exact base state"
                )
        elif revealed_count <= 0:
            raise ValueError("refitted update-state attestation requires a nonempty reveal")

        expected_summary = _summary_document(
            run=run,
            protocol_seal_sha256=self.protocol_seal_sha256,
            prepare_global_seal_sha256=self.prepare_global_seal_sha256,
            base_update_leaf_seal_sha256=self.base_update_leaf_seal_sha256,
            base_contexts_payload_sha256=self.base_contexts_payload_sha256,
            base_model_payload_sha256=self.base_model_payload_sha256,
            base_model_state_sha256=self.base_model_state_sha256,
            reveal_global_seal_sha256=self.reveal_global_seal_sha256,
            reveal_leaf_seal_sha256=self.reveal_leaf_seal_sha256,
            reveal_contexts_payload_sha256=self.reveal_contexts_payload_sha256,
            base_context_count=base_count,
            base_example_ids_sha256=self.base_example_ids_sha256,
            revealed_context_count=revealed_count,
            revealed_example_ids_sha256=self.revealed_example_ids_sha256,
            training_context_count=training_count,
            training_example_ids_sha256=self.training_example_ids_sha256,
            training_example_ids_payload_sha256=payloads["training-example-ids.jsonl"],
            refit=self.refit,
            updated_model_payload_sha256=payloads["updated-model.json"],
        )
        if payloads["update-summary.json"] != sha256_bytes(canonical_json_bytes(expected_summary)):
            raise ValueError("update-state attestation census differs from its summary digest")
        if _attested_state_leaf_seal_sha256(self) != self.state_leaf_seal_sha256:
            raise ValueError("update-state attestation does not reconstruct its authoritative leaf")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": UPDATE_STATE_ATTESTATION_ARTIFACT,
            "run": self.run.document(),
            "publication_identity": _identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "prepare_global_seal_sha256": self.prepare_global_seal_sha256,
            "base_update_leaf_seal_sha256": self.base_update_leaf_seal_sha256,
            "base_contexts_payload_sha256": self.base_contexts_payload_sha256,
            "base_model_payload_sha256": self.base_model_payload_sha256,
            "base_model_state_sha256": self.base_model_state_sha256,
            "reveal_global_seal_sha256": self.reveal_global_seal_sha256,
            "reveal_leaf_seal_sha256": self.reveal_leaf_seal_sha256,
            "reveal_contexts_payload_sha256": self.reveal_contexts_payload_sha256,
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "base_context_count": self.base_context_count,
            "base_example_ids_sha256": self.base_example_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "training_context_count": self.training_context_count,
            "training_example_ids_sha256": self.training_example_ids_sha256,
            "refit": self.refit,
        }

    def index_document(self) -> dict[str, object]:
        """Project the exact payload-free update-global state index row."""

        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "state",
            "run": self.run.document(),
            "relative_path": update_state_relative_path(self.run),
            "leaf_artifact": UPDATE_STATE_ARTIFACT,
            "leaf_seal_sha256": self.state_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "base_context_count": self.base_context_count,
            "base_example_ids_sha256": self.base_example_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "training_context_count": self.training_context_count,
            "training_example_ids_sha256": self.training_example_ids_sha256,
            "refit": self.refit,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())


def _attested_state_leaf_seal_sha256(attestation: UpdateStateAttestation) -> str:
    if type(attestation) is not UpdateStateAttestation:
        raise TypeError("state leaf reconstruction requires an exact attestation")
    payloads = dict(_validate_payload_digests(attestation.payload_sha256))
    receipt = canonical_json_bytes(
        {
            "artifact": UPDATE_STATE_ARTIFACT,
            "metadata": attestation.publication_identity.metadata(
                phase="update",
                scope_id=attestation.run.track_id,
            ),
            "payloads": payloads,
            "predecessor_seals": _state_predecessors(
                run=attestation.run,
                protocol_seal_sha256=attestation.protocol_seal_sha256,
                prepare_global_seal_sha256=attestation.prepare_global_seal_sha256,
                base_update_leaf_seal_sha256=attestation.base_update_leaf_seal_sha256,
                reveal_global_seal_sha256=attestation.reveal_global_seal_sha256,
                reveal_leaf_seal_sha256=attestation.reveal_leaf_seal_sha256,
            ),
            "schema_version": SCHEMA_VERSION,
            "status": "sealed",
        }
    )
    return sha256_bytes(checksum_manifest_bytes({**payloads, RECEIPT_NAME: sha256_bytes(receipt)}))


def update_state_attestation_from_document(value: object) -> UpdateStateAttestation:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "publication_identity",
            "protocol_seal_sha256",
            "prepare_global_seal_sha256",
            "base_update_leaf_seal_sha256",
            "base_contexts_payload_sha256",
            "base_model_payload_sha256",
            "base_model_state_sha256",
            "reveal_global_seal_sha256",
            "reveal_leaf_seal_sha256",
            "reveal_contexts_payload_sha256",
            "state_leaf_seal_sha256",
            "payload_sha256",
            "base_context_count",
            "base_example_ids_sha256",
            "revealed_context_count",
            "revealed_example_ids_sha256",
            "training_context_count",
            "training_example_ids_sha256",
            "refit",
        },
        label="update-state attestation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["artifact"]) is not str
        or raw["artifact"] != UPDATE_STATE_ATTESTATION_ARTIFACT
        or type(raw["refit"]) is not bool
    ):
        raise ValueError("update-state attestation identity or refit type changed")
    payload_map = _exact_object(
        raw["payload_sha256"],
        set(UPDATE_STATE_PAYLOAD_PATHS),
        label="update-state attestation payload map",
    )
    payloads = tuple(
        (path, _sha256(payload_map[path], label=f"update-state attestation payload {path}"))
        for path in UPDATE_STATE_PAYLOAD_PATHS
    )
    attestation = UpdateStateAttestation(
        run=_run_from_document(raw["run"]),
        publication_identity=_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(raw["protocol_seal_sha256"], label="protocol seal"),
        prepare_global_seal_sha256=_sha256(
            raw["prepare_global_seal_sha256"], label="prepare-global seal"
        ),
        base_update_leaf_seal_sha256=_sha256(
            raw["base_update_leaf_seal_sha256"], label="base-update leaf seal"
        ),
        base_contexts_payload_sha256=_sha256(
            raw["base_contexts_payload_sha256"], label="base contexts payload"
        ),
        base_model_payload_sha256=_sha256(
            raw["base_model_payload_sha256"], label="base model payload"
        ),
        base_model_state_sha256=_sha256(raw["base_model_state_sha256"], label="base model state"),
        reveal_global_seal_sha256=_sha256(
            raw["reveal_global_seal_sha256"], label="reveal-global seal"
        ),
        reveal_leaf_seal_sha256=_sha256(raw["reveal_leaf_seal_sha256"], label="reveal leaf seal"),
        reveal_contexts_payload_sha256=_sha256(
            raw["reveal_contexts_payload_sha256"], label="reveal contexts payload"
        ),
        state_leaf_seal_sha256=_sha256(raw["state_leaf_seal_sha256"], label="state leaf seal"),
        payload_sha256=payloads,
        base_context_count=_exact_int(
            raw["base_context_count"], label="base context count", minimum=1
        ),
        base_example_ids_sha256=_sha256(raw["base_example_ids_sha256"], label="base example IDs"),
        revealed_context_count=_exact_int(
            raw["revealed_context_count"], label="revealed context count"
        ),
        revealed_example_ids_sha256=_sha256(
            raw["revealed_example_ids_sha256"], label="revealed example IDs"
        ),
        training_context_count=_exact_int(
            raw["training_context_count"], label="training context count", minimum=1
        ),
        training_example_ids_sha256=_sha256(
            raw["training_example_ids_sha256"], label="training example IDs"
        ),
        refit=raw["refit"],
    )
    if canonical_json_bytes(attestation.document()) != canonical_json_bytes(raw):
        raise ValueError("update-state attestation does not reconstruct its exact document")
    return attestation


def update_state_attestation_from_bytes(payload: bytes) -> UpdateStateAttestation:
    return update_state_attestation_from_document(
        _strict_json_object(payload, label="update-state attestation")
    )


@dataclass(frozen=True, slots=True)
class _AuthenticatedGlobals:
    protocol: ProtocolCapability
    prepare: PrepareCampaignCapability
    reveal: RevealCampaignCapability


def _authenticate_upstream_globals(
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> _AuthenticatedGlobals:
    _identity_document(publication_identity)
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("update-state boundary requires an exact ProtocolCapability")
    if type(prepare_campaign) is not PrepareCampaignCapability:
        raise TypeError("update-state boundary requires an exact PrepareCampaignCapability")
    if type(reveal_campaign) is not RevealCampaignCapability:
        raise TypeError("update-state boundary requires an exact RevealCampaignCapability")
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("update-state boundary requires an exact StageManifestCapability")
    if type(selection_barrier) is not PhaseSeal:
        raise TypeError("update-state boundary requires an exact select-global PhaseSeal")

    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    prepare_digest = _sha256(
        expected_prepare_campaign_seal_sha256,
        label="expected prepare campaign seal",
    )
    prepare = verify_prepare_campaign_capability(
        prepare_campaign,
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=prepare_digest,
        expected_protocol_seal_sha256=protocol.seal.seal_sha256,
    )
    reveal = verify_reveal_campaign_capability(
        reveal_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        expected_stage_global_seal_sha256=_sha256(
            expected_stage_global_seal_sha256,
            label="expected stage-global seal",
        ),
        expected_selection_barrier_seal_sha256=_sha256(
            expected_selection_barrier_seal_sha256,
            label="expected selection-global seal",
        ),
        expected_reveal_campaign_seal_sha256=_sha256(
            expected_reveal_campaign_seal_sha256,
            label="expected reveal-global seal",
        ),
    )
    return _AuthenticatedGlobals(protocol=protocol, prepare=prepare, reveal=reveal)


def _materialize_update_inputs(
    *,
    run: PolicyRunSpec,
    globals_: _AuthenticatedGlobals,
    base_update_seal: PhaseSeal,
    reveal_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> tuple[BaseUpdateCapability, SelectedPoolRevealCapability, PolicyUpdateState]:
    frozen = _require_frozen_run(run, label="update-state worker run")
    if type(base_update_seal) is not PhaseSeal or type(reveal_seal) is not PhaseSeal:
        raise TypeError("update-state worker inputs must be exact rootless PhaseSeals")
    base = base_update_from_campaign(
        globals_.prepare,
        spec=frozen.rotation,
        base_update_seal=base_update_seal,
        publication_identity=publication_identity,
        protocol_capability=globals_.protocol,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    reveal = pool_reveal_from_campaign(
        globals_.reveal,
        run=frozen,
        reveal_seal=reveal_seal,
        publication_identity=publication_identity,
        protocol_capability=globals_.protocol,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    if (
        base.base_update_seal_sha256 != base_update_seal.seal_sha256
        or reveal.reveal_leaf_seal_sha256 != reveal_seal.seal_sha256
        or reveal.reveal_campaign_seal_sha256 != globals_.reveal.seal.seal_sha256
    ):
        raise ValueError("update-state materialized inputs differ from their leaf authorities")
    state = fit_policy_update_state(
        run=frozen,
        base_update=base,
        selected_reveal=reveal,
    )
    return base, reveal, state


def _authenticated_base_payload_provenance(
    *,
    spec: RotationSpec,
    base: BaseUpdateCapability,
    base_update_seal: PhaseSeal,
    prepare_campaign: PrepareCampaignCapability,
) -> tuple[str, str, str]:
    """Match the opened base leaf to all prepare-global payload authorities."""

    base_contexts_digest = _payload_digest(
        base_update_seal,
        "base-contexts.jsonl",
        label="authenticated base-update leaf",
    )
    base_model_digest = _payload_digest(
        base_update_seal,
        "base-model.json",
        label="authenticated base-update leaf",
    )
    base_model_state_digest = sha256_bytes(canonical_json_bytes(base.base_model.document()))
    expected = (
        prepare_campaign.leaf_payload_sha256(
            spec=spec,
            role=BASE_UPDATE_ROLE,
            path="base-contexts.jsonl",
        ),
        prepare_campaign.leaf_payload_sha256(
            spec=spec,
            role=BASE_UPDATE_ROLE,
            path="base-model.json",
        ),
        prepare_campaign.base_model_state_sha256(spec=spec),
    )
    result = (base_contexts_digest, base_model_digest, base_model_state_digest)
    if result != expected:
        raise ValueError("authenticated base-update payload provenance differs from prepare-global")
    return result


def _state_payloads(
    state: PolicyUpdateState,
    *,
    protocol_seal_sha256: str,
    prepare_global_seal_sha256: str,
    base_contexts_payload_sha256: str,
    base_model_payload_sha256: str,
    base_model_state_sha256: str,
    reveal_global_seal_sha256: str,
    reveal_contexts_payload_sha256: str,
) -> dict[str, bytes]:
    training_payload = _training_id_payload(state.training_example_ids)
    model_payload = canonical_json_bytes(state.model.document())
    summary = _summary_document(
        run=state.run,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_global_seal_sha256=prepare_global_seal_sha256,
        base_update_leaf_seal_sha256=state.base_update_seal_sha256,
        base_contexts_payload_sha256=base_contexts_payload_sha256,
        base_model_payload_sha256=base_model_payload_sha256,
        base_model_state_sha256=base_model_state_sha256,
        reveal_global_seal_sha256=reveal_global_seal_sha256,
        reveal_leaf_seal_sha256=state.reveal_leaf_seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_payload_sha256,
        base_context_count=len(state.base_example_ids),
        base_example_ids_sha256=id_stream_sha256(state.base_example_ids),
        revealed_context_count=len(state.revealed_example_ids),
        revealed_example_ids_sha256=id_stream_sha256(
            state.revealed_example_ids,
            allow_empty=True,
        ),
        training_context_count=len(state.training_example_ids),
        training_example_ids_sha256=id_stream_sha256(state.training_example_ids),
        training_example_ids_payload_sha256=sha256_bytes(training_payload),
        refit=state.refit,
        updated_model_payload_sha256=sha256_bytes(model_payload),
    )
    return {
        "training-example-ids.jsonl": training_payload,
        "update-summary.json": canonical_json_bytes(summary),
        "updated-model.json": model_payload,
    }


def _attestation_from_state(
    seal: PhaseSeal,
    state: PolicyUpdateState,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    prepare_global_seal_sha256: str,
    base_contexts_payload_sha256: str,
    base_model_payload_sha256: str,
    base_model_state_sha256: str,
    reveal_global_seal_sha256: str,
    reveal_contexts_payload_sha256: str,
) -> UpdateStateAttestation:
    return UpdateStateAttestation(
        run=state.run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_global_seal_sha256=prepare_global_seal_sha256,
        base_update_leaf_seal_sha256=state.base_update_seal_sha256,
        base_contexts_payload_sha256=base_contexts_payload_sha256,
        base_model_payload_sha256=base_model_payload_sha256,
        base_model_state_sha256=base_model_state_sha256,
        reveal_global_seal_sha256=reveal_global_seal_sha256,
        reveal_leaf_seal_sha256=state.reveal_leaf_seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_payload_sha256,
        state_leaf_seal_sha256=seal.seal_sha256,
        payload_sha256=seal.payload_sha256,
        base_context_count=len(state.base_example_ids),
        base_example_ids_sha256=id_stream_sha256(state.base_example_ids),
        revealed_context_count=len(state.revealed_example_ids),
        revealed_example_ids_sha256=id_stream_sha256(
            state.revealed_example_ids,
            allow_empty=True,
        ),
        training_context_count=len(state.training_example_ids),
        training_example_ids_sha256=id_stream_sha256(state.training_example_ids),
        refit=state.refit,
    )


def _verify_leaf_against_state(
    seal: PhaseSeal,
    state: PolicyUpdateState,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    prepare_global_seal_sha256: str,
    base_contexts_payload_sha256: str,
    base_model_payload_sha256: str,
    base_model_state_sha256: str,
    reveal_global_seal_sha256: str,
    reveal_contexts_payload_sha256: str,
    expected_state_leaf_seal_sha256: str,
) -> UpdateStateAttestation:
    expected_payloads = _state_payloads(
        state,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_global_seal_sha256=prepare_global_seal_sha256,
        base_contexts_payload_sha256=base_contexts_payload_sha256,
        base_model_payload_sha256=base_model_payload_sha256,
        base_model_state_sha256=base_model_state_sha256,
        reveal_global_seal_sha256=reveal_global_seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_payload_sha256,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=UPDATE_STATE_ARTIFACT,
        expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
        expected_predecessor_seals=_state_predecessors(
            run=state.run,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_global_seal_sha256=prepare_global_seal_sha256,
            base_update_leaf_seal_sha256=state.base_update_seal_sha256,
            reveal_global_seal_sha256=reveal_global_seal_sha256,
            reveal_leaf_seal_sha256=state.reveal_leaf_seal_sha256,
        ),
        expected_seal_sha256=_sha256(
            expected_state_leaf_seal_sha256,
            label="expected update-state leaf seal",
        ),
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=state.run.track_id,
    )
    for path, expected in expected_payloads.items():
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"update-state payload differs from authenticated computation: {path}")
    return _attestation_from_state(
        verified,
        state,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_global_seal_sha256=prepare_global_seal_sha256,
        base_contexts_payload_sha256=base_contexts_payload_sha256,
        base_model_payload_sha256=base_model_payload_sha256,
        base_model_state_sha256=base_model_state_sha256,
        reveal_global_seal_sha256=reveal_global_seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_payload_sha256,
    )


def publish_update_state(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    base_update_seal: PhaseSeal,
    reveal_campaign: RevealCampaignCapability,
    reveal_seal: PhaseSeal,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> UpdateStateAttestation:
    """Publish one state after globally authorizing both label-bearing inputs."""

    frozen = _require_frozen_run(run, label="update-state publisher run")
    # This complete authentication deliberately precedes either source bridge.
    globals_ = _authenticate_upstream_globals(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    base, _reveal, state = _materialize_update_inputs(
        run=frozen,
        globals_=globals_,
        base_update_seal=base_update_seal,
        reveal_seal=reveal_seal,
        publication_identity=publication_identity,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    base_contexts_digest, base_model_digest, base_model_state_digest = (
        _authenticated_base_payload_provenance(
            spec=frozen.rotation,
            base=base,
            base_update_seal=base_update_seal,
            prepare_campaign=globals_.prepare,
        )
    )
    reveal_contexts_digest = _payload_digest(
        reveal_seal,
        "contexts.jsonl",
        label="authenticated reveal leaf",
    )
    payloads = _state_payloads(
        state,
        protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
        prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
        base_contexts_payload_sha256=base_contexts_digest,
        base_model_payload_sha256=base_model_digest,
        base_model_state_sha256=base_model_state_digest,
        reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_digest,
    )
    seal = publish_phase(
        destination,
        artifact=UPDATE_STATE_ARTIFACT,
        payloads=payloads,
        predecessor_seals=_state_predecessors(
            run=frozen,
            protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
            prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
            base_update_leaf_seal_sha256=state.base_update_seal_sha256,
            reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
            reveal_leaf_seal_sha256=state.reveal_leaf_seal_sha256,
        ),
        metadata=publication_identity.metadata(phase="update", scope_id=frozen.track_id),
    )
    return _verify_leaf_against_state(
        seal,
        state,
        publication_identity=publication_identity,
        protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
        prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
        base_contexts_payload_sha256=base_contexts_digest,
        base_model_payload_sha256=base_model_digest,
        base_model_state_sha256=base_model_state_digest,
        reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
        reveal_contexts_payload_sha256=reveal_contexts_digest,
        expected_state_leaf_seal_sha256=seal.seal_sha256,
    )


def verify_update_state_phase_capability(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    base_update_seal: PhaseSeal,
    reveal_campaign: RevealCampaignCapability,
    reveal_seal: PhaseSeal,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
) -> UpdateStateAttestation:
    """Recompute and authenticate one state while its two sources are isolated."""

    frozen = _require_frozen_run(run, label="update-state verifier run")
    globals_ = _authenticate_upstream_globals(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    base, _reveal, state = _materialize_update_inputs(
        run=frozen,
        globals_=globals_,
        base_update_seal=base_update_seal,
        reveal_seal=reveal_seal,
        publication_identity=publication_identity,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    base_contexts_digest, base_model_digest, base_model_state_digest = (
        _authenticated_base_payload_provenance(
            spec=frozen.rotation,
            base=base,
            base_update_seal=base_update_seal,
            prepare_campaign=globals_.prepare,
        )
    )
    return _verify_leaf_against_state(
        seal,
        state,
        publication_identity=publication_identity,
        protocol_seal_sha256=globals_.protocol.seal.seal_sha256,
        prepare_global_seal_sha256=globals_.prepare.seal.seal_sha256,
        base_contexts_payload_sha256=base_contexts_digest,
        base_model_payload_sha256=base_model_digest,
        base_model_state_sha256=base_model_state_digest,
        reveal_global_seal_sha256=globals_.reveal.seal.seal_sha256,
        reveal_contexts_payload_sha256=_payload_digest(
            reveal_seal,
            "contexts.jsonl",
            label="authenticated reveal leaf",
        ),
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
    )


def _validated_attestation(value: object) -> UpdateStateAttestation:
    if type(value) is not UpdateStateAttestation:
        raise TypeError("state authority requires an exact UpdateStateAttestation")
    reconstructed = update_state_attestation_from_document(value.document())
    if reconstructed != value:
        raise ValueError("update-state attestation changed during strict reconstruction")
    return reconstructed


def _cross_check_attestation_authorities(
    attestation: UpdateStateAttestation,
    *,
    run: PolicyRunSpec,
    globals_: _AuthenticatedGlobals,
    expected_state_leaf_seal_sha256: str,
) -> RevealIndexRow:
    expected_state = _sha256(
        expected_state_leaf_seal_sha256,
        label="controller-authoritative update-state leaf seal",
    )
    reveal_row = globals_.reveal.index_row(run=run)
    expected_base = globals_.prepare.leaf_seal_sha256(
        spec=run.rotation,
        role=BASE_UPDATE_ROLE,
    )
    expected_base_contexts_payload = globals_.prepare.leaf_payload_sha256(
        spec=run.rotation,
        role=BASE_UPDATE_ROLE,
        path="base-contexts.jsonl",
    )
    expected_base_model_payload = globals_.prepare.leaf_payload_sha256(
        spec=run.rotation,
        role=BASE_UPDATE_ROLE,
        path="base-model.json",
    )
    expected_base_model_state = globals_.prepare.base_model_state_sha256(spec=run.rotation)
    if (
        attestation.run != run
        or attestation.publication_identity != globals_.prepare.publication_identity
        or attestation.protocol_seal_sha256 != globals_.protocol.seal.seal_sha256
        or attestation.prepare_global_seal_sha256 != globals_.prepare.seal.seal_sha256
        or attestation.base_update_leaf_seal_sha256 != expected_base
        or attestation.base_contexts_payload_sha256 != expected_base_contexts_payload
        or attestation.base_model_payload_sha256 != expected_base_model_payload
        or attestation.base_model_state_sha256 != expected_base_model_state
        or attestation.reveal_global_seal_sha256 != globals_.reveal.seal.seal_sha256
        or attestation.reveal_leaf_seal_sha256 != reveal_row.leaf_seal_sha256
        or attestation.revealed_context_count != reveal_row.revealed_context_count
        or attestation.revealed_example_ids_sha256 != reveal_row.revealed_example_ids_sha256
        or attestation.state_leaf_seal_sha256 != expected_state
    ):
        raise ValueError("update-state attestation differs from controller global authority")
    return reveal_row


def update_state_from_authorities(
    state_seal: PhaseSeal,
    *,
    attestation: UpdateStateAttestation,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
) -> UpdateStateCapability:
    """Authorize one raw-label-free state without reopening either source leaf."""

    frozen = _require_frozen_run(run, label="update-state materialization run")
    if type(state_seal) is not PhaseSeal:
        raise TypeError("update-state materialization requires an exact PhaseSeal")

    # Global and procedural authorities are exhausted before any state payload
    # is consulted.  Neither label-bearing source leaf is accepted by this API.
    globals_ = _authenticate_upstream_globals(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    item = _validated_attestation(attestation)
    _cross_check_attestation_authorities(
        item,
        run=frozen,
        globals_=globals_,
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
    )
    verified = verify_phase_capability(
        state_seal,
        expected_artifact=UPDATE_STATE_ARTIFACT,
        expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
        expected_predecessor_seals=_state_predecessors(
            run=frozen,
            protocol_seal_sha256=item.protocol_seal_sha256,
            prepare_global_seal_sha256=item.prepare_global_seal_sha256,
            base_update_leaf_seal_sha256=item.base_update_leaf_seal_sha256,
            reveal_global_seal_sha256=item.reveal_global_seal_sha256,
            reveal_leaf_seal_sha256=item.reveal_leaf_seal_sha256,
        ),
        expected_seal_sha256=item.state_leaf_seal_sha256,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=frozen.track_id,
    )
    if verified.payload_sha256 != item.payload_sha256:
        raise ValueError("update-state leaf payload digests differ from its attestation")

    # This is the first state-leaf payload read.
    training_payload = verified.read_payload_bytes("training-example-ids.jsonl")
    training_ids = _training_ids_from_payload(training_payload)
    payloads = dict(item.payload_sha256)
    if (
        len(training_ids) != item.training_context_count
        or id_stream_sha256(training_ids) != item.training_example_ids_sha256
        or sha256_bytes(training_payload) != payloads["training-example-ids.jsonl"]
    ):
        raise ValueError("update-state training ledger differs from its attestation")

    model_payload = verified.read_payload_bytes("updated-model.json")
    model_document = _strict_json_object(model_payload, label="updated model state")
    model = descriptor_logistic_state_from_document(model_document)
    if (
        canonical_json_bytes(model.document()) != model_payload
        or sha256_bytes(model_payload) != payloads["updated-model.json"]
        or model.training_contexts != item.training_context_count
    ):
        raise ValueError("updated model differs from the authenticated state ledger")

    expected_summary = _summary_document(
        run=frozen,
        protocol_seal_sha256=item.protocol_seal_sha256,
        prepare_global_seal_sha256=item.prepare_global_seal_sha256,
        base_update_leaf_seal_sha256=item.base_update_leaf_seal_sha256,
        base_contexts_payload_sha256=item.base_contexts_payload_sha256,
        base_model_payload_sha256=item.base_model_payload_sha256,
        base_model_state_sha256=item.base_model_state_sha256,
        reveal_global_seal_sha256=item.reveal_global_seal_sha256,
        reveal_leaf_seal_sha256=item.reveal_leaf_seal_sha256,
        reveal_contexts_payload_sha256=item.reveal_contexts_payload_sha256,
        base_context_count=item.base_context_count,
        base_example_ids_sha256=item.base_example_ids_sha256,
        revealed_context_count=item.revealed_context_count,
        revealed_example_ids_sha256=item.revealed_example_ids_sha256,
        training_context_count=item.training_context_count,
        training_example_ids_sha256=item.training_example_ids_sha256,
        training_example_ids_payload_sha256=payloads["training-example-ids.jsonl"],
        refit=item.refit,
        updated_model_payload_sha256=payloads["updated-model.json"],
    )
    summary_payload = verified.read_payload_bytes("update-summary.json")
    _strict_json_object(summary_payload, label="update-state summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("update-state summary differs from its authenticated attestation")

    return UpdateStateCapability(
        run=frozen,
        model=model,
        training_example_ids=training_ids,
        state_leaf_seal_sha256=verified.seal_sha256,
        updated_model_payload_sha256=payloads["updated-model.json"],
    )


__all__ = [
    "UPDATE_STATE_ATTESTATION_ARTIFACT",
    "UPDATE_STATE_SUMMARY_ARTIFACT",
    "UpdateStateAttestation",
    "publish_update_state",
    "update_state_attestation_from_bytes",
    "update_state_attestation_from_document",
    "update_state_from_authorities",
    "update_state_relative_path",
    "verify_update_state_phase_capability",
]

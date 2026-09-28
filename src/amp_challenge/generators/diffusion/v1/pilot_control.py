"""Torch-free authority records for the four-fit pilot score barrier.

The GPU workers publish one immutable readiness receipt only after their
trainer bundle and all five checkpoint pairs have been independently
observed.  The coordinator can publish the score release only after replaying
those observations for all four folds and proving that nodes and CUDA device
UUIDs are distinct.

This module intentionally imports neither Torch nor any trainer, evaluator, or
bundle-writing module.  The one adapter to :class:`CheckpointIdentity` keeps
its import local and executes it only after the companion metadata bytes match
the digest already bound into a verified readiness receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    ARTIFACT,
    CONFIG_SHA256,
    PARENT_CONFIG_SHA256,
    NativeDiffusionV1PilotContract,
)

_SCHEMA_VERSION = 1
_FOLD_KEYS = ("0", "1", "2", "3")
_FOLDS = (0, 1, 2, 3)
_CHECKPOINT_STEPS = (250, 500, 1000, 2000, 4000)
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_CHECKPOINT_DIGEST_FIELDS = (
    "checkpoint_file_sha256",
    "checkpoint_logical_state_sha256",
    "checkpoint_metadata_sha256",
)
_READINESS_FIELDS = (
    "schema_version",
    "artifact",
    "child_contract_sha256",
    "parent_contract_sha256",
    "git_commit",
    "outer_fold",
    "fit_identity_sha256",
    "trainer_bundle_sha256",
    "count_prior_file_sha256",
    "checkpoint_digest_by_step",
    "checkpoint_digest_map_sha256",
    "node_name",
    "device_uuid",
)
_RELEASE_FIELDS = (
    "schema_version",
    "artifact",
    "child_contract_sha256",
    "parent_contract_sha256",
    "git_commit",
    "readiness_receipt_sha256_by_fold",
    "readiness_receipt_digest_map_sha256",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_MAX_RECEIPT_BYTES = 1 << 20
_MAX_IDENTITY_TEXT_BYTES = 255


@dataclass(frozen=True, slots=True)
class CheckpointDigest:
    """Exact physical, logical, and companion-metadata checkpoint digests."""

    checkpoint_file_sha256: str
    checkpoint_logical_state_sha256: str
    checkpoint_metadata_sha256: str

    def __post_init__(self) -> None:
        for field in _CHECKPOINT_DIGEST_FIELDS:
            _sha256(getattr(self, field), label=field)

    def document(self) -> dict[str, str]:
        """Return the contract-ordered JSON value for this checkpoint."""

        return {field: getattr(self, field) for field in _CHECKPOINT_DIGEST_FIELDS}

    @classmethod
    def from_document(cls, value: object, *, label: str = "checkpoint digest") -> CheckpointDigest:
        document = _exact_json_object(value, _CHECKPOINT_DIGEST_FIELDS, label=label)
        return cls(
            checkpoint_file_sha256=_sha256(
                document["checkpoint_file_sha256"],
                label=f"{label}.checkpoint_file_sha256",
            ),
            checkpoint_logical_state_sha256=_sha256(
                document["checkpoint_logical_state_sha256"],
                label=f"{label}.checkpoint_logical_state_sha256",
            ),
            checkpoint_metadata_sha256=_sha256(
                document["checkpoint_metadata_sha256"],
                label=f"{label}.checkpoint_metadata_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainerReadinessObservation:
    """Authenticated values observed outside the receipt being checked.

    The coordinator receives this value object from its bundle/checkpoint
    authentication boundary.  It contains no paths and performs no I/O.
    """

    git_commit: str
    outer_fold: int
    fit_identity_sha256: str
    trainer_bundle_sha256: str
    count_prior_file_sha256: str
    checkpoint_digest_by_step: Mapping[str, CheckpointDigest]
    node_name: str
    device_uuid: str

    def __post_init__(self) -> None:
        _git_commit(self.git_commit)
        _outer_fold(self.outer_fold)
        _sha256(self.fit_identity_sha256, label="fit_identity_sha256")
        _sha256(self.trainer_bundle_sha256, label="trainer_bundle_sha256")
        _sha256(self.count_prior_file_sha256, label="count_prior_file_sha256")
        object.__setattr__(
            self,
            "checkpoint_digest_by_step",
            _freeze_checkpoint_map(self.checkpoint_digest_by_step),
        )
        _identity_text(self.node_name, label="node_name")
        _identity_text(self.device_uuid, label="device_uuid")


@dataclass(frozen=True, slots=True)
class TrainerReadinessReceipt:
    """Immutable, canonical checkpoint-ready receipt for exactly one fold."""

    schema_version: int
    artifact: str
    child_contract_sha256: str
    parent_contract_sha256: str
    git_commit: str
    outer_fold: int
    fit_identity_sha256: str
    trainer_bundle_sha256: str
    count_prior_file_sha256: str
    checkpoint_digest_by_step: Mapping[str, CheckpointDigest]
    checkpoint_digest_map_sha256: str
    node_name: str
    device_uuid: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError("readiness schema_version must be exact integer 1")
        if type(self.artifact) is not str or self.artifact != ARTIFACT:
            raise ValueError("readiness artifact differs from the frozen pilot artifact")
        if self.child_contract_sha256 != CONFIG_SHA256:
            raise ValueError("readiness child contract digest is not the frozen digest")
        if self.parent_contract_sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("readiness parent contract digest is not the frozen digest")
        _git_commit(self.git_commit)
        _outer_fold(self.outer_fold)
        _sha256(self.fit_identity_sha256, label="fit_identity_sha256")
        _sha256(self.trainer_bundle_sha256, label="trainer_bundle_sha256")
        _sha256(self.count_prior_file_sha256, label="count_prior_file_sha256")
        frozen = _freeze_checkpoint_map(self.checkpoint_digest_by_step)
        object.__setattr__(self, "checkpoint_digest_by_step", frozen)
        expected_map_digest = hashlib.sha256(_checkpoint_map_bytes(frozen)).hexdigest()
        _sha256(self.checkpoint_digest_map_sha256, label="checkpoint_digest_map_sha256")
        if self.checkpoint_digest_map_sha256 != expected_map_digest:
            raise ValueError("readiness checkpoint digest-map SHA-256 does not match its map")
        _identity_text(self.node_name, label="node_name")
        _identity_text(self.device_uuid, label="device_uuid")

    def document(self) -> dict[str, object]:
        """Return the exact top-level field sequence declared by the contract."""

        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "artifact": self.artifact,
            "child_contract_sha256": self.child_contract_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "git_commit": self.git_commit,
            "outer_fold": self.outer_fold,
            "fit_identity_sha256": self.fit_identity_sha256,
            "trainer_bundle_sha256": self.trainer_bundle_sha256,
            "count_prior_file_sha256": self.count_prior_file_sha256,
            "checkpoint_digest_by_step": _checkpoint_map_document(self.checkpoint_digest_by_step),
            "checkpoint_digest_map_sha256": self.checkpoint_digest_map_sha256,
            "node_name": self.node_name,
            "device_uuid": self.device_uuid,
        }
        if tuple(document) != _READINESS_FIELDS:  # pragma: no cover - source invariant
            raise RuntimeError("readiness field construction order changed")
        return document

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.document())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        contract: NativeDiffusionV1PilotContract,
    ) -> TrainerReadinessReceipt:
        return parse_trainer_readiness_receipt(payload, contract=contract)


@dataclass(frozen=True, slots=True)
class ScoreReleaseReceipt:
    """Immutable release binding the exact bytes of all four readiness receipts."""

    schema_version: int
    artifact: str
    child_contract_sha256: str
    parent_contract_sha256: str
    git_commit: str
    readiness_receipt_sha256_by_fold: Mapping[str, str]
    readiness_receipt_digest_map_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError("score-release schema_version must be exact integer 1")
        if type(self.artifact) is not str or self.artifact != ARTIFACT:
            raise ValueError("score-release artifact differs from the frozen pilot artifact")
        if self.child_contract_sha256 != CONFIG_SHA256:
            raise ValueError("score-release child contract digest is not the frozen digest")
        if self.parent_contract_sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("score-release parent contract digest is not the frozen digest")
        _git_commit(self.git_commit)
        frozen = _freeze_fold_digest_map(self.readiness_receipt_sha256_by_fold)
        object.__setattr__(self, "readiness_receipt_sha256_by_fold", frozen)
        expected_map_digest = hashlib.sha256(_fold_digest_map_bytes(frozen)).hexdigest()
        _sha256(
            self.readiness_receipt_digest_map_sha256,
            label="readiness_receipt_digest_map_sha256",
        )
        if self.readiness_receipt_digest_map_sha256 != expected_map_digest:
            raise ValueError("score-release digest-map SHA-256 does not match its map")

    def document(self) -> dict[str, object]:
        """Return the exact top-level field sequence declared by the contract."""

        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "artifact": self.artifact,
            "child_contract_sha256": self.child_contract_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "git_commit": self.git_commit,
            "readiness_receipt_sha256_by_fold": dict(self.readiness_receipt_sha256_by_fold),
            "readiness_receipt_digest_map_sha256": (self.readiness_receipt_digest_map_sha256),
        }
        if tuple(document) != _RELEASE_FIELDS:  # pragma: no cover - source invariant
            raise RuntimeError("score-release field construction order changed")
        return document

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.document())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        contract: NativeDiffusionV1PilotContract,
    ) -> ScoreReleaseReceipt:
        return parse_score_release_receipt(payload, contract=contract)


def build_trainer_readiness_receipt(
    *,
    contract: NativeDiffusionV1PilotContract,
    observation: TrainerReadinessObservation,
) -> TrainerReadinessReceipt:
    """Build one readiness receipt solely from authenticated observations."""

    _validate_contract_surface(contract)
    _validate_observation_against_contract(observation, contract=contract)
    checkpoint_map_bytes = contract.readiness_checkpoint_digest_map_bytes(
        _checkpoint_map_document(observation.checkpoint_digest_by_step)
    )
    return TrainerReadinessReceipt(
        schema_version=_SCHEMA_VERSION,
        artifact=ARTIFACT,
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
        git_commit=observation.git_commit,
        outer_fold=observation.outer_fold,
        fit_identity_sha256=observation.fit_identity_sha256,
        trainer_bundle_sha256=observation.trainer_bundle_sha256,
        count_prior_file_sha256=observation.count_prior_file_sha256,
        checkpoint_digest_by_step=observation.checkpoint_digest_by_step,
        checkpoint_digest_map_sha256=hashlib.sha256(checkpoint_map_bytes).hexdigest(),
        node_name=observation.node_name,
        device_uuid=observation.device_uuid,
    )


def verify_trainer_readiness_receipt(
    receipt: TrainerReadinessReceipt,
    *,
    contract: NativeDiffusionV1PilotContract,
    observation: TrainerReadinessObservation,
) -> TrainerReadinessReceipt:
    """Recheck a readiness receipt against independent authenticated values."""

    if type(receipt) is not TrainerReadinessReceipt:
        raise TypeError("receipt must be an exact TrainerReadinessReceipt")
    expected = build_trainer_readiness_receipt(contract=contract, observation=observation)
    if receipt != expected or receipt.canonical_bytes() != expected.canonical_bytes():
        raise ValueError("readiness receipt differs from authenticated observations")
    return receipt


def parse_trainer_readiness_receipt(
    payload: bytes,
    *,
    contract: NativeDiffusionV1PilotContract,
) -> TrainerReadinessReceipt:
    """Strictly parse canonical readiness JSON without trusting its assertions."""

    _validate_contract_surface(contract)
    document = _strict_json(payload, label="trainer readiness receipt")
    values = _exact_json_object(document, _READINESS_FIELDS, label="trainer readiness receipt")
    raw_map = _exact_json_object(
        values["checkpoint_digest_by_step"],
        _CHECKPOINT_KEYS,
        label="trainer readiness checkpoint map",
    )
    digest_map = {
        key: CheckpointDigest.from_document(
            raw_map[key],
            label=f"trainer readiness checkpoint map.{key}",
        )
        for key in _CHECKPOINT_KEYS
    }
    receipt = TrainerReadinessReceipt(
        schema_version=values["schema_version"],  # type: ignore[arg-type]
        artifact=values["artifact"],  # type: ignore[arg-type]
        child_contract_sha256=values["child_contract_sha256"],  # type: ignore[arg-type]
        parent_contract_sha256=values["parent_contract_sha256"],  # type: ignore[arg-type]
        git_commit=values["git_commit"],  # type: ignore[arg-type]
        outer_fold=values["outer_fold"],  # type: ignore[arg-type]
        fit_identity_sha256=values["fit_identity_sha256"],  # type: ignore[arg-type]
        trainer_bundle_sha256=values["trainer_bundle_sha256"],  # type: ignore[arg-type]
        count_prior_file_sha256=values["count_prior_file_sha256"],  # type: ignore[arg-type]
        checkpoint_digest_by_step=digest_map,
        checkpoint_digest_map_sha256=values["checkpoint_digest_map_sha256"],  # type: ignore[arg-type]
        node_name=values["node_name"],  # type: ignore[arg-type]
        device_uuid=values["device_uuid"],  # type: ignore[arg-type]
    )
    checkpoint_map_bytes = contract.readiness_checkpoint_digest_map_bytes(
        _checkpoint_map_document(receipt.checkpoint_digest_by_step)
    )
    if hashlib.sha256(checkpoint_map_bytes).hexdigest() != receipt.checkpoint_digest_map_sha256:
        raise ValueError("readiness checkpoint map differs from the contract encoding")
    if receipt.canonical_bytes() != payload:
        raise ValueError("trainer readiness receipt is not canonical JSON")
    return receipt


def build_score_release_receipt(
    *,
    contract: NativeDiffusionV1PilotContract,
    git_commit: str,
    readiness_receipt_bytes_by_fold: Mapping[str, bytes],
    observations_by_fold: Mapping[str, TrainerReadinessObservation],
) -> ScoreReleaseReceipt:
    """Replay all four readiness checks and build the sole score-release token."""

    _validate_contract_surface(contract)
    _git_commit(git_commit)
    receipt_payloads = _exact_fold_mapping(
        readiness_receipt_bytes_by_fold,
        label="readiness receipt bytes by fold",
    )
    observations = _exact_fold_mapping(
        observations_by_fold,
        label="readiness observations by fold",
    )
    parsed: list[TrainerReadinessReceipt] = []
    readiness_digests: dict[str, str] = {}
    for fold_key in _FOLD_KEYS:
        payload = receipt_payloads[fold_key]
        if type(payload) is not bytes:
            raise TypeError(f"readiness receipt bytes for fold {fold_key} must be exact bytes")
        observation = observations[fold_key]
        if type(observation) is not TrainerReadinessObservation:
            raise TypeError(f"readiness observation for fold {fold_key} has an invalid value type")
        if observation.outer_fold != int(fold_key):
            raise ValueError(f"readiness observation is under the wrong fold key {fold_key}")
        if observation.git_commit != git_commit:
            raise ValueError(f"readiness observation for fold {fold_key} changed Git commit")
        receipt = parse_trainer_readiness_receipt(payload, contract=contract)
        verify_trainer_readiness_receipt(
            receipt,
            contract=contract,
            observation=observation,
        )
        if receipt.outer_fold != int(fold_key) or receipt.git_commit != git_commit:
            raise ValueError(f"readiness receipt is under the wrong fold or commit {fold_key}")
        parsed.append(receipt)
        readiness_digests[fold_key] = hashlib.sha256(payload).hexdigest()
    if len({receipt.outer_fold for receipt in parsed}) != len(_FOLDS):
        raise ValueError("score release requires four distinct outer folds")
    if len({receipt.node_name for receipt in parsed}) != len(_FOLDS):
        raise ValueError("score release requires four distinct producer nodes")
    if len({receipt.device_uuid for receipt in parsed}) != len(_FOLDS):
        raise ValueError("score release requires four distinct CUDA device UUIDs")
    digest_map_bytes = contract.release_readiness_digest_map_bytes(readiness_digests)
    return ScoreReleaseReceipt(
        schema_version=_SCHEMA_VERSION,
        artifact=ARTIFACT,
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
        git_commit=git_commit,
        readiness_receipt_sha256_by_fold=readiness_digests,
        readiness_receipt_digest_map_sha256=hashlib.sha256(digest_map_bytes).hexdigest(),
    )


def verify_score_release_receipt(
    receipt: ScoreReleaseReceipt,
    *,
    contract: NativeDiffusionV1PilotContract,
    git_commit: str,
    readiness_receipt_bytes_by_fold: Mapping[str, bytes],
    observations_by_fold: Mapping[str, TrainerReadinessObservation],
) -> ScoreReleaseReceipt:
    """Rebuild a release from all source evidence and compare exact bytes."""

    if type(receipt) is not ScoreReleaseReceipt:
        raise TypeError("receipt must be an exact ScoreReleaseReceipt")
    expected = build_score_release_receipt(
        contract=contract,
        git_commit=git_commit,
        readiness_receipt_bytes_by_fold=readiness_receipt_bytes_by_fold,
        observations_by_fold=observations_by_fold,
    )
    if receipt != expected or receipt.canonical_bytes() != expected.canonical_bytes():
        raise ValueError("score-release receipt differs from rechecked readiness evidence")
    return receipt


def parse_score_release_receipt(
    payload: bytes,
    *,
    contract: NativeDiffusionV1PilotContract,
) -> ScoreReleaseReceipt:
    """Strictly parse canonical score-release JSON without trusting its map."""

    _validate_contract_surface(contract)
    document = _strict_json(payload, label="score-release receipt")
    values = _exact_json_object(document, _RELEASE_FIELDS, label="score-release receipt")
    raw_map = _exact_json_object(
        values["readiness_receipt_sha256_by_fold"],
        _FOLD_KEYS,
        label="score-release readiness digest map",
    )
    digest_map = {
        key: _sha256(raw_map[key], label=f"score-release readiness digest map.{key}")
        for key in _FOLD_KEYS
    }
    receipt = ScoreReleaseReceipt(
        schema_version=values["schema_version"],  # type: ignore[arg-type]
        artifact=values["artifact"],  # type: ignore[arg-type]
        child_contract_sha256=values["child_contract_sha256"],  # type: ignore[arg-type]
        parent_contract_sha256=values["parent_contract_sha256"],  # type: ignore[arg-type]
        git_commit=values["git_commit"],  # type: ignore[arg-type]
        readiness_receipt_sha256_by_fold=digest_map,
        readiness_receipt_digest_map_sha256=values["readiness_receipt_digest_map_sha256"],  # type: ignore[arg-type]
    )
    digest_map_bytes = contract.release_readiness_digest_map_bytes(digest_map)
    if hashlib.sha256(digest_map_bytes).hexdigest() != receipt.readiness_receipt_digest_map_sha256:
        raise ValueError("score-release readiness map differs from the contract encoding")
    if receipt.canonical_bytes() != payload:
        raise ValueError("score-release receipt is not canonical JSON")
    return receipt


def publish_trainer_readiness_receipt(
    path: str | os.PathLike[str],
    receipt: TrainerReadinessReceipt,
    *,
    contract: NativeDiffusionV1PilotContract,
    observation: TrainerReadinessObservation,
) -> Path:
    """Verify and atomically publish one 0444 no-replace readiness receipt."""

    verify_trainer_readiness_receipt(
        receipt,
        contract=contract,
        observation=observation,
    )
    return _publish_receipt(path, receipt.canonical_bytes())


def publish_score_release_receipt(
    path: str | os.PathLike[str],
    receipt: ScoreReleaseReceipt,
    *,
    contract: NativeDiffusionV1PilotContract,
    git_commit: str,
    readiness_receipt_bytes_by_fold: Mapping[str, bytes],
    observations_by_fold: Mapping[str, TrainerReadinessObservation],
) -> Path:
    """Recheck all four folds and atomically publish the 0444 release token."""

    verify_score_release_receipt(
        receipt,
        contract=contract,
        git_commit=git_commit,
        readiness_receipt_bytes_by_fold=readiness_receipt_bytes_by_fold,
        observations_by_fold=observations_by_fold,
    )
    return _publish_receipt(path, receipt.canonical_bytes())


def trusted_checkpoint_identity_from_readiness(
    metadata_bytes: bytes,
    receipt: TrainerReadinessReceipt,
    step: int,
) -> object:
    """Derive the loader identity after matching trusted metadata bytes.

    The return type is deliberately annotated as ``object`` so importing this
    module has no Torch-bearing type dependency.  At runtime it is the exact
    ``pilot_checkpoint.CheckpointIdentity`` class.
    """

    if type(receipt) is not TrainerReadinessReceipt:
        raise TypeError("receipt must be an exact TrainerReadinessReceipt")
    if type(metadata_bytes) is not bytes or not 0 < len(metadata_bytes) <= _MAX_RECEIPT_BYTES:
        raise ValueError("checkpoint metadata must be non-empty bounded exact bytes")
    if type(step) is not int or step not in _CHECKPOINT_STEPS:
        raise ValueError("step is not an exact pilot checkpoint step")
    key = f"{step:06d}"
    digest = receipt.checkpoint_digest_by_step[key]
    if hashlib.sha256(metadata_bytes).hexdigest() != digest.checkpoint_metadata_sha256:
        raise ValueError("checkpoint metadata bytes differ from the readiness receipt")

    # This is intentionally the only Torch-bearing import in the module and it
    # occurs strictly after the trusted digest comparison above.
    from amp_challenge.generators.diffusion.v1.pilot_checkpoint import CheckpointIdentity

    return CheckpointIdentity(
        checkpoint_file_sha256=digest.checkpoint_file_sha256,
        checkpoint_logical_state_sha256=digest.checkpoint_logical_state_sha256,
        checkpoint_metadata_sha256=digest.checkpoint_metadata_sha256,
        metadata_bytes=metadata_bytes,
    )


def _validate_contract_surface(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if (
        contract.config_sha256 != CONFIG_SHA256
        or contract.parent_config_sha256 != PARENT_CONFIG_SHA256
        or contract.document.get("schema_version") != _SCHEMA_VERSION
        or contract.document.get("artifact") != ARTIFACT
        or contract.checkpoint_steps != _CHECKPOINT_STEPS
        or tuple(fold.outer_fold for fold in contract.folds) != _FOLDS
    ):
        raise ValueError("pilot contract identity or execution grid changed")
    barrier = contract.table("barrier")
    if tuple(barrier["readiness_receipt_fields"]) != _READINESS_FIELDS:
        raise ValueError("pilot readiness receipt field order changed")
    if tuple(barrier["score_release_fields"]) != _RELEASE_FIELDS:
        raise ValueError("pilot score-release receipt field order changed")
    if tuple(barrier["checkpoint_digest_key_order"]) != _CHECKPOINT_KEYS:
        raise ValueError("pilot checkpoint receipt key order changed")
    if tuple(barrier["checkpoint_digest_value_fields"]) != _CHECKPOINT_DIGEST_FIELDS:
        raise ValueError("pilot checkpoint digest fields changed")
    if tuple(barrier["release_digest_key_order"]) != _FOLD_KEYS:
        raise ValueError("pilot release fold key order changed")
    if barrier["required_readiness_receipts"] != len(_FOLDS):
        raise ValueError("pilot readiness receipt count changed")


def _validate_observation_against_contract(
    observation: TrainerReadinessObservation,
    *,
    contract: NativeDiffusionV1PilotContract,
) -> None:
    if type(observation) is not TrainerReadinessObservation:
        raise TypeError("observation must be an exact TrainerReadinessObservation")
    fold = contract.fold(observation.outer_fold)
    expected_fit = contract.fit_identity_sha256(observation.outer_fold)
    if fold.fit_identity_sha256 != expected_fit:
        raise ValueError("contract fold and reconstructed fit identities differ")
    if observation.fit_identity_sha256 != expected_fit:
        raise ValueError("observed fit identity differs from the authenticated contract")
    contract_bytes = contract.readiness_checkpoint_digest_map_bytes(
        _checkpoint_map_document(observation.checkpoint_digest_by_step)
    )
    if contract_bytes != _checkpoint_map_bytes(observation.checkpoint_digest_by_step):
        raise ValueError("observed checkpoint map differs from contract canonicalization")


def _checkpoint_map_document(
    value: Mapping[str, CheckpointDigest],
) -> dict[str, dict[str, str]]:
    frozen = _freeze_checkpoint_map(value)
    return {key: frozen[key].document() for key in _CHECKPOINT_KEYS}


def _checkpoint_map_bytes(value: Mapping[str, CheckpointDigest]) -> bytes:
    return _canonical_json_bytes(_checkpoint_map_document(value))


def _freeze_checkpoint_map(
    value: Mapping[str, CheckpointDigest],
) -> Mapping[str, CheckpointDigest]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint_digest_by_step must be a mapping")
    _exact_mapping_keys(value, _CHECKPOINT_KEYS, label="checkpoint_digest_by_step")
    result: dict[str, CheckpointDigest] = {}
    for key in _CHECKPOINT_KEYS:
        item = value[key]
        if type(item) is not CheckpointDigest:
            raise TypeError(f"checkpoint_digest_by_step.{key} must be a CheckpointDigest")
        # Reconstruction prevents a mutated or forged subclass from crossing
        # the value-object boundary.
        result[key] = CheckpointDigest(**item.document())
    return MappingProxyType(result)


def _freeze_fold_digest_map(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("readiness_receipt_sha256_by_fold must be a mapping")
    _exact_mapping_keys(
        value,
        _FOLD_KEYS,
        label="readiness_receipt_sha256_by_fold",
    )
    return MappingProxyType(
        {
            key: _sha256(
                value[key],
                label=f"readiness_receipt_sha256_by_fold.{key}",
            )
            for key in _FOLD_KEYS
        }
    )


def _fold_digest_map_bytes(value: Mapping[str, str]) -> bytes:
    frozen = _freeze_fold_digest_map(value)
    return _canonical_json_bytes(dict(frozen))


def _exact_fold_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _exact_mapping_keys(value, _FOLD_KEYS, label=label)
    return value


def _exact_mapping_keys(value: Mapping[Any, Any], expected: tuple[str, ...], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} contains a non-string key")
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(expected_set - observed)}, "
            f"extra={sorted(observed - expected_set)}"
        )


def _exact_json_object(value: object, fields: tuple[str, ...], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    _exact_mapping_keys(value, fields, label=label)
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ValueError("git_commit must be a lowercase forty-character Git object ID")
    return value


def _outer_fold(value: object) -> int:
    if type(value) is not int or value not in _FOLDS:
        raise ValueError("outer_fold must be one of the exact integers 0, 1, 2, 3")
    return value


def _identity_text(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_IDENTITY_TEXT_BYTES
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"{label} must be non-empty bounded printable ASCII without whitespace")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("value cannot be serialized as finite canonical UTF-8 JSON") from error


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    if (
        type(payload) is not bytes
        or not 0 < len(payload) <= _MAX_RECEIPT_BYTES
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
    ):
        raise ValueError(f"{label} must be bounded exact bytes with one trailing LF")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value!r}")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid finite UTF-8 JSON: {error}") from error
    if type(document) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    return document


def _publish_receipt(path: str | os.PathLike[str], payload: bytes) -> Path:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_RECEIPT_BYTES:
        raise ValueError("receipt payload must be non-empty bounded exact bytes")
    destination = Path(os.path.abspath(os.fspath(path)))
    if destination.name in {"", ".", ".."}:
        raise ValueError("receipt destination must name one file")
    parent = destination.parent
    _validate_parent_directory(parent)
    if os.path.lexists(destination):
        raise FileExistsError("receipt publication is strictly no-overwrite")

    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = Path(temporary_raw)
    linked = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short receipt write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _validate_parent_directory(parent)
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)
    if not linked:  # pragma: no cover - successful link sets this first
        raise RuntimeError("receipt publication did not reach its atomic commit point")
    if _read_sealed_receipt(destination) != payload:
        raise RuntimeError("published receipt failed its immutable-byte postcondition")
    return destination


def _validate_parent_directory(parent: Path) -> None:
    _reject_symlink_chain(parent)
    try:
        before = os.lstat(parent)
    except OSError as error:
        raise ValueError("receipt parent directory must already exist") from error
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("receipt parent must be a non-symlink directory")
    try:
        descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError("receipt parent directory cannot be safely opened") from error
    try:
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise ValueError("receipt parent directory changed during validation")


def _read_sealed_receipt(path: Path) -> bytes:
    _reject_symlink_chain(path)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError("published receipt is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_RECEIPT_BYTES
    ):
        raise ValueError("published receipt is not a sealed 0444 single-link file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_nlink != 1
        ):
            raise ValueError("published receipt changed before reading")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("published receipt was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("published receipt grew while reading")
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(before, field) != getattr(after, field)
        for field in identity_fields
    ):
        raise ValueError("published receipt changed during snapshot")
    return b"".join(chunks)


def _reject_symlink_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect path component {current}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CheckpointDigest",
    "ScoreReleaseReceipt",
    "TrainerReadinessObservation",
    "TrainerReadinessReceipt",
    "build_score_release_receipt",
    "build_trainer_readiness_receipt",
    "parse_score_release_receipt",
    "parse_trainer_readiness_receipt",
    "publish_score_release_receipt",
    "publish_trainer_readiness_receipt",
    "trusted_checkpoint_identity_from_readiness",
    "verify_score_release_receipt",
    "verify_trainer_readiness_receipt",
]

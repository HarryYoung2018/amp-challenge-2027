"""Strict, role-isolated pool reveals for sequential mixed-acquisition v2.

Every reveal worker first authenticates one persisted pool commitment.  A
no-query worker has no stage or outcome-vault input at all.  A nonempty worker
receives one rootless stage-global capability and one matching pool-vault
capsule, reanchors that capsule through the authenticated stage index, and only
then decodes labels.  Workers return payload-free attestations; the global
assembler never receives reveal, commitment, or vault payload bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.evaluation.sequential_v2_commitments import (
    AuthenticatedPoolCommitmentCapability,
    CommitmentIndexRow,
    decode_pool_commitment,
    pool_commitment_relative_path,
    verify_pool_commitment_campaign_barrier_for_reveal,
    verify_pool_commitment_capability,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_primitives import TARGETS, ContextRow
from amp_challenge.evaluation.sequential_v2_protocol import (
    CEILING,
    EXPECTED_POLICY_RUNS,
    EXPECTED_POOL_COMMITTED_ASSOCIATIONS,
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    policy_run_by_track_id,
    rotation_by_id,
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
from amp_challenge.evaluation.sequential_v2_select import PoolCommitment
from amp_challenge.evaluation.sequential_v2_stage import (
    POOL_OUTCOME_ROLE,
    AuthenticatedLeafCapsule,
    PoolOutcomeVault,
    StageManifestCapability,
    pool_outcome_vault_from_stage_capabilities,
    verify_stage_manifest_capability,
)

SCHEMA_VERSION = 1
POOL_REVEAL_ARTIFACT = "sequential_v2_pool_reveal_v1"
POOL_REVEAL_SUMMARY_ARTIFACT = "sequential_v2_pool_reveal_summary_v1"
REVEAL_LEAF_ATTESTATION_ARTIFACT = "sequential_v2_pool_reveal_attestation_v1"
REVEAL_CAMPAIGN_ARTIFACT = "sequential_v2_pool_reveal_campaign_barrier_v1"

POOL_REVEAL_PAYLOAD_PATHS = (
    "commitment.json",
    "contexts.jsonl",
    "reveal-summary.json",
    "selected-sequence-ids.jsonl",
)
REVEAL_CAMPAIGN_PAYLOAD_PATHS = ("reveal-index.jsonl", "reveal-summary.json")

EXPECTED_EMPTY_REVEALS = 20
EXPECTED_NONEMPTY_REVEALS = 200
EXPECTED_BUDGETED_REVEALS = 180
EXPECTED_CEILING_REVEALS = 20
EXPECTED_BUDGETED_SELECTED_ASSOCIATIONS = 1_800
EXPECTED_CEILING_SELECTED_ASSOCIATIONS = 2_600
EXPECTED_CEILING_CONTEXT_ASSOCIATIONS = 7_868

_EMPTY_ID_STREAM_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CONTEXT_FIELDS = frozenset(
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
    }
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be exact text")
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

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
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


def _strict_jsonl(
    payload: bytes,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if type(payload) is not bytes or (not payload and not allow_empty):
        raise ValueError(f"{label} must be canonical JSON Lines")
    if not payload:
        return ()
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
    canonical = rotation_by_id(value.rotation_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if type(value.policy) is not str or (value.seed is not None and type(value.seed) is not int):
        raise TypeError(f"{label} policy and seed must use exact scalar types")
    canonical = policy_run_by_track_id(value.track_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _identity_document(
    publication_identity: SequentialV2PublicationIdentity,
) -> dict[str, object]:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("publication identity must be exact")
    return {
        "git_commit": publication_identity.git_commit,
        "code_manifest_sha256": publication_identity.code_manifest_sha256,
        "config_sha256": publication_identity.config_sha256,
        "lock_sha256": publication_identity.lock_sha256,
    }


def _identity_from_document(value: object) -> SequentialV2PublicationIdentity:
    raw = _exact_object(
        value,
        {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"},
        label="reveal publication identity",
    )
    git_commit = _text(raw["git_commit"], label="reveal publication git commit")
    if _GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("reveal publication git commit must be forty lowercase hex characters")
    return SequentialV2PublicationIdentity(
        git_commit=git_commit,
        code_manifest_sha256=_sha256(
            raw["code_manifest_sha256"], label="reveal publication code manifest"
        ),
        config_sha256=_sha256(raw["config_sha256"], label="reveal publication config"),
        lock_sha256=_sha256(raw["lock_sha256"], label="reveal publication lock"),
    )


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
        label="reveal run",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("reveal run schema version changed")
    run = policy_run_by_track_id(_text(raw["track_id"], label="reveal track ID"))
    if canonical_json_bytes(raw) != canonical_json_bytes(run.document()):
        raise ValueError("reveal run differs from the frozen policy registry")
    return run


def _id_stream_sha256(values: Sequence[str], *, label: str) -> str:
    if type(values) not in {tuple, list}:
        raise TypeError(f"{label} must be an exact ID sequence")
    identifiers = tuple(values)
    if len(set(identifiers)) != len(identifiers) or any(
        type(value) is not str or _SHA256.fullmatch(value) is None for value in identifiers
    ):
        raise ValueError(f"{label} must contain unique lowercase SHA-256 values")
    return sha256_bytes("".join(f"{value}\n" for value in identifiers).encode("ascii"))


def pool_reveal_relative_path(run: PolicyRunSpec) -> str:
    """Return the frozen logical path for one per-track reveal leaf."""

    _require_frozen_run(run, label="reveal path run")
    return f"reveal/tracks/{run.track_id}"


def _reveal_predecessor(run: PolicyRunSpec) -> str:
    return f"{pool_reveal_relative_path(run)}/SHA256SUMS"


def _protocol_predecessor() -> str:
    return "protocol/SHA256SUMS"


def _stage_predecessor() -> str:
    return "stage/global/SHA256SUMS"


def _selection_predecessor() -> str:
    return "select/global/SHA256SUMS"


def _commitment_predecessor(run: PolicyRunSpec) -> str:
    return f"{pool_commitment_relative_path(run)}/SHA256SUMS"


def _pool_vault_predecessor(run: PolicyRunSpec) -> str:
    return f"stage/rotations/{run.rotation.rotation_id}/{POOL_OUTCOME_ROLE}/SHA256SUMS"


def _target_counts(value: object, *, label: str) -> tuple[tuple[str, int], ...]:
    raw = _exact_object(value, set(TARGETS), label=label)
    result = tuple(
        (target, _exact_int(raw[target], label=f"{label} {target}")) for target in TARGETS
    )
    return result


def _target_count_document(values: tuple[tuple[str, int], ...]) -> dict[str, int]:
    if (
        type(values) is not tuple
        or tuple(target for target, _count in values) != TARGETS
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
            for item in values
        )
    ):
        raise ValueError("per-target reveal counts must use the exact target inventory")
    return dict(values)


def _require_exact_context(row: object, *, label: str) -> ContextRow:
    if type(row) is not ContextRow:
        raise TypeError(f"{label} must be an exact ContextRow")
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
        type(getattr(row, field)) is not int for field in ("fold", "label", "source_observations")
    ):
        raise TypeError(f"{label} fields must use exact scalar types")
    return row


def _context_document(row: ContextRow) -> dict[str, object]:
    value = _require_exact_context(row, label="reveal context")
    return {
        "example_id": value.example_id,
        "assay_context_id": value.assay_context_id,
        "sequence_id": value.sequence_id,
        "sequence": value.sequence,
        "target": value.target,
        "gram": value.gram,
        "fold": value.fold,
        "label": value.label,
        "source_observations": value.source_observations,
    }


def _context_from_document(value: object, *, label: str) -> ContextRow:
    raw = _exact_object(value, _CONTEXT_FIELDS, label=label)
    strings = (
        "example_id",
        "assay_context_id",
        "sequence_id",
        "sequence",
        "target",
        "gram",
    )
    if any(type(raw[field]) is not str for field in strings) or any(
        type(raw[field]) is not int for field in ("fold", "label", "source_observations")
    ):
        raise ValueError(f"{label} fields must use exact JSON scalar types")
    row = ContextRow(**{field: raw[field] for field in _CONTEXT_FIELDS})  # type: ignore[arg-type]
    if _SHA256.fullmatch(row.example_id) is None or row.assay_context_id != row.example_id:
        raise ValueError(f"{label} has an invalid assay-context identity")
    if canonical_json_bytes(_context_document(row)) != canonical_json_bytes(raw):
        raise ValueError(f"{label} does not round-trip exactly")
    return row


def _context_payload(contexts: tuple[ContextRow, ...]) -> bytes:
    if type(contexts) is not tuple:
        raise TypeError("reveal contexts must be an exact tuple")
    for index, row in enumerate(contexts):
        _require_exact_context(row, label=f"reveal context {index}")
    identifiers = tuple(row.example_id for row in contexts)
    if identifiers != tuple(sorted(set(identifiers))):
        raise ValueError("reveal contexts must use unique ascending example IDs")
    return canonical_jsonl_bytes(_context_document(row) for row in contexts)


def _contexts_from_payload(payload: bytes, *, allow_empty: bool) -> tuple[ContextRow, ...]:
    rows = tuple(
        _context_from_document(item, label=f"reveal context {index}")
        for index, item in enumerate(
            _strict_jsonl(payload, label="reveal contexts", allow_empty=allow_empty)
        )
    )
    if _context_payload(rows) != payload:
        raise ValueError("reveal context payload is not its exact canonical reconstruction")
    return rows


def _selected_ids_payload(identifiers: tuple[str, ...]) -> bytes:
    if type(identifiers) is not tuple:
        raise TypeError("selected sequence IDs must be an exact tuple")
    _id_stream_sha256(identifiers, label="selected sequence IDs")
    return canonical_jsonl_bytes({"sequence_id": value} for value in identifiers)


def _selected_ids_from_payload(payload: bytes, *, allow_empty: bool) -> tuple[str, ...]:
    identifiers: list[str] = []
    for index, item in enumerate(
        _strict_jsonl(payload, label="selected sequence IDs", allow_empty=allow_empty)
    ):
        row = _exact_object(item, {"sequence_id"}, label=f"selected sequence ID row {index}")
        identifiers.append(_sha256(row["sequence_id"], label="selected sequence ID"))
    result = tuple(identifiers)
    if _selected_ids_payload(result) != payload:
        raise ValueError("selected sequence-ID payload is not its exact canonical reconstruction")
    return result


def _payload_digest(seal: PhaseSeal, path: str) -> str:
    matches = tuple(digest for current, digest in seal.payload_sha256 if current == path)
    if len(matches) != 1:
        raise ValueError(f"reveal phase lacks one exact payload digest for {path}")
    return _sha256(matches[0], label=f"reveal payload {path}")


def _validate_payload_digest_inventory(
    value: object,
    *,
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
    result = value
    if tuple(path for path, _digest in result) != POOL_REVEAL_PAYLOAD_PATHS or len(
        dict(result)
    ) != len(result):
        raise ValueError(f"{label} has the wrong payload inventory")
    for path, digest in result:
        _sha256(digest, label=f"{label} {path}")
    return result


@dataclass(frozen=True, slots=True)
class RevealLeafAttestation:
    """Payload-free result from one isolated, semantically verifying worker."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    select_global_seal_sha256: str
    stage_global_seal_sha256: str | None
    reveal_leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    commitment_leaf_seal_sha256: str
    pool_outcome_vault_seal_sha256: str | None
    selected_sequence_count: int
    selected_sequence_ids_sha256: str
    revealed_context_count: int
    revealed_example_ids_sha256: str
    per_target_revealed_context_count: tuple[tuple[str, int], ...]
    revealed_source_observations: int
    empty_reason: str | None

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="reveal attestation run")
        _identity_document(self.publication_identity)
        _sha256(self.protocol_seal_sha256, label="reveal attestation protocol seal")
        _sha256(self.select_global_seal_sha256, label="reveal attestation select seal")
        _sha256(self.reveal_leaf_seal_sha256, label="reveal attestation leaf seal")
        _validate_payload_digest_inventory(
            self.payload_sha256,
            label="reveal attestation payload digests",
        )
        _sha256(
            self.commitment_leaf_seal_sha256,
            label="reveal attestation commitment leaf seal",
        )
        if (
            type(self.selected_sequence_count) is not int
            or self.selected_sequence_count != self.run.expected_pool_selection_count
        ):
            raise ValueError("reveal attestation selected count differs from frozen policy")
        _sha256(
            self.selected_sequence_ids_sha256,
            label="reveal attestation selected IDs",
        )
        if type(self.revealed_context_count) is not int or self.revealed_context_count < 0:
            raise ValueError("reveal attestation context count must be nonnegative")
        _sha256(
            self.revealed_example_ids_sha256,
            label="reveal attestation example IDs",
        )
        target_counts = _target_count_document(self.per_target_revealed_context_count)
        if sum(target_counts.values()) != self.revealed_context_count:
            raise ValueError("reveal attestation target counts do not sum to context count")
        if (
            type(self.revealed_source_observations) is not int
            or self.revealed_source_observations < self.revealed_context_count
        ):
            raise ValueError("reveal attestation observation census is invalid")
        if self.run.policy == NO_QUERY:
            if (
                self.stage_global_seal_sha256 is not None
                or self.pool_outcome_vault_seal_sha256 is not None
                or self.selected_sequence_count != 0
                or self.selected_sequence_ids_sha256 != _EMPTY_ID_STREAM_SHA256
                or self.revealed_context_count != 0
                or self.revealed_example_ids_sha256 != _EMPTY_ID_STREAM_SHA256
                or any(target_counts.values())
                or self.revealed_source_observations != 0
                or self.empty_reason != "no_acquired_sequences"
            ):
                raise ValueError("no-query reveal attestation is not exactly empty")
        else:
            _sha256(self.stage_global_seal_sha256, label="reveal attestation stage seal")
            _sha256(
                self.pool_outcome_vault_seal_sha256,
                label="reveal attestation pool vault seal",
            )
            if (
                self.selected_sequence_count <= 0
                or self.revealed_context_count <= 0
                or self.revealed_source_observations <= 0
                or self.empty_reason is not None
            ):
                raise ValueError("nonempty reveal attestation has an empty census")
        payload_digests = dict(self.payload_sha256)
        expected_summary = _summary_document_from_census(
            run=self.run,
            selection_barrier_seal_sha256=self.select_global_seal_sha256,
            commitment_leaf_seal_sha256=self.commitment_leaf_seal_sha256,
            commitment_payload_sha256=payload_digests["commitment.json"],
            pool_outcome_vault_seal_sha256=self.pool_outcome_vault_seal_sha256,
            selected_sequence_count=self.selected_sequence_count,
            selected_sequence_ids_sha256=self.selected_sequence_ids_sha256,
            revealed_context_count=self.revealed_context_count,
            revealed_example_ids_sha256=self.revealed_example_ids_sha256,
            per_target_revealed_context_count=self.per_target_revealed_context_count,
            revealed_source_observations=self.revealed_source_observations,
            empty_reason=self.empty_reason,
        )
        if payload_digests["reveal-summary.json"] != sha256_bytes(
            canonical_json_bytes(expected_summary)
        ):
            raise ValueError("reveal attestation census differs from its sealed summary digest")
        if self.run.policy == NO_QUERY and (
            payload_digests["contexts.jsonl"] != _EMPTY_ID_STREAM_SHA256
            or payload_digests["selected-sequence-ids.jsonl"] != _EMPTY_ID_STREAM_SHA256
        ):
            raise ValueError("no-query reveal attestation does not bind empty payloads")
        if _attested_leaf_seal_sha256(self) != self.reveal_leaf_seal_sha256:
            raise ValueError("reveal attestation does not reconstruct its authoritative leaf seal")

    def index_document(self) -> dict[str, object]:
        """Project the exact label-free global reveal-index row."""

        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "policy": self.run.policy,
            "seed": self.run.seed,
            "selection_kind": self.run.selection_kind,
            "relative_path": pool_reveal_relative_path(self.run),
            "leaf_artifact": POOL_REVEAL_ARTIFACT,
            "leaf_seal_sha256": self.reveal_leaf_seal_sha256,
            "commitment_leaf_seal_sha256": self.commitment_leaf_seal_sha256,
            "pool_outcome_vault_seal_sha256": self.pool_outcome_vault_seal_sha256,
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "per_target_revealed_context_count": _target_count_document(
                self.per_target_revealed_context_count
            ),
            "revealed_source_observations": self.revealed_source_observations,
        }

    def document(self) -> dict[str, object]:
        """Serialize the safe fresh-exec result channel record."""

        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": REVEAL_LEAF_ATTESTATION_ARTIFACT,
            "run": self.run.document(),
            "publication_identity": _identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "select_global_seal_sha256": self.select_global_seal_sha256,
            "stage_global_seal_sha256": self.stage_global_seal_sha256,
            "reveal_leaf_seal_sha256": self.reveal_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "commitment_leaf_seal_sha256": self.commitment_leaf_seal_sha256,
            "pool_outcome_vault_seal_sha256": self.pool_outcome_vault_seal_sha256,
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "per_target_revealed_context_count": _target_count_document(
                self.per_target_revealed_context_count
            ),
            "revealed_source_observations": self.revealed_source_observations,
            "empty_reason": self.empty_reason,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())


@dataclass(frozen=True, slots=True)
class RevealIndexRow:
    """Exact label-free index record for one authenticated reveal leaf."""

    run: PolicyRunSpec
    relative_path: str
    leaf_seal_sha256: str
    commitment_leaf_seal_sha256: str
    pool_outcome_vault_seal_sha256: str | None
    selected_sequence_count: int
    selected_sequence_ids_sha256: str
    revealed_context_count: int
    revealed_example_ids_sha256: str
    per_target_revealed_context_count: tuple[tuple[str, int], ...]
    revealed_source_observations: int

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="reveal index run")
        if type(self.relative_path) is not str or self.relative_path != pool_reveal_relative_path(
            self.run
        ):
            raise ValueError("reveal index path differs from frozen track")
        _sha256(self.leaf_seal_sha256, label="reveal index leaf seal")
        _sha256(self.commitment_leaf_seal_sha256, label="reveal index commitment leaf seal")
        if (
            type(self.selected_sequence_count) is not int
            or self.selected_sequence_count != self.run.expected_pool_selection_count
        ):
            raise ValueError("reveal index selected count differs from frozen policy")
        _sha256(self.selected_sequence_ids_sha256, label="reveal index selected IDs")
        if type(self.revealed_context_count) is not int or self.revealed_context_count < 0:
            raise ValueError("reveal index context count must be nonnegative")
        _sha256(self.revealed_example_ids_sha256, label="reveal index example IDs")
        counts = _target_count_document(self.per_target_revealed_context_count)
        if sum(counts.values()) != self.revealed_context_count:
            raise ValueError("reveal index target counts do not sum to context count")
        if (
            type(self.revealed_source_observations) is not int
            or self.revealed_source_observations < self.revealed_context_count
        ):
            raise ValueError("reveal index observation census is invalid")
        if self.run.policy == NO_QUERY:
            if (
                self.pool_outcome_vault_seal_sha256 is not None
                or self.revealed_context_count != 0
                or self.revealed_example_ids_sha256 != _EMPTY_ID_STREAM_SHA256
                or any(counts.values())
                or self.revealed_source_observations != 0
                or self.selected_sequence_ids_sha256 != _EMPTY_ID_STREAM_SHA256
            ):
                raise ValueError("no-query reveal index row is not exactly empty")
        else:
            _sha256(
                self.pool_outcome_vault_seal_sha256,
                label="reveal index pool vault seal",
            )
            if self.revealed_context_count <= 0:
                raise ValueError("nonempty reveal index row has no contexts")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "policy": self.run.policy,
            "seed": self.run.seed,
            "selection_kind": self.run.selection_kind,
            "relative_path": self.relative_path,
            "leaf_artifact": POOL_REVEAL_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "commitment_leaf_seal_sha256": self.commitment_leaf_seal_sha256,
            "pool_outcome_vault_seal_sha256": self.pool_outcome_vault_seal_sha256,
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "revealed_context_count": self.revealed_context_count,
            "revealed_example_ids_sha256": self.revealed_example_ids_sha256,
            "per_target_revealed_context_count": _target_count_document(
                self.per_target_revealed_context_count
            ),
            "revealed_source_observations": self.revealed_source_observations,
        }


@dataclass(frozen=True, slots=True)
class SelectedPoolRevealCapability:
    """One globally authorized selected-context projection for an update.

    Construction validates the immutable value shape, but construction alone
    is not authority to consume labels.  Update workers obtain this value only
    through :func:`pool_reveal_from_campaign`, which first authenticates the
    complete reveal campaign and the exact indexed leaf under externally
    supplied controller digests.
    """

    run: PolicyRunSpec
    selected_sequence_ids: tuple[str, ...]
    contexts: tuple[ContextRow, ...]
    allowed_example_ids: tuple[str, ...]
    selected_sequence_ids_sha256: str
    revealed_example_ids_sha256: str
    commitment_leaf_seal_sha256: str
    reveal_leaf_seal_sha256: str
    reveal_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="selected reveal run")
        if type(self.selected_sequence_ids) is not tuple or any(
            type(value) is not str for value in self.selected_sequence_ids
        ):
            raise TypeError("selected reveal sequence IDs must be an exact string tuple")
        if type(self.contexts) is not tuple or any(
            type(row) is not ContextRow for row in self.contexts
        ):
            raise TypeError("selected reveal contexts must be an exact ContextRow tuple")
        if type(self.allowed_example_ids) is not tuple or any(
            type(value) is not str for value in self.allowed_example_ids
        ):
            raise TypeError("selected reveal allowed IDs must be an exact string tuple")

        selected_digest = _id_stream_sha256(
            self.selected_sequence_ids,
            label="selected reveal sequence IDs",
        )
        if len(
            self.selected_sequence_ids
        ) != self.run.expected_pool_selection_count or selected_digest != _sha256(
            self.selected_sequence_ids_sha256,
            label="selected reveal sequence-ID digest",
        ):
            raise ValueError("selected reveal sequence census or digest changed")

        _context_payload(self.contexts)
        example_ids = tuple(row.example_id for row in self.contexts)
        if (
            self.allowed_example_ids != example_ids
            or _id_stream_sha256(example_ids, label="selected reveal example IDs")
            != _sha256(
                self.revealed_example_ids_sha256,
                label="selected reveal example-ID digest",
            )
            or any(row.fold != self.run.rotation.pool_fold for row in self.contexts)
        ):
            raise ValueError("selected reveal contexts have the wrong identity, order, or fold")

        _sha256(
            self.commitment_leaf_seal_sha256,
            label="selected reveal commitment leaf seal",
        )
        _sha256(self.reveal_leaf_seal_sha256, label="selected reveal leaf seal")
        _sha256(
            self.reveal_campaign_seal_sha256,
            label="selected reveal campaign seal",
        )
        represented = {row.sequence_id for row in self.contexts}
        if self.run.policy == NO_QUERY:
            if self.selected_sequence_ids or self.contexts or self.allowed_example_ids:
                raise ValueError("no-query selected reveal must be exactly empty")
        elif not self.contexts or represented != set(self.selected_sequence_ids):
            raise ValueError(
                "selected reveal contexts must represent exactly the committed sequences"
            )


@dataclass(frozen=True, slots=True)
class FinalizePoolRevealCapability:
    """Authenticated reveal plus its select-global-bound commitment trace.

    Construction alone is not authority.  Finalization obtains this value only
    from :func:`pool_reveal_with_commitment_from_campaign`, which first runs the
    complete campaign and indexed-leaf authentication performed by
    :func:`pool_reveal_from_campaign`.
    """

    reveal: SelectedPoolRevealCapability
    commitment: PoolCommitment

    def __post_init__(self) -> None:
        if type(self.reveal) is not SelectedPoolRevealCapability:
            raise TypeError("finalize pool reveal requires an exact selected reveal")
        if type(self.commitment) is not PoolCommitment:
            raise TypeError("finalize pool reveal requires an exact pool commitment")
        if (
            self.commitment.run != self.reveal.run
            or self.commitment.selected_sequence_ids != self.reveal.selected_sequence_ids
        ):
            raise ValueError("finalize pool reveal commitment and context projection disagree")


@dataclass(frozen=True, slots=True)
class _DecodedPoolReveal:
    commitment_payload: bytes
    selected_sequence_ids: tuple[str, ...]
    contexts: tuple[ContextRow, ...]
    attestation: RevealLeafAttestation


def reveal_leaf_attestation_from_document(value: object) -> RevealLeafAttestation:
    """Strictly decode one payload-free reveal-worker result."""

    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "publication_identity",
            "protocol_seal_sha256",
            "select_global_seal_sha256",
            "stage_global_seal_sha256",
            "reveal_leaf_seal_sha256",
            "payload_sha256",
            "commitment_leaf_seal_sha256",
            "pool_outcome_vault_seal_sha256",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "revealed_context_count",
            "revealed_example_ids_sha256",
            "per_target_revealed_context_count",
            "revealed_source_observations",
            "empty_reason",
        },
        label="reveal leaf attestation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or type(raw["artifact"]) is not str
        or raw["artifact"] != REVEAL_LEAF_ATTESTATION_ARTIFACT
    ):
        raise ValueError("reveal leaf attestation identity changed")
    payload_raw = _exact_object(
        raw["payload_sha256"],
        set(POOL_REVEAL_PAYLOAD_PATHS),
        label="reveal leaf attestation payload digests",
    )
    payloads = tuple(
        (path, _sha256(payload_raw[path], label=f"reveal attestation payload {path}"))
        for path in POOL_REVEAL_PAYLOAD_PATHS
    )
    stage = raw["stage_global_seal_sha256"]
    if stage is not None:
        stage = _sha256(stage, label="reveal attestation stage seal")
    vault = raw["pool_outcome_vault_seal_sha256"]
    if vault is not None:
        vault = _sha256(vault, label="reveal attestation pool vault seal")
    empty_reason = raw["empty_reason"]
    if empty_reason is not None and type(empty_reason) is not str:
        raise ValueError("reveal attestation empty reason must be text or null")
    attestation = RevealLeafAttestation(
        run=_run_from_document(raw["run"]),
        publication_identity=_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(raw["protocol_seal_sha256"], label="protocol seal"),
        select_global_seal_sha256=_sha256(
            raw["select_global_seal_sha256"], label="selection barrier seal"
        ),
        stage_global_seal_sha256=stage,
        reveal_leaf_seal_sha256=_sha256(raw["reveal_leaf_seal_sha256"], label="reveal leaf seal"),
        payload_sha256=payloads,
        commitment_leaf_seal_sha256=_sha256(
            raw["commitment_leaf_seal_sha256"], label="commitment leaf seal"
        ),
        pool_outcome_vault_seal_sha256=vault,
        selected_sequence_count=_exact_int(
            raw["selected_sequence_count"], label="selected sequence count"
        ),
        selected_sequence_ids_sha256=_sha256(
            raw["selected_sequence_ids_sha256"], label="selected sequence IDs"
        ),
        revealed_context_count=_exact_int(
            raw["revealed_context_count"], label="revealed context count"
        ),
        revealed_example_ids_sha256=_sha256(
            raw["revealed_example_ids_sha256"], label="revealed example IDs"
        ),
        per_target_revealed_context_count=_target_counts(
            raw["per_target_revealed_context_count"],
            label="per-target revealed context count",
        ),
        revealed_source_observations=_exact_int(
            raw["revealed_source_observations"],
            label="revealed source observations",
        ),
        empty_reason=empty_reason,
    )
    if canonical_json_bytes(attestation.document()) != canonical_json_bytes(raw):
        raise ValueError("reveal leaf attestation does not round-trip exactly")
    return attestation


def reveal_leaf_attestation_from_bytes(payload: bytes) -> RevealLeafAttestation:
    """Decode canonical LF-terminated attestation IPC bytes."""

    return reveal_leaf_attestation_from_document(
        _strict_json(payload, label="reveal leaf attestation")
    )


def _identity_metadata(
    publication_identity: SequentialV2PublicationIdentity,
    *,
    scope_id: str,
) -> dict[str, object]:
    _identity_document(publication_identity)
    return publication_identity.metadata(phase="reveal", scope_id=scope_id)


def _verify_identity_metadata(
    seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    *,
    scope_id: str,
) -> None:
    _identity_document(publication_identity)
    publication_identity.verify_metadata(
        seal.metadata_json,
        phase="reveal",
        scope_id=scope_id,
    )


def _authenticate_commitment(
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    *,
    protocol_capability: ProtocolCapability,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> tuple[
    AuthenticatedPoolCommitmentCapability,
    ProtocolCapability,
    str,
]:
    _require_frozen_run(run, label="reveal worker run")
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("reveal worker requires an exact publication identity")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("reveal worker requires a ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    prepare_seal = _sha256(
        expected_prepare_campaign_seal_sha256,
        label="expected prepare campaign seal",
    )
    selection_seal = _sha256(
        expected_selection_barrier_seal_sha256,
        label="expected selection barrier seal",
    )
    verified = verify_pool_commitment_capability(
        commitment_capability,
        protocol_capability=protocol,
        expected_prepare_campaign_seal_sha256=prepare_seal,
        publication_identity=publication_identity,
        expected_selection_barrier_seal_sha256=selection_seal,
    )
    if verified.run != run:
        raise ValueError("reveal commitment capability belongs to another run")
    return verified, protocol, selection_seal


def _require_exact_pool_vault(vault: object, *, spec: RotationSpec) -> PoolOutcomeVault:
    if type(vault) is not PoolOutcomeVault:
        raise TypeError("reveal worker requires an exact PoolOutcomeVault")
    _require_frozen_rotation(vault.spec, label="pool outcome vault rotation")
    if vault.spec != spec:
        raise ValueError("pool outcome vault belongs to another rotation")
    if type(vault.contexts) is not tuple or type(vault.allowed_example_ids) is not tuple:
        raise TypeError("pool outcome vault fields must be exact tuples")
    for index, row in enumerate(vault.contexts):
        _require_exact_context(row, label=f"pool outcome vault context {index}")
    if any(type(value) is not str for value in vault.allowed_example_ids):
        raise TypeError("pool outcome vault allowed IDs must be exact text")
    example_ids = tuple(row.example_id for row in vault.contexts)
    if (
        example_ids != tuple(sorted(set(example_ids)))
        or example_ids != vault.allowed_example_ids
        or any(row.fold != spec.pool_fold for row in vault.contexts)
    ):
        raise ValueError("pool outcome vault has the wrong identity, order, or fold")
    return vault


def _support_sequence_ids(contexts: tuple[ContextRow, ...]) -> tuple[str, ...]:
    grams: dict[str, set[str]] = defaultdict(set)
    sequences: dict[str, str] = {}
    for row in contexts:
        prior = sequences.setdefault(row.sequence_id, row.sequence)
        if prior != row.sequence:
            raise ValueError("pool vault maps one sequence ID to multiple sequences")
        grams[row.sequence_id].add(row.gram)
    return tuple(
        sorted(
            sequence_id
            for sequence_id, observed in grams.items()
            if {"positive", "negative"}.issubset(observed)
        )
    )


def _source_selected_contexts(
    capsule: AuthenticatedLeafCapsule,
    vault: PoolOutcomeVault,
    *,
    selected_sequence_ids: tuple[str, ...],
) -> tuple[tuple[ContextRow, ...], bytes]:
    if type(capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("reveal worker requires an exact AuthenticatedLeafCapsule")
    full_payload = capsule.seal.read_payload_bytes("contexts.jsonl")
    decoded_source = _contexts_from_payload(full_payload, allow_empty=False)
    if decoded_source != vault.contexts or _context_payload(vault.contexts) != full_payload:
        raise ValueError("pool vault capability differs from its exact source context bytes")
    selected = set(selected_sequence_ids)
    support = set(_support_sequence_ids(vault.contexts))
    if not selected.issubset(support):
        raise ValueError("committed sequence IDs are not a subset of reconstructed pool support")
    contexts = tuple(row for row in vault.contexts if row.sequence_id in selected)
    if {row.sequence_id for row in contexts} != selected:
        raise ValueError("revealed contexts do not represent every committed sequence")
    lines = full_payload.splitlines(keepends=True)
    if len(lines) != len(decoded_source):
        raise AssertionError("canonical pool-vault line census changed")
    selected_payload = b"".join(
        line for line, row in zip(lines, decoded_source, strict=True) if row.sequence_id in selected
    )
    if selected_payload != _context_payload(contexts):
        raise ValueError("revealed contexts are not byte-identical source records")
    return contexts, selected_payload


def _summary_document_from_census(
    *,
    run: PolicyRunSpec,
    selection_barrier_seal_sha256: str,
    commitment_leaf_seal_sha256: str,
    commitment_payload_sha256: str,
    pool_outcome_vault_seal_sha256: str | None,
    selected_sequence_count: int,
    selected_sequence_ids_sha256: str,
    revealed_context_count: int,
    revealed_example_ids_sha256: str,
    per_target_revealed_context_count: tuple[tuple[str, int], ...],
    revealed_source_observations: int,
    empty_reason: str | None,
) -> dict[str, object]:
    """Build the exact summary from values safe to carry in an attestation."""

    frozen_run = _require_frozen_run(run, label="reveal summary run")
    selected_count = _exact_int(selected_sequence_count, label="reveal summary selected count")
    if selected_count != frozen_run.expected_pool_selection_count:
        raise ValueError("reveal summary selected count differs from frozen policy")
    context_count = _exact_int(revealed_context_count, label="reveal summary context count")
    observation_count = _exact_int(
        revealed_source_observations,
        label="reveal summary source observations",
    )
    if observation_count < context_count:
        raise ValueError("reveal summary source-observation census is invalid")
    target_counts = _target_count_document(per_target_revealed_context_count)
    if sum(target_counts.values()) != context_count:
        raise ValueError("reveal summary per-target census differs from context count")
    if empty_reason is not None and type(empty_reason) is not str:
        raise ValueError("reveal summary empty reason must be exact text or null")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": POOL_REVEAL_SUMMARY_ARTIFACT,
        "track_id": frozen_run.track_id,
        "rotation_id": frozen_run.rotation.rotation_id,
        "policy": frozen_run.policy,
        "seed": frozen_run.seed,
        "selection_kind": frozen_run.selection_kind,
        "campaign_barrier_seal_sha256": _sha256(
            selection_barrier_seal_sha256,
            label="reveal summary selection barrier seal",
        ),
        "commitment_leaf_seal_sha256": _sha256(
            commitment_leaf_seal_sha256,
            label="reveal summary commitment leaf seal",
        ),
        "commitment_payload_sha256": _sha256(
            commitment_payload_sha256,
            label="reveal summary commitment payload",
        ),
        "pool_outcome_vault_seal_sha256": pool_outcome_vault_seal_sha256,
        "selected_sequence_count": selected_count,
        "selected_sequence_ids_sha256": _sha256(
            selected_sequence_ids_sha256,
            label="reveal summary selected IDs",
        ),
        "revealed_context_count": context_count,
        "revealed_example_ids_sha256": _sha256(
            revealed_example_ids_sha256,
            label="reveal summary example IDs",
        ),
        "per_target_revealed_context_count": target_counts,
        "revealed_source_observations": observation_count,
        "empty_reason": empty_reason,
    }


def _summary_document(
    *,
    run: PolicyRunSpec,
    selection_barrier_seal_sha256: str,
    commitment_leaf_seal_sha256: str,
    commitment_payload: bytes,
    pool_outcome_vault_seal_sha256: str | None,
    selected_sequence_ids: tuple[str, ...],
    contexts: tuple[ContextRow, ...],
) -> dict[str, object]:
    _require_frozen_run(run, label="reveal summary run")
    if type(commitment_payload) is not bytes:
        raise TypeError("reveal summary commitment payload must be exact bytes")
    counts = Counter(row.target for row in contexts)
    return _summary_document_from_census(
        run=run,
        selection_barrier_seal_sha256=selection_barrier_seal_sha256,
        commitment_leaf_seal_sha256=commitment_leaf_seal_sha256,
        commitment_payload_sha256=sha256_bytes(commitment_payload),
        pool_outcome_vault_seal_sha256=pool_outcome_vault_seal_sha256,
        selected_sequence_count=len(selected_sequence_ids),
        selected_sequence_ids_sha256=_id_stream_sha256(
            selected_sequence_ids,
            label="reveal summary selected IDs",
        ),
        revealed_context_count=len(contexts),
        revealed_example_ids_sha256=_id_stream_sha256(
            tuple(row.example_id for row in contexts),
            label="reveal summary example IDs",
        ),
        per_target_revealed_context_count=tuple((target, counts[target]) for target in TARGETS),
        revealed_source_observations=sum(row.source_observations for row in contexts),
        empty_reason="no_acquired_sequences" if run.policy == NO_QUERY else None,
    )


def _leaf_predecessors(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    selection_barrier_seal_sha256: str,
    commitment_leaf_seal_sha256: str,
    stage_global_seal_sha256: str | None,
    pool_outcome_vault_seal_sha256: str | None,
) -> dict[str, str]:
    _require_frozen_run(run, label="reveal predecessor run")
    result = {
        _protocol_predecessor(): _sha256(protocol_seal_sha256, label="protocol seal"),
        _selection_predecessor(): _sha256(
            selection_barrier_seal_sha256,
            label="selection barrier seal",
        ),
        _commitment_predecessor(run): _sha256(
            commitment_leaf_seal_sha256,
            label="commitment leaf seal",
        ),
    }
    if run.policy == NO_QUERY:
        if stage_global_seal_sha256 is not None or pool_outcome_vault_seal_sha256 is not None:
            raise ValueError("no-query reveal cannot bind stage or pool-vault authority")
    else:
        result[_stage_predecessor()] = _sha256(
            stage_global_seal_sha256,
            label="stage global seal",
        )
        result[_pool_vault_predecessor(run)] = _sha256(
            pool_outcome_vault_seal_sha256,
            label="pool outcome vault seal",
        )
    expected_count = 3 if run.policy == NO_QUERY else 5
    if len(result) != expected_count:
        raise AssertionError("reveal predecessor census changed")
    return result


def _attested_leaf_seal_sha256(attestation: RevealLeafAttestation) -> str:
    """Reconstruct a reveal leaf seal from its payload-free attestation."""

    if type(attestation) is not RevealLeafAttestation:
        raise TypeError("reveal leaf reconstruction requires an exact attestation")
    payloads = dict(
        _validate_payload_digest_inventory(
            attestation.payload_sha256,
            label="reveal attestation payload digests",
        )
    )
    receipt = canonical_json_bytes(
        {
            "artifact": POOL_REVEAL_ARTIFACT,
            "metadata": _identity_metadata(
                attestation.publication_identity,
                scope_id=attestation.run.track_id,
            ),
            "payloads": payloads,
            "predecessor_seals": _leaf_predecessors(
                run=attestation.run,
                protocol_seal_sha256=attestation.protocol_seal_sha256,
                selection_barrier_seal_sha256=attestation.select_global_seal_sha256,
                commitment_leaf_seal_sha256=attestation.commitment_leaf_seal_sha256,
                stage_global_seal_sha256=attestation.stage_global_seal_sha256,
                pool_outcome_vault_seal_sha256=(attestation.pool_outcome_vault_seal_sha256),
            ),
            "schema_version": SCHEMA_VERSION,
            "status": "sealed",
        }
    )
    return sha256_bytes(
        checksum_manifest_bytes(
            {
                **payloads,
                RECEIPT_NAME: sha256_bytes(receipt),
            }
        )
    )


def _attestation_from_verified_leaf(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    selection_barrier_seal_sha256: str,
    stage_global_seal_sha256: str | None,
    commitment_leaf_seal_sha256: str,
    pool_outcome_vault_seal_sha256: str | None,
    selected_sequence_ids: tuple[str, ...],
    contexts: tuple[ContextRow, ...],
) -> RevealLeafAttestation:
    counts = Counter(row.target for row in contexts)
    return RevealLeafAttestation(
        run=run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        select_global_seal_sha256=selection_barrier_seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        reveal_leaf_seal_sha256=seal.seal_sha256,
        payload_sha256=seal.payload_sha256,
        commitment_leaf_seal_sha256=commitment_leaf_seal_sha256,
        pool_outcome_vault_seal_sha256=pool_outcome_vault_seal_sha256,
        selected_sequence_count=len(selected_sequence_ids),
        selected_sequence_ids_sha256=_id_stream_sha256(
            selected_sequence_ids,
            label="reveal attestation selected IDs",
        ),
        revealed_context_count=len(contexts),
        revealed_example_ids_sha256=_id_stream_sha256(
            tuple(row.example_id for row in contexts),
            label="reveal attestation example IDs",
        ),
        per_target_revealed_context_count=tuple((target, counts[target]) for target in TARGETS),
        revealed_source_observations=sum(row.source_observations for row in contexts),
        empty_reason="no_acquired_sequences" if run.policy == NO_QUERY else None,
    )


def _decode_pool_reveal(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    protocol_seal_sha256: str,
    selection_barrier_seal_sha256: str,
    stage_global_seal_sha256: str | None,
    pool_outcome_vault_seal_sha256: str | None,
    expected_context_payload: bytes,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> _DecodedPoolReveal:
    _require_frozen_run(run, label="reveal decoder run")
    if type(commitment_capability) is not AuthenticatedPoolCommitmentCapability:
        raise TypeError("reveal decoder requires an exact commitment capability")
    predecessors = _leaf_predecessors(
        run=run,
        protocol_seal_sha256=protocol_seal_sha256,
        selection_barrier_seal_sha256=selection_barrier_seal_sha256,
        commitment_leaf_seal_sha256=commitment_capability.commitment_leaf.seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        pool_outcome_vault_seal_sha256=pool_outcome_vault_seal_sha256,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=POOL_REVEAL_ARTIFACT,
        expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=_sha256(expected_seal_sha256, label="expected reveal leaf seal"),
    )
    _verify_identity_metadata(verified, publication_identity, scope_id=run.track_id)
    commitment_payload = verified.read_payload_bytes("commitment.json")
    expected_commitment_payload = commitment_capability.commitment_leaf.read_payload_bytes(
        "commitment.json"
    )
    if commitment_payload != expected_commitment_payload:
        raise ValueError("reveal commitment payload is not byte-identical to its sealed leaf")
    decoded_commitment = decode_pool_commitment(commitment_payload)
    if (
        decoded_commitment.run != run
        or canonical_json_bytes(decoded_commitment.document()) != commitment_payload
    ):
        raise ValueError("reveal commitment payload differs from its authenticated run")
    selected_payload = verified.read_payload_bytes("selected-sequence-ids.jsonl")
    selected_ids = _selected_ids_from_payload(
        selected_payload,
        allow_empty=run.policy == NO_QUERY,
    )
    if selected_ids != commitment_capability.commitment.selected_sequence_ids:
        raise ValueError("revealed selected IDs differ from the authenticated commitment")
    context_payload = verified.read_payload_bytes("contexts.jsonl")
    contexts = _contexts_from_payload(context_payload, allow_empty=run.policy == NO_QUERY)
    if type(expected_context_payload) is not bytes or context_payload != expected_context_payload:
        raise ValueError("reveal contexts are not the exact authenticated source projection")
    if any(row.fold != run.rotation.pool_fold for row in contexts):
        raise ValueError("reveal context belongs to the wrong acquisition fold")
    if {row.sequence_id for row in contexts} != set(selected_ids):
        raise ValueError("revealed context sequences differ from committed selected IDs")
    expected_summary = _summary_document(
        run=run,
        selection_barrier_seal_sha256=selection_barrier_seal_sha256,
        commitment_leaf_seal_sha256=commitment_capability.commitment_leaf.seal_sha256,
        commitment_payload=commitment_payload,
        pool_outcome_vault_seal_sha256=pool_outcome_vault_seal_sha256,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
    )
    summary_payload = verified.read_payload_bytes("reveal-summary.json")
    summary = _strict_json_object(summary_payload, label="pool reveal summary")
    if (
        summary_payload != canonical_json_bytes(expected_summary)
        or canonical_json_bytes(summary) != summary_payload
    ):
        raise ValueError("pool reveal summary differs from its authenticated payloads")
    attestation = _attestation_from_verified_leaf(
        verified,
        run=run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        selection_barrier_seal_sha256=selection_barrier_seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        commitment_leaf_seal_sha256=commitment_capability.commitment_leaf.seal_sha256,
        pool_outcome_vault_seal_sha256=pool_outcome_vault_seal_sha256,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
    )
    return _DecodedPoolReveal(
        commitment_payload=commitment_payload,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
        attestation=attestation,
    )


def publish_no_query_pool_reveal(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    protocol_capability: ProtocolCapability,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> RevealLeafAttestation:
    """Publish an empty reveal without accepting any stage or vault capability."""

    commitment, protocol, selection_seal = _authenticate_commitment(
        run,
        commitment_capability,
        protocol_capability=protocol_capability,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if run.policy != NO_QUERY:
        raise ValueError("no-query reveal publisher requires the no-query policy")
    commitment_payload = commitment.commitment_leaf.read_payload_bytes("commitment.json")
    summary = _summary_document(
        run=run,
        selection_barrier_seal_sha256=selection_seal,
        commitment_leaf_seal_sha256=commitment.commitment_leaf.seal_sha256,
        commitment_payload=commitment_payload,
        pool_outcome_vault_seal_sha256=None,
        selected_sequence_ids=(),
        contexts=(),
    )
    seal = publish_phase(
        destination,
        artifact=POOL_REVEAL_ARTIFACT,
        payloads={
            "commitment.json": commitment_payload,
            "contexts.jsonl": b"",
            "reveal-summary.json": canonical_json_bytes(summary),
            "selected-sequence-ids.jsonl": b"",
        },
        predecessor_seals=_leaf_predecessors(
            run=run,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            selection_barrier_seal_sha256=selection_seal,
            commitment_leaf_seal_sha256=commitment.commitment_leaf.seal_sha256,
            stage_global_seal_sha256=None,
            pool_outcome_vault_seal_sha256=None,
        ),
        metadata=_identity_metadata(publication_identity, scope_id=run.track_id),
    )
    return _decode_pool_reveal(
        seal,
        run=run,
        commitment_capability=commitment,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        selection_barrier_seal_sha256=selection_seal,
        stage_global_seal_sha256=None,
        pool_outcome_vault_seal_sha256=None,
        expected_context_payload=b"",
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
    ).attestation


def verify_no_query_pool_reveal_phase_capability(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    protocol_capability: ProtocolCapability,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_leaf_seal_sha256: str,
) -> RevealLeafAttestation:
    """Authenticate one empty no-query reveal without any label-bearing input."""

    commitment, protocol, selection_seal = _authenticate_commitment(
        run,
        commitment_capability,
        protocol_capability=protocol_capability,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if run.policy != NO_QUERY:
        raise ValueError("no-query reveal verifier requires the no-query policy")
    return _decode_pool_reveal(
        seal,
        run=run,
        commitment_capability=commitment,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        selection_barrier_seal_sha256=selection_seal,
        stage_global_seal_sha256=None,
        pool_outcome_vault_seal_sha256=None,
        expected_context_payload=b"",
        publication_identity=publication_identity,
        expected_seal_sha256=expected_reveal_leaf_seal_sha256,
    ).attestation


def _authenticate_pool_vault(
    *,
    run: PolicyRunSpec,
    stage_manifest_capability: StageManifestCapability,
    expected_stage_global_seal_sha256: str,
    pool_outcome_vault_capsule: AuthenticatedLeafCapsule,
) -> tuple[StageManifestCapability, PoolOutcomeVault, bytes]:
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("nonempty reveal requires an exact StageManifestCapability")
    stage_seal = _sha256(
        expected_stage_global_seal_sha256,
        label="expected stage global seal",
    )
    verified_stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_seal,
    )
    vault = pool_outcome_vault_from_stage_capabilities(
        verified_stage,
        pool_outcome_vault_capsule,
        spec=run.rotation,
        expected_stage_global_seal_sha256=stage_seal,
    )
    exact_vault = _require_exact_pool_vault(vault, spec=run.rotation)
    source_payload = pool_outcome_vault_capsule.seal.read_payload_bytes("contexts.jsonl")
    if _context_payload(exact_vault.contexts) != source_payload:
        raise ValueError("pool outcome vault values differ from source payload bytes")
    return verified_stage, exact_vault, source_payload


def publish_nonempty_pool_reveal(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    protocol_capability: ProtocolCapability,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    stage_manifest_capability: StageManifestCapability,
    expected_stage_global_seal_sha256: str,
    pool_outcome_vault_capsule: AuthenticatedLeafCapsule,
) -> RevealLeafAttestation:
    """Publish one exact committed projection from one reanchored pool vault."""

    commitment, protocol, selection_seal = _authenticate_commitment(
        run,
        commitment_capability,
        protocol_capability=protocol_capability,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if run.policy == NO_QUERY:
        raise ValueError("nonempty reveal publisher rejects the no-query policy")
    stage, vault, _source_payload = _authenticate_pool_vault(
        run=run,
        stage_manifest_capability=stage_manifest_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        pool_outcome_vault_capsule=pool_outcome_vault_capsule,
    )
    selected_ids = commitment.commitment.selected_sequence_ids
    contexts, context_payload = _source_selected_contexts(
        pool_outcome_vault_capsule,
        vault,
        selected_sequence_ids=selected_ids,
    )
    commitment_payload = commitment.commitment_leaf.read_payload_bytes("commitment.json")
    vault_seal = pool_outcome_vault_capsule.seal.seal_sha256
    summary = _summary_document(
        run=run,
        selection_barrier_seal_sha256=selection_seal,
        commitment_leaf_seal_sha256=commitment.commitment_leaf.seal_sha256,
        commitment_payload=commitment_payload,
        pool_outcome_vault_seal_sha256=vault_seal,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
    )
    seal = publish_phase(
        destination,
        artifact=POOL_REVEAL_ARTIFACT,
        payloads={
            "commitment.json": commitment_payload,
            "contexts.jsonl": context_payload,
            "reveal-summary.json": canonical_json_bytes(summary),
            "selected-sequence-ids.jsonl": _selected_ids_payload(selected_ids),
        },
        predecessor_seals=_leaf_predecessors(
            run=run,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            selection_barrier_seal_sha256=selection_seal,
            commitment_leaf_seal_sha256=commitment.commitment_leaf.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            pool_outcome_vault_seal_sha256=vault_seal,
        ),
        metadata=_identity_metadata(publication_identity, scope_id=run.track_id),
    )
    return _decode_pool_reveal(
        seal,
        run=run,
        commitment_capability=commitment,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        selection_barrier_seal_sha256=selection_seal,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        pool_outcome_vault_seal_sha256=vault_seal,
        expected_context_payload=context_payload,
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
    ).attestation


def verify_nonempty_pool_reveal_phase_capability(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    commitment_capability: AuthenticatedPoolCommitmentCapability,
    protocol_capability: ProtocolCapability,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    stage_manifest_capability: StageManifestCapability,
    expected_stage_global_seal_sha256: str,
    pool_outcome_vault_capsule: AuthenticatedLeafCapsule,
    expected_reveal_leaf_seal_sha256: str,
) -> RevealLeafAttestation:
    """Fully reauthenticate one nonempty reveal and its exact source projection."""

    commitment, protocol, selection_seal = _authenticate_commitment(
        run,
        commitment_capability,
        protocol_capability=protocol_capability,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if run.policy == NO_QUERY:
        raise ValueError("nonempty reveal verifier rejects the no-query policy")
    stage, vault, _source_payload = _authenticate_pool_vault(
        run=run,
        stage_manifest_capability=stage_manifest_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        pool_outcome_vault_capsule=pool_outcome_vault_capsule,
    )
    _contexts, expected_context_payload = _source_selected_contexts(
        pool_outcome_vault_capsule,
        vault,
        selected_sequence_ids=commitment.commitment.selected_sequence_ids,
    )
    return _decode_pool_reveal(
        seal,
        run=run,
        commitment_capability=commitment,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        selection_barrier_seal_sha256=selection_seal,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        pool_outcome_vault_seal_sha256=pool_outcome_vault_capsule.seal.seal_sha256,
        expected_context_payload=expected_context_payload,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_reveal_leaf_seal_sha256,
    ).attestation


def _reveal_index_row_from_document(value: object) -> RevealIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "track_id",
            "rotation_id",
            "policy",
            "seed",
            "selection_kind",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "commitment_leaf_seal_sha256",
            "pool_outcome_vault_seal_sha256",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "revealed_context_count",
            "revealed_example_ids_sha256",
            "per_target_revealed_context_count",
            "revealed_source_observations",
        },
        label="reveal index row",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("reveal index schema version changed")
    run = policy_run_by_track_id(_text(raw["track_id"], label="reveal index track ID"))
    if (
        type(raw["rotation_id"]) is not str
        or raw["rotation_id"] != run.rotation.rotation_id
        or type(raw["policy"]) is not str
        or raw["policy"] != run.policy
        or type(raw["seed"]) is not type(run.seed)
        or raw["seed"] != run.seed
        or type(raw["selection_kind"]) is not str
        or raw["selection_kind"] != run.selection_kind
        or type(raw["leaf_artifact"]) is not str
        or raw["leaf_artifact"] != POOL_REVEAL_ARTIFACT
    ):
        raise ValueError("reveal index identity differs from its frozen track")
    vault = raw["pool_outcome_vault_seal_sha256"]
    if vault is not None:
        vault = _sha256(vault, label="reveal index pool vault seal")
    row = RevealIndexRow(
        run=run,
        relative_path=_text(raw["relative_path"], label="reveal index relative path"),
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="reveal index leaf seal"),
        commitment_leaf_seal_sha256=_sha256(
            raw["commitment_leaf_seal_sha256"],
            label="reveal index commitment leaf seal",
        ),
        pool_outcome_vault_seal_sha256=vault,
        selected_sequence_count=_exact_int(
            raw["selected_sequence_count"],
            label="reveal index selected sequence count",
        ),
        selected_sequence_ids_sha256=_sha256(
            raw["selected_sequence_ids_sha256"],
            label="reveal index selected IDs",
        ),
        revealed_context_count=_exact_int(
            raw["revealed_context_count"],
            label="reveal index context count",
        ),
        revealed_example_ids_sha256=_sha256(
            raw["revealed_example_ids_sha256"],
            label="reveal index example IDs",
        ),
        per_target_revealed_context_count=_target_counts(
            raw["per_target_revealed_context_count"],
            label="reveal index per-target counts",
        ),
        revealed_source_observations=_exact_int(
            raw["revealed_source_observations"],
            label="reveal index source observations",
        ),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("reveal index row does not round-trip exactly")
    return row


def _campaign_summary_document(
    rows: Sequence[RevealIndexRow],
    *,
    reveal_index_sha256: str,
) -> dict[str, object]:
    values = tuple(rows)
    if (
        len(values) != EXPECTED_POLICY_RUNS
        or any(type(row) is not RevealIndexRow for row in values)
        or tuple(row.run for row in values) != ordered_policy_runs()
    ):
        raise ValueError("reveal campaign requires 220 exact rows in frozen track order")
    empty = tuple(row for row in values if row.run.policy == NO_QUERY)
    ceiling = tuple(row for row in values if row.run.policy == CEILING)
    budgeted = tuple(row for row in values if row.run.policy not in {NO_QUERY, CEILING})
    nonempty = (*budgeted, *ceiling)
    target_counts = {
        target: sum(dict(row.per_target_revealed_context_count)[target] for row in values)
        for target in TARGETS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact": REVEAL_CAMPAIGN_ARTIFACT,
        "reveal_leaf_count": len(values),
        "empty_reveal_leaf_count": len(empty),
        "nonempty_reveal_leaf_count": len(nonempty),
        "budgeted_reveal_leaf_count": len(budgeted),
        "ceiling_reveal_leaf_count": len(ceiling),
        "selected_sequence_association_count": sum(row.selected_sequence_count for row in values),
        "budgeted_selected_sequence_association_count": sum(
            row.selected_sequence_count for row in budgeted
        ),
        "ceiling_selected_sequence_association_count": sum(
            row.selected_sequence_count for row in ceiling
        ),
        "revealed_context_association_count": sum(row.revealed_context_count for row in values),
        "budgeted_revealed_context_association_count": sum(
            row.revealed_context_count for row in budgeted
        ),
        "ceiling_revealed_context_association_count": sum(
            row.revealed_context_count for row in ceiling
        ),
        "per_target_revealed_context_association_count": target_counts,
        "revealed_source_observations": sum(row.revealed_source_observations for row in values),
        "reveal_index_sha256": _sha256(
            reveal_index_sha256,
            label="reveal campaign index",
        ),
    }
    if (
        summary["empty_reveal_leaf_count"] != EXPECTED_EMPTY_REVEALS
        or summary["nonempty_reveal_leaf_count"] != EXPECTED_NONEMPTY_REVEALS
        or summary["budgeted_reveal_leaf_count"] != EXPECTED_BUDGETED_REVEALS
        or summary["ceiling_reveal_leaf_count"] != EXPECTED_CEILING_REVEALS
        or summary["selected_sequence_association_count"] != EXPECTED_POOL_COMMITTED_ASSOCIATIONS
        or summary["budgeted_selected_sequence_association_count"]
        != EXPECTED_BUDGETED_SELECTED_ASSOCIATIONS
        or summary["ceiling_selected_sequence_association_count"]
        != EXPECTED_CEILING_SELECTED_ASSOCIATIONS
        or summary["ceiling_revealed_context_association_count"]
        != EXPECTED_CEILING_CONTEXT_ASSOCIATIONS
        or summary["revealed_context_association_count"]
        != summary["budgeted_revealed_context_association_count"]
        + EXPECTED_CEILING_CONTEXT_ASSOCIATIONS
        or sum(target_counts.values()) != summary["revealed_context_association_count"]
    ):
        raise ValueError("reveal campaign global census differs from the frozen graph")
    return summary


def _campaign_predecessors(
    rows: Sequence[RevealIndexRow],
    *,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    select_global_seal_sha256: str,
) -> dict[str, str]:
    values = tuple(rows)
    if tuple(row.run for row in values) != ordered_policy_runs():
        raise ValueError("reveal campaign predecessors require frozen track order")
    result = {
        _protocol_predecessor(): _sha256(protocol_seal_sha256, label="protocol seal"),
        _stage_predecessor(): _sha256(stage_global_seal_sha256, label="stage global seal"),
        _selection_predecessor(): _sha256(
            select_global_seal_sha256,
            label="select global seal",
        ),
    }
    result.update({_reveal_predecessor(row.run): row.leaf_seal_sha256 for row in values})
    if len(result) != 223:
        raise ValueError("reveal campaign predecessor census must be exactly 223")
    return result


def _decode_campaign_indices(seal: PhaseSeal) -> tuple[RevealIndexRow, ...]:
    rows = tuple(
        _reveal_index_row_from_document(item)
        for item in _strict_jsonl(
            seal.read_payload_bytes("reveal-index.jsonl"),
            label="reveal campaign index",
        )
    )
    if tuple(row.run for row in rows) != ordered_policy_runs():
        raise ValueError("reveal campaign index differs from frozen track order")
    return rows


def _decode_safe_campaign(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[tuple[RevealIndexRow, ...], str, str, str]:
    if type(seal) is not PhaseSeal:
        raise TypeError("reveal campaign requires an exact rootless PhaseSeal")
    _identity_document(publication_identity)
    authenticated = verify_phase_capability(
        seal,
        expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=seal.seal_sha256,
    )
    preliminary_rows = _decode_campaign_indices(authenticated)
    anchors = dict(authenticated.predecessor_seals)
    protocol = _sha256(anchors.get(_protocol_predecessor()), label="campaign protocol seal")
    stage = _sha256(anchors.get(_stage_predecessor()), label="campaign stage seal")
    selection = _sha256(anchors.get(_selection_predecessor()), label="campaign select seal")
    expected_predecessors = _campaign_predecessors(
        preliminary_rows,
        protocol_seal_sha256=protocol,
        stage_global_seal_sha256=stage,
        select_global_seal_sha256=selection,
    )
    verified = verify_phase_capability(
        authenticated,
        expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=expected_predecessors,
        expected_seal_sha256=seal.seal_sha256,
    )
    _verify_identity_metadata(verified, publication_identity, scope_id="global")
    rows = _decode_campaign_indices(verified)
    index_payload = verified.read_payload_bytes("reveal-index.jsonl")
    expected_summary = _campaign_summary_document(
        rows,
        reveal_index_sha256=sha256_bytes(index_payload),
    )
    summary_payload = verified.read_payload_bytes("reveal-summary.json")
    _strict_json_object(summary_payload, label="reveal campaign summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("reveal campaign summary differs from its exact index census")
    return rows, protocol, stage, selection


@dataclass(frozen=True, slots=True)
class RevealCampaignCapability:
    """Rootless label-free global reveal barrier for future update workers."""

    seal: PhaseSeal
    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("reveal campaign capability requires an exact PhaseSeal")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("reveal campaign capability requires an exact publication identity")
        _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
        )

    def index_row(self, *, run: PolicyRunSpec) -> RevealIndexRow:
        """Return one label-free indexed reveal identity without sibling payloads."""

        _require_frozen_run(run, label="reveal campaign lookup run")
        rows, _protocol, _stage, _selection = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
        )
        matches = tuple(row for row in rows if row.run == run)
        if len(matches) != 1:
            raise ValueError("reveal campaign lacks one exact requested track")
        return matches[0]


def _campaign_authorities(
    *,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> tuple[
    ProtocolCapability,
    StageManifestCapability,
    tuple[CommitmentIndexRow, ...],
    str,
    str,
]:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("reveal campaign requires an exact publication identity")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("reveal campaign requires a ProtocolCapability")
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("reveal campaign requires a StageManifestCapability")
    if type(selection_barrier) is not PhaseSeal:
        raise TypeError("reveal campaign requires an exact select-global PhaseSeal")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    stage_seal = _sha256(
        expected_stage_global_seal_sha256,
        label="expected stage global seal",
    )
    stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_seal,
    )
    selection_seal = _sha256(
        expected_selection_barrier_seal_sha256,
        label="expected selection barrier seal",
    )
    commitment_rows = verify_pool_commitment_campaign_barrier_for_reveal(
        selection_barrier,
        protocol_capability=protocol,
        expected_prepare_campaign_seal_sha256=_sha256(
            expected_prepare_campaign_seal_sha256,
            label="expected prepare campaign seal",
        ),
        publication_identity=publication_identity,
        expected_seal_sha256=selection_seal,
    )
    return protocol, stage, commitment_rows, stage_seal, selection_seal


def _cross_check_campaign_rows(
    reveal_rows: Sequence[RevealIndexRow],
    commitment_rows: Sequence[CommitmentIndexRow],
    *,
    stage_manifest_capability: StageManifestCapability,
) -> None:
    reveals = tuple(reveal_rows)
    commitments = tuple(commitment_rows)
    if (
        tuple(row.run for row in reveals) != ordered_policy_runs()
        or tuple(row.run for row in commitments) != ordered_policy_runs()
    ):
        raise ValueError("reveal/select campaign rows differ from frozen track order")
    for reveal, commitment in zip(reveals, commitments, strict=True):
        if (
            reveal.commitment_leaf_seal_sha256 != commitment.leaf_seal_sha256
            or reveal.selected_sequence_count != commitment.selected_sequence_count
            or reveal.selected_sequence_ids_sha256 != commitment.selected_sequence_ids_sha256
        ):
            raise ValueError("reveal index differs from authenticated select-global row")
        if reveal.run.policy == NO_QUERY:
            if reveal.pool_outcome_vault_seal_sha256 is not None:
                raise ValueError("no-query reveal index unexpectedly names a pool vault")
        else:
            expected_vault = stage_manifest_capability.leaf(
                spec=reveal.run.rotation,
                role=POOL_OUTCOME_ROLE,
            )
            if reveal.pool_outcome_vault_seal_sha256 != expected_vault.leaf_seal_sha256:
                raise ValueError("reveal index pool vault differs from stage-global index")


def _validate_reveal_attestations(
    attestations: tuple[RevealLeafAttestation, ...],
    *,
    expected_reveal_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> tuple[
    tuple[RevealLeafAttestation, ...],
    tuple[RevealIndexRow, ...],
    ProtocolCapability,
    StageManifestCapability,
    str,
]:
    protocol, stage, commitments, stage_seal, selection_seal = _campaign_authorities(
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if type(attestations) is not tuple:
        raise TypeError("reveal campaign attestations must be one exact immutable tuple")
    if type(expected_reveal_leaf_seal_sha256s) is not tuple:
        raise TypeError("expected reveal leaf digests must be one exact immutable tuple")
    if (
        len(attestations) != EXPECTED_POLICY_RUNS
        or any(type(item) is not RevealLeafAttestation for item in attestations)
        or tuple(item.run for item in attestations) != ordered_policy_runs()
        or len(expected_reveal_leaf_seal_sha256s) != EXPECTED_POLICY_RUNS
    ):
        raise ValueError("reveal campaign requires 220 exact attestations in frozen order")
    expected_leaf_seals = tuple(
        _sha256(value, label=f"expected reveal leaf seal {index}")
        for index, value in enumerate(expected_reveal_leaf_seal_sha256s)
    )
    if len(set(expected_leaf_seals)) != EXPECTED_POLICY_RUNS:
        raise ValueError("expected reveal leaf seal sequence must contain 220 distinct values")
    rows: list[RevealIndexRow] = []
    for index, (item, commitment, expected_leaf) in enumerate(
        zip(attestations, commitments, expected_leaf_seals, strict=True)
    ):
        if reveal_leaf_attestation_from_document(item.document()).canonical_bytes() != (
            item.canonical_bytes()
        ):
            raise ValueError(f"reveal attestation {index} changed during strict reconstruction")
        if item.publication_identity != publication_identity:
            raise ValueError("reveal attestation has the wrong publication identity")
        if (
            item.protocol_seal_sha256 != protocol.seal.seal_sha256
            or item.select_global_seal_sha256 != selection_seal
            or item.reveal_leaf_seal_sha256 != expected_leaf
            or item.commitment_leaf_seal_sha256 != commitment.leaf_seal_sha256
            or dict(item.payload_sha256)["commitment.json"] != commitment.commitment_payload_sha256
            or item.selected_sequence_count != commitment.selected_sequence_count
            or item.selected_sequence_ids_sha256 != commitment.selected_sequence_ids_sha256
        ):
            raise ValueError("reveal attestation differs from controller authority")
        if item.run.policy == NO_QUERY:
            if item.stage_global_seal_sha256 is not None:
                raise ValueError("no-query reveal attestation binds stage-global")
        else:
            expected_vault = stage.leaf(spec=item.run.rotation, role=POOL_OUTCOME_ROLE)
            if (
                item.stage_global_seal_sha256 != stage_seal
                or item.pool_outcome_vault_seal_sha256 != expected_vault.leaf_seal_sha256
            ):
                raise ValueError("reveal attestation differs from stage-global authority")
        rows.append(_reveal_index_row_from_document(item.index_document()))
    result = tuple(rows)
    _cross_check_campaign_rows(
        result,
        commitments,
        stage_manifest_capability=stage,
    )
    return attestations, result, protocol, stage, selection_seal


def _campaign_payloads(rows: tuple[RevealIndexRow, ...]) -> dict[str, bytes]:
    index_payload = canonical_jsonl_bytes(row.document() for row in rows)
    return {
        "reveal-index.jsonl": index_payload,
        "reveal-summary.json": canonical_json_bytes(
            _campaign_summary_document(
                rows,
                reveal_index_sha256=sha256_bytes(index_payload),
            )
        ),
    }


def publish_pool_reveal_campaign_barrier(
    destination: str | Path,
    *,
    attestations: tuple[RevealLeafAttestation, ...],
    expected_reveal_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> RevealCampaignCapability:
    """Publish the label-free 223-predecessor barrier from safe attestations."""

    _items, rows, protocol, stage, selection_seal = _validate_reveal_attestations(
        attestations,
        expected_reveal_leaf_seal_sha256s=expected_reveal_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    payloads = _campaign_payloads(rows)
    seal = publish_phase(
        destination,
        artifact=REVEAL_CAMPAIGN_ARTIFACT,
        payloads=payloads,
        predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            select_global_seal_sha256=selection_seal,
        ),
        metadata=_identity_metadata(publication_identity, scope_id="global"),
    )
    capability = reveal_campaign_capability_from_seal(
        seal,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=stage.seal.seal_sha256,
        expected_selection_barrier_seal_sha256=selection_seal,
        expected_reveal_campaign_seal_sha256=seal.seal_sha256,
    )
    for path, expected in payloads.items():
        if capability.seal.read_payload_bytes(path) != expected:
            raise ValueError(f"reveal campaign payload differs from attestations: {path}")
    return capability


def reveal_campaign_capability_from_seal(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> RevealCampaignCapability:
    """Derive a rootless reveal-global capability under external authorities."""

    protocol, stage, commitments, stage_seal, selection_seal = _campaign_authorities(
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        publication_identity=publication_identity,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    expected_global = _sha256(
        expected_reveal_campaign_seal_sha256,
        label="expected reveal campaign seal",
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_global,
    )
    rows, actual_protocol, actual_stage, actual_selection = _decode_safe_campaign(
        verified,
        publication_identity=publication_identity,
    )
    if (
        actual_protocol != protocol.seal.seal_sha256
        or actual_stage != stage_seal
        or actual_selection != selection_seal
    ):
        raise ValueError("reveal campaign binds the wrong global authorities")
    _cross_check_campaign_rows(
        rows,
        commitments,
        stage_manifest_capability=stage,
    )
    return RevealCampaignCapability(verified, publication_identity)


def verify_reveal_campaign_capability(
    capability: RevealCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> RevealCampaignCapability:
    """Freshly rederive reveal-global before granting update authority."""

    if type(capability) is not RevealCampaignCapability:
        raise TypeError("reveal campaign verification requires a RevealCampaignCapability")
    if capability.publication_identity != publication_identity:
        raise ValueError("reveal campaign capability has the wrong publication identity")
    verified = reveal_campaign_capability_from_seal(
        capability.seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    if capability.seal.seal_sha256 != verified.seal.seal_sha256:
        raise ValueError("reveal campaign capability differs from authoritative derivation")
    return verified


def pool_reveal_from_campaign(
    campaign: RevealCampaignCapability,
    *,
    run: PolicyRunSpec,
    reveal_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> SelectedPoolRevealCapability:
    """Authorize and decode one selected reveal without its source vault.

    The reveal-global barrier's supervised attestations already prove that
    each indexed context payload was the exact selected projection of its pool
    vault.  An update worker therefore needs only the globally indexed reveal
    leaf.  All global and leaf authorities are checked before any payload from
    that label-bearing leaf is read.
    """

    frozen_run = _require_frozen_run(run, label="selected reveal lookup run")
    if type(campaign) is not RevealCampaignCapability:
        raise TypeError("selected reveal lookup requires a RevealCampaignCapability")
    if type(reveal_seal) is not PhaseSeal:
        raise TypeError("selected reveal lookup requires an exact reveal PhaseSeal")
    _identity_document(publication_identity)
    prepare_seal = _sha256(
        expected_prepare_campaign_seal_sha256,
        label="expected prepare campaign seal",
    )
    stage_seal = _sha256(
        expected_stage_global_seal_sha256,
        label="expected stage global seal",
    )
    selection_seal = _sha256(
        expected_selection_barrier_seal_sha256,
        label="expected selection barrier seal",
    )
    campaign_seal = _sha256(
        expected_reveal_campaign_seal_sha256,
        label="expected reveal campaign seal",
    )

    # Authenticate every label-free authority and both global indices before
    # consulting any payload carried by the requested reveal leaf.
    authorized_campaign = verify_reveal_campaign_capability(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=prepare_seal,
        expected_stage_global_seal_sha256=stage_seal,
        expected_selection_barrier_seal_sha256=selection_seal,
        expected_reveal_campaign_seal_sha256=campaign_seal,
    )
    commitment_rows = verify_pool_commitment_campaign_barrier_for_reveal(
        selection_barrier,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=prepare_seal,
        publication_identity=publication_identity,
        expected_seal_sha256=selection_seal,
    )
    row_index = ordered_policy_runs().index(frozen_run)
    commitment_row = commitment_rows[row_index]
    reveal_row = authorized_campaign.index_row(run=frozen_run)
    if (
        commitment_row.run != frozen_run
        or reveal_row.run != frozen_run
        or reveal_row.commitment_leaf_seal_sha256 != commitment_row.leaf_seal_sha256
        or reveal_row.selected_sequence_count != commitment_row.selected_sequence_count
        or reveal_row.selected_sequence_ids_sha256 != commitment_row.selected_sequence_ids_sha256
    ):
        raise ValueError("selected reveal indices disagree for the requested run")

    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    expected_predecessors = _leaf_predecessors(
        run=frozen_run,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        selection_barrier_seal_sha256=selection_seal,
        commitment_leaf_seal_sha256=commitment_row.leaf_seal_sha256,
        stage_global_seal_sha256=(None if frozen_run.policy == NO_QUERY else stage_seal),
        pool_outcome_vault_seal_sha256=reveal_row.pool_outcome_vault_seal_sha256,
    )
    verified_leaf = verify_phase_capability(
        reveal_seal,
        expected_artifact=POOL_REVEAL_ARTIFACT,
        expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
        expected_predecessor_seals=expected_predecessors,
        expected_seal_sha256=reveal_row.leaf_seal_sha256,
    )
    _verify_identity_metadata(
        verified_leaf,
        publication_identity,
        scope_id=frozen_run.track_id,
    )
    if _payload_digest(verified_leaf, "commitment.json") != (
        commitment_row.commitment_payload_sha256
    ):
        raise ValueError("selected reveal binds the wrong commitment payload digest")

    # This is the first point at which the label-bearing reveal leaf is read.
    commitment_payload = verified_leaf.read_payload_bytes("commitment.json")
    commitment = decode_pool_commitment(commitment_payload)
    if (
        commitment.run != frozen_run
        or canonical_json_bytes(commitment.document()) != commitment_payload
        or sha256_bytes(commitment_payload) != commitment_row.commitment_payload_sha256
    ):
        raise ValueError("selected reveal commitment differs from select-global")

    selected_payload = verified_leaf.read_payload_bytes("selected-sequence-ids.jsonl")
    selected_ids = _selected_ids_from_payload(
        selected_payload,
        allow_empty=frozen_run.policy == NO_QUERY,
    )
    selected_digest = _id_stream_sha256(
        selected_ids,
        label="selected reveal sequence IDs",
    )
    if (
        selected_ids != commitment.selected_sequence_ids
        or len(selected_ids) != reveal_row.selected_sequence_count
        or selected_digest != reveal_row.selected_sequence_ids_sha256
    ):
        raise ValueError("selected reveal IDs differ from the authenticated campaign")

    context_payload = verified_leaf.read_payload_bytes("contexts.jsonl")
    contexts = _contexts_from_payload(
        context_payload,
        allow_empty=frozen_run.policy == NO_QUERY,
    )
    example_ids = tuple(row.example_id for row in contexts)
    example_digest = _id_stream_sha256(example_ids, label="selected reveal example IDs")
    counts = Counter(row.target for row in contexts)
    target_counts = tuple((target, counts[target]) for target in TARGETS)
    if (
        any(row.fold != frozen_run.rotation.pool_fold for row in contexts)
        or {row.sequence_id for row in contexts} != set(selected_ids)
        or len(contexts) != reveal_row.revealed_context_count
        or example_digest != reveal_row.revealed_example_ids_sha256
        or target_counts != reveal_row.per_target_revealed_context_count
        or sum(row.source_observations for row in contexts)
        != reveal_row.revealed_source_observations
    ):
        raise ValueError("selected reveal contexts differ from the authenticated campaign")

    expected_summary = _summary_document(
        run=frozen_run,
        selection_barrier_seal_sha256=selection_seal,
        commitment_leaf_seal_sha256=commitment_row.leaf_seal_sha256,
        commitment_payload=commitment_payload,
        pool_outcome_vault_seal_sha256=reveal_row.pool_outcome_vault_seal_sha256,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
    )
    summary_payload = verified_leaf.read_payload_bytes("reveal-summary.json")
    _strict_json_object(summary_payload, label="selected reveal summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("selected reveal summary differs from authenticated contents")

    return SelectedPoolRevealCapability(
        run=frozen_run,
        selected_sequence_ids=selected_ids,
        contexts=contexts,
        allowed_example_ids=example_ids,
        selected_sequence_ids_sha256=selected_digest,
        revealed_example_ids_sha256=example_digest,
        commitment_leaf_seal_sha256=commitment_row.leaf_seal_sha256,
        reveal_leaf_seal_sha256=verified_leaf.seal_sha256,
        reveal_campaign_seal_sha256=authorized_campaign.seal.seal_sha256,
    )


def pool_reveal_with_commitment_from_campaign(
    campaign: RevealCampaignCapability,
    *,
    run: PolicyRunSpec,
    reveal_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
) -> FinalizePoolRevealCapability:
    """Materialize one globally authenticated reveal and commitment trace."""

    reveal = pool_reveal_from_campaign(
        campaign,
        run=run,
        reveal_seal=reveal_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    # The call above returns only after this exact leaf, its commitment payload,
    # and the select/reveal campaign indices have all agreed.  The immutable
    # captured bytes can therefore be decoded once more without accepting a
    # caller-supplied commitment value.
    commitment_payload = reveal_seal.read_payload_bytes("commitment.json")
    commitment = decode_pool_commitment(commitment_payload)
    if (
        canonical_json_bytes(commitment.document()) != commitment_payload
        or commitment.run != reveal.run
        or commitment.selected_sequence_ids != reveal.selected_sequence_ids
    ):
        raise ValueError("finalize commitment differs from authenticated reveal")
    return FinalizePoolRevealCapability(reveal=reveal, commitment=commitment)


__all__ = [
    "POOL_REVEAL_ARTIFACT",
    "POOL_REVEAL_PAYLOAD_PATHS",
    "POOL_REVEAL_SUMMARY_ARTIFACT",
    "REVEAL_CAMPAIGN_ARTIFACT",
    "REVEAL_CAMPAIGN_PAYLOAD_PATHS",
    "REVEAL_LEAF_ATTESTATION_ARTIFACT",
    "FinalizePoolRevealCapability",
    "RevealCampaignCapability",
    "RevealIndexRow",
    "RevealLeafAttestation",
    "SelectedPoolRevealCapability",
    "pool_reveal_from_campaign",
    "pool_reveal_relative_path",
    "pool_reveal_with_commitment_from_campaign",
    "publish_no_query_pool_reveal",
    "publish_nonempty_pool_reveal",
    "publish_pool_reveal_campaign_barrier",
    "reveal_campaign_capability_from_seal",
    "reveal_leaf_attestation_from_bytes",
    "reveal_leaf_attestation_from_document",
    "verify_no_query_pool_reveal_phase_capability",
    "verify_nonempty_pool_reveal_phase_capability",
    "verify_reveal_campaign_capability",
]

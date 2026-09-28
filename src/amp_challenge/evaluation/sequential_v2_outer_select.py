"""Outcome-free outer-selection commitments and their global barrier.

Every policy track receives exactly one authenticated selector-only view from
``update/global``.  A leaf worker re-derives the frozen outer-mean decision,
publishes three byte-exact payloads, and releases only a payload-free
attestation.  The global worker accepts the 220 attestations plus a separately
observed ordered leaf-digest tuple; it never receives a leaf capability.

The resulting campaign capability is label-free.  A later finalization worker
can use :func:`outer_selection_from_campaign` to authenticate and decode one
selected-ID projection without being granted any sibling selection leaf.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.acquisition.sequential_v2_selector import (
    OUTER_MEAN,
    OuterMeanCandidate,
    SelectedSeat,
    SelectionResult,
    select_outer_mean,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_OUTER_CANDIDATES,
    EXPECTED_OUTER_COMMITTED_ASSOCIATIONS,
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    PolicyRunSpec,
    ordered_policy_runs,
    policy_run_by_track_id,
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
from amp_challenge.evaluation.sequential_v2_select import selection_result_document
from amp_challenge.evaluation.sequential_v2_update import OUTER_VIEW_PAYLOAD_PATHS
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
    UpdateOuterViewIndexRow,
    outer_view_from_campaign,
    verify_update_campaign_capability,
)

SCHEMA_VERSION = 1
OUTER_SELECTION_ARTIFACT = "sequential_v2_outer_selection_commitment_v1"
OUTER_SELECTION_COMMITMENT_ARTIFACT = OUTER_SELECTION_ARTIFACT
OUTER_SELECTION_SUMMARY_ARTIFACT = "sequential_v2_outer_selection_summary_v1"
OUTER_SELECTION_ATTESTATION_ARTIFACT = "sequential_v2_outer_selection_attestation_v1"
OUTER_SELECTION_CAMPAIGN_ARTIFACT = "sequential_v2_outer_selection_campaign_barrier_v1"
OUTER_SELECTION_CAMPAIGN_BARRIER_ARTIFACT = OUTER_SELECTION_CAMPAIGN_ARTIFACT

OUTER_SELECTION_PAYLOAD_PATHS = (
    "selected-sequence-ids.jsonl",
    "selection-result.json",
    "selection-summary.json",
)
OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS = (
    "outer-selection-index.jsonl",
    "outer-selection-summary.json",
)

EXPECTED_OUTER_SELECTION_LEAVES = EXPECTED_POLICY_RUNS
EXPECTED_OUTER_SELECTION_PREDECESSORS = EXPECTED_POLICY_RUNS + 2

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be exact text")
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
    if (
        type(value) is not dict
        or set(value) != set(fields)
        or any(type(key) is not str for key in value)
    ):
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


def _identity_document(
    value: SequentialV2PublicationIdentity,
) -> dict[str, object]:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError("outer selection requires an exact publication identity")
    return {
        "git_commit": value.git_commit,
        "code_manifest_sha256": value.code_manifest_sha256,
        "config_sha256": value.config_sha256,
        "lock_sha256": value.lock_sha256,
    }


def _identity_from_document(value: object) -> SequentialV2PublicationIdentity:
    raw = _exact_object(
        value,
        {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"},
        label="outer-selection publication identity",
    )
    git_commit = _text(raw["git_commit"], label="outer-selection git commit")
    if _GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("outer-selection git commit must be forty lowercase hex characters")
    return SequentialV2PublicationIdentity(
        git_commit=git_commit,
        code_manifest_sha256=_sha256(
            raw["code_manifest_sha256"],
            label="outer-selection code manifest",
        ),
        config_sha256=_sha256(raw["config_sha256"], label="outer-selection config"),
        lock_sha256=_sha256(raw["lock_sha256"], label="outer-selection lock"),
    )


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    if value != policy_run_by_track_id(value.track_id):
        raise ValueError(f"{label} differs from the frozen registry")
    return value


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
        label="outer-selection run",
    )
    track_id = _text(raw["track_id"], label="outer-selection track ID")
    run = policy_run_by_track_id(track_id)
    if canonical_json_bytes(raw) != canonical_json_bytes(run.document()):
        raise ValueError("outer-selection run differs from the frozen registry")
    return run


def _payload_digest_tuple(
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
    if tuple(path for path, _digest in value) != paths or len(dict(value)) != len(value):
        raise ValueError(f"{label} path inventory or order changed")
    for path, digest in value:
        _sha256(digest, label=f"{label} {path}")
    return value


def _payload_digests_from_document(
    value: object,
    *,
    paths: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    raw = _exact_object(value, set(paths), label=label)
    return tuple((path, _sha256(raw[path], label=f"{label} {path}")) for path in paths)


def _id_stream_sha256(values: Sequence[str], *, label: str) -> str:
    if type(values) not in {tuple, list}:
        raise TypeError(f"{label} must be an exact ID sequence")
    identifiers = tuple(values)
    if len(set(identifiers)) != len(identifiers) or any(
        type(value) is not str or _SHA256.fullmatch(value) is None for value in identifiers
    ):
        raise ValueError(f"{label} must contain unique lowercase SHA-256 values")
    return sha256_bytes("".join(f"{value}\n" for value in identifiers).encode("ascii"))


def _selected_ids_payload(identifiers: tuple[str, ...]) -> bytes:
    if type(identifiers) is not tuple:
        raise TypeError("outer selected IDs must be an exact tuple")
    if len(identifiers) != 10:
        raise ValueError("outer selection must contain exactly ten IDs")
    _id_stream_sha256(identifiers, label="outer selected IDs")
    return canonical_jsonl_bytes({"sequence_id": value} for value in identifiers)


def _selected_ids_from_payload(payload: bytes) -> tuple[str, ...]:
    identifiers: list[str] = []
    for index, raw in enumerate(_strict_jsonl(payload, label="outer selected sequence IDs")):
        row = _exact_object(
            raw,
            {"sequence_id"},
            label=f"outer selected sequence ID row {index}",
        )
        identifiers.append(_sha256(row["sequence_id"], label=f"outer selected sequence ID {index}"))
    result = tuple(identifiers)
    if _selected_ids_payload(result) != payload:
        raise ValueError("outer selected sequence-ID payload is not its exact reconstruction")
    return result


def _component_census(
    *,
    selected_count: int,
    unique_count: object,
    max_occupancy: object,
    label: str,
) -> tuple[int, int]:
    """Validate that an exact component-count multiset can have this census."""

    selected = _exact_int(selected_count, label=f"{label} selected count", minimum=1)
    unique = _exact_int(unique_count, label=f"{label} unique components", minimum=1)
    occupancy = _exact_int(
        max_occupancy,
        label=f"{label} maximum component occupancy",
        minimum=1,
    )
    minimum_unique = (selected + occupancy - 1) // occupancy
    maximum_unique = selected - occupancy + 1
    if occupancy > 2 or not minimum_unique <= unique <= maximum_unique:
        raise ValueError(f"{label} component census is impossible under the strict cap")
    return unique, occupancy


def outer_selection_relative_path(run: PolicyRunSpec) -> str:
    """Return the frozen logical path of one outer-selection commitment."""

    frozen = _require_frozen_run(run, label="outer-selection path run")
    return f"outer-select/tracks/{frozen.track_id}"


def _protocol_predecessor() -> str:
    return "protocol/SHA256SUMS"


def _update_predecessor() -> str:
    return "update/global/SHA256SUMS"


def _outer_view_predecessor(run: PolicyRunSpec) -> str:
    frozen = _require_frozen_run(run, label="outer-selection predecessor run")
    return f"update/tracks/{frozen.track_id}/outer-view/SHA256SUMS"


def _selection_predecessor(run: PolicyRunSpec) -> str:
    return f"{outer_selection_relative_path(run)}/SHA256SUMS"


def _leaf_predecessors(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    update_global_seal_sha256: str,
    outer_view_leaf_seal_sha256: str,
) -> dict[str, str]:
    result = {
        _protocol_predecessor(): _sha256(
            protocol_seal_sha256,
            label="outer-selection protocol seal",
        ),
        _update_predecessor(): _sha256(
            update_global_seal_sha256,
            label="outer-selection update-global seal",
        ),
        _outer_view_predecessor(run): _sha256(
            outer_view_leaf_seal_sha256,
            label="outer-selection outer-view leaf seal",
        ),
    }
    if len(result) != 3:
        raise AssertionError("outer-selection leaf predecessor census changed")
    return result


def _summary_document_from_safe_fields(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    update_global_seal_sha256: str,
    outer_view_leaf_seal_sha256: str,
    outer_view_payload_sha256: tuple[tuple[str, str], ...],
    outer_candidate_count: int,
    outer_candidate_ids_sha256: str,
    selected_sequence_count: int,
    selected_sequence_ids_sha256: str,
    selected_unique_component_count: int,
    selected_max_component_occupancy: int,
    selected_sequence_ids_payload_sha256: str,
    selection_result_payload_sha256: str,
) -> dict[str, object]:
    frozen = _require_frozen_run(run, label="outer-selection summary run")
    view_payloads = _payload_digest_tuple(
        outer_view_payload_sha256,
        paths=OUTER_VIEW_PAYLOAD_PATHS,
        label="outer-selection summary outer-view payload digests",
    )
    expected_candidates = EXPECTED_SUPPORT_BY_FOLD[frozen.rotation.outer_fold]
    candidate_count = _exact_int(
        outer_candidate_count,
        label="outer-selection summary candidate count",
        minimum=1,
    )
    if candidate_count != expected_candidates:
        raise ValueError("outer-selection summary candidate census changed")
    selected_count = _exact_int(
        selected_sequence_count,
        label="outer-selection summary selected count",
        minimum=1,
    )
    if selected_count != frozen.expected_outer_selection_count or selected_count != 10:
        raise ValueError("outer-selection summary selected count must be exactly ten")
    unique_components, max_occupancy = _component_census(
        selected_count=selected_count,
        unique_count=selected_unique_component_count,
        max_occupancy=selected_max_component_occupancy,
        label="outer-selection summary",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_SELECTION_SUMMARY_ARTIFACT,
        "run": frozen.document(),
        "selector_kind": OUTER_MEAN,
        "protocol_seal_sha256": _sha256(
            protocol_seal_sha256,
            label="outer-selection summary protocol seal",
        ),
        "update_global_seal_sha256": _sha256(
            update_global_seal_sha256,
            label="outer-selection summary update-global seal",
        ),
        "outer_view_leaf_seal_sha256": _sha256(
            outer_view_leaf_seal_sha256,
            label="outer-selection summary outer-view leaf",
        ),
        "outer_view_payload_sha256": dict(view_payloads),
        "outer_candidate_count": candidate_count,
        "outer_candidate_ids_sha256": _sha256(
            outer_candidate_ids_sha256,
            label="outer-selection summary candidate IDs",
        ),
        "selected_sequence_count": selected_count,
        "selected_sequence_ids_sha256": _sha256(
            selected_sequence_ids_sha256,
            label="outer-selection summary selected IDs",
        ),
        "selected_unique_component_count": unique_components,
        "selected_max_component_occupancy": max_occupancy,
        "selected_sequence_ids_payload_sha256": _sha256(
            selected_sequence_ids_payload_sha256,
            label="outer-selection summary selected-ID payload",
        ),
        "selection_result_payload_sha256": _sha256(
            selection_result_payload_sha256,
            label="outer-selection summary result payload",
        ),
    }


def _attested_leaf_seal_sha256(attestation: OuterSelectionAttestation) -> str:
    if type(attestation) is not OuterSelectionAttestation:
        raise TypeError("outer-selection reconstruction requires an exact attestation")
    payloads = dict(
        _payload_digest_tuple(
            attestation.payload_sha256,
            paths=OUTER_SELECTION_PAYLOAD_PATHS,
            label="outer-selection attestation payload digests",
        )
    )
    receipt = canonical_json_bytes(
        {
            "artifact": OUTER_SELECTION_ARTIFACT,
            "metadata": attestation.publication_identity.metadata(
                phase="outer-select",
                scope_id=attestation.run.track_id,
            ),
            "payloads": payloads,
            "predecessor_seals": _leaf_predecessors(
                run=attestation.run,
                protocol_seal_sha256=attestation.protocol_seal_sha256,
                update_global_seal_sha256=attestation.update_global_seal_sha256,
                outer_view_leaf_seal_sha256=attestation.outer_view_leaf_seal_sha256,
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


@dataclass(frozen=True, slots=True)
class OuterSelectionAttestation:
    """Payload-free result from one isolated outer-selection worker."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    update_global_seal_sha256: str
    outer_view_leaf_seal_sha256: str
    outer_view_payload_sha256: tuple[tuple[str, str], ...]
    outer_candidate_count: int
    outer_candidate_ids_sha256: str
    selection_leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    selected_sequence_count: int
    selected_sequence_ids_sha256: str
    selected_unique_component_count: int
    selected_max_component_occupancy: int

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="outer-selection attestation run")
        _identity_document(self.publication_identity)
        _sha256(self.protocol_seal_sha256, label="outer-selection attestation protocol")
        _sha256(
            self.update_global_seal_sha256,
            label="outer-selection attestation update-global",
        )
        _sha256(
            self.outer_view_leaf_seal_sha256,
            label="outer-selection attestation outer-view leaf",
        )
        _payload_digest_tuple(
            self.outer_view_payload_sha256,
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer-selection attestation outer-view payload digests",
        )
        _sha256(
            self.outer_candidate_ids_sha256,
            label="outer-selection attestation candidate IDs",
        )
        _sha256(
            self.selection_leaf_seal_sha256,
            label="outer-selection attestation leaf",
        )
        payloads = dict(
            _payload_digest_tuple(
                self.payload_sha256,
                paths=OUTER_SELECTION_PAYLOAD_PATHS,
                label="outer-selection attestation payload digests",
            )
        )
        expected_summary = _summary_document_from_safe_fields(
            run=self.run,
            protocol_seal_sha256=self.protocol_seal_sha256,
            update_global_seal_sha256=self.update_global_seal_sha256,
            outer_view_leaf_seal_sha256=self.outer_view_leaf_seal_sha256,
            outer_view_payload_sha256=self.outer_view_payload_sha256,
            outer_candidate_count=self.outer_candidate_count,
            outer_candidate_ids_sha256=self.outer_candidate_ids_sha256,
            selected_sequence_count=self.selected_sequence_count,
            selected_sequence_ids_sha256=self.selected_sequence_ids_sha256,
            selected_unique_component_count=self.selected_unique_component_count,
            selected_max_component_occupancy=self.selected_max_component_occupancy,
            selected_sequence_ids_payload_sha256=payloads["selected-sequence-ids.jsonl"],
            selection_result_payload_sha256=payloads["selection-result.json"],
        )
        if payloads["selection-summary.json"] != sha256_bytes(
            canonical_json_bytes(expected_summary)
        ):
            raise ValueError(
                "outer-selection attestation safe fields differ from its summary digest"
            )
        if _attested_leaf_seal_sha256(self) != self.selection_leaf_seal_sha256:
            raise ValueError(
                "outer-selection attestation does not reconstruct its authoritative leaf seal"
            )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": OUTER_SELECTION_ATTESTATION_ARTIFACT,
            "run": self.run.document(),
            "publication_identity": _identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "update_global_seal_sha256": self.update_global_seal_sha256,
            "outer_view_leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "outer_view_payload_sha256": dict(self.outer_view_payload_sha256),
            "outer_candidate_count": self.outer_candidate_count,
            "outer_candidate_ids_sha256": self.outer_candidate_ids_sha256,
            "selection_leaf_seal_sha256": self.selection_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "selected_unique_component_count": self.selected_unique_component_count,
            "selected_max_component_occupancy": self.selected_max_component_occupancy,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())

    def index_document(self) -> dict[str, object]:
        return OuterSelectionIndexRow.from_attestation(self).document()


def outer_selection_attestation_from_document(
    value: object,
) -> OuterSelectionAttestation:
    """Strictly decode one payload-free outer-selection worker result."""

    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "publication_identity",
            "protocol_seal_sha256",
            "update_global_seal_sha256",
            "outer_view_leaf_seal_sha256",
            "outer_view_payload_sha256",
            "outer_candidate_count",
            "outer_candidate_ids_sha256",
            "selection_leaf_seal_sha256",
            "payload_sha256",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "selected_unique_component_count",
            "selected_max_component_occupancy",
        },
        label="outer-selection attestation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["artifact"]) is not str
        or raw["artifact"] != OUTER_SELECTION_ATTESTATION_ARTIFACT
    ):
        raise ValueError("outer-selection attestation identity changed")
    attestation = OuterSelectionAttestation(
        run=_run_from_document(raw["run"]),
        publication_identity=_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(
            raw["protocol_seal_sha256"],
            label="outer-selection attestation protocol",
        ),
        update_global_seal_sha256=_sha256(
            raw["update_global_seal_sha256"],
            label="outer-selection attestation update-global",
        ),
        outer_view_leaf_seal_sha256=_sha256(
            raw["outer_view_leaf_seal_sha256"],
            label="outer-selection attestation outer-view leaf",
        ),
        outer_view_payload_sha256=_payload_digests_from_document(
            raw["outer_view_payload_sha256"],
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer-selection attestation outer-view payload digests",
        ),
        outer_candidate_count=_exact_int(
            raw["outer_candidate_count"],
            label="outer-selection attestation candidates",
            minimum=1,
        ),
        outer_candidate_ids_sha256=_sha256(
            raw["outer_candidate_ids_sha256"],
            label="outer-selection attestation candidate IDs",
        ),
        selection_leaf_seal_sha256=_sha256(
            raw["selection_leaf_seal_sha256"],
            label="outer-selection attestation leaf",
        ),
        payload_sha256=_payload_digests_from_document(
            raw["payload_sha256"],
            paths=OUTER_SELECTION_PAYLOAD_PATHS,
            label="outer-selection attestation payload digests",
        ),
        selected_sequence_count=_exact_int(
            raw["selected_sequence_count"],
            label="outer-selection attestation selected count",
            minimum=1,
        ),
        selected_sequence_ids_sha256=_sha256(
            raw["selected_sequence_ids_sha256"],
            label="outer-selection attestation selected IDs",
        ),
        selected_unique_component_count=_exact_int(
            raw["selected_unique_component_count"],
            label="outer-selection attestation unique components",
            minimum=1,
        ),
        selected_max_component_occupancy=_exact_int(
            raw["selected_max_component_occupancy"],
            label="outer-selection attestation maximum component occupancy",
            minimum=1,
        ),
    )
    if canonical_json_bytes(raw) != attestation.canonical_bytes():
        raise ValueError("outer-selection attestation does not round-trip exactly")
    return attestation


def outer_selection_attestation_from_bytes(payload: bytes) -> OuterSelectionAttestation:
    """Decode one canonical payload-free outer-selection attestation."""

    return outer_selection_attestation_from_document(
        _strict_json_object(payload, label="outer-selection attestation")
    )


@dataclass(frozen=True, slots=True)
class OuterSelectionIndexRow:
    """One safe global-index record for an authenticated selection leaf."""

    run: PolicyRunSpec
    leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    outer_view_leaf_seal_sha256: str
    outer_view_payload_sha256: tuple[tuple[str, str], ...]
    outer_candidate_count: int
    outer_candidate_ids_sha256: str
    selected_sequence_count: int
    selected_sequence_ids_sha256: str
    selected_unique_component_count: int
    selected_max_component_occupancy: int

    def __post_init__(self) -> None:
        frozen = _require_frozen_run(self.run, label="outer-selection index run")
        _sha256(self.leaf_seal_sha256, label="outer-selection index leaf")
        _payload_digest_tuple(
            self.payload_sha256,
            paths=OUTER_SELECTION_PAYLOAD_PATHS,
            label="outer-selection index payload digests",
        )
        _sha256(
            self.outer_view_leaf_seal_sha256,
            label="outer-selection index outer-view leaf",
        )
        _payload_digest_tuple(
            self.outer_view_payload_sha256,
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer-selection index outer-view payload digests",
        )
        expected_candidates = EXPECTED_SUPPORT_BY_FOLD[frozen.rotation.outer_fold]
        if (
            type(self.outer_candidate_count) is not int
            or self.outer_candidate_count != expected_candidates
        ):
            raise ValueError("outer-selection index candidate census changed")
        _sha256(
            self.outer_candidate_ids_sha256,
            label="outer-selection index candidate IDs",
        )
        if type(self.selected_sequence_count) is not int or self.selected_sequence_count != 10:
            raise ValueError("outer-selection index selected count must be exactly ten")
        _sha256(
            self.selected_sequence_ids_sha256,
            label="outer-selection index selected IDs",
        )
        _component_census(
            selected_count=self.selected_sequence_count,
            unique_count=self.selected_unique_component_count,
            max_occupancy=self.selected_max_component_occupancy,
            label="outer-selection index",
        )

    @classmethod
    def from_attestation(
        cls,
        attestation: OuterSelectionAttestation,
    ) -> OuterSelectionIndexRow:
        if type(attestation) is not OuterSelectionAttestation:
            raise TypeError("outer-selection index requires an exact attestation")
        return cls(
            run=attestation.run,
            leaf_seal_sha256=attestation.selection_leaf_seal_sha256,
            payload_sha256=attestation.payload_sha256,
            outer_view_leaf_seal_sha256=attestation.outer_view_leaf_seal_sha256,
            outer_view_payload_sha256=attestation.outer_view_payload_sha256,
            outer_candidate_count=attestation.outer_candidate_count,
            outer_candidate_ids_sha256=attestation.outer_candidate_ids_sha256,
            selected_sequence_count=attestation.selected_sequence_count,
            selected_sequence_ids_sha256=attestation.selected_sequence_ids_sha256,
            selected_unique_component_count=attestation.selected_unique_component_count,
            selected_max_component_occupancy=attestation.selected_max_component_occupancy,
        )

    @property
    def relative_path(self) -> str:
        return outer_selection_relative_path(self.run)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run": self.run.document(),
            "relative_path": self.relative_path,
            "leaf_artifact": OUTER_SELECTION_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "outer_view_leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "outer_view_payload_sha256": dict(self.outer_view_payload_sha256),
            "outer_candidate_count": self.outer_candidate_count,
            "outer_candidate_ids_sha256": self.outer_candidate_ids_sha256,
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "selected_unique_component_count": self.selected_unique_component_count,
            "selected_max_component_occupancy": self.selected_max_component_occupancy,
        }


def _index_row_from_document(value: object) -> OuterSelectionIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "run",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "payload_sha256",
            "outer_view_leaf_seal_sha256",
            "outer_view_payload_sha256",
            "outer_candidate_count",
            "outer_candidate_ids_sha256",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "selected_unique_component_count",
            "selected_max_component_occupancy",
        },
        label="outer-selection index row",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("outer-selection index schema version changed")
    run = _run_from_document(raw["run"])
    if (
        _text(raw["relative_path"], label="outer-selection index path")
        != outer_selection_relative_path(run)
        or _text(raw["leaf_artifact"], label="outer-selection index artifact")
        != OUTER_SELECTION_ARTIFACT
    ):
        raise ValueError("outer-selection index leaf identity changed")
    row = OuterSelectionIndexRow(
        run=run,
        leaf_seal_sha256=_sha256(
            raw["leaf_seal_sha256"],
            label="outer-selection index leaf",
        ),
        payload_sha256=_payload_digests_from_document(
            raw["payload_sha256"],
            paths=OUTER_SELECTION_PAYLOAD_PATHS,
            label="outer-selection index payload digests",
        ),
        outer_view_leaf_seal_sha256=_sha256(
            raw["outer_view_leaf_seal_sha256"],
            label="outer-selection index outer-view leaf",
        ),
        outer_view_payload_sha256=_payload_digests_from_document(
            raw["outer_view_payload_sha256"],
            paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer-selection index outer-view payload digests",
        ),
        outer_candidate_count=_exact_int(
            raw["outer_candidate_count"],
            label="outer-selection index candidates",
            minimum=1,
        ),
        outer_candidate_ids_sha256=_sha256(
            raw["outer_candidate_ids_sha256"],
            label="outer-selection index candidate IDs",
        ),
        selected_sequence_count=_exact_int(
            raw["selected_sequence_count"],
            label="outer-selection index selected count",
            minimum=1,
        ),
        selected_sequence_ids_sha256=_sha256(
            raw["selected_sequence_ids_sha256"],
            label="outer-selection index selected IDs",
        ),
        selected_unique_component_count=_exact_int(
            raw["selected_unique_component_count"],
            label="outer-selection index unique components",
            minimum=1,
        ),
        selected_max_component_occupancy=_exact_int(
            raw["selected_max_component_occupancy"],
            label="outer-selection index maximum component occupancy",
            minimum=1,
        ),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("outer-selection index row does not round-trip exactly")
    return row


@dataclass(frozen=True, slots=True)
class _SelectionMaterial:
    row: UpdateOuterViewIndexRow
    candidates: tuple[OuterMeanCandidate, ...]
    result: SelectionResult
    selected_sequence_ids: tuple[str, ...]
    selected_unique_component_count: int
    selected_max_component_occupancy: int
    payloads: tuple[tuple[str, bytes], ...]


def _validate_outer_result(
    result: SelectionResult,
    *,
    run: PolicyRunSpec,
    candidates: tuple[OuterMeanCandidate, ...],
) -> tuple[int, int]:
    if type(result) is not SelectionResult:
        raise TypeError("outer selector must return an exact SelectionResult")
    seats = tuple(result.seats)
    selected_ids = tuple(item.sequence_id for item in seats)
    if (
        result.rotation_id != run.rotation.rotation_id
        or result.policy != OUTER_MEAN
        or len(seats) != 10
        or any(type(item) is not SelectedSeat for item in seats)
        or len(set(selected_ids)) != 10
        or result.mean_control_sequence_ids != selected_ids
        or result.requested_sequence_ids != selected_ids
        or result.requested_complete is not True
        or dict(result.requested_role_counts) != {"exploit": 10}
        or dict(result.applied_role_counts) != {"exploit": 10}
        or result.scalar_loss is not None
        or result.objective_losses is not None
        or tuple(result.repair_trace)
        or tuple(result.guard_evaluations)
        or result.fallback_reason is not None
    ):
        raise ValueError("outer selection result differs from frozen outer-mean semantics")
    candidates_by_id = {item.sequence_id: item for item in candidates}
    if len(candidates_by_id) != len(candidates) or any(
        sequence_id not in candidates_by_id for sequence_id in selected_ids
    ):
        raise ValueError("outer selection result names a candidate outside its input view")
    component_counts = Counter(
        candidates_by_id[sequence_id].diversity_component_id for sequence_id in selected_ids
    )
    if dict(result.component_counts) != dict(component_counts):
        raise ValueError("outer selection component counts differ from selected candidates")
    for seat in seats:
        candidate = candidates_by_id[seat.sequence_id]
        if (
            seat.requested_role != "exploit"
            or seat.applied_role != "exploit"
            or type(seat.acquisition_score) is not float
            or type(seat.scalar_mean) is not float
            or not math.isfinite(seat.acquisition_score)
            or seat.acquisition_score != candidate.scalar_mean
            or seat.scalar_mean != candidate.scalar_mean
        ):
            raise ValueError("outer selection seat differs from its raw mean score")
    unique_count = len(component_counts)
    max_occupancy = max(component_counts.values())
    if unique_count * 2 < 10 or max_occupancy > 2:
        raise ValueError("outer selection violates the strict component cap")
    return unique_count, max_occupancy


def _authenticate_authorities(
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> tuple[ProtocolCapability, UpdateCampaignCapability]:
    _identity_document(publication_identity)
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("outer selection requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    if type(update_campaign) is not UpdateCampaignCapability:
        raise TypeError("outer selection requires an exact UpdateCampaignCapability")
    update = verify_update_campaign_capability(
        update_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    return protocol, update


def _snapshot_update_outer_view_rows(
    update: UpdateCampaignCapability,
) -> tuple[UpdateOuterViewIndexRow, ...]:
    """Decode one update-global index into its exact ordered selector rows once."""

    rows = update.index_rows()
    views = tuple(row for row in rows if type(row) is UpdateOuterViewIndexRow)
    if (
        len(views) != EXPECTED_POLICY_RUNS
        or tuple(row.run for row in views) != ordered_policy_runs()
    ):
        raise ValueError("update campaign outer-view snapshot differs from frozen track order")
    return views


def _selection_material(
    *,
    run: PolicyRunSpec,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> tuple[ProtocolCapability, UpdateCampaignCapability, _SelectionMaterial]:
    frozen = _require_frozen_run(run, label="outer-selection run")
    if type(outer_view_seal) is not PhaseSeal:
        raise TypeError("outer selection requires an exact outer-view PhaseSeal")
    protocol, update = _authenticate_authorities(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    row = update.outer_view_row(run=frozen)
    candidates = outer_view_from_campaign(
        update,
        run=frozen,
        outer_view_seal=outer_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    result = select_outer_mean(
        candidates,
        outer_fold=frozen.rotation.outer_fold,
        pool_fold=frozen.rotation.pool_fold,
    )
    unique_components, max_occupancy = _validate_outer_result(
        result,
        run=frozen,
        candidates=candidates,
    )
    selected_ids = result.sequence_ids
    selected_payload = _selected_ids_payload(selected_ids)
    result_payload = canonical_json_bytes(selection_result_document(result))
    summary_payload = canonical_json_bytes(
        _summary_document_from_safe_fields(
            run=frozen,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
            outer_view_leaf_seal_sha256=row.leaf_seal_sha256,
            outer_view_payload_sha256=row.payload_sha256,
            outer_candidate_count=row.candidate_count,
            outer_candidate_ids_sha256=row.candidate_ids_sha256,
            selected_sequence_count=len(selected_ids),
            selected_sequence_ids_sha256=_id_stream_sha256(
                selected_ids,
                label="outer selected IDs",
            ),
            selected_unique_component_count=unique_components,
            selected_max_component_occupancy=max_occupancy,
            selected_sequence_ids_payload_sha256=sha256_bytes(selected_payload),
            selection_result_payload_sha256=sha256_bytes(result_payload),
        )
    )
    payload_map = {
        "selected-sequence-ids.jsonl": selected_payload,
        "selection-result.json": result_payload,
        "selection-summary.json": summary_payload,
    }
    return (
        protocol,
        update,
        _SelectionMaterial(
            row=row,
            candidates=candidates,
            result=result,
            selected_sequence_ids=selected_ids,
            selected_unique_component_count=unique_components,
            selected_max_component_occupancy=max_occupancy,
            payloads=tuple((path, payload_map[path]) for path in OUTER_SELECTION_PAYLOAD_PATHS),
        ),
    )


def _attestation_from_verified_leaf(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    update_global_seal_sha256: str,
    material: _SelectionMaterial,
) -> OuterSelectionAttestation:
    return OuterSelectionAttestation(
        run=run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        update_global_seal_sha256=update_global_seal_sha256,
        outer_view_leaf_seal_sha256=material.row.leaf_seal_sha256,
        outer_view_payload_sha256=material.row.payload_sha256,
        outer_candidate_count=material.row.candidate_count,
        outer_candidate_ids_sha256=material.row.candidate_ids_sha256,
        selection_leaf_seal_sha256=seal.seal_sha256,
        payload_sha256=seal.payload_sha256,
        selected_sequence_count=len(material.selected_sequence_ids),
        selected_sequence_ids_sha256=_id_stream_sha256(
            material.selected_sequence_ids,
            label="outer-selection attestation selected IDs",
        ),
        selected_unique_component_count=material.selected_unique_component_count,
        selected_max_component_occupancy=material.selected_max_component_occupancy,
    )


def publish_outer_selection_commitment(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> OuterSelectionAttestation:
    """Re-derive and atomically publish one exact outer-mean commitment."""

    frozen = _require_frozen_run(run, label="outer-selection publication run")
    protocol, update, material = _selection_material(
        run=frozen,
        update_campaign=update_campaign,
        outer_view_seal=outer_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    seal = publish_phase(
        destination,
        artifact=OUTER_SELECTION_ARTIFACT,
        payloads=dict(material.payloads),
        predecessor_seals=_leaf_predecessors(
            run=frozen,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
            outer_view_leaf_seal_sha256=material.row.leaf_seal_sha256,
        ),
        metadata=publication_identity.metadata(
            phase="outer-select",
            scope_id=frozen.track_id,
        ),
    )
    return verify_outer_selection_commitment(
        seal,
        run=frozen,
        update_campaign=update,
        outer_view_seal=outer_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=update.seal.seal_sha256,
        expected_outer_selection_leaf_seal_sha256=seal.seal_sha256,
    )


def verify_outer_selection_commitment(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_leaf_seal_sha256: str,
) -> OuterSelectionAttestation:
    """Re-derive all inputs and require every leaf payload byte to match."""

    frozen = _require_frozen_run(run, label="outer-selection verification run")
    protocol, update, material = _selection_material(
        run=frozen,
        update_campaign=update_campaign,
        outer_view_seal=outer_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=OUTER_SELECTION_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
        expected_predecessor_seals=_leaf_predecessors(
            run=frozen,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
            outer_view_leaf_seal_sha256=material.row.leaf_seal_sha256,
        ),
        expected_seal_sha256=_sha256(
            expected_outer_selection_leaf_seal_sha256,
            label="expected outer-selection leaf seal",
        ),
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="outer-select",
        scope_id=frozen.track_id,
    )
    for path, expected in material.payloads:
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"outer-selection payload differs from re-derived result: {path}")
    return _attestation_from_verified_leaf(
        verified,
        run=frozen,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        update_global_seal_sha256=update.seal.seal_sha256,
        material=material,
    )


def _campaign_summary_document(
    rows: tuple[OuterSelectionIndexRow, ...],
    *,
    index_sha256: str,
) -> dict[str, object]:
    if (
        type(rows) is not tuple
        or len(rows) != EXPECTED_POLICY_RUNS
        or any(type(row) is not OuterSelectionIndexRow for row in rows)
        or tuple(row.run for row in rows) != ordered_policy_runs()
    ):
        raise ValueError("outer-selection campaign requires 220 exact rows in frozen order")
    if len({row.leaf_seal_sha256 for row in rows}) != EXPECTED_POLICY_RUNS:
        raise ValueError("outer-selection campaign leaf seals must be entirely distinct")
    candidate_count = sum(row.outer_candidate_count for row in rows)
    selected_count = sum(row.selected_sequence_count for row in rows)
    unique_component_count = sum(row.selected_unique_component_count for row in rows)
    max_component_occupancy = max(row.selected_max_component_occupancy for row in rows)
    rotation_count = len({row.run.rotation for row in rows})
    if (
        rotation_count != EXPECTED_ROTATIONS
        or candidate_count != EXPECTED_OUTER_CANDIDATES
        or selected_count != EXPECTED_OUTER_COMMITTED_ASSOCIATIONS
    ):
        raise ValueError("outer-selection campaign census differs from the frozen graph")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        "rotation_count": rotation_count,
        "track_count": len(rows),
        "selection_leaf_count": len(rows),
        "outer_candidate_association_count": candidate_count,
        "selected_sequence_association_count": selected_count,
        "selected_unique_component_association_count": unique_component_count,
        "selected_max_component_occupancy": max_component_occupancy,
        "outer_selection_index_sha256": _sha256(
            index_sha256,
            label="outer-selection campaign index digest",
        ),
    }


def _campaign_predecessors(
    rows: tuple[OuterSelectionIndexRow, ...],
    *,
    protocol_seal_sha256: str,
    update_global_seal_sha256: str,
) -> dict[str, str]:
    _campaign_summary_document(rows, index_sha256="0" * 64)
    result = {
        _protocol_predecessor(): _sha256(
            protocol_seal_sha256,
            label="outer-selection campaign protocol seal",
        ),
        _update_predecessor(): _sha256(
            update_global_seal_sha256,
            label="outer-selection campaign update-global seal",
        ),
        **{_selection_predecessor(row.run): row.leaf_seal_sha256 for row in rows},
    }
    if len(result) != EXPECTED_OUTER_SELECTION_PREDECESSORS:
        raise ValueError("outer-selection campaign predecessor census must be exactly 222")
    return result


def _campaign_payloads(
    rows: tuple[OuterSelectionIndexRow, ...],
) -> dict[str, bytes]:
    index_payload = canonical_jsonl_bytes(row.document() for row in rows)
    return {
        "outer-selection-index.jsonl": index_payload,
        "outer-selection-summary.json": canonical_json_bytes(
            _campaign_summary_document(
                rows,
                index_sha256=sha256_bytes(index_payload),
            )
        ),
    }


def _decode_campaign_index(payload: bytes) -> tuple[OuterSelectionIndexRow, ...]:
    rows = tuple(
        _index_row_from_document(raw)
        for raw in _strict_jsonl(payload, label="outer-selection campaign index")
    )
    if len(rows) != EXPECTED_POLICY_RUNS or tuple(row.run for row in rows) != ordered_policy_runs():
        raise ValueError("outer-selection campaign index differs from frozen track order")
    return rows


def _validate_index_row_authority(
    row: OuterSelectionIndexRow,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    update_global_seal_sha256: str,
) -> None:
    """Reconstruct one indexed leaf using only global anchors and safe fields."""

    if type(row) is not OuterSelectionIndexRow:
        raise TypeError("outer-selection leaf reconstruction requires an exact index row")
    OuterSelectionAttestation(
        run=row.run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        update_global_seal_sha256=update_global_seal_sha256,
        outer_view_leaf_seal_sha256=row.outer_view_leaf_seal_sha256,
        outer_view_payload_sha256=row.outer_view_payload_sha256,
        outer_candidate_count=row.outer_candidate_count,
        outer_candidate_ids_sha256=row.outer_candidate_ids_sha256,
        selection_leaf_seal_sha256=row.leaf_seal_sha256,
        payload_sha256=row.payload_sha256,
        selected_sequence_count=row.selected_sequence_count,
        selected_sequence_ids_sha256=row.selected_sequence_ids_sha256,
        selected_unique_component_count=row.selected_unique_component_count,
        selected_max_component_occupancy=row.selected_max_component_occupancy,
    )


def _decode_safe_campaign(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[tuple[OuterSelectionIndexRow, ...], str, str]:
    if type(seal) is not PhaseSeal:
        raise TypeError("outer-selection campaign requires an exact rootless PhaseSeal")
    _identity_document(publication_identity)
    expected = _sha256(
        expected_seal_sha256,
        label="expected outer-selection campaign seal",
    )
    preliminary = verify_phase_capability(
        seal,
        expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected,
    )
    rows = _decode_campaign_index(preliminary.read_payload_bytes("outer-selection-index.jsonl"))
    anchors = dict(preliminary.predecessor_seals)
    protocol = _sha256(
        anchors.get(_protocol_predecessor()),
        label="outer-selection campaign protocol predecessor",
    )
    update = _sha256(
        anchors.get(_update_predecessor()),
        label="outer-selection campaign update predecessor",
    )
    for row in rows:
        _validate_index_row_authority(
            row,
            publication_identity=publication_identity,
            protocol_seal_sha256=protocol,
            update_global_seal_sha256=update,
        )
    verified = verify_phase_capability(
        preliminary,
        expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=protocol,
            update_global_seal_sha256=update,
        ),
        expected_seal_sha256=expected,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="outer-select",
        scope_id="global",
    )
    index_payload = verified.read_payload_bytes("outer-selection-index.jsonl")
    expected_summary = _campaign_summary_document(
        rows,
        index_sha256=sha256_bytes(index_payload),
    )
    summary_payload = verified.read_payload_bytes("outer-selection-summary.json")
    _strict_json_object(summary_payload, label="outer-selection campaign summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("outer-selection campaign summary differs from its exact index census")
    return rows, protocol, update


@dataclass(frozen=True, slots=True)
class OuterSelectionCampaignCapability:
    """Rootless label-free 220-track outer-selection barrier."""

    seal: PhaseSeal
    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("outer-selection campaign capability requires an exact PhaseSeal")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError(
                "outer-selection campaign capability requires an exact publication identity"
            )
        _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )

    @property
    def anchor_digests(self) -> tuple[tuple[str, str], ...]:
        _rows, protocol, update = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )
        return (
            (_protocol_predecessor(), protocol),
            (_update_predecessor(), update),
        )

    def index_rows(self) -> tuple[OuterSelectionIndexRow, ...]:
        rows, _protocol, _update = _decode_safe_campaign(
            self.seal,
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.seal.seal_sha256,
        )
        return rows

    def index_row(self, *, run: PolicyRunSpec) -> OuterSelectionIndexRow:
        frozen = _require_frozen_run(run, label="outer-selection campaign lookup run")
        matches = tuple(row for row in self.index_rows() if row.run == frozen)
        if len(matches) != 1:
            raise ValueError("outer-selection campaign lacks one exact requested track")
        return matches[0]


def _cross_check_rows_against_update(
    rows: tuple[OuterSelectionIndexRow, ...],
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    update: UpdateCampaignCapability,
    update_outer_view_rows: tuple[UpdateOuterViewIndexRow, ...],
) -> None:
    if (
        tuple(row.run for row in rows) != ordered_policy_runs()
        or type(update_outer_view_rows) is not tuple
        or len(update_outer_view_rows) != EXPECTED_POLICY_RUNS
        or any(type(row) is not UpdateOuterViewIndexRow for row in update_outer_view_rows)
        or tuple(row.run for row in update_outer_view_rows) != ordered_policy_runs()
    ):
        raise ValueError("outer-selection rows differ from frozen track order")
    for row, update_row in zip(rows, update_outer_view_rows, strict=True):
        _validate_index_row_authority(
            row,
            publication_identity=publication_identity,
            protocol_seal_sha256=protocol_seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
        )
        if (
            row.outer_view_leaf_seal_sha256 != update_row.leaf_seal_sha256
            or row.outer_view_payload_sha256 != update_row.payload_sha256
            or row.outer_candidate_count != update_row.candidate_count
            or row.outer_candidate_ids_sha256 != update_row.candidate_ids_sha256
        ):
            raise ValueError("outer-selection index differs from authenticated update-global")


def _validate_campaign_inputs(
    *,
    selection_attestations: tuple[OuterSelectionAttestation, ...],
    expected_outer_selection_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> tuple[
    tuple[OuterSelectionIndexRow, ...],
    ProtocolCapability,
    UpdateCampaignCapability,
]:
    protocol, update = _authenticate_authorities(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    if type(selection_attestations) is not tuple:
        raise TypeError("outer-selection campaign attestations must be an exact tuple")
    if type(expected_outer_selection_leaf_seal_sha256s) is not tuple:
        raise TypeError("outer-selection controller leaf digests must be an exact tuple")
    if (
        len(selection_attestations) != EXPECTED_POLICY_RUNS
        or any(type(item) is not OuterSelectionAttestation for item in selection_attestations)
        or tuple(item.run for item in selection_attestations) != ordered_policy_runs()
        or len(expected_outer_selection_leaf_seal_sha256s) != EXPECTED_POLICY_RUNS
    ):
        raise ValueError("outer-selection campaign requires 220 exact attestations in frozen order")
    expected_leaves = tuple(
        _sha256(value, label=f"controller outer-selection leaf digest {index}")
        for index, value in enumerate(expected_outer_selection_leaf_seal_sha256s)
    )
    if len(set(expected_leaves)) != EXPECTED_POLICY_RUNS:
        raise ValueError("controller outer-selection leaf digests must be entirely distinct")
    update_outer_view_rows = _snapshot_update_outer_view_rows(update)
    rows: list[OuterSelectionIndexRow] = []
    for index, (attestation, expected_leaf, update_row) in enumerate(
        zip(
            selection_attestations,
            expected_leaves,
            update_outer_view_rows,
            strict=True,
        )
    ):
        reconstructed = outer_selection_attestation_from_document(attestation.document())
        if reconstructed.canonical_bytes() != attestation.canonical_bytes():
            raise ValueError(f"outer-selection attestation {index} changed during reconstruction")
        if (
            attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol.seal.seal_sha256
            or attestation.update_global_seal_sha256 != update.seal.seal_sha256
            or attestation.outer_view_leaf_seal_sha256 != update_row.leaf_seal_sha256
            or attestation.outer_view_payload_sha256 != update_row.payload_sha256
            or attestation.outer_candidate_count != update_row.candidate_count
            or attestation.outer_candidate_ids_sha256 != update_row.candidate_ids_sha256
            or attestation.selection_leaf_seal_sha256 != expected_leaf
            or _attested_leaf_seal_sha256(attestation) != expected_leaf
        ):
            raise ValueError("outer-selection attestation differs from update/controller authority")
        rows.append(OuterSelectionIndexRow.from_attestation(attestation))
    result = tuple(rows)
    _cross_check_rows_against_update(
        result,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        update=update,
        update_outer_view_rows=update_outer_view_rows,
    )
    _campaign_summary_document(result, index_sha256="0" * 64)
    return result, protocol, update


def publish_outer_selection_campaign_barrier(
    destination: str | Path,
    *,
    selection_attestations: tuple[OuterSelectionAttestation, ...],
    expected_outer_selection_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
) -> OuterSelectionCampaignCapability:
    """Publish the 222-predecessor barrier from payload-free worker results."""

    rows, protocol, update = _validate_campaign_inputs(
        selection_attestations=selection_attestations,
        expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    payloads = _campaign_payloads(rows)
    seal = publish_phase(
        destination,
        artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        payloads=payloads,
        predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
        ),
        metadata=publication_identity.metadata(phase="outer-select", scope_id="global"),
    )
    return verify_outer_selection_campaign_barrier(
        seal,
        selection_attestations=selection_attestations,
        expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        update_campaign=update,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=update.seal.seal_sha256,
        expected_outer_selection_campaign_seal_sha256=seal.seal_sha256,
    )


def verify_outer_selection_campaign_barrier(
    seal: PhaseSeal,
    *,
    selection_attestations: tuple[OuterSelectionAttestation, ...],
    expected_outer_selection_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
) -> OuterSelectionCampaignCapability:
    """Independently reconstruct and authenticate the global barrier."""

    rows, protocol, update = _validate_campaign_inputs(
        selection_attestations=selection_attestations,
        expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    expected_payloads = _campaign_payloads(rows)
    verified = verify_phase_capability(
        seal,
        expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        expected_predecessor_seals=_campaign_predecessors(
            rows,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            update_global_seal_sha256=update.seal.seal_sha256,
        ),
        expected_seal_sha256=_sha256(
            expected_outer_selection_campaign_seal_sha256,
            label="controller outer-selection campaign seal",
        ),
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="outer-select",
        scope_id="global",
    )
    for path, expected in expected_payloads.items():
        if verified.read_payload_bytes(path) != expected:
            raise ValueError(f"outer-selection campaign payload differs from attestations: {path}")
    return OuterSelectionCampaignCapability(verified, publication_identity)


def outer_selection_campaign_capability_from_seal(
    seal: PhaseSeal,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
) -> OuterSelectionCampaignCapability:
    """Derive the rootless campaign under controller-authoritative globals."""

    protocol, update = _authenticate_authorities(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    rows, actual_protocol, actual_update = _decode_safe_campaign(
        seal,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_outer_selection_campaign_seal_sha256,
    )
    if actual_protocol != protocol.seal.seal_sha256 or actual_update != update.seal.seal_sha256:
        raise ValueError("outer-selection campaign binds the wrong global authorities")
    update_outer_view_rows = _snapshot_update_outer_view_rows(update)
    _cross_check_rows_against_update(
        rows,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        update=update,
        update_outer_view_rows=update_outer_view_rows,
    )
    return OuterSelectionCampaignCapability(seal, publication_identity)


def verify_outer_selection_campaign_capability(
    capability: OuterSelectionCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
) -> OuterSelectionCampaignCapability:
    """Freshly rederive the campaign before granting finalization access."""

    if type(capability) is not OuterSelectionCampaignCapability:
        raise TypeError("outer-selection campaign verification requires an exact capability")
    if capability.publication_identity != publication_identity:
        raise ValueError("outer-selection campaign has the wrong publication identity")
    verified = outer_selection_campaign_capability_from_seal(
        capability.seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    if capability.seal.seal_sha256 != verified.seal.seal_sha256:
        raise ValueError("outer-selection campaign differs from authoritative derivation")
    return verified


@dataclass(frozen=True, slots=True)
class SelectedOuterSequencesCapability:
    """One globally authorized selected-ID projection for finalization."""

    run: PolicyRunSpec
    selected_sequence_ids: tuple[str, ...]
    selected_sequence_ids_sha256: str
    selected_unique_component_count: int
    selected_max_component_occupancy: int
    outer_view_leaf_seal_sha256: str
    selection_leaf_seal_sha256: str
    outer_selection_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="selected outer-sequence run")
        if (
            type(self.selected_sequence_ids) is not tuple
            or len(self.selected_sequence_ids) != self.run.expected_outer_selection_count
        ):
            raise ValueError("selected outer-sequence capability must contain exactly ten IDs")
        digest = _id_stream_sha256(
            self.selected_sequence_ids,
            label="selected outer-sequence capability IDs",
        )
        if digest != _sha256(
            self.selected_sequence_ids_sha256,
            label="selected outer-sequence capability ID digest",
        ):
            raise ValueError("selected outer-sequence capability ID digest changed")
        _component_census(
            selected_count=len(self.selected_sequence_ids),
            unique_count=self.selected_unique_component_count,
            max_occupancy=self.selected_max_component_occupancy,
            label="selected outer-sequence capability",
        )
        _sha256(
            self.outer_view_leaf_seal_sha256,
            label="selected outer-sequence capability outer-view leaf",
        )
        _sha256(
            self.selection_leaf_seal_sha256,
            label="selected outer-sequence capability selection leaf",
        )
        _sha256(
            self.outer_selection_campaign_seal_sha256,
            label="selected outer-sequence capability campaign",
        )


def outer_selection_from_campaign(
    campaign: OuterSelectionCampaignCapability,
    *,
    run: PolicyRunSpec,
    selection_seal: PhaseSeal,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
) -> SelectedOuterSequencesCapability:
    """Authenticate and decode one selected-ID leaf for finalization."""

    frozen = _require_frozen_run(run, label="outer-selection campaign decode run")
    if type(selection_seal) is not PhaseSeal:
        raise TypeError("outer-selection campaign decoder requires an exact leaf PhaseSeal")
    authorized = verify_outer_selection_campaign_capability(
        campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    row = authorized.index_row(run=frozen)
    anchors = dict(authorized.anchor_digests)
    protocol_anchor = anchors[_protocol_predecessor()]
    update_anchor = anchors[_update_predecessor()]

    # Authenticate the complete global index and leaf shape before reading one
    # selected-ID payload.  The following re-derivation then proves semantics.
    verified_leaf = verify_phase_capability(
        selection_seal,
        expected_artifact=OUTER_SELECTION_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
        expected_predecessor_seals=_leaf_predecessors(
            run=frozen,
            protocol_seal_sha256=protocol_anchor,
            update_global_seal_sha256=update_anchor,
            outer_view_leaf_seal_sha256=row.outer_view_leaf_seal_sha256,
        ),
        expected_seal_sha256=row.leaf_seal_sha256,
    )
    publication_identity.verify_metadata(
        verified_leaf.metadata_json,
        phase="outer-select",
        scope_id=frozen.track_id,
    )
    if verified_leaf.payload_sha256 != row.payload_sha256:
        raise ValueError("outer-selection leaf payload digests differ from global index")

    protocol, update, material = _selection_material(
        run=frozen,
        update_campaign=update_campaign,
        outer_view_seal=outer_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    if (
        protocol.seal.seal_sha256 != protocol_anchor
        or update.seal.seal_sha256 != update_anchor
        or material.row.leaf_seal_sha256 != row.outer_view_leaf_seal_sha256
        or material.row.payload_sha256 != row.outer_view_payload_sha256
        or material.row.candidate_count != row.outer_candidate_count
        or material.row.candidate_ids_sha256 != row.outer_candidate_ids_sha256
    ):
        raise ValueError("outer-selection leaf re-derivation differs from campaign index")
    for path, expected in material.payloads:
        if verified_leaf.read_payload_bytes(path) != expected:
            raise ValueError(f"outer-selection leaf differs from re-derived semantics: {path}")
    selected_ids = _selected_ids_from_payload(
        verified_leaf.read_payload_bytes("selected-sequence-ids.jsonl")
    )
    selected_digest = _id_stream_sha256(
        selected_ids,
        label="authenticated outer-selection IDs",
    )
    if (
        selected_ids != material.selected_sequence_ids
        or selected_digest != row.selected_sequence_ids_sha256
        or row.selected_sequence_count != len(selected_ids)
        or row.selected_unique_component_count != material.selected_unique_component_count
        or row.selected_max_component_occupancy != material.selected_max_component_occupancy
    ):
        raise ValueError("outer-selection selected IDs differ from campaign authority")
    return SelectedOuterSequencesCapability(
        run=frozen,
        selected_sequence_ids=selected_ids,
        selected_sequence_ids_sha256=selected_digest,
        selected_unique_component_count=row.selected_unique_component_count,
        selected_max_component_occupancy=row.selected_max_component_occupancy,
        outer_view_leaf_seal_sha256=row.outer_view_leaf_seal_sha256,
        selection_leaf_seal_sha256=row.leaf_seal_sha256,
        outer_selection_campaign_seal_sha256=authorized.seal.seal_sha256,
    )


__all__ = [
    "EXPECTED_OUTER_SELECTION_LEAVES",
    "EXPECTED_OUTER_SELECTION_PREDECESSORS",
    "OUTER_SELECTION_ARTIFACT",
    "OUTER_SELECTION_ATTESTATION_ARTIFACT",
    "OUTER_SELECTION_CAMPAIGN_ARTIFACT",
    "OUTER_SELECTION_CAMPAIGN_BARRIER_ARTIFACT",
    "OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS",
    "OUTER_SELECTION_COMMITMENT_ARTIFACT",
    "OUTER_SELECTION_PAYLOAD_PATHS",
    "OUTER_SELECTION_SUMMARY_ARTIFACT",
    "OuterSelectionAttestation",
    "OuterSelectionCampaignCapability",
    "OuterSelectionIndexRow",
    "SelectedOuterSequencesCapability",
    "outer_selection_attestation_from_bytes",
    "outer_selection_attestation_from_document",
    "outer_selection_campaign_capability_from_seal",
    "outer_selection_from_campaign",
    "outer_selection_relative_path",
    "publish_outer_selection_campaign_barrier",
    "publish_outer_selection_commitment",
    "verify_outer_selection_campaign_barrier",
    "verify_outer_selection_campaign_capability",
    "verify_outer_selection_commitment",
]

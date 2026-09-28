"""Prospective per-student replay records; no campaign authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_tree_records import NativeTreeGeneration

CONFIG_SHA256 = "e9b6d6b932d50f9726d814e05f6639537bb7dd8eb1505e6bdfe5b4e64e211d5b"
MAXIMUM_RECORD_BYTES = 64 * 1024**2


def source_bytes():
    root = Path(__file__).resolve().parents[4]
    paths = (
        [
            "configs/diffusion/native_tr2d2_replay_v2.toml",
            "configs/diffusion/native_tr2d2_operators_v1.toml",
        ]
        + [
            "src/amp_challenge/generators/diffusion/" + name + ".py"
            for name in (
                "native_tr2d2_replay_v2_records",
                "native_tr2d2_replay_v2",
                "native_tr2d2_replay_v2_verify",
                "native_tr2d2",
                "native_tree_records",
                "native_tree_replay",
                "native_search_posterior",
                "native_weighted_training",
                "native_baseline_operators",
                "native_endpoint",
                "native_initialization",
                "model",
                "categorical",
                "subset_kernel",
                "replay",
            )
        ]
        + [
            "src/amp_challenge/generators/search/verified_charged_history.py",
            "src/amp_challenge/generators/search/peptide_ga_tunable_v2_records.py",
        ]
    )
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    if result[paths[0]] != CONFIG_SHA256:
        raise ValueError("TR2 replay-v2 declared configuration changed")
    if result[paths[1]] != "a27fa602333f3370fc631c7bdf45a3ce109aeb8ddd76483dbe5980cbcebdabd4":
        raise ValueError("TR2 replay-v1 configuration changed")
    return result


def seal_document(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    encoded = payload.encode()
    if len(encoded) > MAXIMUM_RECORD_BYTES:
        raise ValueError("TR2 replay-v2 serialized record cap exceeded")
    return payload, hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StudentReplayV2:
    triple: str
    origin_generation: int
    behavior_version: int
    behavior_sha256: str
    reference_sha256: str
    eligible_attempt_indices: tuple[int, ...]
    attempt_indices: tuple[int, ...]
    raw_log_weights: tuple[float, ...]
    relative_clamped_log_weights: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    effective_sample_size: float
    maximum_weight: float
    status: str


@dataclass(frozen=True, slots=True)
class ReplayGenerationV2:
    collection: NativeTreeGeneration
    behavior_versions: tuple[tuple[str, int], ...]
    buffers: tuple[StudentReplayV2, ...]
    source_sha256: str
    configuration_sha256: str = CONFIG_SHA256
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False

    @property
    def sha256(self):
        return _json_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ReplayAdvanceV2:
    record_json: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ReplayFailureV2:
    phase: str
    error: str
    completed_prefix: dict
    replacement_commit_permitted: bool = False
    campaign_eligible: bool = False


def ordered_layers(attempts, indices, context):
    """The v2-only replay order; the v1 query order and tree stay untouched."""
    remaining, ordered = list(indices), []
    while remaining:
        front = []
        for index in remaining:
            values = attempts[index].posterior.objectives
            if not any(
                all(
                    a >= b
                    for a, b in zip(attempts[other].posterior.objectives, values, strict=True)
                )
                and any(
                    a > b for a, b in zip(attempts[other].posterior.objectives, values, strict=True)
                )
                for other in remaining
            ):
                front.append(index)
        front.sort(
            key=lambda index: (
                -context.scalarize(attempts[index].posterior.objectives),
                sequence_id(attempts[index].path.endpoint),
            )
        )
        ordered.extend(front)
        used = set(front)
        remaining = [index for index in remaining if index not in used]
    return tuple(ordered)


def prepare_replay_generation(collection, context, *, expected_versions, source_sha256):
    if type(collection) is not NativeTreeGeneration or not 1 <= collection.round_index <= 28:
        raise ValueError("TR2 replay-v2 requires an actual bounded source generation")
    triples = tuple(root.triple for root in collection.roots)
    if (
        len(set(triples)) != len(triples)
        or set(expected_versions) != set(triples)
        or any(
            type(value) is not int or not 0 <= value < collection.round_index
            for value in expected_versions.values()
        )
        or collection.status not in ("complete", "bounded_search_underfill")
        or collection.evaluator_binding.objective_context_sha256 != context.context_sha256
    ):
        raise ValueError("TR2 replay-v2 source generation/context/version inventory differs")
    by_expansion = {row.expansion_index: row.triple for row in collection.expansions}
    buffers = []
    for root in collection.roots:
        eligible = tuple(
            index
            for index, row in enumerate(collection.attempts)
            if by_expansion[row.expansion_index] == root.triple
            and row.rejection_reason is None
            and row.first_attempt_index == index
        )
        indices = ordered_layers(collection.attempts, eligible, context)[:64]
        if len({collection.attempts[index].path.endpoint for index in indices}) != len(indices):
            raise ValueError("TR2 replay-v2 cannot duplicate target rows")
        raw = tuple(
            context.scalarize(collection.attempts[index].posterior.objectives) / 0.1
            + collection.attempts[index].path.reference_log_probability
            - collection.attempts[index].path.behavior_log_probability
            for index in indices
        )
        if not all(map(math.isfinite, raw)):
            raise ValueError("TR2 replay-v2 has nonfinite actual path weights")
        relative = tuple(float(max(-10.0, min(0.0, value - max(raw)))) for value in raw)
        weights = np.exp(np.asarray(relative, dtype=np.float64))
        if len(weights):
            weights /= weights.sum()
        ess = float(1 / (weights @ weights)) if len(weights) else 0.0
        maximum = float(max(weights)) if len(weights) else 0.0
        status = (
            "insufficient_unique_support_no_update"
            if len(indices) < 40
            else "concentrated_actual_weights_no_update"
            if ess / len(indices) < 0.2 or maximum > 0.05
            else "admitted_actual_weights"
        )
        buffers.append(
            StudentReplayV2(
                root.triple,
                collection.round_index,
                expected_versions[root.triple],
                root.behavior_sha256,
                root.reference_sha256,
                eligible,
                indices,
                raw,
                relative,
                tuple(map(float, weights)),
                ess,
                maximum,
                status,
            )
        )
    return ReplayGenerationV2(
        collection,
        tuple((key, expected_versions[key]) for key in triples),
        tuple(buffers),
        source_sha256,
    )

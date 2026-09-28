"""Prospective positive-child replay, requalified against each current posterior.

Replay is selected-endpoint supervision, not independent evidence or trajectory
importance sampling. No private oracle is available here. Legacy teachers are
unchanged; this intervention is explicitly identified in campaign provenance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_evolution_math import contrast_from_moments
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    EndpointOrigin,
    EndpointTarget,
    EndpointTeacher,
)

REPLAY_SPEC = {
    "version": "requalified_positive_replay_v1",
    "capacity": 512,
    "children": 56,
    "parent_anchors": 8,
    "qualification": "current_paired_mean_minus_sd_positive_and_absolute_lcb_at_least_half",
    "selection": "global_top_children_without_parent_group_veto",
    "retention": "current_qualifying_children_only_top_advantage_unique_sequences",
    "minimum_distinct_children": 40,
    "logweight_span": math.log(2),
    "oracle_access": "none",
}
REPLAY_SPEC_SHA256 = _json_hash(REPLAY_SPEC)


@dataclass(frozen=True)
class ReplayCandidate:
    sequence: str
    parent: str
    source_wave_sha256: str
    attempt_sha256: str
    original_generation: int
    original_triple: str
    original_behavior_model_sha256: str
    original_behavior_version: int
    first_seen_ordinal: int
    feasible: bool


class RequalifiedReplayTeacher:
    """Bounded, unique positive replay; stale confidence is never reused."""

    def __init__(self):
        self.bank = ()
        self.generation = 0

    def build(
        self, wave, posterior, cache, *, generation, max_generations, prospective_protocol_sha256
    ):
        if (
            wave.variant != "full"
            or wave.round_index != generation
            or generation != self.generation + 1
            or max_generations != 64
            or wave.max_rounds != max_generations
            or wave.prospective_protocol_sha256 != prospective_protocol_sha256
        ):
            raise ValueError("replay teacher requires contiguous prospective full-method waves")
        prior = self.bank
        inventory = {sequence_id(row.sequence): row for row in prior}
        wave_sha = wave.sha256
        for attempt in wave.attempts:
            if attempt.rejection is not None or attempt.first_seen_ordinal != attempt.ordinal:
                continue
            candidate = ReplayCandidate(
                attempt.trace.endpoint,
                attempt.lineage_parent,
                wave_sha,
                attempt.sha256,
                generation,
                attempt.triple,
                attempt.trace.model_sha256,
                attempt.behavior_version,
                attempt.ordinal,
                attempt.contrast.feasible,
            )
            inventory.setdefault(sequence_id(candidate.sequence), candidate)
        candidates = tuple(inventory.values())
        records, qualified = [], []
        for start in range(0, len(candidates), 512):
            block = candidates[start : start + 512]
            means, covariances = posterior.pairs(
                cache.matrix(tuple(row.sequence for row in block)),
                cache.matrix(tuple(row.parent for row in block)),
            )
            for candidate, mean, covariance in zip(block, means, covariances, strict=True):
                contrast = contrast_from_moments(mean, covariance, feasible=candidate.feasible)
                record = {
                    **asdict(candidate),
                    "sequence_id": sequence_id(candidate.sequence),
                    "lineage_parent_id": sequence_id(candidate.parent),
                    "rebuilt_generation": generation,
                    "posterior_sha256": posterior.sha256,
                    "rebuilt_contrast": asdict(contrast),
                    "metric": contrast.advantage,
                    "accepted": contrast.accepted,
                    "weighting_rule": "paired_mean_minus_sd",
                }
                records.append(record)
                if contrast.accepted:
                    qualified.append((candidate, contrast.advantage, record))
        qualified.sort(key=lambda row: (-row[1], sequence_id(row[0].sequence)))
        selected = qualified[: REPLAY_SPEC["children"]]
        children = {row[0].sequence for row in selected}
        parents = tuple(
            dict.fromkeys(row[0].parent for row in selected if row[0].parent not in children)
        )[:8]
        maximum = max((row[1] for row in selected), default=0.0)
        targets = []
        for candidate, metric, record in selected:
            raw = (metric - maximum) / 0.1
            logweight = float(np.clip(raw, -math.log(2), 0))
            record.update(logweight_unclipped=raw, logweight=logweight)
            targets.append(
                EndpointTarget(
                    candidate.sequence,
                    "positive_child",
                    logweight,
                    EndpointOrigin(
                        _json_hash(record),
                        "native_endpoint",
                        candidate.original_generation,
                        candidate.original_triple,
                        candidate.original_behavior_model_sha256,
                        candidate.original_behavior_version,
                    ),
                )
            )
        for parent in parents:
            origin = {
                "parent": parent,
                "generation": generation,
                "posterior": posterior.sha256,
                "child_origins": [
                    row[0].attempt_sha256 for row in selected if row[0].parent == parent
                ],
            }
            targets.append(
                EndpointTarget(
                    parent,
                    "zero_advantage_parent",
                    float(np.clip(-maximum / 0.1, -math.log(2), 0)),
                    EndpointOrigin(_json_hash(origin), "lineage_parent", None, None, None, None),
                )
            )
        retained = tuple(row[0] for row in qualified[: REPLAY_SPEC["capacity"]])
        evidence = {
            "wave_sha256": wave_sha,
            "variant": wave.variant,
            "posterior_sha256": posterior.sha256,
            "context_sha256": posterior.context_sha256,
            "generation": generation,
            "candidates": records,
            "selected_sequence_ids": [target.sequence_id for target in targets],
            "protected_children": len(selected),
            "zero_advantage_parents": len(parents),
            "selection_semantics": REPLAY_SPEC["selection"],
            "replay_spec": REPLAY_SPEC,
            "replay_spec_sha256": REPLAY_SPEC_SHA256,
            "bank_before_sha256": _json_hash([asdict(row) for row in prior]),
            "bank_after_sha256": _json_hash([asdict(row) for row in retained]),
            "bank_before_count": len(prior),
            "bank_after_count": len(retained),
            "retained_sequence_ids": [sequence_id(row.sequence) for row in retained],
            "selected_replayed_children": sum(
                row[0].original_generation < generation for row in selected
            ),
            "max_generations": max_generations,
            "prospective_protocol_sha256": prospective_protocol_sha256,
        }
        teacher = EndpointTeacher(
            tuple(targets),
            "positive_child",
            posterior.context_sha256,
            _json_hash(evidence),
            generation,
            max_generations=max_generations,
            prospective_protocol_sha256=prospective_protocol_sha256,
        )
        self.bank, self.generation = retained, generation
        return teacher, evidence

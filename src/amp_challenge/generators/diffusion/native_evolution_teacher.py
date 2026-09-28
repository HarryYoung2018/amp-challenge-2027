"""Rebuild selected endpoint targets using only the charged-data learner."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_evolution_math import contrast_from_moments
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    EndpointOrigin,
    EndpointTarget,
    EndpointTeacher,
)


def build_evolution_teacher(
    wave,
    posterior,
    cache,
    *,
    generation: int,
    max_generations: int = 28,
    prospective_protocol_sha256: str | None = None,
):
    """No hidden oracle scores; old contrast evidence is never overwritten.

    Parent groups rank by their best admitted child, then canonical parent ID.
    Up to eight groups contribute at most56 highest-valued distinct children.
    Their distinct no-op parents fit the remaining eight slots. The shared
    trainer independently enforces forty protected children before any mixing.
    """
    if (
        getattr(wave, "max_rounds", 28) != max_generations
        or getattr(wave, "prospective_protocol_sha256", None) != prospective_protocol_sha256
    ):
        raise ValueError("endpoint prospective budget/protocol differs from wave")
    records, candidates, groups = [], [], {}
    usable = tuple(
        attempt
        for attempt in wave.attempts
        if attempt.rejection is None and attempt.first_seen_ordinal == attempt.ordinal
    )
    if usable:
        child_raw = cache.matrix(tuple(attempt.trace.endpoint for attempt in usable))
        if wave.variant == "no_counterfactual":
            means = posterior.means(child_raw)
        else:
            parent_raw = cache.matrix(tuple(attempt.lineage_parent for attempt in usable))
            means, covariances = posterior.pairs(child_raw, parent_raw)
    for index, attempt in enumerate(usable):
        # Feasibility here is the fixed public/context predicate already sealed
        # with generation. It is not imputed assay/organizer safety evidence.
        feasible = attempt.contrast.feasible
        if wave.variant == "no_counterfactual":
            metric = float(means[index].mean())
            accepted = feasible and metric > 0.5
            paired = None
        else:
            paired = contrast_from_moments(means[index], covariances[index], feasible=feasible)
            metric, accepted = paired.advantage, paired.accepted
        record = {
            "attempt_sha256": attempt.sha256,
            "first_seen_ordinal": attempt.first_seen_ordinal,
            "sequence_id": attempt.trace.endpoint_sha256,
            "lineage_parent_id": sequence_id(attempt.lineage_parent),
            "original_triple": attempt.triple,
            "original_behavior_model_sha256": attempt.trace.model_sha256,
            "original_behavior_version": attempt.behavior_version,
            "original_generation": wave.round_index,
            "rebuilt_generation": generation,
            "posterior_sha256": posterior.sha256,
            "presubmission_contrast": asdict(attempt.contrast),
            "rebuilt_contrast": None if paired is None else asdict(paired),
            "metric": metric,
            "accepted": bool(accepted),
            "weighting_rule": "absolute_mean_only" if paired is None else "paired_mean_minus_sd",
        }
        records.append(record)
        if accepted:
            candidates.append((attempt, metric, record))
            groups[attempt.lineage_parent] = max(
                groups.get(attempt.lineage_parent, -math.inf), metric
            )
    parents = tuple(sorted(groups, key=lambda parent: (-groups[parent], sequence_id(parent)))[:8])
    candidates = sorted(
        (row for row in candidates if row[0].lineage_parent in parents),
        key=lambda row: (-row[1], row[0].trace.endpoint_sha256),
    )[:56]
    child_sequences = {row[0].trace.endpoint for row in candidates}
    # A candidate which is itself another selected child's parent retains its
    # positive-child role once; duplicates cannot manufacture replay entropy.
    used_parents = tuple(
        parent
        for parent in parents
        if parent not in child_sequences
        and any(row[0].lineage_parent == parent for row in candidates)
    )
    maximum = max((row[1] for row in candidates), default=0.0)
    targets = []
    for attempt, metric, record in candidates:
        raw = (metric - maximum) / 0.1
        logweight = float(np.clip(raw, -math.log(2), 0))
        record.update(logweight_unclipped=raw, logweight=logweight)
        origin = EndpointOrigin(
            _json_hash(record),
            "native_endpoint",
            wave.round_index,
            attempt.triple,
            attempt.trace.model_sha256,
            attempt.behavior_version,
        )
        targets.append(EndpointTarget(attempt.trace.endpoint, "positive_child", logweight, origin))
    for parent in used_parents:
        evidence = {
            "parent": parent,
            "role": "zero_advantage_parent",
            "first_seen_lineages": [
                row[0].sha256 for row in candidates if row[0].lineage_parent == parent
            ],
            "rebuilt_generation": generation,
            "posterior_sha256": posterior.sha256,
        }
        targets.append(
            EndpointTarget(
                parent,
                "zero_advantage_parent",
                float(np.clip(-maximum / 0.1, -math.log(2), 0)),
                EndpointOrigin(_json_hash(evidence), "lineage_parent", None, None, None, None),
            )
        )
    evidence = {
        "wave_sha256": wave.sha256,
        "variant": wave.variant,
        "posterior_sha256": posterior.sha256,
        "context_sha256": posterior.context_sha256,
        "generation": generation,
        "candidates": records,
        "selected_sequence_ids": [target.sequence_id for target in targets],
        "protected_children": len(candidates),
        "zero_advantage_parents": len(used_parents),
        "selection_semantics": "first_seen_lineage_max_eight_parent_groups_then_top56_children_not_IS",
    }
    if max_generations != 28 or prospective_protocol_sha256 is not None:
        evidence["max_generations"] = max_generations
        evidence["prospective_protocol_sha256"] = prospective_protocol_sha256
    teacher = EndpointTeacher(
        tuple(targets),
        "positive_child",
        posterior.context_sha256,
        _json_hash(evidence),
        generation,
        max_generations=max_generations,
        prospective_protocol_sha256=prospective_protocol_sha256,
    )
    return teacher, evidence

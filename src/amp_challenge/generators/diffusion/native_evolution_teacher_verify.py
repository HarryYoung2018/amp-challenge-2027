"""Independent target selection/weight/origin reconstruction before NN replay.

Uses the accepted Gaussian posterior primitives, not the teacher producer or
its paired-credit helper. Upstream wave/history/features require their own
authentication; this is not a second scientific/provider authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_evolution_verify import _paired_mean
from amp_challenge.generators.diffusion.native_initialization import TRIPLES


def verify_evolution_teacher(
    teacher, evidence, *, wave, history, posterior, cache, expected_behavior_versions
):
    wave.check()
    history.__post_init__()
    posterior.check()
    teacher.__post_init__()
    generation = history.round_index - 1
    if (
        wave.status != "complete"
        or not history.complete
        or wave.round_index != generation
        or posterior.history_sha256 != history.sha256
        or teacher.rebuild_generation != generation
        or teacher.objective_context_sha256 != history.objective_context_sha256
        or posterior.context_sha256 != history.objective_context_sha256
        or teacher.protected_role != "positive_child"
        or set(expected_behavior_versions) != set(TRIPLES)
        or any(
            type(value) is not int or not 0 <= value < wave.round_index
            for value in expected_behavior_versions.values()
        )
    ):
        raise ValueError("teacher verified generation/version/context inputs differ")
    fixed = {
        "wave_sha256": wave.sha256,
        "variant": wave.variant,
        "posterior_sha256": posterior.sha256,
        "context_sha256": posterior.context_sha256,
        "generation": generation,
        "selection_semantics": "first_seen_lineage_max_eight_parent_groups_then_top56_children_not_IS",
    }
    if any(evidence.get(key) != value for key, value in fixed.items()):
        raise ValueError("teacher source/current-posterior binding differs")
    usable = [
        attempt
        for attempt in wave.attempts
        if attempt.rejection is None and attempt.first_seen_ordinal == attempt.ordinal
    ]
    records = evidence["candidates"]
    if len(records) != len(usable):
        raise ValueError("teacher first-seen candidate inventory differs")
    accepted, metrics = [], {}
    for attempt, record in zip(usable, records, strict=True):
        if attempt.behavior_version != expected_behavior_versions[attempt.triple]:
            raise ValueError("teacher source behavior version differs")
        identity = {
            "attempt_sha256": attempt.sha256,
            "first_seen_ordinal": attempt.ordinal,
            "sequence_id": attempt.trace.endpoint_sha256,
            "lineage_parent_id": sequence_id(attempt.lineage_parent),
            "original_triple": attempt.triple,
            "original_behavior_model_sha256": attempt.trace.model_sha256,
            "original_behavior_version": expected_behavior_versions[attempt.triple],
            "original_generation": wave.round_index,
            "rebuilt_generation": generation,
            "posterior_sha256": posterior.sha256,
            "presubmission_contrast": asdict(attempt.contrast),
            "weighting_rule": "absolute_mean_only"
            if wave.variant == "no_counterfactual"
            else "paired_mean_minus_sd",
        }
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("teacher immutable origin identity differs")
        feasible = attempt.contrast.feasible
        raw = cache.matrix((attempt.trace.endpoint, attempt.lineage_parent))
        if wave.variant == "no_counterfactual":
            # The no-CF teacher must not require any paired covariance.
            feature = posterior.transform.apply(raw[:1])
            coefficients = posterior.backend.coefficient_mean + np.einsum(
                "dor,r->do", posterior.backend.coefficient_factor, posterior.backend.latent_mean
            )
            metric = float((feature @ coefficients)[0].mean())
            admit = feasible and metric > 0.5
            if record["rebuilt_contrast"] is not None:
                raise ValueError("no-CF teacher invents paired credit")
        else:
            belief = posterior.joint(raw)
            vector = np.array([0.5, 0.5, -0.5, -0.5])
            difference = _paired_mean(belief.mean)
            variance = max(0.0, float(vector @ belief.covariance.reshape(4, 4) @ vector))
            absolute = float(belief.mean[0].mean())
            risk = absolute - math.sqrt(
                max(
                    0.0,
                    float(
                        np.array([0.5, 0.5]) @ belief.covariance[0, :, 0, :] @ np.array([0.5, 0.5])
                    ),
                )
            )
            metric = difference - math.sqrt(variance)
            admit = bool(feasible and metric > 0 and risk >= 0.5)
            contrast = record["rebuilt_contrast"]
            if set(contrast) != {
                "mean",
                "variance",
                "advantage",
                "absolute_mean",
                "absolute_risk",
                "feasible",
                "accepted",
            }:
                raise ValueError("teacher paired contrast fields differ")
            np.testing.assert_allclose(
                [
                    contrast[key]
                    for key in ("mean", "variance", "advantage", "absolute_mean", "absolute_risk")
                ],
                (difference, variance, metric, absolute, risk),
                atol=1e-10,
                rtol=1e-9,
            )
            if contrast["feasible"] is not feasible or contrast["accepted"] is not admit:
                raise ValueError("teacher paired/absolute eligibility differs")
        if record["accepted"] is not bool(admit):
            raise ValueError("teacher eligibility differs")
        np.testing.assert_allclose(record["metric"], metric, atol=1e-10, rtol=1e-9)
        metrics[attempt.ordinal] = metric
        if admit:
            accepted.append(attempt)
    best_by_parent = {}
    for attempt in accepted:
        parent = attempt.lineage_parent
        best_by_parent[parent] = max(
            best_by_parent.get(parent, -math.inf), metrics[attempt.ordinal]
        )
    parents = sorted(
        best_by_parent, key=lambda parent: (-best_by_parent[parent], sequence_id(parent))
    )[:8]
    children = sorted(
        (attempt for attempt in accepted if attempt.lineage_parent in parents),
        key=lambda attempt: (-metrics[attempt.ordinal], attempt.trace.endpoint_sha256),
    )[:56]
    children_sequences = {attempt.trace.endpoint for attempt in children}
    parents = [
        parent
        for parent in parents
        if parent not in children_sequences
        and any(attempt.lineage_parent == parent for attempt in children)
    ]
    expected_sequences = tuple(attempt.trace.endpoint for attempt in children) + tuple(parents)
    if (
        tuple(target.sequence for target in teacher.targets) != expected_sequences
        or evidence["selected_sequence_ids"] != list(map(sequence_id, expected_sequences))
        or evidence["protected_children"] != len(children)
        or evidence["zero_advantage_parents"] != len(parents)
    ):
        raise ValueError("teacher selected parent groups/order/caps differ")
    selected_ordinals = {attempt.ordinal for attempt in children}
    max_metric = max((metrics[attempt.ordinal] for attempt in children), default=0.0)
    indexed_records = {
        attempt.ordinal: record for attempt, record in zip(usable, records, strict=True)
    }
    for attempt, record in zip(usable, records, strict=True):
        expected_extra = (
            {"logweight", "logweight_unclipped"} if attempt.ordinal in selected_ordinals else set()
        )
        if (
            set(record)
            != set(identity) | {"rebuilt_contrast", "metric", "accepted"} | expected_extra
        ):
            raise ValueError("teacher candidate weight field inventory differs")
    for index, attempt in enumerate(children):
        target, record = teacher.targets[index], indexed_records[attempt.ordinal]
        raw_weight = (metrics[attempt.ordinal] - max_metric) / 0.1
        weight = max(-math.log(2), min(0.0, raw_weight))
        np.testing.assert_allclose(
            (target.log_weight, record["logweight"], record["logweight_unclipped"]),
            (weight, weight, raw_weight),
            atol=1e-10,
            rtol=1e-9,
        )
        if target.role != "positive_child" or asdict(target.origin) != {
            "source_record_sha256": _json_hash(record),
            "origin_kind": "native_endpoint",
            "generation": wave.round_index,
            "checkpoint_triple": attempt.triple,
            "behavior_model_sha256": attempt.trace.model_sha256,
            "behavior_version": expected_behavior_versions[attempt.triple],
        }:
            raise ValueError("teacher selected child origin mapping differs")
    for index, parent in enumerate(parents, len(children)):
        target = teacher.targets[index]
        origin_record = {
            "parent": parent,
            "role": "zero_advantage_parent",
            "first_seen_lineages": [
                attempt.sha256 for attempt in children if attempt.lineage_parent == parent
            ],
            "rebuilt_generation": generation,
            "posterior_sha256": posterior.sha256,
        }
        np.testing.assert_allclose(
            target.log_weight, max(-math.log(2), min(0.0, -max_metric / 0.1)), atol=1e-10, rtol=1e-9
        )
        if target.role != "zero_advantage_parent" or asdict(target.origin) != {
            "source_record_sha256": _json_hash(origin_record),
            "origin_kind": "lineage_parent",
            "generation": None,
            "checkpoint_triple": None,
            "behavior_model_sha256": None,
            "behavior_version": None,
        }:
            raise ValueError("teacher zero-parent origin mapping differs")
    if teacher.target_source_sha256 != _json_hash(evidence):
        raise ValueError("teacher target evidence receipt differs")
    posterior.check()
    return {
        "reconstructed": True,
        "protected_children": len(children),
        "scope": "independent_target_selector_weights_origins_shared_Gaussian_primitives_not_IS_or_oracle_truth",
    }

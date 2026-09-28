"""Separate frontier/check reconstruction using the accepted Gaussian evaluator.

This never calls the evolutionary selector or its pool reducer. It is not a
second independently authored Gaussian conditioning implementation. Its input
wave, charged learner and controller-filter receipts need their own audits.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from amp_challenge.acquisition.soft_kg import (
    EvaluationBatch,
    GaussianSoftKG,
    PreferenceMeasure,
    SoftKGProblem,
)
from amp_challenge.generators.diffusion.native_endpoint import _json_hash


def _same_result(actual, expected):
    expected = asdict(expected)
    if set(actual) != set(expected):
        raise ValueError("KG numerical result fields differ")
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "b":
                if np.asarray(actual[key]).dtype.kind != "b":
                    raise ValueError("KG exact eligibility type differs")
                np.testing.assert_array_equal(actual[key], value)
            else:
                np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-9)
        elif _json_hash(actual[key]) != _json_hash(value):
            raise ValueError("KG result identity/order differs")


def verify_evolution_selection(
    selection, *, wave, posterior, cache, allowed_sequence_ids, expected_filter_sha256
):
    wave.check()
    posterior.check()
    if (
        selection.wave_sha256 != wave.sha256
        or selection.posterior_sha256 != posterior.sha256
        or wave.posterior_sha256 != posterior.sha256
        or selection.external_filter_sha256 != expected_filter_sha256
    ):
        raise ValueError("KG selection source/wave/filter binding differs")
    offered = {wave.attempts[index].trace.endpoint_sha256 for index in wave.shortlist_ordinals}
    if type(allowed_sequence_ids) is not frozenset or not allowed_sequence_ids <= offered:
        raise ValueError("KG expected filter invents an unoffered candidate")
    indices = [
        index
        for index in wave.shortlist_ordinals
        if wave.attempts[index].trace.endpoint_sha256 in allowed_sequence_ids
    ]
    by_mean = sorted(
        indices,
        key=lambda index: (
            -sum(wave.attempts[index].posterior_mean) / 2,
            wave.attempts[index].trace.endpoint_sha256,
        ),
    )
    by_max_draw = sorted(
        indices,
        key=lambda index: (
            -wave.attempts[index].max_thompson_value,
            wave.attempts[index].trace.endpoint_sha256,
        ),
    )
    reduced = []
    for pair in zip(by_mean, by_max_draw, strict=True):
        for index in pair:
            if index not in reduced and len(reduced) < 20:
                reduced.append(index)
    pool = tuple(reduced)
    if selection.pool_ordinals != pool:
        raise ValueError("KG mixed pool reconstruction differs")
    if len(pool) < 14 or wave.status != "complete":
        if (
            selection.status != "stopped_underfilled_kg_pool"
            or selection.selected_ordinals
            or selection.primary
            or selection.post_selection_checks
            or selection.numerical_sha256 != _json_hash([])
        ):
            raise ValueError("KG underfill stop differs")
        return {"reconstructed": True, "scope": "underfilled_frozen_pool"}
    belief = posterior.joint(
        cache.matrix(tuple(wave.attempts[index].trace.endpoint for index in pool))
    )
    if selection.numerical_sha256 != _json_hash(
        [belief.mean.tolist(), belief.covariance.tolist(), posterior.noise.tolist()]
    ):
        raise ValueError("KG frozen joint numerical hash differs")
    positions = tuple(range(len(pool)))
    problem = SoftKGProblem(
        positions, (0, 1), PreferenceMeasure(np.array([[0.5, 0.5]])), np.ones(len(pool))
    )
    batch = EvaluationBatch(positions, np.ones(len(pool)))

    def engine(fantasies, suffix):
        seed = int(_json_hash([wave.semantic_history_sha256, suffix])[:16], 16)
        return GaussianSoftKG(
            problem,
            temperature=0.25,
            observed_outputs=(0, 1),
            n_fantasies=fantasies,
            standard_error_multiplier=1,
            seed=seed,
            relative_eigenvalue_cutoff=1e-10,
            candidate_chunk_size=64,
            fantasy_chunk_size=512,
        )

    evaluator = engine(512, "evolution-kg-primary")
    if wave.variant == "singleton_kg":
        scored = evaluator.score(belief, batch)
        _same_result(selection.primary["singleton"], scored)
        ranked = sorted(
            (index for index in positions if scored.score[index] > 0),
            key=lambda index: (-scored.score[index], index),
        )
        chosen = tuple(ranked[:14])
        expected_eligible = [
            [index for index in positions if index not in chosen[:step]]
            for step in range(len(chosen))
        ]
        if selection.primary["sequential_eligible"] != expected_eligible:
            raise ValueError("singleton sequential eligible sets differ")
        groups = [(index,) for index in chosen] if len(chosen) == 14 else []
        statistics = [
            (float(scored.estimate[index]), float(scored.standard_error[index])) for index in chosen
        ]
    else:
        primary = selection.primary
        if (primary["batch_size"], primary["beam_width"], primary["max_groups_scored"]) != (
            14,
            4,
            768,
        ) or len(primary["depth_trace"]) != 14:
            raise ValueError("joint KG declared beam budgets differ")
        frontier, count, pruned = ((),), 0, False
        for depth, recorded in enumerate(primary["depth_trace"], 1):
            groups = tuple(
                sorted(
                    {
                        tuple(sorted((*group, point)))
                        for group in frontier
                        for point in positions
                        if point not in group
                    }
                )
            )
            available = 768 - count - (14 - depth)
            if len(groups) > available:
                raise ValueError("KG replay frontier exceeds predeclared cap")
            scored = evaluator.score_joint_groups(belief, batch, groups, max_groups=available)
            _same_result(recorded["scored"], scored)
            ranked = sorted(
                range(len(groups)), key=lambda index: (-scored.score[index], groups[index])
            )
            frontier = tuple(groups[index] for index in ranked[:4])
            count += len(groups)
            trimmed = max(0, len(groups) - len(frontier)) if depth < 14 else 0
            pruned |= trimmed > 0
            if (
                recorded["depth"],
                recorded["generated_group_count"],
                recorded["completion_feasible_group_count"],
                recorded["beam_pruned_group_count"],
                recorded["remaining_group_budget"],
            ) != (depth, len(groups), len(groups), trimmed, 768 - count) or tuple(
                map(tuple, recorded["retained_evaluation_batches"])
            ) != frontier:
                raise ValueError("KG beam frontier/order/census differs")
        if primary["total_groups_scored"] != count:
            raise ValueError("KG total scoring budget differs")
        if primary["approximation_status"] != (
            "beam_pruned" if pruned else "exact_frontier_covered"
        ):
            raise ValueError("KG approximation declaration differs")
        positive = [index for index in ranked if scored.score[index] > 0]
        chosen = () if not positive else groups[positive[0]]
        groups = [chosen] if chosen else []
        statistics = (
            []
            if not positive
            else [(float(scored.estimate[positive[0]]), float(scored.standard_error[positive[0]]))]
        )
    expected_checks = []
    status, selected = "stopped_no_positive_kg_action", ()
    if groups:
        checked = engine(1024, "evolution-kg-independent-check").score_joint_groups(
            belief, batch, groups, max_groups=14
        )
        for index, (mean, se) in enumerate(statistics):
            other, other_se = float(checked.estimate[index]), float(checked.standard_error[index])
            tolerance = 3 * math.sqrt(se**2 + other_se**2) + 1e-6
            passed = (
                all(math.isfinite(value) for value in (mean, se, other, other_se))
                and mean - se > 0
                and se <= 0.02
                and abs(mean - other) <= tolerance
            )
            expected_checks.append(
                {
                    "group": list(groups[index]),
                    "primary_mean": mean,
                    "primary_se": se,
                    "check_mean": other,
                    "check_se": other_se,
                    "agreement_tolerance": tolerance,
                    "passed": passed,
                }
            )
        status = (
            "selected"
            if all(row["passed"] for row in expected_checks)
            else "stopped_kg_mc_instability"
        )
        if status == "selected":
            selected = tuple(pool[index] for index in chosen)
    if selection.selected_ordinals != selected or selection.status != status:
        raise ValueError("KG final action/ranking/stop differs")
    if len(selection.post_selection_checks) != len(expected_checks):
        raise ValueError("KG fresh-fantasy inventory differs")
    for actual, expected in zip(selection.post_selection_checks, expected_checks, strict=True):
        if (
            set(actual) != set(expected)
            or actual["group"] != expected["group"]
            or actual["passed"] != expected["passed"]
        ):
            raise ValueError("KG fresh check identities/status differ")
        for key in expected.keys() - {"group", "passed"}:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-9)
    posterior.check()
    return {
        "reconstructed": True,
        "scope": "separate_frontier_and_check_reconstruction_shared_gaussian_evaluator_not_oracle_truth",
    }

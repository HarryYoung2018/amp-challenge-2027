"""Bounded real joint-Gaussian acquisition, not a proxy oracle caller."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from amp_challenge.acquisition.soft_kg import (
    EvaluationBatch,
    GaussianSoftKG,
    PreferenceMeasure,
    SoftKGProblem,
)
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string


def numerical_document(value):
    if is_dataclass(value):
        return numerical_document(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: numerical_document(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [numerical_document(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class EvolutionSelection:
    wave_sha256: str
    posterior_sha256: str
    external_filter_sha256: str
    pool_ordinals: tuple[int, ...]
    selected_ordinals: tuple[int, ...]
    status: str
    primary: dict
    post_selection_checks: tuple[dict, ...]
    numerical_sha256: str

    @property
    def sha256(self):
        return _json_hash(asdict(self))


def reduced_pool(wave, allowed_ids):
    candidates = [
        index
        for index in wave.shortlist_ordinals
        if sequence_id(wave.attempts[index].trace.endpoint) in allowed_ids
    ]
    by_mean = sorted(
        candidates,
        key=lambda index: (
            -sum(wave.attempts[index].posterior_mean) / 2,
            sequence_id(wave.attempts[index].trace.endpoint),
        ),
    )
    by_draw = sorted(
        candidates,
        key=lambda index: (
            -wave.attempts[index].max_thompson_value,
            sequence_id(wave.attempts[index].trace.endpoint),
        ),
    )
    result = []
    for pair in zip(by_mean, by_draw, strict=True):
        for index in pair:
            if index not in result:
                result.append(index)
                if len(result) == 20:
                    return tuple(result)
    return tuple(result)


def select_evolution_queries(
    wave,
    posterior,
    cache,
    *,
    allowed_sequence_ids: frozenset[str],
    external_filter_sha256: str,
    deadline,
    fixed_budget: bool = False,
    precision_recheck: bool = False,
):
    """Filter is controller-owned; it sees only already proposed candidates.

    No veto/reserve identity is fed back into search credit or training. A fresh
    diagnostic fantasy set evaluates the chosen actions but never reranks them.
    A protocol-bound fixed budget removes only the positive-gain stopping rule;
    finite values, precision, agreement, eligibility and deadlines remain gates.
    """
    wave.check()
    if type(precision_recheck) is not bool or (precision_recheck and not fixed_budget):
        raise ValueError("precision recheck is an explicit fixed-budget amendment")
    if type(fixed_budget) is not bool or (
        fixed_budget
        and (wave.variant != "full" or not hash_string(wave.prospective_protocol_sha256))
    ):
        raise ValueError(
            "fixed-budget native allocation requires a prospective full-method protocol"
        )
    if not hash_string(external_filter_sha256) or type(allowed_sequence_ids) is not frozenset:
        raise ValueError("explicit controller filter binding required")
    offered = {
        sequence_id(wave.attempts[index].trace.endpoint) for index in wave.shortlist_ordinals
    }
    if not allowed_sequence_ids <= offered or posterior.sha256 != wave.posterior_sha256:
        raise ValueError("filter/posterior not bound to offered wave")
    pool = reduced_pool(wave, allowed_sequence_ids)
    primary, checks, selected = {}, [], ()
    status = "stopped_underfilled_kg_pool"
    numerical = _json_hash([])
    if len(pool) >= 14 and wave.status == "complete":
        deadline.check("before_kg_joint")
        features = cache.matrix(tuple(wave.attempts[index].trace.endpoint for index in pool))
        belief = posterior.joint(features)
        numerical = _json_hash(
            [belief.mean.tolist(), belief.covariance.tolist(), posterior.noise.tolist()]
        )
        indices = tuple(range(len(pool)))
        problem = SoftKGProblem(
            indices, (0, 1), PreferenceMeasure(np.array([[0.5, 0.5]])), np.ones(len(pool))
        )
        batch = EvaluationBatch(indices, np.ones(len(pool)))
        seed = int(_json_hash([wave.semantic_history_sha256, "evolution-kg-primary"])[:16], 16)
        engine = GaussianSoftKG(
            problem,
            temperature=0.25,
            observed_outputs=(0, 1),
            n_fantasies=512,
            standard_error_multiplier=1,
            seed=seed,
            relative_eigenvalue_cutoff=1e-10,
            candidate_chunk_size=64,
            fantasy_chunk_size=512,
        )
        if wave.variant == "singleton_kg":
            # Fixed belief/problem: rescoring after only removing IDs gives the
            # same singleton marginals. Retain the exact sequential eligible sets.
            result = engine.score(belief, batch)
            chosen = result.ranked_evaluation_indices[:14]
            primary = {
                "singleton": numerical_document(result),
                "sequential_eligible": [
                    list(index for index in indices if index not in chosen[:step])
                    for step in range(len(chosen))
                ],
            }
            groups = [(index,) for index in chosen] if len(chosen) == 14 else []
            stats = [
                (float(result.estimate[index]), float(result.standard_error[index]))
                for index in chosen
            ]
        else:
            result = engine.select_joint_beam(
                belief,
                batch,
                batch_size=14,
                beam_width=4,
                max_groups_scored=768,
                fixed_budget=fixed_budget,
            )
            primary = numerical_document(result)
            if fixed_budget:
                primary["optional_rule_would_stop_on_fixed_budget_frontier"] = not bool(
                    result.final_result.selected_evaluation_indices
                )
                primary["legacy_frontier_recomputed"] = False
            chosen = result.selected_evaluation_indices
            groups = [chosen] if len(chosen) == 14 else []
            position = result.final_result.evaluation_batches.index(chosen) if chosen else None
            stats = (
                []
                if position is None
                else [
                    (
                        float(result.final_result.estimate[position]),
                        float(result.final_result.standard_error[position]),
                    )
                ]
            )
        deadline.check("after_kg_primary")
        status = "stopped_no_positive_kg_action"
        if groups:
            check_seed = int(
                _json_hash([wave.semantic_history_sha256, "evolution-kg-independent-check"])[:16],
                16,
            )
            checker = GaussianSoftKG(
                problem,
                temperature=0.25,
                observed_outputs=(0, 1),
                n_fantasies=1024,
                standard_error_multiplier=1,
                seed=check_seed,
                relative_eigenvalue_cutoff=1e-10,
                candidate_chunk_size=64,
                fantasy_chunk_size=512,
            )
            checked = checker.score_joint_groups(belief, batch, groups, max_groups=14)
            for index, (mean, se) in enumerate(stats):
                check_mean, check_se = (
                    float(checked.estimate[index]),
                    float(checked.standard_error[index]),
                )
                tolerance = 3 * math.sqrt(se * se + check_se * check_se) + 1e-6
                passed = (
                    all(math.isfinite(value) for value in (mean, se, check_mean, check_se))
                    and (fixed_budget or mean - se > 0)
                    and se <= 0.02
                    and abs(mean - check_mean) <= tolerance
                )
                checks.append(
                    {
                        "group": list(groups[index]),
                        "primary_mean": mean,
                        "primary_se": se,
                        "check_mean": check_mean,
                        "check_se": check_se,
                        "agreement_tolerance": tolerance,
                        "passed": passed,
                    }
                )
                if fixed_budget:
                    checks[-1]["positive_penalized_gain"] = mean - se > 0
                    checks[-1]["positivity_is_diagnostic_not_veto"] = True
            deadline.check("after_kg_fresh_check")
            if precision_recheck and len(checks) == 1 and not checks[0]["passed"]:
                original = checks[0]
                # Only finite, adequately precise disagreements qualify. Do not
                # turn bad numerical values or a precision failure into a retry.
                values = (mean, se, check_mean, check_se)
                if (
                    all(math.isfinite(value) for value in values)
                    and 0 <= se <= 0.02
                    and 0 <= check_se <= 0.02
                ):
                    from amp_challenge.generators.diffusion.native_acquisition_precision import (
                        recheck_fixed_batch,
                    )

                    recheck = recheck_fixed_batch(
                        problem,
                        belief,
                        batch,
                        groups[0],
                        wave.semantic_history_sha256,
                        deadline=deadline.deadline,
                    )
                    original["initial_passed"] = False
                    original["precision_recheck"] = recheck
                    original["passed"] = recheck["passed"]
                    deadline.check("after_kg_bounded_precision_recheck")
            status = (
                "selected" if all(row["passed"] for row in checks) else "stopped_kg_mc_instability"
            )
            if status == "selected":
                selected = tuple(pool[index] for index in chosen)
        posterior.check()
    return EvolutionSelection(
        wave.sha256,
        posterior.sha256,
        external_filter_sha256,
        pool,
        selected,
        status,
        primary,
        tuple(checks),
        numerical,
    )

"""Independent fixed-OOF proper scores, pair events, intervals and literal gates."""

from __future__ import annotations

import math

import numpy as np

from amp_challenge.benchmarks.ensemble_audit_math import (
    PREDICTORS,
    agree,
    family_interval,
    pair_scores,
    proper_scores,
    row_weights,
)


def _summaries(rows, probability, supported, dominant):
    indices = {
        "all": list(range(len(rows))),
        "dominant_union_holdout": [],
        "excluding_dominant_union": [],
    }
    for index, row in enumerate(rows):
        indices[
            "dominant_union_holdout"
            if row["union_component_id"] == dominant
            else "excluding_dominant_union"
        ].append(index)
    for field in ("canonical_target", "gram", "oof_fold"):
        for value in sorted({row[field] for row in rows}):
            indices[f"{field}:{value}"] = [i for i, row in enumerate(rows) if row[field] == value]
    result = {}
    for name, selected in indices.items():
        subset = [rows[index] for index in selected]
        if not subset:
            result[name] = None
            continue
        retained = sum(bool(supported[index]) for index in selected)
        result[name] = {
            "contexts": len(subset),
            "sequences": len({row["sequence_id"] for row in subset}),
            "unions": len({row["union_component_id"] for row in subset}),
            "positive": sum(row["label"] for row in subset),
            "primary_teacher_supported_rows": retained,
            "primary_teacher_supported_row_coverage": retained / len(subset),
            **proper_scores(subset, probability[selected]),
        }
    return result


def reliability(rows, probability):
    weights = row_weights(rows)
    result = []
    for index, (lower, upper) in enumerate(
        zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1], strict=True)
    ):
        selected = [
            i for i, p in enumerate(probability) if lower <= p and (p < upper or index == 4)
        ]
        total = math.fsum(float(weights[i]) for i in selected)
        result.append(
            {
                "bin": index,
                "rows": len(selected),
                "equal_union_mass": total,
                "mean_probability": math.fsum(float(weights[i] * probability[i]) for i in selected)
                / total
                if total
                else None,
                "observed_positive_fraction": math.fsum(
                    float(weights[i]) * rows[i]["label"] for i in selected
                )
                / total
                if total
                else None,
            }
        )
    return result


def reconstruct_statistics(rows, predictors, supported, pairs, seeds, members, recorded):
    """Verify numeric endpoints before using serialized values for exact gates."""
    means = {name: np.mean(predictors[name], axis=1) for name in PREDICTORS}
    reports = {
        name: _summaries(rows, p, supported, pairs["dominant_union"]) for name, p in means.items()
    }
    group_ids = sorted({row["union_component_id"] for row in rows})
    comparisons, margins = {}, {"raw_ood": 0.01, "prior": 0.02}
    sensitive = []
    for baseline, margin in margins.items():
        differences = []
        for group in group_ids:
            indices = [i for i, row in enumerate(rows) if row["union_component_id"] == group]
            subrows = [rows[i] for i in indices]
            differences.append(
                proper_scores(subrows, means["calibrated_ood"][indices])["component_log_loss"]
                - proper_scores(subrows, means[baseline][indices])["component_log_loss"]
            )
        key = f"calibrated_ood_minus_{baseline}"
        interval = family_interval(differences, 17009)
        endpoint = {
            "observed_unions": len(group_ids),
            "difference": math.fsum(differences) / len(differences),
            "family_95_interval": interval,
            "engineering_margin": margin,
        }
        actual = recorded["comparisons"][key]
        agree({k: actual[k] for k in endpoint}, endpoint, name=key)
        comparisons[key] = {
            **endpoint,
            "noninferiority_passed": actual["family_95_interval"][1] <= margin,
        }
        if abs(actual["family_95_interval"][1] - margin) <= 1e-10:
            sensitive.append(key + ".upper_margin")
    member_matrix = predictors["calibrated_ood"]
    joint, independent = pair_scores(pairs["primary"], rows, member_matrix)
    differences = joint - independent
    pair_groups = {pair[key] for pair in pairs["primary"] for key in ("left_union", "right_union")}
    enough = len(joint) >= 15 and len(pair_groups) >= 30
    interval = family_interval(differences, 17011) if enough else None
    seeds_report = []
    nonpositive = 0
    for index, seed in enumerate(seeds):
        values = member_matrix[:, index * members : (index + 1) * members]
        mixed, product = pair_scores(pairs["primary"], rows, values)
        difference = (
            math.fsum(float(value) for value in mixed - product) / len(mixed)
            if len(mixed)
            else None
        )
        item = {
            "seed": seed,
            "actual_bootstrap_model_members": members,
            "all_row_scores": proper_scores(rows, values.mean(axis=1)),
            "joint_difference": difference,
        }
        agree(recorded["actual_seed_ensembles"][index], item, name=f"seed-{seed}")
        saved = recorded["actual_seed_ensembles"][index]["joint_difference"]
        nonpositive += saved is not None and saved <= 0
        if saved is not None and abs(saved) <= 1e-10:
            sensitive.append(f"seed-{seed}.joint_zero")
        seeds_report.append(item)
    item = {
        "pairs": len(joint),
        "distinct_unions": len(pair_groups),
        "eligible_for_primary_endpoint": enough,
        "difference": math.fsum(float(value) for value in differences) / len(differences)
        if len(differences)
        else None,
        "family_95_interval": interval,
        "nonpositive_seed_ensembles": nonpositive,
    }
    saved = recorded["comparisons"]["joint_minus_independent"]
    agree({key: saved[key] for key in item}, item, name="joint-endpoints")
    item["pair_gate_passed"] = bool(
        enough and saved["family_95_interval"][1] < 0 and nonpositive >= 4
    )
    comparisons["joint_minus_independent"] = item
    if enough and abs(saved["family_95_interval"][1]) <= 1e-10:
        sensitive.append("joint_minus_independent.upper_zero")
    if saved["difference"] is not None and abs(saved["difference"]) <= 1e-10:
        sensitive.append("joint_minus_independent.difference_zero")
    stress, stress_product = pair_scores(pairs["dominant_stress"], rows, member_matrix)
    coverage = sum(bool(keep) for keep in supported) / len(rows)
    dominant_difference = (
        reports["calibrated_ood"]["dominant_union_holdout"]["component_log_loss"]
        - reports["prior"]["dominant_union_holdout"]["component_log_loss"]
    )
    saved_dominant = recorded["qualifications"]["dominant_logloss_difference_from_prior"]
    agree(saved_dominant, dominant_difference, name="dominant-difference")
    if abs(saved_dominant - 0.03) <= 1e-10:
        sensitive.append("dominant_difference.margin")
    mean_passed = (
        all(
            comparisons[f"calibrated_ood_minus_{baseline}"]["noninferiority_passed"]
            for baseline in margins
        )
        and saved_dominant <= 0.03
        and coverage >= 0.8
    )
    report = {
        "predictors": reports,
        "reliability": {name: reliability(rows, value) for name, value in means.items()},
        "comparisons": comparisons,
        "actual_seed_ensembles": seeds_report,
        "dominant_pair_stress": {
            "pairs": len(stress),
            "mixture_log_loss": math.fsum(float(value) for value in stress) / len(stress)
            if len(stress)
            else None,
            "independent_log_loss": math.fsum(float(value) for value in stress_product)
            / len(stress_product)
            if len(stress)
            else None,
            "descriptive_only": True,
        },
        "qualifications": {
            "exploratory_mean_passed": mean_passed,
            "exploratory_joint_passed": mean_passed and item["pair_gate_passed"],
            "supported_row_coverage": coverage,
            "dominant_logloss_difference_from_prior": dominant_difference,
            "production_input_eligible": False,
            "teacher_moments_available_to_adaptive_search": False,
        },
        "uncertainty_scope": "fixed_OOF_conditional_whole_union_or_disjoint_union_pair_resampling_not_latent_epistemic_calibration",
        "bootstrap_replicates": 2000,
        "family_comparisons": 3,
        "all_rows_scored": len(rows),
    }
    return report, sensitive

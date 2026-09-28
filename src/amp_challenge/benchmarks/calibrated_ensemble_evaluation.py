"""Label-blind pair inventory, exact finite-mixture moments and frozen gates.

All teacher member probabilities remain controller-private. Event-prediction
scores do not establish latent epistemic calibration or authorize exposure to
an adaptive learner.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from amp_challenge.benchmarks.calibrated_ensemble_contract import PREDICTORS, require
from amp_challenge.benchmarks.oracle_activity_independent import scores, weights

PAIR_FIELDS = {
    "example_id",
    "sequence_id",
    "union_component_id",
    "oof_fold",
    "canonical_target",
    "gram",
}
TARGETS = (("escherichia_coli", "negative"), ("staphylococcus_aureus", "positive"))


def _hash(*values):
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def build_pairs(sequences, metadata):
    require(
        all(set(row) == PAIR_FIELDS for row in metadata),
        "pair construction accepts label-free metadata only",
    )
    require(len({row["example_id"] for row in metadata}) == len(metadata), "duplicate pair context")
    counts = Counter(row["union_component_id"] for row in sequences)
    dominant = min(counts, key=lambda uid: (-counts[uid], uid))
    used, primary = set(), []
    for fold in range(5):
        for target, gram in TARGETS:
            groups = defaultdict(list)
            for index, row in enumerate(metadata):
                if (
                    row["oof_fold"] == fold
                    and (row["canonical_target"], row["gram"]) == (target, gram)
                    and row["union_component_id"] not in used
                ):
                    groups[row["union_component_id"]].append(index)
            ordered = sorted(groups, key=lambda uid: (_hash("pair-v1", fold, target, uid), uid))
            for offset in range(0, len(ordered) - 1, 2):
                left, right = ordered[offset : offset + 2]
                selected = [
                    min(
                        groups[uid],
                        key=lambda index: (
                            _hash("context-v1", metadata[index]["example_id"]),
                            metadata[index]["example_id"],
                        ),
                    )
                    for uid in (left, right)
                ]
                primary.append(
                    {
                        "outer_fold": fold,
                        "canonical_target": target,
                        "gram": gram,
                        "left_index": selected[0],
                        "right_index": selected[1],
                        "left_union": left,
                        "right_union": right,
                    }
                )
                used.update((left, right))
    stress = []
    for target, gram in TARGETS:
        available = sorted(
            [
                i
                for i, row in enumerate(metadata)
                if row["union_component_id"] == dominant
                and (row["canonical_target"], row["gram"]) == (target, gram)
            ],
            key=lambda i: (
                _hash("context-v1", metadata[i]["example_id"]),
                metadata[i]["example_id"],
            ),
        )
        while available and len(stress) < 16:
            left = available.pop(0)
            right = next(
                (
                    i
                    for i in available
                    if metadata[i]["sequence_id"] != metadata[left]["sequence_id"]
                ),
                None,
            )
            if right is None:
                break
            available.remove(right)
            stress.append(
                {
                    "outer_fold": metadata[left]["oof_fold"],
                    "canonical_target": target,
                    "gram": gram,
                    "left_index": left,
                    "right_index": right,
                    "left_union": dominant,
                    "right_union": dominant,
                }
            )
    return {
        "primary": primary,
        "dominant_stress": stress,
        "dominant_union": dominant,
        "outcomes_used_in_construction": False,
    }


def finite_mixture_moments(probabilities):
    """Rows=candidates, columns=equal-mass members; exact mixture divisor M."""
    values = np.asarray(probabilities, dtype=np.float64)
    require(
        values.ndim == 2
        and 1 <= values.shape[0] <= 858
        and 1 <= values.shape[1] <= 40
        and np.isfinite(values).all()
        and np.all((values >= 0) & (values <= 1)),
        "invalid bounded member matrix",
    )
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    covariance = centered @ centered.T / values.shape[1]
    noise = (values * (1 - values)).mean(axis=1)
    require(
        np.allclose(np.diag(covariance) + noise, mean * (1 - mean), atol=1e-12, rtol=0),
        "finite-mixture observation variance identity failed",
    )
    return mean, covariance, noise


def pair_losses(pairs, rows, probabilities):
    probability = np.asarray(probabilities, dtype=np.float64)
    require(
        probability.ndim == 2
        and probability.shape[0] == len(rows)
        and np.isfinite(probability).all()
        and np.all((probability >= 0) & (probability <= 1)),
        "invalid pair member probabilities",
    )
    mixture, independent = [], []
    for pair in pairs:
        left, right = pair["left_index"], pair["right_index"]
        require(
            rows[left]["oof_fold"] == rows[right]["oof_fold"] == pair["outer_fold"],
            "pair crosses independently trained outer teachers",
        )
        outcomes = [
            probability[index] if rows[index]["label"] else 1 - probability[index]
            for index in (left, right)
        ]
        joint = float(np.mean(outcomes[0] * outcomes[1]))
        product = float(outcomes[0].mean() * outcomes[1].mean())
        mixture.append(float(-np.log(np.clip(joint, 1e-12, 1.0))))
        independent.append(float(-np.log(np.clip(product, 1e-12, 1.0))))
    return np.asarray(mixture), np.asarray(independent)


def _subset_reports(rows, probabilities, dominant, supported):
    subsets = {
        "all": list(range(len(rows))),
        "dominant_union_holdout": [
            i for i, row in enumerate(rows) if row["union_component_id"] == dominant
        ],
        "excluding_dominant_union": [
            i for i, row in enumerate(rows) if row["union_component_id"] != dominant
        ],
    }
    for field in ("canonical_target", "gram", "oof_fold"):
        for value in sorted({row[field] for row in rows}):
            subsets[f"{field}:{value}"] = [i for i, row in enumerate(rows) if row[field] == value]
    result = {}
    for name, indices in subsets.items():
        if not indices:
            result[name] = None
            continue
        selected = [rows[index] for index in indices]
        result[name] = {
            "contexts": len(indices),
            "sequences": len({row["sequence_id"] for row in selected}),
            "unions": len({row["union_component_id"] for row in selected}),
            "positive": sum(row["label"] for row in selected),
            "primary_teacher_supported_rows": int(supported[indices].sum()),
            "primary_teacher_supported_row_coverage": float(supported[indices].mean()),
            **scores(selected, probabilities[indices]),
        }
    return result


def _reliability(rows, probability):
    weight = weights(rows)
    labels = np.asarray([row["label"] for row in rows])
    bins = np.minimum(
        np.searchsorted([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], probability, side="right") - 1, 4
    )
    report = []
    for index in range(5):
        selected = bins == index
        mass = float(weight[selected].sum())
        report.append(
            {
                "bin": index,
                "rows": int(selected.sum()),
                "equal_union_mass": mass,
                "mean_probability": float(np.dot(weight[selected], probability[selected]) / mass)
                if mass
                else None,
                "observed_positive_fraction": float(
                    np.dot(weight[selected], labels[selected]) / mass
                )
                if mass
                else None,
            }
        )
    return report


def _family_interval(values, seed):
    values = np.asarray(values, dtype=np.float64)
    require(
        values.ndim == 1 and values.size > 0 and np.isfinite(values).all(),
        "invalid paired score differences",
    )
    draws = np.random.default_rng(seed).integers(0, len(values), size=(2000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.05 / 6, 1 - 0.05 / 6]).tolist()


def evaluate_study(rows, predictor_members, supported, pairs, seeds, members_per_seed):
    require(set(predictor_members) == set(PREDICTORS), "predictor inventory differs")
    means = {name: values.mean(axis=1) for name, values in predictor_members.items()}
    dominant = pairs["dominant_union"]
    reports = {
        name: _subset_reports(rows, values, dominant, supported) for name, values in means.items()
    }
    groups = sorted({row["union_component_id"] for row in rows})
    comparisons = {}
    for baseline, margin in (("raw_ood", 0.01), ("prior", 0.02)):
        differences = []
        for group in groups:
            selected = [i for i, row in enumerate(rows) if row["union_component_id"] == group]
            subset = [rows[index] for index in selected]
            differences.append(
                scores(subset, means["calibrated_ood"][selected])["component_log_loss"]
                - scores(subset, means[baseline][selected])["component_log_loss"]
            )
        interval = _family_interval(differences, 17009)
        comparisons[f"calibrated_ood_minus_{baseline}"] = {
            "observed_unions": len(groups),
            "difference": float(np.mean(differences)),
            "family_95_interval": interval,
            "engineering_margin": margin,
            "noninferiority_passed": bool(interval[1] <= margin),
        }
    mixed, independent = pair_losses(pairs["primary"], rows, predictor_members["calibrated_ood"])
    pair_groups = {pair[key] for pair in pairs["primary"] for key in ("left_union", "right_union")}
    require(
        len(pair_groups) == 2 * len(pairs["primary"]),
        "primary pair groups are not globally disjoint",
    )
    eligible = len(mixed) >= 15 and len(pair_groups) >= 30
    difference = mixed - independent
    interval = _family_interval(difference, 17011) if eligible else None
    seed_reports = []
    for index, seed in enumerate(seeds):
        selected = slice(index * members_per_seed, (index + 1) * members_per_seed)
        seeded = predictor_members["calibrated_ood"][:, selected]
        left, right = pair_losses(pairs["primary"], rows, seeded)
        seed_reports.append(
            {
                "seed": seed,
                "actual_bootstrap_model_members": members_per_seed,
                "all_row_scores": scores(rows, seeded.mean(axis=1)),
                "joint_difference": float((left - right).mean()) if len(left) else None,
            }
        )
    nonpositive = sum(
        row["joint_difference"] is not None and row["joint_difference"] <= 0 for row in seed_reports
    )
    stress_joint, stress_independent = pair_losses(
        pairs["dominant_stress"], rows, predictor_members["calibrated_ood"]
    )
    comparisons["joint_minus_independent"] = {
        "pairs": len(mixed),
        "distinct_unions": len(pair_groups),
        "eligible_for_primary_endpoint": eligible,
        "difference": float(difference.mean()) if len(difference) else None,
        "family_95_interval": interval,
        "nonpositive_seed_ensembles": nonpositive,
        "pair_gate_passed": bool(eligible and interval[1] < 0 and nonpositive >= 4),
    }
    coverage = float(np.mean(supported))
    dominant_difference = (
        reports["calibrated_ood"]["dominant_union_holdout"]["component_log_loss"]
        - reports["prior"]["dominant_union_holdout"]["component_log_loss"]
    )
    mean_passed = bool(
        all(
            comparisons[f"calibrated_ood_minus_{name}"]["noninferiority_passed"]
            for name in ("raw_ood", "prior")
        )
        and dominant_difference <= 0.03
        and coverage >= 0.8
    )
    return {
        "predictors": reports,
        "reliability": {name: _reliability(rows, value) for name, value in means.items()},
        "comparisons": comparisons,
        "actual_seed_ensembles": seed_reports,
        "dominant_pair_stress": {
            "pairs": len(stress_joint),
            "mixture_log_loss": float(stress_joint.mean()) if len(stress_joint) else None,
            "independent_log_loss": float(stress_independent.mean()) if len(stress_joint) else None,
            "descriptive_only": True,
        },
        "qualifications": {
            "exploratory_mean_passed": mean_passed,
            "exploratory_joint_passed": bool(
                mean_passed and comparisons["joint_minus_independent"]["pair_gate_passed"]
            ),
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

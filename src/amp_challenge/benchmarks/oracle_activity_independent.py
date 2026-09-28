"""Independent NumPy/stdlib reconstruction of the frozen binary-activity audit.

No fitting, producer prediction, weighting, or summary function is imported.
The stored inner models have no coefficients; their probabilities are checked
across executions, while their priors, supports and selection scores reconstruct.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

ARMS = ("descriptors_context", "esm_context", "spectral_context", "esm_spectral_context")
PENALTIES = (0.01, 0.1, 1.0, 10.0)
SEEDS = (17, 42, 91, 137, 271)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_values(left, right, *, location="root", atol=1e-10, rtol=0.0):
    """Exact structure/types/integers; frozen floating-point tolerances only."""
    require(type(left) is type(right), f"type mismatch: {location}")
    if isinstance(left, dict):
        require(set(left) == set(right), f"key mismatch: {location}")
        for key in left:
            compare_values(
                left[key], right[key], location=f"{location}.{key}", atol=atol, rtol=rtol
            )
    elif isinstance(left, list | tuple):
        require(len(left) == len(right), f"length mismatch: {location}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_values(a, b, location=f"{location}[{index}]", atol=atol, rtol=rtol)
    elif isinstance(left, float):
        if location.rsplit(".", 1)[-1] == "penalty":
            atol = rtol = 0.0
        require(
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= atol + rtol * abs(right),
            f"floating-point mismatch: {location}",
        )
    else:
        require(left == right, f"identity mismatch: {location}")


def assign_unions(sequences, folds):
    members = Counter(row["union_component_id"] for row in sequences)
    require(len(members) >= folds, "insufficient assignment groups")
    loads = [0 for _ in range(folds)]
    assigned = {}
    for group, count in sorted(members.items(), key=lambda item: (-item[1], item[0])):
        destination = loads.index(min(loads))
        assigned[group] = destination
        loads[destination] += count
    return assigned


def weights(rows):
    counts = Counter(row["union_component_id"] for row in rows)
    require(bool(counts), "empty score/training support")
    return np.array([1 / len(counts) / counts[row["union_component_id"]] for row in rows])


def scores(rows, probabilities):
    require(len(rows) == len(probabilities) and bool(rows), "unaligned score rows")
    losses, errors = [], []
    grouped = defaultdict(list)
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        require(type(row["label"]) is int and row["label"] in (0, 1), "nonbinary label")
        require(math.isfinite(probability) and 0 <= probability <= 1, "invalid probability")
        clipped = min(max(float(probability), 1e-12), 1 - 1e-12)
        losses.append(-math.log(clipped) if row["label"] else -math.log1p(-clipped))
        errors.append((float(probability) - row["label"]) ** 2)
        grouped[row["union_component_id"]].append(index)

    def balanced(values):
        return math.fsum(
            math.fsum(values[i] for i in indices) / len(indices) for indices in grouped.values()
        ) / len(grouped)

    return {
        "component_log_loss": balanced(losses),
        "component_brier": balanced(errors),
        "row_log_loss": math.fsum(losses) / len(rows),
        "row_brier": math.fsum(errors) / len(rows),
    }


def prior_details(training, row):
    matched = [
        r
        for r in training
        if (r["canonical_target"], r["gram"]) == (row["canonical_target"], row["gram"])
    ]
    groups = len({r["union_component_id"] for r in matched})
    classes = sorted({r["label"] for r in matched})
    if len({r["label"] for r in training}) == 1:
        reason = "constant_training_labels"
    elif not matched:
        reason = "unseen_target_gram_context"
    elif groups < 2:
        reason = "context_fewer_than_two_training_unions"
    elif len(classes) == 1:
        reason = "constant_context_labels"
    else:
        reason = None
    if matched:
        selected, level = matched, "target_gram"
    else:
        gram = [r for r in training if r["gram"] == row["gram"]]
        selected, level = (gram, "gram") if gram else (training, "global")
    labels = defaultdict(list)
    for selected_row in selected:
        labels[selected_row["union_component_id"]].append(selected_row["label"])
    probability = (
        math.fsum(math.fsum(values) / len(values) for values in labels.values()) + 0.5
    ) / (len(labels) + 1)
    return {
        "abstained": reason is not None,
        "reason": reason,
        "fallback_prior_level": level if reason else None,
        "training_context_unions": groups,
        "training_context_classes": classes,
        "diagnostic_prior_probability": probability,
        "diagnostic_prior_level": level,
        "diagnostic_prior_training_unions": len(labels),
    }


def _subsets(rows, dominant, prior=False):
    subsets = {
        "all": rows,
        "dominant_union_holdout": [r for r in rows if r["union_component_id"] == dominant],
        "excluding_dominant_union": [r for r in rows if r["union_component_id"] != dominant],
    }
    for field in ("canonical_target", "gram", "oof_fold"):
        for value in sorted({r[field] for r in rows}):
            subsets[f"{field}:{value}"] = [r for r in rows if r[field] == value]
    output = {}
    for name, subset in subsets.items():
        if not subset:
            output[name] = None
            continue
        details = (
            {
                "prior_level_counts": dict(Counter(r["diagnostic_prior_level"] for r in subset)),
                "training_prior_union_support_counts": dict(
                    Counter(str(r["diagnostic_prior_training_unions"]) for r in subset)
                ),
                "training_context_union_support_counts": dict(
                    Counter(str(r["training_context_unions"]) for r in subset)
                ),
            }
            if prior
            else {
                "abstentions": sum(r["abstained"] for r in subset),
                "fallback_reasons": dict(Counter(r["reason"] for r in subset if r["abstained"])),
            }
        )
        output[name] = {
            "contexts": len(subset),
            "sequences": len({r["sequence_id"] for r in subset}),
            "unions": len({r["union_component_id"] for r in subset}),
            "positive": sum(r["label"] for r in subset),
            "negative": sum(r["label"] == 0 for r in subset),
            **details,
            **scores(
                subset,
                [r["diagnostic_prior_probability"] if prior else r["probability"] for r in subset],
            ),
        }
    return output


def reconstruct_report(predictions, sequences, replicates=2000):
    counts = Counter(r["union_component_id"] for r in sequences)
    dominant = sorted(counts, key=lambda uid: (-counts[uid], uid))[0]
    arm_rows = {arm: [r for r in predictions if r["arm"] == arm] for arm in ARMS}
    group_loss = {}
    for arm, rows in arm_rows.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["union_component_id"]].append(row)
        group_loss[arm] = {
            uid: scores(subset, [r["probability"] for r in subset])
            for uid, subset in groups.items()
        }
    ids = sorted(group_loss[ARMS[0]])
    comparisons = {}
    for arm in ARMS[1:]:
        require(set(group_loss[arm]) == set(ids), "unpaired bootstrap groups")
        delta = np.array(
            [
                [
                    group_loss[arm][uid][metric] - group_loss[ARMS[0]][uid][metric]
                    for metric in ("component_log_loss", "component_brier")
                ]
                for uid in ids
            ]
        )
        bootstrap = []
        for seed in SEEDS:
            indices = np.random.default_rng(seed).integers(len(ids), size=(replicates, len(ids)))
            sampled = np.mean(delta[indices], axis=1)
            bootstrap.append(
                {
                    "seed": seed,
                    "log_loss_bonferroni_95_family_interval": np.quantile(
                        sampled[:, 0], [0.05 / 6, 1 - 0.05 / 6]
                    ).tolist(),
                    "brier_descriptive_95_interval": np.quantile(
                        sampled[:, 1], [0.025, 0.975]
                    ).tolist(),
                }
            )
        comparisons[arm] = {
            "difference_direction": "arm_minus_descriptors_lower_is_better",
            "observed_unions": len(ids),
            "log_loss_difference": float(np.mean(delta[:, 0])),
            "brier_difference": float(np.mean(delta[:, 1])),
            "primary_seed": 17,
            "bootstrap_results": bootstrap,
        }
    return {
        "arms": {arm: _subsets(rows, dominant) for arm, rows in arm_rows.items()},
        "diagnostic_prior_baseline": {
            "scope": "train_only_smoothed_context_hierarchy_descriptive_not_primary_comparison",
            "optimizer_fits": 0,
            "primary_comparison_family_member": False,
            "scores": _subsets(arm_rows[ARMS[0]], dominant, prior=True),
        },
        "paired_comparisons": comparisons,
        "dominant_union_id": dominant,
        "dominant_union_sequences": counts[dominant],
        "uncertainty_scope": "conditional_OOF_predictor_whole_union_resampling_not_retraining_variance",
        "primary_family": "three_log_loss_differences_against_descriptors_context",
        "bootstrap_replicates_per_seed": replicates,
        "calibration_fitted": False,
        "binary_label_not_continuous_MIC": True,
        "safety_model_fitted": False,
        "production_input_eligible": False,
        "scientific_superiority_accepted": False,
    }


def _keyed(rows, fields):
    keyed = {tuple(row[field] for field in fields): row for row in rows}
    require(len(keyed) == len(rows), f"duplicate identities: {fields}")
    return keyed


def reconstruct_execution(
    document, sequences, contexts, matrices, *, penalties=PENALTIES, replicates=2000
):
    """Validate saved models/predictions without an optimizer or an oracle call."""
    outer = assign_unions(sequences, 5)
    require(
        all(outer[r["union_component_id"]] == r["oof_fold"] for r in sequences),
        "outer assignment differs",
    )
    by_id = _keyed(contexts, ("example_id",))
    predictions = _keyed(document["predictions"], ("arm", "example_id"))
    require(
        set(predictions) == {(arm, row["example_id"]) for arm in ARMS for row in contexts},
        "outer prediction inventory differs",
    )
    models = _keyed(document["models"], ("arm", "outer_fold"))
    require(
        set(models) == {(arm, fold) for arm in ARMS for fold in range(5)}, "model inventory differs"
    )
    inner_predictions = _keyed(
        document["inner_predictions"], ("arm", "outer_fold", "inner_fold", "penalty", "example_id")
    )
    expected_inner_keys = set()
    assignments = []
    for fold in range(5):
        indices = [i for i, row in enumerate(contexts) if row["oof_fold"] != fold]
        heldout = [row for row in contexts if row["oof_fold"] == fold]
        training = [contexts[i] for i in indices]
        inner = assign_unions([r for r in sequences if r["oof_fold"] != fold], 3)
        assignments.append({"outer_fold": fold, "inner_union_folds": inner})
        for arm in ARMS:
            selection = []
            for penalty in penalties:
                pooled_rows, pooled_probabilities = [], []
                for inner_fold in range(3):
                    fitted = [r for r in training if inner[r["union_component_id"]] != inner_fold]
                    validating = [
                        r for r in training if inner[r["union_component_id"]] == inner_fold
                    ]
                    require(bool(fitted) and bool(validating), "empty inner split")
                    for row in validating:
                        key = (arm, fold, inner_fold, penalty, row["example_id"])
                        expected_inner_keys.add(key)
                        require(key in inner_predictions, "missing inner prediction")
                        observed = inner_predictions[key]
                        detail = prior_details(fitted, row)
                        expected = {
                            "arm": arm,
                            "outer_fold": fold,
                            "inner_fold": inner_fold,
                            "penalty": penalty,
                            "example_id": row["example_id"],
                            "probability": observed["probability"],
                            **detail,
                        }
                        compare_values(observed, expected, location="inner prediction")
                        if detail["abstained"]:
                            compare_values(
                                observed["probability"],
                                detail["diagnostic_prior_probability"],
                                location="inner abstention probability",
                            )
                        pooled_rows.append(row)
                        pooled_probabilities.append(observed["probability"])
                selection.append({"penalty": penalty, **scores(pooled_rows, pooled_probabilities)})
            # Ties use the serialized (audited within tolerance) inner scores.
            # Independent summation may differ by an ulp, but must not change an
            # explicitly exact tie in the declared producer selection rule.
            model = models[(arm, fold)]
            compare_values(model["inner_selection"], selection, location="inner selection scores")
            chosen = min(
                model["inner_selection"],
                key=lambda result: (result["component_log_loss"], -result["penalty"]),
            )["penalty"]
            require(model["penalty"] == chosen, "selected penalty identity differs")
            _reconstruct_outer_model(model, matrices[arm], indices, training, contexts)
            for row in heldout:
                observed = predictions[(arm, row["example_id"])]
                detail = prior_details(training, row)
                expected = {"arm": arm, **row, "probability": observed["probability"], **detail}
                compare_values(observed, expected, location="outer prediction identity/support")
                index = next(
                    i
                    for i, candidate in enumerate(contexts)
                    if candidate["example_id"] == row["example_id"]
                )
                probability = (
                    detail["diagnostic_prior_probability"]
                    if detail["abstained"]
                    else _model_probability(model, matrices[arm][index], row)
                )
                compare_values(
                    observed["probability"], probability, location="outer model probability"
                )
    require(set(inner_predictions) == expected_inner_keys, "extra inner prediction identity")
    compare_values(document["inner_assignments"], assignments, location="inner assignments", atol=0)
    require(len(by_id) == len(contexts), "duplicate contexts")
    return reconstruct_report(document["predictions"], sequences, replicates)


def _design_row(model, values, row):
    standardized = (np.asarray(values) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return np.concatenate(
        (
            [1.0],
            standardized,
            [float(row["canonical_target"] == value) for value in model["targets"]],
            [float(row["gram"] == value) for value in model["grams"]],
        )
    )


def _model_probability(model, values, row):
    logit = float(np.dot(_design_row(model, values, row), np.asarray(model["coefficients"])))
    return 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))


def _reconstruct_outer_model(model, matrix, indices, training, contexts):
    require(
        model["training_example_ids"] == sorted(r["example_id"] for r in training),
        "model training IDs differ",
    )
    require(
        model["training_union_ids"] == sorted({r["union_component_id"] for r in training}),
        "model training unions differ",
    )
    require(
        list(model["targets"]) == sorted({r["canonical_target"] for r in training}),
        "training target vocabulary differs",
    )
    require(
        list(model["grams"]) == sorted({r["gram"] for r in training}),
        "training Gram vocabulary differs",
    )
    require(
        model["constant_training_labels"] is (len({r["label"] for r in training}) == 1),
        "constant-label support differs",
    )
    x = np.asarray(matrix, dtype=np.float64)[indices]
    weight = weights(training)
    mean = np.sum(x * weight[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * weight[:, None], axis=0))
    scale[(scale == 0) | np.all(x == x[0], axis=0)] = 1.0
    compare_values(model["mean"], mean.tolist(), location="training mean", atol=1e-8, rtol=1e-8)
    compare_values(model["scale"], scale.tolist(), location="training scale", atol=1e-8, rtol=1e-8)
    require(np.all(np.asarray(model["scale"]) > 0), "nonpositive training scale")
    coefficients = np.asarray(model["coefficients"])
    require(
        coefficients.shape == (1 + x.shape[1] + len(model["targets"]) + len(model["grams"]),),
        "coefficient schema differs",
    )
    require(np.isfinite(coefficients).all(), "nonfinite coefficients")
    require(
        type(model["iterations"]) is int and 0 <= model["iterations"] <= 1000,
        "invalid iteration count",
    )
    if model["constant_training_labels"]:
        require(
            model["objective"] is None and np.all(coefficients == 0) and model["iterations"] == 0,
            "constant-label bypass differs",
        )
    else:
        logits = np.asarray(
            [np.dot(_design_row(model, matrix[i], contexts[i]), coefficients) for i in indices]
        )
        labels = np.asarray([r["label"] for r in training])
        objective = float(
            np.sum(weight * (np.logaddexp(0, logits) - labels * logits))
            + model["penalty"] * np.sum(coefficients[1:] ** 2) / 2
        )
        compare_values(model["objective"], objective, location="penalized training objective")


def compare_executions(first, second):
    """Apply predeclared identities/tolerances, excluding scheduler/elapsed fields."""
    for field in ("predictions", "inner_predictions", "inner_assignments", "metrics"):
        compare_values(first[field], second[field], location=field)
    require(len(first["models"]) == len(second["models"]), "model count differs")
    for one, two in zip(first["models"], second["models"], strict=True):
        require(set(one) == set(two), "model keys differ")
        for field in one:
            if field == "iterations":
                continue  # Optimizer iteration counts are descriptive, not a declared identity.
            atol, rtol = (
                (1e-8, 1e-8) if field in ("mean", "scale", "coefficients") else (1e-10, 0.0)
            )
            if field == "penalty":
                atol = 0.0
            compare_values(one[field], two[field], location=f"model.{field}", atol=atol, rtol=rtol)
    return {
        "passed": True,
        "optimizer_iteration_counts_equal": [m["iterations"] for m in first["models"]]
        == [m["iterations"] for m in second["models"]],
    }

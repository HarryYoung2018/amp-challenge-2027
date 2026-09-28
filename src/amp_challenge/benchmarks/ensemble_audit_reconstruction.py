"""Independently replay every saved nested model without invoking an optimizer."""

from __future__ import annotations

import numpy as np

from amp_challenge.benchmarks.ensemble_audit_math import (
    CLAIMS,
    PAIR_FIELDS,
    PREDICTORS,
    SEEDS,
    VIEWS,
    agree,
    ood_partition,
    pair_inventory,
    prior,
    require,
    row_weights,
    sigmoid,
    transform,
    union_assignment,
)
from amp_challenge.benchmarks.ensemble_audit_statistics import reconstruct_statistics

ARRAYS = {
    "model_parameters",
    "bootstrap_group_draws",
    "outer_raw_view_members",
    "outer_calibrated_unmasked_members",
    "outer_context_supported",
    "outer_ood_abstained",
    "outer_hierarchical_prior",
    "inner_raw_view_members",
    "inner_effective_mixture_members",
    "inner_context_row_indices",
    "inner_calibration_mask",
    "inner_hierarchical_prior",
    "inner_context_supported",
    "inner_ood_abstained",
    "outer_covariance_blocks",
    "outer_latent_probability_variance",
    "outer_bernoulli_noise_diagonal",
    *("predictor_" + name for name in PREDICTORS),
}


def _partition(name, seq_indices, training, prediction, sequences, rows, geometry):
    by_sequence = {row["sequence_id"]: index for index, row in enumerate(sequences)}
    query_indices = [by_sequence[rows[index]["sequence_id"]] for index in prediction]
    train_rows = [rows[index] for index in training]
    details = [prior(train_rows, rows[index]) for index in prediction]
    record = {
        "identity": name,
        "sequence_indices": seq_indices,
        "training_context_indices": training,
        "prediction_context_indices": prediction,
        **ood_partition(
            geometry[seq_indices], [sequences[i] for i in seq_indices], geometry[query_indices]
        ),
        "prediction_context_details": details,
    }
    support = np.asarray([detail["reason"] is None for detail in details], dtype=bool)
    priors = np.asarray([detail["diagnostic_prior_probability"] for detail in details])
    rejected = np.asarray(record["prediction_ood_abstained"], dtype=bool)
    return record, support, priors, rejected


def reconstruct(result, sequences, rows, matrices, geometry, *, seeds=SEEDS, members=8):
    require(
        1 <= len(sequences) <= 415
        and 1 <= len(rows) <= 858
        and 1 <= len(seeds) <= 5
        and 1 <= members <= 8,
        "reconstruction envelope differs",
    )
    require(set(result["arrays"]) == ARRAYS, "saved array inventory differs")
    require(set(matrices) == set(VIEWS), "independent feature inventory differs")
    arrays = result["arrays"]
    require(
        all(
            array.dtype in (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))
            and np.isfinite(array).all()
            for array in arrays.values()
        ),
        "unsafe saved numerical array",
    )
    agree(result["claims"], CLAIMS, name="claims")
    agree(result["seeds"], list(seeds), name="seeds")
    agree(result["members_per_seed"], members, name="members")
    assignments = union_assignment(sequences, 5)
    require(
        all(
            row["namespace"] == "oracle"
            and row["oof_fold"] == assignments[row["union_component_id"]]
            for row in sequences
        ),
        "outer assignment or oracle namespace differs",
    )
    by_sequence = {row["sequence_id"]: row for row in sequences}
    require(
        len(by_sequence) == len(sequences)
        and len({row["example_id"] for row in rows}) == len(rows),
        "input IDs repeat",
    )
    require(
        all(
            row["namespace"] == "oracle"
            and row["sequence_id"] in by_sequence
            and row["union_component_id"] == by_sequence[row["sequence_id"]]["union_component_id"]
            and row["oof_fold"] == by_sequence[row["sequence_id"]]["oof_fold"]
            for row in rows
        ),
        "context/sequence identity differs",
    )
    # Freeze independent pairing using strictly label-free metadata before scoring.
    pairs = pair_inventory(sequences, [{key: row[key] for key in PAIR_FIELDS} for row in rows])
    agree(result["pairs"], pairs, name="pairs", exact=True)
    count = len(seeds) * members
    require(
        len(result["models"]) == 40 * count
        and len(result["calibrators"]) == 5 * count
        and len(result["partitions"]) == 20
        and len(result["outer_folds"]) == len(result["covariance_blocks"]) == 5,
        "complete fit/partition inventory differs",
    )
    expected = {
        "outer_raw_view_members": np.zeros((len(rows), count, 2)),
        "outer_calibrated_unmasked_members": np.zeros((len(rows), count)),
        "outer_context_supported": np.zeros(len(rows), dtype=bool),
        "outer_ood_abstained": np.zeros(len(rows), dtype=bool),
        "outer_hierarchical_prior": np.zeros(len(rows)),
    }
    model_position, parameter_offset, calibrator_position, group_offset, inner_offset = (
        0,
        0,
        0,
        0,
        0,
    )
    inner_results, actual_draws, outer_ledgers = [], [], []
    diagnostics = {
        "base_gradient_max_absolute": 0.0,
        "calibration_projected_gradient_max_absolute": 0.0,
        "base_optimizer_invocations": 0,
        "calibration_optimizer_invocations": 0,
    }

    def replay_model(identity, training, prediction, draw):
        nonlocal model_position, parameter_offset
        record = result["models"][model_position]
        training_rows = [rows[index] for index in training]
        targets = sorted({row["canonical_target"] for row in training_rows})
        grams = sorted({row["gram"] for row in training_rows})
        x = matrices[identity["view"]]
        dimension = x.shape[1]
        coefficient_count = 1 + dimension + len(targets) + len(grams)
        total = 2 * dimension + coefficient_count
        values = arrays["model_parameters"][parameter_offset : parameter_offset + total]
        require(values.shape == (total,), "model parameter span differs")
        saved_mean, saved_scale, beta = (
            values[:dimension],
            values[dimension : 2 * dimension],
            values[2 * dimension :],
        )
        require(np.all(saved_scale > 0), "nonpositive saved feature scale")
        weights = row_weights(training_rows, draw)
        mean, scale = transform(x[training], weights)
        agree(saved_mean, mean, name="training-mean", coefficient=True)
        agree(saved_scale, scale, name="training-scale", coefficient=True)

        def design(indices):
            context = np.asarray(
                [
                    [int(rows[index]["canonical_target"] == value) for value in targets]
                    + [int(rows[index]["gram"] == value) for value in grams]
                    for index in indices
                ]
            )
            return np.concatenate(
                (np.ones((len(indices), 1)), (x[indices] - saved_mean) / saved_scale, context),
                axis=1,
            )

        labels = np.asarray([row["label"] for row in training_rows])
        require(
            all(type(row["label"]) is int and row["label"] in (0, 1) for row in training_rows),
            "nonbinary training outcome",
        )
        constant = len(set(labels)) == 1
        if constant:
            agree(beta, np.zeros_like(beta), name="constant-placeholder", exact=True)
            objective = None
            require(record["iterations"] == 0, "constant fit invoked iterations")
        else:
            train_design = design(training)
            logits = train_design @ beta
            objective = float(
                np.sum(weights * (np.logaddexp(0, logits) - labels * logits))
                + np.sum(beta[1:] ** 2) / 2
            )
            gradient = np.sum(
                (weights * (sigmoid(logits) - labels))[:, None] * train_design, axis=0
            )
            gradient[1:] += beta[1:]
            diagnostics["base_gradient_max_absolute"] = max(
                diagnostics["base_gradient_max_absolute"], float(np.max(np.abs(gradient)))
            )
            diagnostics["base_optimizer_invocations"] += 1
            require(
                type(record["iterations"]) is int and 1 <= record["iterations"] <= 1000,
                "base iteration envelope differs",
            )
        metadata = {
            **identity,
            "model_id": model_position,
            "parameter_offset": parameter_offset,
            "feature_dimension": dimension,
            "coefficient_count": coefficient_count,
            "targets": targets,
            "grams": grams,
            "constant_training_labels": constant,
            "iterations": record["iterations"],
            "objective": objective,
        }
        agree(record, metadata, name=f"model-{model_position}")
        model_position += 1
        parameter_offset += total
        return sigmoid(design(prediction) @ beta)

    for outer in range(5):
        training = [i for i, row in enumerate(rows) if row["oof_fold"] != outer]
        held = [i for i, row in enumerate(rows) if row["oof_fold"] == outer]
        seq_train = [i for i, row in enumerate(sequences) if row["oof_fold"] != outer]
        train_rows = [rows[i] for i in training]
        assignment = union_assignment([sequences[i] for i in seq_train], 3)
        record, supported, probabilities, rejected = _partition(
            f"outer-{outer}", seq_train, training, held, sequences, rows, geometry
        )
        agree(result["partitions"][4 * outer], record, name=f"outer-{outer}")
        expected["outer_context_supported"][held] = supported
        expected["outer_ood_abstained"][held] = rejected
        expected["outer_hierarchical_prior"][held] = probabilities
        lookup = {index: position for position, index in enumerate(training)}
        inside = {
            "raw_view_members": np.zeros((len(training), count, 2)),
            "context_row_indices": np.asarray(training, dtype=np.int64),
            "context_supported": np.zeros(len(training), dtype=bool),
            "ood_abstained": np.zeros(len(training), dtype=bool),
            "hierarchical_prior": np.zeros(len(training)),
            "effective_mixture_members": np.zeros((len(training), count)),
        }
        inner_parts = []
        for inner in range(3):
            current = [i for i in training if assignment[rows[i]["union_component_id"]] != inner]
            prediction = [i for i in training if assignment[rows[i]["union_component_id"]] == inner]
            seq_current = [
                i for i in seq_train if assignment[sequences[i]["union_component_id"]] != inner
            ]
            name = f"outer-{outer}-inner-{inner}"
            part, supported, probabilities, rejected = _partition(
                name, seq_current, current, prediction, sequences, rows, geometry
            )
            agree(result["partitions"][4 * outer + inner + 1], part, name=name)
            positions = [lookup[i] for i in prediction]
            inside["context_supported"][positions] = supported
            inside["ood_abstained"][positions] = rejected
            inside["hierarchical_prior"][positions] = probabilities
            inner_parts.append((inner, name, current, prediction, positions))
        eligible = inside["context_supported"] & ~inside["ood_abstained"]
        inside["calibration_mask"] = eligible
        groups = sorted({row["union_component_id"] for row in train_rows})
        outer_record = {
            "outer_fold": outer,
            "training_context_indices": training,
            "heldout_context_indices": held,
            "inner_assignment": assignment,
            "group_ids": groups,
            "draw_offset": group_offset,
            "draw_shape": [count, len(groups)],
            "inner_row_offset": inner_offset,
            "inner_row_count": len(training),
        }
        agree(result["outer_folds"][outer], outer_record, name="outer-ledger", exact=True)
        outer_ledgers.append(outer_record)
        group_offset += count * len(groups)
        inner_offset += len(training)
        for seed_index, seed in enumerate(seeds):
            for local in range(members):
                member = seed_index * members + local
                generator = np.random.Generator(
                    np.random.PCG64(np.random.SeedSequence([seed, outer, local]))
                )
                masses = generator.exponential(size=len(groups))
                actual_draws.append(masses)
                draw = {group: float(mass) for group, mass in zip(groups, masses, strict=True)}
                for inner, name, current, prediction, positions in inner_parts:
                    for view, label in enumerate(VIEWS):
                        identity = {
                            "outer_fold": outer,
                            "inner_fold": inner,
                            "partition_id": name,
                            "seed": seed,
                            "member_index": member,
                            "view": label,
                        }
                        inside["raw_view_members"][positions, member, view] = replay_model(
                            identity, current, prediction, draw
                        )
                probabilities = inside["raw_view_members"][:, member, :].mean(axis=1)
                inside["effective_mixture_members"][:, member] = np.where(
                    eligible, probabilities, inside["hierarchical_prior"]
                )
                chosen = [rows[i] for i, keep in zip(training, eligible, strict=True) if keep]
                chosen_ids = [i for i, keep in zip(training, eligible, strict=True) if keep]
                union_counts = [
                    len({row["union_component_id"] for row in chosen if row["label"] == label})
                    for label in (0, 1)
                ]
                group_count = len({row["union_component_id"] for row in chosen})
                fitted = group_count >= 10 and min(union_counts) >= 3
                calibration = result["calibrators"][calibrator_position]
                a, b = calibration["a"], calibration["b"]
                require(
                    type(a) is float and type(b) is float and 0.05 <= a <= 5 and -5 <= b <= 5,
                    "calibration coefficient bounds differ",
                )
                if fitted:
                    clipped = np.maximum(1e-6, np.minimum(probabilities[eligible], 1 - 1e-6))
                    logits = np.log(clipped) - np.log1p(-clipped)
                    score = a * logits + b
                    labels = np.asarray([row["label"] for row in chosen])
                    weights = row_weights(chosen, draw)
                    objective = float(
                        np.sum(weights * (np.logaddexp(0, score) - labels * score))
                        + 0.05 * ((a - 1) ** 2 + b**2)
                    )
                    gradient = np.asarray(
                        [
                            np.sum(weights * (sigmoid(score) - labels) * logits) + 0.1 * (a - 1),
                            np.sum(weights * (sigmoid(score) - labels)) + 0.1 * b,
                        ]
                    )
                    projected = np.asarray([a, b]) - np.clip(
                        np.asarray([a, b]) - gradient, [0.05, -5], [5, 5]
                    )
                    diagnostics["calibration_projected_gradient_max_absolute"] = max(
                        diagnostics["calibration_projected_gradient_max_absolute"],
                        float(np.max(np.abs(projected))),
                    )
                    diagnostics["calibration_optimizer_invocations"] += 1
                    require(
                        type(calibration["iterations"]) is int
                        and 1 <= calibration["iterations"] <= 1000,
                        "calibration iteration envelope differs",
                    )
                else:
                    require(
                        (a, b, calibration["iterations"]) == (1.0, 0.0, 0),
                        "identity calibration differs",
                    )
                    objective = None
                expected_calibration = {
                    "outer_fold": outer,
                    "seed": seed,
                    "member_index": member,
                    "calibration_context_indices": chosen_ids,
                    "a": a,
                    "b": b,
                    "fitted": fitted,
                    "eligible_unions": group_count,
                    "negative_unions": union_counts[0],
                    "positive_unions": union_counts[1],
                    "iterations": calibration["iterations"],
                    "objective": objective,
                }
                agree(calibration, expected_calibration, name=f"calibrator-{calibrator_position}")
                calibrator_position += 1
                for view, label in enumerate(VIEWS):
                    identity = {
                        "outer_fold": outer,
                        "inner_fold": None,
                        "partition_id": f"outer-{outer}",
                        "seed": seed,
                        "member_index": member,
                        "view": label,
                    }
                    expected["outer_raw_view_members"][held, member, view] = replay_model(
                        identity, training, held, draw
                    )
                outer_probability = expected["outer_raw_view_members"][held, member, :].mean(axis=1)
                if fitted:
                    clipped = np.clip(outer_probability, 1e-6, 1 - 1e-6)
                    calibrated = sigmoid(a * (np.log(clipped) - np.log1p(-clipped)) + b)
                else:
                    calibrated = outer_probability
                expected["outer_calibrated_unmasked_members"][held, member] = calibrated
        inner_results.append(inside)
    require(
        parameter_offset == len(arrays["model_parameters"]),
        "unreferenced or missing model parameters",
    )
    expected["bootstrap_group_draws"] = np.concatenate(actual_draws)
    agree(
        arrays["bootstrap_group_draws"],
        expected["bootstrap_group_draws"],
        name="bootstrap-draws",
        exact=True,
    )
    for key in inner_results[0]:
        expected["inner_" + key] = np.concatenate([part[key] for part in inner_results])
    raw = expected["outer_raw_view_members"].mean(axis=2)
    prior_matrix = np.repeat(expected["outer_hierarchical_prior"][:, None], count, axis=1)
    support = expected["outer_context_supported"]
    active = support & ~expected["outer_ood_abstained"]
    expected["predictor_prior"] = prior_matrix
    for name, values, keep in (
        ("raw", raw, support),
        ("raw_ood", raw, active),
        ("calibrated", expected["outer_calibrated_unmasked_members"], support),
        ("calibrated_ood", expected["outer_calibrated_unmasked_members"], active),
    ):
        expected["predictor_" + name] = np.where(keep[:, None], values, prior_matrix)
    covariance, noise, latent, cov_offset = [], np.zeros(len(rows)), np.zeros(len(rows)), 0
    for outer in outer_ledgers:
        indices = outer["heldout_context_indices"]
        values = expected["predictor_calibrated_ood"][indices]
        mean = values.mean(axis=1)
        # Sum member outer products explicitly; not the producer Gram operation.
        block = np.zeros((len(indices), len(indices)))
        for member in range(count):
            deviation = values[:, member] - mean
            block += np.outer(deviation, deviation) / count
        covariance.append(block.ravel())
        noise[indices] = np.sum(values * (1 - values), axis=1) / count
        latent[indices] = np.diag(block)
        require(
            np.allclose(latent[indices] + noise[indices], mean * (1 - mean), atol=1e-10, rtol=0),
            "independent total variance identity",
        )
        metadata = {
            "outer_fold": outer["outer_fold"],
            "row_indices": indices,
            "offset": cov_offset,
            "shape": list(block.shape),
            "finite_mixture_divisor": count,
            "unsupported_rows_are_scoring_placeholders_not_query_uncertainty": True,
        }
        agree(
            result["covariance_blocks"][outer["outer_fold"]],
            metadata,
            name="covariance-ledger",
            exact=True,
        )
        cov_offset += block.size
    expected["outer_covariance_blocks"] = np.concatenate(covariance)
    expected["outer_latent_probability_variance"] = latent
    expected["outer_bernoulli_noise_diagonal"] = noise
    require(
        set(expected) == ARRAYS - {"model_parameters"},
        "independent reconstruction inventory incomplete",
    )
    for name, value in expected.items():
        agree(arrays[name], value, name=name, exact=name == "bootstrap_group_draws")
    # Statistics use independently authenticated serialized probabilities. This
    # preserves exact histogram/bin and machine-zero decision identities while
    # all numeric statistics are recomputed using the independent path.
    report, sensitive = reconstruct_statistics(
        rows,
        {name: arrays["predictor_" + name] for name in PREDICTORS},
        active,
        pairs,
        seeds,
        members,
        result["metrics"],
    )
    report.update(
        {
            "fit_attempts": {"base": 40 * count, "calibration": 5 * count},
            "total_fit_attempts": 45 * count,
            "claims": CLAIMS,
        }
    )
    agree(result["metrics"], report, name="all-reconstructed-statistics")
    return {
        "independent_reconstruction_passed": True,
        "models_replayed": model_position,
        "calibrators_replayed": calibrator_position,
        "reconstructed_metrics": report,
        "optimization_diagnostics_not_new_acceptance_thresholds": diagnostics,
        "numerical_resolution_sensitive_endpoints": sensitive,
        "resolution_sensitive_gain_is_not_robust_scientific_evidence": bool(sensitive),
        "optimizer_invoked": False,
        "production_input_eligible": False,
        "teacher_moments_available_to_adaptive_search": False,
    }


def compare_runs(first, second):
    """Reproduction complements, never replaces, independent reconstruction."""
    require(set(first["arrays"]) == set(second["arrays"]) == ARRAYS, "reproduction array inventory")
    for key in ARRAYS:
        agree(
            second["arrays"][key],
            first["arrays"][key],
            name=key,
            coefficient=key == "model_parameters",
            exact=key == "bootstrap_group_draws",
        )
    iterations_equal = True
    for key in ("models", "calibrators"):
        require(len(first[key]) == len(second[key]), "reproduction fit count")
        for left, right in zip(first[key], second[key], strict=True):
            iterations_equal &= left["iterations"] == right["iterations"]
            agree(
                {name: value for name, value in right.items() if name != "iterations"},
                {name: value for name, value in left.items() if name != "iterations"},
                name=key,
            )
    for key in (
        "partitions",
        "outer_folds",
        "covariance_blocks",
        "pairs",
        "metrics",
        "seeds",
        "members_per_seed",
        "claims",
    ):
        agree(
            second[key],
            first[key],
            name=key,
            exact=key
            in {"outer_folds", "covariance_blocks", "pairs", "seeds", "members_per_seed", "claims"},
        )
    return {
        "reproduction_passed": True,
        "optimizer_iteration_counts_equal": iterations_equal,
        "probability_metric_atol": 1e-10,
        "probability_metric_rtol": 0.0,
        "coefficient_transform_atol": 1e-8,
        "coefficient_transform_rtol": 1e-8,
        "adaptive_tolerance_changes": False,
    }

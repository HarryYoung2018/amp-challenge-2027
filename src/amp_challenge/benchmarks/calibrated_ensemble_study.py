"""Nested controller-private ensemble fitting with complete compact ledgers.

The numerical routine supports bounded synthetic fixtures. The real-data runner
separately enforces the exact census, 40 members and 1,800-attempt declaration.
No final all-oracle fit, charged query interface or adaptive posterior is built.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from amp_challenge.benchmarks.calibrated_ensemble_contract import (
    SEEDS,
    VIEWS,
    private_teacher_claims,
    require,
)
from amp_challenge.benchmarks.calibrated_ensemble_evaluation import (
    PAIR_FIELDS,
    build_pairs,
    evaluate_study,
    finite_mixture_moments,
)
from amp_challenge.benchmarks.calibrated_ensemble_models import (
    fit_calibrator,
    fit_ood_geometry,
    fit_weighted_model,
)
from amp_challenge.benchmarks.oracle_activity_independent import assign_unions, prior_details


class ModelArchive:
    def __init__(self):
        self.chunks = []
        self.size = 0
        self.models = []

    def add(self, model, **identity):
        parameters = np.concatenate((model.mean, model.scale, model.coefficients))
        record = {
            **identity,
            "model_id": len(self.models),
            "parameter_offset": self.size,
            "feature_dimension": len(model.mean),
            "coefficient_count": len(model.coefficients),
            "targets": list(model.targets),
            "grams": list(model.grams),
            "constant_training_labels": model.constant_training_labels,
            "iterations": model.iterations,
            "objective": model.objective,
        }
        self.chunks.append(parameters)
        self.size += len(parameters)
        self.models.append(record)


def _partition(identity, sequence_indices, train_indices, test_indices, sequences, rows, geometry):
    training_sequences = [sequences[i] for i in sequence_indices]
    training_rows = [rows[i] for i in train_indices]
    test_rows = [rows[i] for i in test_indices]
    fitted = fit_ood_geometry(geometry[sequence_indices], training_sequences)
    sequence_index = {row["sequence_id"]: i for i, row in enumerate(sequences)}
    query_features = geometry[[sequence_index[row["sequence_id"]] for row in test_rows]]
    distances = fitted.distances(query_features)
    support_details = [prior_details(training_rows, row) for row in test_rows]
    supported = np.asarray([not detail["abstained"] for detail in support_details])
    priors = np.asarray([detail["diagnostic_prior_probability"] for detail in support_details])
    ood = distances > fitted.threshold
    return (
        {
            "identity": identity,
            "sequence_indices": sequence_indices,
            "training_context_indices": train_indices,
            "prediction_context_indices": test_indices,
            "ood_mean": fitted.mean.tolist(),
            "ood_scale": fitted.scale.tolist(),
            "ood_threshold": fitted.threshold,
            "ood_training_reference_distances": fitted.training_reference_distances.tolist(),
            "prediction_distances": distances.tolist(),
            "prediction_context_details": support_details,
            "prediction_ood_abstained": ood.tolist(),
        },
        supported,
        priors,
        ood,
    )


def fit_study(
    sequences, rows, matrices, geometry, *, seeds=SEEDS, members_per_seed=8, progress=None
):
    require(
        1 <= len(seeds) <= 5
        and len(set(seeds)) == len(seeds)
        and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds),
        "invalid bounded stochastic seeds",
    )
    require(type(members_per_seed) is int and 1 <= members_per_seed <= 8, "invalid member budget")
    require(1 <= len(sequences) <= 415 and 1 <= len(rows) <= 858, "study row budget exceeded")
    require(
        set(matrices) == set(VIEWS)
        and all(
            np.asarray(matrix).ndim == 2 and len(matrix) == len(rows) and np.isfinite(matrix).all()
            for matrix in matrices.values()
        ),
        "study feature inventory differs",
    )
    require(
        np.asarray(geometry).shape == (len(sequences), 321),
        "OOD geometry feature alignment differs",
    )
    matrices = {name: np.asarray(matrix, dtype=np.float64) for name, matrix in matrices.items()}
    geometry = np.asarray(geometry, dtype=np.float64)
    require(
        len({row["sequence_id"] for row in sequences}) == len(sequences)
        and len({row["example_id"] for row in rows}) == len(rows),
        "duplicate study input identities",
    )
    by_sequence = {row["sequence_id"]: row for row in sequences}
    require(
        all(row["namespace"] == "oracle" for row in sequences)
        and all(
            row["namespace"] == "oracle"
            and row["sequence_id"] in by_sequence
            and row["union_component_id"] == by_sequence[row["sequence_id"]]["union_component_id"]
            and row["oof_fold"] == by_sequence[row["sequence_id"]]["oof_fold"]
            for row in rows
        ),
        "oracle namespace or context grouping differs",
    )
    outer_assignment = assign_unions(sequences, 5)
    require(
        all(outer_assignment[row["union_component_id"]] == row["oof_fold"] for row in sequences),
        "frozen outer union assignment differs",
    )
    # Pair construction occurs before labels enter any fitted statistic.
    pairs = build_pairs(sequences, [{key: row[key] for key in PAIR_FIELDS} for row in rows])
    member_count = len(seeds) * members_per_seed
    outer_raw_views = np.zeros((len(rows), member_count, 2))
    outer_calibrated_unmasked = np.zeros((len(rows), member_count))
    outer_context_supported = np.zeros(len(rows), dtype=bool)
    outer_ood = np.zeros(len(rows), dtype=bool)
    outer_prior = np.zeros(len(rows))
    outer_seen = np.zeros(len(rows), dtype=bool)
    archive = ModelArchive()
    partitions, outer_ledgers, calibrators = [], [], []
    inner_chunks, inner_effective_chunks, inner_row_indices, inner_masks = [], [], [], []
    inner_prior_chunks, inner_context_chunks, inner_ood_chunks = [], [], []
    draws, draw_offset = [], 0
    attempts = {"base": 0, "calibration": 0}

    def attempt(stage, **identity):
        attempts[stage] += 1
        require(sum(attempts.values()) <= 1800, "fit attempt ceiling exceeded")
        if progress is not None:
            progress(
                {
                    "event": "fit_attempt_start",
                    "stage": stage,
                    "attempts": dict(attempts),
                    **identity,
                }
            )

    for outer in range(5):
        train_indices = [i for i, row in enumerate(rows) if row["oof_fold"] != outer]
        test_indices = [i for i, row in enumerate(rows) if row["oof_fold"] == outer]
        sequence_indices = [i for i, row in enumerate(sequences) if row["oof_fold"] != outer]
        require(bool(train_indices) and bool(test_indices), "empty outer activity partition")
        training_rows = [rows[i] for i in train_indices]
        training_sequences = [sequences[i] for i in sequence_indices]
        inner_assignment = assign_unions(training_sequences, 3)
        partition, supported, priors, ood = _partition(
            f"outer-{outer}",
            sequence_indices,
            train_indices,
            test_indices,
            sequences,
            rows,
            geometry,
        )
        partitions.append(partition)
        outer_context_supported[test_indices] = supported
        outer_prior[test_indices] = priors
        outer_ood[test_indices] = ood
        outer_seen[test_indices] = True
        row_positions = {index: position for position, index in enumerate(train_indices)}
        inner_supported = np.zeros(len(train_indices), dtype=bool)
        inner_ood = np.zeros(len(train_indices), dtype=bool)
        inner_prior = np.zeros(len(train_indices))
        inner_views = np.zeros((len(train_indices), member_count, 2))
        inner_parts = []
        for inner in range(3):
            fitted = [
                i for i in train_indices if inner_assignment[rows[i]["union_component_id"]] != inner
            ]
            held = [
                i for i in train_indices if inner_assignment[rows[i]["union_component_id"]] == inner
            ]
            fitted_sequences = [
                i
                for i in sequence_indices
                if inner_assignment[sequences[i]["union_component_id"]] != inner
            ]
            require(bool(fitted) and bool(held), "empty inner activity partition")
            part, support, prior, inner_rejected = _partition(
                f"outer-{outer}-inner-{inner}",
                fitted_sequences,
                fitted,
                held,
                sequences,
                rows,
                geometry,
            )
            partitions.append(part)
            positions = [row_positions[index] for index in held]
            inner_supported[positions] = support
            inner_prior[positions] = prior
            inner_ood[positions] = inner_rejected
            inner_parts.append((inner, fitted, held, positions, part["identity"]))
        eligible = inner_supported & ~inner_ood
        group_ids = sorted({row["union_component_id"] for row in training_rows})
        outer_ledgers.append(
            {
                "outer_fold": outer,
                "training_context_indices": train_indices,
                "heldout_context_indices": test_indices,
                "inner_assignment": inner_assignment,
                "group_ids": group_ids,
                "draw_offset": draw_offset,
                "draw_shape": [member_count, len(group_ids)],
                "inner_row_offset": sum(len(chunk) for chunk in inner_row_indices),
                "inner_row_count": len(train_indices),
            }
        )
        effective = np.zeros((len(train_indices), member_count))
        for seed_index, seed in enumerate(seeds):
            for local_member in range(members_per_seed):
                member = seed_index * members_per_seed + local_member
                random = np.random.default_rng(np.random.SeedSequence([seed, outer, local_member]))
                raw_weights = random.exponential(1.0, size=len(group_ids))
                require(
                    np.isfinite(raw_weights).all() and np.all(raw_weights > 0),
                    "bootstrap draw is invalid; no redraw allowed",
                )
                group_draw = dict(zip(group_ids, raw_weights, strict=True))
                draws.append(raw_weights)
                draw_offset += len(raw_weights)
                for inner, fitted, held, positions, part_id in inner_parts:
                    fitted_rows, held_rows = [rows[i] for i in fitted], [rows[i] for i in held]
                    for view_index, view in enumerate(VIEWS):
                        identity = {
                            "outer_fold": outer,
                            "inner_fold": inner,
                            "partition_id": part_id,
                            "seed": seed,
                            "member_index": member,
                            "view": view,
                        }
                        attempt("base", **identity)
                        model = fit_weighted_model(matrices[view][fitted], fitted_rows, group_draw)
                        archive.add(model, **identity)
                        inner_views[positions, member, view_index] = model.predict(
                            matrices[view][held], held_rows
                        )
                raw_inner_mixture = inner_views[:, member].mean(axis=1)
                effective[:, member] = np.where(eligible, raw_inner_mixture, inner_prior)
                attempt("calibration", outer_fold=outer, seed=seed, member_index=member)
                calibrator = fit_calibrator(
                    effective[:, member], training_rows, eligible, group_draw
                )
                calibrators.append(
                    {
                        "outer_fold": outer,
                        "seed": seed,
                        "member_index": member,
                        "calibration_context_indices": [
                            index
                            for index, keep in zip(train_indices, eligible, strict=True)
                            if keep
                        ],
                        **asdict(calibrator),
                    }
                )
                for view_index, view in enumerate(VIEWS):
                    identity = {
                        "outer_fold": outer,
                        "inner_fold": None,
                        "partition_id": f"outer-{outer}",
                        "seed": seed,
                        "member_index": member,
                        "view": view,
                    }
                    attempt("base", **identity)
                    model = fit_weighted_model(
                        matrices[view][train_indices], training_rows, group_draw
                    )
                    archive.add(model, **identity)
                    outer_raw_views[test_indices, member, view_index] = model.predict(
                        matrices[view][test_indices], [rows[i] for i in test_indices]
                    )
                outer_calibrated_unmasked[test_indices, member] = calibrator.predict(
                    outer_raw_views[test_indices, member].mean(axis=1)
                )
        inner_chunks.append(inner_views)
        inner_effective_chunks.append(effective)
        inner_row_indices.append(np.asarray(train_indices, dtype=np.int64))
        inner_masks.append(eligible)
        inner_prior_chunks.append(inner_prior)
        inner_context_chunks.append(inner_supported)
        inner_ood_chunks.append(inner_ood)
    require(outer_seen.all(), "outer prediction coverage incomplete")
    expected = 5 * member_count
    require(
        attempts == {"base": expected * 8, "calibration": expected},
        "nested fit-attempt accounting differs",
    )
    supported = outer_context_supported & ~outer_ood
    raw = outer_raw_views.mean(axis=2)
    prior_matrix = np.broadcast_to(outer_prior[:, None], raw.shape).copy()
    predictors = {
        "prior": prior_matrix,
        "raw": np.where(outer_context_supported[:, None], raw, prior_matrix),
        "raw_ood": np.where(supported[:, None], raw, prior_matrix),
        "calibrated": np.where(
            outer_context_supported[:, None], outer_calibrated_unmasked, prior_matrix
        ),
        "calibrated_ood": np.where(supported[:, None], outer_calibrated_unmasked, prior_matrix),
    }
    covariance_chunks, covariance_ledger = [], []
    noise, latent_variance = np.zeros(len(rows)), np.zeros(len(rows))
    covariance_offset = 0
    for outer in outer_ledgers:
        indices = outer["heldout_context_indices"]
        _, covariance, observed_noise = finite_mixture_moments(
            predictors["calibrated_ood"][indices]
        )
        covariance_chunks.append(covariance.reshape(-1))
        covariance_ledger.append(
            {
                "outer_fold": outer["outer_fold"],
                "row_indices": indices,
                "offset": covariance_offset,
                "shape": list(covariance.shape),
                "finite_mixture_divisor": member_count,
                "unsupported_rows_are_scoring_placeholders_not_query_uncertainty": True,
            }
        )
        covariance_offset += covariance.size
        noise[indices], latent_variance[indices] = observed_noise, np.diag(covariance)
    arrays = {
        "model_parameters": np.concatenate(archive.chunks),
        "bootstrap_group_draws": np.concatenate(draws),
        "outer_raw_view_members": outer_raw_views,
        "outer_calibrated_unmasked_members": outer_calibrated_unmasked,
        "outer_context_supported": outer_context_supported,
        "outer_ood_abstained": outer_ood,
        "outer_hierarchical_prior": outer_prior,
        "inner_raw_view_members": np.concatenate(inner_chunks),
        "inner_effective_mixture_members": np.concatenate(inner_effective_chunks),
        "inner_context_row_indices": np.concatenate(inner_row_indices),
        "inner_calibration_mask": np.concatenate(inner_masks),
        "inner_hierarchical_prior": np.concatenate(inner_prior_chunks),
        "inner_context_supported": np.concatenate(inner_context_chunks),
        "inner_ood_abstained": np.concatenate(inner_ood_chunks),
        "outer_covariance_blocks": np.concatenate(covariance_chunks),
        "outer_latent_probability_variance": latent_variance,
        "outer_bernoulli_noise_diagonal": noise,
        **{f"predictor_{name}": value for name, value in predictors.items()},
    }
    require(
        all(
            value.dtype in (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))
            and np.isfinite(value).all()
            for value in arrays.values()
        ),
        "invalid saved array dtype or finiteness",
    )
    report = evaluate_study(rows, predictors, supported, pairs, seeds, members_per_seed)
    report.update(
        {
            "fit_attempts": attempts,
            "total_fit_attempts": sum(attempts.values()),
            "claims": private_teacher_claims(),
        }
    )
    return {
        "arrays": arrays,
        "models": archive.models,
        "calibrators": calibrators,
        "partitions": partitions,
        "outer_folds": outer_ledgers,
        "covariance_blocks": covariance_ledger,
        "pairs": pairs,
        "metrics": report,
        "seeds": list(seeds),
        "members_per_seed": members_per_seed,
        "claims": private_teacher_claims(),
    }

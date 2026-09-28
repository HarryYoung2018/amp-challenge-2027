"""Independent NumPy/stdlib reconstruction primitives; no producer imports."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

CONFIG = "configs/benchmarks/oracle_calibrated_ensemble_v1.toml"
CONFIG_SHA = "67cec93d3ed7c431fbf5910e4bb2d5676165623caa54fd3b98617336f85c78c0"
PRODUCER_COMMIT = "69dca98b6eed722c5b31fe9acb19f93c95bd2860"
SEEDS = (310013, 310019, 310033, 310049, 310063)
VIEWS = ("descriptors_context", "esm_context")
PREDICTORS = ("prior", "raw", "raw_ood", "calibrated", "calibrated_ood")
PAIR_FIELDS = {
    "example_id",
    "sequence_id",
    "union_component_id",
    "oof_fold",
    "canonical_target",
    "gram",
}
TARGETS = (("escherichia_coli", "negative"), ("staphylococcus_aureus", "positive"))
CLAIMS = {
    "teacher_controller_private": True,
    "teacher_moments_available_to_adaptive_search": False,
    "generated_label_kind": "MODEL_PROXY",
    "generated_query_count_kind": "unique_surrogate_oracle_calls_not_real_measurements",
    "adaptive_training_inputs": "charged_revealed_records_and_explicitly_allowed_label_free_priors_only",
    "final_all_oracle_refit_authorized": False,
    "final_query_mapping_authorized": False,
    "production_input_eligible": False,
    "search_superiority_accepted": False,
    "learned_HC50_hard_constraint_allowed": False,
    "oracle_calls": 0,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def agree(actual, expected, *, name="value", coefficient=False, exact=False):
    """No adaptive tolerance; bool/int identities are never compared as floats."""
    if isinstance(expected, np.ndarray):
        actual = np.asarray(actual)
        require(
            actual.shape == expected.shape and actual.dtype == expected.dtype,
            f"{name}: array shape/dtype differs",
        )
        if exact or expected.dtype.kind in "biu":
            require(np.array_equal(actual, expected), f"{name}: exact array differs")
        else:
            require(
                np.isfinite(actual).all() and np.isfinite(expected).all(),
                f"{name}: nonfinite array",
            )
            require(
                np.allclose(
                    actual,
                    expected,
                    atol=1e-8 if coefficient else 1e-10,
                    rtol=1e-8 if coefficient else 0,
                ),
                f"{name}: numerical array differs",
            )
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{name}: keys differ")
        for key, value in expected.items():
            agree(
                actual[key],
                value,
                name=f"{name}.{key}",
                coefficient=coefficient or key in {"ood_mean", "ood_scale", "a", "b"},
                exact=exact,
            )
    elif isinstance(expected, list | tuple):
        require(
            isinstance(actual, list | tuple) and len(actual) == len(expected),
            f"{name}: sequence differs",
        )
        for index, value in enumerate(expected):
            agree(
                actual[index], value, name=f"{name}[{index}]", coefficient=coefficient, exact=exact
            )
    elif type(expected) is float:
        require(
            type(actual) is float and math.isfinite(actual) and math.isfinite(expected),
            f"{name}: invalid scalar",
        )
        tolerance = (1e-8 + 1e-8 * abs(expected)) if coefficient else 1e-10
        require(
            actual == expected if exact else abs(actual - expected) <= tolerance,
            f"{name}: numerical scalar differs",
        )
    else:
        require(
            type(actual) is type(expected) and actual == expected, f"{name}: exact identity differs"
        )


def union_assignment(sequences, fold_count):
    sizes = Counter(sequence["union_component_id"] for sequence in sequences)
    require(len(sizes) >= fold_count, "not enough whole unions")
    loads = [0] * fold_count
    result = {}
    for group in sorted(sizes, key=lambda group: (-sizes[group], group)):
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        result[group] = fold
        loads[fold] += sizes[group]
    return result


def row_weights(rows, draws=None):
    counts = Counter(row["union_component_id"] for row in rows)
    require(bool(counts), "empty weighted rows")
    if draws is None:
        draws = {key: 1.0 for key in counts}
    require(
        set(counts) <= set(draws)
        and all(math.isfinite(draws[key]) and draws[key] > 0 for key in counts),
        "invalid current group masses",
    )
    denominator = math.fsum(draws[key] for key in counts)
    return np.array(
        [
            draws[row["union_component_id"]] / denominator / counts[row["union_component_id"]]
            for row in rows
        ]
    )


def transform(features, weights):
    # Independently derived weighted population moments; no producer helper.
    mean = np.sum(weights[:, None] * features, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * np.square(features - mean), axis=0))
    constant = np.max(features, axis=0) == np.min(features, axis=0)
    scale[(scale == 0) | constant] = 1.0
    return mean, scale


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -values))


def prior(training, query):
    matching = [
        row
        for row in training
        if row["canonical_target"] == query["canonical_target"] and row["gram"] == query["gram"]
    ]
    count = len({row["union_component_id"] for row in matching})
    classes = sorted({row["label"] for row in matching})
    reason = None
    if len({row["label"] for row in training}) == 1:
        reason = "constant_training_labels"
    elif not matching:
        reason = "unseen_target_gram_context"
    elif count < 2:
        reason = "context_fewer_than_two_training_unions"
    elif len(classes) == 1:
        reason = "constant_context_labels"
    selected, level = matching, "target_gram"
    if not selected:
        selected, level = [row for row in training if row["gram"] == query["gram"]], "gram"
    if not selected:
        selected, level = training, "global"
    values = defaultdict(list)
    for row in selected:
        values[row["union_component_id"]].append(row["label"])
    probability = (
        math.fsum(math.fsum(labels) / len(labels) for labels in values.values()) + 0.5
    ) / (len(values) + 1)
    return {
        "abstained": reason is not None,
        "reason": reason,
        "fallback_prior_level": level if reason is not None else None,
        "training_context_unions": count,
        "training_context_classes": classes,
        "diagnostic_prior_probability": probability,
        "diagnostic_prior_level": level,
        "diagnostic_prior_training_unions": len(values),
    }


def distances(left, right):
    # Rowwise direct differences avoid dot-product cancellation and bound memory.
    return np.array(
        [np.sqrt(np.sum(np.square(right - row), axis=1) / left.shape[1]) for row in left]
    )


def ood_partition(train_x, train_sequences, query_x):
    weights = row_weights(train_sequences)
    mean, scale = transform(train_x, weights)
    standardized = (train_x - mean) / scale
    distance = distances(standardized, standardized)
    group = np.asarray([row["union_component_id"] for row in train_sequences])
    distance[group[:, None] == group[None, :]] = np.inf
    nearest = np.min(distance, axis=1)
    require(np.isfinite(nearest).all(), "missing other-union OOD support")
    ordered = sorted(range(len(nearest)), key=lambda index: (nearest[index], index))
    mass = 0.0
    chosen = ordered[-1]
    for index in ordered:
        mass += weights[index]
        if mass >= 0.95:
            chosen = index
            break
    threshold = float(nearest[chosen])
    query_distances = distances((query_x - mean) / scale, standardized).min(axis=1)
    return {
        "ood_mean": mean.tolist(),
        "ood_scale": scale.tolist(),
        "ood_threshold": threshold,
        "ood_training_reference_distances": nearest.tolist(),
        "prediction_distances": query_distances.tolist(),
        "prediction_ood_abstained": (query_distances > threshold).tolist(),
    }


def _hashed(*pieces):
    return hashlib.sha256("|".join(str(piece) for piece in pieces).encode()).hexdigest()


def pair_inventory(sequences, metadata):
    require(
        all(set(row) == PAIR_FIELDS for row in metadata), "pair builder cannot inspect outcomes"
    )
    require(len({row["example_id"] for row in metadata}) == len(metadata), "duplicate pair context")
    dominant = sorted(
        Counter(row["union_component_id"] for row in sequences).items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    consumed, primary, stress = set(), [], []

    def ordered_context(indices):
        return sorted(
            indices,
            key=lambda i: (
                _hashed("context-v1", metadata[i]["example_id"]),
                metadata[i]["example_id"],
            ),
        )

    def pair(left, right, target, gram):
        return {
            "outer_fold": metadata[left]["oof_fold"],
            "canonical_target": target,
            "gram": gram,
            "left_index": left,
            "right_index": right,
            "left_union": metadata[left]["union_component_id"],
            "right_union": metadata[right]["union_component_id"],
        }

    for outer in range(5):
        for target, gram in TARGETS:
            grouped = defaultdict(list)
            for index, row in enumerate(metadata):
                if (
                    row["oof_fold"] == outer
                    and row["canonical_target"] == target
                    and row["gram"] == gram
                    and row["union_component_id"] not in consumed
                ):
                    grouped[row["union_component_id"]].append(index)
            available = sorted(
                grouped, key=lambda uid: (_hashed("pair-v1", outer, target, uid), uid)
            )
            while len(available) >= 2:
                first, second, *available = available
                primary.append(
                    pair(
                        ordered_context(grouped[first])[0],
                        ordered_context(grouped[second])[0],
                        target,
                        gram,
                    )
                )
                consumed |= {first, second}
    for target, gram in TARGETS:
        available = ordered_context(
            [
                i
                for i, row in enumerate(metadata)
                if row["union_component_id"] == dominant
                and row["canonical_target"] == target
                and row["gram"] == gram
            ]
        )
        while len(available) >= 2 and len(stress) < 16:
            first, *available = available
            compatible = [
                i for i in available if metadata[i]["sequence_id"] != metadata[first]["sequence_id"]
            ]
            if not compatible:
                break
            second = compatible[0]
            available.remove(second)
            stress.append(pair(first, second, target, gram))
    return {
        "primary": primary,
        "dominant_stress": stress,
        "dominant_union": dominant,
        "outcomes_used_in_construction": False,
    }


def proper_scores(rows, probabilities):
    require(len(rows) == len(probabilities) and bool(rows), "score alignment")
    components = defaultdict(list)
    losses, errors = [], []
    for index, (row, value) in enumerate(zip(rows, probabilities, strict=True)):
        require(
            type(row["label"]) is int
            and row["label"] in (0, 1)
            and math.isfinite(value)
            and 0 <= value <= 1,
            "invalid score input",
        )
        p = max(1e-12, min(float(value), 1 - 1e-12))
        losses.append(-math.log(p) if row["label"] == 1 else -math.log1p(-p))
        errors.append((float(value) - row["label"]) ** 2)
        components[row["union_component_id"]].append(index)
    result = {}
    for name, values in (("log_loss", losses), ("brier", errors)):
        result["row_" + name] = math.fsum(values) / len(rows)
        result["component_" + name] = math.fsum(
            math.fsum(values[i] for i in indices) / len(indices) for indices in components.values()
        ) / len(components)
    return result


def pair_scores(pairs, rows, member_probabilities):
    joint, independent = [], []
    for pair in pairs:
        first, second = pair["left_index"], pair["right_index"]
        require(
            rows[first]["oof_fold"] == rows[second]["oof_fold"] == pair["outer_fold"],
            "cross-teacher pair",
        )
        left = [
            float(p) if rows[first]["label"] else 1 - float(p) for p in member_probabilities[first]
        ]
        right = [
            float(p) if rows[second]["label"] else 1 - float(p)
            for p in member_probabilities[second]
        ]
        size = len(left)
        event = math.fsum(a * b for a, b in zip(left, right, strict=True)) / size
        separate = (math.fsum(left) / size) * (math.fsum(right) / size)
        joint.append(-math.log(max(event, 1e-12)))
        independent.append(-math.log(max(separate, 1e-12)))
    return np.asarray(joint), np.asarray(independent)


def family_interval(values, seed):
    # Same predeclared random-index algorithm, separately implemented statistics.
    values = np.asarray(values)
    sampled = np.random.Generator(np.random.PCG64(seed)).integers(
        len(values), size=(2000, len(values))
    )
    means = sorted(
        math.fsum(float(values[index]) for index in sample) / len(values) for sample in sampled
    )

    def quantile(probability):
        position = probability * (len(means) - 1)
        lower = math.floor(position)
        return means[lower] + (position - lower) * (
            means[min(lower + 1, len(means) - 1)] - means[lower]
        )

    return [quantile(0.05 / 6), quantile(1 - 0.05 / 6)]

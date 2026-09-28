"""Independent numerical readback of the bounded TR2 continuation probe.

Shares native state/kernel primitives and structural node-plan reconstruction;
never invokes the producer, its KL routine, or its summary implementation.
Whole-tree PUCT authentication remains the existing tree verifier's job.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX, canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _seed,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_tr2_operator_kl import (
    MEASURE,
    PATHS,
    realized_node_plan,
    source_sha256,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(actual, expected, label):
    _require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
        "TR2 readback " + label + " differs",
    )


def verify_tr2_operator_record(units, generation, candidate, record, report):
    plan = realized_node_plan(units, generation)
    triple = record["triple"]
    by_triple = {unit.triple: unit for unit in units}
    _require(triple in by_triple, "TR2 readback student differs")
    unit = by_triple[triple]
    models = tuple(canonical_model_logical_hash(m) for m in (unit.model, candidate, unit.reference))
    _require(
        candidate.config == unit.model.config == unit.reference.config,
        "TR2 readback model architecture differs",
    )
    _require(
        tuple(record["models"]) == models and (models[0], models[2]) == plan["models"][triple],
        "TR2 readback model identities differ",
    )
    _require(
        record["source_sha256"] == source_sha256() and record["plan_sha256"] == _json_hash(plan),
        "TR2 readback source/plan differs",
    )
    _require(
        record["measure"] == MEASURE
        and record["scope"] == "frozen_realized_node_law_not_global_adaptive_search"
        and record["state_weighting"] == "uniform_path_then_uniform_active_suffix_states",
        "TR2 readback measure differs",
    )
    _require(
        type(record["candidate_index"]) is int and 0 <= record["candidate_index"] <= 8,
        "TR2 readback candidate index differs",
    )
    seed = int(
        _json_hash([plan["seed"], plan["round_index"], triple, "tr2-actual-continuation-kl-v1"])[
            :16
        ],
        16,
    )
    _require(
        record["seed"] == seed and len(record["paths"]) == PATHS,
        "TR2 readback path count/seed differs",
    )
    rows = plan["students"][triple]
    indices = [int(_seed(seed, i, "tr2-kl-node", 0).integers(len(rows))) for i in range(PATHS)]
    tokens = [np.array(rows[index]["tokens"], dtype=np.int64) for index in indices]
    terms, current_logp, reference_logp, active_terms = (
        [[] for _ in range(PATHS)] for _ in range(4)
    )
    for ordinal, (index, path) in enumerate(zip(indices, record["paths"], strict=True)):
        node = rows[index]
        _require(
            path["ordinal"] == ordinal
            and path["node_index"] == index
            and _json_hash(path["node"]) == _json_hash(node),
            "TR2 readback selected node differs",
        )
        _close(path["node_log_probability"], -math.log(len(rows)), "node probability")
        _require(len(path["steps"]) == node["start_level"], "TR2 readback suffix length differs")
    for level in range(max(rows[index]["start_level"] for index in indices), 0, -1):
        active = [i for i, index in enumerate(indices) if level <= rows[index]["start_level"]]
        states = tuple(
            NativeTransitionState(tokens[i], rows[indices[i]]["length"], level) for i in active
        )
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        reference = _validated_transition_kernels(unit.reference, states, NATIVE_ENDPOINT_DEFAULTS)
        for ordinal, p, q in zip(active, current, reference, strict=True):
            step = record["paths"][ordinal]["steps"][rows[indices[ordinal]]["start_level"] - level]
            _require(
                step["level"] == level
                and step["before_tokens_sha256"] == _json_hash(tokens[ordinal].tolist()),
                "TR2 readback before-state differs",
            )
            draw = p.sample(_seed(seed, ordinal, "tr2d2-commit-" + triple, level))
            _require(
                tuple(step["draw"]["positions"]) == draw.positions
                and tuple(step["draw"]["residues"]) == draw.residues,
                "TR2 readback sampled event differs",
            )
            _close(
                step["draw"]["log_probability"],
                draw.log_probability,
                "current transition probability",
            )
            reference_probability = q.log_probability(draw.positions, draw.residues)
            _close(
                step["reference_log_probability"],
                reference_probability,
                "reference transition probability",
            )
            _require(
                p.masked_positions == q.masked_positions and p.commit_count == q.commit_count,
                "TR2 readback subset support differs",
            )
            value = 0.0
            if p.commit_count:
                value = max(
                    0.0,
                    p.commit_count
                    / len(p.masked_positions)
                    * math.fsum(
                        float(a) * math.log(float(a) / float(b))
                        for a, b in zip(
                            p.residue_probabilities.flat, q.residue_probabilities.flat, strict=True
                        )
                    ),
                )
                active_terms[ordinal].append(value)
            _close(step["transition_kl"], value, "complete transition KL")
            terms[ordinal].append(value)
            current_logp[ordinal].append(draw.log_probability)
            reference_logp[ordinal].append(reference_probability)
            tokens[ordinal][list(draw.positions)] = draw.residues
            _require(
                step["after_tokens_sha256"] == _json_hash(tokens[ordinal].tolist()),
                "TR2 readback after-state differs",
            )
    values, weighted = [], []
    for ordinal, path in enumerate(record["paths"]):
        _require(
            not np.any(tokens[ordinal] == MASK_TOKEN_INDEX) and bool(active_terms[ordinal]),
            "TR2 readback incomplete continuation",
        )
        value = math.fsum(terms[ordinal])
        values.append(value)
        _close(path["conditional_kl"], value, "path KL")
        _close(
            path["current_log_probability"],
            math.fsum(current_logp[ordinal]),
            "path current probability",
        )
        _close(
            path["reference_log_probability"],
            math.fsum(reference_logp[ordinal]),
            "path reference probability",
        )
        weighted.extend(
            (term, 1 / (PATHS * len(active_terms[ordinal]))) for term in active_terms[ordinal]
        )
    mean = math.fsum(values) / PATHS
    mcse = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (PATHS * (PATHS - 1)))
    cumulative, p99 = 0.0, max(term for term, _ in weighted)
    for term, mass in sorted(weighted):
        cumulative += mass
        if cumulative >= 0.99:
            p99 = term
            break
    for label, value in (
        ("mean", mean),
        ("monte_carlo_standard_error", mcse),
        ("active_transition_p99", p99),
    ):
        _close(record[label], value, label)
        _close(getattr(report, label), value, "report " + label)
    expected_report = dict(
        source_sha256=record["source_sha256"],
        plan_sha256=record["plan_sha256"],
        receipt_sha256=_json_hash(record),
        old_model_sha256=models[0],
        candidate_model_sha256=models[1],
        reference_model_sha256=models[2],
        path_count=PATHS,
        scope="DECLARED_CONDITIONAL_OPERATOR_PATH_NOT_GLOBAL_SEARCH",
    )
    _require(
        all(asdict(report)[key] == value for key, value in expected_report.items()),
        "TR2 readback report identity differs",
    )
    _require(
        models
        == tuple(canonical_model_logical_hash(m) for m in (unit.model, candidate, unit.reference)),
        "TR2 readback mutated models",
    )
    return {
        "reconstructed": True,
        "path_count": PATHS,
        "mean": mean,
        "monte_carlo_standard_error": mcse,
        "active_transition_p99": p99,
        "scientific_evidence_accepted": False,
    }

"""Independent conditional node/draw/endpoint/predicate/bound/work reconstruction.

Shares existing native kernels and structural node-state reconstruction, never
the matched sampler, its node-choice function or its statistical implementation.
Original tree PUCT authentication and external timing remain separate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _seed,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_matched_feasibility_verify import (
    verify_matched_summary,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy
from amp_challenge.generators.diffusion.native_tr2_operator_kl import (
    realized_node_plan,
    source_sha256,
)


def _source():
    root = Path(__file__).resolve().parent
    return _json_hash(
        {
            "operator": source_sha256(),
            "component": {
                name: hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest()
                for name in (
                    "native_tr2_matched_feasibility",
                    "native_tr2_matched_feasibility_verify",
                    "native_matched_feasibility",
                    "native_matched_feasibility_verify",
                    "native_ga_partial_work",
                    "native_initialization",
                )
            },
        }
    )


def _require(condition, message):
    if not condition:
        raise ValueError("independent TR2 feasibility " + message)


def verify_tr2_matched_feasibility(
    record,
    *,
    units,
    proposals,
    generation,
    predicate,
    predicate_sha256,
    context_sha256,
    expected_source_sha256,
    expected_plan_sha256,
    check=lambda: None,
):
    check()
    _require(
        tuple(u.triple for u in units) == TRIPLES and len(proposals) == 10,
        "needs all ten checkpoints",
    )
    node_plan = realized_node_plan(units, generation)
    node_sha = _json_hash(node_plan)
    old_ids = tuple(canonical_model_logical_hash(u.model) for u in units)
    proposed_ids = tuple(canonical_model_logical_hash(m) for m in proposals)
    scope = "equal_checkpoint_uniform_realized_tr2_node_not_adaptive_search"
    draws = {}
    for triple in TRIPLES:
        seed = int(_json_hash([node_sha, triple, "tr2-matched-nodes-v1"])[:16], 16)
        draws[triple] = tuple(
            int(
                _seed(seed, i, "uniform-realized-expansion", 0).integers(
                    len(node_plan["students"][triple])
                )
            )
            for i in range(16)
        )
    expected = {
        "node_plan_sha256": node_sha,
        "old_models": old_ids,
        "common_proposed_models": proposed_ids,
        "predicate_sha256": predicate_sha256,
        "context_sha256": context_sha256,
        "source_sha256": expected_source_sha256,
        "draws": draws,
        "candidate_scales": tuple(2.0**-k for k in range(9)),
        "scope": scope,
    }
    _require(
        _source() == expected_source_sha256 == record["source_sha256"]
        and _json_hash(expected) == expected_plan_sha256 == record["plan_sha256"]
        and record["status"] == "complete"
        and len(record["students"]) == 10,
        "source, fixed plan or completeness differs",
    )
    index = record["candidate_index"]
    _require(type(index) is int and 0 <= index < 9, "candidate index differs")
    pairs, seen = [], {}
    rows, calls = 0, 0
    for unit, proposal, raw in zip(units, proposals, record["students"], strict=True):
        check()
        _require(
            getattr(predicate, "source_sha256", None) == predicate_sha256,
            "predicate source differs",
        )
        candidate = interpolate_policy(unit.model, proposal, index)
        candidate_id = canonical_model_logical_hash(candidate)
        chosen = draws[unit.triple]
        nodes = tuple(node_plan["students"][unit.triple][i] for i in chosen)
        seed = int(_json_hash([node_sha, unit.triple, "tr2-matched-paths-v1"])[:16], 16)
        _require(
            raw["triple"] == unit.triple
            and raw["old_model_sha256"] == unit.policy_sha256
            and raw["candidate_model_sha256"] == candidate_id
            and raw["path_seed"] == seed
            and tuple(raw["node_indices"]) == chosen
            and _json_hash(raw["nodes"]) == _json_hash(nodes)
            and raw["node_log_probability"] == -math.log(len(node_plan["students"][unit.triple])),
            "selected node, checkpoint, model or stream differs",
        )
        endpoints = []
        for model, key in ((unit.model, "old_paths"), (candidate, "candidate_paths")):
            paths = raw[key]
            _require(len(paths) == 16, "path count differs")
            tokens = [np.array(n["tokens"], dtype=np.int64) for n in nodes]
            for i, (node, path) in enumerate(zip(nodes, paths, strict=True)):
                _require(
                    path["ordinal"] == i and len(path["steps"]) == node["start_level"],
                    "path ordinal/length differs",
                )
            for level in range(max(n["start_level"] for n in nodes), 0, -1):
                check()
                active = [i for i, n in enumerate(nodes) if level <= n["start_level"]]
                states = tuple(
                    NativeTransitionState(tokens[i], nodes[i]["length"], level) for i in active
                )
                kernels = _validated_transition_kernels(model, states, NATIVE_ENDPOINT_DEFAULTS)
                # Reconstructed commit support determines which rows invoked the model.
                nonzero = sum(kernel.commit_count > 0 for kernel in kernels)
                rows += nonzero
                calls += (nonzero + 15) // 16
                for i, kernel in zip(active, kernels, strict=True):
                    step = paths[i]["steps"][nodes[i]["start_level"] - level]
                    _require(
                        step["level"] == level
                        and step["before_tokens_sha256"] == _json_hash(tokens[i].tolist()),
                        "before-state differs",
                    )
                    draw = kernel.sample(_seed(seed, i, "tr2d2-commit-" + unit.triple, level))
                    actual = step["draw"]
                    _require(
                        tuple(actual["positions"]) == draw.positions
                        and tuple(actual["residues"]) == draw.residues,
                        "conditional sampled event differs",
                    )
                    _require(
                        type(actual["log_probability"]) in (int, float)
                        and math.isclose(
                            actual["log_probability"],
                            draw.log_probability,
                            rel_tol=1e-9,
                            abs_tol=1e-10,
                        ),
                        "conditional event probability differs",
                    )
                    tokens[i][list(draw.positions)] = draw.residues
                    _require(
                        step["after_tokens_sha256"] == _json_hash(tokens[i].tolist()),
                        "after-state differs",
                    )
                check()
            for token_row, node, path in zip(tokens, nodes, paths, strict=True):
                _require(
                    all(0 <= int(r) < 20 for r in token_row[: node["length"]]),
                    "incomplete continuation",
                )
                endpoint = PeptideVocabulary().decode(np.array([token_row]))[0]
                _require(path["endpoint"] == endpoint, "endpoint differs")
                endpoints.append(endpoint)
        check()
        flags = predicate(tuple(endpoints))
        _require(
            type(flags) is tuple and len(flags) == 32 and all(type(v) is bool for v in flags),
            "public flags malformed",
        )
        _require(
            type(raw["flags"]) is list
            and all(type(v) is bool for v in raw["flags"])
            and raw["flags"] == list(flags),
            "public predicate replay differs",
        )
        check()
        _require(
            getattr(predicate, "source_sha256", None) == predicate_sha256
            and canonical_model_logical_hash(candidate) == candidate_id,
            "predicate/candidate changed",
        )
        for endpoint, flag in zip(endpoints, flags, strict=True):
            _require(
                endpoint not in seen or seen[endpoint] is flag, "repeated-sequence contradiction"
            )
            seen[endpoint] = flag
        pairs.extend(zip(flags[:16], flags[16:], strict=True))
    verify_matched_summary(record["summary"], tuple(pairs), expected_scope=scope)
    keys = ("row_forwards", "forward_calls", "backward_calls", "grad_enabled_forwards")
    delta = dict(row_forwards=rows, forward_calls=calls, backward_calls=0, grad_enabled_forwards=0)
    before, after = record["work_before"], record["work_after"]
    for work in (before, after):
        _require(
            work["shared_inner_subphase_split_measured"] is False
            and work["external_feature_model_work_counted_here"] is False,
            "work scope differs",
        )
        for phase in work["phases"].values():
            _require(
                set(phase) == set(keys) and all(type(v) is int and v >= 0 for v in phase.values()),
                "work counts differ",
            )
        _require(
            set(work["total"]) == set(keys)
            and all(type(v) is int for v in work["total"].values())
            and work["total"]
            == {key: sum(p[key] for p in work["phases"].values()) for key in keys},
            "work sums differ",
        )
    for phase in set(before["phases"]) | set(after["phases"]):
        observed = {
            key: after["phases"].get(phase, {}).get(key, 0)
            - before["phases"].get(phase, {}).get(key, 0)
            for key in keys
        }
        _require(
            observed
            == (delta if phase == "tr2_matched_feasibility_sampling" else dict.fromkeys(keys, 0)),
            "actual sampling work differs",
        )
    _require(
        rows <= 20480
        and calls <= 1280
        and after["total"] == {k: before["total"][k] + delta[k] for k in keys},
        "work ceiling differs",
    )
    check()
    _require(
        old_ids == tuple(canonical_model_logical_hash(u.model) for u in units)
        and proposed_ids == tuple(canonical_model_logical_hash(m) for m in proposals)
        and generation.sha256 == node_plan["generation_sha256"]
        and _source() == expected_source_sha256
        and getattr(predicate, "source_sha256", None) == predicate_sha256,
        "frozen source/model/generation changed",
    )
    return {
        "reconstructed": True,
        "paired_paths": 160,
        "native_paths": 320,
        "sampling_work_delta": delta,
        "passed": record["summary"]["passed"],
        "scope": scope,
        "original_tree_authenticated_here": False,
        "external_timing_authenticated": False,
        "scientific_evidence_accepted": False,
    }

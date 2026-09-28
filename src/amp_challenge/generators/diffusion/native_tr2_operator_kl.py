"""Current-policy conditional continuation KL at frozen realized TR2 tree nodes.

The node law is uniform over realized expansions within one student, not a
claim about future adaptive PUCT selection or endpoint-marginal search KL.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _seed,
    _state_contract,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import OperatorPathReport
from amp_challenge.generators.diffusion.native_tree_records import NativeTreeGeneration
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl

PATHS = 8
MEASURE = "uniform_realized_expansion_then_current_conditional_suffix"


def source_sha256():
    root = Path(__file__).resolve().parent
    return _json_hash(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "native_tr2_operator_kl.py",
                "native_tr2_operator_kl_verify.py",
                "native_tr2d2.py",
                "native_tree_records.py",
                "native_endpoint.py",
                "model.py",
                "categorical.py",
                "subset_kernel.py",
                "replay.py",
                "native_shared_endpoint.py",
            )
        }
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _prefix_tokens(unit, root, transitions):
    tokens = np.full(unit.model.config.max_length, PAD_TOKEN_INDEX, dtype=np.int64)
    tokens[: root.length] = MASK_TOKEN_INDEX
    for index, step in enumerate(transitions):
        level = unit.model.config.levels - index
        _require(
            step.level == level and step.before_tokens_sha256 == _json_hash(tokens.tolist()),
            "TR2 node prefix state/level differs",
        )
        positions, count = _state_contract(
            NativeTransitionState(tokens, root.length, level), unit.model.config
        )
        draw = step.draw
        _require(
            len(draw.positions) == len(draw.residues) == count
            and tuple(sorted(set(draw.positions))) == draw.positions
            and set(draw.positions) <= set(positions),
            "TR2 node prefix subset differs",
        )
        _require(
            all(type(r) is int and 0 <= r < 20 for r in draw.residues),
            "TR2 node prefix residue differs",
        )
        tokens[list(draw.positions)] = draw.residues
        _require(
            step.after_tokens_sha256 == _json_hash(tokens.tolist()),
            "TR2 node prefix after-state differs",
        )
    return tuple(map(int, tokens))


def realized_node_plan(units, generation):
    """Reconstruct realized node states without using posterior rewards.

    Existing native tree verification authenticates the original PUCT/sampling
    law separately. This verifies the state/prefix chain needed for the new
    conditional diagnostic; it is not a replacement whole-tree verifier.
    """
    _require(
        type(generation) is NativeTreeGeneration
        and generation.status in ("complete", "bounded_search_underfill"),
        "complete retained native generation required",
    )
    _require(type(units) is tuple and 1 <= len(units) <= 10, "bounded native students required")
    _require(
        1 <= len(generation.expansions) <= 100
        and len(generation.attempts) == 8 * len(generation.expansions),
        "TR2 node plan exceeds realized expansion budget",
    )
    by_triple = {unit.triple: unit for unit in units}
    roots = {row.triple: row for row in generation.roots}
    _require(
        len(by_triple) == len(roots) == len(units) and set(by_triple) == set(roots),
        "TR2 node-plan student inventory differs",
    )
    nodes, plans = {}, {triple: [] for triple in roots}
    for triple, unit in by_triple.items():
        unit.check()
        root = roots[triple]
        _require(
            (root.behavior_sha256, root.reference_sha256)
            == (unit.policy_sha256, unit.reference_sha256),
            "TR2 node plan behavior/reference differs",
        )
        nodes[triple] = {0: ()}
    seen, seen_attempts = set(), set()
    for expansion in generation.expansions:
        _require(
            expansion.triple in roots and expansion.expansion_index not in seen,
            "TR2 expansion identity differs",
        )
        seen.add(expansion.expansion_index)
        triple = expansion.triple
        unit, root = by_triple[triple], roots[triple]
        _require(
            expansion.selected_node in nodes[triple], "TR2 selected node has no retained prefix"
        )
        prefix = nodes[triple][expansion.selected_node]
        level = unit.model.config.levels - len(prefix)
        _require(
            level > 0 and len(expansion.attempt_indices) == 8,
            "TR2 expansion is not an active eight-child node",
        )
        plans[triple].append(
            {
                "expansion_index": expansion.expansion_index,
                "selected_node": expansion.selected_node,
                "length": root.length,
                "start_level": level,
                "tokens": _prefix_tokens(unit, root, prefix),
            }
        )
        for index in expansion.attempt_indices:
            _require(
                type(index) is int and 0 <= index < len(generation.attempts),
                "TR2 attempt index differs",
            )
            _require(index not in seen_attempts, "TR2 attempt reused across expansions")
            seen_attempts.add(index)
            attempt = generation.attempts[index]
            _require(
                attempt.attempt_index == index
                and attempt.expansion_index == expansion.expansion_index
                and attempt.child_node == len(nodes[triple]),
                "TR2 retained child identity differs",
            )
            _require(
                attempt.path.transitions[: len(prefix)] == prefix
                and len(prefix) < attempt.child_prefix_steps <= unit.model.config.levels,
                "TR2 child prefix differs",
            )
            child = attempt.path.transitions[: attempt.child_prefix_steps]
            _prefix_tokens(unit, root, child)
            # A child node ends at the first active edge after the selected prefix.
            _require(
                all(not step.draw.positions for step in child[len(prefix) : -1])
                and bool(child[-1].draw.positions),
                "TR2 child prefix active-edge boundary differs",
            )
            nodes[triple][attempt.child_node] = child
    _require(len(seen_attempts) == len(generation.attempts), "TR2 unconsumed attempt differs")
    _require(
        all(1 <= len(rows) <= 10 for rows in plans.values()), "TR2 realized expansion count differs"
    )
    return {
        "measure": MEASURE,
        "path_count": PATHS,
        "generation_sha256": generation.sha256,
        "round_index": generation.round_index,
        "seed": generation.seed,
        "students": plans,
        "models": {t: (u.policy_sha256, u.reference_sha256) for t, u in by_triple.items()},
    }


class TR2OperatorGuard:
    def __init__(self, units, generation, *, deadline, clock=time.monotonic):
        now = clock()
        _require(
            math.isfinite(deadline) and now < deadline <= now + 7200,
            "TR2 guard needs an original bounded deadline",
        )
        self.deadline, self.clock = deadline, clock
        self.plan = realized_node_plan(units, generation)
        self.plan_sha256, self.source_sha256 = _json_hash(self.plan), source_sha256()
        self.records, self.attempted, self.failed = [], set(), False
        self._check()

    def _check(self):
        _require(not self.failed, "TR2 operator guard failure cannot restart")
        _require(
            _json_hash(self.plan) == self.plan_sha256 and source_sha256() == self.source_sha256,
            "TR2 operator guard source/plan changed",
        )
        if self.clock() >= self.deadline:
            raise TimeoutError("TR2 operator guard original deadline exhausted")

    def evaluate(self, old, candidate, reference, *, triple, candidate_index):
        self._check()
        _require(
            triple in self.plan["students"]
            and type(candidate_index) is int
            and 0 <= candidate_index <= 8,
            "TR2 operator candidate identity differs",
        )
        key = (triple, candidate_index)
        _require(key not in self.attempted, "TR2 operator candidate cannot resample")
        models = tuple(canonical_model_logical_hash(m) for m in (old, candidate, reference))
        _require(
            (models[0], models[2]) == self.plan["models"][triple]
            and old.config == candidate.config == reference.config,
            "TR2 operator model bindings differ",
        )
        self.attempted.add(key)
        paths = []
        try:
            rows = self.plan["students"][triple]
            seed = int(
                _json_hash(
                    [
                        self.plan["seed"],
                        self.plan["round_index"],
                        triple,
                        "tr2-actual-continuation-kl-v1",
                    ]
                )[:16],
                16,
            )
            # Candidate index is omitted: all backtracks use the same parent/draw keys.
            selected = [
                int(_seed(seed, i, "tr2-kl-node", 0).integers(len(rows))) for i in range(PATHS)
            ]
            tokens = [
                np.asarray(rows[index]["tokens"], dtype=np.int64).copy() for index in selected
            ]
            for ordinal, index in enumerate(selected):
                paths.append(
                    {
                        "ordinal": ordinal,
                        "node_index": index,
                        "node": rows[index],
                        "node_log_probability": -math.log(len(rows)),
                        "steps": [],
                    }
                )
            for level in range(max(rows[index]["start_level"] for index in selected), 0, -1):
                self._check()
                active = [
                    i for i, index in enumerate(selected) if level <= rows[index]["start_level"]
                ]
                states = tuple(
                    NativeTransitionState(tokens[i], rows[selected[i]]["length"], level)
                    for i in active
                )
                current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
                frozen = _validated_transition_kernels(reference, states, NATIVE_ENDPOINT_DEFAULTS)
                for i, kernel, ref in zip(active, current, frozen, strict=True):
                    draw = kernel.sample(_seed(seed, i, "tr2d2-commit-" + triple, level))
                    before = _json_hash(tokens[i].tolist())
                    tokens[i][list(draw.positions)] = draw.residues
                    paths[i]["steps"].append(
                        {
                            "level": level,
                            "before_tokens_sha256": before,
                            "after_tokens_sha256": _json_hash(tokens[i].tolist()),
                            "draw": asdict(draw),
                            "reference_log_probability": ref.log_probability(
                                draw.positions, draw.residues
                            ),
                            "transition_kl": complete_subset_commit_kl(kernel, ref),
                        }
                    )
            values, weights, active_terms = [], [], []
            for path in paths:
                terms = [s["transition_kl"] for s in path["steps"]]
                active = [s["transition_kl"] for s in path["steps"] if s["draw"]["positions"]]
                _require(bool(active), "TR2 continuation has no active transitions")
                values.append(math.fsum(terms))
                active_terms.extend(active)
                weights.extend([1 / (PATHS * len(active))] * len(active))
                path["conditional_kl"] = values[-1]
                path["current_log_probability"] = math.fsum(
                    s["draw"]["log_probability"] for s in path["steps"]
                )
                path["reference_log_probability"] = math.fsum(
                    s["reference_log_probability"] for s in path["steps"]
                )
            mean = float(np.mean(values))
            mcse = float(np.std(values, ddof=1) / math.sqrt(PATHS))
            p99 = float(summarize_kl(active_terms, weights=weights).p99)
            record = {
                "source_sha256": self.source_sha256,
                "plan_sha256": self.plan_sha256,
                "triple": triple,
                "candidate_index": candidate_index,
                "models": models,
                "seed": seed,
                "paths": paths,
                "mean": mean,
                "monte_carlo_standard_error": mcse,
                "active_transition_p99": p99,
                "measure": MEASURE,
                "state_weighting": "uniform_path_then_uniform_active_suffix_states",
                "scope": "frozen_realized_node_law_not_global_adaptive_search",
            }
            receipt = _json_hash(record)
            _require(
                tuple(canonical_model_logical_hash(m) for m in (old, candidate, reference))
                == models,
                "TR2 guard mutated supplied models",
            )
            self.records.append(record)
            self._check()
            return OperatorPathReport(
                self.source_sha256, self.plan_sha256, receipt, *models, PATHS, mean, mcse, p99
            )
        except BaseException:
            self.failed = True
            raise

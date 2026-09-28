"""Paid matched continuations at frozen realized TR2 nodes; no model publication."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
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
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_matched_feasibility import (
    summarize_matched_feasibility,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy
from amp_challenge.generators.diffusion.native_tr2_operator_kl import (
    realized_node_plan,
    source_sha256,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string

SCOPE = "equal_checkpoint_uniform_realized_tr2_node_not_adaptive_search"
PHASE = "tr2_matched_feasibility_sampling"


def tr2_matched_source():
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


def tr2_matched_draws(node_plan):
    draws = {}
    for triple in TRIPLES:
        seed = int(_json_hash([_json_hash(node_plan), triple, "tr2-matched-nodes-v1"])[:16], 16)
        nodes = node_plan["students"][triple]
        draws[triple] = tuple(
            int(_seed(seed, i, "uniform-realized-expansion", 0).integers(len(nodes)))
            for i in range(16)
        )
    return draws


class TR2MatchedFeasibility:
    """Fixed ten-student, nine-scale family. Failure spends and closes the instance."""

    def __init__(
        self,
        units,
        proposals,
        generation,
        *,
        predicate,
        predicate_sha256,
        context_sha256,
        expected_source_sha256,
        counter,
        check,
        deadline_poll=None,
    ):
        if type(counter) is not NativeWorkCounter:
            raise TypeError("TR2 matched feasibility requires an exact active work counter")
        self.units, self.proposals = tuple(units), tuple(proposals)
        if tuple(u.triple for u in self.units) != TRIPLES or len(self.proposals) != 10:
            raise ValueError("TR2 matched feasibility requires all ten ordered checkpoints")
        counter.require_active(self.units)
        if any(not any(h is counter.hook for h in m._forward_hooks.values()) for m in proposals):
            raise ValueError("TR2 matched proposal is outside the active work counter")
        if (
            not callable(predicate)
            or not callable(check)
            or (deadline_poll is not None and not callable(deadline_poll))
            or not all(
                hash_string(v) for v in (predicate_sha256, context_sha256, expected_source_sha256)
            )
        ):
            raise ValueError("TR2 matched external requirement differs")
        self.generation = generation
        self.node_plan = realized_node_plan(self.units, generation)
        self.node_plan_sha256 = _json_hash(self.node_plan)
        self.predicate, self.predicate_sha256 = predicate, predicate_sha256
        self.context_sha256, self.source_sha256 = context_sha256, expected_source_sha256
        self.counter, self.check = counter, check
        self.deadline_poll = check if deadline_poll is None else deadline_poll
        self.old_hashes = tuple(u.policy_sha256 for u in self.units)
        self.proposal_hashes = tuple(canonical_model_logical_hash(m) for m in self.proposals)
        self.draws = tr2_matched_draws(self.node_plan)
        self.plan_sha256 = _json_hash(self.document())
        self.records, self._completed, self._seen = [], [], {}
        self._failed = False
        self._identity = (
            self.units,
            self.proposals,
            self.generation,
            self.predicate,
            counter,
            check,
            self.deadline_poll,
        )
        self._guard()

    def document(self):
        return {
            "node_plan_sha256": self.node_plan_sha256,
            "old_models": self.old_hashes,
            "common_proposed_models": self.proposal_hashes,
            "predicate_sha256": self.predicate_sha256,
            "context_sha256": self.context_sha256,
            "source_sha256": self.source_sha256,
            "draws": self.draws,
            "candidate_scales": tuple(2.0**-k for k in range(9)),
            "scope": SCOPE,
        }

    def _guard(self):
        self.check()
        self.counter.require_active(self.units)
        if (
            not all(
                a is b
                for a, b in zip(
                    self._identity,
                    (
                        self.units,
                        self.proposals,
                        self.generation,
                        self.predicate,
                        self.counter,
                        self.check,
                        self.deadline_poll,
                    ),
                    strict=True,
                )
            )
            or tr2_matched_source() != self.source_sha256
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
            or _json_hash(self.document()) != self.plan_sha256
            or _json_hash(self.node_plan) != self.node_plan_sha256
            or self.generation.sha256 != self.node_plan["generation_sha256"]
            or tuple(canonical_model_logical_hash(u.model) for u in self.units) != self.old_hashes
            or tuple(canonical_model_logical_hash(m) for m in self.proposals)
            != self.proposal_hashes
            or any(_json_hash(self.records[i]) != pin for i, pin in enumerate(self._completed))
            or any(
                not any(h is self.counter.hook for h in m._forward_hooks.values())
                for m in self.proposals
            )
        ):
            raise ValueError("TR2 matched fixed source/plan/predicate/model/record changed")
        self.check()

    def _sample(self, model, triple, seed, nodes, paths):
        tokens = [np.array(node["tokens"], dtype=np.int64) for node in nodes]
        paths.extend({"ordinal": i, "steps": [], "endpoint": None} for i in range(16))
        for level in range(max(n["start_level"] for n in nodes), 0, -1):
            self.deadline_poll()
            active = [i for i, n in enumerate(nodes) if level <= n["start_level"]]
            states = tuple(
                NativeTransitionState(tokens[i], nodes[i]["length"], level) for i in active
            )
            with self.counter.at(PHASE):
                kernels = _validated_transition_kernels(model, states, NATIVE_ENDPOINT_DEFAULTS)
            for i, kernel in zip(active, kernels, strict=True):
                draw = kernel.sample(_seed(seed, i, "tr2d2-commit-" + triple, level))
                before = _json_hash(tokens[i].tolist())
                tokens[i][list(draw.positions)] = draw.residues
                paths[i]["steps"].append(
                    {
                        "level": level,
                        "before_tokens_sha256": before,
                        "after_tokens_sha256": _json_hash(tokens[i].tolist()),
                        "draw": asdict(draw),
                    }
                )
            # Completed work/steps survive even if the original check expires here.
            self.deadline_poll()
        for i, node in enumerate(nodes):
            residues = tokens[i][: node["length"]]
            if any(not 0 <= int(r) < 20 for r in residues):
                raise ValueError("TR2 matched continuation did not complete")
            paths[i]["endpoint"] = PeptideVocabulary().decode(np.array([tokens[i]]))[0]

    def evaluate(self, *, candidate_index):
        if (
            self._failed
            or type(candidate_index) is not int
            or candidate_index != len(self.records)
            or not 0 <= candidate_index < 9
        ):
            raise ValueError("TR2 matched candidate cannot restart, skip or expand")
        record = {
            "candidate_index": candidate_index,
            "plan_sha256": self.plan_sha256,
            "source_sha256": self.source_sha256,
            "students": [],
            "summary": None,
            "status": "started",
            "work_before": self.counter.document(),
            "work_after": None,
        }
        self.records.append(record)
        try:
            self._guard()
            pairs = []
            for unit, proposal in zip(self.units, self.proposals, strict=True):
                self._guard()
                candidate = interpolate_policy(unit.model, proposal, candidate_index)
                candidate_sha = canonical_model_logical_hash(candidate)
                selected = self.draws[unit.triple]
                nodes = tuple(self.node_plan["students"][unit.triple][i] for i in selected)
                seed = int(
                    _json_hash([self.node_plan_sha256, unit.triple, "tr2-matched-paths-v1"])[:16],
                    16,
                )
                row = {
                    "triple": unit.triple,
                    "old_model_sha256": unit.policy_sha256,
                    "candidate_model_sha256": candidate_sha,
                    "path_seed": seed,
                    "node_indices": selected,
                    "nodes": nodes,
                    "node_log_probability": -math.log(len(self.node_plan["students"][unit.triple])),
                    "old_paths": [],
                    "candidate_paths": [],
                    "flags": None,
                }
                record["students"].append(row)
                for model, key in ((unit.model, "old_paths"), (candidate, "candidate_paths")):
                    self._sample(model, unit.triple, seed, nodes, row[key])
                    self._guard()
                sequences = tuple(
                    p["endpoint"] for key in ("old_paths", "candidate_paths") for p in row[key]
                )
                flags = self.predicate(sequences)
                if (
                    type(flags) is not tuple
                    or len(flags) != 32
                    or any(type(v) is not bool for v in flags)
                ):
                    raise ValueError("TR2 matched predicate must return exact paired Boolean flags")
                row["flags"] = list(flags)
                self._guard()
                if canonical_model_logical_hash(candidate) != candidate_sha:
                    raise ValueError("TR2 matched candidate changed during callback")
                for sequence, flag in zip(sequences, flags, strict=True):
                    if sequence in self._seen and self._seen[sequence] is not flag:
                        raise ValueError("TR2 matched repeated-sequence predicate contradiction")
                    self._seen[sequence] = flag
                pairs.extend(zip(flags[:16], flags[16:], strict=True))
            record["summary"] = summarize_matched_feasibility(tuple(pairs), scope=SCOPE)
            record["work_after"] = self.counter.document()
            before, after = record["work_before"]["total"], record["work_after"]["total"]
            if any(
                not 0 <= after[k] - before[k] <= cap
                for k, cap in (
                    ("row_forwards", 20480),
                    ("forward_calls", 1280),
                    ("backward_calls", 0),
                    ("grad_enabled_forwards", 0),
                )
            ):
                raise RuntimeError("TR2 matched native work ceiling exceeded")
            self._guard()
            record["status"] = "complete"
            if sum(len(str(r).encode()) for r in self.records) > 64 * 1024**2:
                raise ValueError("TR2 matched retained record budget exceeded")
            self._completed.append(_json_hash(record))
            self._guard()
            return record
        except BaseException as error:
            self._failed = True
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}"[:512],
                work_after=self.counter.document(),
            )
            raise

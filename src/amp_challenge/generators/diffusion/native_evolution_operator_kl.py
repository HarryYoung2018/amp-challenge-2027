"""Actual operator-conditional current-policy path KL, not global search KL."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    _seed,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_evolution_math import operator_plan
from amp_challenge.generators.diffusion.native_evolution_records import OPERATORS
from amp_challenge.generators.diffusion.native_proposals import (
    replay_native_trace,
    sample_native_proposals,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import OperatorPathReport
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl


class EvolutionOperatorGuard:
    """Eight paths/student, stratified uniformly over four operators.

    Parent law is the frozen realized two-stage allocation, not an assertion
    that future adaptive frontier choices follow the same marginal law.
    """

    def __init__(self, units, wave, *, seed: int, deadline):
        self.seed, self.deadline = seed, deadline
        self.round_index = wave.round_index
        self.corpora = {unit.triple: unit.sequences for unit in units}
        self.parents = {}
        for unit in units:
            masses = {}
            for allocation in wave.allocations:
                if allocation.triple == unit.triple:
                    for branch, count in zip(allocation.branches, allocation.counts, strict=True):
                        masses[branch.parent] = masses.get(branch.parent, 0) + count
            if sum(masses.values()) != 48:
                raise ValueError("operator guard requires complete48-attempt parent law")
            self.parents[unit.triple] = tuple(sorted(masses.items()))
        self.plan_sha256 = self._plan()
        self.source_sha256 = self._source()
        self.records = []

    def _plan(self):
        return _json_hash(
            [
                self.parents,
                self.corpora,
                self.seed,
                self.round_index,
                "uniform_four_operators_two_current_paths_each",
            ]
        )

    def mixture_summaries(self):
        """Equal ten-checkpoint stratified summaries, never pooled IID MCSE."""
        summaries = []
        for candidate_index in sorted({row["candidate_index"] for row in self.records}):
            rows = [row for row in self.records if row["candidate_index"] == candidate_index]
            if len({row["triple"] for row in rows}) != len(rows):
                raise ValueError("duplicate operator student diagnostic")
            base = {
                "candidate_index": candidate_index,
                "completed_checkpoints": len(rows),
                "receipt_sha256s": [_json_hash(row) for row in rows],
            }
            if set(row["triple"] for row in rows) != set(self.parents):
                summaries.append({**base, "status": "partial_not_equal_ten_mixture"})
                continue
            values, weights = [], []
            for row in rows:
                for path in row["paths"]:
                    active = [
                        term
                        for term, step in zip(
                            path["transition_kl"], path["trace"]["steps"], strict=True
                        )
                        if step["draw"]["positions"]
                    ]
                    values.extend(active)
                    weights.extend([1 / (80 * len(active))] * len(active))
            summaries.append(
                {
                    **base,
                    "status": "complete_equal_ten_mixture",
                    "mean": float(np.mean([row["mean"] for row in rows])),
                    "stratified_mcse": math.sqrt(
                        math.fsum(row["stratified_mcse"] ** 2 for row in rows)
                    )
                    / 10,
                    "active_transition_p99": summarize_kl(values, weights=weights).p99,
                    "scope": "uniform_checkpoint_operator_path_not_global_search",
                }
            )
        return summaries

    @staticmethod
    def _source():
        root = Path(__file__).resolve().parent
        return _json_hash(
            {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "native_evolution_operator_kl.py",
                    "native_evolution_math.py",
                    "native_proposals.py",
                    "native_endpoint.py",
                    "subset_kernel.py",
                )
            }
        )

    def evaluate(self, old, candidate, reference, *, triple, candidate_index):
        self.deadline.check("before_actual_operator_kl")
        if self._source() != self.source_sha256 or self._plan() != self.plan_sha256:
            raise ValueError("actual operator guard source changed")
        rows = self.parents[triple]
        probabilities = np.asarray([count / 48 for _, count in rows])
        unit = SimpleNamespace(model=candidate, sequences=self.corpora[triple])
        path_seed = int(
            _json_hash([self.seed, self.round_index, triple, "evolution-actual-operator-paths"])[
                :16
            ],
            16,
        )
        templates, levels, plans = [], [], []
        for op_index, operator in enumerate(OPERATORS):
            for replicate in range(2):
                ordinal = 2 * op_index + replicate
                rng = _seed(path_seed, ordinal, "evolution-operator-parent-" + triple, 0)
                index = int(rng.choice(len(rows), p=probabilities))
                parent = rows[index][0]
                template, level, length_logp = operator_plan(
                    unit, parent, operator, seed=path_seed, ordinal=ordinal
                )
                templates.append(template)
                levels.append(level)
                plans.append(
                    {
                        "operator": operator,
                        "parent": parent,
                        "parent_log_probability": math.log(probabilities[index]),
                        "length_log_probability": length_logp,
                        "operator_weight": 0.25,
                        "replicate": replicate,
                    }
                )
        traces = sample_native_proposals(
            candidate,
            tuple(templates),
            start_levels=tuple(levels),
            seed=path_seed,
            ordinals=tuple(range(8)),
        )
        paths, active_values, active_weights = [], [], []
        for plan, trace in zip(plans, traces, strict=True):
            current_logp, states = replay_native_trace(candidate, trace, authenticate_sampling=True)
            reference_logp, _ = replay_native_trace(reference, trace)
            current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
            frozen = _validated_transition_kernels(reference, states, NATIVE_ENDPOINT_DEFAULTS)
            terms = [
                complete_subset_commit_kl(left, right)
                for left, right in zip(current, frozen, strict=True)
            ]
            active = [
                term for term, kernel in zip(terms, current, strict=True) if kernel.commit_count
            ]
            if not active:
                raise ValueError("operator path has no active transition")
            active_values.extend(active)
            active_weights.extend([1 / (8 * len(active))] * len(active))
            shared = (
                plan["parent_log_probability"] + plan["length_log_probability"] + math.log(0.25)
            )
            paths.append(
                {
                    **plan,
                    "trace": asdict(trace),
                    "conditional_kl": math.fsum(terms),
                    "transition_kl": terms,
                    "current_augmented_log_probability": shared + current_logp,
                    "reference_augmented_log_probability": shared + reference_logp,
                    "sampled_log_ratio": current_logp - reference_logp,
                }
            )
        values = np.array([row["conditional_kl"] for row in paths]).reshape(4, 2)
        mean = float(values.mean())
        mcse = float(np.sqrt(np.sum(values.var(axis=1, ddof=1) / 2) / 16))
        p99 = summarize_kl(active_values, weights=active_weights).p99
        identities = tuple(
            canonical_model_logical_hash(model) for model in (old, candidate, reference)
        )
        record = {
            "plan_sha256": self.plan_sha256,
            "source_sha256": self.source_sha256,
            "triple": triple,
            "candidate_index": candidate_index,
            "models": identities,
            "paths": paths,
            "mean": mean,
            "stratified_mcse": mcse,
            "active_transition_p99": p99,
            "state_weighting": "uniform_operator_then_uniform_two_paths_then_uniform_active_states_within_path",
            "scope": "conditional_frozen_parent_operator_law_not_global_adaptive_search",
        }
        receipt = _json_hash(record)
        self.records.append(record)
        self.deadline.check("after_actual_operator_kl")
        if self._source() != self.source_sha256 or self._plan() != self.plan_sha256:
            raise ValueError("actual operator guard source changed")
        return OperatorPathReport(
            self.source_sha256, self.plan_sha256, receipt, *identities, 8, mean, mcse, float(p99)
        )

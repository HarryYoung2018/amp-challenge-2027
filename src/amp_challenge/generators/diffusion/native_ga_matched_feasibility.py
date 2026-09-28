"""Paid paired feasibility for the exact legal GA partial-native attempt law."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    GA_MATCHED_CONFIG_SHA256,
    PartialGuardPlan,
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_matched_feasibility import (
    summarize_matched_feasibility,
)
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string

SCOPE = "conditional_legal_ga_partial_native_attempt_not_selected_pool"


def ga_matched_source():
    root = Path(__file__).resolve().parent
    return _json_hash(
        {
            "partial_sources": source_identities(),
            "component": {
                name: hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest()
                for name in (
                    "native_ga_matched_feasibility",
                    "native_ga_matched_feasibility_verify",
                )
            },
        }
    )


def ga_matched_draws(plan):
    """IID legal ordinals; no stratification, quotas, deduplication or redraw."""
    draws = []
    for replicate in range(160):
        key = _json_hash([plan.numerical_sha256, replicate, "ga-matched-iid-legal-v1"])
        rng = np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16)))
        legal = plan.native_ordinal + int(rng.integers(plan.remaining_attempts))
        student = plan.checkpoint_order[legal % 10]
        parent_index = (legal - plan.native_ordinal) % 128
        draws.append(
            {
                "replicate": replicate,
                "legal_native_ordinal": legal,
                "triple": TRIPLES[student],
                "parent_index": parent_index,
                "parent": plan.parents[parent_index],
                "log_probability": -math.log(plan.remaining_attempts),
            }
        )
    return tuple(draws)


@dataclass(frozen=True)
class GAMatchedRequirement:
    predicate: object
    predicate_sha256: str
    context_sha256: str
    source_sha256: str

    def document(self):
        if (
            not callable(self.predicate)
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
            or not all(
                hash_string(v)
                for v in (self.predicate_sha256, self.context_sha256, self.source_sha256)
            )
            or ga_matched_source() != self.source_sha256
        ):
            raise ValueError("GA matched public requirement changed")
        return {
            "amendment_sha256": GA_MATCHED_CONFIG_SHA256,
            "predicate_sha256": self.predicate_sha256,
            "context_sha256": self.context_sha256,
            "source_sha256": self.source_sha256,
            "scope": SCOPE,
        }


class GAMatchedFeasibility:
    """Fixed common proposal and nine-scale plan; failed instances cannot retry."""

    def __init__(
        self,
        units,
        proposals,
        plan,
        *,
        predicate,
        predicate_sha256,
        context_sha256,
        expected_source_sha256,
        counter,
        check,
    ):
        if type(plan) is not PartialGuardPlan or type(counter) is not NativeWorkCounter:
            raise TypeError("GA matched feasibility needs exact plan and active work counter")
        plan.__post_init__()
        self.units, self.proposals = tuple(units), tuple(proposals)
        if tuple(unit.triple for unit in units) != TRIPLES or len(proposals) != 10:
            raise ValueError("GA matched feasibility requires all ten models")
        if (
            tuple(unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True))
            != plan.units
        ):
            raise ValueError("GA matched old units differ from public legal plan")
        if (
            not callable(predicate)
            or not callable(check)
            or not all(
                hash_string(v) for v in (predicate_sha256, context_sha256, expected_source_sha256)
            )
            or _json_hash(source_identities()) != plan.source_sha256
        ):
            raise ValueError("GA matched external requirement differs")
        counter.require_active(units)
        if any(
            not any(hook is counter.hook for hook in model._forward_hooks.values())
            for model in proposals
        ):
            raise ValueError("GA matched proposal is outside the active native work counter")
        self.plan, self.predicate = plan, predicate
        self.predicate_sha256, self.context_sha256 = predicate_sha256, context_sha256
        self.source_sha256, self.counter, self.check = expected_source_sha256, counter, check
        self.old_hashes = tuple(unit.policy_sha256 for unit in units)
        self.proposal_hashes = tuple(canonical_model_logical_hash(model) for model in proposals)
        self.draws = ga_matched_draws(plan)
        self.plan_sha256 = _json_hash(self.document())
        self.records, self._completed, self._seen = [], [], {}
        self._failed = False
        self._identity = (
            self.units,
            self.proposals,
            self.plan,
            self.predicate,
            self.counter,
            self.check,
        )
        self._guard()

    def document(self):
        return {
            "partial_plan_sha256": self.plan.sha256,
            "old_models": self.old_hashes,
            "common_proposed_models": self.proposal_hashes,
            "predicate_sha256": self.predicate_sha256,
            "context_sha256": self.context_sha256,
            "source_sha256": self.source_sha256,
            "draws": self.draws,
            "start_level": 32,
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
                        self.plan,
                        self.predicate,
                        self.counter,
                        self.check,
                    ),
                    strict=True,
                )
            )
            or ga_matched_source() != self.source_sha256
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
            or _json_hash(self.document()) != self.plan_sha256
            or tuple(canonical_model_logical_hash(unit.model) for unit in self.units)
            != self.old_hashes
            or tuple(canonical_model_logical_hash(model) for model in self.proposals)
            != self.proposal_hashes
            or any(_json_hash(self.records[i]) != pin for i, pin in enumerate(self._completed))
        ):
            raise ValueError("GA matched fixed source/plan/predicate/model/record changed")
        self.check()

    def evaluate(self, *, candidate_index):
        if (
            self._failed
            or type(candidate_index) is not int
            or candidate_index != len(self.records)
            or not 0 <= candidate_index < 9
        ):
            raise ValueError("GA matched candidate cannot restart, skip or expand")
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
            pairs = {}
            for unit, proposal in zip(self.units, self.proposals, strict=True):
                self._guard()
                candidate = interpolate_policy(unit.model, proposal, candidate_index)
                candidate_sha = canonical_model_logical_hash(candidate)
                draws = tuple(draw for draw in self.draws if draw["triple"] == unit.triple)
                path_seed = int(
                    _json_hash(
                        [self.plan.numerical_sha256, unit.triple, "ga-matched-common-path-v1"]
                    )[:16],
                    16,
                )
                row = {
                    "triple": unit.triple,
                    "old_model_sha256": unit.policy_sha256,
                    "candidate_model_sha256": candidate_sha,
                    "path_seed": path_seed,
                    "draws": draws,
                    "old_traces": [],
                    "candidate_traces": [],
                    "flags": None,
                }
                record["students"].append(row)
                paths = []
                for model, key in ((unit.model, "old_traces"), (candidate, "candidate_traces")):
                    traces = []
                    for start in range(0, len(draws), 128):
                        chunk = draws[start : start + 128]
                        with self.counter.at("ga_matched_feasibility_sampling"):
                            sampled = sample_native_proposals(
                                model,
                                tuple(draw["parent"] for draw in chunk),
                                start_levels=(32,) * len(chunk),
                                seed=path_seed,
                                ordinals=tuple(draw["replicate"] for draw in chunk),
                            )
                        traces.extend(sampled)
                        row[key].extend(asdict(trace) for trace in sampled)
                        self._guard()
                    paths.append(tuple(traces))
                sequences = tuple(trace.endpoint for traces in paths for trace in traces)
                flags = self.predicate(sequences) if sequences else ()
                if (
                    type(flags) is not tuple
                    or len(flags) != 2 * len(draws)
                    or any(type(v) is not bool for v in flags)
                ):
                    raise ValueError("GA matched predicate must return exact paired Boolean flags")
                row["flags"] = list(flags)
                self._guard()
                if canonical_model_logical_hash(candidate) != candidate_sha:
                    raise ValueError("GA matched candidate changed during predicate")
                for sequence, flag in zip(sequences, flags, strict=True):
                    if sequence in self._seen and self._seen[sequence] is not flag:
                        raise ValueError("GA matched repeated-sequence predicate contradiction")
                    self._seen[sequence] = flag
                n = len(draws)
                for draw, old, current in zip(draws, flags[:n], flags[n:], strict=True):
                    pairs[draw["replicate"]] = old, current
            record["summary"] = summarize_matched_feasibility(
                tuple(pairs[i] for i in range(160)), scope=SCOPE
            )
            record["work_after"] = self.counter.document()
            before, after = record["work_before"]["total"], record["work_after"]["total"]
            if any(
                after[key] - before[key] > cap
                for key, cap in (
                    ("row_forwards", 10240),
                    ("forward_calls", 1216),
                    ("backward_calls", 0),
                    ("grad_enabled_forwards", 0),
                )
            ):
                raise RuntimeError("GA matched complete native work ceiling exceeded")
            self._guard()
            record["status"] = "complete"
            if sum(len(str(row).encode()) for row in self.records) > 32 * 1024**2:
                raise ValueError("GA matched retained record budget exceeded")
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

"""Paid paired native feasibility probes under a frozen proposal family.

This component does not publish models or certify a public predicate. Atomic
update/outer admission is a separate consumer connection.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import _json_hash, _seed
from amp_challenge.generators.diffusion.native_evolution_math import operator_plan
from amp_challenge.generators.diffusion.native_evolution_records import OPERATORS
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string

SCOPE = "fixed_equal_checkpoint_operator_parent_mixture_not_global_search"
MAX_CANDIDATES = 9
PAIRS = 160
TAIL_LOG = math.log(2 * MAX_CANDIDATES / 0.05)
MAX_BYTES = 32 * 1024**2


def _binary_kl(q, p):
    if q == p:
        return 0.0
    if p in (0.0, 1.0):
        return math.inf
    if q == 0.0:
        return -math.log1p(-p)
    if q == 1.0:
        return -math.log(p)
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def _mean_bound(count, *, upper):
    q, threshold = count / PAIRS, TAIL_LOG / PAIRS
    if upper and count == PAIRS:
        return 1.0
    if not upper and count == 0:
        return 0.0
    lo, hi = (q, 1.0) if upper else (0.0, q)
    for _ in range(80):
        midpoint = (lo + hi) / 2
        if midpoint in (lo, hi):
            break
        inside = _binary_kl(q, midpoint) <= threshold
        if inside == upper:
            lo = midpoint
        else:
            hi = midpoint
    return (
        min(1.0, math.nextafter(hi, math.inf)) if upper else max(0.0, math.nextafter(lo, -math.inf))
    )


def summarize_matched_feasibility(pairs, *, scope=SCOPE):
    if scope not in (
        SCOPE,
        "conditional_legal_ga_partial_native_attempt_not_selected_pool",
        "equal_checkpoint_uniform_realized_tr2_node_not_adaptive_search",
    ):
        raise ValueError("undeclared matched feasibility measure")
    if (
        type(pairs) is not tuple
        or len(pairs) != PAIRS
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(flag) is not bool for flag in pair)
            for pair in pairs
        )
    ):
        raise ValueError("exact160 Boolean old/current pairs required")
    losses = sum(old and not current for old, current in pairs)
    gains = sum(current and not old for old, current in pairs)
    upper_loss, lower_gain = _mean_bound(losses, upper=True), _mean_bound(gains, upper=False)
    upper_drop = math.nextafter(upper_loss - lower_gain, math.inf)
    return {
        "pairs": PAIRS,
        "old_passes": sum(old for old, _ in pairs),
        "candidate_passes": sum(current for _, current in pairs),
        "losses": losses,
        "gains": gains,
        "empirical_drop": (losses - gains) / PAIRS,
        "loss_mean_upper": upper_loss,
        "gain_mean_lower": lower_gain,
        "drop_upper": upper_drop,
        "drop_limit": 0.05,
        "tail_error": 0.05 / (2 * MAX_CANDIDATES),
        "family_candidates": MAX_CANDIDATES,
        "confidence_scope": "nominal95_percent_simultaneous_per_frozen_nine_candidate_update",
        "scope": scope,
        "passed": upper_drop <= 0.05,
    }


def matched_feasibility_source():
    root = Path(__file__).resolve().parents[4]
    names = (
        "native_matched_feasibility",
        "native_matched_feasibility_verify",
        "native_evolution_records",
        "native_initialization",
        "native_baseline_operators",
        "native_proposals",
        "native_evolution_math",
        "native_endpoint",
        "native_shared_endpoint",
        "native_shared_endpoint_verify",
        "model",
        "categorical",
        "subset_kernel",
    )
    return _json_hash(
        {
            name: hashlib.sha256(
                (root / "src/amp_challenge/generators/diffusion" / (name + ".py")).read_bytes()
            ).hexdigest()
            for name in names
        }
    )


@dataclass(frozen=True)
class MatchedFeasibilityRequirement:
    """Caller-pinned public requirement, independent of the proposed update."""

    wave: object
    predicate: object
    predicate_sha256: str
    context_sha256: str
    source_sha256: str

    def document(self):
        self.wave.check()
        if (
            self.wave.status != "complete"
            or not callable(self.predicate)
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
            or not all(
                hash_string(value)
                for value in (self.predicate_sha256, self.context_sha256, self.source_sha256)
            )
            or matched_feasibility_source() != self.source_sha256
        ):
            raise ValueError("matched feasibility external requirement differs")
        return {
            "contract": "native_matched_feasibility_v1_20260914",
            "wave_sha256": self.wave.sha256,
            "predicate_sha256": self.predicate_sha256,
            "context_sha256": self.context_sha256,
            "source_sha256": self.source_sha256,
        }


class NativeMatchedFeasibility:
    """A single fixed nine-scale family; failures spend and close this instance."""

    def __init__(
        self, units, proposals, wave, *, predicate, predicate_sha256, context_sha256, seed, deadline
    ):
        wave.check()
        if tuple(unit.triple for unit in units) != TRIPLES or len(proposals) != 10:
            raise ValueError("matched feasibility requires all ten ordered checkpoints")
        if (
            wave.status != "complete"
            or not callable(predicate)
            or not all(hash_string(value) for value in (predicate_sha256, context_sha256))
            or type(seed) is not int
            or not 0 <= seed < 2**63
        ):
            raise ValueError("matched feasibility wave/predicate/context/seed differs")
        self.units, self.proposals = tuple(units), tuple(proposals)
        self.predicate, self.predicate_sha256 = predicate, predicate_sha256
        self.context_sha256, self.seed, self.deadline = context_sha256, seed, deadline
        self.parents = {}
        for unit in units:
            unit.check()
            masses = {}
            for allocation in wave.allocations:
                if allocation.triple == unit.triple:
                    for branch, count in zip(allocation.branches, allocation.counts, strict=True):
                        masses[branch.parent] = masses.get(branch.parent, 0) + count
            if sum(masses.values()) != 48:
                raise ValueError("matched feasibility needs full48-attempt parent law")
            self.parents[unit.triple] = tuple(sorted(masses.items()))
        self.old_hashes = tuple(unit.policy_sha256 for unit in units)
        self.proposal_hashes = tuple(canonical_model_logical_hash(model) for model in proposals)
        self.wave_sha256, self.round_index = wave.sha256, wave.round_index
        self.source_sha256 = matched_feasibility_source()
        self.plan_sha256 = _json_hash(self.plan())
        self.records, self._seen = [], {}
        self._completed_hashes = []
        self._failed = False
        self._identity = (self.predicate, self.deadline, self.units, self.proposals)
        self._guard()

    def plan(self):
        return {
            "wave_sha256": self.wave_sha256,
            "round_index": self.round_index,
            "parents": self.parents,
            "corpora": {unit.triple: unit.sequences for unit in self.units},
            "old_models": self.old_hashes,
            "common_proposed_models": self.proposal_hashes,
            "predicate_sha256": self.predicate_sha256,
            "context_sha256": self.context_sha256,
            "source_sha256": self.source_sha256,
            "seed": self.seed,
            "replicates_per_operator_checkpoint": 4,
            "candidate_scales": tuple(2.0**-k for k in range(9)),
            "scope": SCOPE,
        }

    def _guard(self):
        self.deadline.check("matched_feasibility_guard")
        if any(
            _json_hash(self.records[index]) != expected
            for index, expected in enumerate(self._completed_hashes)
        ):
            raise ValueError("completed matched feasibility record changed")
        if (
            not all(
                left is right
                for left, right in zip(
                    self._identity,
                    (self.predicate, self.deadline, self.units, self.proposals),
                    strict=True,
                )
            )
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
        ):
            raise ValueError("matched predicate identity/source changed")
        if (
            matched_feasibility_source() != self.source_sha256
            or _json_hash(self.plan()) != self.plan_sha256
        ):
            raise ValueError("matched feasibility source or fixed family changed")
        if (
            tuple(canonical_model_logical_hash(unit.model) for unit in self.units)
            != self.old_hashes
            or tuple(canonical_model_logical_hash(model) for model in self.proposals)
            != self.proposal_hashes
        ):
            raise ValueError("matched old/proposed model changed")
        self.deadline.check("matched_feasibility_guard_complete")

    def evaluate(self, *, candidate_index):
        from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy

        if (
            self._failed
            or type(candidate_index) is not int
            or candidate_index != len(self.records)
            or not 0 <= candidate_index < 9
        ):
            raise ValueError("matched candidate cannot restart, skip or expand its fixed family")
        record = {
            "plan_sha256": self.plan_sha256,
            "source_sha256": self.source_sha256,
            "candidate_index": candidate_index,
            "students": [],
            "status": "started",
            "summary": None,
        }
        self.records.append(record)
        try:
            self._guard()
            pairs = []
            for unit, proposal in zip(self.units, self.proposals, strict=True):
                self._guard()
                candidate = interpolate_policy(unit.model, proposal, candidate_index)
                candidate_hash = canonical_model_logical_hash(candidate)
                path_seed = int(
                    _json_hash(
                        [self.seed, self.round_index, unit.triple, "matched-public-feasibility-v1"]
                    )[:16],
                    16,
                )
                parents = self.parents[unit.triple]
                probabilities = np.asarray([mass / 48 for _, mass in parents])
                templates, levels, plans = [], [], []
                for operator_index, operator in enumerate(OPERATORS):
                    for replicate in range(4):
                        ordinal = operator_index * 4 + replicate
                        rng = _seed(
                            path_seed, ordinal, "matched-feasibility-parent-" + unit.triple, 0
                        )
                        parent = parents[int(rng.choice(len(parents), p=probabilities))][0]
                        template, level, length_logp = operator_plan(
                            SimpleNamespace(model=unit.model, sequences=unit.sequences),
                            parent,
                            operator,
                            seed=path_seed,
                            ordinal=ordinal,
                        )
                        templates.append(template)
                        levels.append(level)
                        plans.append(
                            {
                                "operator": operator,
                                "replicate": replicate,
                                "parent": parent,
                                "template": template,
                                "level": level,
                                "length_log_probability": length_logp,
                            }
                        )
                row = {
                    "triple": unit.triple,
                    "old_model_sha256": unit.policy_sha256,
                    "candidate_model_sha256": candidate_hash,
                    "path_seed": path_seed,
                    "plans": plans,
                    "old_traces": [],
                    "candidate_traces": [],
                    "flags": None,
                }
                record["students"].append(row)
                old = sample_native_proposals(
                    unit.model,
                    tuple(templates),
                    start_levels=tuple(levels),
                    seed=path_seed,
                    ordinals=tuple(range(16)),
                )
                row["old_traces"] = [asdict(trace) for trace in old]
                self._guard()
                current = sample_native_proposals(
                    candidate,
                    tuple(templates),
                    start_levels=tuple(levels),
                    seed=path_seed,
                    ordinals=tuple(range(16)),
                )
                row["candidate_traces"] = [asdict(trace) for trace in current]
                self._guard()
                sequences = tuple(trace.endpoint for trace in (*old, *current))
                flags = self.predicate(sequences)
                if (
                    type(flags) is not tuple
                    or len(flags) != 32
                    or any(type(flag) is not bool for flag in flags)
                ):
                    raise ValueError("matched predicate must return32 exact Boolean values")
                row["flags"] = list(flags)
                self._guard()
                if canonical_model_logical_hash(candidate) != candidate_hash:
                    raise ValueError("matched predicate changed candidate model")
                for sequence, flag in zip(sequences, flags, strict=True):
                    if sequence in self._seen and self._seen[sequence] is not flag:
                        raise ValueError("matched predicate changed repeated sequence feasibility")
                    self._seen[sequence] = flag
                pairs.extend(zip(flags[:16], flags[16:], strict=True))
            record["summary"] = summarize_matched_feasibility(tuple(pairs))
            self._guard()
            record["status"] = "complete"
            # Keep the same bounded failure reserve across every candidate.
            from amp_challenge.representations.run_feature_cache_records import canonical

            if len(canonical(self.records)) > MAX_BYTES:
                raise ValueError("matched feasibility retained record budget exceeded")
            self._completed_hashes.append(_json_hash(record))
            self._guard()
            return record
        except BaseException as error:
            self._failed = True
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"[:512]
            raise

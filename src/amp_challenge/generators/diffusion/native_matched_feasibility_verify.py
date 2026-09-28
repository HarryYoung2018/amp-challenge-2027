"""Independent paired-plan/trace/predicate reconstruction, not public-data admission.

Uses existing native operator planning, interpolation and trace scorers. It does
not call the new probe producer or its statistical summary function.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import _json_hash, _seed
from amp_challenge.generators.diffusion.native_evolution_math import operator_plan
from amp_challenge.generators.diffusion.native_evolution_records import OPERATORS
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import _trace


def _current_source():
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


def verify_matched_summary(
    summary,
    pairs,
    *,
    expected_scope="fixed_equal_checkpoint_operator_parent_mixture_not_global_search",
):
    if expected_scope not in (
        "fixed_equal_checkpoint_operator_parent_mixture_not_global_search",
        "conditional_legal_ga_partial_native_attempt_not_selected_pool",
        "equal_checkpoint_uniform_realized_tr2_node_not_adaptive_search",
    ):
        raise ValueError("independent undeclared matched feasibility measure")
    if len(pairs) != 160 or any(type(v) is not bool for pair in pairs for v in pair):
        raise ValueError("independent feasibility needs160 Boolean pairs")
    losses = sum(a and not b for a, b in pairs)
    gains = sum(b and not a for a, b in pairs)

    def bound(k, upper):
        q = k / 160
        if (upper and k == 160) or (not upper and k == 0):
            return q
        threshold = math.log(360) / 160
        left, right = (q, 1.0) if upper else (0.0, q)
        for _ in range(100):
            p = (left + right) * 0.5
            if p in (left, right):
                break
            terms = (
                -math.log1p(-p)
                if k == 0
                else -math.log(p)
                if k == 160
                else q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
            )
            if (terms > threshold) == upper:
                right = p
            else:
                left = p
        return right if upper else left

    upper_loss, lower_gain = bound(losses, True), bound(gains, False)
    for key, value in {
        "empirical_drop": (losses - gains) / 160,
        "loss_mean_upper": upper_loss,
        "gain_mean_lower": lower_gain,
        "drop_upper": upper_loss - lower_gain,
    }.items():
        if not math.isclose(summary[key], value, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("independent feasibility probability bound differs")
    expected = {
        "pairs": 160,
        "old_passes": sum(a for a, _ in pairs),
        "candidate_passes": sum(b for _, b in pairs),
        "losses": losses,
        "gains": gains,
        "drop_limit": 0.05,
        "tail_error": 0.05 / 18,
        "family_candidates": 9,
        "confidence_scope": "nominal95_percent_simultaneous_per_frozen_nine_candidate_update",
        "scope": expected_scope,
        "passed": upper_loss - lower_gain <= 0.05,
    }
    if set(summary) != set(expected) | {
        "empirical_drop",
        "loss_mean_upper",
        "gain_mean_lower",
        "drop_upper",
    } or any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("independent feasibility counts/scope/decision differ")
    return True


def verify_native_matched_feasibility(
    record,
    *,
    units,
    proposals,
    wave,
    predicate,
    predicate_sha256,
    context_sha256,
    seed,
    expected_plan_sha256,
    expected_source_sha256,
    check,
):
    wave.check()
    check()
    if _current_source() != expected_source_sha256:
        raise ValueError("independent feasibility actual source differs")
    if (
        tuple(unit.triple for unit in units) != TRIPLES
        or len(proposals) != 10
        or record["status"] != "complete"
    ):
        raise ValueError("independent feasibility requires complete ten-checkpoint record")
    parents = {}
    for unit in units:
        masses = {}
        for allocation in wave.allocations:
            if allocation.triple == unit.triple:
                for branch, count in zip(allocation.branches, allocation.counts, strict=True):
                    masses[branch.parent] = masses.get(branch.parent, 0) + count
        if sum(masses.values()) != 48:
            raise ValueError("independent feasibility parent mass differs")
        parents[unit.triple] = tuple(sorted(masses.items()))
    identities = tuple(canonical_model_logical_hash(unit.model) for unit in units)
    proposal_ids = tuple(canonical_model_logical_hash(model) for model in proposals)
    plan = {
        "wave_sha256": wave.sha256,
        "round_index": wave.round_index,
        "parents": parents,
        "corpora": {unit.triple: unit.sequences for unit in units},
        "old_models": identities,
        "common_proposed_models": proposal_ids,
        "predicate_sha256": predicate_sha256,
        "context_sha256": context_sha256,
        "source_sha256": expected_source_sha256,
        "seed": seed,
        "replicates_per_operator_checkpoint": 4,
        "candidate_scales": tuple(2.0**-k for k in range(9)),
        "scope": "fixed_equal_checkpoint_operator_parent_mixture_not_global_search",
    }
    if (
        _json_hash(plan) != expected_plan_sha256
        or record["plan_sha256"] != expected_plan_sha256
        or record["source_sha256"] != expected_source_sha256
    ):
        raise ValueError("independent feasibility externally pinned plan/source differs")
    index = record["candidate_index"]
    if type(index) is not int or not 0 <= index < 9 or len(record["students"]) != 10:
        raise ValueError("independent feasibility family/record length differs")
    seen, pairs = {}, []
    for unit, proposal, row in zip(units, proposals, record["students"], strict=True):
        check()
        if getattr(predicate, "source_sha256", None) != predicate_sha256:
            raise ValueError("independent public predicate source differs")
        candidate = interpolate_policy(unit.model, proposal, index)
        candidate_sha = canonical_model_logical_hash(candidate)
        path_seed = int(
            _json_hash([seed, wave.round_index, unit.triple, "matched-public-feasibility-v1"])[:16],
            16,
        )
        if (
            row["triple"],
            row["old_model_sha256"],
            row["candidate_model_sha256"],
            row["path_seed"],
        ) != (unit.triple, unit.policy_sha256, candidate_sha, path_seed):
            raise ValueError("independent feasibility model/seed differs")
        if any(len(row[name]) != 16 for name in ("plans", "old_traces", "candidate_traces")):
            raise ValueError("independent feasibility path census differs")
        old_traces, current_traces = (
            tuple(map(_trace, row["old_traces"])),
            tuple(map(_trace, row["candidate_traces"])),
        )
        for ordinal, (raw_plan, old, current) in enumerate(
            zip(row["plans"], old_traces, current_traces, strict=True)
        ):
            operator, replicate = OPERATORS[ordinal // 4], ordinal % 4
            law = parents[unit.triple]
            rng = _seed(path_seed, ordinal, "matched-feasibility-parent-" + unit.triple, 0)
            parent = law[int(rng.choice(len(law), p=np.asarray([count / 48 for _, count in law])))][
                0
            ]
            template, level, logp = operator_plan(
                SimpleNamespace(model=unit.model, sequences=unit.sequences),
                parent,
                operator,
                seed=path_seed,
                ordinal=ordinal,
            )
            expected = {
                "operator": operator,
                "replicate": replicate,
                "parent": parent,
                "template": template,
                "level": level,
                "length_log_probability": logp,
            }
            if raw_plan != expected:
                raise ValueError("independent feasibility paired template/length law differs")
            for model, trace in ((unit.model, old), (candidate, current)):
                if (trace.parent, trace.start_level, trace.seed, trace.ordinal) != (
                    template,
                    level,
                    path_seed,
                    ordinal,
                ):
                    raise ValueError("independent feasibility matched random stream differs")
                replay_native_trace(model, trace, authenticate_sampling=True)
            check()
        sequences = tuple(trace.endpoint for trace in (*old_traces, *current_traces))
        flags = predicate(sequences)
        check()
        if (
            type(flags) is not tuple
            or len(flags) != 32
            or any(type(flag) is not bool for flag in flags)
            or list(flags) != row["flags"]
        ):
            raise ValueError("independent feasibility predicate flags differ")
        if (
            getattr(predicate, "source_sha256", None) != predicate_sha256
            or canonical_model_logical_hash(candidate) != candidate_sha
        ):
            raise ValueError("independent feasibility predicate/source/model drift")
        for sequence, flag in zip(sequences, flags, strict=True):
            if sequence in seen and seen[sequence] is not flag:
                raise ValueError("independent feasibility repeated sequence differs")
            seen[sequence] = flag
        pairs.extend(zip(flags[:16], flags[16:], strict=True))
    if identities != tuple(
        canonical_model_logical_hash(unit.model) for unit in units
    ) or proposal_ids != tuple(canonical_model_logical_hash(model) for model in proposals):
        raise ValueError("independent feasibility frozen family changed")
    if _current_source() != expected_source_sha256:
        raise ValueError("independent feasibility actual source changed")
    verify_matched_summary(record["summary"], pairs)
    check()
    return {
        "reconstructed": True,
        "paired_paths": 160,
        "native_paths": 320,
        "passed": record["summary"]["passed"],
        "scientific_evidence_accepted": False,
    }

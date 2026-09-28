"""Independent legal-ordinal, native-path, predicate, bound and work replay."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_ga_partial_verify import _sampling_cost
from amp_challenge.generators.diffusion.native_matched_feasibility_verify import (
    verify_matched_summary,
)
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_shared_endpoint import interpolate_policy
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import _trace


def _source():
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


def verify_ga_matched_feasibility(
    record,
    *,
    units,
    proposals,
    plan,
    predicate,
    predicate_sha256,
    context_sha256,
    expected_source_sha256,
    expected_plan_sha256,
    check=lambda: None,
):
    """Only complete candidates; failed-prefix timing/work authority is external."""
    check()
    plan.__post_init__()
    if (
        tuple(unit.triple for unit in units) != TRIPLES
        or len(proposals) != 10
        or tuple(unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True))
        != plan.units
        or _source() != expected_source_sha256
        or _json_hash(source_identities()) != plan.source_sha256
        or record["source_sha256"] != expected_source_sha256
        or record["status"] != "complete"
        or len(record["students"]) != 10
    ):
        raise ValueError("independent GA feasibility source/old-state/completeness differs")
    old_ids = tuple(canonical_model_logical_hash(unit.model) for unit in units)
    proposed_ids = tuple(canonical_model_logical_hash(model) for model in proposals)
    draws = []
    for i in range(160):
        material = _json_hash([plan.numerical_sha256, i, "ga-matched-iid-legal-v1"])
        random = np.random.Generator(np.random.PCG64DXSM(int(material[:32], 16)))
        legal = plan.native_ordinal + int(random.integers(plan.remaining_attempts))
        parent = (legal - plan.native_ordinal) % 128
        draws.append(
            {
                "replicate": i,
                "legal_native_ordinal": legal,
                "triple": TRIPLES[plan.checkpoint_order[legal % 10]],
                "parent_index": parent,
                "parent": plan.parents[parent],
                "log_probability": -math.log(plan.remaining_attempts),
            }
        )
    scope = "conditional_legal_ga_partial_native_attempt_not_selected_pool"
    expected = {
        "partial_plan_sha256": plan.sha256,
        "old_models": old_ids,
        "common_proposed_models": proposed_ids,
        "predicate_sha256": predicate_sha256,
        "context_sha256": context_sha256,
        "source_sha256": expected_source_sha256,
        "draws": tuple(draws),
        "start_level": 32,
        "candidate_scales": tuple(2.0**-k for k in range(9)),
        "scope": scope,
    }
    if (
        _json_hash(expected) != expected_plan_sha256
        or record["plan_sha256"] != expected_plan_sha256
    ):
        raise ValueError("independent GA feasibility fixed family/plan differs")
    index = record["candidate_index"]
    if type(index) is not int or not 0 <= index < 9:
        raise ValueError("independent GA feasibility candidate index differs")
    pairs, seen = {}, {}
    rows, calls = 0, 0
    for unit, proposal, raw in zip(units, proposals, record["students"], strict=True):
        check()
        if getattr(predicate, "source_sha256", None) != predicate_sha256:
            raise ValueError("independent GA feasibility predicate source differs")
        selected = [draw for draw in draws if draw["triple"] == unit.triple]
        candidate = interpolate_policy(unit.model, proposal, index)
        candidate_sha = canonical_model_logical_hash(candidate)
        seed = int(
            _json_hash([plan.numerical_sha256, unit.triple, "ga-matched-common-path-v1"])[:16], 16
        )
        if (
            raw["triple"] != unit.triple
            or raw["old_model_sha256"] != unit.policy_sha256
            or raw["candidate_model_sha256"] != candidate_sha
            or raw["path_seed"] != seed
            or list(raw["draws"]) != selected
        ):
            raise ValueError(
                "independent GA feasibility legal-ordinal/parent/checkpoint law differs"
            )
        paths = []
        for model, key in ((unit.model, "old_traces"), (candidate, "candidate_traces")):
            if len(raw[key]) != len(selected):
                raise ValueError("independent GA feasibility missing path")
            traces = tuple(_trace(value) for value in raw[key])
            for draw, trace in zip(selected, traces, strict=True):
                if (trace.parent, trace.start_level, trace.seed, trace.ordinal) != (
                    draw["parent"],
                    32,
                    seed,
                    draw["replicate"],
                ):
                    raise ValueError("independent GA feasibility matched path stream differs")
                replay_native_trace(model, trace, authenticate_sampling=True)
                check()
            for start in range(0, len(selected), 128):
                n, m = _sampling_cost(raw[key][start : start + 128])
                rows += n
                calls += m
            paths.append(traces)
        seqs = tuple(trace.endpoint for traces in paths for trace in traces)
        flags = predicate(seqs) if seqs else ()
        check()
        if (
            type(flags) is not tuple
            or len(flags) != 2 * len(selected)
            or any(type(v) is not bool for v in flags)
            or list(flags) != raw["flags"]
        ):
            raise ValueError("independent GA feasibility Boolean predicate differs")
        for sequence, flag in zip(seqs, flags, strict=True):
            if sequence in seen and seen[sequence] is not flag:
                raise ValueError("independent GA feasibility repeated-sequence contradiction")
            seen[sequence] = flag
        for draw, a, b in zip(
            selected, flags[: len(selected)], flags[len(selected) :], strict=True
        ):
            pairs[draw["replicate"]] = a, b
        if (
            getattr(predicate, "source_sha256", None) != predicate_sha256
            or canonical_model_logical_hash(candidate) != candidate_sha
        ):
            raise ValueError("independent GA feasibility callback changed source/model")
    verify_matched_summary(
        record["summary"], tuple(pairs[i] for i in range(160)), expected_scope=scope
    )
    keys = ("row_forwards", "forward_calls", "backward_calls", "grad_enabled_forwards")
    delta = {
        "row_forwards": rows,
        "forward_calls": calls,
        "backward_calls": 0,
        "grad_enabled_forwards": 0,
    }
    before, after = record["work_before"], record["work_after"]
    for work in (before, after):
        if (
            work["shared_inner_subphase_split_measured"] is not False
            or work["external_feature_model_work_counted_here"] is not False
        ):
            raise ValueError("independent GA feasibility work scope differs")
        for phase in work["phases"].values():
            if set(phase) != set(keys) or any(type(v) is not int or v < 0 for v in phase.values()):
                raise ValueError("independent GA feasibility work counts differ")
        if work["total"] != {
            key: sum(phase[key] for phase in work["phases"].values()) for key in keys
        }:
            raise ValueError("independent GA feasibility work sums differ")
    for phase in set(before["phases"]) | set(after["phases"]):
        observed = {
            key: after["phases"].get(phase, {}).get(key, 0)
            - before["phases"].get(phase, {}).get(key, 0)
            for key in keys
        }
        if observed != (
            delta if phase == "ga_matched_feasibility_sampling" else dict.fromkeys(keys, 0)
        ):
            raise ValueError("independent GA feasibility actual sampling work differs")
    if (
        rows > 10240
        or calls > 1216
        or after["total"] != {key: before["total"][key] + delta[key] for key in keys}
    ):
        raise ValueError("independent GA feasibility work bound differs")
    if (
        old_ids != tuple(canonical_model_logical_hash(unit.model) for unit in units)
        or proposed_ids != tuple(canonical_model_logical_hash(model) for model in proposals)
        or _source() != expected_source_sha256
    ):
        raise ValueError("independent GA feasibility frozen model/source changed")
    check()
    return {
        "reconstructed": True,
        "paired_paths": 160,
        "native_paths": 320,
        "sampling_work_delta": delta,
        "passed": record["summary"]["passed"],
        "scientific_evidence_accepted": False,
    }

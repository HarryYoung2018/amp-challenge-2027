"""Actual old-occupancy and candidate-path KL for the declared GA operator law."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    MEASURE,
    PartialGuardPlan,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_proposals import (
    replay_native_trace,
    sample_native_proposals,
)
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl


def path_seed(plan, triple, role):
    return int(_json_hash([plan.numerical_sha256, triple, role, "probe-paths"])[:16], 16)


def parent_draws(plan, triple):
    student = tuple(row[0] for row in plan.units).index(triple)
    residue = plan.checkpoint_order.index(student)
    first = plan.native_ordinal + (residue - plan.native_ordinal) % 10
    count = (plan.native_ordinal + plan.remaining_attempts - 1 - first) // 10 + 1
    if count <= 0:
        raise ValueError("partial guard student has no legal attempt support")
    draws = []
    for replicate in range(8):
        key = _json_hash([plan.numerical_sha256, triple, replicate, "iid-legal-ordinal"])
        rng = np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16)))
        legal_ordinal = first + 10 * int(rng.integers(count))
        parent_index = (legal_ordinal - plan.native_ordinal) % 128
        draws.append(
            {
                "replicate": replicate,
                "legal_native_ordinal": legal_ordinal,
                "parent_index": parent_index,
                "parent": plan.parents[parent_index],
                "legal_student_count": count,
                "log_probability": -math.log(count),
            }
        )
    return tuple(draws)


def active_terms(terms, states, model):
    from amp_challenge.generators.diffusion.native_endpoint import _state_contract

    values = [
        float(value)
        for value, state in zip(terms, states, strict=True)
        if _state_contract(state, model.config)[1] > 0
    ]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise FloatingPointError("partial guard invalid active transition terms")
    return values


def trajectory_summary(rows, name):
    values, masses = [], []
    for row in rows:
        active = row[name]
        values.extend(active)
        masses.extend([1 / (len(rows) * len(active))] * len(active))
    return asdict(summarize_kl(values, weights=masses))


class GAPartialOperatorGuard:
    def __init__(self, plan, units, *, check, counter):
        if type(plan) is not PartialGuardPlan:
            raise TypeError("exact partial guard plan required")
        plan.__post_init__()
        if type(counter) is not NativeWorkCounter:
            raise TypeError("partial guard requires the exact native work counter")
        counter.require_active(units)
        if (
            tuple(unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True))
            != plan.units
        ):
            raise ValueError("partial guard actual old behavior/initializer differs")
        self.plan, self.units, self.check, self.counter = plan, tuple(units), check, counter
        self.old_rows, self.old_states, self.records = [], {}, []

    def prepare_old(self):
        if self.old_rows:
            raise ValueError("partial old occupancy cannot be resampled")
        for unit in self.units:
            self.check()
            draws = parent_draws(self.plan, unit.triple)
            row = {"triple": unit.triple, "draws": draws, "traces": [], "authenticated": False}
            self.old_rows.append(row)
            with self.counter.at("partial_old_sampling"):
                traces = sample_native_proposals(
                    unit.model,
                    tuple(draw["parent"] for draw in draws),
                    start_levels=(32,) * 8,
                    seed=path_seed(self.plan, unit.triple, "old"),
                    ordinals=tuple(range(8)),
                )
            row["traces"] = [asdict(trace) for trace in traces]
            states = []
            for trace in traces:
                self.check()
                with self.counter.at("partial_old_authentication"):
                    _, path_states = replay_native_trace(
                        unit.model, trace, authenticate_sampling=True
                    )
                states.append(path_states)
            self.old_states[unit.triple] = tuple(states)
            row["authenticated"] = True
            self.check()

    def evaluate(self, unit, candidate, *, candidate_index):
        self.check()
        if type(candidate_index) is not int or not 0 <= candidate_index <= 8:
            raise ValueError("partial candidate index differs")
        old_states = self.old_states[unit.triple]
        if len(old_states) != 8:
            raise ValueError("partial guard lacks complete old occupancy")
        record = {
            "triple": unit.triple,
            "candidate_index": candidate_index,
            "models": [
                canonical_model_logical_hash(model)
                for model in (unit.model, candidate, unit.reference)
            ],
            "plan_sha256": self.plan.sha256,
            "local_paths": [],
            "candidate_paths": [],
            "local_summary": None,
            "path_mean": None,
            "path_mcse": None,
            "reference_active_summary": None,
            "complete": False,
            "measure": MEASURE,
        }
        self.records.append(record)
        for states in old_states:
            self.check()
            with self.counter.at("partial_old_candidate_kernels"):
                old = _validated_transition_kernels(unit.model, states, NATIVE_ENDPOINT_DEFAULTS)
                current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
            terms = [
                complete_subset_commit_kl(left, right)
                for left, right in zip(old, current, strict=True)
            ]
            record["local_paths"].append(
                {"transition_kl": terms, "active": active_terms(terms, states, unit.model)}
            )
        draws = parent_draws(self.plan, unit.triple)
        with self.counter.at("partial_candidate_sampling"):
            traces = sample_native_proposals(
                candidate,
                tuple(draw["parent"] for draw in draws),
                start_levels=(32,) * 8,
                seed=path_seed(self.plan, unit.triple, "candidate"),
                ordinals=tuple(range(8)),
            )
        for draw, trace in zip(draws, traces, strict=True):
            self.check()
            row = {
                "draw": draw,
                "trace": asdict(trace),
                "transition_kl": None,
                "active": None,
                "path_sum": None,
            }
            record["candidate_paths"].append(row)
            with self.counter.at("partial_candidate_reference_replay"):
                current_logp, states = replay_native_trace(
                    candidate, trace, authenticate_sampling=True
                )
                reference_logp, _ = replay_native_trace(unit.reference, trace)
            with self.counter.at("partial_candidate_reference_kernels"):
                current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
                frozen = _validated_transition_kernels(
                    unit.reference, states, NATIVE_ENDPOINT_DEFAULTS
                )
            terms = [
                complete_subset_commit_kl(left, right)
                for left, right in zip(current, frozen, strict=True)
            ]
            row.update(
                transition_kl=terms,
                active=active_terms(terms, states, candidate),
                path_sum=math.fsum(terms),
                current_joint_log_probability=draw["log_probability"] + current_logp,
                reference_joint_log_probability=draw["log_probability"] + reference_logp,
                sampled_conditional_log_ratio=current_logp - reference_logp,
            )
        sums = np.asarray([row["path_sum"] for row in record["candidate_paths"]])
        record.update(
            local_summary=trajectory_summary(record["local_paths"], "active"),
            path_mean=float(sums.mean()),
            path_mcse=float(sums.std(ddof=1) / math.sqrt(8)),
            reference_active_summary=trajectory_summary(record["candidate_paths"], "active"),
            complete=True,
        )
        self.check()
        if record["models"] != [
            canonical_model_logical_hash(model) for model in (unit.model, candidate, unit.reference)
        ]:
            raise ValueError("partial guard models changed during diagnostics")
        return record


def combined_summary(shared_rows, partial_rows, plan):
    """Keep legacy pooled-active and new equal-path active weighting distinct."""
    if (
        len(shared_rows) != 10
        or len(partial_rows) != 10
        or tuple(row["triple"] for row in partial_rows) != tuple(unit[0] for unit in plan.units)
    ):
        raise ValueError("partial guard summary lacks ordered ten students")
    summaries = []
    counts = [parent_draws(plan, unit[0])[0]["legal_student_count"] for unit in plan.units]
    for label, masses in (
        ("equal_ten", [0.1] * 10),
        ("legal_ordinal_mass", [count / plan.remaining_attempts for count in counts]),
    ):
        old_local, old_local_mass, fullmask, fullmask_mass = [], [], [], []
        local, local_mass, reference, reference_mass = [], [], [], []
        # Original anchor quantiles are reconstructed from their exact per-student
        # empirical summaries in the update layer; here per-student guards remain
        # mandatory. Means and fullmask/new-active mixtures are explicit.
        for weight, shared, partial in zip(masses, shared_rows, partial_rows, strict=True):
            values = shared["fullmask_reference"]["active_transitions"]
            fullmask.extend(values)
            fullmask_mass.extend([weight / len(values)] * len(values))
            for target, weights, key in (
                (local, local_mass, "local_paths"),
                (reference, reference_mass, "candidate_paths"),
            ):
                for path in partial[key]:
                    target.extend(path["active"])
                    weights.extend([weight / (8 * len(path["active"]))] * len(path["active"]))
            old_local.append(shared["local"]["local_old_candidate"]["mean"])
            old_local_mass.append(weight)
        summaries.append(
            {
                "mixture": label,
                "student_masses": masses,
                "old_local_mean": math.fsum(
                    value * weight for value, weight in zip(old_local, old_local_mass, strict=True)
                ),
                "old_local_p99_upper_bound": max(
                    row["local"]["local_old_candidate"]["p99"] for row in shared_rows
                ),
                "fullmask_mean": math.fsum(
                    weight * row["fullmask_reference"]["mean"]
                    for weight, row in zip(masses, shared_rows, strict=True)
                ),
                "fullmask_mcse": math.sqrt(
                    math.fsum(
                        (weight * row["fullmask_reference"]["monte_carlo_standard_error"]) ** 2
                        for weight, row in zip(masses, shared_rows, strict=True)
                    )
                ),
                "fullmask_active": asdict(summarize_kl(fullmask, weights=fullmask_mass)),
                "partial_local": asdict(summarize_kl(local, weights=local_mass)),
                "partial_path_mean": math.fsum(
                    weight * row["path_mean"]
                    for weight, row in zip(masses, partial_rows, strict=True)
                ),
                "partial_path_mcse": math.sqrt(
                    math.fsum(
                        (weight * row["path_mcse"]) ** 2
                        for weight, row in zip(masses, partial_rows, strict=True)
                    )
                ),
                "partial_reference_active": asdict(summarize_kl(reference, weights=reference_mass)),
            }
        )
    per_student = all(
        old["local"]["local_old_candidate"]["mean"] <= 0.01
        and old["local"]["local_old_candidate"]["p99"] <= 0.02
        and old["fullmask_reference"]["mean"] <= 0.08
        and old["fullmask_reference"]["active_summary"]["p99"] <= 0.02
        and new["local_summary"]["mean"] <= 0.01
        and new["local_summary"]["p99"] <= 0.02
        and new["path_mean"] <= 0.08
        and new["reference_active_summary"]["p99"] <= 0.02
        for old, new in zip(shared_rows, partial_rows, strict=True)
    )
    passed = per_student and all(
        row["old_local_mean"] <= 0.01
        and row["old_local_p99_upper_bound"] <= 0.02
        and row["fullmask_mean"] <= 0.08
        and row["fullmask_active"]["p99"] <= 0.02
        and row["partial_local"]["mean"] <= 0.01
        and row["partial_local"]["p99"] <= 0.02
        and row["partial_path_mean"] <= 0.08
        and row["partial_reference_active"]["p99"] <= 0.02
        for row in summaries
    )
    return {"mixtures": summaries, "all_per_student_passed": per_student, "all_kl_passed": passed}

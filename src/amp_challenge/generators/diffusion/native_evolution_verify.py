"""Independent numerical/event reconstruction; never calls the new producer.

Uses the accepted native trace scorer and Gaussian backend, not an independently
implemented neural network. External feature/oracle receipt authenticity remains
upstream; every required input is explicitly supplied, not read from a teacher.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule
from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    _seed,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_evolution_records import (
    DIRECTIONS,
    OPERATORS,
    EvolutionBranch,
    semantic_history,
)
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import _trace
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl


def _paired_mean(mean):
    """Reconstruct the fixed half/half contrast without a cancelling flat sum.

    Scale and subtract each child/parent objective first. In particular, equal
    vectors give exactly zero under the unchanged strict positive-advantage gate.
    This reader does not call the producer's paired-contrast implementation.
    """
    values = np.asarray(mean, dtype=np.float64)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("independent paired mean requires two finite objective vectors")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    first = float(values[0, 0] / scale - values[1, 0] / scale)
    second = float(values[0, 1] / scale - values[1, 1] / scale)
    result = ((first + second) * 0.5) * scale
    if not math.isfinite(result):
        raise ValueError("independent paired mean is not representable in float64")
    return result


def verify_evolution_wave(
    wave,
    *,
    units,
    history,
    posterior,
    cache,
    expected_source_sha256: str,
    eligible_charged_ids: frozenset[str],
    prior_branches=(),
    seen_before=None,
    expected_behavior_versions=None,
):
    """Verify recorded collection numerics and quotas without regenerating it.

    It does not certify external wall time, upstream learner fitting, or the
    truth of the fixed-context feasibility callback. Those receipts are supplied
    to the independent campaign auditor separately.
    prior_branches are the externally reconstructed post-response credit/yield
    state (empty only for common64). They are not inferred from this wave's own
    allocation claims. verify_evolution_branch_update reconstructs that state.
    """
    wave.check()
    posterior.check()
    if wave.round_index != history.round_index:
        raise ValueError("evolution origin generation differs from charged history")
    if expected_behavior_versions is None:
        if history.round_index != 1:
            raise ValueError("later evolution replay needs reconstructed behavior versions")
        expected_behavior_versions = {unit.triple: 0 for unit in units}
    if set(expected_behavior_versions) != {unit.triple for unit in units} or any(
        type(value) is not int or not 0 <= value < history.round_index
        for value in expected_behavior_versions.values()
    ):
        raise ValueError("evolution expected behavior version inventory differs")
    if (
        wave.history_sha256,
        wave.semantic_history_sha256,
        wave.posterior_sha256,
        wave.feature_binding_sha256,
        wave.source_sha256,
    ) != (
        history.sha256,
        semantic_history(history, eligible_charged_ids),
        posterior.sha256,
        cache.binding.sha256,
        expected_source_sha256,
    ):
        raise ValueError("evolution audit expected source/history/features differ")
    if wave.policy_identities != tuple((unit.triple, unit.policy_sha256) for unit in units):
        raise ValueError("evolution audit generating models differ")
    for index, observed in enumerate(wave.frozen_draws):
        seed = int(
            _json_hash(
                [
                    semantic_history(history, eligible_charged_ids),
                    "evolution-frozen-coefficient",
                    index,
                ]
            )[:16],
            16,
        )
        np.testing.assert_array_equal(
            observed, posterior.backend.draw_latent(seed=seed, count=1)[0]
        )
    if len(wave.frozen_draws) > 40 or (wave.status == "complete" and len(wave.frozen_draws) != 40):
        raise ValueError("evolution audit frozen draw inventory differs")
    if wave.status != "complete" and not wave.attempts and not wave.allocations:
        if wave.shortlist_ordinals:
            raise ValueError("early evolution stop invents a shortlist")
        if wave.status == "stopped_no_eligible_parent" and (
            eligible_charged_ids != frozenset() or wave.frozen_draws
        ):
            raise ValueError("empty-parent stop inventory differs")
        return {
            "reconstructed": True,
            "complete": False,
            "attempts": 0,
            "scope": "early_stop_prefix_external_failure_cause_not_authenticated",
        }
    if len(wave.frozen_draws) != 40:
        raise ValueError("native attempts require all frozen branch functions")
    offset = int(_json_hash([history.seed, "evolution-round-robin"])[:16], 16) % 10
    if wave.status == "complete" and (len(wave.attempts) != 480 or len(wave.allocations) != 20):
        raise ValueError("complete wave lacks exact ten by48 proposal quotas")
    training = set().union(*(unit.initialization.training_sequence_ids for unit in units))
    charged = {sequence_id(row.sequence) for row in history.observations}
    seen = dict(seen_before or {})
    by_triple = {unit.triple: unit for unit in units}
    successful = {
        sequence_id(row.sequence) for row in history.observations if row.status == "successful"
    }
    if (
        type(eligible_charged_ids) is not frozenset
        or not eligible_charged_ids
        or not eligible_charged_ids <= successful
        or (history.round_index == 1 and prior_branches)
        or (history.round_index > 1 and len(prior_branches) != 40)
    ):
        raise ValueError("explicit parent eligibility/prior credit state differs")
    prior = {(branch.triple, branch.branch): branch for branch in prior_branches}
    roots = tuple(
        row.sequence
        for row in history.observations
        if sequence_id(row.sequence) in eligible_charged_ids
    )
    expected_branches_by_key = {}
    for unit_index, unit in enumerate(units):
        for branch_index in range(4):
            values = posterior.draw(
                cache.matrix(roots), wave.frozen_draws[unit_index * 4 + branch_index]
            ) @ np.asarray(DIRECTIONS[branch_index])
            parent = min(
                zip(values, roots, strict=True), key=lambda row: (-row[0], sequence_id(row[1]))
            )[1]
            old = prior.get((unit.triple, branch_index))
            expected_branches_by_key[(unit.triple, branch_index)] = (
                EvolutionBranch(unit.triple, branch_index, parent)
                if old is None
                else replace(old, parent=parent)
            )
    flags = {}
    evaluated = [
        index for index, attempt in enumerate(wave.attempts) if attempt.contrast is not None
    ]
    expected_maxima = {}
    for start in range(0, len(evaluated), 128):
        indices = evaluated[start : start + 128]
        raw = cache.matrix(tuple(wave.attempts[index].trace.endpoint for index in indices))
        full = posterior.backend.evaluate_draws(
            posterior.transform.apply(raw), np.asarray(wave.frozen_draws)
        )
        maxima = np.max(np.sum(full * np.asarray(DIRECTIONS * 10)[:, None, :], axis=2), axis=0)
        expected_maxima.update(zip(indices, maxima, strict=True))
    cached_by_id = {sequence_id(seq): seq for seq in cache.rows}
    for event in wave.feature_events:
        if event["kind"] == "public_feasibility":
            for key, flag in zip(event["sequence_ids"], event["flags"], strict=True):
                if type(flag) is not bool or (key in flags and flags[key] != flag):
                    raise ValueError("fixed feasibility changed within a sealed wave")
                flags[key] = flag
        elif event["kind"] == "cache_hit":
            seq = cached_by_id.get(event["sequence_id"])
            if seq is None:
                raise ValueError("audited feature cache hit missing")
            vector, receipt, row, numerical = cache.rows[seq]
            if (receipt, row, numerical) != (
                event["receipt_sha256"],
                event["receipt_row"],
                event["feature_sha256"],
            ) or _json_hash(vector.tolist()) != numerical:
                raise ValueError("audited feature cache receipt/order/bytes differ")
    for allocation_index, allocation in enumerate(wave.allocations):
        if allocation_index == 10:
            for unit_index, unit in enumerate(units):
                for branch_index in range(4):
                    key = (unit.triple, branch_index)
                    old = expected_branches_by_key[key]
                    parent_value = float(
                        posterior.draw(
                            cache.matrix((old.parent,)),
                            wave.frozen_draws[unit_index * 4 + branch_index],
                        )[0]
                        @ np.asarray(DIRECTIONS[branch_index])
                    )
                    choices = [(parent_value, old.parent)] + [
                        (attempt.thompson_value, attempt.trace.endpoint)
                        for attempt in wave.attempts[:240]
                        if (attempt.triple, attempt.branch) == key and attempt.rejection is None
                    ]
                    parent = min(choices, key=lambda row: (-row[0], sequence_id(row[1])))[1]
                    expected_branches_by_key[key] = replace(old, parent=parent)
        expected_triple = units[(allocation_index % 10 + offset) % 10].triple
        if allocation.stage != allocation_index // 10 or allocation.triple != expected_triple:
            raise ValueError("evolution round-robin/stage assignment differs")
        if allocation.branches != tuple(
            expected_branches_by_key[(expected_triple, index)] for index in range(4)
        ):
            raise ValueError("evolution frozen-function parent/prior credit state differs")
        scores = []
        for branch in allocation.branches:
            a, b = (
                branch.useful_descendants + 1,
                branch.charged_descendants - branch.useful_descendants + 1,
            )
            score = a / (a + b) + math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            scores.append(score * (1 if wave.variant == "no_counterfactual" else branch.credit))
        ideal = np.asarray(scores) * 16 / sum(scores)
        counts = np.floor(ideal).astype(int)
        order = sorted(range(4), key=lambda index: (-(ideal[index] - counts[index]), index))
        for index in order[: 16 - int(counts.sum())]:
            counts[index] += 1
        if allocation.counts != tuple(map(int, counts + 2)) or not np.allclose(
            allocation.scores, scores, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("evolution credit/yield largest-remainder allocation differs")
        if allocation.parent_ids != tuple(
            sequence_id(branch.parent) for branch in allocation.branches
        ):
            raise ValueError("evolution parent inventory differs")
        expected_branches = tuple(
            branch
            for branch, count in zip(allocation.branches, allocation.counts, strict=True)
            for _ in range(count)
        )
        for local, branch in enumerate(expected_branches):
            position = allocation_index * 24 + local
            if position >= len(wave.attempts):
                if wave.status == "complete":
                    raise ValueError("complete evolution attempt prefix truncated")
                break
            attempt = wave.attempts[position]
            if (
                type(attempt.behavior_version) is not int
                or attempt.triple not in expected_behavior_versions
                or attempt.behavior_version != expected_behavior_versions[attempt.triple]
            ):
                raise ValueError("evolution origin behavior version differs")
            ordinal = (history.round_index - 1) * 480 + position
            operator = OPERATORS[
                int(_seed(history.seed, ordinal, "evolution-operator", 0).integers(4))
            ]
            if (
                attempt.ordinal,
                attempt.stage,
                attempt.triple,
                attempt.branch,
                attempt.lineage_parent,
                attempt.operator,
            ) != (
                ordinal,
                allocation.stage,
                allocation.triple,
                branch.branch,
                branch.parent,
                operator,
            ):
                raise ValueError("evolution authoritative lineage/operator/ordinal differs")
            unit = by_triple[attempt.triple]
            template, level, length_logp = _expected_operator(
                unit, branch.parent, operator, history.seed, ordinal
            )
            trace = attempt.trace
            if (
                (trace.parent, trace.seed, trace.ordinal, trace.start_level)
                != (template, history.seed, ordinal, level)
                or attempt.operator_log_probability != -math.log(4)
                or attempt.length_log_probability != length_logp
            ):
                raise ValueError("evolution complete conditional operator factors differ")
            logp, _ = replay_native_trace(unit.model, trace, authenticate_sampling=True)
            if not math.isclose(
                attempt.complete_conditional_log_probability,
                logp + length_logp - math.log(4),
                abs_tol=1e-6,
                rel_tol=1e-5,
            ):
                raise ValueError("evolution complete conditional path log probability differs")
            key = trace.endpoint_sha256
            first = seen.get(key, ordinal)
            expected_rejection = (
                "training_overlap"
                if key in training
                else "already_charged"
                if key in charged
                else "duplicate"
                if key in seen
                else "identity_noop"
                if trace.endpoint == branch.parent
                else None
            )
            seen.setdefault(key, ordinal)
            if attempt.first_seen_ordinal != first:
                raise ValueError("evolution duplicate first-lineage credit changed")
            if attempt.contrast is not None:
                feasible = flags[key]
                expected_rejection = expected_rejection or (
                    None if feasible else "public_infeasible"
                )
                belief = posterior.joint(cache.matrix((trace.endpoint, branch.parent)))
                direction = np.array([0.5, 0.5, -0.5, -0.5])
                mean = _paired_mean(belief.mean)
                variance = max(0.0, float(direction @ belief.covariance.reshape(4, 4) @ direction))
                absolute = float(belief.mean[0].mean())
                np.testing.assert_allclose(
                    attempt.posterior_mean, belief.mean[0], atol=1e-10, rtol=1e-9
                )
                absolute_risk = absolute - math.sqrt(
                    max(
                        0.0,
                        float(
                            np.array([0.5, 0.5])
                            @ belief.covariance[0, :, 0, :]
                            @ np.array([0.5, 0.5])
                        ),
                    )
                )
                contrast = attempt.contrast
                np.testing.assert_allclose(
                    (
                        contrast.mean,
                        contrast.variance,
                        contrast.advantage,
                        contrast.absolute_mean,
                        contrast.absolute_risk,
                    ),
                    (mean, variance, mean - math.sqrt(variance), absolute, absolute_risk),
                    atol=1e-10,
                    rtol=1e-9,
                )
                if contrast.feasible != feasible or contrast.accepted != bool(
                    feasible and mean - math.sqrt(variance) > 0 and absolute_risk >= 0.5
                ):
                    raise ValueError("evolution paired/absolute target gate differs")
                unit_index = tuple(by_triple).index(unit.triple)
                sampled = posterior.draw(
                    cache.matrix((trace.endpoint,)),
                    wave.frozen_draws[4 * unit_index + branch.branch],
                )[0]
                np.testing.assert_allclose(
                    attempt.thompson_value,
                    sampled @ np.array(DIRECTIONS[branch.branch]),
                    atol=1e-10,
                    rtol=1e-9,
                )
                np.testing.assert_allclose(
                    attempt.max_thompson_value, expected_maxima[position], atol=1e-10, rtol=1e-9
                )
            if attempt.rejection != expected_rejection:
                raise ValueError("evolution support/duplicate/feasibility rejection differs")
            if wave.status == "complete" and attempt.contrast is None:
                raise ValueError("complete evolution attempt lacks frozen evaluation")
    if wave.status == "complete":
        eligible = [
            index for index, attempt in enumerate(wave.attempts) if attempt.rejection is None
        ]
        mean_rank = sorted(
            eligible,
            key=lambda index: (
                -sum(wave.attempts[index].posterior_mean) / 2,
                wave.attempts[index].trace.endpoint_sha256,
            ),
        )
        draw_rank = sorted(
            eligible,
            key=lambda index: (
                -wave.attempts[index].max_thompson_value,
                wave.attempts[index].trace.endpoint_sha256,
            ),
        )
        expected_shortlist = []
        for pair in zip(mean_rank, draw_rank, strict=True):
            for index in pair:
                if index not in expected_shortlist and len(expected_shortlist) < 256:
                    expected_shortlist.append(index)
        if wave.shortlist_ordinals != tuple(expected_shortlist):
            raise ValueError("evolution mixed mean/frozen-direction shortlist differs")
    return {
        "reconstructed": True,
        "attempts": len(wave.attempts),
        "complete": wave.status == "complete",
        "scope": "independent_event_and_numeric_reconstruction_not_oracle_or_timing_authentication",
    }


def verify_evolution_branch_update(
    *,
    previous_branches,
    previous_wave,
    history,
    posterior,
    cache,
    eligible_charged_ids,
    actual_branches,
):
    """Rebuild observed-yield and decayed paired credit from charged records."""
    values = {
        (row.triple, row.branch): replace(row, credit=1 + 0.9 * (row.credit - 1))
        for row in previous_branches
    }
    attempts = {
        attempt.trace.endpoint: attempt
        for attempt in previous_wave.attempts
        if attempt.rejection is None
    }
    for observation in history.observations[-16:]:
        attempt = attempts.get(observation.sequence)
        if attempt is None:
            continue
        key = (attempt.triple, attempt.branch)
        branch = values[key]
        useful = (
            observation.status == "successful"
            and sequence_id(observation.sequence) in eligible_charged_ids
            and sum(observation.objectives) / 2 > 0.5
        )
        credit = 1.0
        if previous_wave.variant != "no_counterfactual":
            belief = posterior.joint(cache.matrix((attempt.trace.endpoint, attempt.lineage_parent)))
            vector = np.array([0.5, 0.5, -0.5, -0.5])
            advantage = _paired_mean(belief.mean) - math.sqrt(
                max(0.0, float(vector @ belief.covariance.reshape(4, 4) @ vector))
            )
            credit = float(
                np.clip(
                    0.9 * branch.credit
                    + 0.1 * math.exp(float(np.clip(advantage / 0.1, math.log(0.25), math.log(4)))),
                    0.25,
                    4,
                )
            )
        values[key] = replace(
            branch,
            credit=credit,
            charged_descendants=branch.charged_descendants + 1,
            useful_descendants=branch.useful_descendants + int(useful),
        )
    expected = tuple(values[(branch.triple, branch.branch)] for branch in previous_branches)
    if len(actual_branches) != len(expected) or any(
        replace(actual, credit=reference.credit) != reference
        or not math.isclose(actual.credit, reference.credit, abs_tol=1e-10, rel_tol=1e-9)
        for actual, reference in zip(actual_branches, expected, strict=True)
    ):
        raise ValueError("charged branch-yield/counterfactual update differs")
    return {
        "reconstructed": True,
        "branches": len(expected),
        "scope": "charged_history_and_frozen_paired_posterior_not_causal_credit",
    }


def _expected_operator(unit, parent, operator, seed, ordinal):
    if operator == "full_regeneration":
        index = int(_seed(seed, ordinal, "evolution-full-length", 0).integers(len(unit.sequences)))
        length = len(unit.sequences[index])
        template = min(seq for seq in unit.sequences if len(seq) == length)
        return (
            template,
            unit.model.config.levels,
            math.log(sum(len(seq) == length for seq in unit.sequences) / len(unit.sequences)),
        )
    desired = (
        1
        if operator == "single_site"
        else math.ceil(len(parent) * (0.25 if operator == "quarter_remask" else 0.5))
    )
    for level in range(1, unit.model.config.levels + 1):
        if (
            int(
                CosineMaskSchedule().mask_counts(
                    len(parent), level, total_levels=unit.model.config.levels
                )[0]
            )
            >= desired
        ):
            return parent, level, 0.0
    raise ValueError("no native operator schedule support")


def verify_evolution_operator_report(
    old, candidate, reference, report, triple, candidate_index, *, guard
):
    """Reconstruct guarded trajectories from saved records, never call evaluate."""
    matches = [record for record in guard.records if _json_hash(record) == report["receipt_sha256"]]
    if len(matches) != 1:
        raise ValueError("operator report lacks unique authenticated trajectory record")
    record = matches[0]
    identities = tuple(canonical_model_logical_hash(model) for model in (old, candidate, reference))
    if (
        record["triple"] != triple
        or record["candidate_index"] != candidate_index
        or record["plan_sha256"] != guard.plan_sha256
        or record["source_sha256"] != guard.source_sha256
        or len(record["paths"]) != 8
        or tuple(record["models"]) != identities
        or tuple(
            report[key]
            for key in ("old_model_sha256", "candidate_model_sha256", "reference_model_sha256")
        )
        != identities
        or report["path_count"] != 8
    ):
        raise ValueError("operator report source/plan/index differs")
    unit = type("AuditUnit", (), {"model": candidate, "sequences": guard.corpora[triple]})()
    rows = guard.parents[triple]
    probabilities = np.array([count / 48 for _, count in rows])
    path_seed = int(
        _json_hash([guard.seed, guard.round_index, triple, "evolution-actual-operator-paths"])[:16],
        16,
    )
    values, active, weights = [], [], []
    for index, row in enumerate(record["paths"]):
        operator = OPERATORS[index // 2]
        rng = _seed(path_seed, index, "evolution-operator-parent-" + triple, 0)
        parent_index = int(rng.choice(len(rows), p=probabilities))
        parent = rows[parent_index][0]
        template, level, length_logp = _expected_operator(unit, parent, operator, path_seed, index)
        trace = _trace(row["trace"])
        if (
            row["operator"],
            row["parent"],
            row["parent_log_probability"],
            row["length_log_probability"],
            row["replicate"],
            trace.parent,
            trace.start_level,
            trace.seed,
            trace.ordinal,
        ) != (
            operator,
            parent,
            math.log(probabilities[parent_index]),
            length_logp,
            index % 2,
            template,
            level,
            path_seed,
            index,
        ):
            raise ValueError("actual operator/parent/length/path stream differs")
        current_logp, states = replay_native_trace(candidate, trace, authenticate_sampling=True)
        reference_logp, _ = replay_native_trace(reference, trace)
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        frozen = _validated_transition_kernels(reference, states, NATIVE_ENDPOINT_DEFAULTS)
        terms = [complete_subset_commit_kl(p, q) for p, q in zip(current, frozen, strict=True)]
        if row["operator_weight"] != 0.25:
            raise ValueError("actual operator mixture weight differs")
        np.testing.assert_allclose(row["transition_kl"], terms, atol=1e-10, rtol=1e-9)
        shared = math.log(probabilities[parent_index]) + length_logp + math.log(0.25)
        np.testing.assert_allclose(
            (
                row["conditional_kl"],
                row["current_augmented_log_probability"],
                row["reference_augmented_log_probability"],
                row["sampled_log_ratio"],
            ),
            (
                math.fsum(terms),
                shared + current_logp,
                shared + reference_logp,
                current_logp - reference_logp,
            ),
            atol=1e-10,
            rtol=1e-9,
        )
        values.append(math.fsum(terms))
        selected = [
            term for term, kernel in zip(terms, current, strict=True) if kernel.commit_count
        ]
        active.extend(selected)
        weights.extend([1 / (8 * len(selected))] * len(selected))
    groups = np.array(values).reshape(4, 2)
    order = np.argsort(active, kind="stable")
    cumulative = np.cumsum(np.asarray(weights)[order])
    p99 = float(np.asarray(active)[order[min(np.searchsorted(cumulative, 0.99), len(active) - 1)]])
    np.testing.assert_allclose(
        (report["mean"], report["monte_carlo_standard_error"], report["active_transition_p99"]),
        (groups.mean(), np.sqrt(np.sum(groups.var(axis=1, ddof=1) / 2) / 16), p99),
        atol=1e-10,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        (record["mean"], record["stratified_mcse"], record["active_transition_p99"]),
        (report["mean"], report["monte_carlo_standard_error"], report["active_transition_p99"]),
        atol=1e-10,
        rtol=1e-9,
    )
    return {"reconstructed": True, "scope": "frozen_parent_operator_conditional_path_not_global_KL"}

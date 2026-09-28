"""Independent recorded-tree reconstruction; never calls collection/evaluation.

Replays exact native actions and RNG, positive PUCT choices, backpropagation,
exclusions, global Pareto buffers and weights. Posterior scores remain externally
authenticated inputs, not independently established scientific truth.
"""

from __future__ import annotations

import math

import numpy as np

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _seed,
    _state_contract,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_proposals import PROBABILITY_ATOL, PROBABILITY_RTOL
from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.diffusion.native_tree_records import (
    TR2D2_CONFIG_SHA256,
    NativeTreeGeneration,
    TreeReplayBuffer,
    TreeRoot,
    normalized_tree_weights,
    pareto_indices,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string


def _close(left, right):
    if not np.allclose(left, right, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL):
        raise ValueError("recorded tree neural probability differs beyond declared tolerance")


def replay_tree_path(unit, root, path, *, seed: int, prefix=(), suffix_ordinal: int):
    """Full root-to-endpoint scorer, including prefixes sampled in earlier iterations."""
    levels = unit.model.config.levels
    if (
        len(path.transitions) != levels
        or path.transitions[: len(prefix)] != prefix
        or path.scope
        != "conditional_checkpoint_length_model_path_not_whole_search_or_endpoint_marginal"
        or len(path.endpoint) != root.length
    ):
        raise ValueError("tree path prefix/length/scope differs")
    tokens = (
        PeptideVocabulary()
        .encode(("A" * root.length,), max_length=unit.model.config.max_length)
        .tokens[0]
    )
    tokens[: root.length] = MASK_TOKEN_INDEX
    states, after_states, counts = [], [], []
    for index, step in enumerate(path.transitions):
        level = levels - index
        if step.level != level or step.before_tokens_sha256 != _json_hash(tokens.tolist()):
            raise ValueError("tree path state/level chain differs")
        if index >= len(prefix) and step.rng_ordinal != suffix_ordinal:
            raise ValueError("tree path suffix random ordinal differs")
        state = NativeTransitionState(tokens, root.length, level)
        _, count = _state_contract(state, unit.model.config)
        states.append(state)
        counts.append(count)
        if len(step.draw.positions) != count or len(step.draw.residues) != count:
            raise ValueError("tree path commit count differs")
        if any(
            type(position) is not int or not 0 <= position < root.length
            for position in step.draw.positions
        ):
            raise ValueError("tree path position support differs")
        if any(tokens[position] != MASK_TOKEN_INDEX for position in step.draw.positions):
            raise ValueError("tree path overwrites an unmasked token")
        tokens[list(step.draw.positions)] = step.draw.residues
        if step.after_tokens_sha256 != _json_hash(tokens.tolist()):
            raise ValueError("tree path after-state chain differs")
        after_states.append(tokens.copy())
    kernels = _validated_transition_kernels(unit.model, tuple(states), NATIVE_ENDPOINT_DEFAULTS)
    references = _validated_transition_kernels(
        unit.reference, tuple(states), NATIVE_ENDPOINT_DEFAULTS
    )
    behavior_logps, reference_logps = [], []
    for step, kernel, reference in zip(path.transitions, kernels, references, strict=True):
        draw = kernel.sample(
            _seed(seed, step.rng_ordinal, "tr2d2-commit-" + unit.triple, step.level)
        )
        if (draw.positions, draw.residues) != (step.draw.positions, step.draw.residues):
            raise ValueError("tree path replayed native draw differs")
        logp = kernel.log_probability(step.draw.positions, step.draw.residues)
        logref = reference.log_probability(step.draw.positions, step.draw.residues)
        _close(logp, step.draw.log_probability)
        _close(logref, step.reference_log_probability)
        behavior_logps.append(logp)
        reference_logps.append(logref)
    endpoint = PeptideVocabulary().decode(np.stack([tokens]))[0]
    if endpoint != path.endpoint:
        raise ValueError("tree path endpoint differs")
    _close(math.fsum(behavior_logps), path.behavior_log_probability)
    _close(math.fsum(reference_logps), path.reference_log_probability)
    positive = next(index for index in range(len(prefix), levels) if counts[index] > 0)
    return positive + 1, after_states[positive]


def _front_order(attempts, indices, context, *, all_layers):
    remaining, result = list(indices), []
    while remaining:
        front = [
            remaining[i]
            for i in pareto_indices(tuple(attempts[i].posterior.objectives for i in remaining))
        ]
        result.extend(
            sorted(
                front,
                key=lambda i: (
                    -context.scalarize(attempts[i].posterior.objectives),
                    sequence_id(attempts[i].path.endpoint),
                ),
            )
        )
        if not all_layers:
            break
        remaining = [i for i in remaining if i not in set(front)]
    return tuple(result)


def verify_native_tree_generation(
    ensemble, generation: NativeTreeGeneration, *, expected_posterior_sha256: str
) -> None:
    """Rebuild from a supplied exact pre-update ensemble; no producer rerun."""
    from amp_challenge.generators.diffusion.native_tr2d2 import NativeTR2D2Ensemble

    if type(ensemble) is not NativeTR2D2Ensemble or type(generation) is not NativeTreeGeneration:
        raise TypeError("tree verification record types differ")
    history = ensemble._history
    if history is None or history.round_index == 29:
        raise ValueError("tree verification requires pre-update nonterminal history")
    if (
        (generation.round_index, generation.seed, generation.history_sha256)
        != (history.round_index, ensemble.seed, history.sha256)
        or generation.configuration_sha256 != TR2D2_CONFIG_SHA256
        or any(
            flag is not False
            for flag in (
                generation.campaign_eligible,
                generation.scientific_evidence_accepted,
                generation.production_eligible,
            )
        )
        or len(generation.attempts) > 800
        or len(generation.expansions) > 100
        or len(generation.roots) != len(ensemble._units)
    ):
        raise ValueError("tree generation source/budget/qualification differs")
    binding = generation.evaluator_binding
    if type(binding) is not FrozenNativePosteriorBinding:
        raise TypeError("tree evaluator binding type differs")
    binding.__post_init__()
    if (
        not hash_string(expected_posterior_sha256)
        or binding.posterior_sha256 != expected_posterior_sha256
    ):
        raise ValueError("tree posterior differs from independently supplied identity")
    if (
        binding.history_sha256,
        binding.objective_context_sha256,
        binding.feature_source_sha256,
        binding.evaluator_source_sha256,
    ) != (
        history.sha256,
        ensemble.context.context_sha256,
        ensemble.feature_source_sha256,
        ensemble.evaluator_source_sha256,
    ):
        raise ValueError("tree evaluator fixed code/features/history differ")
    generation.check_output_budget()
    forest = {}
    for unit, root in zip(ensemble._units, generation.roots, strict=True):
        unit.check()
        index = int(
            _seed(
                ensemble.seed, history.round_index, "tr2d2-root-length-" + unit.triple, 0
            ).integers(len(unit.sequences))
        )
        anchor = unit.sequences[index]
        mass = sum(len(seq) == len(anchor) for seq in unit.sequences) / len(unit.sequences)
        expected_root = TreeRoot(
            unit.triple,
            len(anchor),
            math.log(mass),
            sequence_id(anchor),
            unit.policy_sha256,
            unit.reference_sha256,
        )
        if root != expected_root:
            raise ValueError("tree root empirical length/initializer binding differs")
        forest[unit.triple] = [
            dict(
                level=unit.model.config.levels,
                prefix=(),
                parent=None,
                children=[],
                visits=0,
                total=np.zeros(2),
                edge=0.0,
            )
        ]

    def can_expand(nodes, index):
        node = nodes[index]
        return node["level"] > 0 and (
            not node["children"] or any(can_expand(nodes, child) for child in node["children"])
        )

    attempts = generation.attempts
    offset = int(_json_hash([ensemble.seed, "tr2d2-equal-mixture-v1"])[:16], 16) % len(
        ensemble._units
    )
    seen, known_scores, eligible = {}, {}, {unit.triple: [] for unit in ensemble._units}
    charged = {sequence_id(row.sequence) for row in history.observations}
    expected_attempt, cursor = 0, 0
    for expansion_index in range(10 * len(ensemble._units)):
        unit = ensemble._units[
            (expansion_index % len(ensemble._units) + offset) % len(ensemble._units)
        ]
        root = next(root for root in generation.roots if root.triple == unit.triple)
        nodes = forest[unit.triple]
        if not can_expand(nodes, 0):
            continue
        if cursor >= len(generation.expansions):
            raise ValueError("tree silently omitted a bounded expansion")
        expansion = generation.expansions[cursor]
        cursor += 1
        if (
            (expansion.expansion_index, expansion.triple, expansion.iteration)
            != (expansion_index, unit.triple, expansion_index // len(ensemble._units))
            or expansion.attempt_indices != tuple(range(expected_attempt, expected_attempt + 8))
            or not hash_string(expansion.posterior_receipt_sha256)
        ):
            raise ValueError("tree expansion order/inventory differs")
        current, choice_index = 0, 0
        while nodes[current]["children"]:
            if choice_index >= len(expansion.selections):
                raise ValueError("tree missing selection probability")
            choice = expansion.selections[choice_index]
            choice_index += 1
            available = tuple(
                child for child in nodes[current]["children"] if can_expand(nodes, child)
            )
            scores = tuple(
                tuple(
                    map(
                        float,
                        nodes[child]["total"] / nodes[child]["visits"]
                        + 0.1
                        * math.exp(nodes[child]["edge"])
                        * math.sqrt(nodes[current]["visits"])
                        / (1 + nodes[child]["visits"]),
                    )
                )
                for child in available
            )
            frontier = tuple(available[i] for i in pareto_indices(scores))
            ordinal = (history.round_index - 1) * 100 + expansion_index
            selected = int(
                _seed(ensemble.seed, ordinal, "tr2d2-select-" + unit.triple, current).choice(
                    frontier
                )
            )
            if (
                choice.node_id,
                choice.eligible_children,
                choice.pareto_children,
                choice.selected_child,
            ) != (current, available, frontier, selected):
                raise ValueError("tree Pareto/selection replay differs")
            _close(choice.puct_scores, scores)
            _close(choice.conditional_log_probability, -math.log(len(frontier)))
            current = selected
        if choice_index != len(expansion.selections) or expansion.selected_node != current:
            raise ValueError("tree selected leaf/prefix differs")
        parent = nodes[current]
        for slot, index in enumerate(expansion.attempt_indices):
            if index >= len(attempts):
                raise ValueError("tree missing terminal attempt")
            row = attempts[index]
            if (row.attempt_index, row.expansion_index, row.child_node) != (
                index,
                expansion_index,
                len(nodes),
            ):
                raise ValueError("tree child/attempt identity differs")
            ordinal = (history.round_index - 1) * 800 + expansion_index * 8 + slot
            prefix_size, _ = replay_tree_path(
                unit,
                root,
                row.path,
                seed=ensemble.seed,
                prefix=parent["prefix"],
                suffix_ordinal=ordinal,
            )
            if row.child_prefix_steps != prefix_size:
                raise ValueError("tree active edge boundary differs")
            row.posterior.validate(ensemble.context)
            key = sequence_id(row.path.endpoint)
            if key in known_scores and known_scores[key] != row.posterior:
                raise ValueError("tree repeated posterior scores differ")
            known_scores[key] = row.posterior
            reason = (
                "generator_training_overlap"
                if key in ensemble._training_ids
                else "previously_charged"
                if key in charged
                else "generated_duplicate"
                if key in seen
                else "cheap_constraints_failed"
                if not row.posterior.feasible
                else None
            )
            first = seen.setdefault(key, index)
            if (row.rejection_reason, row.first_attempt_index) != (reason, first):
                raise ValueError("tree exclusion/duplicate replay differs")
            if reason is None:
                eligible[unit.triple].append(index)
            retained = _front_order(
                attempts, eligible[unit.triple], ensemble.context, all_layers=False
            )[:20]
            if (
                row.buffer_after_attempt_indices != retained
                or row.conditional_retention_probability != float(index in retained)
            ):
                raise ValueError("tree deterministic retention replay differs")
            child = dict(
                level=unit.model.config.levels - prefix_size,
                prefix=row.path.transitions[:prefix_size],
                parent=current,
                children=[],
                visits=0,
                total=np.zeros(2),
                edge=math.fsum(
                    step.draw.log_probability
                    for step in row.path.transitions[len(parent["prefix"]) : prefix_size]
                ),
            )
            nodes[current]["children"].append(len(nodes))
            nodes.append(child)
            ancestor = len(nodes) - 1
            reward = np.asarray(row.posterior.objectives) if reason is None else np.zeros(2)
            while ancestor is not None:
                nodes[ancestor]["visits"] += 1
                nodes[ancestor]["total"] += reward
                ancestor = nodes[ancestor]["parent"]
            expected_attempt += 1
    if cursor != len(generation.expansions) or expected_attempt != len(attempts):
        raise ValueError("tree hidden extra expansion/attempt differs")
    expected_buffers = []
    for unit in ensemble._units:
        indices = _front_order(attempts, eligible[unit.triple], ensemble.context, all_layers=False)[
            :20
        ]
        if indices:
            logweights = tuple(
                ensemble.context.scalarize(attempts[index].posterior.objectives) / 0.1
                + attempts[index].path.reference_log_probability
                - attempts[index].path.behavior_log_probability
                for index in indices
            )
            weights, ess = normalized_tree_weights(logweights)
            maximum = max(weights)
            expected_buffers.append(
                TreeReplayBuffer(
                    unit.triple,
                    indices,
                    logweights,
                    weights,
                    ess,
                    maximum,
                    ess / len(weights) >= 0.20 and maximum <= 0.05,
                )
            )
        else:
            expected_buffers.append(TreeReplayBuffer(unit.triple, (), (), (), 0.0, 0.0, False))
    if generation.replay_buffers != tuple(expected_buffers):
        raise ValueError("tree offpolicy weights/buffer reconstruction differs")
    ranked = _front_order(
        attempts,
        tuple(i for i, row in enumerate(attempts) if row.rejection_reason is None),
        ensemble.context,
        all_layers=True,
    )[:256]
    shortlist = tuple(attempts[i].path.endpoint for i in ranked)
    status = "complete" if len(shortlist) == 256 else "bounded_search_underfill"
    if (generation.shortlisted_sequences, generation.status) != (shortlist, status):
        raise ValueError("tree shortlist/status reconstruction differs")
    for unit in ensemble._units:
        unit.check()

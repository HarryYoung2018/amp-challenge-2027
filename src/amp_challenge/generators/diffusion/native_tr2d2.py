"""Actual bounded native tree collection and off-policy neural distillation.

No oracle port, spectral features, Thompson direction, KG, or production switch.
All numerical defaults were declared before implementation in the pinned TOML.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_baseline_operators import (
    NativeBaselineStep,
    NormalizedObjectiveContext,
    _NativeUnit,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _seed,
    _validated_transition_kernels,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    read_native_posterior,
)
from amp_challenge.generators.diffusion.native_tree_records import (
    TR2D2_CONFIG_SHA256,
    NativeTreeGeneration,
    TreeAttempt,
    TreeExpansion,
    TreePath,
    TreeReplayBuffer,
    TreeRoot,
    TreeSelection,
    TreeTransition,
    normalized_tree_weights,
    pareto_indices,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


@dataclass
class _Node:
    tokens: np.ndarray
    level: int
    prefix: tuple[TreeTransition, ...]
    parent: int | None
    children: list[int] = field(default_factory=list)
    visits: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(2))
    edge_log_probability: float = 0.0


def _expandable(nodes: list[_Node], index: int) -> bool:
    node = nodes[index]
    if node.level == 0:
        return False
    return not node.children or any(_expandable(nodes, child) for child in node.children)


def _select(nodes: list[_Node], *, seed: int, ordinal: int, triple: str):
    current, selections = 0, []
    if not _expandable(nodes, 0):
        return None, ()
    while nodes[current].children:
        node = nodes[current]
        eligible = tuple(child for child in node.children if _expandable(nodes, child))
        scores = tuple(
            tuple(
                map(
                    float,
                    nodes[child].total / nodes[child].visits
                    + 0.1
                    * math.exp(nodes[child].edge_log_probability)
                    * math.sqrt(node.visits)
                    / (1 + nodes[child].visits),
                )
            )
            for child in eligible
        )
        frontier = tuple(eligible[index] for index in pareto_indices(scores))
        selected = int(_seed(seed, ordinal, "tr2d2-select-" + triple, current).choice(frontier))
        selections.append(
            TreeSelection(current, eligible, scores, frontier, selected, -math.log(len(frontier)))
        )
        current = selected
    return current, tuple(selections)


def _root(unit, *, seed: int, round_index: int):
    index = int(
        _seed(seed, round_index, "tr2d2-root-length-" + unit.triple, 0).integers(
            len(unit.sequences)
        )
    )
    anchor = unit.sequences[index]
    length = len(anchor)
    length_mass = sum(len(seq) == length for seq in unit.sequences) / len(unit.sequences)
    tokens = (
        PeptideVocabulary().encode((anchor,), max_length=unit.model.config.max_length).tokens[0]
    )
    tokens[:length] = MASK_TOKEN_INDEX
    return (
        TreeRoot(
            unit.triple,
            length,
            math.log(length_mass),
            sequence_id(anchor),
            unit.policy_sha256,
            unit.reference_sha256,
        ),
        [_Node(tokens, unit.model.config.levels, (), None)],
    )


def _rollout_children(unit, root, node, *, seed: int, ordinals: tuple[int, ...]):
    """One active edge per child followed by a real batched native rollout.

    Prefixes were sampled earlier under this same generation-frozen behavior.
    Every draw records its own ordinal; reused prefixes do not get re-randomized.
    """
    tokens = np.stack([node.tokens.copy() for _ in ordinals])
    paths = [list(node.prefix) for _ in ordinals]
    children = [None] * len(ordinals)
    for level in range(node.level, 0, -1):
        states = tuple(NativeTransitionState(row, root.length, level) for row in tokens)
        kernels = _validated_transition_kernels(unit.model, states, NATIVE_ENDPOINT_DEFAULTS)
        references = _validated_transition_kernels(unit.reference, states, NATIVE_ENDPOINT_DEFAULTS)
        for row, (kernel, reference) in enumerate(zip(kernels, references, strict=True)):
            draw = kernel.sample(_seed(seed, ordinals[row], "tr2d2-commit-" + unit.triple, level))
            before = _json_hash(tokens[row].tolist())
            tokens[row, list(draw.positions)] = draw.residues
            paths[row].append(
                TreeTransition(
                    level,
                    ordinals[row],
                    before,
                    _json_hash(tokens[row].tolist()),
                    draw,
                    reference.log_probability(draw.positions, draw.residues),
                )
            )
            if children[row] is None and kernel.commit_count > 0:
                children[row] = _Node(
                    tokens[row].copy(),
                    level - 1,
                    tuple(paths[row]),
                    None,
                    edge_log_probability=math.fsum(
                        step.draw.log_probability for step in paths[row][len(node.prefix) :]
                    ),
                )
    if any(child is None for child in children):
        raise ValueError("tree expansion has no active transition")
    endpoints = PeptideVocabulary().decode(tokens)
    traces = tuple(
        TreePath(
            endpoint,
            tuple(path),
            math.fsum(step.draw.log_probability for step in path),
            math.fsum(step.reference_log_probability for step in path),
        )
        for endpoint, path in zip(endpoints, paths, strict=True)
    )
    return children, traces


def _rank(attempts, indices, context, *, layers: bool):
    remaining, result = list(indices), []
    while remaining:
        front = [
            remaining[index]
            for index in pareto_indices(
                tuple(attempts[index].posterior.objectives for index in remaining)
            )
        ]
        front.sort(
            key=lambda index: (
                -context.scalarize(attempts[index].posterior.objectives),
                sequence_id(attempts[index].path.endpoint),
            )
        )
        result.extend(front)
        if not layers:
            break
        used = set(front)
        remaining = [index for index in remaining if index not in used]
    return tuple(result)


def _buffer_record(triple, attempts, indices, context):
    if not indices:
        return TreeReplayBuffer(triple, (), (), (), 0.0, 0.0, False)
    logweights = tuple(
        context.scalarize(attempts[index].posterior.objectives) / 0.1
        + attempts[index].path.reference_log_probability
        - attempts[index].path.behavior_log_probability
        for index in indices
    )
    weights, ess = normalized_tree_weights(logweights)
    maximum = max(weights)
    return TreeReplayBuffer(
        triple,
        tuple(indices),
        logweights,
        weights,
        ess,
        maximum,
        ess / len(weights) >= 0.20 and maximum <= 0.05,
    )


def _collect(
    units,
    history,
    context,
    evaluator,
    binding,
    training_ids,
    *,
    feature_batching_sha256=None,
    checkpoint=None,
    retain=None,
):
    forest = {
        unit.triple: _root(unit, seed=history.seed, round_index=history.round_index)
        for unit in units
    }
    offset = int(_json_hash([history.seed, "tr2d2-equal-mixture-v1"])[:16], 16) % len(units)
    charged = {sequence_id(row.sequence) for row in history.observations}
    attempts, expansions, seen, known_scores = [], [], {}, {}
    eligible_by_unit = {unit.triple: [] for unit in units}
    retained = {unit.triple: () for unit in units}
    grouped = feature_batching_sha256 is not None
    if grouped:
        from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
            GroupedNativePosterior,
            check_batching_configuration,
        )

        check_batching_configuration(feature_batching_sha256)

    def prepare(iteration, slot):
        unit = units[(slot + offset) % len(units)]
        root, nodes = forest[unit.triple]
        expansion_index = iteration * len(units) + slot
        select_ordinal = (history.round_index - 1) * 100 + expansion_index
        if grouped and checkpoint is not None:
            checkpoint("before_expansion_preparation")
        selected, selections = _select(
            nodes, seed=history.seed, ordinal=select_ordinal, triple=unit.triple
        )
        if selected is None:
            return None
        ordinals = tuple(
            (history.round_index - 1) * 800 + expansion_index * 8 + child for child in range(8)
        )
        descriptor = dict(
            expansion_index=expansion_index,
            triple=unit.triple,
            iteration=iteration,
            selected_node=selected,
            ordinals=ordinals,
        )
        if grouped and retain is not None:
            retain("preparing_expansion", descriptor)
        children, paths = _rollout_children(
            unit, root, nodes[selected], seed=history.seed, ordinals=ordinals
        )
        if grouped and retain is not None:
            retain("prepared_expansion", {**descriptor, "paths": paths})
        if grouped and checkpoint is not None:
            checkpoint("after_expansion_preparation")
        return unit, nodes, expansion_index, selected, selections, children, paths

    def apply(item, batch, iteration):
        unit, nodes, expansion_index, selected, selections, children, paths = item
        indices = tuple(range(len(attempts), len(attempts) + 8))
        expansions.append(
            TreeExpansion(
                expansion_index,
                unit.triple,
                iteration,
                selected,
                selections,
                indices,
                batch.receipt_sha256,
            )
        )
        for child, path, score in zip(children, paths, batch.scores, strict=True):
            index = len(attempts)
            key = sequence_id(path.endpoint)
            if key in known_scores and known_scores[key] != score:
                raise ValueError("frozen posterior changed scores for repeated sequence")
            known_scores[key] = score
            reason = (
                "generator_training_overlap"
                if key in training_ids
                else "previously_charged"
                if key in charged
                else "generated_duplicate"
                if key in seen
                else "cheap_constraints_failed"
                if not score.feasible
                else None
            )
            first = seen.setdefault(key, index)
            child_id = len(nodes)
            child.parent = selected
            nodes.append(child)
            nodes[selected].children.append(child_id)
            attempts.append(
                TreeAttempt(
                    index,
                    expansion_index,
                    child_id,
                    len(child.prefix),
                    path,
                    score,
                    reason,
                    first,
                    (),
                    0.0,
                )
            )
            if reason is None:
                eligible_by_unit[unit.triple].append(index)
            retained[unit.triple] = _rank(
                attempts, eligible_by_unit[unit.triple], context, layers=False
            )[:20]
            attempts[-1] = replace(
                attempts[-1],
                buffer_after_attempt_indices=retained[unit.triple],
                conditional_retention_probability=float(index in retained[unit.triple]),
            )
            reward = np.asarray(score.objectives) if reason is None else np.zeros(2)
            ancestor = child_id
            while ancestor is not None:
                nodes[ancestor].visits += 1
                nodes[ancestor].total += reward
                ancestor = nodes[ancestor].parent

    for iteration in range(10):
        if not grouped:
            for slot in range(len(units)):
                item = prepare(iteration, slot)
                if item is None:
                    continue
                batch = read_native_posterior(
                    evaluator,
                    tuple(path.endpoint for path in item[-1]),
                    expected_binding=binding,
                    context=context,
                )
                apply(item, batch, iteration)
            continue
        pending = [
            item for slot in range(len(units)) if (item := prepare(iteration, slot)) is not None
        ]
        if not pending:
            continue
        groups = tuple(tuple(path.endpoint for path in item[-1]) for item in pending)
        if evaluator.binding != binding:
            raise ValueError("TR2 grouped posterior binding changed before acquisition")
        result = evaluator.evaluate_groups(groups)
        if (
            evaluator.binding != binding
            or type(result) is not GroupedNativePosterior
            or len(result.groups) != len(pending)
        ):
            raise ValueError("TR2 grouped posterior inventory differs")
        result.__post_init__()
        for item, sequences, batch in zip(pending, groups, result.groups, strict=True):
            if batch.sequence_ids != tuple(map(sequence_id, sequences)) or len(batch.scores) != 8:
                raise ValueError("TR2 grouped posterior slice order differs")
            for score in batch.scores:
                score.validate(context)
            if checkpoint is not None:
                checkpoint("before_expansion_application")
            apply(item, batch, iteration)
            if retain is not None:
                retain("applied_expansion", item[2])
            if checkpoint is not None:
                checkpoint("after_expansion_application")
    buffers = tuple(
        _buffer_record(unit.triple, attempts, retained[unit.triple], context) for unit in units
    )
    shortlist = _rank(
        attempts,
        tuple(index for index, row in enumerate(attempts) if row.rejection_reason is None),
        context,
        layers=True,
    )[:256]
    generation = NativeTreeGeneration(
        history.round_index,
        history.seed,
        history.sha256,
        binding,
        tuple(forest[unit.triple][0] for unit in units),
        tuple(expansions),
        tuple(attempts),
        buffers,
        tuple(attempts[index].path.endpoint for index in shortlist),
        "complete" if len(shortlist) == 256 else "bounded_search_underfill",
    )
    generation.check_output_budget()
    return generation


@dataclass(frozen=True, slots=True)
class NativeTreeAdvance:
    triple: str
    round_index: int
    history_sha256: str
    previous_generation_sha256: str | None
    status: str
    steps: tuple[NativeBaselineStep, ...]
    old_policy_sha256: str
    new_policy_sha256: str
    saved_buffer: TreeReplayBuffer | None
    configuration_sha256: str = TR2D2_CONFIG_SHA256
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


def _update_unit(unit, history, context, generation):
    unit.check()
    buffer = (
        None
        if generation is None
        else next(row for row in generation.replay_buffers if row.triple == unit.triple)
    )
    old_sha, working, steps = unit.policy_sha256, unit.model, []
    status = "unchanged_by_declared_schedule"
    if generation is not None and history.round_index < 29:
        if not buffer.attempt_indices:
            status = "empty_pareto_buffer_no_update"
        else:
            sequences = tuple(
                generation.attempts[index].path.endpoint for index in buffer.attempt_indices
            )
            # Saved posterior rewards/path weights do NOT change when theta changes.
            weights = np.asarray(buffer.normalized_weights)
            replay_seed = int(_json_hash([history.seed, unit.triple, "tr2d2-training"])[:16], 16)
            for step in range(4):
                ordinal = history.round_index * 4 + step
                replay = build_weighted_replay(
                    working,
                    sequences,
                    weights,
                    context_id=context.context_sha256,
                    seed=replay_seed,
                    ordinal=ordinal,
                )
                probes = build_weighted_replay(
                    unit.reference,
                    sequences,
                    weights,
                    context_id=context.context_sha256,
                    seed=replay_seed,
                    ordinal=ordinal,
                    active_probes=True,
                )
                direction = propose_weighted_direction(working, replay)
                candidate = endpoint_candidate(working, direction)
                diagnostics = weighted_anchor_diagnostics(
                    working, candidate, unit.reference, probes, NATIVE_ENDPOINT_DEFAULTS
                )
                gradient = hashlib.sha256()
                for name, tensor in direction.gradients:
                    gradient.update(name.encode() + b"\0" + tensor.numpy().tobytes())
                after = canonical_model_logical_hash(candidate)
                steps.append(
                    NativeBaselineStep(
                        replay,
                        probes,
                        direction.base_model_sha256,
                        after,
                        gradient.hexdigest(),
                        direction.objective_before,
                        direction.gradient_norm_before_clip,
                        True,
                        0,
                        False,
                        diagnostics,
                    )
                )
                working = candidate
            status = "four_frozen_buffer_offpolicy_native_updates"
    new = copy.copy(unit)
    new.model, new.policy_sha256 = working, canonical_model_logical_hash(working)
    return new, NativeTreeAdvance(
        unit.triple,
        history.round_index,
        history.sha256,
        None if generation is None else generation.sha256,
        status,
        tuple(steps),
        old_sha,
        new.policy_sha256,
        buffer,
    )


class NativeTR2D2Ensemble:
    """Research-only native operator; outer controller owns scientific authority."""

    def __init__(
        self,
        initializations,
        generator_sequences,
        context,
        *,
        run_id,
        seed,
        oracle_bundle_sha256,
        feature_source_sha256,
        evaluator_source_sha256,
    ):
        if type(context) is not NormalizedObjectiveContext:
            raise TypeError("tree normalized objective context differs")
        context.__post_init__()
        if (
            type(run_id) is not str
            or not run_id
            or type(seed) is not int
            or not 0 <= seed < 2**63
            or any(
                not hash_string(value)
                for value in (oracle_bundle_sha256, feature_source_sha256, evaluator_source_sha256)
            )
            or not 1 <= len(initializations) <= 10
            or len(initializations) != len(generator_sequences)
        ):
            raise ValueError("tree fixed source/run/initializer inventory differs")
        units = tuple(
            _NativeUnit(init, sequences)
            for init, sequences in zip(initializations, generator_sequences, strict=True)
        )
        if (
            len({unit.triple for unit in units}) != len(units)
            or len({unit.model.config for unit in units}) != 1
        ):
            raise ValueError("tree initializer triples/architectures differ")
        self._units = tuple(sorted(units, key=lambda unit: unit.triple))
        self.protocol_ten_checkpoint_mixture = tuple(unit.triple for unit in self._units) == TRIPLES
        self.context, self.run_id, self.seed = context, run_id, seed
        self.oracle_bundle_sha256 = oracle_bundle_sha256
        self.feature_source_sha256, self.evaluator_source_sha256 = (
            feature_source_sha256,
            evaluator_source_sha256,
        )
        self._training_ids = frozenset(sequence_id(seq) for unit in units for seq in unit.sequences)
        self._history, self._generation, self._receipts = None, None, ()
        self._generation_feature_batching_sha256 = None

    @property
    def policy_identities(self):
        return tuple((unit.triple, unit.policy_sha256) for unit in self._units)

    def advance(self, history, *, expected_previous_head_sha256):
        if type(history) is not VerifiedHistorySnapshot:
            raise TypeError("tree requires shared verified charged history")
        history.__post_init__()
        if (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
            history.previous_wave_head_sha256,
        ) != (
            self.run_id,
            self.seed,
            self.context.context_sha256,
            self.oracle_bundle_sha256,
            expected_previous_head_sha256,
        ):
            raise ValueError("tree history run/context/source/head differs")
        if any(sequence_id(row.sequence) in self._training_ids for row in history.observations):
            raise ValueError("tree charged history overlaps generator training")
        if self._history is not None and history.sha256 == self._history.sha256:
            return self._receipts
        expected_round = 1 if self._history is None else self._history.round_index + 1
        if history.round_index != expected_round:
            raise ValueError("tree history skipped/repeated a wave")
        if self._history is not None and (
            self._generation is None
            or history.observations[: len(self._history.observations)] != self._history.observations
        ):
            raise ValueError("tree history rewrites prefix or precedes sealed generation")
        if not history.complete:
            return ()
        replacements = tuple(
            _update_unit(unit, history, self.context, self._generation) for unit in self._units
        )
        self._units = tuple(pair[0] for pair in replacements)
        self._receipts = tuple(pair[1] for pair in replacements)
        self._history, self._generation = history, None
        return self._receipts

    def collect(
        self,
        evaluator,
        *,
        expected_posterior_sha256,
        feature_batching_sha256=None,
        checkpoint=None,
        retain=None,
    ):
        if self._history is None or self._history.round_index == 29:
            raise ValueError("tree collection needs a complete nonterminal history")
        binding = FrozenNativePosteriorBinding(
            self._history.sha256,
            self.context.context_sha256,
            expected_posterior_sha256,
            self.feature_source_sha256,
            self.evaluator_source_sha256,
        )
        if evaluator.binding != binding:
            raise ValueError("tree evaluator does not match fixed code/features/history")
        for unit in self._units:
            unit.check()
        if self._generation is not None:
            if self._generation.evaluator_binding != binding:
                raise ValueError("tree generation cannot change its frozen posterior")
            if self._generation_feature_batching_sha256 != feature_batching_sha256:
                raise ValueError("tree generation cannot change its feature-batching mode")
            return self._generation
        generation = _collect(
            self._units,
            self._history,
            self.context,
            evaluator,
            binding,
            self._training_ids,
            feature_batching_sha256=feature_batching_sha256,
            checkpoint=checkpoint,
            retain=retain,
        )
        for unit in self._units:
            unit.check()
        self._generation = generation
        self._generation_feature_batching_sha256 = feature_batching_sha256
        return generation

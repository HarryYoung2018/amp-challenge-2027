"""Executed frozen-native directional tree/refinement adapter, research only.

Outer code owns authentic posterior fitting, charged oracle transport, reserve
composition, compliance, persistent publication and hard process deadlines.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    _NativeUnit,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _state_contract,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_mp2d_math import (
    CONFIG_SHA256,
    DIRECTIONS,
    angular_decision,
    archive_rewards,
    commit,
    event_draw,
    greedy_completions,
    keyed_rng,
    mpi_values,
    pareto_indices,
    remask,
    stable_softmax,
    vector_ucb,
)
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorScore,
    read_native_posterior,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def source_identities() -> dict[str, str]:
    root = Path(__file__).resolve().parents[4]
    paths = [
        "configs/diffusion/native_mp2d_operators_v1.toml",
        *(
            "src/amp_challenge/generators/diffusion/" + name + ".py"
            for name in (
                "native_mp2d_math",
                "native_mp2d_operators",
                "native_search_posterior",
                "native_baseline_operators",
                "native_initialization",
                "native_proposals",
                "native_endpoint",
                "subset_kernel",
                "categorical",
                "model",
            )
        ),
    ]
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    if result[paths[0]] != CONFIG_SHA256:
        raise ValueError("MP2D predeclared configuration bytes differ")
    return result


def semantic_history(history):
    return _json_hash(
        [
            CONFIG_SHA256,
            history.seed,
            history.round_index,
            history.objective_context_sha256,
            [
                (row.charge_index, row.sequence, row.status, row.objectives)
                for row in history.observations
            ],
        ]
    )


@dataclass(frozen=True, slots=True)
class NativeMP2DWave:
    record_json: str
    sha256: str
    candidates: tuple[str, ...]
    stop_reason: str
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


class _Stop(Exception):
    pass


@dataclass
class _Node:
    tokens: np.ndarray
    level: int
    endpoint: str
    score: NativePosteriorScore
    prior: float = 1.0
    children: list[int] = field(default_factory=list)
    visits: int = 0
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(2))
    blocked: bool = False


@dataclass
class _Tree:
    ordinal: int
    unit: object
    seed: str
    direction_index: int
    noise: int
    nodes: list[_Node]
    archive: dict[str, NativePosteriorScore]
    angle: float = 45.0
    ema: float = 0.3


class _Runner:
    def __init__(self, units, history, context, evaluator, binding, deadline, clock, replay_stop):
        self.units, self.history, self.context = units, history, context
        self.evaluator, self.binding = evaluator, binding
        self.deadline, self.clock, self.replay_stop = deadline, clock, replay_stop
        self.stream = semantic_history(history)
        self.seed = int(self.stream[:16], 16)
        self.directions = keyed_rng(self.stream, "directions").permutation(64).tolist()
        self.training = set().union(
            *(set(unit.initialization.training_sequence_ids) for unit in units)
        )
        self.charged = {sequence_id(row.sequence) for row in history.observations}
        self.events, self.responses = [], []
        self.scores, self.pool, self.seen = {}, {}, set()
        self.attempts = self.checks = 0
        self.stop_check = None
        self.seed_rows = []
        self.pending = []
        self.pending_bootstrap = []

    def check(self):
        self.checks += 1
        if self.replay_stop == self.checks or (
            self.replay_stop is None and self.clock() >= self.deadline
        ):
            self.stop_check = self.checks
            raise _Stop("partial_deadline_stop")

    def reserve_attempts(self, count):
        self.check()
        if self.attempts + count > 1024:
            raise _Stop("partial_attempt_cap")
        begin = self.attempts
        self.attempts += count
        return begin

    def rejection(self, sequence):
        key = sequence_id(sequence)
        return (
            "generator_training_overlap"
            if key in self.training
            else "previously_charged"
            if key in self.charged
            else None
        )

    def evaluate(self, sequences):
        requested = tuple(
            dict.fromkeys(
                seq for seq in sequences if seq not in self.scores and self.rejection(seq) is None
            )
        )
        if len(self.scores) + len(requested) > 1024:
            raise _Stop("partial_posterior_cap")
        for start in range(0, len(requested), 128):
            self.check()
            chunk = requested[start : start + 128]
            batch = read_native_posterior(
                self.evaluator, chunk, expected_binding=self.binding, context=self.context
            )
            self.responses.append({"sequences": list(chunk), "batch": asdict(batch)})
            self.scores.update(zip(chunk, batch.scores, strict=True))
            self.check()

    def bootstrap(self):
        offset = int(_json_hash([self.history.seed, "mp2d-equal-mixture-v1"])[:16], 16) % len(
            self.units
        )
        while len(self.seed_rows) < 8:
            count = 8 - len(self.seed_rows)
            begin = self.reserve_attempts(count)
            planned = []
            for index in range(begin, begin + count):
                unit = self.units[
                    (offset + (self.history.round_index - 1) * 8 + index) % len(self.units)
                ]
                parent = unit.sequences[
                    int(keyed_rng(self.stream, "length", index).integers(len(unit.sequences)))
                ]
                planned.append((unit, parent, index))
                self.pending_bootstrap.append(
                    {"attempt": index, "triple": unit.triple, "parent": parent, "trace": None}
                )
            traces = {}
            for unit in self.units:
                group = [row for row in planned if row[0] is unit]
                if group:
                    self.check()
                    sampled = sample_native_proposals(
                        unit.model,
                        tuple(row[1] for row in group),
                        start_levels=(unit.model.config.levels,) * len(group),
                        seed=self.seed,
                        ordinals=tuple(row[2] for row in group),
                    )
                    traces.update(
                        (row[2], trace) for row, trace in zip(group, sampled, strict=True)
                    )
                    for row, trace in zip(group, sampled, strict=True):
                        next(item for item in self.pending_bootstrap if item["attempt"] == row[2])[
                            "trace"
                        ] = asdict(trace)
                    self.check()
            for unit, _, index in planned:
                trace = traces[index]
                reason = self.rejection(trace.endpoint) or (
                    "generated_duplicate" if trace.endpoint in self.seen else None
                )
                self.seen.add(trace.endpoint)
                self.events.append(
                    {
                        "kind": "bootstrap",
                        "attempt": index,
                        "triple": unit.triple,
                        "trace": asdict(trace),
                        "rejection": reason,
                    }
                )
                if reason is None:
                    self.seed_rows.append((unit, trace.endpoint))
            self.pending_bootstrap = []
        self.evaluate([seq for _, seq in self.seed_rows])
        for _, seq in self.seed_rows:
            if self.scores[seq].feasible:
                self.pool[seq] = self.scores[seq]

    def start_trees(self, seeds, pass_index):
        trees = []
        for index, (unit, seq) in enumerate(seeds):
            self.check()
            ordinal = pass_index * 8 + index
            direction = self.directions[((self.history.round_index - 1) * 16 + ordinal) % 64]
            noise = int(
                keyed_rng(self.stream, "noise", ordinal).integers(unit.model.config.levels + 1)
            )
            tokens, positions, logp = remask(unit.model, seq, noise, self.stream, ordinal)
            node = _Node(tokens, noise, seq, self.scores[seq], blocked=noise == 0)
            archive = {seq: node.score} if node.score.feasible else {}
            trees.append(_Tree(ordinal, unit, seq, direction, noise, [node], archive))
            self.events.append(
                {
                    "kind": "root",
                    "tree": ordinal,
                    "triple": unit.triple,
                    "seed": seq,
                    "direction_index": direction,
                    "noise": noise,
                    "positions": positions,
                    "remask_log_probability": logp,
                    "tokens_sha256": _json_hash(tokens.tolist()),
                }
            )
        return trees

    def select(self, tree, expansion):
        for node in reversed(tree.nodes):
            if node.children and all(tree.nodes[index].blocked for index in node.children):
                node.blocked = True
        path = [0]
        while True:
            node = tree.nodes[path[-1]]
            available = [index for index in node.children if not tree.nodes[index].blocked]
            if not available:
                return None if node.blocked or node.children else path
            values = vector_ucb(
                [tree.nodes[index].rewards for index in available],
                [tree.nodes[index].visits for index in available],
                [tree.nodes[index].prior for index in available],
                node.visits,
            )
            front = pareto_indices(values, deduplicate=False)
            selected = int(
                keyed_rng(self.stream, "select", tree.ordinal, expansion, len(path)).choice(front)
            )
            path.append(available[selected])

    def expand(self, trees, expansion):
        planned, rollout_rows = [], []
        for tree in trees:
            self.check()
            path = self.select(tree, expansion)
            if path is None:
                continue
            node = tree.nodes[path[-1]]
            level = node.level
            skipped = []
            while level:
                state = NativeTransitionState(node.tokens, len(node.endpoint), level)
                if _state_contract(state, tree.unit.model.config)[1]:
                    break
                skipped.append(level)
                level -= 1
            if not level:
                node.blocked = True
                self.events.append(
                    {
                        "kind": "empty_leaf",
                        "tree": tree.ordinal,
                        "expansion": expansion,
                        "path": path,
                        "skipped_levels": skipped,
                    }
                )
                continue
            begin = self.reserve_attempts(8)
            kernel = _validated_transition_kernels(
                tree.unit.model, (state,), NATIVE_ENDPOINT_DEFAULTS
            )[0]
            children = []
            for child in range(8):
                draw = event_draw(
                    kernel, rng=keyed_rng(self.stream, "child", tree.ordinal, expansion, child)
                )
                tokens, action = commit(node.tokens, level, draw, "gumbel_subset")
                children.append({"attempt": begin + child, "action": action, "tokens": tokens})
                self.pending.append(
                    {
                        "attempt": begin + child,
                        "tree": tree.ordinal,
                        "expansion": expansion,
                        "triple": tree.unit.triple,
                        "action": action,
                        "completion": None,
                    }
                )
                rollout_rows.append((tree.unit.model, tokens, len(node.endpoint), level - 1))
            planned.append((tree, path, level, skipped, children))
        completions = iter(greedy_completions(rollout_rows, check=self.check))
        pending = []
        for _, _, _, _, children in planned:
            for child in children:
                child["endpoint"], child["rollout"] = next(completions)
                next(row for row in self.pending if row["attempt"] == child["attempt"])[
                    "completion"
                ] = {"endpoint": child["endpoint"], "rollout": child["rollout"]}
                reason = self.rejection(child["endpoint"]) or (
                    "generated_duplicate" if child["endpoint"] in self.seen else None
                )
                self.seen.add(child["endpoint"])
                child["rejection"] = reason
                if reason is None:
                    pending.append(child["endpoint"])
        self.evaluate(pending)
        self.check()
        for tree, path, level, skipped, children in planned:
            parent = tree.nodes[path[-1]]
            scores = [
                self.scores.get(
                    child["endpoint"], NativePosteriorScore(parent.score.objectives, False)
                )
                for child in children
            ]
            feasible = tuple(
                child["rejection"] is None and score.feasible
                for child, score in zip(children, scores, strict=True)
            )
            decision = angular_decision(
                parent.score.objectives,
                [score.objectives for score in scores],
                feasible,
                DIRECTIONS[tree.direction_index],
                tree.angle,
                tree.ema,
            )
            rewards = archive_rewards(
                [scores[index].objectives for index in decision.retained],
                [score.objectives for score in tree.archive.values()],
            )
            total = rewards.sum(axis=0) if len(rewards) else np.zeros(2)
            tree.angle, tree.ema = decision.new_angle, decision.new_ema
            new_indices = []
            for index in decision.retained:
                child, score = children[index], scores[index]
                new_indices.append(len(tree.nodes))
                tree.nodes.append(
                    _Node(
                        child["tokens"],
                        level - 1,
                        child["endpoint"],
                        score,
                        math.exp(child["action"]["base_log_probability"]),
                        blocked=level == 1,
                    )
                )
                tree.archive.setdefault(child["endpoint"], score)
                self.pool.setdefault(child["endpoint"], score)
            parent.children.extend(new_indices)
            if not new_indices:
                parent.blocked = True
            archive_rows = list(tree.archive.items())
            tree.archive = {
                archive_rows[index][0]: archive_rows[index][1]
                for index in pareto_indices([score.objectives for _, score in archive_rows])
            }
            for index in path:
                tree.nodes[index].rewards += total
                tree.nodes[index].visits += 1
            self.events.append(
                {
                    "kind": "expansion",
                    "tree": tree.ordinal,
                    "expansion": expansion,
                    "path": path,
                    "skipped_levels": skipped,
                    "children": [
                        {key: value for key, value in child.items() if key != "tokens"}
                        for child in children
                    ],
                    "decision": asdict(decision),
                    "rewards": rewards.tolist(),
                    "new_node_indices": new_indices,
                    "archive": list(tree.archive),
                    "ancestor_visits": [tree.nodes[index].visits for index in path],
                    "ancestor_rewards": [tree.nodes[index].rewards.tolist() for index in path],
                }
            )
        self.pending = []

    def refine(self, trees):
        result = []
        for tree in trees:
            self.check()
            choices = list(tree.archive)
            if tree.noise == 0 or not choices:
                selected, values = tree.seed, []
            else:
                delta = (
                    np.asarray([tree.archive[seq].objectives for seq in choices])
                    - self.scores[tree.seed].objectives
                )
                values = mpi_values(delta, DIRECTIONS[tree.direction_index], tree.noise).tolist()
                selected = choices[
                    int(
                        keyed_rng(self.stream, "refine", tree.ordinal).choice(
                            len(choices), p=stable_softmax(values)
                        )
                    )
                ]
            result.append((tree.unit, selected))
            self.events.append(
                {
                    "kind": "refinement",
                    "tree": tree.ordinal,
                    "choices": choices,
                    "mpi": values,
                    "selected": selected,
                }
            )
        return result

    def run(self):
        status = "complete_declared_wave_search_not_campaign"
        try:
            self.bootstrap()
            seeds = self.seed_rows
            for pass_index in range(2):
                trees = self.start_trees(seeds, pass_index)
                for expansion in range(4):
                    self.expand(trees, expansion)
                seeds = self.refine(trees)
            self.check()
        except _Stop as error:
            status = str(error)
        # Ordered Pareto layers, equal-weight means within a layer, sequence-ID
        # tie break. Only gate-retained feasible candidates plus feasible seeds.
        remaining, candidates = list(self.pool), []
        while remaining and len(candidates) < 256:
            front = pareto_indices(
                [self.pool[seq].objectives for seq in remaining], deduplicate=False
            )
            layer = sorted(
                (remaining[index] for index in front),
                key=lambda seq: (
                    -self.context.scalarize(self.pool[seq].objectives),
                    sequence_id(seq),
                ),
            )
            candidates.extend(layer[: 256 - len(candidates)])
            selected = set(front)
            remaining = [seq for index, seq in enumerate(remaining) if index not in selected]
        return status, tuple(candidates)


def run_native_mp2d(
    initializations,
    corpora,
    history,
    context,
    evaluator,
    *,
    expected_binding,
    deadline: float,
    clock=time.monotonic,
    _replay_stop=None,
) -> NativeMP2DWave:
    """One actual native wave, no oracle transport and no controller mutation.

    deadline is the outer scientific monotonic deadline, never a fresh 7200s.
    Local enforcement checks bounded stages; caller must preempt blocked calls.
    _replay_stop is internal deterministic reconstruction of a recorded time stop.
    """
    started = clock()
    if not math.isfinite(deadline):
        raise ValueError("explicit finite outer deadline required")
    if type(history) is not VerifiedHistorySnapshot:
        raise TypeError("MP2D needs the shared verified history record")
    history.__post_init__()
    if not history.complete or history.round_index == 29:
        raise ValueError("MP2D needs complete pre-adaptive charged history")
    if (
        type(context) is not NormalizedObjectiveContext
        or history.objective_context_sha256 != context.context_sha256
    ):
        raise ValueError("MP2D objective context differs")
    context.__post_init__()
    if (
        type(expected_binding) is not FrozenNativePosteriorBinding
        or expected_binding.history_sha256 != history.sha256
        or expected_binding.objective_context_sha256 != context.context_sha256
        or evaluator.binding != expected_binding
    ):
        raise ValueError("MP2D expected frozen posterior/history binding differs")
    expected_binding.__post_init__()
    initializations, corpora = tuple(initializations), tuple(corpora)
    if (
        not 1 <= len(initializations) <= 10
        or len(initializations) != len(corpora)
        or len({item.triple for item in initializations}) != len(initializations)
    ):
        raise ValueError("MP2D initializer inventory differs")
    if tuple(item.triple for item in initializations) != tuple(
        sorted(item.triple for item in initializations)
    ):
        raise ValueError("MP2D initializer order differs")
    sources = source_identities()
    units = tuple(
        _NativeUnit(init, corpus) for init, corpus in zip(initializations, corpora, strict=True)
    )
    runner = _Runner(
        units,
        history,
        context,
        evaluator,
        expected_binding,
        min(deadline, started + 120),
        clock,
        _replay_stop,
    )
    status, candidates = runner.run()
    for unit in units:
        unit.check()
    if evaluator.binding != expected_binding:
        raise ValueError("MP2D posterior binding changed during search")
    if source_identities() != sources:
        raise ValueError("MP2D source changed during search")
    for init in initializations:
        if canonical_model_logical_hash(init.model) != init.checkpoint_logical_sha256:
            raise ValueError("MP2D caller model changed during search")
    if runner.stop_check is None:
        try:
            runner.check()
        except _Stop as error:
            status = str(error)
    payload = {
        "artifact": "native_mp2d_wave_v1",
        "config_sha256": CONFIG_SHA256,
        "history_sha256": history.sha256,
        "posterior": asdict(expected_binding),
        "semantic_stream_sha256": runner.stream,
        "source_identities": sources,
        "initializations": [
            {
                "triple": unit.triple,
                "model_sha256": unit.policy_sha256,
                "checkpoint_sha256": unit.initialization.checkpoint_file_sha256,
                "audit_sha256": unit.initialization.audit_sha256,
                "manifest_sha256": unit.initialization.manifest_sha256,
                "training_ids_sha256": _json_hash(unit.initialization.training_sequence_ids),
            }
            for unit in units
        ],
        "events": runner.events,
        "pending_expansion_attempts": runner.pending,
        "pending_bootstrap_attempts": runner.pending_bootstrap,
        "posterior_responses": runner.responses,
        "attempts": runner.attempts,
        "unique_posterior_rows": len(runner.scores),
        "checks": runner.checks,
        "stop_check": runner.stop_check,
        "candidates": candidates,
        "stop_reason": status,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
        "scope": "posterior_scored_worker_pool_not_oracle_truth_or_private_seat_plan",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(serialized.encode()) > 128 * 1024**2:
        raise ValueError("MP2D wave exceeds bounded publication size")
    return NativeMP2DWave(
        serialized, hashlib.sha256(serialized.encode()).hexdigest(), candidates, status
    )

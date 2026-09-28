"""Equal-ten MP2D-style v2: fixed attempt strata, four score opportunities.

No new policy training, oracle transport, feature authority or production gate.
The frozen v1 math/select/refine primitives are reused, not altered.
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
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_mp2d_math import (
    DIRECTIONS,
    angular_decision,
    archive_rewards,
    commit,
    event_draw,
    greedy_completions,
    keyed_rng,
    pareto_indices,
    remask,
)
from amp_challenge.generators.diffusion.native_mp2d_operators import (
    _Runner,
    _Stop,
    _Tree,
)
from amp_challenge.generators.diffusion.native_mp2d_operators import (
    source_identities as original_source_identities,
)
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorScore,
    read_native_posterior,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

CONFIG_SHA256 = "7dfb024ea7d1bbe1c6396cac995157db1c2169e495b616da7479dfe474030e4e"
CHILD_COUNTS = (11, 12, 12, 12)
COMPLETE = "complete_equal_ten_assigned_wave_not_campaign"
RECEIPT_LIMIT = 128 * 1024**2


def source_identities():
    result = original_source_identities()
    root = Path(__file__).resolve().parents[4]
    config = "configs/diffusion/native_mp2d_equal_ten_v2.toml"
    for path in (
        config,
        "src/amp_challenge/generators/diffusion/native_mp2d_equal_ten.py",
        "src/amp_challenge/generators/diffusion/native_mp2d_equal_ten_verify.py",
        "src/amp_challenge/generators/diffusion/native_mp2d_verify.py",
    ):
        result[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    if result[config] != CONFIG_SHA256:
        raise ValueError("equal-ten predeclared configuration differs")
    return result


def semantic_history(history, eligible_query_ids):
    if (
        type(eligible_query_ids) is not frozenset
        or any(type(value) is not str for value in eligible_query_ids)
        or not eligible_query_ids
        <= {row.query_id for row in history.observations if row.status == "successful"}
    ):
        raise ValueError("equal-ten eligibility must explicitly name successful current charges")
    return _json_hash(
        [
            "native-mp2d-equal-ten-v2",
            CONFIG_SHA256,
            history.seed,
            history.round_index,
            history.objective_context_sha256,
            [
                (
                    row.charge_index,
                    row.sequence,
                    row.status,
                    row.query_id in eligible_query_ids,
                    row.objectives if row.query_id in eligible_query_ids else None,
                )
                for row in history.observations
            ],
        ]
    )


def checkpoint_counts(slots, events):
    result = {}
    for triple in TRIPLES:
        rows = [row for row in slots if row["triple"] == triple]
        tree_ids = {
            event["tree"]
            for event in events
            if event["kind"] == "root" and event["triple"] == triple
        }
        result[triple] = {
            "assigned": len(rows),
            "bootstrap_assigned": sum(row["phase"] == "bootstrap" for row in rows),
            "child_assigned": sum(row["phase"] == "child" for row in rows),
            "native_active": sum(
                row["state"] in ("native_active", "native_completed") for row in rows
            ),
            "native_completed": sum(row["state"] == "native_completed" for row in rows),
            "identity_noop": sum(row["state"] == "blocked_identity" for row in rows),
            "reserved_only": sum(row["state"] == "reserved" for row in rows),
            "root_accepted": sum(
                event["kind"] == "bootstrap"
                and event["triple"] == triple
                and event["root_accepted"]
                for event in events
            ),
            "child_gate_retained": sum(
                len(event["decision"]["retained"])
                for event in events
                if event["kind"] == "expansion" and event["tree"] in tree_ids
            ),
        }
    return result


@dataclass(frozen=True, slots=True)
class EqualTenMP2DWave:
    record_json: str
    sha256: str
    candidates: tuple[str, ...]
    stop_reason: str
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False
    campaign_eligible: bool = False


@dataclass
class _Node:
    tokens: np.ndarray
    level: int
    endpoint: str
    # None means not yet observed. It is never a surrogate numeric mean.
    score: NativePosteriorScore | None
    prior: float = 1.0
    children: list[int] = field(default_factory=list)
    visits: int = 0
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(2))
    blocked: bool = False


class _EqualTenRunner(_Runner):
    def __init__(self, *args, eligible_query_ids):
        super().__init__(*args)
        self.stream = semantic_history(self.history, eligible_query_ids)
        self.seed = int(self.stream[:16], 16)
        self.directions = keyed_rng(self.stream, "directions").permutation(64).tolist()
        permutation = keyed_rng(self.stream, "checkpoint-order").permutation(10)
        self.order = tuple(self.units[int(index)] for index in permutation)
        self.slots, self.score_opportunities, self.cache_origins = [], [], {}
        self.bootstrap_rounds = 0

    def reserve(self, count, phase, stage):
        self.check()
        if self.attempts + count * 10 > 1020:
            raise _Stop("partial_assigned_attempt_cap")
        rows = []
        for local in range(count):
            for unit in self.order:
                row = {
                    "attempt": self.attempts,
                    "triple": unit.triple,
                    "phase": phase,
                    "stage": stage,
                    "local": local,
                    "state": "reserved",
                }
                self.attempts += 1
                self.slots.append(row)
                rows.append(row)
        return rows

    def bootstrap(self):
        roots = {}
        for bootstrap_round in range(55):
            rows = self.reserve(1, "bootstrap", bootstrap_round)
            self.bootstrap_rounds += 1
            self.pending_bootstrap = []
            traces = {}
            for row, unit in zip(rows, self.order, strict=True):
                parent = unit.sequences[
                    int(
                        keyed_rng(self.stream, "length", row["attempt"]).integers(
                            len(unit.sequences)
                        )
                    )
                ]
                self.pending_bootstrap.append(
                    {
                        "attempt": row["attempt"],
                        "triple": unit.triple,
                        "parent": parent,
                        "trace": None,
                    }
                )
            for row, unit, pending in zip(rows, self.order, self.pending_bootstrap, strict=True):
                self.check()
                row["state"] = "native_active"
                trace = sample_native_proposals(
                    unit.model,
                    (pending["parent"],),
                    start_levels=(unit.model.config.levels,),
                    seed=self.seed,
                    ordinals=(row["attempt"],),
                )[0]
                traces[row["attempt"]] = trace
                pending["trace"] = asdict(trace)
                row["state"] = "native_completed"
                self.check()
            for row, unit in zip(rows, self.order, strict=True):
                trace = traces[row["attempt"]]
                reason = self.rejection(trace.endpoint) or (
                    "generated_duplicate" if trace.endpoint in self.seen else None
                )
                self.seen.add(trace.endpoint)
                accepted = unit.triple not in roots and reason is None
                if accepted:
                    roots[unit.triple] = trace.endpoint
                self.events.append(
                    {
                        "kind": "bootstrap",
                        "attempt": row["attempt"],
                        "triple": unit.triple,
                        "trace": asdict(trace),
                        "rejection": reason,
                        "root_accepted": accepted,
                        "already_filled": unit.triple in roots and not accepted,
                    }
                )
            self.pending_bootstrap = []
            if len(roots) == 10:
                self.seed_rows = [(unit, roots[unit.triple]) for unit in self.order]
                return
        raise _Stop("partial_bootstrap_root_cap")

    def start_trees(self, seeds, pass_index):
        trees = []
        for index, (unit, seq) in enumerate(seeds):
            self.check()
            ordinal = pass_index * 10 + index
            direction = self.directions[((self.history.round_index - 1) * 20 + ordinal) % 64]
            noise = int(
                keyed_rng(self.stream, "noise", ordinal).integers(unit.model.config.levels + 1)
            )
            tokens, positions, logp = remask(unit.model, seq, noise, self.stream, ordinal)
            score = self.scores.get(seq)
            node = _Node(tokens, noise, seq, score, blocked=noise == 0)
            archive = {seq: score} if score is not None and score.feasible else {}
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

    def score_stage(self, sequences, stage):
        if stage != len(self.score_opportunities) or stage not in range(4):
            raise ValueError("equal-ten score opportunity cannot repeat or reorder")
        ordered = tuple(dict.fromkeys(seq for seq in sequences if self.rejection(seq) is None))
        requested = tuple(seq for seq in ordered if seq not in self.scores)
        if len(requested) > 120 or len(self.scores) + len(requested) > 480:
            raise _Stop("partial_fixed_score_admission_cap")
        opportunity = {
            "stage": stage,
            "ordered_sequences": list(ordered),
            "requested_sequences": list(requested),
            "cache_hits": [
                {"sequence": seq, **self.cache_origins[seq]}
                for seq in ordered
                if seq in self.scores
            ],
            "response_index": None,
            "state": "reserved",
        }
        self.score_opportunities.append(opportunity)
        self.check()
        if requested:
            batch = read_native_posterior(
                self.evaluator, requested, expected_binding=self.binding, context=self.context
            )
            index = len(self.responses)
            self.responses.append({"sequences": list(requested), "batch": asdict(batch)})
            opportunity["response_index"] = index
            self.scores.update(zip(requested, batch.scores, strict=True))
            for offset, seq in enumerate(requested):
                self.cache_origins[seq] = {
                    "response_index": index,
                    "row": offset,
                    "receipt_sha256": batch.receipt_sha256,
                }
        opportunity["state"] = "complete"
        self.check()

    def materialize_roots(self, trees):
        for tree in trees:
            score = self.scores[tree.seed]
            if tree.nodes[0].score is None:
                tree.nodes[0].score = score
                if score.feasible:
                    tree.archive.setdefault(tree.seed, score)
                    self.pool.setdefault(tree.seed, score)
            elif tree.nodes[0].score != score:
                raise ValueError("equal-ten cached root mean changed")

    def expand_stage(self, trees, stage):
        count, expansion = CHILD_COUNTS[stage], stage % 2
        slots = self.reserve(count, "child", stage)
        tree_rows = {
            tree.ordinal: [row for row in slots if row["triple"] == tree.unit.triple]
            for tree in trees
        }
        planned, rollout_rows = [], []
        for tree in trees:
            self.check()
            rows = tree_rows[tree.ordinal]
            for row in rows:
                row["tree"] = tree.ordinal
            path = self.select(tree, expansion)
            if path is None:
                for row in rows:
                    row.update(state="blocked_identity", endpoint=tree.seed)
                self.events.append(
                    {
                        "kind": "quota_noop",
                        "tree": tree.ordinal,
                        "stage": stage,
                        "reason": "blocked_tree",
                        "attempts": [row["attempt"] for row in rows],
                        "endpoint": tree.seed,
                    }
                )
                continue
            node = tree.nodes[path[-1]]
            level, skipped = node.level, []
            while level:
                state = NativeTransitionState(node.tokens, len(node.endpoint), level)
                if _state_contract(state, tree.unit.model.config)[1]:
                    break
                skipped.append(level)
                level -= 1
            if not level:
                node.blocked = True
                for row in rows:
                    row.update(state="blocked_identity", endpoint=node.endpoint)
                self.events.append(
                    {
                        "kind": "quota_noop",
                        "tree": tree.ordinal,
                        "stage": stage,
                        "reason": "empty_leaf",
                        "path": path,
                        "skipped_levels": skipped,
                        "attempts": [row["attempt"] for row in rows],
                        "endpoint": node.endpoint,
                    }
                )
                continue
            kernel = _validated_transition_kernels(
                tree.unit.model, (state,), NATIVE_ENDPOINT_DEFAULTS
            )[0]
            children = []
            for child, row in enumerate(rows):
                draw = event_draw(
                    kernel, rng=keyed_rng(self.stream, "child", tree.ordinal, expansion, child)
                )
                tokens, action = commit(node.tokens, level, draw, "gumbel_subset")
                row["state"] = "native_active"
                children.append({"attempt": row["attempt"], "action": action, "tokens": tokens})
                self.pending.append(
                    {
                        "attempt": row["attempt"],
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
        score_rows = [tree.seed for tree in trees] if stage == 0 else []
        for _, _, _, _, children in planned:
            for child in children:
                child["endpoint"], child["rollout"] = next(completions)
                slot = self.slots[child["attempt"]]
                slot["state"] = "native_completed"
                pending = next(row for row in self.pending if row["attempt"] == child["attempt"])
                pending["completion"] = {"endpoint": child["endpoint"], "rollout": child["rollout"]}
                reason = self.rejection(child["endpoint"]) or (
                    "generated_duplicate" if child["endpoint"] in self.seen else None
                )
                self.seen.add(child["endpoint"])
                child["rejection"] = reason
                # Retain valid duplicate references as explicit cache hits, but
                # they cannot pass the child gate or cause a new feature call.
                if (
                    reason is None
                    or child["endpoint"] in self.scores
                    or (stage == 0 and child["endpoint"] in {tree.seed for tree in trees})
                ):
                    score_rows.append(child["endpoint"])
        self.score_stage(score_rows, stage)
        self.materialize_roots(trees)
        self.check()
        for tree, path, level, skipped, children in planned:
            parent = tree.nodes[path[-1]]
            if parent.score is None:
                raise ValueError("equal-ten cannot gate on an unscored root")
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

    def run(self):
        try:
            self.bootstrap()
            seeds = self.seed_rows
            for pass_index in range(2):
                trees = self.start_trees(seeds, pass_index)
                for stage in range(pass_index * 2, pass_index * 2 + 2):
                    self.expand_stage(trees, stage)
                seeds = self.refine(trees)
            self.check()
            remaining, candidates = list(self.pool), []
            while remaining and len(candidates) < 256:
                self.check()
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
            return COMPLETE, tuple(candidates)
        except _Stop as error:
            return str(error), ()


def run_native_mp2d_equal_ten(
    initializations,
    corpora,
    history,
    context,
    evaluator,
    *,
    expected_binding,
    eligible_query_ids: frozenset[str],
    deadline: float,
    clock=time.monotonic,
    _replay_stop=None,
) -> EqualTenMP2DWave:
    """Caller bounds deadline by the original controller wave AND scientific epoch.

    In particular upstream posterior fit time is not refunded. Local entry+120
    only shortens that existing deadline; it cannot renew an already begun wave.
    """
    started = clock()
    if not math.isfinite(deadline):
        raise ValueError("equal-ten needs the original finite deadline")
    if type(history) is not VerifiedHistorySnapshot:
        raise TypeError("equal-ten needs a verified charged-history snapshot")
    history.__post_init__()
    semantic_history(history, eligible_query_ids)
    if not history.complete or history.round_index == 29:
        raise ValueError("equal-ten needs complete pre-adaptive history")
    if (
        type(context) is not NormalizedObjectiveContext
        or history.objective_context_sha256 != context.context_sha256
    ):
        raise ValueError("equal-ten objective context differs")
    context.__post_init__()
    if type(expected_binding) is not FrozenNativePosteriorBinding or (
        expected_binding.history_sha256 != history.sha256
        or expected_binding.objective_context_sha256 != context.context_sha256
        or evaluator.binding != expected_binding
    ):
        raise ValueError("equal-ten frozen posterior/history binding differs")
    expected_binding.__post_init__()
    initializations, corpora = tuple(initializations), tuple(corpora)
    if tuple(item.triple for item in initializations) != TRIPLES or len(corpora) != 10:
        raise ValueError("equal-ten requires all ten ordered audited checkpoints")
    sources = source_identities()
    units = tuple(
        _NativeUnit(init, corpus) for init, corpus in zip(initializations, corpora, strict=True)
    )
    runner = _EqualTenRunner(
        units,
        history,
        context,
        evaluator,
        expected_binding,
        min(deadline, started + 120),
        clock,
        _replay_stop,
        eligible_query_ids=eligible_query_ids,
    )
    status, candidates = runner.run()

    def check_bindings():
        for unit in units:
            unit.check()
        if evaluator.binding != expected_binding or source_identities() != sources:
            raise ValueError("equal-ten provider/source changed during execution")
        for init in initializations:
            if canonical_model_logical_hash(init.model) != init.checkpoint_logical_sha256:
                raise ValueError("equal-ten caller model changed")

    check_bindings()
    payload = {
        "artifact": "native_mp2d_equal_ten_wave_v2",
        "config_sha256": CONFIG_SHA256,
        "history_sha256": history.sha256,
        "posterior": asdict(expected_binding),
        "eligible_query_ids": sorted(eligible_query_ids),
        "semantic_stream_sha256": runner.stream,
        "source_identities": sources,
        "checkpoint_order": [unit.triple for unit in runner.order],
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
        "slots": runner.slots,
        "bootstrap_rounds": runner.bootstrap_rounds,
        "checkpoint_counts": checkpoint_counts(runner.slots, runner.events),
        "pending_expansion_attempts": runner.pending,
        "pending_bootstrap_attempts": runner.pending_bootstrap,
        "posterior_responses": runner.responses,
        "score_opportunities": runner.score_opportunities,
        "attempts": runner.attempts,
        "unique_posterior_rows": len(runner.scores),
        "scope": "equal_assigned_attempt_mixture_not_equal_active_rollouts_yield_or_selected_outputs",
        "scientific_evidence_accepted": False,
        "production_eligible": False,
        "campaign_eligible": False,
    }

    def encode(*, final_check_pending=False):
        payload.update(
            checks=runner.checks + int(final_check_pending),
            stop_check=runner.stop_check,
            candidates=candidates,
            stop_reason=status,
            completed_balanced_wave=status == COMPLETE,
        )
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(serialized.encode()) > RECEIPT_LIMIT:
            raise ValueError("equal-ten receipt exceeds original128MiB cap")
        return serialized, hashlib.sha256(serialized.encode()).hexdigest()

    serialized, digest = encode(final_check_pending=runner.stop_check is None)
    check_bindings()
    if runner.stop_check is None:
        try:
            runner.check()
        except _Stop as error:
            status, candidates = str(error), ()
            serialized, digest = encode()
    return EqualTenMP2DWave(serialized, digest, candidates, status)

"""Real native updates behind a prospective, per-student replay overlay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_search_posterior import read_native_posterior
from amp_challenge.generators.diffusion.native_tr2d2 import NativeTR2D2Ensemble
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2_records import (
    CONFIG_SHA256,
    ReplayAdvanceV2,
    ReplayFailureV2,
    prepare_replay_generation,
    seal_document,
    source_bytes,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def replay_document(replay):
    return {
        "sha256": replay.sha256,
        "sequences": replay.sequences,
        "context_id": replay.context_id,
        "weights": replay.weights.tolist(),
        "states": [
            {"tokens": row.tokens.tolist(), "length": row.length, "level": row.level}
            for row in replay.states
        ],
    }


class NativeTR2D2ReplayV2:
    """The unchanged tree is composed with new explicit replay/commit records.

    The outer caller authenticates history and the original absolute deadline,
    hard-preempts blocking native/provider work, and owns durable publication.
    This class is not the scientific campaign controller or an oracle client.
    """

    def __init__(self, *args, scientific_deadline, clock=time.monotonic, **kwargs):
        started = clock()
        if (
            not math.isfinite(scientific_deadline)
            or not started < scientific_deadline <= started + 7200
        ):
            raise ValueError("TR2 replay-v2 requires a bounded original scientific deadline")
        self.clock, self.scientific_deadline = clock, scientific_deadline
        self.tree = NativeTR2D2Ensemble(*args, **kwargs)
        self.source = source_bytes()
        self.versions = {unit.triple: 0 for unit in self.tree._units}
        self.generation = self.receipt = self.failure = None
        self.collection_operation = None
        self._collection_batching_sha256 = None
        self._generation_sha256 = None
        self._wave_deadlines = {}
        self._attempted_collections = set()
        self._attempted_advances = set()

    @property
    def policy_identities(self):
        return self.tree.policy_identities

    def _check(self):
        if self.failure is not None:
            raise ValueError("TR2 replay-v2 terminal failure cannot restart")
        if source_bytes() != self.source:
            raise ValueError("TR2 replay-v2 numerical source changed")
        for unit in self.tree._units:
            unit.check()

    def _time(self, round_index, phase):
        if self.clock() >= self._wave_deadlines[round_index]:
            raise TimeoutError("TR2 replay-v2 original deadline exhausted at " + phase)

    def _failed(self, phase, error, prefix):
        self.failure = ReplayFailureV2(phase, f"{type(error).__name__}: {error}"[:512], prefix)

    def _history_admission(self, history, expected_previous_head_sha256):
        if type(history) is not VerifiedHistorySnapshot:
            raise TypeError("TR2 replay-v2 needs the exact verified charged history")
        history.__post_init__()
        tree = self.tree
        if (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
            history.previous_wave_head_sha256,
        ) != (
            tree.run_id,
            tree.seed,
            tree.context.context_sha256,
            tree.oracle_bundle_sha256,
            expected_previous_head_sha256,
        ):
            raise ValueError("TR2 replay-v2 history context/run/source/head differs")
        if any(sequence_id(row.sequence) in tree._training_ids for row in history.observations):
            raise ValueError("TR2 replay-v2 charged history overlaps generator training")
        old = tree._history
        if old is not None and history.sha256 == old.sha256:
            return "already_advanced"
        if history.round_index != (1 if old is None else old.round_index + 1):
            raise ValueError("TR2 replay-v2 history skipped or repeated a round")
        if old is not None and (
            self.generation is None
            or history.observations[: len(old.observations)] != old.observations
        ):
            raise ValueError("TR2 replay-v2 missing collection or changed charged prefix")
        return "ready" if history.complete else "paused_incomplete_history"

    def advance(self, history, *, expected_previous_head_sha256, outer_deadline):
        """outer_deadline includes the ORIGINAL controller wave start and learner fit.

        Supply min(original scientific deadline, original wave start + 180),
        never only a later two-hour deadline after upstream work. Local checks
        can shorten that bound but cannot authenticate the caller's prior epoch.
        Preflight rejection does not start a neural attempt; a begun attempt's
        failure is terminal, and incomplete-history pauses retain the first cap.
        """
        self._check()
        admission = self._history_admission(history, expected_previous_head_sha256)
        if admission == "already_advanced":
            return self.receipt
        now = self.clock()
        if not math.isfinite(outer_deadline) or outer_deadline > self.scientific_deadline:
            raise ValueError("TR2 replay-v2 outer deadline exceeds original scientific clock")
        deadline = min(now + 180, outer_deadline, self.scientific_deadline)
        self._wave_deadlines[history.round_index] = min(
            self._wave_deadlines.get(history.round_index, deadline), deadline
        )
        if admission == "paused_incomplete_history":
            self._time(history.round_index, "incomplete_history")
            return None
        if history.round_index in self._attempted_advances:
            raise ValueError("TR2 replay-v2 advance cannot retry")
        self._attempted_advances.add(history.round_index)
        payload = {
            "configuration_sha256": CONFIG_SHA256,
            "source_sha256": _json_hash(self.source),
            "history_sha256": history.sha256,
            "round_index": history.round_index,
            "previous_generation_sha256": None
            if self.generation is None
            else self.generation.sha256,
            "status": "started",
            "students": [],
            "campaign_eligible": False,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }
        try:
            self._time(history.round_index, "before_replay_admission")
            plan = self.generation
            if plan is not None:
                if plan.sha256 != self._generation_sha256:
                    raise ValueError("TR2 replay-v2 saved generation changed")
                expected = prepare_replay_generation(
                    plan.collection,
                    self.tree.context,
                    expected_versions=self.versions,
                    source_sha256=_json_hash(self.source),
                )
                if plan != expected or plan.collection.round_index + 1 != history.round_index:
                    raise ValueError("TR2 replay-v2 generation freshness or actual versions differ")
                if plan.collection.history_sha256 != self.tree._history.sha256:
                    raise ValueError("TR2 replay-v2 generation does not bind previous history")
                for unit, buffer in zip(self.tree._units, plan.buffers, strict=True):
                    if (
                        buffer.triple,
                        buffer.behavior_sha256,
                        buffer.reference_sha256,
                        buffer.behavior_version,
                    ) != (
                        unit.triple,
                        unit.policy_sha256,
                        unit.reference_sha256,
                        self.versions[unit.triple],
                    ):
                        raise ValueError("TR2 replay-v2 true behavior model/version differs")
            # Every student's real support/weight/source gates are established
            # above before the first student can execute any SGD.
            replacements, versions = [], dict(self.versions)
            for index, unit in enumerate(self.tree._units):
                buffer = None if plan is None else plan.buffers[index]
                row = {
                    "triple": unit.triple,
                    "old_policy_sha256": unit.policy_sha256,
                    "new_policy_sha256": unit.policy_sha256,
                    "reference_sha256": unit.reference_sha256,
                    "old_version": versions[unit.triple],
                    "new_version": versions[unit.triple],
                    "trajectory_version_lag": None
                    if buffer is None
                    else versions[unit.triple] - buffer.behavior_version,
                    "saved_buffer": None if buffer is None else asdict(buffer),
                    "steps": [],
                    "status": "unchanged_common_initial"
                    if plan is None
                    else "unchanged_terminal_round29"
                    if history.round_index == 29
                    else buffer.status,
                }
                payload["students"].append(row)
                if (
                    row["trajectory_version_lag"] is not None
                    and not 0 <= row["trajectory_version_lag"] <= 1
                ):
                    raise ValueError("TR2 replay-v2 true trajectory version lag exceeded")
                working = unit.model
                if (
                    buffer is not None
                    and history.round_index < 29
                    and buffer.status == "admitted_actual_weights"
                ):
                    sequences = tuple(
                        plan.collection.attempts[i].path.endpoint for i in buffer.attempt_indices
                    )
                    weights = np.array(buffer.normalized_weights)
                    seed = int(_json_hash([history.seed, unit.triple, "tr2d2-training"])[:16], 16)
                    for step in range(4):
                        self._time(history.round_index, "before_native_step")
                        ordinal = history.round_index * 4 + step
                        replay = build_weighted_replay(
                            working,
                            sequences,
                            weights,
                            context_id=self.tree.context.context_sha256,
                            seed=seed,
                            ordinal=ordinal,
                        )
                        probes = build_weighted_replay(
                            unit.reference,
                            sequences,
                            weights,
                            context_id=self.tree.context.context_sha256,
                            seed=seed,
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
                        row["steps"].append(
                            {
                                "replay": replay_document(replay),
                                "probes": replay_document(probes),
                                "before": direction.base_model_sha256,
                                "after": after,
                                "gradient_sha256": gradient.hexdigest(),
                                "objective": direction.objective_before,
                                "gradient_norm": direction.gradient_norm_before_clip,
                                "diagnostics": asdict(diagnostics),
                                "kl_enforced": False,
                            }
                        )
                        working = candidate
                        self._time(history.round_index, "after_native_step")
                    row["status"] = "four_actual_weighted_steps"
                    versions[unit.triple] += 1
                replacement = copy.copy(unit)
                replacement.model = working
                replacement.policy_sha256 = canonical_model_logical_hash(working)
                row["new_policy_sha256"] = replacement.policy_sha256
                row["new_version"] = versions[unit.triple]
                replacements.append(replacement)
            payload["status"] = "atomic_tuple_ready"
            raw, digest = seal_document(payload)
            self._check()
            self._time(history.round_index, "after_advance_receipt_serialization")
        except (
            ValueError,
            TypeError,
            RuntimeError,
            FloatingPointError,
            TimeoutError,
            OSError,
        ) as error:
            self._failed("advance", error, payload)
            raise
        # Publication means an in-process sealed handoff, not durable campaign I/O.
        replacement_tree = copy.copy(self.tree)
        replacement_tree._units = tuple(replacements)
        replacement_tree._history, replacement_tree._generation = history, None
        self.tree, self.versions = replacement_tree, versions
        self.receipt = ReplayAdvanceV2(raw, digest)
        self.generation = self._generation_sha256 = None
        return self.receipt

    def collect(self, evaluator, *, expected_posterior_sha256, feature_batching_sha256=None):
        self._check()
        grouped = feature_batching_sha256 is not None
        if grouped:
            from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
            from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
                GroupedNativePosterior,
                batching_source,
                check_batching_configuration,
            )

            check_batching_configuration(feature_batching_sha256)
        history = self.tree._history
        if history is None or history.round_index == 29:
            raise ValueError("TR2 replay-v2 needs a complete nonterminal history")
        if self.generation is not None:
            if (
                self.generation.collection.evaluator_binding.posterior_sha256
                != expected_posterior_sha256
                or self._collection_batching_sha256 != feature_batching_sha256
            ):
                raise ValueError("TR2 replay-v2 cannot replace the frozen posterior")
            return self.generation
        if history.round_index in self._attempted_collections:
            raise ValueError("TR2 replay-v2 collection cannot retry")
        self._attempted_collections.add(history.round_index)
        prefix = {
            "history_sha256": history.sha256,
            "completed_posterior_batches": [],
            "completed_collection": None,
            "replacement_commit_permitted": False,
        }
        owner = self
        fixed_binding = evaluator.binding
        counter = NativeWorkCounter(self.tree._units) if grouped else None
        self.collection_operation = None
        if grouped:
            prefix.update(
                feature_batching_sha256=feature_batching_sha256,
                feature_batching_source_sha256=batching_source(),
                prepared_expansions=[],
                applied_expansion_indices=[],
                active_expansion=None,
                active_grouped_evaluation=None,
                completed_grouped_batches=[],
                native_work=counter.document(),
            )

        def checkpoint(stage):
            owner._time(history.round_index, stage)

        def retain(event, value):
            if event == "preparing_expansion":
                prefix["active_expansion"] = value
            elif event == "prepared_expansion":
                prefix["prepared_expansions"].append(value)
                prefix["active_expansion"] = None
            elif event == "applied_expansion":
                prefix["applied_expansion_indices"].append(value)
            prefix["native_work"] = counter.document()

        def retain_scores(event, payload):
            # Capture returned feature bytes before any closing deadline check.
            if event == "raw_features":
                prefix["active_grouped_evaluation"] = {"raw": json.loads(payload), "groups": []}
            elif event == "scored_group":
                prefix["active_grouped_evaluation"]["groups"].append(json.loads(payload))
            owner._time(history.round_index, "after_grouped_" + event)

        class RecordedProvider:
            @property
            def binding(self):
                return evaluator.binding

            def evaluate(self, sequences):
                owner._time(history.round_index, "before_posterior")
                result = read_native_posterior(
                    evaluator, sequences, expected_binding=fixed_binding, context=owner.tree.context
                )
                prefix["completed_posterior_batches"].append(
                    {"sequences": sequences, "result": asdict(result)}
                )
                owner._time(history.round_index, "after_posterior")
                return result

            def evaluate_groups(self, groups):
                owner._time(history.round_index, "before_grouped_posterior")
                if evaluator.binding != fixed_binding:
                    raise ValueError("TR2 grouped posterior binding differs")
                result = evaluator.evaluate_groups(groups, retain=retain_scores)
                if (
                    evaluator.binding != fixed_binding
                    or type(result) is not GroupedNativePosterior
                    or len(result.groups) != len(groups)
                ):
                    raise ValueError("TR2 grouped posterior result differs")
                result.__post_init__()
                prefix["completed_grouped_batches"].append(json.loads(result.record_payload))
                prefix["active_grouped_evaluation"] = None
                prefix["completed_posterior_batches"].extend(
                    {"sequences": sequences, "result": asdict(batch)}
                    for sequences, batch in zip(groups, result.groups, strict=True)
                )
                owner._time(history.round_index, "after_grouped_posterior")
                return result

        try:
            self._time(history.round_index, "before_legacy_collection")
            candidate_tree = copy.copy(self.tree)
            with (
                counter if grouped else nullcontext(),
                counter.at("tr2_collection") if grouped else nullcontext(),
            ):
                collection = candidate_tree.collect(
                    RecordedProvider(),
                    expected_posterior_sha256=expected_posterior_sha256,
                    feature_batching_sha256=feature_batching_sha256,
                    checkpoint=checkpoint if grouped else None,
                    retain=retain if grouped else None,
                )
            if grouped:
                prefix["native_work"] = counter.document()
            prefix["completed_collection"] = collection
            generation = prepare_replay_generation(
                collection,
                self.tree.context,
                expected_versions=self.versions,
                source_sha256=_json_hash(self.source),
            )
            _, digest = seal_document(asdict(generation))
            operation = None
            if grouped:
                if batching_source() != prefix["feature_batching_source_sha256"]:
                    raise ValueError("TR2 grouped-feature source changed during collection")
                operation, _ = seal_document(
                    {
                        "artifact": "native_tr2_grouped_collection_v1",
                        "configuration_sha256": feature_batching_sha256,
                        "source_sha256": prefix["feature_batching_source_sha256"],
                        "history_sha256": history.sha256,
                        "generation_sha256": generation.sha256,
                        "prepared_expansions": [
                            {
                                **{key: value for key, value in row.items() if key != "paths"},
                                "path_sha256s": [_json_hash(asdict(path)) for path in row["paths"]],
                            }
                            for row in prefix["prepared_expansions"]
                        ],
                        "applied_expansion_indices": prefix["applied_expansion_indices"],
                        "posterior_batches": prefix["completed_grouped_batches"],
                        "native_work": prefix["native_work"],
                        "scientific_evidence_accepted": False,
                    }
                )
            self._check()
            self._time(history.round_index, "after_collection_serialization")
        except BaseException as error:
            if not grouped and not isinstance(
                error,
                ValueError | TypeError | RuntimeError | FloatingPointError | TimeoutError | OSError,
            ):
                raise
            if grouped:
                prefix["native_work"] = counter.document()
            self._failed("collect", error, prefix)
            raise
        self.tree, self.generation = candidate_tree, generation
        self._generation_sha256 = digest
        self._collection_batching_sha256 = feature_batching_sha256
        self.collection_operation = None if operation is None else operation.encode()
        return generation

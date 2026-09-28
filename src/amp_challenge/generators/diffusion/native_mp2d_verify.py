"""Native MP2D reconstruction without live posterior calls or new authority.

Two checks: separately written conditional-state/event reconstruction, followed
by full deterministic engine re-execution against the recorded response table.
The latter shares the producer and is not independent algorithm authorship.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.native_proposals import (
    NativeProposalStep,
    NativeProposalTrace,
    replay_native_trace,
)
from amp_challenge.generators.diffusion.native_search_posterior import (
    NativePosteriorBatch,
    NativePosteriorScore,
)
from amp_challenge.generators.diffusion.subset_kernel import SubsetCommitDraw


def _rng(stream, *key):
    payload = json.dumps(
        ["native-mp2d-v1", stream, key], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return np.random.Generator(
        np.random.PCG64DXSM(int(hashlib.sha256(payload).hexdigest()[:32], 16))
    )


def _trace(raw):
    values = dict(raw)
    values["remasked_positions"] = tuple(values["remasked_positions"])
    steps = []
    for step in values["steps"]:
        draw = step["draw"]
        steps.append(
            NativeProposalStep(
                step["level"],
                step["before_tokens_sha256"],
                step["after_tokens_sha256"],
                SubsetCommitDraw(
                    tuple(draw["positions"]), tuple(draw["residues"]), draw["log_probability"]
                ),
            )
        )
    values["steps"] = tuple(steps)
    return NativeProposalTrace(**values)


def _step(model, tokens, length, raw, rng=None):
    state = NativeTransitionState(tokens, length, raw["level"])
    kernel = _validated_transition_kernels(model, (state,), NATIVE_ENDPOINT_DEFAULTS)[0]
    if not kernel.commit_count:
        positions, residues = (), ()
    elif rng is None:
        priorities = [
            (float(row.max()), pos)
            for pos, row in zip(kernel.masked_positions, kernel.residue_probabilities, strict=True)
        ]
        positions = tuple(
            sorted(
                pos
                for _, pos in sorted(priorities, key=lambda pair: (-pair[0], pair[1]))[
                    : kernel.commit_count
                ]
            )
        )
        residues = tuple(
            int(kernel.residue_probabilities[kernel.masked_positions.index(pos)].argmax())
            for pos in positions
        )
    else:
        positions = tuple(
            sorted(
                int(pos)
                for pos in rng.choice(kernel.masked_positions, kernel.commit_count, replace=False)
            )
        )
        residues = tuple(
            int(
                (
                    np.log(kernel.residue_probabilities[kernel.masked_positions.index(pos)])
                    + rng.gumbel(size=20)
                ).argmax()
            )
            for pos in positions
        )
    if (
        raw["positions"] != list(positions)
        or raw["residues"] != list(residues)
        or raw["before_sha256"] != _json_hash(tokens.tolist())
    ):
        raise ValueError("MP2D conditional event/draw reconstruction differs")
    logp = kernel.log_probability(positions, residues)
    if not math.isclose(logp, raw["base_log_probability"], rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("MP2D complete conditional action probability differs")
    after = np.array(tokens, copy=True)
    after[list(positions)] = residues
    if raw["after_sha256"] != _json_hash(after.tolist()):
        raise ValueError("MP2D conditional state identity differs")
    return after


def reconstruct_mp2d_native_events(initializations, payload):
    models = {init.triple: init.model for init in initializations}
    nodes, roots = {}, {}
    stream = payload["semantic_stream_sha256"]
    for event in payload["events"]:
        if event["kind"] == "bootstrap":
            replay_native_trace(
                models[event["triple"]], _trace(event["trace"]), authenticate_sampling=True
            )
        elif event["kind"] == "root":
            model, seq, level = models[event["triple"]], event["seed"], event["noise"]
            count = int(
                CosineMaskSchedule().mask_counts(len(seq), level, total_levels=model.config.levels)[
                    0
                ]
            )
            positions = tuple(
                sorted(
                    map(
                        int,
                        _rng(stream, "remask", event["tree"]).choice(
                            len(seq), count, replace=False
                        ),
                    )
                )
            )
            tokens = (
                PeptideVocabulary().encode((seq,), max_length=model.config.max_length).tokens[0]
            )
            tokens[list(positions)] = MASK_TOKEN_INDEX
            if (
                list(positions) != event["positions"]
                or event["tokens_sha256"] != _json_hash(tokens.tolist())
                or not math.isclose(
                    event["remask_log_probability"],
                    -math.log(math.comb(len(seq), count)),
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("MP2D root remask reconstruction differs")
            roots[event["tree"]] = (model, len(seq))
            nodes[event["tree"]] = [tokens]
        elif event["kind"] == "expansion":
            model, length = roots[event["tree"]]
            parent = nodes[event["tree"]][event["path"][-1]]
            children = []
            for index, child in enumerate(event["children"]):
                tokens = _step(
                    model,
                    parent,
                    length,
                    child["action"],
                    _rng(stream, "child", event["tree"], event["expansion"], index),
                )
                children.append(tokens.copy())
                expected_level = child["action"]["level"] - 1
                for raw in child["rollout"]:
                    if raw["level"] != expected_level or raw["mode"] != "greedy_map":
                        raise ValueError("MP2D greedy rollout level/mode differs")
                    tokens = _step(model, tokens, length, raw)
                    expected_level -= 1
                if (
                    expected_level
                    or PeptideVocabulary().decode(tokens[None, :])[0] != child["endpoint"]
                ):
                    raise ValueError("MP2D clean completion reconstruction differs")
            for retained in event["decision"]["retained"]:
                nodes[event["tree"]].append(children[retained])


def verify_native_mp2d(initializations, corpora, history, context, wave, *, expected_binding):
    """Reconstruct one bounded receipt against caller-supplied immutable inputs."""
    from amp_challenge.generators.diffusion.native_mp2d_operators import (
        NativeMP2DWave,
        run_native_mp2d,
    )

    if (
        type(wave) is not NativeMP2DWave
        or wave.scientific_evidence_accepted is not False
        or wave.production_eligible is not False
    ):
        raise ValueError("MP2D wave qualification differs")
    if (
        type(wave.record_json) is not str
        or len(wave.record_json.encode()) > 128 * 1024**2
        or hashlib.sha256(wave.record_json.encode()).hexdigest() != wave.sha256
    ):
        raise ValueError("MP2D wave content seal differs")
    payload = json.loads(wave.record_json)
    if payload["history_sha256"] != history.sha256 or payload["posterior"] != asdict(
        expected_binding
    ):
        raise ValueError("MP2D reconstruction history/posterior binding differs")
    if len(payload["posterior_responses"]) > 1024 or len(payload["events"]) > 2048:
        raise ValueError("MP2D reconstruction inventory exceeds budget")
    reconstruct_mp2d_native_events(initializations, payload)

    class RecordedPosterior:
        binding = expected_binding

        def __init__(self):
            self.index = 0

        def evaluate(self, sequences):
            if self.index >= len(payload["posterior_responses"]):
                raise ValueError("MP2D replay requested an unrecorded posterior batch")
            row = payload["posterior_responses"][self.index]
            self.index += 1
            if list(sequences) != row["sequences"]:
                raise ValueError("MP2D replay posterior request identity/order differs")
            batch = row["batch"]
            return NativePosteriorBatch(
                tuple(batch["sequence_ids"]),
                tuple(
                    NativePosteriorScore(tuple(score["objectives"]), score["feasible"])
                    for score in batch["scores"]
                ),
                batch["receipt_sha256"],
            )

    evaluator = RecordedPosterior()
    replayed = run_native_mp2d(
        initializations,
        corpora,
        history,
        context,
        evaluator,
        expected_binding=expected_binding,
        deadline=1e12,
        clock=lambda: 0.0,
        _replay_stop=payload["stop_check"],
    )
    if evaluator.index != len(payload["posterior_responses"]) or replayed != wave:
        raise ValueError("MP2D complete numerical/provenance replay differs")
    return {
        "reconstructed": True,
        "posterior_rows": payload["unique_posterior_rows"],
        "attempts": payload["attempts"],
        "stop_reason": wave.stop_reason,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }

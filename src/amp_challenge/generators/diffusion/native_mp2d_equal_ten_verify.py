"""Independent v2 schedule/cache/native reconstruction, then disclosed engine replay.

The final engine replay is not independent MCTS authorship. Native event math
shares the accepted, separately authored v1 checker, never a live posterior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_mp2d_verify import (
    _rng,
    _step,
    _trace,
    reconstruct_mp2d_native_events,
)
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_search_posterior import (
    NativePosteriorBatch,
    NativePosteriorScore,
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sequence_id(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def reconstruct_equal_ten_schedule(
    initializations, corpora, history, context, payload, *, expected_eligible_query_ids
):
    """No imports of v2 producer, allocation, cache or counting functions."""
    if tuple(init.triple for init in initializations) != TRIPLES or len(corpora) != 10:
        raise ValueError("equal-ten reconstruction requires exact ten inputs")
    config = "7dfb024ea7d1bbe1c6396cac995157db1c2169e495b616da7479dfe474030e4e"
    if (
        type(expected_eligible_query_ids) is not frozenset
        or any(type(value) is not str for value in expected_eligible_query_ids)
        or not expected_eligible_query_ids
        <= {row.query_id for row in history.observations if row.status == "successful"}
        or payload["eligible_query_ids"] != sorted(expected_eligible_query_ids)
    ):
        raise ValueError("equal-ten external eligibility binding differs")
    rows = [
        (
            row.charge_index,
            row.sequence,
            row.status,
            row.query_id in expected_eligible_query_ids,
            row.objectives if row.query_id in expected_eligible_query_ids else None,
        )
        for row in history.observations
    ]
    stream = _hash(
        [
            "native-mp2d-equal-ten-v2",
            config,
            history.seed,
            history.round_index,
            history.objective_context_sha256,
            rows,
        ]
    )
    order = [TRIPLES[int(index)] for index in _rng(stream, "checkpoint-order").permutation(10)]
    if (
        payload["semantic_stream_sha256"] != stream
        or payload["checkpoint_order"] != order
        or payload["config_sha256"] != config
    ):
        raise ValueError("equal-ten semantic checkpoint schedule differs")
    slots, events = payload["slots"], payload["events"]
    # A deadline may expire before the first reserved bootstrap round.
    if (
        type(payload["bootstrap_rounds"]) is not int
        or not 0 <= payload["bootstrap_rounds"] <= 55
        or (payload["bootstrap_rounds"] == 0 and slots)
    ):
        raise ValueError("equal-ten bootstrap allocation differs")
    bootstrap_size = payload["bootstrap_rounds"] * 10
    expected = [
        ("bootstrap", round_index, 0, triple)
        for round_index in range(payload["bootstrap_rounds"])
        for triple in order
    ]
    stage_sizes = (110, 120, 120, 120)
    remaining = len(slots) - bootstrap_size
    stages = 0
    while remaining > 0 and stages < 4:
        remaining -= stage_sizes[stages]
        expected.extend(
            ("child", stages, local, triple)
            for local in range((11, 12, 12, 12)[stages])
            for triple in order
        )
        stages += 1
    if (
        remaining
        or len(slots) != len(expected)
        or len(slots) > 1020
        or payload["attempts"] != len(slots)
    ):
        raise ValueError("equal-ten stage quota must be whole and bounded")
    valid_states = {"reserved", "native_active", "native_completed", "blocked_identity"}
    for index, (row, (phase, stage, local, triple)) in enumerate(zip(slots, expected, strict=True)):
        if (row["attempt"], row["phase"], row["stage"], row["local"], row["triple"]) != (
            index,
            phase,
            stage,
            local,
            triple,
        ) or row["state"] not in valid_states:
            raise ValueError("equal-ten attempt ordinal/checkpoint/quota differs")
        if phase == "bootstrap" and row["state"] == "blocked_identity":
            raise ValueError("bootstrap cannot be a fabricated identity draw")
        if "tree" in row and row["tree"] != (stage // 2) * 10 + order.index(triple):
            raise ValueError("equal-ten child tree/checkpoint binding differs")
    training = set().union(*(set(init.training_sequence_ids) for init in initializations))
    charged = {_sequence_id(row.sequence) for row in history.observations}
    seen, accepted, child_rows = set(), {}, {stage: [] for stage in range(4)}
    models = {init.triple: init.model for init in initializations}
    corpora_by_triple = dict(zip(TRIPLES, map(sorted, corpora), strict=True))
    bootstrap_events = [event for event in events if event["kind"] == "bootstrap"]
    if len(bootstrap_events) % 10 or len(bootstrap_events) > bootstrap_size:
        raise ValueError("bootstrap adjudication must complete a ten-model round")

    def rejection(seq):
        return (
            "generator_training_overlap"
            if _sequence_id(seq) in training
            else "previously_charged"
            if _sequence_id(seq) in charged
            else None
        )

    for index, event in enumerate(bootstrap_events):
        triple, trace = order[index % 10], event["trace"]
        corpus = corpora_by_triple[triple]
        parent = corpus[int(_rng(stream, "length", index).integers(len(corpus)))]
        reason = rejection(trace["endpoint"]) or (
            "generated_duplicate" if trace["endpoint"] in seen else None
        )
        was_filled = triple in accepted
        keep = not was_filled and reason is None
        if (
            event["attempt"] != index
            or event["triple"] != triple
            or event["rejection"] != reason
            or event["root_accepted"] is not keep
            or event["already_filled"] is not was_filled
            or trace["parent"] != parent
            or trace["ordinal"] != index
            or trace["seed"] != int(stream[:16], 16)
            or trace["start_level"] != models[triple].config.levels
            or slots[index]["state"] != "native_completed"
        ):
            raise ValueError("equal-ten first-root/bootstrap provenance differs")
        seen.add(trace["endpoint"])
        if keep:
            accepted[triple] = trace["endpoint"]
        if len(accepted) == 10 and (index + 1) % 10 == 0 and index + 1 != len(bootstrap_events):
            raise ValueError("equal-ten bootstrap continued after all roots filled")
    roots = {event["tree"]: event for event in events if event["kind"] == "root"}
    root_events = [event for event in events if event["kind"] == "root"]
    if [event["tree"] for event in root_events] != list(range(len(root_events))) or len(
        root_events
    ) > 20:
        raise ValueError("equal-ten root prefix order differs")
    directions = _rng(stream, "directions").permutation(64).tolist()
    refined = {
        event["tree"]: event["selected"] for event in events if event["kind"] == "refinement"
    }
    for ordinal, event in roots.items():
        triple = order[ordinal % 10]
        seed = accepted.get(triple) if ordinal < 10 else refined.get(ordinal - 10)
        noise = int(_rng(stream, "noise", ordinal).integers(models[triple].config.levels + 1))
        if (
            event["triple"] != triple
            or event["seed"] != seed
            or event["noise"] != noise
            or event["direction_index"]
            != directions[((history.round_index - 1) * 20 + ordinal) % 64]
        ):
            raise ValueError("equal-ten fixed root/direction/component differs")
    covered = set(range(len(bootstrap_events)))
    for event in events:
        if event["kind"] == "expansion":
            stage = (event["tree"] // 10) * 2 + event["expansion"]
            if stage not in range(4) or len(event["children"]) != (11, 12, 12, 12)[stage]:
                raise ValueError("equal-ten expansion child count differs")
            for local, child in enumerate(event["children"]):
                index = child["attempt"]
                if index in covered or not 0 <= index < len(slots):
                    raise ValueError("equal-ten child attempt reused")
                row = slots[index]
                if (row["tree"], row["stage"], row["local"], row["state"]) != (
                    event["tree"],
                    stage,
                    local,
                    "native_completed",
                ):
                    raise ValueError("equal-ten expansion slot binding differs")
                covered.add(index)
                child_rows[stage].append((event["tree"], local, child))
        elif event["kind"] == "quota_noop":
            ordinal, stage = event["tree"], event["stage"]
            expected_ids = [
                row["attempt"]
                for row in slots
                if row["phase"] == "child" and row["stage"] == stage and row.get("tree") == ordinal
            ]
            if event["attempts"] != expected_ids or len(expected_ids) != (11, 12, 12, 12)[stage]:
                raise ValueError("equal-ten no-op quota differs")
            for index in expected_ids:
                if (
                    index in covered
                    or slots[index]["state"] != "blocked_identity"
                    or slots[index]["endpoint"] != event["endpoint"]
                ):
                    raise ValueError("equal-ten no-op state/provenance differs")
                covered.add(index)
    for pending in payload["pending_bootstrap_attempts"]:
        index = pending["attempt"]
        if (
            index in covered
            or not 0 <= index < bootstrap_size
            or pending["triple"] != slots[index]["triple"]
        ):
            raise ValueError("equal-ten pending bootstrap slot differs")
        triple = slots[index]["triple"]
        corpus = corpora_by_triple[triple]
        expected_parent = corpus[int(_rng(stream, "length", index).integers(len(corpus)))]
        if pending["parent"] != expected_parent:
            raise ValueError("equal-ten pending bootstrap parent law differs")
        trace = pending["trace"]
        if trace is not None:
            if (
                slots[index]["state"] != "native_completed"
                or trace["ordinal"] != index
                or trace["parent"] != expected_parent
                or trace["seed"] != int(stream[:16], 16)
                or trace["start_level"] != models[triple].config.levels
                or trace["model_sha256"]
                != initializations[TRIPLES.index(triple)].checkpoint_logical_sha256
            ):
                raise ValueError("equal-ten pending native trace differs")
            replay_native_trace(
                models[pending["triple"]], _trace(trace), authenticate_sampling=True
            )
        elif slots[index]["state"] not in ("reserved", "native_active"):
            raise ValueError("equal-ten pending bootstrap completion missing")
        covered.add(index)
    for pending in payload["pending_expansion_attempts"]:
        index = pending["attempt"]
        if index in covered or not 0 <= index < len(slots):
            raise ValueError("equal-ten pending child reused")
        row = slots[index]
        stage = row["stage"]
        if (pending["triple"], pending["tree"], pending["expansion"]) != (
            row["triple"],
            row["tree"],
            stage % 2,
        ):
            raise ValueError("equal-ten pending child component differs")
        completion = pending["completion"]
        if row["state"] != ("native_completed" if completion is not None else "native_active"):
            raise ValueError("equal-ten pending completion state differs")
        if completion is not None:
            child_rows[stage].append((row["tree"], row["local"], {"attempt": index, **completion}))
        covered.add(index)
    if any(row["state"] != "reserved" for index, row in enumerate(slots) if index not in covered):
        raise ValueError("equal-ten consumed slot has no reconstruction evidence")

    cache, origins, response_index = {}, {}, 0
    opportunities = payload["score_opportunities"]
    if len(opportunities) > 4 or len(payload["posterior_responses"]) > 4:
        raise ValueError("equal-ten exceeds four score opportunities")
    for stage in range(4):
        sequences = (
            [roots[index]["seed"] for index in range(10)] if stage == 0 and len(roots) >= 10 else []
        )
        for _, _, child in sorted(child_rows[stage], key=lambda item: (item[0], item[1])):
            seq = child["endpoint"]
            reason = rejection(seq) or ("generated_duplicate" if seq in seen else None)
            if "rejection" in child and child["rejection"] != reason:
                raise ValueError("equal-ten child rejection stream differs")
            seen.add(seq)
            if reason is None or seq in cache or (stage == 0 and seq in accepted.values()):
                sequences.append(seq)
        if stage >= len(opportunities):
            continue
        opportunity = opportunities[stage]
        ordered = list(dict.fromkeys(seq for seq in sequences if rejection(seq) is None))
        requested = [seq for seq in ordered if seq not in cache]
        hits = [{"sequence": seq, **origins[seq]} for seq in ordered if seq in cache]
        if (
            opportunity["stage"] != stage
            or opportunity["ordered_sequences"] != ordered
            or opportunity["requested_sequences"] != requested
            or opportunity["cache_hits"] != hits
            or len(requested) > 120
        ):
            raise ValueError("equal-ten delayed-root/score/cache admission differs")
        if opportunity["state"] == "reserved":
            if opportunity["response_index"] is not None or stage != len(opportunities) - 1:
                raise ValueError("equal-ten uncompleted score opportunity differs")
            continue
        if opportunity["state"] != "complete":
            raise ValueError("equal-ten score completion status differs")
        if requested:
            if opportunity["response_index"] != response_index:
                raise ValueError("equal-ten response order differs")
            response = payload["posterior_responses"][response_index]
            batch = response["batch"]
            if (
                response["sequences"] != requested
                or batch["sequence_ids"] != list(map(_sequence_id, requested))
                or len(batch["scores"]) != len(requested)
            ):
                raise ValueError("equal-ten response IDs differ")
            for offset, (seq, score) in enumerate(zip(requested, batch["scores"], strict=True)):
                NativePosteriorScore(tuple(score["objectives"]), score["feasible"]).validate(
                    context
                )
                cache[seq] = score
                origins[seq] = {
                    "response_index": response_index,
                    "row": offset,
                    "receipt_sha256": batch["receipt_sha256"],
                }
            response_index += 1
        elif opportunity["response_index"] is not None:
            raise ValueError("equal-ten empty opportunity cannot invent response")
    if (
        response_index != len(payload["posterior_responses"])
        or len(cache) != payload["unique_posterior_rows"]
        or len(cache) > 480
    ):
        raise ValueError("equal-ten score totals differ")
    counts = {}
    for triple in TRIPLES:
        rows = [row for row in slots if row["triple"] == triple]
        trees = [ordinal for ordinal, root in roots.items() if root["triple"] == triple]
        states = [row["state"] for row in rows]
        counts[triple] = {
            "assigned": len(rows),
            "bootstrap_assigned": sum(row["phase"] == "bootstrap" for row in rows),
            "child_assigned": sum(row["phase"] == "child" for row in rows),
            "native_active": states.count("native_active") + states.count("native_completed"),
            "native_completed": states.count("native_completed"),
            "identity_noop": states.count("blocked_identity"),
            "reserved_only": states.count("reserved"),
            "root_accepted": int(triple in accepted),
            "child_gate_retained": sum(
                len(event["decision"]["retained"])
                for event in events
                if event["kind"] == "expansion" and event["tree"] in trees
            ),
        }
    if payload["checkpoint_counts"] != counts:
        raise ValueError("equal-ten assigned/native/no-op count claims differ")
    completed = payload["stop_reason"] == "complete_equal_ten_assigned_wave_not_campaign"
    if payload["completed_balanced_wave"] is not completed or (
        completed
        and (
            len(roots) != 20
            or len(opportunities) != 4
            or any(
                row["child_assigned"] != 47
                or row["root_accepted"] != 1
                or row["reserved_only"]
                or row["native_active"] != row["native_completed"]
                for row in counts.values()
            )
        )
    ):
        raise ValueError("equal-ten complete-wave balance claim differs")
    if not completed and payload["candidates"]:
        raise ValueError("equal-ten partial wave cannot release candidates")
    reconstruct_mp2d_native_events(initializations, payload)
    _reconstruct_pending_native(initializations, payload)
    return counts


def _reconstruct_pending_native(initializations, payload):
    models = {init.triple: init.model for init in initializations}
    nodes, trees = {}, {}
    for event in payload["events"]:
        if event["kind"] == "root":
            tokens = (
                PeptideVocabulary()
                .encode((event["seed"],), max_length=models[event["triple"]].config.max_length)
                .tokens[0]
            )
            tokens[event["positions"]] = MASK_TOKEN_INDEX
            nodes[event["tree"]] = [tokens]
            trees[event["tree"]] = event
        elif event["kind"] == "expansion":
            parent = nodes[event["tree"]][event["path"][-1]]
            for index in event["decision"]["retained"]:
                child = event["children"][index]
                tokens = np.array(parent, copy=True)
                tokens[child["action"]["positions"]] = child["action"]["residues"]
                nodes[event["tree"]].append(tokens)
    for pending in payload["pending_expansion_attempts"]:
        tree, action = trees[pending["tree"]], pending["action"]
        model = models[pending["triple"]]
        # A before-state digest identifies the exact already reconstructed tree
        # node; it cannot introduce an arbitrary masked state for pending work.
        parents = [
            tokens
            for tokens in nodes[pending["tree"]]
            if _hash(tokens.tolist()) == action["before_sha256"]
        ]
        if not parents:
            raise ValueError("equal-ten pending native parent state unavailable")
        local = payload["slots"][pending["attempt"]]["local"]
        tokens = _step(
            model,
            parents[0],
            len(tree["seed"]),
            action,
            _rng(
                payload["semantic_stream_sha256"],
                "child",
                pending["tree"],
                pending["expansion"],
                local,
            ),
        )
        completion = pending["completion"]
        if completion is not None:
            level = action["level"] - 1
            for step in completion["rollout"]:
                if step["level"] != level:
                    raise ValueError("equal-ten pending rollout level differs")
                tokens = _step(model, tokens, len(tree["seed"]), step)
                level -= 1
            if level or PeptideVocabulary().decode(tokens[None, :])[0] != completion["endpoint"]:
                raise ValueError("equal-ten pending clean completion differs")


def verify_native_mp2d_equal_ten(
    initializations,
    corpora,
    history,
    context,
    wave,
    *,
    expected_binding,
    expected_eligible_query_ids,
):
    from amp_challenge.generators.diffusion.native_mp2d_equal_ten import (
        EqualTenMP2DWave,
        run_native_mp2d_equal_ten,
    )

    if type(wave) is not EqualTenMP2DWave or any(
        value is not False
        for value in (
            wave.scientific_evidence_accepted,
            wave.production_eligible,
            wave.campaign_eligible,
        )
    ):
        raise ValueError("equal-ten engineering-only wave type differs")
    if (
        type(wave.record_json) is not str
        or len(wave.record_json.encode()) > 128 * 1024**2
        or hashlib.sha256(wave.record_json.encode()).hexdigest() != wave.sha256
    ):
        raise ValueError("equal-ten wave content seal differs")
    payload = json.loads(wave.record_json)
    if (
        payload["history_sha256"] != history.sha256
        or payload["posterior"] != asdict(expected_binding)
        or payload["artifact"] != "native_mp2d_equal_ten_wave_v2"
    ):
        raise ValueError("equal-ten replay history/posterior identity differs")
    if len(payload["events"]) > 2048 or len(payload["slots"]) > 1020:
        raise ValueError("equal-ten replay inventory exceeds cap")
    counts = reconstruct_equal_ten_schedule(
        initializations,
        corpora,
        history,
        context,
        payload,
        expected_eligible_query_ids=expected_eligible_query_ids,
    )

    class RecordedPosterior:
        binding = expected_binding

        def __init__(self):
            self.index = 0

        def evaluate(self, sequences):
            if self.index >= len(payload["posterior_responses"]):
                raise ValueError("equal-ten replay asked for extra posterior call")
            row = payload["posterior_responses"][self.index]
            self.index += 1
            if list(sequences) != row["sequences"]:
                raise ValueError("equal-ten replay posterior request order differs")
            return NativePosteriorBatch(
                tuple(row["batch"]["sequence_ids"]),
                tuple(
                    NativePosteriorScore(tuple(score["objectives"]), score["feasible"])
                    for score in row["batch"]["scores"]
                ),
                row["batch"]["receipt_sha256"],
            )

    evaluator = RecordedPosterior()
    replayed = run_native_mp2d_equal_ten(
        initializations,
        corpora,
        history,
        context,
        evaluator,
        expected_binding=expected_binding,
        eligible_query_ids=expected_eligible_query_ids,
        deadline=1e12,
        clock=lambda: 0.0,
        _replay_stop=payload["stop_check"],
    )
    if replayed != wave or evaluator.index != len(payload["posterior_responses"]):
        raise ValueError("equal-ten same-engine numerical/receipt replay differs")
    return {
        "reconstructed": True,
        "checkpoint_counts": counts,
        "scientific_evidence_accepted": False,
        "campaign_eligible": False,
        "production_eligible": False,
    }

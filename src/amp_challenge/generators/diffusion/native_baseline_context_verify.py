"""Independent eligibility, training and replacement reconstruction.

Never imports the context-update producer or its selector. Row admission,
ordering, weights, corruption and update control flow are reconstructed here.
The accepted vocabulary/schedule/RNG and native gradient, endpoint and KL
primitives are shared; this is not an independent neural implementation.
External applicability and clock truth remain controller responsibilities.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, fields

import numpy as np

from amp_challenge.generators.diffusion import native_baseline_context_records as records
from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_baseline_operators import (
    BASELINE_CONFIG_SHA256,
    NativeBaselineAdvance,
    NativeBaselineEnsemble,
    NativeBaselineStep,
    NativeCandidatePool,
    NormalizedObjectiveContext,
    _NativeUnit,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeEndpointConfig,
    NativeTransitionState,
    _json_hash,
    _seed,
    _validate_model,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_initialization import (
    TRIPLES,
    AuditedNativeInitialization,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    NativeWeightedReplay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def _same(left, right, label):
    if _json_hash(left) != _json_hash(right):
        raise ValueError("native context reconstructed " + label + " differs")


def _need(condition, label):
    if not condition:
        raise ValueError("native context reconstructed " + label + " differs")


def _keys(value, expected, label):
    _need(type(value) is dict and set(value) == set(expected), label + " schema")


def _unit_document(unit):
    if (
        type(unit) is not _NativeUnit
        or type(unit.initialization) is not AuditedNativeInitialization
    ):
        raise TypeError("native context exact unit/initializer required")
    for field in fields(AuditedNativeInitialization):
        value = getattr(unit.initialization, field.name)
        if field.name in (
            "training_folds",
            "training_sequence_ids",
            "training_union_component_ids",
        ):
            subtype = int if field.name == "training_folds" else str
            _need(
                type(value) is tuple and all(type(item) is subtype for item in value),
                "initializer tuple metadata types",
            )
        elif field.name in ("production_input_eligible", "scientific_evidence_accepted"):
            _need(type(value) is bool, "initializer qualification type")
        elif field.name != "model":
            _need(type(value) is str, "initializer string metadata type")
    if (
        type(unit.triple) is not str
        or unit.triple not in TRIPLES
        or unit.triple != unit.initialization.triple
        or type(unit.sequences) is not tuple
        or not 1 <= len(unit.sequences) <= 1113
        or any(
            type(sequence) is not str
            or not 8 <= len(sequence) <= 50
            or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
            for sequence in unit.sequences
        )
        or tuple(sorted(set(unit.sequences))) != unit.sequences
        or tuple(sorted(sequence_id(sequence) for sequence in unit.sequences))
        != unit.initialization.training_sequence_ids
    ):
        raise ValueError("native context generator corpus/initializer identity differs")
    for model in (unit.model, unit.reference, unit.initialization.model):
        _need(type(model.config) is NativeDenoiserConfig, "exact native model configuration type")
        _validate_model(model, NATIVE_ENDPOINT_DEFAULTS)
    unit.check()
    if (
        canonical_model_logical_hash(unit.initialization.model)
        != unit.initialization.checkpoint_logical_sha256
        or unit.reference_sha256 != unit.initialization.checkpoint_logical_sha256
        or unit.model.config != unit.reference.config
        or unit.model.config != unit.initialization.model.config
    ):
        raise ValueError("native context frozen initializer/reference lineage differs")
    return {
        "triple": unit.triple,
        "sequences": unit.sequences,
        "initialization": {
            field.name: getattr(unit.initialization, field.name)
            for field in fields(AuditedNativeInitialization)
            if field.name != "model"
        },
        "model_config": asdict(unit.model.config),
        "policy_sha256": unit.policy_sha256,
        "reference_sha256": unit.reference_sha256,
    }


def _replacement_lineage(old, new):
    before, after = _unit_document(old), _unit_document(new)
    for field in fields(AuditedNativeInitialization):
        if field.name != "model" and type(getattr(old.initialization, field.name)) is not type(
            getattr(new.initialization, field.name)
        ):
            raise TypeError("native context replacement initializer metadata type differs")
    before.pop("policy_sha256")
    after.pop("policy_sha256")
    _same(after, before, "replacement full initializer/corpus/reference lineage")


def _caller_state(units):
    """Object/value snapshots for caller-owned flags and existing gradients."""
    snapshots = []
    for unit in units:
        for model in (unit.model, unit.reference, unit.initialization.model):
            parameters = []
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                payload = (
                    None
                    if gradient is None
                    else hashlib.sha256(gradient.detach().cpu().numpy().tobytes()).hexdigest()
                )
                parameters.append(
                    (
                        name,
                        id(parameter),
                        parameter.requires_grad,
                        None if gradient is None else id(gradient),
                        None
                        if gradient is None
                        else (
                            str(gradient.dtype),
                            str(gradient.device),
                            tuple(gradient.shape),
                            payload,
                        ),
                    )
                )
            snapshots.append(
                (
                    id(model),
                    tuple(
                        (name, id(module), type(module.training).__name__, module.training)
                        for name, module in model.named_modules()
                    ),
                    tuple(parameters),
                )
            )
    return tuple(snapshots)


def _selected_rows(unit, history, eligibility, mode, step):
    admitted = []
    for row in history.observations:
        if row.query_id not in eligibility.query_ids:
            continue
        # Structural admission has already required successful normalized rows.
        value = math.fsum((0.5 * row.objectives[0], 0.5 * row.objectives[1]))
        if mode == "arcadiamp_style_iterative_d3pm" and value < 0.5:
            continue
        admitted.append((row, value))
    if not admitted:
        return None

    def key(sequence, role):
        return _json_hash(
            [
                "native-baseline-replay-v1",
                history.seed,
                unit.triple,
                history.round_index,
                step,
                role,
                sequence,
            ]
        )

    anchors = tuple(sorted(unit.sequences, key=lambda sequence: key(sequence, "anchor"))[:64])
    revealed = tuple(sorted(admitted, key=lambda pair: key(pair[0].sequence, "revealed"))[:64])
    values = np.asarray([pair[1] for pair in revealed], dtype=np.float64)
    if mode == "diffusion_reward_kl_no_search":
        weights = np.exp(np.clip((values - values.max()) / 0.1, -5.0, 0.0))
    else:
        weights = np.clip(values, 0.05, 1.0)
    normalized = np.concatenate(
        (np.full(len(anchors), 0.5 / len(anchors)), 0.5 * weights / weights.sum())
    )
    return (
        anchors + tuple(pair[0].sequence for pair in revealed),
        normalized,
        tuple(pair[0].query_id for pair in revealed),
    )


def _corruption(model, sequences, weights, *, context_id, seed, ordinal, probes):
    """Rebuild the semantic row-specific mask law without the producer builder."""
    vocabulary = PeptideVocabulary()
    encoded = vocabulary.encode(sequences, max_length=model.config.max_length)
    states = []
    for index, sequence in enumerate(sequences):
        if not model.config.min_length <= len(sequence) <= model.config.max_length:
            raise ValueError("native context replay sequence is outside model support")
        generator = _seed(seed, ordinal, "weighted-replay-" + sequence, 0)
        counts = CosineMaskSchedule().mask_counts(
            np.full(model.config.levels + 1, len(sequence), dtype=np.int64),
            np.arange(model.config.levels + 1),
            total_levels=model.config.levels,
        )
        levels = [
            level
            for level in range(1, model.config.levels + 1)
            if not probes or counts[level] > counts[level - 1]
        ]
        level = int(generator.choice(levels))
        positions = generator.choice(len(sequence), size=int(counts[level]), replace=False)
        tokens = encoded.tokens[index].copy()
        tokens[positions] = MASK_TOKEN_INDEX
        states.append(NativeTransitionState(tokens, len(sequence), level))
    return NativeWeightedReplay(sequences, context_id, tuple(states), weights)


def _replay_document(replay):
    return {
        "sequences": replay.sequences,
        "context_id": replay.context_id,
        "states": [
            {"tokens": state.tokens.tolist(), "length": state.length, "level": state.level}
            for state in replay.states
        ],
        "weights": replay.weights.tolist(),
    }


def _reconstruct_unit(unit, history, eligibility, context, checkpoint):
    working = unit.model
    status = "unchanged_by_declared_schedule"
    steps, selected_ids = [], []
    scheduled = (context.mode == "diffusion_reward_kl_no_search" and history.round_index == 1) or (
        context.mode == "arcadiamp_style_iterative_d3pm" and 2 <= history.round_index <= 28
    )
    if scheduled:
        for index in range(1 if context.mode == "diffusion_reward_kl_no_search" else 4):
            checkpoint()
            selected = _selected_rows(unit, history, eligibility, context.mode, index)
            if selected is None:
                status = "no_supported_revealed_rows_no_update"
                break
            sequences, weights, query_ids = selected
            seed = int(_json_hash([history.seed, unit.triple, "native-training"])[:16], 16)
            ordinal = history.round_index * 4 + index
            replay = _corruption(
                working,
                sequences,
                weights,
                context_id=context.objective.context_sha256,
                seed=seed,
                ordinal=ordinal,
                probes=False,
            )
            probes = _corruption(
                unit.reference,
                sequences,
                weights,
                context_id=context.objective.context_sha256,
                seed=seed,
                ordinal=ordinal,
                probes=True,
            )
            checkpoint()
            direction = propose_weighted_direction(working, replay, NATIVE_ENDPOINT_DEFAULTS)
            gradient = hashlib.sha256()
            for name, tensor in direction.gradients:
                gradient.update(name.encode() + b"\0" + tensor.numpy().tobytes())
            enforce = context.mode == "diffusion_reward_kl_no_search"
            for backtracks in range(
                NATIVE_ENDPOINT_DEFAULTS.maximum_backtracks + 1 if enforce else 1
            ):
                checkpoint()
                candidate = endpoint_candidate(working, direction, backtracks=backtracks)
                diagnostics = weighted_anchor_diagnostics(
                    working, candidate, unit.reference, probes, NATIVE_ENDPOINT_DEFAULTS
                )
                local = diagnostics.local_old_candidate
                frozen = diagnostics.candidate_frozen_reference
                accepted = not enforce or (
                    local.mean <= 0.01
                    and local.p99 <= 0.02
                    and frozen.mean <= 0.05
                    and frozen.p99 <= 0.02
                )
                if accepted:
                    break
            after = (
                canonical_model_logical_hash(candidate) if accepted else direction.base_model_sha256
            )
            steps.append(
                {
                    "replay": _replay_document(replay),
                    "probes": _replay_document(probes),
                    "model_before_sha256": direction.base_model_sha256,
                    "model_after_sha256": after,
                    "gradient_sha256": gradient.hexdigest(),
                    "objective_before": direction.objective_before,
                    "gradient_norm": direction.gradient_norm_before_clip,
                    "accepted": accepted,
                    "backtracks": backtracks,
                    "kl_enforced": enforce,
                    "diagnostics": asdict(diagnostics),
                }
            )
            selected_ids.append(query_ids)
            if not accepted:
                status = "constrained_update_rejected_policy_frozen"
                break
            working = candidate
            status = "updated"
    return {
        "triple": unit.triple,
        "round_index": history.round_index,
        "history_sha256": history.sha256,
        "history_receipt_sha256": history.receipt_sha256,
        "status": status,
        "steps": steps,
        "old_policy_sha256": unit.policy_sha256,
        "new_policy_sha256": canonical_model_logical_hash(working),
        "frozen_reference_sha256": unit.reference_sha256,
        "configuration_sha256": BASELINE_CONFIG_SHA256,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }, selected_ids


_ENVELOPE_KEYS = {
    "artifact",
    "configuration_sha256",
    "context",
    "expected",
    "history",
    "eligibility",
    "numerical_input_sha256",
    "previous_envelope_sha256",
    "status",
    "old_inner",
    "advances",
    "selected_query_ids",
    "new_units",
    "sources",
    "timing",
    "failure",
    "scientific_evidence_accepted",
    "campaign_eligible",
    "production_eligible",
}


def _history(raw):
    _keys(raw, (field.name for field in fields(VerifiedHistorySnapshot)), "raw history")
    _need(type(raw["observations"]) is list, "history observation array")
    observations = []
    for row in raw["observations"]:
        _keys(row, (field.name for field in fields(ChargedObservation)), "charged observation")
        values = row["objectives"]
        _need(values is None or type(values) is list, "charged objective array")
        observations.append(
            ChargedObservation(**{**row, "objectives": None if values is None else tuple(values)})
        )
    result = VerifiedHistorySnapshot(**{**raw, "observations": tuple(observations)})
    _same(records.plain(result), raw, "canonical history")
    return result


def _eligibility(raw):
    _keys(raw, (field.name for field in fields(records.NativeBaselineEligibility)), "eligibility")
    ids = raw["query_ids"]
    _need(type(ids) is list and all(type(value) is str for value in ids), "eligible query array")
    _need(ids == sorted(set(ids)), "canonical unique eligible query array")
    return records.NativeBaselineEligibility(**{**raw, "query_ids": frozenset(ids)})


def _expectations(raw):
    _keys(raw, (field.name for field in fields(records.NativeBaselineExpectations)), "expectations")
    ids = raw["eligible_query_ids"]
    _need(type(ids) is list and all(type(value) is str for value in ids), "expected query array")
    _need(ids == sorted(set(ids)), "canonical expected query array")
    return records.NativeBaselineExpectations(**{**raw, "eligible_query_ids": frozenset(ids)})


def _numerical_digest(context, history, eligibility):
    # Independently reconstructed semantic input; NEVER consumed as an RNG seed.
    observations = []
    for row in history.observations:
        admitted = row.query_id in eligibility.query_ids
        observations.append(
            {
                "sequence": row.sequence,
                "status": row.status,
                "eligible": admitted,
                "objectives": list(row.objectives) if admitted else None,
            }
        )
    return records.document_sha(
        {
            "mode": context.mode,
            "config": BASELINE_CONFIG_SHA256,
            "context": context.objective.context_sha256,
            "seed": history.seed,
            "round": history.round_index,
            "observations": observations,
        }
    )


def _admit(history, eligibility, context, expected):
    history.__post_init__()
    eligibility.__post_init__()
    expected.__post_init__()
    _need(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
            history.previous_wave_head_sha256,
        )
        == (
            context.run_id,
            context.seed,
            context.objective.context_sha256,
            context.oracle_bundle_sha256,
            expected.previous_wave_head_sha256,
        ),
        "external run/seed/context/bundle/head",
    )
    _need(
        history.sha256 == expected.history_sha256 == eligibility.history_sha256,
        "external full raw history",
    )
    _need(
        history.objective_context_sha256
        == expected.objective_context_sha256
        == eligibility.objective_context_sha256,
        "external objective context",
    )
    _need(
        eligibility.source_sha256 == expected.eligibility_source_sha256
        and eligibility.receipt_sha256 == expected.eligibility_receipt_sha256
        and eligibility.query_ids == expected.eligible_query_ids,
        "external exact applicability source/receipt/subset",
    )
    successful = frozenset(
        row.query_id for row in history.observations if row.status == "successful"
    )
    _need(eligibility.query_ids <= successful, "applicability successful subset")


def _inner_document(inner, context):
    if type(inner) is not NativeBaselineEnsemble:
        raise TypeError("native context verifier requires exact pre-update inner ensemble")
    _need(type(inner.context) is NormalizedObjectiveContext, "inner objective type")
    inner.context.__post_init__()
    _need(
        type(inner.seed) is int
        and type(inner.mode) is str
        and type(inner.run_id) is str
        and type(inner.oracle_bundle_sha256) is str,
        "inner exact scalar identity types",
    )
    _need(
        type(inner.config) is NativeEndpointConfig and inner.config == NATIVE_ENDPOINT_DEFAULTS,
        "inner unchanged numerical configuration",
    )
    _need(type(inner._units) is tuple and 1 <= len(inner._units) <= 10, "inner unit inventory")
    units = [_unit_document(unit) for unit in inner._units]
    triples = tuple(unit.triple for unit in inner._units)
    _need(triples == tuple(sorted(set(triples))), "ordered unique mixture triples")
    _need(len({unit.model.config for unit in inner._units}) == 1, "mixture model compatibility")
    _need(
        type(inner.protocol_ten_checkpoint_mixture) is bool
        and inner.protocol_ten_checkpoint_mixture == (triples == TRIPLES),
        "mixture scope",
    )
    training = frozenset(
        sequence_id(sequence) for unit in inner._units for sequence in unit.sequences
    )
    _need(
        type(inner._training_ids) is frozenset and inner._training_ids == training,
        "full generator-training exclusion union",
    )
    _need(type(inner._receipts) is tuple, "inner receipts type")
    _need(
        all(
            type(receipt) is NativeBaselineAdvance
            and type(receipt.steps) is tuple
            and all(type(step) is NativeBaselineStep for step in receipt.steps)
            for receipt in inner._receipts
        ),
        "exact retained advance/step types",
    )
    _need(
        (inner.mode, inner.run_id, inner.seed, inner.oracle_bundle_sha256)
        == (context.mode, context.run_id, context.seed, context.oracle_bundle_sha256),
        "inner external identity",
    )
    _same(
        records.plain(inner.context), records.plain(context.objective), "inner normalized context"
    )
    if inner._history is not None:
        _need(type(inner._history) is VerifiedHistorySnapshot, "inner history type")
        inner._history.__post_init__()
        _need(inner._history.complete, "inner last completed history")
    if inner._pool is not None:
        _need(
            type(inner._pool) is NativeCandidatePool and inner._history is not None,
            "inner pool type/history",
        )
        _need(
            (
                inner._pool.mode,
                inner._pool.round_index,
                inner._pool.history_sha256,
                inner._pool.policy_sha256,
            )
            == (
                inner.mode,
                inner._history.round_index,
                inner._history.sha256,
                inner.policy_identities,
            ),
            "inner pool policy/history binding",
        )
    return {
        "mode": inner.mode,
        "run_id": inner.run_id,
        "seed": inner.seed,
        "objective": records.plain(inner.context),
        "bundle": inner.oracle_bundle_sha256,
        "config": records.plain(inner.config),
        "units": units,
        "history_sha256": None if inner._history is None else inner._history.sha256,
        "pool_sha256": None if inner._pool is None else records.document_sha(inner._pool),
        "receipts_sha256": records.document_sha(inner._receipts),
        "training_ids": sorted(training),
        "ten_checkpoint_mixture": triples == TRIPLES,
    }


def _envelope_document(envelope, context, sources):
    if type(envelope) is not records.ContextEnvelope:
        raise TypeError("native context verifier requires exact envelope")
    document = envelope.document()
    _keys(document, _ENVELOPE_KEYS, "envelope")
    _need(
        document["artifact"] == records.ARTIFACT
        and document["configuration_sha256"] == records.CONFIG_SHA256,
        "new envelope contract",
    )
    _same(document["context"], records.plain(context), "envelope external context")
    _same(document["sources"], sources, "executed source inventory")
    _need(
        all(
            document[key] is False
            for key in ("scientific_evidence_accepted", "campaign_eligible", "production_eligible")
        ),
        "non-promoting envelope scope",
    )
    _need(
        document["status"] in ("completed", "paused_incomplete_history")
        and document["failure"] is None,
        "accepting envelope status; failed work is evidence only",
    )
    history = _history(document["history"])
    eligibility = _eligibility(document["eligibility"])
    expected = _expectations(document["expected"])
    _admit(history, eligibility, context, expected)
    _need(
        document["previous_envelope_sha256"] == expected.previous_update_envelope_sha256,
        "recorded predecessor expectation",
    )
    _need((document["status"] == "completed") == history.complete, "history completeness/status")
    _need(
        document["numerical_input_sha256"] == _numerical_digest(context, history, eligibility),
        "eligible-value numerical input digest",
    )
    _need(
        type(document["advances"]) is list
        and type(document["selected_query_ids"]) is list
        and type(document["new_units"]) is list,
        "update arrays",
    )
    _need(type(document["timing"]) is list and bool(document["timing"]), "producer timing trace")
    last = None
    for item in document["timing"]:
        _keys(item, ("phase", "at_monotonic"), "timing entry")
        _need(type(item["phase"]) is str and bool(item["phase"]), "timing phase")
        at = records.finite_clock(item["at_monotonic"])
        _need(
            at < records.finite_clock(context.deadline_monotonic) and (last is None or at >= last),
            "recorded original-clock ordering/bound",
        )
        last = at
    return document, history, eligibility


def _growth(inner, previous, history, eligibility, context, expected, old_inner):
    if previous is None:
        _need(
            expected.previous_update_envelope_sha256 is None
            and inner._history is None
            and inner._pool is None
            and not inner._receipts,
            "initial trusted predecessor",
        )
        _need(
            all(unit.policy_sha256 == unit.reference_sha256 for unit in inner._units),
            "initial policy equals audited initializer; no unrecorded prior update",
        )
    else:
        prior, prior_history, prior_eligibility = previous
        _need(
            history.sha256 != prior_history.sha256,
            "same-history work must reuse the retained idempotent envelope",
        )
        _need(
            history.observations[: len(prior_history.observations)] == prior_history.observations,
            "prior raw charged prefix",
        )
        _need(
            eligibility.source_sha256 == prior_eligibility.source_sha256,
            "fixed applicability source",
        )
        _need(
            all(
                (row.query_id in eligibility.query_ids)
                == (row.query_id in prior_eligibility.query_ids)
                for row in prior_history.observations
            ),
            "immutable prior-row membership",
        )
        _same(prior["new_units"], old_inner["units"], "trusted predecessor resulting units")
        if prior["status"] == "completed":
            _need(
                inner._history is not None and inner._history.sha256 == prior_history.sha256,
                "trusted predecessor completed history",
            )
            _same(prior["advances"], records.plain(inner._receipts), "trusted predecessor receipts")
        else:
            _same(prior["old_inner"], old_inner, "paused predecessor unchanged inner")
            _need(
                not prior["advances"] and not prior["selected_query_ids"],
                "paused predecessor no SGD",
            )
            _need(
                history.round_index == prior_history.round_index,
                "paused predecessor same unfinished round",
            )
    _need(
        not any(sequence_id(row.sequence) in inner._training_ids for row in history.observations),
        "all-raw-history generator-training exclusion",
    )
    next_round = 1 if inner._history is None else inner._history.round_index + 1
    _need(history.round_index == next_round, "contiguous raw-history round")
    if inner._history is not None:
        _need(
            context.mode == "arcadiamp_style_iterative_d3pm", "frozen reward response prohibition"
        )
        _need(
            history.observations[: len(inner._history.observations)] == inner._history.observations,
            "committed raw-history prefix",
        )
        _need(
            inner._pool is not None and inner._pool.status == "complete",
            "completed previous candidate pool prerequisite",
        )


def verify_context_advance(
    inner,
    previous_envelope,
    envelope,
    *,
    context,
    expected,
    expected_new_units=None,
    monotonic=time.monotonic,
) -> dict:
    """Verify one NEW paused/completed admission against its trusted pre-state.

    The caller must independently supply context/expectations and the actual
    last accepted (including paused) envelope. This is not an issuer or a
    durable chain loader. Failed envelopes do not confer acceptance. The inner
    ensemble's existing pool is trusted pre-state; new pool traces still need
    the separately exposed eligibility-bound native pool verifier.
    """
    if (
        type(context) is not records.NativeBaselineContext
        or type(expected) is not records.NativeBaselineExpectations
    ):
        raise TypeError("native context exact external context/expectations required")
    context.__post_init__()
    expected.__post_init__()
    _need(callable(monotonic), "external monotonic clock")
    deadline = records.finite_clock(context.deadline_monotonic)
    context_before, expected_before = records.plain(context), records.plain(expected)
    sources = records.source_identities()
    _need(records.document_sha(sources) == context.implementation_sha256, "actual source pin")
    old_inner = _inner_document(inner, context)
    old_inner_sha = records.document_sha(old_inner)
    document, history, eligibility = _envelope_document(envelope, context, sources)
    envelope_identity = (envelope.payload, envelope.sha256)
    _same(document["expected"], expected_before, "independently supplied expectations")
    _same(document["old_inner"], old_inner, "actual trusted pre-update ensemble")
    previous = None
    previous_identity = None
    if previous_envelope is not None:
        previous = _envelope_document(previous_envelope, context, sources)
        previous_identity = (previous_envelope.payload, previous_envelope.sha256)
        _need(
            previous_envelope.sha256 == expected.previous_update_envelope_sha256,
            "independently expected predecessor seal",
        )
    _growth(inner, previous, history, eligibility, context, expected, old_inner)
    if expected_new_units is not None:
        _need(
            type(expected_new_units) is tuple and len(expected_new_units) == len(inner._units),
            "staged replacement inventory",
        )
        for old, new in zip(inner._units, expected_new_units, strict=True):
            _replacement_lineage(old, new)
    replacement_before = (
        None
        if expected_new_units is None
        else records.document_sha([_unit_document(unit) for unit in expected_new_units])
    )
    caller_before = _caller_state(inner._units)
    staged_caller_before = None if expected_new_units is None else _caller_state(expected_new_units)
    last_clock = None

    def bindings():
        context.__post_init__()
        expected.__post_init__()
        _same(records.plain(context), context_before, "caller context stability")
        _same(records.plain(expected), expected_before, "caller expectations stability")
        _same(records.source_identities(), sources, "source stability")
        _need(
            type(envelope.payload) is bytes
            and type(envelope.sha256) is str
            and (envelope.payload, envelope.sha256) == envelope_identity,
            "envelope stability",
        )
        if previous_envelope is not None:
            _need(
                type(previous_envelope.payload) is bytes
                and type(previous_envelope.sha256) is str
                and (previous_envelope.payload, previous_envelope.sha256) == previous_identity,
                "predecessor envelope stability",
            )
        _need(
            records.document_sha(_inner_document(inner, context)) == old_inner_sha,
            "trusted pre-update ensemble stability",
        )
        _need(
            _caller_state(inner._units) == caller_before,
            "caller modes/requires-grad/gradient stability",
        )
        if expected_new_units is not None:
            _need(
                records.document_sha([_unit_document(unit) for unit in expected_new_units])
                == replacement_before,
                "staged replacement stability",
            )
            _need(
                _caller_state(expected_new_units) == staged_caller_before,
                "staged caller modes/requires-grad/gradient stability",
            )

    def checkpoint():
        nonlocal last_clock
        now = records.finite_clock(monotonic())
        _need(
            now < deadline and (last_clock is None or now >= last_clock),
            "original external deadline/monotonic ordering",
        )
        _need(
            last_clock is not None or deadline - now <= 7200,
            "standalone entry remaining original deadline exceeds 7200 seconds",
        )
        last_clock = now
        # No external callback follows these checks before work/return.
        bindings()

    checkpoint()
    _need(document["timing"][-1]["at_monotonic"] <= last_clock, "producer/verifier clock ordering")
    if previous is not None:
        _need(
            previous[0]["timing"][-1]["at_monotonic"] <= document["timing"][0]["at_monotonic"],
            "predecessor/current original-clock ordering",
        )
    advances, selections, new_units = [], [], []
    if history.complete:
        for unit in inner._units:
            checkpoint()
            advance, selected = _reconstruct_unit(unit, history, eligibility, context, checkpoint)
            advances.append(advance)
            selections.append(selected)
            new_units.append(
                {**_unit_document(unit), "policy_sha256": advance["new_policy_sha256"]}
            )
    else:
        new_units = old_inner["units"]
    _same(document["advances"], advances, "independent numerical advances")
    _same(document["selected_query_ids"], selections, "independent per-step revealed query order")
    _same(document["new_units"], new_units, "independent replacement identities/lineage")
    if expected_new_units is not None:
        _same(
            [_unit_document(unit) for unit in expected_new_units],
            new_units,
            "actual staged replacement models/lineage",
        )
    checkpoint()
    return document

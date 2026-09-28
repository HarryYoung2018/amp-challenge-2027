"""Independent v2 retention/admission/update reconstruction, not production I/O.

Never imports the v2 selector or update producer. The accepted v1 tree auditor
and native neural primitives are shared, not independently authored networks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, fields

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import _NativeUnit, sequence_id
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_initialization import AuditedNativeInitialization
from amp_challenge.generators.diffusion.native_tr2d2 import NativeTR2D2Ensemble
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2_records import (
    CONFIG_SHA256,
    MAXIMUM_RECORD_BYTES,
    ReplayAdvanceV2,
    ReplayGenerationV2,
)
from amp_challenge.generators.diffusion.native_tree_replay import verify_native_tree_generation
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def _same(left, right, name):
    if _json_hash(left) != _json_hash(right):
        raise ValueError("TR2 replay-v2 reconstructed " + name + " differs")


def _verify_replacement_lineage(old, new):
    if (
        type(old) is not _NativeUnit
        or type(new) is not _NativeUnit
        or type(old.initialization) is not AuditedNativeInitialization
        or type(new.initialization) is not AuditedNativeInitialization
    ):
        raise TypeError("TR2 replay-v2 exact replacement unit/initializer types required")
    if type(new.sequences) is not tuple or new.sequences != old.sequences:
        raise ValueError("TR2 replay-v2 replacement generator corpus differs")
    for field in fields(AuditedNativeInitialization):
        if field.name == "model":
            continue
        old_value = getattr(old.initialization, field.name)
        new_value = getattr(new.initialization, field.name)
        if type(new_value) is not type(old_value):
            raise TypeError("TR2 replay-v2 replacement initializer metadata types differ")
        _same(new_value, old_value, "replacement initializer " + field.name)
    if (
        tuple(sorted(sequence_id(seq) for seq in new.sequences))
        != new.initialization.training_sequence_ids
        or canonical_model_logical_hash(new.initialization.model)
        != old.initialization.checkpoint_logical_sha256
    ):
        raise ValueError("TR2 replay-v2 replacement initializer training/model binding differs")
    for name in ("model", "reference"):
        _same(
            asdict(getattr(new, name).config),
            asdict(getattr(old, name).config),
            "replacement " + name + " configuration",
        )


def _ordered(source, eligible, context):
    remaining, ordered = list(eligible), []
    while remaining:
        values = np.asarray([source.attempts[index].posterior.objectives for index in remaining])
        front = []
        for row, index in enumerate(remaining):
            dominators = (values >= values[row]).all(axis=1) & (values > values[row]).any(axis=1)
            if not dominators.any():
                front.append(index)
        front = sorted(
            front,
            key=lambda index: (
                -context.scalarize(source.attempts[index].posterior.objectives),
                sequence_id(source.attempts[index].path.endpoint),
            ),
        )
        ordered.extend(front)
        remaining = [index for index in remaining if index not in front]
    return ordered[:64]


def verify_replay_generation_v2(
    tree, generation, *, expected_versions, expected_source_sha256, expected_posterior_sha256
):
    if type(tree) is not NativeTR2D2Ensemble or type(generation) is not ReplayGenerationV2:
        raise TypeError("TR2 replay-v2 exact tree/generation record types required")
    if type(expected_versions) is not dict:
        raise TypeError("TR2 replay-v2 exact expected-version dictionary required")
    source = generation.collection
    if (
        generation.configuration_sha256 != CONFIG_SHA256
        or generation.source_sha256 != expected_source_sha256
        or any(
            flag is not False
            for flag in (
                generation.campaign_eligible,
                generation.scientific_evidence_accepted,
                generation.production_eligible,
            )
        )
        or set(expected_versions) != {unit.triple for unit in tree._units}
        or any(
            type(value) is not int or not 0 <= value < source.round_index
            for value in expected_versions.values()
        )
    ):
        raise ValueError("TR2 replay-v2 expected source/actual version inventory differs")
    verify_native_tree_generation(tree, source, expected_posterior_sha256=expected_posterior_sha256)
    _same(
        generation.behavior_versions,
        [(unit.triple, expected_versions[unit.triple]) for unit in tree._units],
        "behavior versions",
    )
    if len(generation.buffers) != len(tree._units):
        raise ValueError("TR2 replay-v2 student inventory differs")
    expansion_owner = {row.expansion_index: row.triple for row in source.expansions}
    for unit, root, observed in zip(tree._units, source.roots, generation.buffers, strict=True):
        eligible = [
            index
            for index, attempt in enumerate(source.attempts)
            if expansion_owner[attempt.expansion_index] == unit.triple
            and attempt.rejection_reason is None
            and attempt.first_attempt_index == index
        ]
        indices = _ordered(source, eligible, tree.context)
        raw = [
            tree.context.scalarize(source.attempts[index].posterior.objectives) / 0.1
            + source.attempts[index].path.reference_log_probability
            - source.attempts[index].path.behavior_log_probability
            for index in indices
        ]
        if not all(map(math.isfinite, raw)):
            raise ValueError("TR2 replay-v2 nonfinite reconstructed path weights")
        relative = [min(0.0, max(-10.0, value - max(raw))) for value in raw]
        values = np.exp(np.array(relative, dtype=np.float64))
        if len(values):
            values /= values.sum()
        ess = float(1 / np.dot(values, values)) if len(values) else 0.0
        maximum = float(values.max()) if len(values) else 0.0
        status = (
            "insufficient_unique_support_no_update"
            if len(indices) < 40
            else "concentrated_actual_weights_no_update"
            if ess / len(indices) < 0.2 or maximum > 0.05
            else "admitted_actual_weights"
        )
        expected = {
            "triple": unit.triple,
            "origin_generation": source.round_index,
            "behavior_version": expected_versions[unit.triple],
            "behavior_sha256": unit.policy_sha256,
            "reference_sha256": unit.reference_sha256,
            "eligible_attempt_indices": eligible,
            "attempt_indices": indices,
            "raw_log_weights": raw,
            "relative_clamped_log_weights": relative,
            "normalized_weights": values.tolist(),
            "effective_sample_size": ess,
            "maximum_weight": maximum,
            "status": status,
        }
        _same(asdict(observed), expected, "selection/path weights/admission")
        if (
            root.behavior_sha256 != unit.policy_sha256
            or root.reference_sha256 != unit.reference_sha256
        ):
            raise ValueError("TR2 replay-v2 source behavior/reference model differs")
    return {
        "reconstructed": True,
        "students": len(generation.buffers),
        "scope": "per_student_real_path_weights_not_exact_IS_or_campaign_authority",
    }


def _replay_document(value):
    return {
        "sha256": value.sha256,
        "sequences": value.sequences,
        "context_id": value.context_id,
        "weights": value.weights.tolist(),
        "states": [
            {"tokens": state.tokens.tolist(), "length": state.length, "level": state.level}
            for state in value.states
        ],
    }


def verify_replay_advance_v2(
    tree,
    generation,
    history,
    receipt,
    *,
    expected_versions,
    expected_new_units,
    expected_new_versions,
    expected_source_sha256,
    expected_previous_head_sha256,
    expected_posterior_sha256,
    reconstruct_private_units=False,
):
    # Explicit proposal-only mode for the v4 aggregate guard reader. Existing
    # callers must still supply and authenticate their actual replacement units.
    if type(reconstruct_private_units) is not bool or (
        reconstruct_private_units != (expected_new_units is None)
    ):
        raise ValueError("TR2 replay-v2 private reconstruction mode differs")
    if (
        type(tree) is not NativeTR2D2Ensemble
        or type(history) is not VerifiedHistorySnapshot
        or type(receipt) is not ReplayAdvanceV2
        or (generation is not None and type(generation) is not ReplayGenerationV2)
    ):
        raise TypeError("TR2 replay-v2 exact update record types required")
    if (
        type(expected_versions) is not dict
        or type(expected_new_versions) is not dict
        or any(type(value) is not int for value in expected_new_versions.values())
    ):
        raise TypeError("TR2 replay-v2 exact integer version dictionaries required")
    history.__post_init__()
    if not history.complete or (
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
        raise ValueError("TR2 replay-v2 update history context/identity differs")
    if any(sequence_id(row.sequence) in tree._training_ids for row in history.observations):
        raise ValueError("TR2 replay-v2 update history has training overlap")
    previous = tree._history
    if (
        (previous is None) != (generation is None)
        or history.round_index != (1 if previous is None else previous.round_index + 1)
        or (
            previous is not None
            and history.observations[: len(previous.observations)] != previous.observations
        )
    ):
        raise ValueError("TR2 replay-v2 update history freshness differs")
    if (
        set(expected_versions) != {unit.triple for unit in tree._units}
        or any(type(value) is not int or value < 0 for value in expected_versions.values())
        or (previous is None and any(expected_versions.values()))
    ):
        raise ValueError("TR2 replay-v2 expected pre-update policy versions differ")
    if generation is not None:
        verify_replay_generation_v2(
            tree,
            generation,
            expected_versions=expected_versions,
            expected_source_sha256=expected_source_sha256,
            expected_posterior_sha256=expected_posterior_sha256,
        )
        if generation.collection.round_index + 1 != history.round_index:
            raise ValueError("TR2 replay-v2 stale behavior generation")
    encoded = receipt.record_json.encode()
    if len(encoded) > MAXIMUM_RECORD_BYTES or hashlib.sha256(encoded).hexdigest() != receipt.sha256:
        raise ValueError("TR2 replay-v2 update receipt byte seal differs")
    document = json.loads(receipt.record_json)
    fixed = {
        "configuration_sha256": CONFIG_SHA256,
        "source_sha256": expected_source_sha256,
        "history_sha256": history.sha256,
        "round_index": history.round_index,
        "previous_generation_sha256": None if generation is None else generation.sha256,
        "status": "atomic_tuple_ready",
        "campaign_eligible": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }
    if set(document) != set(fixed) | {"students"}:
        raise ValueError("TR2 replay-v2 update receipt fields differ")
    _same({key: document[key] for key in fixed}, fixed, "update binding")
    if len(document["students"]) != len(tree._units) or (
        expected_new_units is not None and len(expected_new_units) != len(tree._units)
    ):
        raise ValueError("TR2 replay-v2 update student inventory differs")
    new_versions = dict(expected_versions)
    derived_units = []
    supplied_units = tree._units if reconstruct_private_units else expected_new_units
    for index, (unit, new_unit, row) in enumerate(
        zip(tree._units, supplied_units, document["students"], strict=True)
    ):
        _verify_replacement_lineage(unit, new_unit)
        unit.check()
        buffer = None if generation is None else generation.buffers[index]
        allowed = (
            buffer is not None
            and history.round_index < 29
            and buffer.status == "admitted_actual_weights"
        )
        status = (
            "four_actual_weighted_steps"
            if allowed
            else (
                "unchanged_common_initial"
                if generation is None
                else "unchanged_terminal_round29"
                if history.round_index == 29
                else buffer.status
            )
        )
        lag = None if buffer is None else expected_versions[unit.triple] - buffer.behavior_version
        if lag is not None and not 0 <= lag <= 1:
            raise ValueError("TR2 replay-v2 update true trajectory lag differs")
        expected = {
            "triple": unit.triple,
            "old_policy_sha256": unit.policy_sha256,
            "reference_sha256": unit.reference_sha256,
            "old_version": expected_versions[unit.triple],
            "new_version": expected_versions[unit.triple] + int(allowed),
            "trajectory_version_lag": lag,
            "saved_buffer": None if buffer is None else asdict(buffer),
            "status": status,
        }
        _same({key: row[key] for key in expected}, expected, "student binding/version/status")
        if set(row) != set(expected) | {"steps", "new_policy_sha256"} or len(row["steps"]) != (
            4 if allowed else 0
        ):
            raise ValueError("TR2 replay-v2 student step inventory differs")
        working = unit.model
        if allowed:
            sequences = tuple(
                generation.collection.attempts[i].path.endpoint for i in buffer.attempt_indices
            )
            seed = int(_json_hash([history.seed, unit.triple, "tr2d2-training"])[:16], 16)
            for step, recorded in enumerate(row["steps"]):
                replay = build_weighted_replay(
                    working,
                    sequences,
                    np.array(buffer.normalized_weights),
                    context_id=tree.context.context_sha256,
                    seed=seed,
                    ordinal=history.round_index * 4 + step,
                )
                probes = build_weighted_replay(
                    unit.reference,
                    sequences,
                    np.array(buffer.normalized_weights),
                    context_id=tree.context.context_sha256,
                    seed=seed,
                    ordinal=history.round_index * 4 + step,
                    active_probes=True,
                )
                _same(
                    recorded["replay"], _replay_document(replay), "fresh corruption/frozen weights"
                )
                _same(recorded["probes"], _replay_document(probes), "active probes")
                direction = propose_weighted_direction(working, replay)
                candidate = endpoint_candidate(working, direction)
                gradient = hashlib.sha256()
                for name, tensor in direction.gradients:
                    gradient.update(name.encode() + b"\0" + tensor.numpy().tobytes())
                expected_step = {
                    "before": direction.base_model_sha256,
                    "after": canonical_model_logical_hash(candidate),
                    "gradient_sha256": gradient.hexdigest(),
                    "objective": direction.objective_before,
                    "gradient_norm": direction.gradient_norm_before_clip,
                    "kl_enforced": False,
                    "diagnostics": asdict(
                        weighted_anchor_diagnostics(
                            working, candidate, unit.reference, probes, NATIVE_ENDPOINT_DEFAULTS
                        )
                    ),
                }
                _same(
                    {key: recorded[key] for key in expected_step},
                    expected_step,
                    "actual neural step",
                )
                if set(recorded) != set(expected_step) | {"replay", "probes"}:
                    raise ValueError("TR2 replay-v2 neural record fields differ")
                working = candidate
        final_sha = canonical_model_logical_hash(working)
        if reconstruct_private_units:
            new_unit = copy.copy(unit)
            new_unit.model, new_unit.policy_sha256 = working, final_sha
            derived_units.append(new_unit)
        if (
            new_unit.triple,
            new_unit.policy_sha256,
            row["new_policy_sha256"],
            canonical_model_logical_hash(new_unit.model),
            new_unit.reference_sha256,
        ) != (unit.triple, final_sha, final_sha, final_sha, unit.reference_sha256):
            raise ValueError("TR2 replay-v2 actual committed model identity differs")
        new_versions[unit.triple] += int(allowed)
        unit.check()
        new_unit.check()
    if expected_new_versions != new_versions:
        raise ValueError("TR2 replay-v2 actual committed version inventory differs")
    return {
        "reconstructed": True,
        "students": len(new_versions),
        "scope": "actual_neural_replay_not_external_timing_failure_causality_or_oracle_authentication",
        **(
            {"reconstructed_private_units": tuple(derived_units)}
            if reconstruct_private_units
            else {}
        ),
    }

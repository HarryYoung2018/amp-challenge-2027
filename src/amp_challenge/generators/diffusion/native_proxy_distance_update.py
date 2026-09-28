"""Experimental clipped endpoint fitting with matched conditional-distance gates.

This preserves the native charged teacher and ten-student ensemble. It is a
versioned replacement for the historical update, not a verification of that
update. Sampled residue checks are NOT a whole-peptide distribution guarantee;
the consuming sampler must separately implement its bounded policy mixture.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import numpy as np
import torch

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.distribution_distance_study import (
    AMINO_ACIDS,
    amino_acid_ground_cost,
    categorical_distance_diagnostics,
)
from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX, canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    SharedEndpointUpdate,
    endpoint_rng,
    interpolate_policy,
    selected_anchors,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    TRIPLES,
    EndpointTeacher,
    teacher_admission,
)
from amp_challenge.generators.diffusion.native_weighted_training import build_weighted_replay
from amp_challenge.workflows.peptide_distance_calibration import clipped_dataset_target_loss

METRICS = frozenset(("kl", "reverse_kl", "wasserstein_1", "total_variation", "jensen_shannon"))


def competition_teacher_weights(teacher):
    """Use available positive examples; no arbitrary forty-child admission floor."""
    teacher.__post_init__()
    count = len(teacher.protected_indices)
    if count == 0:
        return {
            "admitted": False,
            "reason": "no_qualifying_positive_children",
            "protected_rows": 0,
        }, None
    logits = np.asarray([row.log_weight for row in teacher.targets], dtype=np.float64)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite training weights")
    return {
        "admitted": True,
        "reason": "available_positive_children",
        "protected_rows": count,
        "effective_sample_size": float(1 / np.sum(weights**2)),
        "maximum_weight": float(weights.max()),
    }, weights


def _batch(model, sequences, weights, teacher, seed):
    replay = build_weighted_replay(
        model,
        sequences,
        weights,
        context_id=teacher.objective_context_sha256,
        seed=seed,
        ordinal=0,
        active_probes=True,
    )
    device = next(model.parameters()).device
    tokens = torch.tensor(np.stack([state.tokens for state in replay.states]), device=device)
    lengths = torch.tensor([state.length for state in replay.states], device=device)
    levels = torch.tensor([state.level for state in replay.states], device=device)
    attention = torch.arange(model.config.max_length, device=device)[None, :] < lengths[:, None]
    masked = tokens == MASK_TOKEN_INDEX
    if torch.any(masked & ~attention) or not torch.all(masked.sum(1) > 0):
        raise ValueError("distance fitting needs active, valid masked residues")
    vocab = PeptideVocabulary()
    if vocab.alphabet != AMINO_ACIDS:
        raise ValueError("ground cost does not match model alphabet")
    clean = torch.tensor(
        vocab.encode(sequences, max_length=model.config.max_length).tokens.copy(), device=device
    )
    targets = torch.where(masked, clean, 0)
    return replay, (tokens, attention, levels, lengths), masked, targets


def _probabilities(model, batch):
    return torch.log_softmax(model(*batch).double(), dim=-1)


def update_proxy_distance_endpoints(
    units,
    teacher: EndpointTeacher,
    *,
    seed: int,
    deadline: float,
    metric: str,
    metric_limit: float,
    clock=time.monotonic,
    training_steps: int = 4,
    learning_rate: float = 2e-4,
    clip_width: float = 0.05,
    max_backtracks: int = 8,
    allow_small_teacher: bool = False,
):
    """Return ten private replacements and a receipt; originals stay immutable.

    Positive teacher weights define clipped weighted endpoint likelihood fitting,
    not on-policy reinforcement learning. Fixed corruption states are shared by
    all optimization steps and backtracks; all five metrics use identical rows.
    Each inspected conditional must satisfy TV<=.05 and the selected metric cap.
    There is deliberately no inherited KL/reference/feasibility gate: such gates
    would confound the declared metric treatment. This scope is explicit below.
    """
    if metric not in METRICS or not math.isfinite(metric_limit) or metric_limit < 0:
        raise ValueError("invalid distance treatment")
    if type(seed) is not int or not 0 <= seed < 2**63 or not math.isfinite(deadline):
        raise ValueError("invalid seed or original deadline")
    if type(training_steps) is not int or not 1 <= training_steps <= 16:
        raise ValueError("training_steps must be in [1,16]")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not 0 < clip_width <= 0.05:
        raise ValueError("invalid learning rate or five-percent clipping width")
    if type(max_backtracks) is not int or not 0 <= max_backtracks <= 8:
        raise ValueError("max_backtracks must be in [0,8]")
    units = tuple(units)
    if tuple(unit.triple for unit in units) != TRIPLES:
        raise ValueError("all ten ordered native students are required")
    if type(teacher) is not EndpointTeacher:
        raise TypeError("native charged endpoint teacher is required")
    teacher.__post_init__()
    for unit in units:
        unit.check()
    training_ids = set().union(*(set(unit.initialization.training_sequence_ids) for unit in units))
    if training_ids.intersection(target.sequence_id for target in teacher.targets):
        raise ValueError("teacher targets overlap generator training namespace")
    original = tuple(unit.policy_sha256 for unit in units)
    admission, weights = (
        competition_teacher_weights if allow_small_teacher else teacher_admission
    )(teacher)
    accepted, chosen, replacements = False, None, units
    status, error = "insufficient_targets_no_update", None
    training, candidates = [], []
    cost = amino_acid_ground_cost()

    def check():
        if clock() >= deadline:
            raise TimeoutError("proxy distance update original deadline")

    try:
        check()
        if admission["admitted"]:
            proposals, probes = [], []
            for unit in units:
                check()
                anchors = selected_anchors(unit, seed)
                sequences = anchors + tuple(target.sequence for target in teacher.targets)
                combined = np.concatenate((np.full(64, 0.5 / 64), 0.5 * weights))
                training_seed = int(
                    endpoint_rng(seed, teacher.semantic_sha256, unit.triple).integers(2**63)
                )
                replay, batch, mask, targets = _batch(
                    unit.model, sequences, combined, teacher, training_seed
                )
                with torch.no_grad():
                    before = _probabilities(unit.model, batch).detach()
                    previous_target = before.gather(-1, targets[..., None]).squeeze(-1)
                proposal = copy.deepcopy(unit.model).eval()
                optimizer = torch.optim.Adam(proposal.parameters(), lr=learning_rate)
                # Mean row reduction then reproduces the declared weighted average.
                advantage = torch.tensor(combined * len(combined), device=before.device)[:, None]
                losses = []
                for _ in range(training_steps):
                    check()
                    optimizer.zero_grad(set_to_none=True)
                    current = (
                        _probabilities(proposal, batch).gather(-1, targets[..., None]).squeeze(-1)
                    )
                    loss = clipped_dataset_target_loss(
                        current, previous_target, advantage, mask, clip_width
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        proposal.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                rows = torch.arange(len(sequences), device=before.device)
                positions = mask.long().argmax(1)
                old_probability = before[rows, positions].exp().cpu().numpy()
                probes.append((batch, rows, positions, old_probability))
                proposals.append(proposal)
                training.append(
                    {
                        "triple": unit.triple,
                        "training_seed": training_seed,
                        "sequences": sequences,
                        "weights": combined.tolist(),
                        "replay_sha256": replay.sha256,
                        "losses": losses,
                        "tokens": batch[0].cpu().tolist(),
                        "lengths": batch[3].cpu().tolist(),
                        "levels": batch[2].cpu().tolist(),
                        "positions": positions.cpu().tolist(),
                        "old_probabilities": old_probability.tolist(),
                        "proposal_sha256": canonical_model_logical_hash(proposal),
                    }
                )
            for backtracks in range(max_backtracks + 1):
                record = {"backtracks": backtracks, "students": []}
                candidates.append(record)
                models, passed = [], True
                for unit, proposal, probe in zip(units, proposals, probes, strict=True):
                    check()
                    candidate = interpolate_policy(unit.model, proposal, backtracks)
                    batch, rows, positions, before = probe
                    with torch.no_grad():
                        after = (
                            _probabilities(candidate, batch)[rows, positions].exp().cpu().numpy()
                        )
                    diagnostics = categorical_distance_diagnostics(before, after, cost)
                    check()
                    row_passed = bool(
                        np.max(diagnostics[metric]) <= metric_limit
                        and np.max(diagnostics["total_variation"]) <= 0.05
                    )
                    passed = passed and row_passed
                    record["students"].append(
                        {
                            "triple": unit.triple,
                            "candidate_sha256": canonical_model_logical_hash(candidate),
                            "new_probabilities": after.tolist(),
                            "passed": row_passed,
                            "distances": {
                                name: values.tolist() for name, values in diagnostics.items()
                            },
                        }
                    )
                    models.append(candidate)
                record["passed"] = passed
                if passed:
                    new_units = []
                    for unit, model in zip(units, models, strict=True):
                        new = copy.copy(unit)
                        new.model, new.policy_sha256 = model, canonical_model_logical_hash(model)
                        new_units.append(new)
                    replacements = tuple(new_units)
                    accepted, chosen, status = True, backtracks, "accepted_guarded_update"
                    break
            if not accepted:
                status = "all_backtracks_rejected_ten_students_unchanged"
        check()
    except (TimeoutError, ValueError, FloatingPointError, RuntimeError, TypeError) as failure:
        replacements, accepted, chosen = units, False, None
        status = (
            "partial_deadline_no_commit"
            if isinstance(failure, TimeoutError)
            else "numerical_or_guard_failure_no_commit"
        )
        error = {"type": type(failure).__name__, "message": str(failure)[:512]}
    for unit in units:
        unit.check()
    if tuple(unit.policy_sha256 for unit in units) != original:
        raise ValueError("original native students changed")
    payload = {
        "artifact": "native_proxy_distance_update_v1",
        "teacher_sha256": teacher.sha256,
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "seed": seed,
        "metric": metric,
        "metric_limit": metric_limit,
        "conditional_tv_limit": 0.05,
        "clip_width": clip_width,
        "training_steps": training_steps,
        "learning_rate": learning_rate,
        "objective": "clipped_positive_weighted_endpoint_likelihood_not_on_policy",
        "probe_scope": "first_masked_residue_per_anchor_and_teacher_sequence",
        "global_five_percent_constraint_proven": False,
        "external_whole_trajectory_mixture_required": True,
        "legacy_operator_reference_kl_and_matched_feasibility_enforced": False,
        "ground_cost_sha256": hashlib.sha256(cost.tobytes()).hexdigest(),
        "old_models": list(zip(TRIPLES, original, strict=True)),
        "new_models": [(unit.triple, unit.policy_sha256) for unit in replacements],
        "admission": admission,
        "training": training,
        "candidates": candidates,
        "accepted": accepted,
        "status": status,
        "backtracks": chosen,
        "error": error,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if accepted and clock() >= deadline:
        replacements, accepted, chosen = units, False, None
        status = "partial_deadline_no_commit"
        payload.update(
            accepted=False,
            status=status,
            backtracks=None,
            new_models=list(zip(TRIPLES, original, strict=True)),
            error={"type": "TimeoutError", "message": "receipt encoding exceeded deadline"},
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt = SharedEndpointUpdate(
        encoded,
        hashlib.sha256(encoded.encode()).hexdigest(),
        status,
        accepted,
        chosen,
        metric in ("kl", "reverse_kl"),
        False,
    )
    return replacements, receipt

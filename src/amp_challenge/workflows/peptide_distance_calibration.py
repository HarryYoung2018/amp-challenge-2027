"""Real-checkpoint clipped-update distance calibration, NOT a search comparison.

Uses charged starting labels only. All metrics are computed post hoc on the same
retained proposals. Calibration is explicitly scoped to sampled masked-residue
conditional distributions, not the full peptide or diffusion-path distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from amp_challenge.evaluation.peptide_proxy_protocol import load_protocol, verify_initial_manifest
from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.distribution_distance_study import (
    AMINO_ACIDS,
    amino_acid_ground_cost,
    categorical_distance_diagnostics,
)
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    canonical_model_logical_hash,
    configure_deterministic_runtime,
)
from amp_challenge.generators.diffusion.native_initialization import (
    load_audited_native_initialization,
)
from amp_challenge.generators.diffusion.native_weighted_training import build_weighted_replay

SCRATCH = Path("/lustre/scratch/users/yonghan.yang/amp_challenge/data/runs")


def clipped_dataset_target_loss(current, previous, advantage, masked, width=0.05):
    """Clipped fitting to observed dataset targets, not on-policy action sampling.

    Retain gradients at ratio=1. Clipping limits the surrogate incentive, not
    actual probability drift; only subsequent distribution checks measure drift.
    """
    if current.shape != previous.shape or masked.shape != current.shape:
        raise ValueError("target log probabilities and mask must have matching shapes")
    if current.ndim != 2 or advantage.shape != (current.shape[0], 1):
        raise ValueError("expected row-by-position probabilities and row advantages")
    if masked.dtype != torch.bool or not torch.all(masked.sum(1) > 0):
        raise ValueError("every row must contain at least one masked residue")
    if not all(torch.isfinite(value).all() for value in (current, previous, advantage)):
        raise ValueError("clipped fitting inputs must be finite")
    if not 0 <= width < 1:
        raise ValueError("clipping width must lie in [0,1)")
    ratio = torch.exp(current - previous.detach())
    objective = torch.minimum(
        ratio * advantage, torch.clamp(ratio, 1 - width, 1 + width) * advantage
    )
    loss = -((objective * masked).sum(1) / masked.sum(1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite clipped training objective")
    return loss


def calibrate(initial: Path, protocol_path: Path, output: Path, triple: str, device: str) -> dict:
    protocol = load_protocol(protocol_path)
    manifest = json.loads(initial.read_bytes())
    verify_initial_manifest(manifest, protocol)
    runtime = configure_deterministic_runtime(manifest["seed"])
    output.mkdir(parents=True, exist_ok=False)
    init = load_audited_native_initialization(
        SCRATCH / "generator-namespace-checkpoints-v1-20260909-cnRHxX",
        SCRATCH / "generator-namespace-checkpoint-audit-v1-20260909-2pogvt/verification.json",
        triple=triple,
    )
    old = init.model.to(device).eval()
    candidate = copy.deepcopy(old)
    rows = manifest["records"][:128]  # deterministic, label-independent subset
    sequences = tuple(row["sequence"] for row in rows)
    replay = build_weighted_replay(
        old,
        sequences,
        np.full(128, 1 / 128),
        context_id="distance-development-calibration",
        seed=manifest["seed"],
        ordinal=0,
        active_probes=True,
    )
    vocab = PeptideVocabulary()
    if vocab.alphabet != AMINO_ACIDS:
        raise ValueError("distance ground metric and native vocabulary differ")
    tokens = torch.tensor(np.stack([row.tokens for row in replay.states]), device=device)
    lengths = torch.tensor([row.length for row in replay.states], device=device)
    levels = torch.tensor([row.level for row in replay.states], device=device)
    attention = torch.arange(old.config.max_length, device=device)[None, :] < lengths[:, None]
    masked = tokens == MASK_TOKEN_INDEX
    if torch.any(masked & ~attention) or not torch.all(masked.sum(1) > 0):
        raise ValueError("replay must mask at least one valid residue in every row")
    clean = torch.tensor(
        vocab.encode(sequences, max_length=old.config.max_length).tokens.copy(), device=device
    )
    targets = torch.where(masked, clean, 0)

    def log_probs(model):
        return torch.log_softmax(model(tokens, attention, levels, lengths).double(), dim=-1)

    with torch.no_grad():
        previous = log_probs(old).detach()
        previous_target = previous.gather(-1, targets[..., None]).squeeze(-1)
    rewards = np.asarray([row["oracle_score"] for row in rows])
    advantage = torch.tensor(
        (rewards - rewards.mean()) / max(float(rewards.std()), 1e-8), device=device
    )[:, None]
    optimizer = torch.optim.Adam(candidate.parameters(), lr=2e-4)
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        current = log_probs(candidate).gather(-1, targets[..., None]).squeeze(-1)
        loss = clipped_dataset_target_loss(current, previous_target, advantage, masked)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    # One fixed masked residue per sequence; same probes for every metric/scale.
    positions = masked.long().argmax(dim=1)
    indices = torch.arange(len(rows), device=device)
    before = previous[indices, positions].exp().cpu().numpy()
    cost = amino_acid_ground_cost()
    np.savez(
        output / "probe_inputs.npz",
        tokens=tokens.cpu().numpy(),
        lengths=lengths.cpu().numpy(),
        levels=levels.cpu().numpy(),
        positions=positions.cpu().numpy(),
        old_probabilities=before,
        ground_cost=cost,
        clean_targets=clean.cpu().numpy(),
        rewards=rewards,
        advantages=advantage.cpu().numpy(),
    )
    torch.save(
        {name: value.detach().cpu() for name, value in candidate.state_dict().items()},
        output / "proposed_model_state.pt",
    )
    result = {
        "artifact": "native_checkpoint_distance_calibration_v1",
        "triple": triple,
        "initial_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": init.checkpoint_file_sha256,
        "initial_model_sha256": canonical_model_logical_hash(old),
        "proposed_model_sha256": canonical_model_logical_hash(candidate),
        "runtime": runtime,
        "objective": "clipped_reward_weighted_dataset_target_fitting_not_on_policy",
        "dropout_during_fitting": False,
        "probe_selection": "first_masked_position_of_first128_frozen_initial_rows",
        "training_steps": 4,
        "training_rows": 128,
        "clip_ratio_width": 0.05,
        "losses": losses,
        "additional_oracle_evaluations": 0,
        "scope": "128_fixed_masked_residue_conditionals_not_whole_peptide_distribution",
        "global_five_percent_constraint_proven": False,
        "method_winner_selected": False,
        "scientific_comparison_runs_completed": 0,
        "ground_cost_sha256": hashlib.sha256(cost.tobytes()).hexdigest(),
        "scales": [],
    }
    old_state, proposed_state = old.state_dict(), candidate.state_dict()
    for exponent in range(9):
        scale = 2.0**-exponent
        model = copy.deepcopy(old)
        model.load_state_dict(
            {
                name: value + scale * (proposed_state[name] - value)
                for name, value in old_state.items()
            }
        )
        with torch.no_grad():
            after = log_probs(model)[indices, positions].exp().cpu().numpy()
        diagnostics = categorical_distance_diagnostics(before, after, cost)
        np.savez(output / f"scale-{exponent}.npz", new_probabilities=after, **diagnostics)
        summary = {
            name: {"mean": float(value.mean()), "max": float(value.max())}
            for name, value in diagnostics.items()
        }
        result["scales"].append(
            {
                "scale": scale,
                "model_sha256": canonical_model_logical_hash(model),
                "distances": summary,
                "sampled_row_tv_within_five_percent": bool(
                    np.max(diagnostics["total_variation"]) <= 0.05
                ),
            }
        )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triple", default="012")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(calibrate(args.initial, args.protocol, args.output, args.triple, args.device)))


if __name__ == "__main__":
    main()

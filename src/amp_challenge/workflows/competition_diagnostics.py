"""Small held-out denoising comparison and shared-budget campaign diagnostics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    load_safetensors_checkpoint,
)
from amp_challenge.generators.diffusion.namespace_checkpoint_train import (
    corrupted_batch,
    objective_for_batch,
)
from amp_challenge.workflows.competition_train import BASE, load_trained_units


def evaluate_historical_oracle(root):
    from amp_challenge.models.competition_oracle import prediction_metrics
    from amp_challenge.workflows.peptide_proxy_initial import EXAMPLES

    rows = [json.loads(line) for line in EXAMPLES.read_text().splitlines()]
    path = EXAMPLES.parent / "oof_predictions.csv"
    with path.open() as stream:
        historical = {
            row["example_id"]: row
            for row in csv.DictReader(stream)
            if row["model"] == "descriptor_logistic"
        }
    if set(historical) != {row["example_id"] for row in rows}:
        raise ValueError("historical oracle evaluated a different example inventory")
    probabilities = []
    for row in rows:
        old = historical[row["example_id"]]
        if (int(old["fold"]), int(old["label"]), old["sequence"]) != (
            row["fold"],
            row["label"],
            row["sequence"],
        ):
            raise ValueError("historical held-out fold or label differs")
        probabilities.append(float(old["probability"]))
    labels = [row["label"] for row in rows]
    report = {
        "historical_descriptor_exact_recipe": prediction_metrics(labels, probabilities),
        "current_ensemble": json.loads((Path(root) / "oracle/report.json").read_text())["metrics"][
            "embedding_ensemble"
        ],
        "historical_predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "same_existing_grouped_folds_not_new_external_biological_validation",
        "historical_method_choice_used_fold4": True,
    }
    (Path(root) / "oracle_baseline_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def evaluate_oracle_device_agreement(root):
    """Re-evaluate existing observations, without feeding anything back to search."""
    from amp_challenge.models.competition_oracle import CompetitionOracle

    root = Path(root)
    initial = json.loads((root / "initial-73121/initial.json").read_text())
    baseline = root / "genetic_algorithm-73121-cpu-complete"
    if not baseline.is_dir():
        baseline = root / "genetic_algorithm-73121"
    events = [
        json.loads(line) for line in (baseline / "campaign/events.jsonl").read_text().splitlines()
    ]
    rows = initial["records"] + [
        event for event in events if event["kind"] == "outcome" and event["status"] == "success"
    ]
    oracle = CompetitionOracle(root / "oracle", device="cuda")
    differences = []
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        actual = oracle.score([row["sequence"] for row in batch])
        differences.extend(abs(actual - np.asarray([row["oracle_score"] for row in batch])))
    report = {
        "rows": len(rows),
        "processor_only_baseline": str(baseline),
        "maximum_absolute_score_difference": float(np.max(differences)),
        "mean_absolute_score_difference": float(np.mean(differences)),
        "scores_differing_by_more_than_1e_6": int(np.count_nonzero(np.asarray(differences) > 1e-6)),
        "scope": "posthoc_cpu_vs_cuda_oracle_check_no_search_feedback_no_new_candidates",
        "additional_validation_score_computations_outside_search_budget": len(rows),
    }
    (root / "oracle_device_agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def evaluate_generators(root, *, device="cpu"):
    torch.set_num_threads(1)
    units, corpora = load_trained_units(Path(root) / "depth6", device=device)
    union = set(seq for corpus in corpora for seq in corpus)
    reports = []
    for ordinal, (unit, corpus) in enumerate(zip(units, corpora, strict=True)):
        older = (
            NativeDenoiser(
                NativeDenoiserConfig(layers=2, hidden_dim=128, attention_heads=4, ffn_dim=384)
            )
            .to(device)
            .eval()
        )
        load_safetensors_checkpoint(older, BASE / f"{ordinal:02d}" / "checkpoint.safetensors")
        heldout = sorted(union - set(corpus))
        losses = {"two_layers": [], "six_layers": []}
        with torch.inference_mode():
            for level in (8, 24, 40, 56):
                for start in range(0, len(heldout), 64):
                    sequences = heldout[start : start + 64]
                    levels = np.full(len(sequences), level, dtype=np.int64)
                    seeds = tuple(
                        640000 + ordinal * 10000 + level * 100 + start + i
                        for i in range(len(sequences))
                    )
                    batch = corrupted_batch(sequences, sequences, levels, seeds)
                    for name, model in (("two_layers", older), ("six_layers", unit.model)):
                        values = (
                            objective_for_batch(model, batch, levels, torch.device(device))
                            .row_losses.cpu()
                            .numpy()
                        )
                        losses[name].extend(map(float, values))
        reports.append(
            {
                "member": unit.triple,
                "heldout_sequences": len(heldout),
                "two_layer_loss": float(np.mean(losses["two_layers"])),
                "six_layer_loss": float(np.mean(losses["six_layers"])),
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    report = {
        "scope": "heldout_generator_folds_masked_token_loss_not_activity",
        "members": reports,
        "mean_two_layer_loss": float(np.mean([r["two_layer_loss"] for r in reports])),
        "mean_six_layer_loss": float(np.mean([r["six_layer_loss"] for r in reports])),
        "comparison_caveat": "same_fold_membership_and_member_seeds_but_new_minibatch_random_stream",
    }
    (Path(root) / "generator_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--historical-oracle", action="store_true")
    parser.add_argument("--oracle-device-agreement", action="store_true")
    args = parser.parse_args()
    if args.oracle_device_agreement:
        evaluate_oracle_device_agreement(args.root)
    elif args.historical_oracle:
        evaluate_historical_oracle(args.root)
    else:
        evaluate_generators(args.root, device=args.device)


if __name__ == "__main__":
    main()

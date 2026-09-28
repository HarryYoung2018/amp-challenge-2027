"""Train/evaluate native evolutionary search against matched competition controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from amp_challenge.evaluation.peptide_proxy_campaign import run_campaign, verify_campaign
from amp_challenge.evaluation.peptide_proxy_protocol import (
    build_initial_manifest,
    fingerprint,
    load_protocol,
)
from amp_challenge.models.competition_oracle import CompetitionOracle

ROOT = Path("/lustre/scratch/users/yonghan.yang/amp_challenge/competition-20260920")
PREVIOUS = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/data/runs/native-proxy-fixed-budget-20260914-Hf77gt/initial"
)
PROTOCOL = Path("configs/search/peptide_proxy_distance_v3.toml")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize_update_distances(records):
    """Summarize recorded accepted probes, without fitting or querying an oracle."""
    backtracks, maxima = {}, {}
    for record in records:
        if not record["accepted"]:
            continue
        chosen = record["backtracks"]
        key = str(chosen)
        backtracks[key] = backtracks.get(key, 0) + 1
        candidate = next(row for row in record["candidates"] if row["backtracks"] == chosen)
        for student in candidate["students"]:
            for metric, values in student["distances"].items():
                maxima[metric] = max(maxima.get(metric, 0.0), float(np.max(values)))
    return {
        "accepted_backtrack_counts": backtracks,
        "accepted_probe_maximum_distances": maxima,
        "probe_scope": "recorded_conditional_residue_probes_not_global_sequence_distances",
    }


class PrefetchEmbeddings:
    """Batch label-free features; scoring still occurs only inside the charged loop."""

    def __init__(self, provider, embedder):
        self.provider, self.embedder = provider, embedder
        self.capabilities = provider.capabilities

    def propose(self, history, batch_size):
        proposals = self.provider.propose(history, batch_size)
        self.embedder.encode(proposals)
        return proposals

    def finalize(self, history):
        return self.provider.finalize(history)


def prepare(root=ROOT, *, seed=73121, device="cuda"):
    from amp_challenge.workflows.competition_train import load_trained_units

    protocol = load_protocol(PROTOCOL)
    output = root / f"initial-{seed}"
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((PREVIOUS / f"seed-{seed}.json").read_text())
    reserve = json.loads((PREVIOUS / f"reserves-{seed}.json").read_text())
    exclusions = json.loads((PREVIOUS / "exclusions.json").read_text())
    _, corpora = load_trained_units(root / "depth6", device="cpu")
    if not set(seq for corpus in corpora for seq in corpus) <= set(exclusions["sequences"]):
        raise ValueError("generator corpus not included in common exclusions")
    oracle = CompetitionOracle(root / "oracle", device=device)
    sequences = [row["sequence"] for row in old["records"]]
    scores = oracle.score(sequences)
    records = [
        {**row, "oracle_score": float(score)}
        for row, score in zip(old["records"], scores, strict=True)
    ]
    initial = build_initial_manifest(
        records,
        seed=seed,
        oracle_id="target_embedding_ensemble_v1",
        oracle_sha256=oracle.model_sha256,
        dataset_sha256=old["dataset_sha256"],
        protocol=protocol,
    )
    write_json(output / "initial.json", initial)
    write_json(output / "reserves.json", reserve)
    write_json(output / "exclusions.json", exclusions)
    write_json(
        output / "preparation.json",
        {
            "initial_count": len(records),
            "same_sequences_as_previous": sequences == [row["sequence"] for row in old["records"]],
            "new_oracle_evaluations": len(records),
            "oracle": oracle.model_sha256,
        },
    )
    return initial


def execute(
    root=ROOT,
    *,
    seed=73121,
    arm="evolutionary_kl",
    disable_updates=False,
    device="cuda",
    output=None,
):
    import torch

    from amp_challenge.generators.diffusion.native_baseline_operators import _NativeUnit
    from amp_challenge.representations.competition_features import CompetitionFeatures
    from amp_challenge.workflows.competition_train import load_trained_units
    from amp_challenge.workflows.native_proxy_baselines import make_baseline_provider
    from amp_challenge.workflows.native_proxy_evolution import load_native_proxy_provider

    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + 7000
    protocol = load_protocol(PROTOCOL)
    inputs = root / f"initial-{seed}"
    initial = json.loads((inputs / "initial.json").read_text())
    reserves = json.loads((inputs / "reserves.json").read_text())["sequences"]
    excluded = frozenset(json.loads((inputs / "exclusions.json").read_text())["sequences"])
    output = Path(output or root / f"{arm}{'-no-updates' if disable_updates else ''}-{seed}")
    output.mkdir(parents=True, exist_ok=False)
    oracle = CompetitionOracle(root / "oracle", device=device)
    repository = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if arm.startswith("evolutionary_"):
        provider = load_native_proxy_provider(
            protocol,
            arm_name=arm,
            run_id=f"competition-{arm}-{seed}",
            seed=seed,
            oracle_sha256=oracle.model_sha256,
            context_sha256=fingerprint(["target-ensemble-mean", oracle.model_sha256]),
            checkpoints=None,
            audit=None,
            repository=repository,
            commit=commit,
            bundle=None,
            output_root=output / "provider",
            reserves=reserves,
            excluded_sequences=excluded,
            deadline=deadline,
            device=device,
            admission_evidence=f"six-layer competition source {commit}",
            model_updates_disabled=disable_updates,
            replay_teacher=True,
            precision_recheck=True,
            trained_release=root / "depth6",
            feature_provider=CompetitionFeatures(oracle.embedder),
            policy_storage=output / "checkpoint-cache",
            allow_small_teacher=True,
        )
    else:
        units = None
        if arm == "categorical":
            initializations, corpora = load_trained_units(root / "depth6", device=device)
            units = tuple(
                _NativeUnit(init, corpus)
                for init, corpus in zip(initializations, corpora, strict=True)
            )
        provider = make_baseline_provider(
            protocol,
            arm_name=arm,
            seed=seed,
            reserves=reserves,
            excluded_sequences=excluded,
            deadline=deadline,
            units=units,
        )
    write_json(
        output / "configuration.json",
        {
            "commit": commit,
            "arm": arm,
            "seed": seed,
            "device": device,
            "disable_updates": disable_updates,
            "initial_manifest": initial["manifest_sha256"],
            "oracle": oracle.model_sha256,
            "depth6": str(root / "depth6"),
            "acquisition_precision": "two_fresh_2048_draws_existing_tested_policy",
            "teacher_admission": "available_positive_children_no_40_child_floor",
            "scope": "one_seed_engineering_comparison_not_conclusive_method_ranking",
        },
    )
    result = run_campaign(
        protocol=protocol,
        initial_manifest=initial,
        provider=PrefetchEmbeddings(provider, oracle.embedder),
        oracle=oracle,
        oracle_id=initial["oracle_id"],
        oracle_sha256=oracle.model_sha256,
        output_dir=output / "campaign",
        max_seconds=deadline - time.monotonic(),
        excluded_sequences=excluded,
    )
    write_json(output / "accounting.json", verify_campaign(output / "campaign", protocol))
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "failure": result["failure"],
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    if result["status"] != "complete":
        raise RuntimeError(f"campaign did not finish: {result['failure']}")
    return result


def evaluate(root=ROOT, *, seed=73121):
    from amp_challenge.workflows.generate import (
        DEFAULT_REFERENCE,
        ReferenceIndex,
        _read_fasta_sequences,
    )

    reference = ReferenceIndex(_read_fasta_sequences(DEFAULT_REFERENCE))
    results = []
    for name in (
        "evolutionary_kl",
        "evolutionary_wasserstein",
        "evolutionary_total_variation",
        "evolutionary_kl-no-updates",
        "categorical",
        "genetic_algorithm",
    ):
        directory = root / f"{name}-{seed}"
        if not (directory / "campaign/result.json").is_file():
            continue
        result = json.loads((directory / "campaign/result.json").read_text())
        events = [
            json.loads(line)
            for line in (directory / "campaign/events.jsonl").read_text().splitlines()
        ]
        scores = [row["oracle_score"] for row in events[0]["manifest"]["records"]]
        scores.extend(
            event["oracle_score"]
            for event in events
            if event["kind"] == "outcome" and event["status"] == "success"
        )
        scores = np.sort(scores)[::-1]
        paid_scores = {
            row["sequence"]: row["oracle_score"] for row in events[0]["manifest"]["records"]
        }
        paid_scores.update(
            {
                event["sequence"]: event["oracle_score"]
                for event in events
                if event["kind"] == "outcome" and event["status"] == "success"
            }
        )
        eligible_scores = []
        for sequence in sorted(
            paid_scores, key=lambda sequence: (-paid_scores[sequence], sequence)
        ):
            if reference.max_ratio(sequence)[0] <= 0.8:
                eligible_scores.append(paid_scores[sequence])
                if len(eligible_scores) == 100:
                    break
        update_summary = {}
        ingests = sorted((directory / "provider/native").glob("ingest-*.json"))
        if ingests:
            updates = json.loads(ingests[-1].read_text())["updates"]
            reasons = {}
            trained = accepted = changed_students = 0
            teacher_children = set()
            teacher_appearances = 0
            future_pairs = {}
            observed = {
                row["sequence"]: (-1, row["oracle_score"])
                for row in events[0]["manifest"]["records"]
            }
            observed.update(
                {
                    event["sequence"]: (event["evaluation_index"], event["oracle_score"])
                    for event in events
                    if event["kind"] == "outcome" and event["status"] == "success"
                }
            )
            for update in updates:
                record = json.loads(update["update"]["record_json"])
                trained += bool(record.get("training"))
                accepted += bool(record["accepted"])
                if record["accepted"]:
                    previous_models = dict(record["old_models"])
                    changed_students += sum(
                        previous_models[triple] != identity
                        for triple, identity in record["new_models"]
                    )
                reason = record["status"]
                reasons[reason] = reasons.get(reason, 0) + 1
                teacher = update["teacher"]
                selected_ids = set(teacher.get("selected_sequence_ids", []))
                cutoff = 16 * teacher["generation"]
                for candidate in teacher.get("candidates", []):
                    if candidate["sequence_id"] not in selected_ids:
                        continue
                    sequence, parent = candidate["sequence"], candidate["parent"]
                    teacher_children.add(sequence)
                    teacher_appearances += bool(record.get("training"))
                    if (
                        sequence in observed
                        and parent in observed
                        and observed[sequence][0] >= cutoff
                        and observed[parent][0] < cutoff
                    ):
                        future_pairs.setdefault(
                            sequence, observed[sequence][1] - observed[parent][1]
                        )
            update_summary = {
                "opportunities": len(updates),
                "reached_training": trained,
                "accepted": accepted,
                "changed_student_checkpoints": changed_students,
                "statuses": reasons,
                "unique_teacher_children": len(teacher_children),
                "teacher_training_appearances": teacher_appearances,
                "teacher_appearance_unit": "one_selected_child_in_one_ten_model_update_not_minibatch_repetitions",
                "subsequently_evaluated_teacher_children": len(future_pairs),
                "subsequent_positive_parent_contrasts": sum(
                    delta > 0 for delta in future_pairs.values()
                ),
                "teacher_validation_scope": "only_later_paid_child_outcomes_with_already_known_parent_not_in_sample_agreement",
                **summarize_update_distances(
                    json.loads(update["update"]["record_json"]) for update in updates
                ),
            }
        results.append(
            {
                "method": name,
                "status": result["status"],
                "evaluations": result["additional_charged_evaluations"],
                "failure": result["failure"],
                "best": float(scores[0]),
                "top10_mean": float(scores[:10].mean()),
                "top100_mean": float(scores[:100].mean()),
                "top100_lower_decile": float(np.quantile(scores[:100], 0.1)),
                "reference_eligible_paid_top_count": len(eligible_scores),
                "reference_eligible_paid_top_mean": float(np.mean(eligible_scores))
                if eligible_scores
                else None,
                "seconds": result["elapsed_seconds"],
                "model_updates": update_summary,
            }
        )
    report = {
        "seed": seed,
        "scope": "one_seed_matched_computational_oracle_not_biological_validation",
        "results": results,
    }
    write_json(root / f"comparison-{seed}.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=73121)
    parser.add_argument("--arm", default="evolutionary_kl")
    parser.add_argument("--disable-updates", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, seed=args.seed, device=args.device)
    elif args.mode == "run":
        execute(
            args.root,
            seed=args.seed,
            arm=args.arm,
            disable_updates=args.disable_updates,
            device=args.device,
            output=args.output,
        )
    else:
        evaluate(args.root, seed=args.seed)


if __name__ == "__main__":
    main()

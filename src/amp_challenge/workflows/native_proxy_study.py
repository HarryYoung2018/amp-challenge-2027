"""Frozen matched scalar-proxy release and actual native study execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from time import monotonic

from amp_challenge.benchmarks.frozen_peptide_proxy import (
    ACCEPTED_EXAMPLES_SHA256,
    FrozenPeptideProxy,
)
from amp_challenge.evaluation.peptide_proxy_campaign import run_campaign, verify_campaign
from amp_challenge.evaluation.peptide_proxy_protocol import (
    build_initial_manifest,
    fingerprint,
    load_protocol,
    verify_initial_manifest,
)
from amp_challenge.workflows.peptide_proxy_initial import CANDIDATES, EXAMPLES, eligible_candidates

SCRATCH = Path("/lustre/scratch/users/yonghan.yang/amp_challenge")
CHECKPOINTS = SCRATCH / "data/runs/generator-namespace-checkpoints-v1-20260909-cnRHxX"
AUDIT = (
    SCRATCH / "data/runs/generator-namespace-checkpoint-audit-v1-20260909-2pogvt/verification.json"
)
BUNDLE = SCRATCH / "models/ampdiffusion-1a862af9"


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def prepare(protocol_path, old_protocol_path, old_initial, output):
    protocol, previous = load_protocol(protocol_path), load_protocol(old_protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    examples = EXAMPLES.read_bytes()
    if hashlib.sha256(examples).hexdigest() != ACCEPTED_EXAMPLES_SHA256:
        raise ValueError("oracle training inventory changed")
    excluded = {json.loads(line)["sequence"] for line in examples.splitlines() if line}
    generator_sources = {}
    for ordinal in range(10):
        directory = CHECKPOINTS / f"{ordinal:02d}"
        payload = (directory / "training_projection.jsonl").read_bytes()
        manifest = json.loads((directory / "manifest.json").read_text())
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["artifacts"]["training_projection.jsonl"]["sha256"]:
            raise ValueError("generator training inventory changed")
        generator_sources[str(directory)] = digest
        excluded.update(json.loads(line)["sequence"] for line in payload.splitlines() if line)
    candidates, counts = eligible_candidates(CANDIDATES.read_bytes(), excluded)
    proxy = FrozenPeptideProxy(old_initial / "oracle/model_states.json")
    proxy.freeze(output / "oracle")
    save(
        output / "exclusions.json",
        {
            "sequences": sorted(excluded),
            "sha256": fingerprint(sorted(excluded)),
            "generator_sources": generator_sources,
        },
    )
    releases = []
    for seed in protocol.seeds:
        old = json.loads((old_initial / f"seed-{seed}.json").read_text())
        verify_initial_manifest(old, previous)
        if any(row["sequence"] in excluded for row in old["records"]):
            raise ValueError(
                "previous frozen initialization overlaps common exclusions; do not silently replace"
            )
        new = build_initial_manifest(
            old["records"],
            seed=seed,
            oracle_id=old["oracle_id"],
            oracle_sha256=old["oracle_sha256"],
            dataset_sha256=old["dataset_sha256"],
            protocol=protocol,
        )
        if new["records"] != old["records"]:
            raise ValueError("initial order or scores changed")
        save(output / f"seed-{seed}.json", new)
        initial_sequences = {row["sequence"] for row in old["records"]}
        reserves = sorted(
            (row["sequence"] for row in candidates if row["sequence"] not in initial_sequences),
            key=lambda seq: (
                fingerprint([protocol.protocol_sha256, seed, "common-reserves", seq]),
                seq,
            ),
        )[:128]
        if len(reserves) != 128:
            raise ValueError("insufficient unscored common reserves")
        save(
            output / f"reserves-{seed}.json",
            {
                "seed": seed,
                "sequences": reserves,
                "sha256": fingerprint(reserves),
                "selection": "sequence_hash_no_reward_access",
            },
        )
        releases.append(
            {
                "seed": seed,
                "old_manifest": old["manifest_sha256"],
                "manifest": new["manifest_sha256"],
                "reserves": fingerprint(reserves),
            }
        )
    report = {
        "protocol": protocol.protocol_sha256,
        "initial_physical_calls": 0,
        "reused_paid_initial_scores": 512 * len(protocol.seeds),
        "candidate_counts": counts,
        "releases": releases,
        "exclusions_sha256": fingerprint(sorted(excluded)),
    }
    save(output / "complete.json", report)
    return report


def execute(
    protocol_path,
    initial_root,
    output,
    *,
    seed,
    arm_name,
    max_seconds,
    device,
    model_updates_disabled=False,
    replay_teacher=False,
    precision_recheck=False,
):
    if type(model_updates_disabled) is not bool or (
        model_updates_disabled and not arm_name.startswith("evolutionary_")
    ):
        raise ValueError("model-update intervention requires a native evolutionary arm")
    if type(replay_teacher) is not bool or (
        replay_teacher and not arm_name.startswith("evolutionary_")
    ):
        raise ValueError("replay teacher intervention requires a native evolutionary arm")
    if type(precision_recheck) is not bool or (precision_recheck and not replay_teacher):
        raise ValueError("precision recheck requires an explicit replay-teacher intervention")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 7200:
        raise ValueError(
            "native campaign requires an original deadline no greater than 7200 seconds"
        )
    started = monotonic()
    deadline = started + max_seconds
    protocol = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    provider = None
    try:
        initial = json.loads((initial_root / f"seed-{seed}.json").read_text())
        verify_initial_manifest(initial, protocol)
        release = json.loads((initial_root / "complete.json").read_text())
        if initial["seed"] != seed or release["protocol"] != protocol.protocol_sha256:
            raise ValueError("initial seed or prepared protocol differs")
        exclusion_doc = json.loads((initial_root / "exclusions.json").read_text())
        excluded = frozenset(exclusion_doc["sequences"])
        if fingerprint(sorted(excluded)) != exclusion_doc["sha256"]:
            raise ValueError("common exclusions changed")
        reserve_doc = json.loads((initial_root / f"reserves-{seed}.json").read_text())
        reserves = tuple(reserve_doc["sequences"])
        if reserve_doc["seed"] != seed or fingerprint(list(reserves)) != reserve_doc["sha256"]:
            raise ValueError("common reserves changed")
        release_seed = next(row for row in release["releases"] if row["seed"] == seed)
        if (
            release_seed["manifest"] != initial["manifest_sha256"]
            or release_seed["reserves"] != reserve_doc["sha256"]
            or release["exclusions_sha256"] != exclusion_doc["sha256"]
        ):
            raise ValueError("prepared common inputs do not match release")
        repository = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository):
            raise ValueError("experiment source must be committed and clean")
        evidence = fingerprint(
            {
                "commit": commit,
                "protocol": protocol.protocol_sha256,
                "initial": initial["manifest_sha256"],
                "reserves": reserve_doc["sha256"],
                "exclusions": exclusion_doc["sha256"],
            }
        )
        if replay_teacher:
            from amp_challenge.generators.diffusion.native_acquisition_precision import SPEC
            from amp_challenge.generators.diffusion.native_replay_teacher import (
                REPLAY_SPEC,
                REPLAY_SPEC_SHA256,
            )

            save(
                output / "intervention.json",
                {
                    "mode": "requalified_positive_replay_v1",
                    "replay_spec": REPLAY_SPEC,
                    "replay_spec_sha256": REPLAY_SPEC_SHA256,
                    "precision_recheck": precision_recheck,
                    "acquisition_precision_spec": SPEC if precision_recheck else None,
                    "acquisition_precision_spec_sha256": fingerprint(SPEC)
                    if precision_recheck
                    else None,
                    "model_updates_disabled": model_updates_disabled,
                    "base_protocol_sha256": protocol.protocol_sha256,
                    "arm": arm_name,
                    "seed": seed,
                    "source_commit": commit,
                    "initial_manifest_sha256": initial["manifest_sha256"],
                    "reserves_sha256": reserve_doc["sha256"],
                    "exclusions_sha256": exclusion_doc["sha256"],
                    "oracle_sha256": initial["oracle_sha256"],
                    "protocol_role": "unchanged_budget_and_input_backbone_with_explicit_intervention",
                    "counterfactual_generation_and_acquisition_retained": not precision_recheck,
                    "candidate_generation_and_acquisition_ranking_retained": True,
                    "distance_thresholds_and_clipping_unchanged": True,
                },
            )
        elif model_updates_disabled:
            save(
                output / "intervention.json",
                {
                    "mode": "no_model_updates",
                    "model_updates_disabled": True,
                    "base_protocol_sha256": protocol.protocol_sha256,
                    "arm": arm_name,
                    "seed": seed,
                    "source_commit": commit,
                    "initial_manifest_sha256": initial["manifest_sha256"],
                    "reserves_sha256": reserve_doc["sha256"],
                    "exclusions_sha256": exclusion_doc["sha256"],
                    "oracle_sha256": initial["oracle_sha256"],
                    "protocol_role": "unchanged_budget_and_input_backbone_with_explicit_intervention",
                    "teacher_and_posterior_retained": True,
                    "counterfactual_generation_and_acquisition_retained": True,
                },
            )
        if arm_name.startswith("evolutionary_"):
            from amp_challenge.workflows.native_proxy_evolution import load_native_proxy_provider

            provider = load_native_proxy_provider(
                protocol,
                arm_name=arm_name,
                run_id=f"native-proxy-{arm_name}-{seed}",
                seed=seed,
                oracle_sha256=initial["oracle_sha256"],
                context_sha256=fingerprint(["scalar-proxy-replicas", initial["oracle_sha256"]]),
                checkpoints=CHECKPOINTS,
                audit=AUDIT,
                repository=repository,
                commit=commit,
                bundle=BUNDLE,
                output_root=output / "provider",
                reserves=reserves,
                excluded_sequences=excluded,
                deadline=deadline,
                device=device,
                admission_evidence=evidence,
                model_updates_disabled=model_updates_disabled,
                replay_teacher=replay_teacher,
                precision_recheck=precision_recheck,
            )
        else:
            from amp_challenge.workflows.native_proxy_baselines import build_baseline_provider

            provider = build_baseline_provider(
                protocol,
                arm_name=arm_name,
                seed=seed,
                initial=initial,
                reserves=reserves,
                excluded_sequences=excluded,
                checkpoints=CHECKPOINTS,
                audit=AUDIT,
                device=device,
                deadline=deadline,
                output_root=output / "provider",
                admission_evidence=evidence,
            )
        proxy = FrozenPeptideProxy(
            initial_root / "oracle/model_states.json", expected_sha256=initial["oracle_sha256"]
        )
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("setup exhausted original full campaign deadline")
        result = run_campaign(
            protocol=protocol,
            initial_manifest=initial,
            provider=provider,
            oracle=proxy,
            oracle_id=initial["oracle_id"],
            oracle_sha256=initial["oracle_sha256"],
            output_dir=output / "campaign",
            max_seconds=remaining,
            excluded_sequences=excluded,
        )
        audit = verify_campaign(output / "campaign", protocol)
        save(output / "accounting.json", audit)
        report = {
            "status": result["status"],
            "failure": result["failure"],
            "seed": seed,
            "arm": arm_name,
            "commit": commit,
            "protocol": protocol.protocol_sha256,
            "whole_elapsed_seconds_including_setup": monotonic() - started,
            "max_seconds_including_setup": max_seconds,
            "charged": result["additional_charged_evaluations"],
            "model_updates_disabled": model_updates_disabled,
            "replay_teacher": replay_teacher,
            "precision_recheck": precision_recheck,
        }
        save(output / "complete.json", report)
        return report
    except BaseException as exc:
        save(
            output / "setup_or_runner_failure.json",
            {"type": type(exc).__name__, "message": str(exc), "elapsed": monotonic() - started},
        )
        raise
    finally:
        if provider is not None and hasattr(provider, "features"):
            provider.features.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/search/peptide_proxy_distance_v2.toml")
    )
    parser.add_argument(
        "--old-protocol", type=Path, default=Path("configs/search/peptide_proxy_distance_v1.toml")
    )
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73121)
    parser.add_argument("--arm", default="evolutionary_kl")
    parser.add_argument("--max-seconds", type=float, default=7000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = (
        prepare(args.protocol, args.old_protocol, args.initial, args.output)
        if args.mode == "prepare"
        else execute(
            args.protocol,
            args.initial,
            args.output,
            seed=args.seed,
            arm_name=args.arm,
            max_seconds=args.max_seconds,
            device=args.device,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

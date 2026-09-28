"""Audit replay admission and matched observed outcomes, without oracle calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from amp_challenge.evaluation.peptide_proxy_campaign import verify_campaign
from amp_challenge.evaluation.peptide_proxy_protocol import load_protocol
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_replay_teacher import (
    REPLAY_SPEC,
    REPLAY_SPEC_SHA256,
    ReplayCandidate,
)
from amp_challenge.workflows.native_proxy_control_report import (
    compare_traces,
    evaluation_trace,
    verify_provider_inputs,
)
from amp_challenge.workflows.native_proxy_report import observed_curve


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate(row):
    return {field.name: row[field.name] for field in fields(ReplayCandidate)}


def audit_teacher(record, previous_bank, generation):
    """Independently reconstruct selection and retention from recorded moments.

    This does not refit the posterior or recompute neural features. It tests
    the recorded current moments, source identities, and sequential replay bank.
    """
    require(record["replay_spec"] == REPLAY_SPEC, "replay specification differs")
    require(record["replay_spec_sha256"] == REPLAY_SPEC_SHA256, "replay specification hash")
    require(record["generation"] == generation, "teacher generation")
    require(record["bank_before_sha256"] == _json_hash(previous_bank), "replay bank chain")
    require(record["bank_before_count"] == len(previous_bank), "replay bank size")
    rows = record["candidates"]
    require(len({row["sequence_id"] for row in rows}) == len(rows), "duplicate child")
    require(
        [candidate(row) for row in rows[: len(previous_bank)]] == previous_bank,
        "original replay lineage changed",
    )
    qualified = []
    require(
        all(row["original_generation"] == generation for row in rows[len(previous_bank) :]),
        "fresh inventory contains unbound historical candidates",
    )
    for row in rows:
        contrast = row["rebuilt_contrast"]
        require(
            row["sequence_id"] == sequence_id(row["sequence"])
            and row["lineage_parent_id"] == sequence_id(row["parent"])
            and row["feasible"] is contrast["feasible"],
            "candidate identity or feasibility",
        )
        require(
            row["rebuilt_generation"] == generation
            and row["posterior_sha256"] == record["posterior_sha256"]
            and 1 <= row["original_generation"] <= generation,
            "stale posterior/origin",
        )
        advantage = contrast["mean"] - math.sqrt(max(0, contrast["variance"]))
        require(
            math.isclose(advantage, row["metric"], rel_tol=1e-12, abs_tol=1e-12),
            "advantage differs from moments",
        )
        accepted = bool(contrast["feasible"] and advantage > 0 and contrast["absolute_risk"] >= 0.5)
        require(
            row["accepted"] is accepted and contrast["accepted"] is accepted,
            "qualification gate differs",
        )
        if accepted:
            qualified.append(row)
    qualified.sort(key=lambda row: (-row["metric"], row["sequence_id"]))
    selected = qualified[:56]
    children = {row["sequence"] for row in selected}
    parents = tuple(
        dict.fromkeys(row["parent"] for row in selected if row["parent"] not in children)
    )[:8]
    require(
        record["protected_children"] == len(selected)
        and record["zero_advantage_parents"] == len(parents),
        "teacher selection size",
    )
    require(
        record["selected_sequence_ids"]
        == [row["sequence_id"] for row in selected] + [sequence_id(parent) for parent in parents],
        "global child ranking differs",
    )
    retained = [candidate(row) for row in qualified[:512]]
    require(
        record["bank_after_sha256"] == _json_hash(retained)
        and record["bank_after_count"] == len(retained),
        "retained replay bank differs",
    )
    maximum = max((row["metric"] for row in selected), default=0)
    for row in selected:
        expected = max(-math.log(2), min(0, (row["metric"] - maximum) / 0.1))
        require(math.isclose(row["logweight"], expected, abs_tol=1e-12), "child weight differs")
    return retained


def audit_mixtures(root, accepted_updates):
    record = read(root / "provider/native/policy-mixture-64.json")
    identity = record.pop("sha256")
    require(_json_hash(record) == identity, "mixture record hash")
    require(len(record["updates"]) == accepted_updates, "mixture update count")
    current = {
        row[0]: {row[1]: Fraction(1)}
        for row in read(root / "provider/native/provisioned.json")["checkpoint_policy_identities"]
    }

    def weights(policy):
        return {
            row["component_id"]: Fraction(row["weight_numerator"], row["weight_denominator"])
            for row in policy["components"]
        }

    for update in record["updates"]:
        require(
            {row["triple"] for row in update} == set(current) and len(update) == 10,
            "partial mixture update",
        )
        for row in update:
            triple = row["triple"]
            require(weights(row["previous_policy"]) == current[triple], "mixture reference changed")
            alpha = Fraction(row["alpha_numerator"], row["alpha_denominator"])
            require(alpha == Fraction(1, 20), "five-percent mixture weight changed")
            expected = {key: value * (1 - alpha) for key, value in current[triple].items()}
            proposal = row["proposal_component_id"]
            expected[proposal] = expected.get(proposal, Fraction(0)) + alpha
            require(weights(row["accepted_policy"]) == expected, "mixture arithmetic/pruning")
            current[triple] = expected
    require(
        {key: weights(value) for key, value in record["mixtures"].items()} == current,
        "final mixture differs",
    )
    return {
        "accepted_ensemble_updates": accepted_updates,
        "exact_update_mass": "1/20",
        "scope": record["bound_scope"],
        "sampler_record_sha256": identity,
    }


def audit_updates(root, seed, protocol):
    arm = next(row for row in protocol.arms if row.name == "evolutionary_kl")
    marker = read(root / "intervention.json")
    require(
        marker.get("precision_recheck", False) is False,
        "teacher-only report cannot silently include an acquisition-policy amendment",
    )
    complete = read(root / "complete.json")
    provisioned = read(root / "provider/native/provisioned.json")
    require(
        marker["source_commit"] == complete["commit"]
        and complete["replay_teacher"] is True
        and provisioned["replay_teacher"] is True,
        "replay execution provenance",
    )
    require(
        marker["mode"] == REPLAY_SPEC["version"]
        and marker["replay_spec_sha256"] == REPLAY_SPEC_SHA256,
        "missing replay intervention",
    )
    require(
        marker["seed"] == seed
        and marker["arm"] == arm.name
        and marker["base_protocol_sha256"] == protocol.protocol_sha256,
        "intervention backbone",
    )
    require(marker["model_updates_disabled"] is False, "updating report received disabled control")
    seen, previous_bank, waves = {}, [], []
    for number in range(1, 66):
        document = read(root / "provider/native" / f"ingest-{number:02d}.json")
        require(
            document["status"] == ("terminal_history" if number == 65 else "ready"),
            "incomplete ingest",
        )
        entries = document["updates"]
        require(len(entries) == min(number - 1, 63), "missing update opportunities")
        identities = []
        for entry in entries:
            receipt = entry["update"]
            identity = receipt["sha256"]
            identities.append(identity)
            digest = _json_hash(entry)
            if identity in seen:
                require(seen[identity] == digest, "cumulative update changed")
                continue
            require(
                hashlib.sha256(receipt["record_json"].encode()).hexdigest() == identity,
                "receipt digest mismatch",
            )
            payload = json.loads(receipt["record_json"])
            previous_bank = audit_teacher(entry["teacher"], previous_bank, number - 1)
            children = entry["teacher"]["protected_children"]
            admission = payload["admission"]["admitted"]
            require(admission is (children >= 40), "admission minimum changed")
            if admission:
                for name in ("protected_targets", "all_targets", "anchors", "combined"):
                    summary = payload["admission"][name]
                    require(
                        summary["passed"] is True
                        and summary["ess_fraction"] >= 0.2
                        and summary["maximum_weight"] <= 0.05 + 1e-15,
                        "weight concentration",
                    )
            require(
                payload["seed"] == seed
                and payload["metric"] == arm.constraint
                and payload["metric_limit"] == arm.threshold
                and payload["conditional_tv_limit"] == 0.05,
                "distance treatment changed",
            )
            require(
                receipt["accepted"] is payload["accepted"]
                and receipt["status"] == payload["status"],
                "receipt envelope differs",
            )
            if payload["accepted"]:
                require(
                    admission and len(payload["training"]) == 10, "accepted without full training"
                )
                selected = payload["candidates"][payload["backtracks"]]
                require(
                    selected["passed"] is True and len(selected["students"]) == 10,
                    "partial distance checks",
                )
                for student in selected["students"]:
                    require(
                        max(student["distances"]["kl"]) <= arm.threshold
                        and max(student["distances"]["total_variation"]) <= 0.05,
                        "accepted distance exceeds unchanged limits",
                    )
            waves.append(
                {
                    "seed": seed,
                    "wave": number - 1,
                    "children": children,
                    "replayed_children": entry["teacher"]["selected_replayed_children"],
                    "bank_size": len(previous_bank),
                    "admitted": admission,
                    "trained_students": len(payload["training"]),
                    "accepted": payload["accepted"],
                    "status": payload["status"],
                    "backtracks": payload["backtracks"],
                }
            )
            seen[identity] = digest
        require(identities == list(seen), "receipt order changed")
    require(len(waves) == 63, "incomplete update history")
    return waves


def campaign(root, protocol, release, seed):
    audit = verify_campaign(root / "campaign", protocol)
    events = [
        json.loads(line) for line in (root / "campaign/events.jsonl").read_text().splitlines()
    ]
    initial = events[0]["manifest"]
    require(initial == read(release / f"seed-{seed}.json"), "initial release differs")
    inputs = verify_provider_inputs(root, release, seed)
    result = read(root / "campaign/result.json")
    require(
        result["status"] == "complete" and result["additional_charged_evaluations"] == 1024,
        "full budget required",
    )
    return {
        "audit": audit,
        "inputs": inputs,
        "initial": initial["manifest_sha256"],
        "oracle": initial["oracle_sha256"],
        "result": result,
        "curve": observed_curve(initial, events, 1024),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol("configs/search/peptide_proxy_distance_v3.toml")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    for seed in protocol.seeds:
        root = args.root / f"evolutionary_kl-{seed}"
        control_root = args.controls / f"evolutionary_kl-{seed}"
        updating = campaign(root, protocol, args.release, seed)
        control = campaign(control_root, protocol, args.release, seed)
        require(
            updating["inputs"] == control["inputs"], "matched model/reserve/exclusion inputs differ"
        )
        require(
            updating["initial"] == control["initial"] and updating["oracle"] == control["oracle"],
            "matched initialization/oracle differs",
        )
        waves = audit_updates(root, seed, protocol)
        mixture = audit_mixtures(root, sum(row["accepted"] for row in waves))
        endpoint, reference = updating["curve"][-1], control["curve"][-1]
        runs.append(
            {
                "seed": seed,
                "decisions": len(waves),
                "admissions": sum(row["admitted"] for row in waves),
                "complete_trainings": sum(row["trained_students"] == 10 for row in waves),
                "accepted_updates": sum(row["accepted"] for row in waves),
                "best_observed_score": endpoint["best_observed_score"],
                "top_10_mean_score": endpoint["top_10_mean_score"],
                "best_difference": endpoint["best_observed_score"]
                - reference["best_observed_score"],
                "top_10_difference": endpoint["top_10_mean_score"] - reference["top_10_mean_score"],
                "failed_evaluations": updating["result"]["failed_evaluations"],
                "cached_repeats": updating["result"]["cached_repeats"],
                "comparison": compare_traces(
                    evaluation_trace(root / "campaign"),
                    evaluation_trace(control_root / "campaign"),
                    1024,
                    16,
                ),
                "waves": waves,
                "mixture_audit": mixture,
            }
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in runs[-1].items()
                    if key not in ("waves", "comparison")
                }
            ),
            flush=True,
        )
    report = {
        "artifact": "requalified_replay_teacher_audit_v1",
        "runs": runs,
        "scope": "recorded_moments_receipts_and_matched_outcomes_not_feature_or_training_reexecution",
        "replay_spec_sha256": REPLAY_SPEC_SHA256,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

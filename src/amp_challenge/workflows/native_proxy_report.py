"""Read-only, common-outcome audit of the prospective frozen-oracle experiment.

No oracle calls, model fitting, or treatment selection. Curves use unique paid
successful non-late observations, including the identical frozen initialization.
Distance receipts are hash-verified and summarized, not numerically re-executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from amp_challenge.evaluation.peptide_proxy_campaign import verify_campaign
from amp_challenge.evaluation.peptide_proxy_protocol import (
    fingerprint,
    load_protocol,
    verify_initial_manifest,
)


def _read(path):
    return json.loads(Path(path).read_text())


def observed_curve(initial, events, charged):
    """One point per logical evaluation; failures/repeats never create extra value."""
    scores = {row["sequence"]: float(row["oracle_score"]) for row in initial["records"]}
    outcomes = {event["evaluation_index"]: event for event in events if event["kind"] == "outcome"}

    def point(budget):
        best = sorted(scores.values(), reverse=True)
        return {
            "additional_evaluations": budget,
            "best_observed_score": best[0],
            "top_10_mean_score": math.fsum(best[:10]) / min(10, len(best)),
            "unique_successful_peptides": len(best),
        }

    curve = [point(0)]
    for index in range(charged):
        event = outcomes.get(index)
        if event is not None and event["status"] == "success" and not event["late"]:
            sequence, score = event["sequence"], float(event["oracle_score"])
            if not math.isfinite(score):
                raise ValueError("nonfinite successful oracle score")
            if sequence in scores:
                raise ValueError(
                    "repeated peptide incorrectly recorded as fresh successful evaluation"
                )
            scores[sequence] = score
        if (
            event is not None
            and event["status"] == "duplicate"
            and (
                event["sequence"] not in scores
                or scores[event["sequence"]] != event["oracle_score"]
            )
        ):
            raise ValueError("duplicate result does not reproduce a prior paid score")
        curve.append(point(index + 1))
    return curve


def summarize_updates(run_root):
    """Deduplicate cumulative ingestion records by authenticated receipt identity."""
    records = {}
    for path in sorted((Path(run_root) / "provider/native").glob("ingest-*.json")):
        for entry in _read(path).get("updates", []):
            receipt = entry["update"]
            raw, identity = receipt["record_json"], receipt["sha256"]
            if hashlib.sha256(raw.encode()).hexdigest() != identity:
                raise ValueError("endpoint update receipt hash mismatch")
            payload = json.loads(raw)
            if payload["accepted"] != receipt["accepted"] or payload["status"] != receipt["status"]:
                raise ValueError("endpoint receipt outer acceptance differs from payload")
            if identity in records and records[identity] != payload:
                raise ValueError("duplicate update receipt has inconsistent payload")
            records[identity] = payload
    summaries = []
    for identity, payload in records.items():
        distances = {}
        chosen = payload.get("backtracks")
        for candidate in payload.get("candidates", []):
            if not payload["accepted"] or candidate["backtracks"] != chosen:
                continue
            for student in candidate["students"]:
                for metric, values in student.get("distances", {}).items():
                    distances.setdefault(metric, []).extend(values)
        summaries.append(
            {
                "receipt_sha256": identity,
                "status": payload["status"],
                "teacher_admitted": payload.get("admission", {}).get("admitted", False),
                "accepted": payload["accepted"],
                "metric": payload.get("metric"),
                "metric_limit": payload.get("metric_limit"),
                "backtracks": chosen,
                "candidate_checks": len(payload.get("candidates", [])),
                "accepted_recorded_distances": {
                    key: {
                        "count": len(values),
                        "mean": math.fsum(values) / len(values),
                        "maximum": max(values),
                    }
                    for key, values in distances.items()
                    if values
                },
            }
        )
    return {
        "unique_updates": len(summaries),
        "teacher_admissions": sum(row["teacher_admitted"] for row in summaries),
        "accepted_updates": sum(row["accepted"] for row in summaries),
        "metric_candidate_checks": sum(row["candidate_checks"] for row in summaries),
        "receipts": summaries,
        "verification_scope": "receipt_hash_and_acceptance_consistency_not_numerical_reexecution",
    }


def audit_run(campaign, protocol, release):
    campaign, release = Path(campaign), Path(release)
    audit = verify_campaign(campaign, protocol)
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    result = _read(campaign / "result.json")
    initial = events[0]["manifest"]
    released = _read(release / f"seed-{initial['seed']}.json")
    verify_initial_manifest(released, protocol)
    if initial != released:
        raise ValueError("run initialization differs from frozen common seed release")
    exclusions = _read(release / "exclusions.json")
    if fingerprint(sorted(exclusions["sequences"])) != exclusions["sha256"]:
        raise ValueError("release exclusions fingerprint differs")
    if events[0]["excluded_sequences"] != sorted(exclusions["sequences"]):
        raise ValueError("run exclusions differ from common release")
    if result["arm_name"] not in {arm.name for arm in protocol.arms}:
        raise ValueError("undeclared treatment")
    for event in events:
        if (
            event["kind"] == "outcome"
            and event["status"] in ("success", "duplicate")
            and event["sequence"] in exclusions["sequences"]
        ):
            raise ValueError("excluded peptide received an accepted outcome")
    curves = observed_curve(initial, events, result["additional_charged_evaluations"])
    updates = summarize_updates(campaign.parent)
    native = result["arm_name"].startswith("evolutionary_")
    failure = result["failure"]
    return {
        "directory": str(campaign.parent.resolve()),
        "seed": result["seed"],
        "arm": result["arm_name"],
        "status": result["status"],
        "failure": failure,
        "algorithmic_stop": bool(failure and failure["type"] == "NativeMethodStopped"),
        "charged_evaluations": result["additional_charged_evaluations"],
        "failed_evaluations": result["failed_evaluations"],
        "cached_repeats": result["cached_repeats"],
        "complete_budget": result["status"] == "complete"
        and result["additional_charged_evaluations"] == 1024,
        "initial_manifest_sha256": initial["manifest_sha256"],
        "oracle_sha256": initial["oracle_sha256"],
        "audit": audit,
        "curve": curves,
        "endpoint": curves[-1],
        "updates": updates,
        "distance_treatment_exercised": native and updates["accepted_updates"] > 0,
        "no_accepted_native_policy_updates": native and updates["accepted_updates"] == 0,
    }


def matched_comparisons(runs, protocol):
    by_key = {}
    for row in runs:
        key = (row["arm"], row["seed"])
        if key in by_key:
            raise ValueError(
                "multiple runs for one treatment/seed; choose a prospective attempt policy, not the best retry"
            )
        by_key[key] = row
    comparisons = []
    for arm in protocol.arms:
        for baseline in ("genetic_algorithm", "categorical"):
            if arm.name == baseline:
                continue
            pairs = []
            for seed in protocol.seeds:
                candidate, reference = by_key.get((arm.name, seed)), by_key.get((baseline, seed))
                if (
                    not candidate
                    or not reference
                    or not candidate["complete_budget"]
                    or not reference["complete_budget"]
                ):
                    continue
                if (candidate["initial_manifest_sha256"], candidate["oracle_sha256"]) != (
                    reference["initial_manifest_sha256"],
                    reference["oracle_sha256"],
                ):
                    raise ValueError("paired runs do not share initialization and scoring oracle")
                pairs.append(
                    {
                        "seed": seed,
                        "best_observed_difference": candidate["endpoint"]["best_observed_score"]
                        - reference["endpoint"]["best_observed_score"],
                        "top_10_mean_difference": candidate["endpoint"]["top_10_mean_score"]
                        - reference["endpoint"]["top_10_mean_score"],
                        "distance_treatment_exercised": candidate["distance_treatment_exercised"],
                        "curve_differences": [
                            {
                                "additional_evaluations": left["additional_evaluations"],
                                "best_observed_difference": left["best_observed_score"]
                                - right["best_observed_score"],
                                "top_10_mean_difference": left["top_10_mean_score"]
                                - right["top_10_mean_score"],
                            }
                            for left, right in zip(
                                candidate["curve"], reference["curve"], strict=True
                            )
                        ],
                    }
                )
            all_matched = len(pairs) == len(protocol.seeds)
            comparisons.append(
                {
                    "arm": arm.name,
                    "baseline": baseline,
                    "matched_complete_seeds": len(pairs),
                    "all_declared_seeds_matched": all_matched,
                    "pairs": pairs,
                    "mean_best_observed_difference": math.fsum(
                        row["best_observed_difference"] for row in pairs
                    )
                    / len(pairs)
                    if pairs
                    else None,
                    "mean_top_10_difference": math.fsum(
                        row["top_10_mean_difference"] for row in pairs
                    )
                    / len(pairs)
                    if pairs
                    else None,
                    "distance_comparison_interpretable": all_matched
                    and arm.name.startswith("evolutionary_")
                    and all(row["distance_treatment_exercised"] for row in pairs),
                    "winner_claimed": False,
                }
            )
    return comparisons


def distance_arm_comparisons(runs, protocol):
    """All three pairings, retaining inactive seeds in the complete cohort.

    Differences are left minus right. Aggregate means require every declared
    seed to have a completed budget in both arms; no active-only mean is made.
    """
    names = ("evolutionary_kl", "evolutionary_wasserstein", "evolutionary_total_variation")
    arms = {arm.name: arm for arm in protocol.arms}
    by_key = {}
    for row in runs:
        key = (row["arm"], row["seed"])
        if key in by_key:
            raise ValueError("multiple runs for one distance treatment/seed")
        by_key[key] = row
    result = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if left not in arms or right not in arms:
                continue
            pairs, missing = [], []
            for seed in protocol.seeds:
                a, b = by_key.get((left, seed)), by_key.get((right, seed))
                if not a or not b or not a["complete_budget"] or not b["complete_budget"]:
                    missing.append(seed)
                    continue
                if (a["initial_manifest_sha256"], a["oracle_sha256"]) != (
                    b["initial_manifest_sha256"],
                    b["oracle_sha256"],
                ):
                    raise ValueError("distance pair differs in initial dataset or oracle")
                pairs.append(
                    {
                        "seed": seed,
                        "best_observed_difference": a["endpoint"]["best_observed_score"]
                        - b["endpoint"]["best_observed_score"],
                        "top_10_mean_difference": a["endpoint"]["top_10_mean_score"]
                        - b["endpoint"]["top_10_mean_score"],
                        "left_update_active": a["distance_treatment_exercised"],
                        "right_update_active": b["distance_treatment_exercised"],
                    }
                )
            complete = len(pairs) == len(protocol.seeds)
            result.append(
                {
                    "left_arm": left,
                    "right_arm": right,
                    "difference_direction": "left_minus_right",
                    "left_metric_threshold": arms[left].threshold,
                    "right_metric_threshold": arms[right].threshold,
                    "matched_complete_seeds": len(pairs),
                    "all_declared_seeds_matched": complete,
                    "missing_or_incomplete_seeds": missing,
                    "pairs": pairs,
                    "mean_best_observed_difference": math.fsum(
                        row["best_observed_difference"] for row in pairs
                    )
                    / len(pairs)
                    if complete
                    else None,
                    "mean_top_10_mean_difference": math.fsum(
                        row["top_10_mean_difference"] for row in pairs
                    )
                    / len(pairs)
                    if complete
                    else None,
                    "both_update_active_seeds": sum(
                        row["left_update_active"] and row["right_update_active"] for row in pairs
                    ),
                    "either_update_active_seeds": sum(
                        row["left_update_active"] or row["right_update_active"] for row in pairs
                    ),
                    "inactive_seeds_retained": True,
                    "active_subset_aggregate_computed": False,
                    "thresholds_claimed_equal_strength": False,
                    "winner_claimed": False,
                }
            )
    return result


def report(protocol_path, release, results_root, output):
    protocol = load_protocol(protocol_path)
    runs, invalid = [], []
    roots = sorted(
        {
            path.parent
            for path in Path(results_root).rglob("events.jsonl")
            if path.parent.name == "campaign"
        }
    )
    for campaign in roots:
        try:
            runs.append(audit_run(campaign, protocol, release))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            invalid.append(
                {
                    "directory": str(campaign.parent.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    failures = [
        {"directory": str(path.parent.resolve()), "failure": _read(path)}
        for path in sorted(Path(results_root).rglob("setup_or_runner_failure.json"))
    ]
    comparisons = matched_comparisons(runs, protocol)
    document = {
        "artifact": "native_proxy_posthoc_report_v1",
        "protocol_sha256": protocol.protocol_sha256,
        "claim_scope": "frozen_computational_activity_proxy_not_biological_validation",
        "metric_scope": "common_unique_paid_observed_scores_not_method_specific_final_rankings",
        "runs": runs,
        "invalid_or_incomplete_journals": invalid,
        "setup_or_runner_failures": failures,
        "planned_runs": len(protocol.arms) * len(protocol.seeds),
        "completed_runs": sum(row["complete_budget"] for row in runs),
        "native_runs_without_accepted_updates": sum(
            row["no_accepted_native_policy_updates"] for row in runs
        ),
        "comparisons": comparisons,
        "distance_arm_comparisons": distance_arm_comparisons(runs, protocol),
        "winner_claimed": False,
        "interpretation": "Exploratory operating points; unmatched or unexercised distance treatments cannot select a winner.",
    }
    document["sha256"] = fingerprint(document)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
    lines = [
        "# Frozen peptide-oracle comparison",
        "",
        document["interpretation"],
        "",
        f"Completed full budgets: {document['completed_runs']}/{document['planned_runs']}.",
        f"Native runs with no accepted model updates: {document['native_runs_without_accepted_updates']}.",
        "",
        "Scores below use a common paid-observation rule, not each method's final ranking.",
        "",
        "| Method | Seed | Status | Additional evaluations | Best observed | Top ten mean | Accepted updates |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['status']} | {row['charged_evaluations']} | {row['endpoint']['best_observed_score']:.6g} | {row['endpoint']['top_10_mean_score']:.6g} | {row['updates']['accepted_updates']} |"
        )
    lines.extend(
        [
            "",
            f"Invalid or incomplete journals: {len(invalid)}. Setup/runner failure records: {len(failures)}.",
            "",
            "No method winner or biological efficacy claim is made. Paired differences and every budget point are retained in report.json.",
        ]
    )
    lines.extend(
        [
            "",
            "## Direct distance-treatment comparisons",
            "",
            "Differences are left minus right. Means retain every declared seed, including seeds with no accepted updates, and are shown only for a fully completed paired cohort. These are exploratory threshold settings, not equal-strength constraints.",
            "",
            "| Left treatment | Right treatment | Complete pairs | Mean best-score difference | Mean top-ten difference | Both update-active |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in document["distance_arm_comparisons"]:
        best = comparison["mean_best_observed_difference"]
        top = comparison["mean_top_10_mean_difference"]
        best_text, top_text = (
            ("not available" if best is None else f"{best:.6g}"),
            ("not available" if top is None else f"{top:.6g}"),
        )
        lines.append(
            f"| {comparison['left_arm']} | {comparison['right_arm']} | {comparison['matched_complete_seeds']} / {len(protocol.seeds)} | {best_text} | {top_text} | {comparison['both_update_active_seeds']} |"
        )
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.protocol, args.release, args.results_root, args.output)
    print(json.dumps({key: result[key] for key in ("completed_runs", "planned_runs", "sha256")}))


if __name__ == "__main__":
    main()

"""Audit the matched no-model-update control without new oracle evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint, load_protocol
from amp_challenge.workflows.native_proxy_report import audit_run

DISTANCE_ARMS = (
    "evolutionary_kl",
    "evolutionary_wasserstein",
    "evolutionary_total_variation",
)
CONTROL_LABEL = "no_model_updates"
TRIPLES = tuple("".join(map(str, row)) for row in combinations(range(5), 3))


def _read(path):
    return json.loads(Path(path).read_text())


def verify_disabled_run(root, protocol, seed, release=None):
    """Require an explicit intervention and authenticated, training-free receipts."""
    root = Path(root)
    marker = _read(root / "intervention.json")
    expected = {
        "model_updates_disabled": True,
        "base_protocol_sha256": protocol.protocol_sha256,
        "arm": "evolutionary_kl",
        "seed": seed,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("missing or mismatched no-model-update intervention")
    if marker.get("model_updates_disabled") is not True:
        raise ValueError("disabled marker must be boolean true")
    provisioned = _read(root / "provider/native/provisioned.json")
    completion = _read(root / "complete.json")
    with (root / "campaign/events.jsonl").open() as stream:
        initialization = json.loads(next(stream))
    initial = initialization["manifest"]
    fixed_models = provisioned["checkpoint_policy_identities"]
    if (
        tuple(row[0] for row in fixed_models) != TRIPLES
        or any(not isinstance(row[1], str) or len(row[1]) != 64 for row in fixed_models)
        or provisioned.get("model_updates_disabled") is not True
    ):
        raise ValueError("control does not provision the full ten frozen models")
    provenance = {
        "source_commit": completion["commit"],
        "initial_manifest_sha256": initial["manifest_sha256"],
        "oracle_sha256": initial["oracle_sha256"],
        "reserves_sha256": provisioned["reserve_sequence_sha256"],
        "exclusions_sha256": fingerprint(initialization["excluded_sequences"]),
    }
    if (
        any(marker.get(key) != value for key, value in provenance.items())
        or provisioned["excluded_sequence_sha256"] != provenance["exclusions_sha256"]
        or completion["seed"] != seed
        or completion["arm"] != "evolutionary_kl"
        or completion["protocol"] != protocol.protocol_sha256
        or initial["seed"] != seed
    ):
        raise ValueError("intervention provenance differs from run or provisioned inputs")
    if release is not None:
        released_initial = _read(Path(release) / f"seed-{seed}.json")
        reserves = _read(Path(release) / f"reserves-{seed}.json")
        exclusions = _read(Path(release) / "exclusions.json")
        if (
            initial != released_initial
            or reserves["seed"] != seed
            or fingerprint(reserves["sequences"]) != reserves["sha256"]
            or reserves["sha256"] != provenance["reserves_sha256"]
            or fingerprint(sorted(exclusions["sequences"])) != exclusions["sha256"]
            or exclusions["sha256"] != provenance["exclusions_sha256"]
        ):
            raise ValueError("intervention differs from historical initial, reserves or exclusions")
    receipts, entries = {}, {}
    paths = sorted((root / "provider/native").glob("ingest-*.json"))
    if [path.name for path in paths] != [
        f"ingest-{number:02d}.json" for number in range(1, protocol.adaptive_rounds + 2)
    ]:
        raise ValueError("missing or extra ingest rounds")
    for number, path in enumerate(paths, 1):
        document = _read(path)
        if document["status"] != (
            "terminal_history" if number == protocol.adaptive_rounds + 1 else "ready"
        ):
            raise ValueError("ingest did not complete")
        identities = []
        for entry in document["updates"]:
            receipt = entry["update"]
            raw, identity = receipt["record_json"], receipt["sha256"]
            identities.append(identity)
            if hashlib.sha256(raw.encode()).hexdigest() != identity:
                raise ValueError("disabled receipt hash mismatch")
            payload = json.loads(raw)
            if (
                receipt["accepted"] is not False
                or payload["accepted"] is not False
                or receipt["status"] != "model_updates_disabled"
                or payload["status"] != "model_updates_disabled"
                or payload.get("model_updates_disabled") is not True
                or payload.get("training_executed") is not False
                or payload.get("distance_checks_executed") is not False
                or payload.get("training") != []
                or payload.get("candidates") != []
                or payload.get("old_models") != fixed_models
                or payload.get("new_models") != fixed_models
                or payload.get("backtracks") is not None
                or receipt.get("backtracks") is not None
                or receipt.get("kl_enforced") is not False
                or receipt.get("operator_guard_present") is not False
            ):
                raise ValueError(
                    "control receipt permits training, distance checks, or model changes"
                )
            entry_identity = fingerprint(entry)
            if identity in entries:
                if entries[identity] != entry_identity:
                    raise ValueError("cumulative disabled receipt envelope changed")
                continue
            if not 2 <= number <= protocol.adaptive_rounds:
                raise ValueError("unexpected initial or terminal disabled decision")
            if entry["teacher"]["generation"] != number - 1 or payload["seed"] != seed:
                raise ValueError("disabled teacher wave or seed differs")
            entries[identity], receipts[identity] = entry_identity, payload
        if identities != list(entries) or len(identities) != len(set(identities)):
            raise ValueError("cumulative disabled decisions reordered, truncated or duplicated")
        if len(entries) != min(number - 1, protocol.adaptive_rounds - 1):
            raise ValueError("missing or extra disabled decisions")
    return {
        "intervention": marker,
        "unique_disabled_decisions": len(receipts),
        "teacher_admissions_without_training": sum(
            row.get("admission", {}).get("admitted", False) for row in receipts.values()
        ),
        "accepted_updates": 0,
        "training_records": 0,
        "distance_checks": 0,
        "receipt_sha256s": sorted(receipts),
        "contiguous_ingests": len(paths),
        "expected_disabled_decisions": protocol.adaptive_rounds - 1,
        "fixed_checkpoint_policy_identities": fixed_models,
        "provenance": provenance,
        "historical_release_inputs_verified": release is not None,
    }


def evaluation_trace(campaign, *, allow_partial=False):
    """Compare paid outcomes, not elapsed times or metadata that differ by execution."""
    rows = {}
    for line in (Path(campaign) / "events.jsonl").read_text().splitlines(keepends=True):
        if allow_partial and not line.endswith("\n"):
            break
        event = json.loads(line)
        if event["kind"] != "outcome":
            continue
        index = event["evaluation_index"]
        if index in rows:
            raise ValueError("duplicate outcome index")
        rows[index] = {key: event[key] for key in ("sequence", "oracle_score", "status", "late")}
    return rows


def compare_traces(updating, disabled, charged, batch_size):
    differences, equal = [], 0
    for index in range(charged):
        left, right = updating.get(index), disabled.get(index)
        if left is not None and right is not None and left == right:
            equal += 1
            continue
        differences.append(
            {
                "evaluation_index": index,
                "additional_evaluation_number": index + 1,
                "wave_number": index // batch_size + 1,
                "updating": left,
                "no_model_updates": right,
            }
        )
    return {
        "compared_evaluations": charged,
        "exactly_equal_sequence_score_status_outcomes": equal,
        "all_evaluations_identical": equal == charged,
        "first_divergence": differences[0] if differences else None,
        "differing_evaluations": len(differences),
        "updating_trace_sha256": fingerprint(sorted(updating.items())),
        "no_model_updates_trace_sha256": fingerprint(sorted(disabled.items())),
        "comparison_scope": "sequence_score_status_and_late_flag_not_runtime_or_source_identity",
    }


def paired_comparison(updating, disabled, updating_trace, disabled_trace, batch_size):
    if not updating["complete_budget"] or not disabled["complete_budget"]:
        raise ValueError("paired comparison requires both full budgets")
    for key in ("seed", "initial_manifest_sha256", "oracle_sha256", "charged_evaluations"):
        if updating[key] != disabled[key]:
            raise ValueError(f"paired runs differ in {key}")
    curve = [
        {
            "additional_evaluations": left["additional_evaluations"],
            "best_observed_difference": left["best_observed_score"] - right["best_observed_score"],
            "top_10_mean_difference": left["top_10_mean_score"] - right["top_10_mean_score"],
        }
        for left, right in zip(updating["curve"], disabled["curve"], strict=True)
    ]
    if any(
        left["additional_evaluations"] != right["additional_evaluations"]
        for left, right in zip(updating["curve"], disabled["curve"], strict=True)
    ):
        raise ValueError("paired curve indices differ")
    trace = compare_traces(
        updating_trace, disabled_trace, updating["charged_evaluations"], batch_size
    )
    zero_updates = updating["updates"]["accepted_updates"] == 0
    return {
        "seed": updating["seed"],
        "updating_arm": updating["arm"],
        "control_label": CONTROL_LABEL,
        "difference_direction": "updating_minus_no_model_updates",
        "best_observed_difference": curve[-1]["best_observed_difference"],
        "top_10_mean_difference": curve[-1]["top_10_mean_difference"],
        "updating_accepted_updates": updating["updates"]["accepted_updates"],
        "historical_zero_update_seed": zero_updates,
        "zero_update_reproduction_matches": trace["all_evaluations_identical"]
        if zero_updates
        else None,
        "evaluation_comparison": trace,
        "curve_differences": curve,
        "source_commit_equal": updating["execution"]["commit"] == disabled["execution"]["commit"],
        "updating_execution": updating["execution"],
        "control_execution": disabled["execution"],
        "equal_walltime_claimed": False,
        "bitwise_model_or_feature_equivalence_claimed": False,
    }


def _execution(root):
    completion = _read(Path(root) / "complete.json")
    if not completion.get("commit"):
        raise ValueError("missing execution source commit")
    return {
        key: completion.get(key)
        for key in (
            "commit",
            "whole_elapsed_seconds_including_setup",
            "max_seconds_including_setup",
        )
    }


def verify_provider_inputs(root, release, seed):
    """Bind both historical and control providers to the same reserved inventory."""
    provisioned = _read(Path(root) / "provider/native/provisioned.json")
    reserves = _read(Path(release) / f"reserves-{seed}.json")
    exclusions = _read(Path(release) / "exclusions.json")
    if (
        reserves["seed"] != seed
        or fingerprint(reserves["sequences"]) != reserves["sha256"]
        or provisioned["reserve_sequence_sha256"] != reserves["sha256"]
        or fingerprint(sorted(exclusions["sequences"])) != exclusions["sha256"]
        or provisioned["excluded_sequence_sha256"] != exclusions["sha256"]
    ):
        raise ValueError("provider reserve or exclusion inventory differs from historical release")
    return {
        "reserves_sha256": reserves["sha256"],
        "exclusions_sha256": exclusions["sha256"],
        "checkpoint_policy_identities": provisioned["checkpoint_policy_identities"],
    }


def prefix_report(protocol_path, historical_root, control_root, output):
    """Observe contiguous paid outcome prefixes; this is not a campaign audit."""
    protocol = load_protocol(protocol_path)
    rows, missing = [], []
    for seed in protocol.seeds:
        control = Path(control_root) / f"evolutionary_kl-{seed}" / "campaign"
        if not (control / "events.jsonl").exists():
            missing.append(seed)
            continue
        trace = evaluation_trace(control, allow_partial=True)
        length = 0
        while length in trace:
            length += 1
        prefix = {index: trace[index] for index in range(length)}
        for arm in DISTANCE_ARMS:
            historical = evaluation_trace(Path(historical_root) / f"{arm}-{seed}" / "campaign")
            historical_prefix = {
                index: historical[index] for index in range(length) if index in historical
            }
            rows.append(
                {
                    "seed": seed,
                    "updating_arm": arm,
                    "control_outcomes_observed": len(trace),
                    "control_contiguous_prefix_length": length,
                    "comparison": compare_traces(
                        historical_prefix, prefix, length, protocol.batch_size
                    ),
                }
            )
    document = {
        "artifact": "native_proxy_control_ongoing_prefix_comparison_v1",
        "is_final_audit": False,
        "complete_budget_claimed": False,
        "claim_scope": "read_only_contiguous_outcome_prefix_not_authenticated_campaign_audit",
        "missing_control_seeds": missing,
        "pairs": rows,
    }
    document["sha256"] = fingerprint(document)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "prefix.json").open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
    return document


def endpoint_means(runs, seeds):
    summaries = []
    for label in (*DISTANCE_ARMS, CONTROL_LABEL):
        rows = [row for row in runs if row["report_label"] == label and row["complete_budget"]]
        identifiers = [row["seed"] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate endpoint seed")
        complete = set(identifiers) == set(seeds)
        summaries.append(
            {
                "treatment": label,
                "completed_seeds": len(rows),
                "all_declared_seeds_matched": complete,
                "mean_best_observed_score": math.fsum(
                    row["endpoint"]["best_observed_score"] for row in rows
                )
                / len(rows)
                if complete
                else None,
                "mean_top_10_score": math.fsum(row["endpoint"]["top_10_mean_score"] for row in rows)
                / len(rows)
                if complete
                else None,
            }
        )
    return summaries


def report(protocol_path, release, historical_root, control_root, output):
    protocol = load_protocol(protocol_path)
    historical_root, control_root = Path(historical_root), Path(control_root)
    runs, invalid, pairs = [], [], []
    by_key, traces = {}, {}
    for label, root, arms in (
        ("historical", historical_root, DISTANCE_ARMS),
        (CONTROL_LABEL, control_root, ("evolutionary_kl",)),
    ):
        for arm in arms:
            for seed in protocol.seeds:
                directory = root / f"{arm}-{seed}"
                try:
                    row = audit_run(directory / "campaign", protocol, release)
                    if row["arm"] != arm or row["seed"] != seed:
                        raise ValueError("run identity differs from requested seed and arm")
                    row["execution"] = _execution(directory)
                    row["provider_inputs"] = verify_provider_inputs(directory, release, seed)
                    row["report_label"] = arm if label == "historical" else CONTROL_LABEL
                    if label == CONTROL_LABEL:
                        row["disabled_verification"] = verify_disabled_run(
                            directory, protocol, seed, release
                        )
                        if row["updates"]["accepted_updates"] != 0:
                            raise ValueError("control contains accepted updates")
                    key = (row["report_label"], seed)
                    by_key[key] = row
                    traces[key] = evaluation_trace(directory / "campaign")
                    runs.append(row)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    invalid.append(
                        {"directory": str(directory), "error": f"{type(exc).__name__}: {exc}"}
                    )
    comparisons = []
    for arm in DISTANCE_ARMS:
        arm_pairs = []
        for seed in protocol.seeds:
            a, b = by_key.get((arm, seed)), by_key.get((CONTROL_LABEL, seed))
            if not a or not b or not a["complete_budget"] or not b["complete_budget"]:
                continue
            if a["provider_inputs"] != b["provider_inputs"]:
                raise ValueError("paired providers differ in checkpoints, reserves or exclusions")
            pair = paired_comparison(
                a, b, traces[(arm, seed)], traces[(CONTROL_LABEL, seed)], protocol.batch_size
            )
            pairs.append(pair)
            arm_pairs.append(pair)
        complete = len(arm_pairs) == len(protocol.seeds)
        comparisons.append(
            {
                "updating_arm": arm,
                "control_label": CONTROL_LABEL,
                "complete_pairs": len(arm_pairs),
                "all_declared_seeds_matched": complete,
                "mean_best_observed_difference": math.fsum(
                    p["best_observed_difference"] for p in arm_pairs
                )
                / len(arm_pairs)
                if complete
                else None,
                "mean_top_10_mean_difference": math.fsum(
                    p["top_10_mean_difference"] for p in arm_pairs
                )
                / len(arm_pairs)
                if complete
                else None,
                "zero_update_seeds": sum(p["historical_zero_update_seed"] for p in arm_pairs),
                "zero_update_reproduction_matches": sum(
                    p["zero_update_reproduction_matches"] is True for p in arm_pairs
                ),
                "inactive_seeds_retained": True,
            }
        )
    document = {
        "artifact": "native_proxy_no_model_updates_comparison_v1",
        "protocol_sha256": protocol.protocol_sha256,
        "claim_scope": "frozen_descriptor_proxy_not_biological_validation",
        "control_label": CONTROL_LABEL,
        "control_arm_identifier_for_manifest_and_random_stream": "evolutionary_kl",
        "intervention": "disable_endpoint_training_only_retain_teacher_acquisition_and_sampling_pipeline",
        "difference_direction": "updating_minus_no_model_updates",
        "equal_walltime_claimed": False,
        "equal_source_commit_claimed": False,
        "winner_claimed": False,
        "runs": runs,
        "invalid_or_incomplete_runs": invalid,
        "completed_control_runs": sum(
            r["report_label"] == CONTROL_LABEL and r["complete_budget"] for r in runs
        ),
        "planned_control_runs": len(protocol.seeds),
        "comparisons": comparisons,
        "pairs": pairs,
        "endpoint_means": endpoint_means(runs, protocol.seeds),
    }
    document["sha256"] = fingerprint(document)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
    lines = [
        "# Native no-model-update control",
        "",
        f"Completed controls: {document['completed_control_runs']} / {document['planned_control_runs']}. Invalid or missing runs: {len(invalid)}.",
        "",
        "The control retains the historical initialization, oracle, counterfactual teacher, acquisition, and sampling pipeline; endpoint model training is explicitly disabled. Its internal Kullback-Leibler arm identifier preserves the manifest and random stream, and does not label it as an updating treatment.",
        "",
        "Differences below are updating minus no model updates. Means require all five seeds, including seeds with no historical accepted update.",
        "",
        "| Updating treatment | Complete pairs | Mean best difference | Mean top-ten difference | Identical zero-update reproductions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        best, top = row["mean_best_observed_difference"], row["mean_top_10_mean_difference"]
        lines.append(
            f"| {row['updating_arm']} | {row['complete_pairs']} | {'unavailable' if best is None else format(best, '.8g')} | {'unavailable' if top is None else format(top, '.8g')} | {row['zero_update_reproduction_matches']} / {row['zero_update_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Absolute endpoint means",
            "",
            "| Treatment | Complete seeds | Mean best score | Mean top-ten score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in document["endpoint_means"]:
        best, top = row["mean_best_observed_score"], row["mean_top_10_score"]
        lines.append(
            f"| {row['treatment']} | {row['completed_seeds']} | {'unavailable' if best is None else format(best, '.9f')} | {'unavailable' if top is None else format(top, '.9f')} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed evaluation reproduction",
            "",
            "| Treatment | Seed | Accepted updates | Best difference | Top-ten difference | First differing additional evaluation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        first = pair["evaluation_comparison"]["first_divergence"]
        lines.append(
            f"| {pair['updating_arm']} | {pair['seed']} | {pair['updating_accepted_updates']} | {pair['best_observed_difference']:.8g} | {pair['top_10_mean_difference']:.8g} | {'none' if first is None else first['additional_evaluation_number']} |"
        )
    lines.extend(
        [
            "",
            "Every additional evaluation is compared for sequence, score, status, and lateness. Exact trace matches do not establish bitwise equivalence of intermediate features or models. Source commits and elapsed times are retained in report.json; equal source and equal wall time are not claimed. These are frozen descriptor-oracle scores, not biological validation.",
        ]
    )
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args()
    if args.prefix_only:
        result = prefix_report(args.protocol, args.historical_root, args.control_root, args.output)
        print(
            json.dumps(
                {"is_final_audit": False, "pairs": len(result["pairs"]), "sha256": result["sha256"]}
            )
        )
        return
    result = report(
        args.protocol, args.release, args.historical_root, args.control_root, args.output
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("completed_control_runs", "planned_control_runs", "sha256")
            }
        )
    )


if __name__ == "__main__":
    main()

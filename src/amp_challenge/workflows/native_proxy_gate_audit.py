"""Count native teacher, training and distance gates without rerunning experiments.

Only complete, contiguous native campaigns are accepted. Weight-concentration
exceptions precede receipt creation in the producer, so their absence is proved
by completed campaigns and all expected decisions, not by missing error rows.
This checks recorded gate values, not neural training or distance recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint, load_protocol

DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[3] / "configs/search/peptide_proxy_distance_v3.toml"
)
TRIPLES = tuple("".join(map(str, row)) for row in combinations(range(5), 3))
COUNTERS = (
    "update_decisions",
    "teacher_distinct_child_failures",
    "teacher_weight_concentration_failures",
    "teacher_admissions",
    "training_reached",
    "training_completed",
    "distance_reached",
    "distance_final_failures",
    "accepted_updates",
    "other_update_failures",
    "distance_candidate_checks",
    "distance_candidate_rejections",
    "distance_student_checks",
    "distance_student_rejections",
    "selected_metric_student_rejections",
    "common_total_variation_student_rejections",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(raw):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(raw, parse_constant=reject)


def _maximum(values):
    require(
        type(values) is list
        and values
        and all(type(value) in (int, float) and math.isfinite(value) for value in values),
        "distance rows must be nonempty and finite",
    )
    return max(values)


def classify_update(entry, *, arm, seed, wave):
    """One decision per originating generation wave, not per student/backtrack."""
    receipt = entry["update"]
    encoded = receipt["record_json"]
    require(
        isinstance(encoded, str)
        and hashlib.sha256(encoded.encode()).hexdigest() == receipt["sha256"],
        "update receipt hash differs",
    )
    payload = parse(encoded)
    require(payload["artifact"] == "native_proxy_distance_update_v1", "wrong receipt artifact")
    require(
        all(payload[key] == receipt[key] for key in ("accepted", "status", "backtracks")),
        "receipt envelope differs",
    )
    require(
        payload["seed"] == seed
        and payload["metric"] == arm.constraint
        and payload["metric_limit"] == arm.threshold
        and payload["conditional_tv_limit"] == 0.05,
        "receipt treatment or seed differs",
    )
    teacher = entry["teacher"]
    require(teacher["generation"] == wave, "teacher generation differs from originating wave")
    require(
        all(
            row["original_generation"] == wave and row["rebuilt_generation"] == wave
            for row in teacher["candidates"]
        ),
        "teacher candidate wave origin differs",
    )
    admission = payload["admission"]
    admitted = admission["admitted"]
    require(type(admitted) is bool and type(payload["accepted"]) is bool, "invalid gate boolean")
    protected = teacher["protected_children"]
    require(type(protected) is int and 0 <= protected <= 56, "invalid distinct child count")
    counts = dict.fromkeys(COUNTERS, 0)
    counts["update_decisions"] = 1
    counts["teacher_admissions"] = int(admitted)
    training, candidates = payload["training"], payload["candidates"]
    if not admitted:
        require(
            admission["reason"] == "insufficient_distinct_protected_targets"
            and admission["protected_rows"] == protected
            and protected < 40
            and not training
            and not candidates
            and payload["status"] == "insufficient_targets_no_update"
            and not payload["accepted"],
            "teacher rejection is not a pre-training distinct-child failure",
        )
        counts["teacher_distinct_child_failures"] = 1
    else:
        require(
            protected >= 40 and admission["reason"] == "source_specific_weight_guards_passed",
            "admitted teacher does not pass distinct-child gate",
        )
        for name in ("protected_targets", "all_targets", "anchors", "combined"):
            summary = admission[name]
            require(
                summary["passed"] is True
                and summary["ess_fraction"] >= 0.2
                and summary["maximum_weight"] <= 0.05 + 1e-15,
                "admitted teacher fails recorded weight guards",
            )
        require(admission["protected_targets"]["rows"] == protected, "protected row count differs")
    counts["training_reached"] = int(bool(training))
    counts["training_completed"] = int(tuple(row["triple"] for row in training) == TRIPLES)
    counts["distance_reached"] = int(bool(candidates))
    require(
        tuple(row["triple"] for row in training) == TRIPLES[: len(training)],
        "training student prefix differs",
    )
    if candidates:
        require(counts["training_completed"] == 1, "distance checks preceded complete training")
    details = []
    for index, candidate in enumerate(candidates):
        require(candidate["backtracks"] == index, "backtracking order differs")
        students = candidate["students"]
        require(
            tuple(row["triple"] for row in students) == TRIPLES,
            "partial student guard checks cannot be counted as complete decisions",
        )
        failures, metric_failures, tv_failures = [], [], []
        for student in students:
            metric_failed = _maximum(student["distances"][arm.constraint]) > arm.threshold
            tv_failed = _maximum(student["distances"]["total_variation"]) > 0.05
            require(
                student["passed"] is (not metric_failed and not tv_failed),
                "student gate disagrees with recorded distances",
            )
            if metric_failed:
                metric_failures.append(student["triple"])
            if tv_failed:
                tv_failures.append(student["triple"])
            if metric_failed or tv_failed:
                failures.append(student["triple"])
        require(candidate["passed"] is (not failures), "candidate gate differs from students")
        counts["distance_candidate_checks"] += 1
        counts["distance_candidate_rejections"] += int(bool(failures))
        counts["distance_student_checks"] += len(students)
        counts["distance_student_rejections"] += len(failures)
        counts["selected_metric_student_rejections"] += len(metric_failures)
        counts["common_total_variation_student_rejections"] += len(tv_failures)
        details.append(
            {
                "backtracks": index,
                "passed": candidate["passed"],
                "rejected_students": failures,
                "selected_metric_rejected_students": metric_failures,
                "common_total_variation_rejected_students": tv_failures,
            }
        )
    if payload["accepted"]:
        require(
            admitted
            and payload["status"] == "accepted_guarded_update"
            and candidates
            and payload["backtracks"] == len(candidates) - 1
            and candidates[-1]["passed"] is True
            and all(row["passed"] is False for row in candidates[:-1]),
            "accepted update is not the first passing candidate",
        )
        counts["accepted_updates"] = 1
    elif payload["status"] == "all_backtracks_rejected_ten_students_unchanged":
        require(
            admitted and len(candidates) == 9 and all(row["passed"] is False for row in candidates),
            "final distance failure lacks exhausted backtracking evidence",
        )
        counts["distance_final_failures"] = 1
    elif admitted:
        counts["other_update_failures"] = 1
    require(
        sum(
            counts[name]
            for name in (
                "teacher_distinct_child_failures",
                "accepted_updates",
                "distance_final_failures",
                "other_update_failures",
            )
        )
        == 1,
        "update decision categories do not partition decisions",
    )
    return {
        "arm": arm.name,
        "seed": seed,
        "originating_wave": wave,
        "observed_at_ingest_round": wave + 1,
        "additional_evaluations_observed": wave * 16,
        "receipt_sha256": receipt["sha256"],
        "teacher_usable_distinct_candidates": len(teacher["candidates"]),
        "teacher_contrast_qualifying_candidates": sum(
            row["accepted"] for row in teacher["candidates"]
        ),
        "teacher_protected_children_after_selection": protected,
        "teacher_admission_reason": admission["reason"],
        "status": payload["status"],
        "accepted_backtracks": payload["backtracks"],
        "counts": counts,
        "distance_candidates": details,
    }


def summarize(rows):
    return {name: sum(row["counts"][name] for row in rows) for name in COUNTERS}


def audit_ingests(documents, *, arm, seed, rounds):
    """Require cumulative, unchanged receipts and attribute newly appended ones."""
    seen, rows = {}, []
    last_round = 0
    for expected, (number, document) in enumerate(documents, 1):
        require(number == expected and number <= rounds + 1, "missing ingest rounds")
        last_round = number
        identities = []
        require(
            document["status"] == ("terminal_history" if number == rounds + 1 else "ready"),
            "ingest did not complete",
        )
        for entry in document["updates"]:
            identity = entry["update"]["sha256"]
            identities.append(identity)
            digest = fingerprint(entry)
            if identity in seen:
                require(seen[identity] == digest, "cumulative receipt changed")
                continue
            require(2 <= number <= rounds, "unexpected initial or terminal policy update")
            rows.append(classify_update(entry, arm=arm, seed=seed, wave=number - 1))
            seen[identity] = digest
        require(
            identities == list(seen) and len(identities) == len(set(identities)),
            "cumulative receipts reordered or truncated",
        )
        require(
            len(seen) == min(max(number - 1, 0), rounds - 1), "missing or extra update decisions"
        )
    require(last_round == rounds + 1, "missing ingest rounds")
    return rows


def render_markdown(report):
    lines = [
        "# Native update-gate audit",
        "",
        "Counts are unique wave-level decisions, not students or backtracking candidates. "
        "Wave w is assessed after its responses, at ingest round w+1. The final wave has no unused policy update.",
        "",
        "## By treatment and seed",
        "",
        "| Treatment | Seed | Decisions | <40 children | Weight failure | Trained | Final distance failure | Accepted | Rejected backtrack candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        c = run["counts"]
        lines.append(
            f"| {run['arm']} | {run['seed']} | {c['update_decisions']} | "
            f"{c['teacher_distinct_child_failures']} | {c['teacher_weight_concentration_failures']} | "
            f"{c['training_reached']} | {c['distance_final_failures']} | {c['accepted_updates']} | "
            f"{c['distance_candidate_rejections']} |"
        )
    lines.extend(
        [
            "",
            "Weight-concentration exceptions occur before receipt creation. Zero such failures here "
            "is conditional on verified complete campaigns and every expected decision being present. "
            "Incomplete campaigns are rejected by this audit, not assigned zero failures.",
            "",
            "A rejected backtracking candidate can belong to an ultimately accepted update. "
            "Selected-distance and common total-variation student failures can overlap and must not be added.",
            "",
            "## Every originating wave",
            "",
            "| Treatment | Seed | Wave | Ingest | Protected children | Teacher reason | Training reached | Final distance failure | Accepted | Backtrack rejections |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["waves"]:
        c = row["counts"]
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['originating_wave']} | "
            f"{row['observed_at_ingest_round']} | {row['teacher_protected_children_after_selection']} | "
            f"{row['teacher_admission_reason']} | {c['training_reached']} | "
            f"{c['distance_final_failures']} | {c['accepted_updates']} | {c['distance_candidate_rejections']} |"
        )
    return "\n".join(lines) + "\n"


def audit_results(results_root, output, *, protocol_path=DEFAULT_PROTOCOL):
    root, output = Path(results_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        protocol = load_protocol(protocol_path)
        native_arms = {arm.name: arm for arm in protocol.arms if arm.method == "evolutionary"}
        inputs, runs, waves, keys = {}, [], [], set()

        def read(path):
            raw = path.read_bytes()
            inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            return parse(raw)

        for directory in sorted(
            path.parent for path in root.glob("*/provider/native/ingest-01.json")
        ):
            run_root = directory.parents[1]
            result = read(run_root / "campaign/result.json")
            arm, seed = result["arm_name"], result["seed"]
            require(arm in native_arms and seed in protocol.seeds, "unexpected native arm or seed")
            require((arm, seed) not in keys, "duplicate treatment/seed run")
            keys.add((arm, seed))
            require(
                result["protocol_sha256"] == protocol.protocol_sha256
                and result["status"] == "complete"
                and result["additional_charged_evaluations"] == protocol.additional_evaluations
                and result["failure"] is None,
                "incomplete campaign cannot exclude unrecorded teacher exceptions",
            )
            documents = (
                (int(path.stem.split("-")[-1]), read(path))
                for path in sorted(directory.glob("ingest-*.json"))
            )
            rows = audit_ingests(
                documents, arm=native_arms[arm], seed=seed, rounds=protocol.adaptive_rounds
            )
            waves.extend(rows)
            runs.append(
                {
                    "run": str(run_root.relative_to(root)),
                    "arm": arm,
                    "seed": seed,
                    "counts": summarize(rows),
                }
            )
        require(
            keys == {(arm, seed) for arm in native_arms for seed in protocol.seeds},
            "complete declared native treatment/seed matrix required",
        )
        report = {
            "artifact": "native_proxy_update_gate_audit_v1",
            "status": "verified",
            "protocol_sha256": protocol.protocol_sha256,
            "counts": summarize(waves),
            "runs": runs,
            "waves": waves,
            "by_arm": {
                arm: summarize([row for row in waves if row["arm"] == arm]) for arm in native_arms
            },
            "by_seed": {
                str(seed): summarize([row for row in waves if row["seed"] == seed])
                for seed in protocol.seeds
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256s": inputs,
            "scope": "receipt_integrity_and_recorded_gate_accounting_not_numerical_distance_reexecution",
            "weight_failure_accounting": "zero_inferred_from_complete_campaigns_and_all_expected_decisions; exceptions_precede_receipts",
            "wave_convention": "originating_wave_w_is_updated_at_ingest_w_plus_one; final_wave_has_no_unused_update",
        }
        report["sha256"] = fingerprint(report)
        (output / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        (output / "report.md").write_text(render_markdown(report))
        return report
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps({"status": "failed", "type": type(error).__name__, "message": str(error)})
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    report = audit_results(args.results_root, args.output, protocol_path=args.protocol)
    print(
        json.dumps(
            {"status": report["status"], "counts": report["counts"], "sha256": report["sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Post-hoc sequence-composition checks on already paid frozen-oracle outcomes.

No scoring calls, fitting, causal conclusions, homology tests or provider changes.
Composition shifts can identify concerns worth further investigation; they cannot
establish whether an optimization gain is biological or an oracle exploit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint, load_protocol
from amp_challenge.workflows.native_proxy_report import audit_run

ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = frozenset("AVILMFWY")
DESCRIPTORS = ("length", "charge_composition_index", "hydrophobic_fraction")


def descriptors(sequence):
    if not isinstance(sequence, str) or not 8 <= len(sequence) <= 50 or set(sequence) - ALPHABET:
        raise ValueError("expected canonical peptide of length eight through fifty")
    length = len(sequence)
    return {
        "length": length,
        "charge_composition_index": (
            sum(sequence.count(aa) for aa in "KR") - sum(sequence.count(aa) for aa in "DE")
        )
        / length,
        "hydrophobic_fraction": sum(sequence.count(aa) for aa in HYDROPHOBIC) / length,
        "length_at_upper_boundary_50": length == 50,
    }


def summarize(rows):
    rows = tuple(rows)
    if not rows or len({row["sequence"] for row in rows}) != len(rows):
        raise ValueError("summary requires nonempty unique peptides")
    values = [descriptors(row["sequence"]) for row in rows]
    return {
        "peptides": len(rows),
        "unique_sequences": len({row["sequence"] for row in rows}),
        "exact_unique_fraction": len({row["sequence"] for row in rows}) / len(rows),
        "mean_oracle_score": math.fsum(row["oracle_score"] for row in rows) / len(rows),
        "mean": {key: math.fsum(row[key] for row in values) / len(values) for key in DESCRIPTORS},
        "minimum": {key: min(row[key] for row in values) for key in DESCRIPTORS},
        "maximum": {key: max(row[key] for row in values) for key in DESCRIPTORS},
        "length_50_count": sum(row["length_at_upper_boundary_50"] for row in values),
        "length_50_fraction": sum(row["length_at_upper_boundary_50"] for row in values)
        / len(values),
        "homology_assessed": False,
        "exact_uniqueness_is_not_sequence_dissimilarity": True,
    }


def sequence_summary(initial, events):
    """Deterministic score ranking, ties resolved by canonical sequence text."""
    initial_rows = [
        {"sequence": row["sequence"], "oracle_score": float(row["oracle_score"])}
        for row in initial["records"]
    ]
    observed = {row["sequence"]: row for row in initial_rows}
    for event in events:
        if event["kind"] == "outcome" and event["status"] == "success" and not event["late"]:
            if event["sequence"] in observed:
                raise ValueError("fresh successful evaluation duplicates a paid sequence")
            score = float(event["oracle_score"])
            if not math.isfinite(score):
                raise ValueError("nonfinite paid score")
            observed[event["sequence"]] = {"sequence": event["sequence"], "oracle_score": score}

    def ranking(rows):
        return sorted(rows, key=lambda row: (-row["oracle_score"], row["sequence"]))

    initial_top = ranking(initial_rows)[:10]
    final_top = ranking(observed.values())[:10]
    initial_ids = {row["sequence"] for row in initial_rows}
    before, after = summarize(initial_top), summarize(final_top)
    return {
        "initial_all": summarize(initial_rows),
        "initial_top_10": before,
        "final_top_10": after,
        "final_best_sequence": {**final_top[0], **descriptors(final_top[0]["sequence"])},
        "initial_best_sequence": {**initial_top[0], **descriptors(initial_top[0]["sequence"])},
        "final_top_10_sequences": [
            {
                **row,
                **descriptors(row["sequence"]),
                "from_initial_dataset": row["sequence"] in initial_ids,
            }
            for row in final_top
        ],
        "top_10_new_sequences": sum(row["sequence"] not in initial_ids for row in final_top),
        "all_observed_unique_sequences": len(observed),
        "top_10_mean_descriptor_change_from_initial_top_10": {
            key: after["mean"][key] - before["mean"][key] for key in DESCRIPTORS
        },
        "top_10_score_change_from_initial_top_10": after["mean_oracle_score"]
        - before["mean_oracle_score"],
        "flags": {
            "any_top_peptide_at_length_50": after["length_50_count"] > 0,
            "all_top_peptides_at_length_50": after["length_50_count"] == after["peptides"],
            "length_boundary_fraction_increased": after["length_50_fraction"]
            > before["length_50_fraction"],
            "charge_composition_index_increased": after["mean"]["charge_composition_index"]
            > before["mean"]["charge_composition_index"],
            "homology_unknown": True,
            "proxy_optimization_not_biological_validation": True,
            "causal_exploitation_claimed": False,
        },
    }


def build_diagnostics(protocol_path, report_path, results_root, release, output):
    protocol = load_protocol(protocol_path)
    source = json.loads(Path(report_path).read_text())
    if fingerprint({key: value for key, value in source.items() if key != "sha256"}) != source.get(
        "sha256"
    ):
        raise ValueError("input report fingerprint mismatch")
    if source["protocol_sha256"] != protocol.protocol_sha256:
        raise ValueError("input report belongs to another protocol")
    results_root = Path(results_root).resolve()
    runs, skipped, seen = [], [], set()
    for reported in source["runs"]:
        run_root = Path(reported["directory"]).resolve()
        if not run_root.is_relative_to(results_root):
            raise ValueError("reported run is outside the supplied results root")
        key = (reported["arm"], reported["seed"])
        if key in seen:
            raise ValueError("duplicate treatment and seed in input report")
        seen.add(key)
        verified = audit_run(run_root / "campaign", protocol, release)
        for field in (
            "arm",
            "seed",
            "complete_budget",
            "charged_evaluations",
            "initial_manifest_sha256",
            "oracle_sha256",
            "audit",
            "endpoint",
        ):
            if verified[field] != reported[field]:
                raise ValueError(f"input report no longer matches journal: {field}")
        if not verified["complete_budget"]:
            skipped.append(
                {
                    "arm": key[0],
                    "seed": key[1],
                    "status": verified["status"],
                    "reason": "incomplete_budget_not_used_for_composition_comparison",
                }
            )
            continue
        events = [
            json.loads(line)
            for line in (run_root / "campaign/events.jsonl").read_text().splitlines()
        ]
        summary = sequence_summary(events[0]["manifest"], events)
        if not math.isclose(
            summary["final_top_10"]["mean_oracle_score"],
            verified["endpoint"]["top_10_mean_score"],
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise ValueError("common top-ten score no longer matches audited report")
        runs.append(
            {
                "arm": key[0],
                "seed": key[1],
                "terminal_event_sha256": verified["audit"]["terminal_event_sha256"],
                **summary,
            }
        )
    lookup = {(row["arm"], row["seed"]): row for row in runs}
    paired = []
    for row in runs:
        for baseline in ("genetic_algorithm", "categorical"):
            reference = lookup.get((baseline, row["seed"]))
            if reference is None or baseline == row["arm"]:
                continue
            paired.append(
                {
                    "arm": row["arm"],
                    "baseline": baseline,
                    "seed": row["seed"],
                    "top_10_score_difference": row["final_top_10"]["mean_oracle_score"]
                    - reference["final_top_10"]["mean_oracle_score"],
                    "top_10_mean_descriptor_differences": {
                        key: row["final_top_10"]["mean"][key]
                        - reference["final_top_10"]["mean"][key]
                        for key in DESCRIPTORS
                    },
                    "best_sequence_descriptor_differences": {
                        key: row["final_best_sequence"][key] - reference["final_best_sequence"][key]
                        for key in DESCRIPTORS
                    },
                    "length_50_fraction_difference": row["final_top_10"]["length_50_fraction"]
                    - reference["final_top_10"]["length_50_fraction"],
                }
            )
    result = {
        "artifact": "native_proxy_composition_diagnostics_v1",
        "source_report_sha256": source["sha256"],
        "protocol_sha256": protocol.protocol_sha256,
        "runs": runs,
        "skipped_runs": skipped,
        "paired_complete_seed_comparisons": paired,
        "definitions": {
            "charge_composition_index": "(count(K)+count(R)-count(D)-count(E))/length; not net charge at a specified pH; histidine and termini omitted",
            "hydrophobic_fraction": "count of residues in AVILMFWY divided by length; declared descriptive set, not a hydropathy scale",
            "diversity": "exact sequence uniqueness only; no homology or structural diversity assessment",
            "ranking": "descending paid frozen-oracle score; exact ties resolved lexicographically by sequence",
        },
        "interpretation": "Composition and boundary shifts are descriptive associations, not proof of oracle exploitation, causal mechanism, or biological efficacy. No new labels or fits were obtained.",
        "winner_claimed": False,
        "causal_exploitation_claimed": False,
        "homology_assessed": False,
        "additional_oracle_calls": 0,
    }
    result["sha256"] = fingerprint(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "diagnostics.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    lines = [
        "# Post-hoc peptide composition diagnostics",
        "",
        result["interpretation"],
        "",
        "Charge index means (K + R - D - E) / length, not physical charge at a particular pH. Hydrophobic residues are defined here as AVILMFWY.",
        "",
        "| Method | Seed | Top-ten mean score | Mean length | Mean charge index | Mean hydrophobic fraction | Length-50 count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        top = row["final_top_10"]
        lines.append(
            f"| {row['arm']} | {row['seed']} | {top['mean_oracle_score']:.6g} | {top['mean']['length']:.4g} | {top['mean']['charge_composition_index']:.4g} | {top['mean']['hydrophobic_fraction']:.4g} | {top['length_50_count']} / {top['peptides']} |"
        )
    lines.extend(
        [
            "",
            "Initial-dataset contrasts, best sequences, paired baseline differences, and explicit boundary flags are retained in diagnostics.json. Exact uniqueness does not establish low homology.",
        ]
    )
    with (output / "diagnostics.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "report", "results-root", "release", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build_diagnostics(
        args.protocol, args.report, args.results_root, args.release, args.output
    )
    print(json.dumps({"completed_runs": len(result["runs"]), "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()

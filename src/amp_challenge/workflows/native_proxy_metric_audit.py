"""Recompute accepted-update metrics and exact mixture arithmetic from receipts.

This reexecutes the existing numerical distance implementation. It is not an
independent distance implementation, neural-weight replay, oracle query, or
verification of complete peptide-distribution distances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint, load_protocol
from amp_challenge.generators.diffusion.distribution_distance_study import (
    amino_acid_ground_cost,
    categorical_distance_diagnostics,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES

METRICS = ("kl", "reverse_kl", "total_variation", "jensen_shannon", "wasserstein_1")
SCOPE = "raw_denoising_trajectory_conditional_fixed_parent_start_level_kernel"
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[3] / "configs/search/peptide_proxy_distance_v3.toml"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(payload):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject)


def _pin(value):
    return type(value) is str and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _models(rows):
    require(
        type(rows) is list and all(type(row) is list and len(row) == 2 for row in rows),
        "model inventory shape differs",
    )
    require(
        tuple(row[0] for row in rows) == TRIPLES and all(_pin(row[1]) for row in rows),
        "ten ordered model identities required",
    )
    return dict(rows)


def _weights(policy):
    require(
        policy["contract"] == "frozen_complete_trajectory_policy_mixture_v1",
        "mixture contract differs",
    )
    require(
        policy["sampling_scope"] == "one_frozen_component_per_complete_peptide_trajectory"
        and policy["provider_freezing_required"] is True,
        "mixture sampling scope differs",
    )
    result = {}
    for row in policy["components"]:
        identity, numerator, denominator = (
            row["component_id"],
            row["weight_numerator"],
            row["weight_denominator"],
        )
        require(
            _pin(identity)
            and identity not in result
            and type(numerator) is int
            and numerator > 0
            and type(denominator) is int
            and denominator > 0,
            "mixture component identity or rational weight differs",
        )
        result[identity] = Fraction(numerator, denominator)
    require(result and sum(result.values()) == 1, "mixture weights must sum exactly to one")
    return result


def verify_mixture(snapshot, old_models, new_models, *, previous=None):
    """Independently reconstruct rational updates, without calling mixture producer."""
    require(
        snapshot["sha256"]
        == fingerprint({key: value for key, value in snapshot.items() if key != "sha256"}),
        "mixture snapshot hash differs",
    )
    require(
        snapshot["bound_scope"] == SCOPE
        and snapshot["filtered_candidate_or_whole_search_bound_claimed"] is False,
        "mixture global search claim differs",
    )
    updates = snapshot["updates"]
    previous_updates = [] if previous is None else previous["updates"]
    require(
        len(updates) == len(previous_updates) + 1 and updates[:-1] == previous_updates,
        "mixture update history changed or skipped",
    )
    require(
        tuple(row["triple"] for row in updates[-1]) == TRIPLES,
        "accepted mixture must contain all ten models",
    )
    require(
        snapshot["latest_proposals"] == new_models and set(snapshot["mixtures"]) == set(TRIPLES),
        "mixture latest models differ",
    )
    for row in updates[-1]:
        triple = row["triple"]
        before, after = _weights(row["previous_policy"]), _weights(row["accepted_policy"])
        expected_before = (
            {old_models[triple]: Fraction(1)}
            if previous is None
            else _weights(previous["mixtures"][triple])
        )
        require(
            before == expected_before and old_models[triple] in before,
            "mixture previous distribution differs",
        )
        require(
            type(row["alpha_numerator"]) is int
            and type(row["alpha_denominator"]) is int
            and row["alpha_denominator"] > 0,
            "mixture update fraction malformed",
        )
        require(
            Fraction(row["alpha_numerator"], row["alpha_denominator"]) == Fraction(1, 20),
            "mixture update mass must equal one twentieth",
        )
        require(
            row["proposal_component_id"] == new_models[triple],
            "mixture proposal is not accepted trained model",
        )
        expected = {identity: weight * Fraction(19, 20) for identity, weight in before.items()}
        expected[new_models[triple]] = expected.get(new_models[triple], Fraction(0)) + Fraction(
            1, 20
        )
        require(
            after == expected and after == _weights(snapshot["mixtures"][triple]),
            "mixture is not exact nineteen-twentieths old plus one-twentieth proposal",
        )
        require(
            row["native_sampler_connected"] is True
            and row["bound_scope"] == SCOPE
            and Fraction(row["global_total_variation_upper_bound"]) == Fraction(1, 20),
            "mixture connection or bound differs",
        )


def audit_update(entry, *, metric, threshold, seed, previous_mixture=None, previous_models=None):
    receipt = entry["update"]
    encoded = receipt["record_json"]
    require(
        type(encoded) is str and hashlib.sha256(encoded.encode()).hexdigest() == receipt["sha256"],
        "update receipt hash differs",
    )
    payload = parse(encoded)
    require(payload["artifact"] == "native_proxy_distance_update_v1", "update artifact differs")
    require(
        type(receipt["accepted"]) is bool
        and receipt["accepted"] is payload["accepted"]
        and receipt["status"] == payload["status"]
        and receipt["backtracks"] == payload["backtracks"],
        "update envelope differs from hashed payload",
    )
    require(
        payload["metric"] == metric
        and payload["metric_limit"] == threshold
        and payload["seed"] == seed,
        "update differs from declared arm threshold or seed",
    )
    require(
        payload["conditional_tv_limit"] == 0.05 and payload["clip_width"] == 0.05,
        "common clipping or total-variation limit differs",
    )
    old_models, new_models = _models(payload["old_models"]), _models(payload["new_models"])
    if previous_models is not None:
        require(
            old_models == previous_models,
            "model history does not continue previous accepted update",
        )
    if not receipt["accepted"]:
        require(old_models == new_models, "rejected update changed models")
        return (
            {"receipt_sha256": receipt["sha256"], "accepted": False, "status": receipt["status"]},
            previous_mixture,
            old_models,
        )
    require(
        payload["status"] == "accepted_guarded_update"
        and payload["error"] is None
        and payload["admission"]["admitted"] is True,
        "accepted update admission/status differs",
    )
    require(
        payload["global_five_percent_constraint_proven"] is False
        and payload["external_whole_trajectory_mixture_required"] is True,
        "sampled metric scope misrepresented",
    )
    cost = amino_acid_ground_cost()
    require(
        hashlib.sha256(cost.tobytes()).hexdigest() == payload["ground_cost_sha256"],
        "ground-cost matrix differs",
    )
    require(
        tuple(row["triple"] for row in payload["training"]) == TRIPLES,
        "training does not contain ten ordered students",
    )
    candidates = payload["candidates"]
    chosen = payload["backtracks"]
    require(
        type(chosen) is int and 0 <= chosen <= 8 and len(candidates) == chosen + 1,
        "accepted backtracking prefix differs",
    )
    require(
        [row["backtracks"] for row in candidates] == list(range(chosen + 1))
        and all(row["passed"] is False for row in candidates[:-1])
        and candidates[-1]["passed"] is True,
        "accepted update is not first passing backtrack",
    )
    students = candidates[-1]["students"]
    require(
        tuple(row["triple"] for row in students) == TRIPLES,
        "accepted candidate lacks all ten students",
    )
    maxima, total_rows = {name: 0.0 for name in METRICS}, 0
    for training, student in zip(payload["training"], students, strict=True):
        old, new = (
            np.asarray(training["old_probabilities"], dtype=float),
            np.asarray(student["new_probabilities"], dtype=float),
        )
        require(
            old.ndim == 2
            and old.shape == new.shape
            and old.shape[1] == 20
            and 1 <= len(old) <= 128
            and len(old) == len(training["sequences"]),
            "saved probability-row shape differs",
        )
        require(
            student["passed"] is True
            and student["candidate_sha256"] == new_models[student["triple"]],
            "accepted student identity/flag differs",
        )
        computed = categorical_distance_diagnostics(old, new, cost)
        require(
            set(computed) == set(student["distances"]) == set(METRICS),
            "five-metric inventory differs",
        )
        for name in METRICS:
            saved = np.asarray(student["distances"][name], dtype=float)
            require(
                saved.shape == computed[name].shape
                and np.all(np.isfinite(saved))
                and np.all(np.isfinite(computed[name])),
                "saved distance shape or finiteness differs",
            )
            np.testing.assert_allclose(
                computed[name],
                saved,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"saved {name} differs from numerical recomputation",
            )
            maxima[name] = max(maxima[name], float(np.max(computed[name])))
        require(
            float(np.max(computed[metric])) <= threshold
            and float(np.max(computed["total_variation"])) <= 0.05,
            "accepted update violates declared distance limit",
        )
        total_rows += len(old)
    mixture = entry["whole_trajectory_mixture"]
    verify_mixture(mixture, old_models, new_models, previous=previous_mixture)
    return (
        {
            "receipt_sha256": receipt["sha256"],
            "accepted": True,
            "metric": metric,
            "threshold": threshold,
            "backtracks": chosen,
            "students": 10,
            "probability_rows": total_rows,
            "maximum_recomputed_distances": maxima,
            "exact_mixture_update_mass": "1/20",
        },
        mixture,
        new_models,
    )


def audit_results(results_root, output, *, protocol_path=DEFAULT_PROTOCOL):
    results_root, output = Path(results_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        protocol = load_protocol(protocol_path)
        files = sorted(
            path
            for path in results_root.rglob("ingest-*.json")
            if path.parent.name == "native" and path.parent.parent.name == "provider"
        )
        require(files, "no native ingest records found; cannot claim accepted-update verification")
        grouped = {}
        for path in files:
            grouped.setdefault(path.parents[2], []).append(path)
        reports, inputs = [], {}
        for run_root, paths in sorted(grouped.items()):
            result_path = run_root / "campaign/result.json"
            result_bytes = result_path.read_bytes()
            result = parse(result_bytes)
            inputs[str(result_path.relative_to(results_root))] = hashlib.sha256(
                result_bytes
            ).hexdigest()
            require(
                result["protocol_sha256"] == protocol.protocol_sha256
                and result["seed"] in protocol.seeds,
                "run does not bind declared study protocol/seed",
            )
            arm = next(arm for arm in protocol.arms if arm.name == result["arm_name"])
            require(arm.method == "evolutionary", "native updates attributed to a different method")
            seen, audited, previous_mixture, previous_models = {}, [], None, None
            for path in paths:
                raw = path.read_bytes()
                inputs[str(path.relative_to(results_root))] = hashlib.sha256(raw).hexdigest()
                document = parse(raw)
                identities = []
                for entry in document["updates"]:
                    identity = entry["update"]["sha256"]
                    identities.append(identity)
                    entry_hash = fingerprint(entry)
                    if identity in seen:
                        require(
                            seen[identity] == entry_hash,
                            "duplicate receipt changed across cumulative ingests",
                        )
                        continue
                    report, previous_mixture, previous_models = audit_update(
                        entry,
                        metric=arm.constraint,
                        threshold=arm.threshold,
                        seed=result["seed"],
                        previous_mixture=previous_mixture,
                        previous_models=previous_models,
                    )
                    seen[identity] = entry_hash
                    audited.append(report)
                require(
                    len(identities) == len(set(identities)) and identities == list(seen),
                    "cumulative update history reordered, truncated or duplicated",
                )
            reports.append(
                {
                    "run": str(run_root.relative_to(results_root)),
                    "arm": arm.name,
                    "seed": result["seed"],
                    "unique_updates": len(audited),
                    "accepted_updates": sum(row["accepted"] for row in audited),
                    "updates": audited,
                }
            )
        report = {
            "artifact": "native_proxy_metric_numerical_reexecution_v1",
            "status": "verified",
            "protocol_sha256": protocol.protocol_sha256,
            "runs": reports,
            "unique_updates": sum(row["unique_updates"] for row in reports),
            "accepted_updates": sum(row["accepted_updates"] for row in reports),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "distance_implementation_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "generators/diffusion/distribution_distance_study.py"
                ).read_bytes()
            ).hexdigest(),
            "ground_cost_sha256": hashlib.sha256(amino_acid_ground_cost().tobytes()).hexdigest(),
            "kl_direction": "new_relative_to_old",
            "reverse_kl_direction": "old_relative_to_new",
            "input_sha256s": inputs,
            "numerical_tolerance": {"rtol": 1e-10, "atol": 1e-12},
            "scope": "existing_distance_numerics_reexecuted_and_exact_rational_mixture_arithmetic",
            "not_verified": [
                "neural_training_replay",
                "saved_probabilities_against_model_weights",
                "trajectory_replay",
                "biological_activity",
                "whole_search_distribution_bound",
            ],
        }
        report["accepted_update_evidence_present"] = report["accepted_updates"] > 0
        report["report_sha256"] = fingerprint(report)
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        return report
    except BaseException as error:
        with (output / "failure.json").open("x") as stream:
            json.dump(
                {"status": "failed", "type": type(error).__name__, "message": str(error)},
                stream,
                sort_keys=True,
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_results(args.results_root, args.output, protocol_path=args.protocol),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

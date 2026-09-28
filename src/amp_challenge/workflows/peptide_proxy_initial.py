"""Produce shared 512-row starts without scoring the candidate pool in advance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from amp_challenge.benchmarks.frozen_peptide_proxy import (
    ACCEPTED_EXAMPLES_SHA256,
    FrozenPeptideProxy,
)
from amp_challenge.evaluation.peptide_proxy_protocol import (
    build_initial_manifest,
    load_protocol,
    select_initial_sequences,
    write_initial_manifest,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

EXAMPLES = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/benchmarks/"
    "oracle-gate1-union-v1/223248/0/gate1/examples.jsonl"
)
CANDIDATES = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/generation/"
    "candidate-pool-v1-validations/225275/0/candidates.jsonl"
)
CANDIDATES_SHA256 = "77315bfbec533f5587123714199c2dd2eb2a51a16d95957ab53927494983dbf2"


def eligible_candidates(payload: bytes, training: set[str]) -> tuple[list[dict], dict]:
    """Validate a pinned inventory before selection; never inspect reward labels."""
    if hashlib.sha256(payload).hexdigest() != CANDIDATES_SHA256:
        raise ValueError("accepted candidate inventory changed")
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    candidates = []
    counts = {"inventory": 0, "ineligible": 0, "exact_training_overlap": 0, "admitted": 0}
    for line in payload.splitlines():
        if not line.strip():
            raise ValueError("candidate inventory contains an empty record")
        row = json.loads(line)
        sequence = row["sequence"]
        identifier = row["sequence_id"]
        if not isinstance(sequence, str) or not 8 <= len(sequence) <= 50:
            raise ValueError("candidate sequence must have length 8 to 50")
        if identifier != canonical_sequence_id(sequence) or sequence != canonicalize_sequence(
            sequence
        ):
            raise ValueError("candidate sequence identity is not canonical")
        if type(row["length"]) is not int or row["length"] != len(sequence):
            raise ValueError("candidate recorded length disagrees")
        if type(row["library_eligible"]) is not bool:
            raise ValueError("candidate eligibility must be boolean")
        if identifier in seen_ids or sequence in seen_sequences:
            raise ValueError("candidate inventory contains duplicate identity")
        seen_ids.add(identifier)
        seen_sequences.add(sequence)
        counts["inventory"] += 1
        if not row["library_eligible"]:
            counts["ineligible"] += 1
            continue
        if sequence in training:
            counts["exact_training_overlap"] += 1
            continue
        candidates.append({"peptide_id": identifier, "sequence": sequence})
    counts["admitted"] = len(candidates)
    return candidates, counts


def produce(protocol_path: Path, output: Path) -> dict:
    protocol = load_protocol(protocol_path)
    # Exclusive root is also a fail-stop boundary: never overwrite partial work.
    output.mkdir(parents=True, exist_ok=False)
    proxy = FrozenPeptideProxy()
    proxy.freeze(output / "oracle")
    example_bytes = EXAMPLES.read_bytes()
    if hashlib.sha256(example_bytes).hexdigest() != ACCEPTED_EXAMPLES_SHA256:
        raise ValueError("oracle training inventory changed")
    training = {json.loads(line)["sequence"] for line in example_bytes.splitlines() if line}
    candidate_bytes = CANDIDATES.read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    candidates, candidate_counts = eligible_candidates(candidate_bytes, training)
    report = {
        "artifact": "paired_proxy_initialization_v1",
        "protocol": protocol.protocol_sha256,
        "candidate_sha256": candidate_sha,
        "excluded_exact_oracle_training": True,
        "candidate_counts": candidate_counts,
        "training_inventory_sha256": ACCEPTED_EXAMPLES_SHA256,
        "reward_role": "activity_prediction_diagnostic_not_selected_quickvina_reward",
        "selection": "seeded_without_replacement_before_any_initial_labels_are_queried",
        "homology_exclusion_claimed": False,
        "oracle": proxy.manifest(),
        "accounting": protocol.accounting(),
        "seeds": [],
        "scientific_comparison_runs_completed": 0,
    }
    for seed in protocol.seeds:
        chosen = select_initial_sequences(candidates, seed=seed, protocol=protocol)
        # Retain intent BEFORE any scoring; a failed run cannot silently resample.
        intent = {"seed": seed, "oracle_sha256": proxy.model_sha256, "records": chosen}
        write_initial_manifest(output / f"seed-{seed}-intent.json", intent)
        scores = proxy.score([row["sequence"] for row in chosen])
        scored = [
            {**row, "oracle_score": float(score)} for row, score in zip(chosen, scores, strict=True)
        ]
        manifest = build_initial_manifest(
            scored,
            seed=seed,
            oracle_id="frozen_activity_proxy_v1",
            oracle_sha256=proxy.model_sha256,
            dataset_sha256=candidate_sha,
            protocol=protocol,
        )
        write_initial_manifest(output / f"seed-{seed}.json", manifest)
        report["seeds"].append(
            {
                "seed": seed,
                "manifest_sha256": manifest["manifest_sha256"],
                "evaluations": len(scored),
                "minimum": float(scores.min()),
                "mean": float(scores.mean()),
                "maximum": float(scores.max()),
            }
        )
    write_initial_manifest(output / "complete.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(produce(args.protocol, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

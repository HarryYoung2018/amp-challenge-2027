"""Generate challenge FASTAs from a trained evolutionary checkpoint mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from amp_challenge.workflows.generate import (
    DEFAULT_REFERENCE,
    GenerationSummary,
    ReferenceIndex,
    _read_fasta_sequences,
)


def choose_portfolio(sequences, target_scores, target_names, embeddings, reference, *, top_k=100):
    """Balanced potency portfolio with soft embedding diversity and exact eligibility."""
    from amp_challenge.benchmarks.frozen_peptide_proxy import TARGET_PANEL
    from amp_challenge.models.competition_oracle import normalized_rows

    gram = dict(TARGET_PANEL)
    positive = [i for i, target in enumerate(target_names) if gram[target] == "positive"]
    negative = [i for i, target in enumerate(target_names) if gram[target] == "negative"]
    if len(sequences) < top_k or not positive or not negative:
        raise ValueError("portfolio requires enough candidates and both Gram panels")
    scores = np.asarray(target_scores)
    utilities = (
        scores.mean(axis=1),
        scores[:, positive].mean(axis=1),
        scores[:, negative].mean(axis=1),
    )
    vectors = normalized_rows(embeddings)
    similarity = np.zeros(len(sequences))
    available = np.ones(len(sequences), dtype=bool)
    selected, metadata = [], []
    for rank in range(top_k):
        # Interleave 50% broad, 25% Gram-positive, 25% Gram-negative throughout ranks.
        objective = (0, 1, 0, 2)[rank % 4]
        values = utilities[objective] - 0.10 * similarity
        values = np.where(available, values, -np.inf)
        for index in np.argsort(-values, kind="stable"):
            if not available[index]:
                continue
            ratio, _ = reference.max_ratio(sequences[index], threshold=0.8)
            if ratio > 0.8:
                available[index] = False
                continue
            selected.append(int(index))
            metadata.append(
                {
                    "rank": rank + 1,
                    "sequence": sequences[index],
                    "objective": ("broad", "gram_positive", "gram_negative")[objective],
                    "target_predictions": dict(
                        zip(target_names, map(float, scores[index]), strict=True)
                    ),
                    "internal_utility": float(values[index]),
                    "max_reference_ratio": ratio,
                }
            )
            available[index] = False
            similarity = np.maximum(similarity, vectors @ vectors[index])
            break
        else:
            raise ValueError("not enough reference-eligible peptides for the portfolio")
    return selected, metadata


def write_fastas(sequences, selected, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    library, top = output / "library.fasta", output / "top.fasta"
    library.write_text("".join(f">amp_{i + 1:05d}\n{seq}\n" for i, seq in enumerate(sequences)))
    top.write_text(
        "".join(
            f">rank_{rank + 1:03d}_amp_{index + 1:05d}\n{sequences[index]}\n"
            for rank, index in enumerate(selected)
        )
    )
    return library, top


def validate_output(sequences, selected, reference, *, n_sequences, top_k):
    if len(sequences) != n_sequences or len(set(sequences)) != n_sequences:
        raise ValueError("library count or uniqueness is invalid")
    if any(not 8 <= len(seq) <= 50 or set(seq) - set("ACDEFGHIKLMNPQRSTVWY") for seq in sequences):
        raise ValueError("library contains an invalid peptide")
    if set(sequences) & set(reference.sequences):
        raise ValueError("library overlaps organizer reference")
    if (
        len(selected) != top_k
        or len(set(selected)) != top_k
        or any(not 0 <= index < n_sequences for index in selected)
    ):
        raise ValueError("top portfolio membership is invalid")
    if any(reference.max_ratio(sequences[index])[0] > 0.8 for index in selected):
        raise ValueError("top portfolio exceeds organizer similarity threshold")


def generate_evolutionary_submission(
    *,
    release,
    oracle_root,
    n_sequences=50_000,
    top_k=100,
    seed=42,
    output_dir=Path("generate"),
    reference_path=DEFAULT_REFERENCE,
    device="cuda",
):
    import torch

    from amp_challenge.generators.diffusion.distribution_policy_mixture import FrozenPolicyMixture
    from amp_challenge.generators.diffusion.model import (
        NativeDenoiser,
        NativeDenoiserConfig,
        canonical_model_logical_hash,
        configure_deterministic_runtime,
    )
    from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
    from amp_challenge.models.competition_oracle import CompetitionOracle

    if n_sequences < top_k or top_k < 1:
        raise ValueError("invalid requested library or portfolio size")
    torch.set_num_threads(1)
    configure_deterministic_runtime(seed)
    release = Path(release)
    result = json.loads((release / "campaign/result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("generation requires a completed evolutionary run")
    indices = sorted((release / "provider/native/policies").glob("index-*.json"))
    if len(indices) != 1:
        raise ValueError("select one final checkpoint index explicitly")
    index = json.loads(indices[0].read_text())
    snapshot = index["sampler"]
    models = {(item["triple"], item["model_sha256"]): item for item in index["checkpoints"]}
    mixtures = {}
    for triple, mixture in snapshot["mixtures"].items():
        entries = mixture["components"]
        mixtures[triple] = FrozenPolicyMixture(
            tuple(row["component_id"] for row in entries),
            tuple(Fraction(row["weight_numerator"], row["weight_denominator"]) for row in entries),
        )
    events = [
        json.loads(line) for line in (release / "campaign/events.jsonl").read_text().splitlines()
    ]
    paid = {row["sequence"]: row["oracle_score"] for row in events[0]["manifest"]["records"]}
    paid.update(
        {
            event["sequence"]: event["oracle_score"]
            for event in events
            if event["kind"] == "outcome" and event["status"] == "success"
        }
    )
    parents = sorted(paid, key=lambda seq: (-paid[seq], seq))[:100]
    reference = ReferenceIndex(_read_fasta_sequences(Path(reference_path)))
    forbidden = set(reference.sequences)
    oracle = CompetitionOracle(oracle_root, device=device)
    if oracle.model_sha256 != result["oracle_sha256"]:
        raise ValueError("generation scorer differs from trained search reward")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    triples = tuple(sorted(mixtures))
    sequences, seen = [], set()
    raw_count = 0
    # New submission inference budget, separate from the 512+1024 search benchmark.
    while len(sequences) < n_sequences:
        count = min(max(n_sequences - len(sequences), min(1024, n_sequences)), 60_000)
        if raw_count + count > n_sequences * 5 + 1024:
            raise RuntimeError("generation exhausted its duplicate-rejection budget")
        groups = {}
        for ordinal in range(raw_count, raw_count + count):
            triple = triples[int(rng.integers(len(triples)))]
            component = mixtures[triple].sample_component(rng)
            parent = parents[int(rng.integers(len(parents)))]
            # Full-mask exploration and local evolutionary refinement; fixed before scoring.
            level = 64 if ordinal % 2 == 0 else 16
            groups.setdefault((triple, component), []).append((ordinal, parent, level))
        generated = []
        for (triple, identity), rows in sorted(groups.items()):
            artifact = models[triple, identity]
            path = indices[0].parent / artifact["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["file_sha256"]:
                raise ValueError("generation checkpoint differs from saved policy")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            with torch.random.fork_rng(devices=[]):
                model = NativeDenoiser(NativeDenoiserConfig(**payload["config"]))
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            if canonical_model_logical_hash(model) != identity:
                raise ValueError("generation model does not match selected component")
            for start in range(0, len(rows), 128):
                batch = rows[start : start + 128]
                traces = sample_native_proposals(
                    model,
                    tuple(row[1] for row in batch),
                    start_levels=tuple(row[2] for row in batch),
                    seed=seed,
                    ordinals=tuple(row[0] for row in batch),
                )
                generated.extend((trace.ordinal, trace.endpoint) for trace in traces)
            del model
        for _, sequence in sorted(generated):
            if sequence not in seen and sequence not in forbidden:
                sequences.append(sequence)
                seen.add(sequence)
                if len(sequences) == n_sequences:
                    break
        raw_count += count
        print(
            json.dumps({"raw_generated": raw_count, "unique_library": len(sequences)}), flush=True
        )
    predictions = []
    for start in range(0, len(sequences), 512):
        predictions.append(oracle.predict_targets(sequences[start : start + 512]))
    target_scores = np.vstack(predictions)
    selected, metadata = choose_portfolio(
        sequences,
        target_scores,
        tuple(oracle.model.targets),
        oracle.embedder.encode(sequences),
        reference,
        top_k=top_k,
    )
    validate_output(sequences, selected, reference, n_sequences=n_sequences, top_k=top_k)
    library, top = write_fastas(sequences, selected, output_dir)
    output = Path(output_dir)
    metadata_path = output / "top_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        output / "library_predictions.npz",
        target_names=np.asarray(tuple(oracle.model.targets)),
        predictions=target_scores,
    )
    library_hash, top_hash = (
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (library, top)
    )
    top_mean = target_scores[selected].mean(axis=1)
    from rapidfuzz.distance import Indel

    pair_similarities = [
        Indel.normalized_similarity(sequences[a], sequences[b])
        for pos, a in enumerate(selected)
        for b in selected[pos + 1 :]
    ]
    from amp_challenge.generators.baseline import FEATURE_NAMES, baseline_feature_matrix

    descriptors = baseline_feature_matrix(sequences)
    manifest = {
        "generator": "six_layer_evolutionary_exact_checkpoint_mixture",
        "release": str(release),
        "checkpoint_index_sha256": hashlib.sha256(indices[0].read_bytes()).hexdigest(),
        "oracle_sha256": oracle.model_sha256,
        "seed": seed,
        "library_size": len(sequences),
        "top_size": len(selected),
        "raw_proposals": raw_count,
        "accepted_search_updates": len(snapshot["updates"]),
        "submission_scored_sequences": len(sequences),
        "search_evaluations_separate": 1024,
        "library_sha256": library_hash,
        "top_sha256": top_hash,
        "sampling": "one_exact_mixture_component_per_trajectory_half_full_mask_half_level16_top100_paid_parents",
        "five_percent_bound_claimed_for_selected_library": False,
        "target_names": list(oracle.model.targets),
        "library_target_quantiles": np.quantile(target_scores, [0.1, 0.5, 0.9], axis=0).tolist(),
        "library_descriptor_names": list(FEATURE_NAMES),
        "library_descriptor_quantiles": np.quantile(descriptors, [0.1, 0.5, 0.9], axis=0).tolist(),
        "descriptor_scope": "physicochemical_and_synthesis_proxies_not_measured_safety",
        "top_reference_similarity_quantiles": np.quantile(
            [row["max_reference_ratio"] for row in metadata], [0.1, 0.5, 0.9]
        ).tolist(),
        "top_mean_activity": float(top_mean.mean()),
        "top_lower_decile_activity": float(np.quantile(top_mean, 0.1)),
        "top_mean_pairwise_indel_similarity": float(np.mean(pair_similarities))
        if pair_similarities
        else 0.0,
        "top_max_pairwise_indel_similarity": float(max(pair_similarities))
        if pair_similarities
        else 0.0,
        "eligible_top": len(selected),
        "warning": "Internal activity predictions, not official aggregation score, biological efficacy, or safety validation.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return GenerationSummary(
        library,
        top,
        metadata_path,
        manifest_path,
        len(sequences),
        len(selected),
        seed,
        library_hash,
        top_hash,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate_evolutionary_submission(
        release=args.release,
        oracle_root=args.oracle,
        output_dir=args.output,
        n_sequences=args.n_sequences,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(
        json.dumps({"library": str(result.library_path), "top": str(result.top_path)}), flush=True
    )


if __name__ == "__main__":
    main()

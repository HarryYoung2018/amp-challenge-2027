"""Replay one recorded failed acquisition batch; no search, generation, or oracle calls."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from pathlib import Path
from time import monotonic


def read(path):
    return json.loads(path.read_text())


def reconstruct(root, generation):
    import numpy as np

    from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
    from amp_challenge.generators.diffusion.native_endpoint import _json_hash
    from amp_challenge.generators.diffusion.native_evolution_records import semantic_history
    from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
    from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
    from amp_challenge.models.charged_probability_learner import (
        GeneratorFeatureTransform,
        fit_charged_learner,
    )
    from amp_challenge.representations.peptide_esm import digest

    native = root / "provider/native"
    provisioned = read(native / "provisioned.json")
    wave = read(native / f"wave-{generation:02d}.json")
    selection = read(native / f"selection-{generation:02d}.json")
    history_data = read(native / f"history-{generation:02d}.json")
    observations = tuple(
        ChargedObservation(
            **{**row, "objectives": None if row["objectives"] is None else tuple(row["objectives"])}
        )
        for row in history_data.pop("observations")
    )
    history = VerifiedHistorySnapshot(**history_data, observations=observations)
    if history.sha256 != wave["history_sha256"] or _json_hash(wave) != selection["wave_sha256"]:
        raise ValueError("saved history, wave and selection identities differ")
    with (root / "campaign/events.jsonl").open() as stream:
        excluded = set(json.loads(next(stream))["excluded_sequences"])
    eligible = tuple(
        row
        for row in observations
        if row.status == "successful"
        and row.sequence not in excluded
        and 8 <= len(row.sequence) <= 50
        and set(row.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
    )
    if (
        semantic_history(history, frozenset(sequence_id(row.sequence) for row in eligible))
        != wave["semantic_history_sha256"]
    ):
        raise ValueError("charged semantic history differs")
    pool_sequences = tuple(
        wave["attempts"][i]["trace"]["endpoint"] for i in selection["pool_ordinals"]
    )
    wanted = {row.sequence for row in eligible} | set(pool_sequences)
    features, receipts = {}, {}
    for batch in sorted((root / "provider/features").glob("batch-*")):
        directory = batch / "features"
        payload = (directory / "sequences.jsonl").read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        if not any(row["sequence"] in wanted - features.keys() for row in rows):
            continue
        manifest_bytes = (directory / "manifest.json").read_bytes()
        pin = read(batch / "response.json")["manifest_sha256"]
        manifest = json.loads(manifest_bytes)
        if digest(manifest_bytes) != pin or digest(payload) != manifest["sequence_input_sha256"]:
            raise ValueError("saved feature manifest or sequence identity differs")
        data = (directory / "esm_length_spectral.npy").read_bytes()
        binding = manifest["arrays"]["esm_length_spectral.npy"]
        if digest(data) != binding["sha256"] or len(data) != binding["bytes"]:
            raise ValueError("saved feature vector bytes differ")
        values = np.load(io.BytesIO(data), allow_pickle=False)
        if list(values.shape) != binding["shape"] or values.dtype.str != binding["dtype"]:
            raise ValueError("saved feature vector layout differs")
        for index, row in enumerate(rows):
            if row["sequence"] in wanted and row["sequence"] not in features:
                features[row["sequence"]] = values[index].copy()
                receipts[row["sequence"]] = pin
        if wanted <= features.keys():
            break
    if not wanted <= features.keys():
        raise ValueError("missing saved features; this microcheck never regenerates them")
    transform = GeneratorFeatureTransform(
        provisioned["feature_binding"]["representation"],
        provisioned["transform_mean"],
        provisioned["transform_scale"],
        provisioned["transform_source_sha256"],
    )
    if transform.sha256 != provisioned["transform_sha256"]:
        raise ValueError("frozen transform differs")
    learner = fit_charged_learner(
        history,
        transform,
        feature_sequence_ids=tuple(sequence_id(row.sequence) for row in eligible),
        raw_features=np.stack([features[row.sequence] for row in eligible]),
        feature_receipt_sha256=_json_hash([receipts[row.sequence] for row in eligible]),
        eligible_query_ids=tuple(row.query_id for row in eligible),
        scalar_replicas=True,
    )
    belief = learner.joint(np.stack([features[seq] for seq in pool_sequences]))
    numerical = _json_hash(
        [belief.mean.tolist(), belief.covariance.tolist(), learner.observation_noise.tolist()]
    )
    if numerical != selection["numerical_sha256"]:
        raise ValueError(
            "reconstructed joint Gaussian does not exactly match recorded numerical hash"
        )
    return (
        belief,
        wave,
        selection,
        {
            "joint_numerical_sha256": numerical,
            "charged_rows": len(eligible),
            "pool_size": len(pool_sequences),
            "feature_manifests": sorted(set(receipts.values())),
            "scope": "saved_vector_hashes_and_charged_posterior_not_neural_feature_reexecution",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--generation", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4:
        raise RuntimeError("use the supported four-CPU compute runner")
    import numpy as np

    from amp_challenge.acquisition.soft_kg import (
        EvaluationBatch,
        GaussianSoftKG,
        PreferenceMeasure,
        SoftKGProblem,
    )
    from amp_challenge.generators.diffusion.native_acquisition_precision import recheck_fixed_batch
    from amp_challenge.generators.diffusion.native_endpoint import _json_hash

    args.output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    belief, wave, selection, reconstruction = reconstruct(args.run, args.generation)
    if (
        selection["status"] != "stopped_kg_mc_instability"
        or len(selection["post_selection_checks"]) != 1
    ):
        raise ValueError("microcheck requires one recorded failed joint-batch verification")
    original = selection["post_selection_checks"][0]
    group = tuple(original["group"])
    positions = tuple(range(reconstruction["pool_size"]))
    problem = SoftKGProblem(
        positions, (0, 1), PreferenceMeasure(np.array([[0.5, 0.5]])), np.ones(len(positions))
    )
    batch = EvaluationBatch(positions, np.ones(len(positions)))
    reproduction = []
    for fantasies, suffix, mean_key, se_key in (
        (512, "evolution-kg-primary", "primary_mean", "primary_se"),
        (1024, "evolution-kg-independent-check", "check_mean", "check_se"),
    ):
        engine = GaussianSoftKG(
            problem,
            temperature=0.25,
            observed_outputs=(0, 1),
            n_fantasies=fantasies,
            standard_error_multiplier=1,
            seed=int(_json_hash([wave["semantic_history_sha256"], suffix])[:16], 16),
            relative_eigenvalue_cutoff=1e-10,
            candidate_chunk_size=64,
            fantasy_chunk_size=512,
        )
        result = engine.score_joint_groups(belief, batch, [group], max_groups=1)
        actual = [float(result.estimate[0]), float(result.standard_error[0])]
        np.testing.assert_allclose(
            actual, [original[mean_key], original[se_key]], atol=1e-12, rtol=1e-9
        )
        reproduction.append(actual)
    check = recheck_fixed_batch(
        problem, belief, batch, group, wave["semantic_history_sha256"], deadline=monotonic() + 5
    )
    report = {
        "artifact": "single_failed_acquisition_microcheck_v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "original_run": str(args.run),
        "generation": args.generation,
        "reconstruction": reconstruction,
        "original_check": original,
        "reproduced_original_estimates": reproduction,
        "bounded_recheck": check,
        "elapsed_seconds": monotonic() - started,
        "additional_oracle_evaluations": 0,
        "candidate_reranking": False,
        "campaign_resumed": False,
        "scope": "one_retrospective_numerical_check_not_a_completed_campaign_or_general_reliability_claim",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

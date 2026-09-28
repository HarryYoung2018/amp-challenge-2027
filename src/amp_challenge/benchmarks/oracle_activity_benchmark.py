"""Pinned real oracle-only activity benchmark; run only after source/config review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import scipy

from amp_challenge.benchmarks.oracle_activity_nested import (
    ARMS,
    LAMBDAS,
    SEEDS,
    SOLVER,
    nested_validation,
    summarize_predictions,
)
from amp_challenge.data import generator_oracle_namespace_split as namespace
from amp_challenge.data.oracle_modeling_projection import read_projection
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.representations.peptide_esm import load_features

CONFIG = "configs/benchmarks/oracle_activity_nested_v1.toml"
PROJECTION_SHA = "eabed8247a42928c054d1f88c956025d806b7d0c0f68117ae54774beab7f6089"
FEATURE_SHA = "c6c49f570be0788496b04ce2c3248b23d5babeb560b046f6aef5b006d3b325f9"
FEATURE_RECEIPT_SHA = "50aacc146c6ff25be92f2222f38552805a8ca14a52ae9e5fb175f213d087f23b"
DATA_FILES = (
    "predictions.jsonl",
    "models.jsonl",
    "inner_predictions.jsonl",
    "inner_assignments.json",
    "metrics.json",
)
OUTPUT_FILES = (*DATA_FILES, "manifest.json", "SHA256SUMS")
MARKER = "oracle_only_activity_benchmark_complete_v1"
SOURCES = (
    CONFIG,
    "src/amp_challenge/benchmarks/oracle_activity_nested.py",
    "src/amp_challenge/benchmarks/oracle_activity_benchmark.py",
    "cluster/slurm/oracle_activity_nested_v1.sbatch",
    "src/amp_challenge/data/oracle_modeling_projection.py",
    "src/amp_challenge/data/generator_oracle_namespace_split.py",
    "src/amp_challenge/representations/peptide_esm.py",
    "src/amp_challenge/representations/laplacian.py",
    "src/amp_challenge/descriptors.py",
    "src/amp_challenge/sequences.py",
    "src/amp_challenge/constants.py",
    "pyproject.toml",
    "uv.lock",
)
CLAIMS = {
    "research_activity_fitting_performed": True,
    "safety_model_fitted": False,
    "continuous_MIC_inferred": False,
    "search_superiority_accepted": False,
    "campaign_execution_authorized": False,
    "oracle_calls_authorized": False,
    "production_input_eligible": False,
}


def canonical(value) -> bytes:
    return namespace._canonical(value)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonl(value: list[dict]) -> bytes:
    return b"".join(canonical(row) for row in value)


def pinned_file(path: Path, pin: str, maximum=32 * 1024**2) -> bytes:
    descriptor = namespace._open_root(path.parent)
    try:
        image = namespace._snapshot(descriptor, path.name, maximum, immutable=True)
    finally:
        os.close(descriptor)
    if image.sha256 != pin:
        raise ValueError(f"artifact pin differs: {path.name}")
    return image.payload


def check_config(config: dict) -> None:
    checks = {
        "schema_version": 1,
        "artifact": "oracle_only_activity_nested_benchmark_v1",
        "scope": "predictive_binary_MIC16_validation_not_search_superiority",
        "projection_manifest_sha256": PROJECTION_SHA,
        "feature_manifest_sha256": FEATURE_SHA,
        "feature_verification_sha256": FEATURE_RECEIPT_SHA,
        "arms": list(ARMS),
        "penalties": list(LAMBDAS),
        "outer_folds": 5,
        "inner_folds": 3,
        "diagnostic_baseline": "same_train_only_smoothed_context_hierarchy_for_all_outer_rows_including_supported_contexts",
        "diagnostic_baseline_comparisons": "descriptive_proper_scores_support_and_dominant_union_stress_only_no_primary_family_expansion",
        "bootstrap_replicates": 2000,
        "bootstrap_primary_seed": SEEDS[0],
        "bootstrap_mc_check_seeds": list(SEEDS[1:]),
        "maximum_fits_per_execution": 260,
        "maximum_executions": 2,
        "maximum_fits_total": 520,
        "maximum_cpus_per_execution": 2,
        "maximum_memory_gib": 4,
        "maximum_wall_seconds_per_execution": 1200,
        "maximum_output_bytes": 134217728,
        "automatic_retries": 0,
        "gpu_count": 0,
        "oracle_calls": 0,
        "safety_model_fitting_allowed": False,
        "continuous_MIC_inference_allowed": False,
        "search_superiority_accepted": False,
        "production_input_eligible": False,
        **{f"solver_{key}": value for key, value in SOLVER.items()},
        "reproduction": {
            "exact_fields": [
                "outer_fold_assignments",
                "inner_union_fold_assignments",
                "arm_identity",
                "example_sequence_union_identity",
                "training_example_and_union_ids",
                "context_vocabularies",
                "abstention_and_support_metadata",
                "diagnostic_prior_level_and_support",
                "selected_penalties",
                "fit_attempt_counts",
            ],
            "probabilities_absolute_tolerance": 1e-10,
            "probabilities_relative_tolerance": 0.0,
            "metrics_absolute_tolerance": 1e-10,
            "metrics_relative_tolerance": 0.0,
            "coefficients_transforms_absolute_tolerance": 1e-8,
            "coefficients_transforms_relative_tolerance": 1e-8,
            "adaptive_tolerance_changes_allowed": False,
        },
    }
    if any(
        config.get(key) != value or type(config.get(key)) is not type(value)
        for key, value in checks.items()
    ):
        raise ValueError("predeclared benchmark numerical/resource contract differs")


def select_oracle_matrices(sequences, contexts, feature_rows, arrays):
    """Select exact oracle IDs before any descriptor or learned transform/fitting."""
    selected = {row["sequence_id"]: row for row in sequences}
    cache = {row["sequence_id"]: (index, row["sequence"]) for index, row in enumerate(feature_rows)}
    if len(selected) != len(sequences) or len(cache) != len(feature_rows):
        raise ValueError("duplicate sequence identity")
    if any(row["namespace"] != "oracle" for row in sequences):
        raise ValueError("non-oracle namespace in modeling input")
    ids = sorted(selected)
    if any(sid not in cache or selected[sid]["sequence"] != cache[sid][1] for sid in ids):
        raise ValueError("oracle feature identity mismatch")
    indices = np.asarray([cache[sid][0] for sid in ids])
    esm = np.asarray(arrays["esm_length"][indices], dtype=np.float64)
    spectral = np.asarray(arrays["spectral"][indices], dtype=np.float64)
    descriptors = np.asarray(
        [list(compute_descriptors(selected[sid]["sequence"]).as_dict().values()) for sid in ids]
    )
    oracle_matrices = {
        ARMS[0]: descriptors,
        ARMS[1]: esm,
        ARMS[2]: np.column_stack((spectral, esm[:, -1])),
        ARMS[3]: np.column_stack((esm, spectral)),
    }
    by_id = {sid: index for index, sid in enumerate(ids)}
    if any(row["namespace"] != "oracle" or row["sequence_id"] not in selected for row in contexts):
        raise ValueError("non-oracle activity context")
    context_indices = np.asarray([by_id[row["sequence_id"]] for row in contexts])
    return {name: values[context_indices] for name, values in oracle_matrices.items()}


def load_data(projection: Path, features: Path, feature_receipt: Path):
    payloads = read_projection(projection, expected_manifest_sha256=PROJECTION_SHA)
    receipt = json.loads(pinned_file(feature_receipt, FEATURE_RECEIPT_SHA))
    # Receipt contents are externally pinned; the public feature reader rechecks
    # the full label-free cache. Only the selected oracle arrays enter fitting.
    feature_rows, arrays, feature_manifest = load_features(features, FEATURE_SHA)
    sequences = [json.loads(line) for line in payloads["oracle_sequences.jsonl"].splitlines()]
    contexts = [json.loads(line) for line in payloads["activity_contexts.jsonl"].splitlines()]
    matrices = select_oracle_matrices(sequences, contexts, feature_rows, arrays)
    return (
        sequences,
        contexts,
        matrices,
        {
            "feature_receipt": receipt,
            "feature_manifest_sha256": digest(canonical(feature_manifest)),
            "projection_manifest_sha256": digest(payloads["manifest.json"]),
        },
    )


def run(
    *,
    repository: Path,
    projection: Path,
    features: Path,
    feature_receipt: Path,
    expected_commit: str,
    reviewed_config_sha256: str,
    output: Path,
) -> dict:
    started = time.monotonic()

    def git(*args):
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(repository), *args],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )

    def authenticate_source():
        if git("rev-parse", "HEAD").decode().strip() != expected_commit or git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise ValueError("benchmark requires exact clean reviewed commit")
        inventory = {}
        for name in SOURCES:
            source = git("show", f"{expected_commit}:{name}")
            if (repository / name).read_bytes() != source:
                raise ValueError("benchmark source differs from committed blob")
            inventory[name] = digest(source)
        return inventory

    source = authenticate_source()
    if source[CONFIG] != reviewed_config_sha256:
        raise ValueError("configuration differs from reviewed predeclaration")
    config = tomllib.loads((repository / CONFIG).read_text())
    check_config(config)
    if output.exists() or output.with_name(output.name + ".complete").exists():
        raise ValueError("refuse existing benchmark output")
    sequences, contexts, matrices, inputs = load_data(projection, features, feature_receipt)
    census = {
        "oracle_sequences": len(sequences),
        "oracle_union_components": len({row["union_component_id"] for row in sequences}),
        "activity_contexts": len(contexts),
        "activity_sequences": len({row["sequence_id"] for row in contexts}),
        "activity_union_components": len({row["union_component_id"] for row in contexts}),
        "activity_positive": sum(row["label"] == 1 for row in contexts),
        "activity_negative": sum(row["label"] == 0 for row in contexts),
        "dominant_union_sequences": max(
            sum(r["union_component_id"] == row["union_component_id"] for r in sequences)
            for row in sequences
        ),
    }
    if census != config["expected"]:
        raise ValueError("accepted oracle census differs")
    result = nested_validation(sequences, contexts, matrices)
    report = summarize_predictions(result["predictions"], sequences)
    report.update(
        {"scope": config["scope"], "census": census, "fit_attempts": result["fit_attempts"]}
    )
    payloads = {
        "predictions.jsonl": jsonl(result["predictions"]),
        "models.jsonl": jsonl(result["models"]),
        "inner_predictions.jsonl": jsonl(result["inner_predictions"]),
        "inner_assignments.json": canonical(result["inner_assignments"]),
        "metrics.json": canonical(report),
    }
    elapsed = time.monotonic() - started
    if (
        elapsed >= config["maximum_wall_seconds_per_execution"]
        or sum(map(len, payloads.values())) > config["maximum_output_bytes"]
    ):
        raise ValueError("benchmark exceeded predeclared time/output budget")
    if authenticate_source() != source:
        raise ValueError("source inventory changed during benchmark")
    manifest = {
        "schema_version": 1,
        "artifact": config["artifact"],
        "claims": CLAIMS,
        "producer_commit": expected_commit,
        "config_sha256": reviewed_config_sha256,
        "source_inventory": source,
        "inputs": inputs,
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": os.sys.version,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "elapsed_seconds": elapsed,
        },
        "files": {
            name: {"sha256": digest(value), "bytes": len(value)} for name, value in payloads.items()
        },
    }
    payloads["manifest.json"] = canonical(manifest)
    manifest_sha = digest(payloads["manifest.json"])
    payloads["SHA256SUMS"] = b"".join(
        f"{digest(value)}  {name}\n".encode() for name, value in sorted(payloads.items())
    )
    binding = namespace._publish_claimed_tree(
        output,
        payloads,
        marker_artifact=MARKER,
        identity={
            "producer_commit": expected_commit,
            "manifest_sha256": manifest_sha,
            "config_sha256": reviewed_config_sha256,
        },
        maximum_file_bytes=32 * 1024**2,
    )
    return {
        "output": str(output),
        "manifest_sha256": manifest_sha,
        "completion_marker_sha256": binding.marker_sha256,
        "fit_attempts": result["fit_attempts"],
        "claims": CLAIMS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "projection", "features", "feature-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--reviewed-config-sha256", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

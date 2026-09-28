"""Authenticate and independently reconstruct two controller-private executions."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np

from amp_challenge.benchmarks.ensemble_audit_math import (
    CLAIMS,
    CONFIG,
    CONFIG_SHA,
    PRODUCER_COMMIT,
    SEEDS,
    VIEWS,
    agree,
    require,
)
from amp_challenge.benchmarks.ensemble_audit_reconstruction import ARRAYS, compare_runs, reconstruct
from amp_challenge.data import generator_oracle_namespace_split as sealed
from amp_challenge.data.oracle_modeling_projection import read_projection
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.representations.peptide_esm import load_features

PROJECTION_SHA = "eabed8247a42928c054d1f88c956025d806b7d0c0f68117ae54774beab7f6089"
FEATURE_SHA = "c6c49f570be0788496b04ce2c3248b23d5babeb560b046f6aef5b006d3b325f9"
FEATURE_RECEIPT_SHA = "50aacc146c6ff25be92f2222f38552805a8ca14a52ae9e5fb175f213d087f23b"
BUDGET_ROOT = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/data/runs/oracle-calibrated-ensemble-v1-budget-67cec93d3ed7"
)
JSON_RECORDS = (
    "models",
    "calibrators",
    "partitions",
    "outer_folds",
    "covariance_blocks",
    "pairs",
    "metrics",
)
PAYLOAD_FILES = {
    *(f"arrays/{name}.npy" for name in ARRAYS),
    *(name + ".json" for name in JSON_RECORDS),
    "oracle_sequences.json",
    "activity_contexts.json",
    "array_schema.json",
}
PRODUCER_SOURCES = (
    CONFIG,
    "docs/benchmarks/oracle_calibrated_ensemble_v1.md",
    *(
        f"src/amp_challenge/benchmarks/calibrated_ensemble_{name}.py"
        for name in ("contract", "models", "evaluation", "study", "runner")
    ),
    "src/amp_challenge/benchmarks/oracle_activity_independent.py",
    "src/amp_challenge/data/oracle_modeling_projection.py",
    "src/amp_challenge/data/generator_oracle_namespace_split.py",
    "src/amp_challenge/representations/peptide_esm.py",
    "src/amp_challenge/representations/laplacian.py",
    "src/amp_challenge/descriptors.py",
    "src/amp_challenge/sequences.py",
    "src/amp_challenge/constants.py",
    "cluster/slurm/oracle_calibrated_ensemble_v1.sbatch",
    "tests/test_calibrated_ensemble.py",
    "pyproject.toml",
    "uv.lock",
)
AUDIT_SOURCES = (
    *(
        f"src/amp_challenge/benchmarks/ensemble_audit_{name}.py"
        for name in ("math", "statistics", "reconstruction", "runner")
    ),
    "docs/benchmarks/oracle_calibrated_independent_audit_v1.md",
    "tests/test_ensemble_independent_audit.py",
    "cluster/slurm/audit_oracle_calibrated_ensemble_v1.sbatch",
    *PRODUCER_SOURCES,
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def parse_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)


def git(repository, *arguments):
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )


def authenticate_source(repository, expected):
    require(re.fullmatch("[0-9a-f]{40}", expected) is not None, "invalid audit source pin")
    require(
        git(repository, "rev-parse", "HEAD").decode().strip() == expected
        and not git(repository, "status", "--porcelain", "--untracked-files=all"),
        "audit requires exact clean source",
    )
    inventory = {}
    for name in AUDIT_SOURCES:
        payload = git(repository, "show", f"{expected}:{name}")
        require((repository / name).read_bytes() == payload, "audit source blob differs")
        inventory[name] = digest(payload)
    require(inventory[CONFIG] == CONFIG_SHA, "audit configuration pin differs")
    return inventory


def _private_directory(path):
    info = path.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o700
        and info.st_uid == os.getuid(),
        "controller-private parent must be owner-only regular directory",
    )


def read_run(path, manifest_sha, execution, repository):
    require(
        path == BUDGET_ROOT / f"execution-{execution}" / "controller-private-bundle",
        "scientific slot path differs",
    )
    _private_directory(BUDGET_ROOT)
    _private_directory(path.parent)
    snapshots, marker, binding = sealed._read_committed_tree(
        path,
        expected_marker_artifact="oracle_calibrated_ensemble_study_complete_v1",
        expected_files=PAYLOAD_FILES | {"manifest.json", "SHA256SUMS"},
        maximum_file_bytes=32 * 1024**2,
        maximum_json_depth=16,
        maximum_json_containers=4096,
        maximum_json_string_bytes=65536,
    )
    require(snapshots["manifest.json"].sha256 == manifest_sha, "external manifest pin differs")
    require(
        sum(image.size for image in snapshots.values()) <= 128 * 1024**2,
        "scientific artifact budget exceeded",
    )
    manifest = parse_json(snapshots["manifest.json"].payload)
    identity = {
        "producer_commit": PRODUCER_COMMIT,
        "config_sha256": CONFIG_SHA,
        "manifest_sha256": manifest_sha,
        "execution_index": execution,
    }
    agree(marker["identity"], identity, name="completion-identity", exact=True)
    require(
        manifest["producer_commit"] == PRODUCER_COMMIT
        and manifest["config_sha256"] == CONFIG_SHA
        and manifest["execution_index"] == execution
        and manifest["schema_version"] == 1
        and manifest["artifact"] == "oracle_only_calibrated_ensemble_study_v1"
        and manifest["scope"]
        == "exploratory_binary_MIC_le_16_micromolar_predictive_validation_not_search_superiority",
        "source or scientific scope differs",
    )
    agree(manifest["claims"], CLAIMS, name="manifest-claims", exact=True)
    agree(manifest["seeds"], list(SEEDS), name="model-seeds", exact=True)
    require(
        manifest["members_per_seed"] == 8
        and manifest["pooled_members"] == 40
        and manifest["fit_attempts"] == {"base": 1600, "calibration": 200},
        "scientific attempt envelope differs",
    )
    expected_source = {
        name: digest(git(repository, "show", f"{PRODUCER_COMMIT}:{name}"))
        for name in PRODUCER_SOURCES
    }
    agree(manifest["source_inventory"], expected_source, name="producer-inventory", exact=True)
    require(set(manifest["files"]) == PAYLOAD_FILES, "saved payload inventory differs")
    for name in PAYLOAD_FILES:
        agree(
            manifest["files"][name],
            {"sha256": snapshots[name].sha256, "bytes": snapshots[name].size},
            name=name,
            exact=True,
        )
    expected_sums = b"".join(
        f"{snapshots[name].sha256}  {name}\n".encode()
        for name in sorted(PAYLOAD_FILES | {"manifest.json"})
    )
    require(snapshots["SHA256SUMS"].payload == expected_sums, "checksum ledger differs")
    require(
        manifest["inputs"]["projection_manifest_sha256"] == PROJECTION_SHA
        and manifest["inputs"]["feature_manifest_sha256"] == FEATURE_SHA
        and digest(sealed._canonical(manifest["inputs"]["feature_receipt"])) == FEATURE_RECEIPT_SHA,
        "accepted input receipt differs",
    )
    parent = sealed._open_root(path.parent)
    try:
        claim = sealed._snapshot(parent, "CLAIM.json", 65536, immutable=True)
    finally:
        os.close(parent)
    require(claim.sha256 == manifest["claim_sha256"], "scientific slot claim pin differs")
    agree(
        parse_json(claim.payload),
        {
            "artifact": "oracle_ensemble_execution_claim_v1",
            "execution_index": execution,
            "maximum_fit_attempts": 1800,
            "failed_claims_cannot_be_reused": True,
            "producer_commit": PRODUCER_COMMIT,
            "config_sha256": CONFIG_SHA,
            "job_id": manifest["runtime"]["job_id"],
        },
        name="execution-claim",
        exact=True,
    )
    arrays = {}
    schema = parse_json(snapshots["array_schema.json"].payload)
    require(set(schema) == ARRAYS, "array schema inventory differs")
    for name in ARRAYS:
        array = np.load(io.BytesIO(snapshots[f"arrays/{name}.npy"].payload), allow_pickle=False)
        require(
            array.dtype in (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))
            and np.isfinite(array).all(),
            "unsafe array encoding",
        )
        agree(
            schema[name],
            {"shape": list(array.shape), "dtype": array.dtype.str},
            name=name,
            exact=True,
        )
        arrays[name] = array
    result = {name: parse_json(snapshots[name + ".json"].payload) for name in JSON_RECORDS}
    result.update(
        {
            "arrays": arrays,
            "seeds": manifest["seeds"],
            "members_per_seed": 8,
            "claims": manifest["claims"],
        }
    )
    require(0 < manifest["runtime"]["elapsed_seconds"] < 1200, "manifest time budget differs")
    return (
        result,
        manifest,
        parse_json(snapshots["oracle_sequences.json"].payload),
        parse_json(snapshots["activity_contexts.json"].payload),
        {
            "manifest_sha256": manifest_sha,
            "completion_marker_sha256": binding.marker_sha256,
            "claim_sha256": claim.sha256,
            "path": str(path),
        },
    )


def load_inputs(projection, features):
    payloads = read_projection(projection, expected_manifest_sha256=PROJECTION_SHA)
    sequences = [parse_json(line) for line in payloads["oracle_sequences.jsonl"].splitlines()]
    rows = [parse_json(line) for line in payloads["activity_contexts.jsonl"].splitlines()]
    feature_rows, arrays, _ = load_features(features, FEATURE_SHA)
    indexed = {row["sequence_id"]: (row, index) for index, row in enumerate(feature_rows)}
    require(len(indexed) == len(feature_rows), "repeated feature ID")
    selected, geometry = {}, []
    for sequence in sequences:
        require(sequence["namespace"] == "oracle", "generator reached audit preprocessing")
        row, index = indexed[sequence["sequence_id"]]
        require(row["sequence"] == sequence["sequence"], "sequence feature identity differs")
        esm = np.asarray(arrays["esm_length"][index], dtype=np.float64)
        descriptor = np.asarray(list(compute_descriptors(sequence["sequence"]).as_dict().values()))
        selected[sequence["sequence_id"]] = (descriptor, esm)
        geometry.append(esm)
    require(
        all(row["namespace"] == "oracle" and row["sequence_id"] in selected for row in rows),
        "non-oracle modeling row",
    )
    matrices = {
        view: np.asarray([selected[row["sequence_id"]][index] for row in rows])
        for index, view in enumerate(VIEWS)
    }
    census = {
        "oracle_sequences": len(sequences),
        "oracle_union_components": len({row["union_component_id"] for row in sequences}),
        "activity_contexts": len(rows),
        "activity_sequences": len({row["sequence_id"] for row in rows}),
        "activity_union_components": len({row["union_component_id"] for row in rows}),
        "activity_positive": sum(row["label"] == 1 for row in rows),
        "activity_negative": sum(row["label"] == 0 for row in rows),
        "dominant_union_sequences": max(
            sum(other["union_component_id"] == row["union_component_id"] for other in sequences)
            for row in sequences
        ),
    }
    return sequences, rows, matrices, np.asarray(geometry), census


def terminal_allocations(manifests):
    ids = [manifest["runtime"]["job_id"] for manifest in manifests]
    require(
        len(set(ids)) == 2
        and all(type(job) is str and re.fullmatch("[1-9][0-9]*", job) for job in ids),
        "separately scheduled job identities differ",
    )
    fields = (
        "JobIDRaw",
        "State",
        "ExitCode",
        "Account",
        "Partition",
        "AllocCPUS",
        "AllocNodes",
        "AllocTRES",
        "ElapsedRaw",
        "TimelimitRaw",
        "Restarts",
        "NodeList",
    )
    payload = subprocess.check_output(
        ["/usr/bin/sacct", "-j", ",".join(ids), "--noheader", "-P", "--format=" + ",".join(fields)],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    records = []
    for line in payload.decode().splitlines():
        values = line.split("|")
        require(len(values) == len(fields), "scheduler schema differs")
        record = dict(zip(fields, values, strict=True))
        if record["JobIDRaw"] in ids:
            records.append(record)
    require(
        len(records) == 2
        and {row["JobIDRaw"] for row in records} == set(ids)
        and len({row["NodeList"] for row in records}) == 2,
        "independent-node allocation inventory differs",
    )
    by_job = {manifest["runtime"]["job_id"]: manifest for manifest in manifests}
    for record in records:
        tres = dict(item.split("=", 1) for item in record["AllocTRES"].split(","))
        require(
            record["State"] == "COMPLETED"
            and record["ExitCode"] == "0:0"
            and record["Account"] == "bio"
            and record["Partition"] == "standard"
            and record["AllocCPUS"] == "2"
            and record["AllocNodes"] == "1"
            and record["Restarts"] == "0"
            and 0 < int(record["ElapsedRaw"]) <= 1200
            and int(record["TimelimitRaw"]) == 20
            and tres.get("cpu") == "2"
            and tres.get("mem") in {"4G", "4096M"}
            and tres.get("node") == "1"
            and not any("gpu" in key for key in tres),
            "terminal resource envelope differs",
        )
        require(
            record["NodeList"] == by_job[record["JobIDRaw"]]["runtime"]["node"],
            "runtime node binding differs",
        )
    return {
        "terminal_allocations": records,
        "scheduler_stdout_sha256": digest(payload),
        "fit_attempts_total": 3600,
        "total_allocated_cpu_seconds": sum(2 * int(record["ElapsedRaw"]) for record in records),
        "gpu_seconds": 0,
        "oracle_calls": 0,
        "scientific_fit_budget_closed": True,
    }


def run(
    *,
    repository,
    expected_commit,
    first,
    first_manifest_sha256,
    second,
    second_manifest_sha256,
    projection,
    features,
    output,
):
    start = time.monotonic()
    require(
        re.fullmatch("[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", "")) is not None,
        "audit requires CPU Slurm",
    )
    _private_directory(output.parent)
    source = authenticate_source(repository, expected_commit)
    protocol = tomllib.loads((repository / CONFIG).read_text())
    sequences, rows, matrices, geometry, census = load_inputs(projection, features)
    agree(census, protocol["expected"], name="accepted-census", exact=True)
    loaded = [
        read_run(first, first_manifest_sha256, 0, repository),
        read_run(second, second_manifest_sha256, 1, repository),
    ]
    resources = terminal_allocations([item[1] for item in loaded])
    reconstructions, bindings = [], []
    for result, manifest, saved_sequences, saved_rows, binding in loaded:
        agree(saved_sequences, sequences, name="sealed-sequences", exact=True)
        agree(saved_rows, rows, name="sealed-activity-rows", exact=True)
        agree(manifest["inputs"]["census"], census, name="manifest-census", exact=True)
        checked = reconstruct(result, sequences, rows, matrices, geometry)
        diagnostics = checked["optimization_diagnostics_not_new_acceptance_thresholds"]
        agree(
            manifest["actual_optimizer_invocations"],
            {
                "base": diagnostics["base_optimizer_invocations"],
                "calibration": diagnostics["calibration_optimizer_invocations"],
            },
            name="actual-optimizer-counts",
            exact=True,
        )
        reconstructions.append(checked)
        bindings.append(binding)
    reproduction = compare_runs(loaded[0][0], loaded[1][0])
    require(time.monotonic() - start < 600, "audit wall budget exceeded")
    agree(
        authenticate_source(repository, expected_commit),
        source,
        name="audit-source-end",
        exact=True,
    )
    document = {
        "artifact": "independent_calibrated_ensemble_reproduction_v1",
        "audit_commit": expected_commit,
        "producer_commit": PRODUCER_COMMIT,
        "config_sha256": CONFIG_SHA,
        "audit_source_inventory": source,
        "inputs": bindings,
        "independent_reconstructions": reconstructions,
        "reproduction": reproduction,
        "resources": resources,
        "claims": CLAIMS,
        "shared_infrastructure": [
            "sealed_filesystem_and_projection_reader",
            "pinned_feature_loader_and_deterministic_descriptors",
            "NumPy_PCG64_not_independently_reimplemented_RNG",
        ],
        "limitations": [
            "no_optimizer_or_global_optimality_proof",
            "fixed_OOF_conditional_uncertainty_not_retraining_variance",
            "predictive_events_not_latent_epistemic_calibration",
            "same_account_not_independent_laboratory",
            "teacher_artifacts_remain_controller_private",
            "no_final_all_oracle_refit_or_search_campaign_authority",
        ],
        "runtime": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": os.environ.get("SLURMD_NODENAME"),
            "elapsed_seconds": time.monotonic() - start,
        },
    }
    payload = sealed._canonical(document)
    require(len(payload) <= 8 * 1024**2, "audit artifact budget exceeded")
    binding = sealed._publish_claimed_tree(
        output,
        {"verification.json": payload},
        marker_artifact="independent_calibrated_ensemble_audit_complete_v1",
        identity={
            "audit_commit": expected_commit,
            "producer_commit": PRODUCER_COMMIT,
            "config_sha256": CONFIG_SHA,
            "verification_sha256": digest(payload),
        },
        maximum_file_bytes=8 * 1024**2,
    )
    return {
        "output": str(output),
        "verification_sha256": digest(payload),
        "completion_marker_sha256": binding.marker_sha256,
        "independent_reconstruction_passed": True,
        "reproduction_passed": True,
        "production_input_eligible": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "first", "second", "projection", "features", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-commit", "first-manifest-sha256", "second-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

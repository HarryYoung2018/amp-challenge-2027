"""Authenticate and independently check two frozen oracle activity executions.

This program never fits a model. Read-only verification may run on CPU Slurm;
its newly published receipt grants no search or production authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np

from amp_challenge.benchmarks.oracle_activity_independent import (
    ARMS,
    compare_executions,
    compare_values,
    reconstruct_execution,
    require,
)
from amp_challenge.data import generator_oracle_namespace_split as sealed
from amp_challenge.data.oracle_modeling_projection import read_projection
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.representations.peptide_esm import load_features

PRODUCER_COMMIT = "cc472a3b90509ee814016fcd49eb8c218d090356"
CONFIG = "configs/benchmarks/oracle_activity_nested_v1.toml"
CONFIG_SHA = "286930676ca292ce735bdf956cd5fd0582dceab33539aa4eef42032798791c76"
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
PRODUCER_SOURCES = (
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
AUDIT_SOURCES = (
    "src/amp_challenge/benchmarks/oracle_activity_independent.py",
    "src/amp_challenge/benchmarks/oracle_activity_reproduction.py",
    "tests/test_oracle_activity_independent.py",
    "docs/benchmarks/oracle_activity_independent_audit_v1.md",
    "cluster/slurm/audit_oracle_activity_reproduction_v1.sbatch",
    *PRODUCER_SOURCES,
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


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def git(repository, *args):
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repository), *args], env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    )


def authenticate_source(repository, expected):
    require(
        git(repository, "rev-parse", "HEAD").decode().strip() == expected,
        "audit source HEAD differs",
    )
    require(
        not git(repository, "status", "--porcelain", "--untracked-files=all"),
        "audit checkout is dirty",
    )
    inventory = {}
    for relative in AUDIT_SOURCES:
        payload = git(repository, "show", f"{expected}:{relative}")
        require((repository / relative).read_bytes() == payload, "audit source blob differs")
        inventory[relative] = digest(payload)
    return inventory


def _json(payload):
    def no_duplicates(pairs):
        parsed = {}
        for key, value in pairs:
            require(key not in parsed, "duplicate JSON key")
            parsed[key] = value
        return parsed

    def no_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=no_constant)


def read_run(path, manifest_sha, repository):
    snapshots, marker, binding = sealed._read_committed_tree(
        path,
        expected_marker_artifact="oracle_only_activity_benchmark_complete_v1",
        expected_files=set((*DATA_FILES, "manifest.json", "SHA256SUMS")),
        maximum_file_bytes=32 * 1024**2,
        maximum_json_depth=16,
        maximum_json_containers=4096,
        maximum_json_string_bytes=65536,
    )
    require(snapshots["manifest.json"].sha256 == manifest_sha, "run manifest pin differs")
    manifest = _json(snapshots["manifest.json"].payload)
    require(
        marker["identity"]
        == {
            "producer_commit": PRODUCER_COMMIT,
            "manifest_sha256": manifest_sha,
            "config_sha256": CONFIG_SHA,
        },
        "run completion identity differs",
    )
    require(
        manifest["producer_commit"] == PRODUCER_COMMIT
        and manifest["config_sha256"] == CONFIG_SHA
        and manifest["claims"] == CLAIMS
        and manifest["artifact"] == "oracle_only_activity_nested_benchmark_v1"
        and manifest["schema_version"] == 1,
        "run source/qualification differs",
    )
    expected_sources = {
        name: digest(git(repository, "show", f"{PRODUCER_COMMIT}:{name}"))
        for name in PRODUCER_SOURCES
    }
    require(
        manifest["source_inventory"] == expected_sources, "run producer source inventory differs"
    )
    require(set(manifest["files"]) == set(DATA_FILES), "run payload inventory differs")
    total = 0
    for name in DATA_FILES:
        image = snapshots[name]
        total += image.size
        require(
            manifest["files"][name] == {"sha256": image.sha256, "bytes": image.size},
            "run payload binding differs",
        )
    require(total <= 128 * 1024**2, "run output budget exceeded")
    expected_sums = b"".join(
        f"{snapshots[name].sha256}  {name}\n".encode()
        for name in sorted((*DATA_FILES, "manifest.json"))
    )
    require(snapshots["SHA256SUMS"].payload == expected_sums, "run checksum inventory differs")
    require(
        manifest["inputs"]["projection_manifest_sha256"] == PROJECTION_SHA
        and manifest["inputs"]["feature_manifest_sha256"] == FEATURE_SHA,
        "run accepted input pins differ",
    )
    require(
        digest(sealed._canonical(manifest["inputs"]["feature_receipt"])) == FEATURE_RECEIPT_SHA,
        "run feature audit receipt differs",
    )
    parsed = {
        name.removesuffix(".jsonl").removesuffix(".json"): [
            _json(line) for line in snapshots[name].payload.splitlines()
        ]
        if name.endswith(".jsonl")
        else _json(snapshots[name].payload)
        for name in DATA_FILES
    }
    require(0 < manifest["runtime"]["elapsed_seconds"] < 1200, "run wall limit differs")
    return (
        parsed,
        manifest,
        {
            "manifest_sha256": manifest_sha,
            "completion_marker_sha256": binding.marker_sha256,
            "output": str(path),
        },
    )


def load_independent_inputs(projection, features):
    payloads = read_projection(projection, expected_manifest_sha256=PROJECTION_SHA)
    sequences = [_json(line) for line in payloads["oracle_sequences.jsonl"].splitlines()]
    contexts = [_json(line) for line in payloads["activity_contexts.jsonl"].splitlines()]
    feature_rows, arrays, _ = load_features(features, FEATURE_SHA)
    cache_index = {row["sequence_id"]: index for index, row in enumerate(feature_rows)}
    require(len(cache_index) == len(feature_rows), "feature identity repeats")
    selected = {}
    for row in sequences:
        require(row["namespace"] == "oracle", "non-oracle sequence reached audit")
        index = cache_index[row["sequence_id"]]
        require(feature_rows[index]["sequence"] == row["sequence"], "feature sequence differs")
        esm = np.asarray(arrays["esm_length"][index], dtype=np.float64)
        spectral = np.asarray(arrays["spectral"][index], dtype=np.float64)
        selected[row["sequence_id"]] = (
            np.asarray(list(compute_descriptors(row["sequence"]).as_dict().values())),
            esm,
            np.concatenate((spectral, [esm[-1]])),
            np.concatenate((esm, spectral)),
        )
    require(
        all(row["namespace"] == "oracle" and row["sequence_id"] in selected for row in contexts),
        "non-oracle context reached audit",
    )
    matrices = {
        arm: np.asarray([selected[row["sequence_id"]][index] for row in contexts])
        for index, arm in enumerate(ARMS)
    }
    return sequences, contexts, matrices


def _census(sequences, contexts):
    groups = {}
    for row in sequences:
        groups[row["union_component_id"]] = groups.get(row["union_component_id"], 0) + 1
    return {
        "oracle_sequences": len(sequences),
        "oracle_union_components": len(groups),
        "activity_contexts": len(contexts),
        "activity_sequences": len({row["sequence_id"] for row in contexts}),
        "activity_union_components": len({row["union_component_id"] for row in contexts}),
        "activity_positive": sum(row["label"] == 1 for row in contexts),
        "activity_negative": sum(row["label"] == 0 for row in contexts),
        "dominant_union_sequences": max(groups.values()),
    }


def terminal_allocations(manifests):
    """Require two terminal successful, non-restarted, bounded CPU allocations."""
    ids = [manifest["runtime"]["job_id"] for manifest in manifests]
    require(
        len(set(ids)) == 2
        and all(isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job) for job in ids),
        "invalid separately scheduled job identities",
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
        require(len(values) == len(fields), "scheduler field count differs")
        record = dict(zip(fields, values, strict=True))
        if record["JobIDRaw"] in ids:
            records.append(record)
    require(
        len(records) == 2 and {row["JobIDRaw"] for row in records} == set(ids),
        "scheduler allocation inventory differs",
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
            and 0 < int(record["TimelimitRaw"]) <= 20,
            "terminal allocation violates frozen resource envelope",
        )
        require(
            tres.get("cpu") == "2"
            and tres.get("mem") in {"4G", "4096M"}
            and tres.get("node") == "1"
            and not any("gpu" in key for key in tres),
            "terminal allocation TRES differs",
        )
        require(
            record["NodeList"] == by_job[record["JobIDRaw"]]["runtime"]["node"],
            "runtime node differs from scheduler",
        )
    return {
        "terminal_allocations": records,
        "scheduler_stdout_sha256": digest(payload),
        "fit_attempts_total": 520,
        "total_allocated_cpu_seconds": sum(2 * int(record["ElapsedRaw"]) for record in records),
        "gpu_seconds": 0,
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
    started = time.monotonic()
    source = authenticate_source(repository, expected_commit)
    require(source[CONFIG] == CONFIG_SHA, "audit config pin differs")
    config = tomllib.loads((repository / CONFIG).read_text())
    one, one_manifest, one_identity = read_run(first, first_manifest_sha256, repository)
    two, two_manifest, two_identity = read_run(second, second_manifest_sha256, repository)
    require(
        first != second and one_manifest["runtime"]["job_id"] != two_manifest["runtime"]["job_id"],
        "reproduction requires separately scheduled executions",
    )
    resources = terminal_allocations((one_manifest, two_manifest))
    sequences, contexts, matrices = load_independent_inputs(projection, features)
    census = _census(sequences, contexts)
    require(census == config["expected"], "audit census differs")
    for document in (one, two):
        reconstructed = reconstruct_execution(document, sequences, contexts, matrices)
        reconstructed.update({"scope": config["scope"], "census": census, "fit_attempts": 260})
        compare_values(
            document["metrics"], reconstructed, location="independently reconstructed report"
        )
    comparison = compare_executions(one, two)
    require(authenticate_source(repository, expected_commit) == source, "audit source changed")
    receipt = {
        "schema_version": 1,
        "artifact": "oracle_activity_independent_reproduction_v1",
        "audit_commit": expected_commit,
        "audit_source_inventory": source,
        "producer_commit": PRODUCER_COMMIT,
        "config_sha256": CONFIG_SHA,
        "executions": [one_identity, two_identity],
        "resources": resources,
        "independent_reconstruction_passed": True,
        "reproduction": comparison,
        "tolerances": config["reproduction"],
        "scope": "binary_MIC16_predictive_validation_only",
        "limitations": [
            "same-account audit is not an independent laboratory",
            "inner coefficients were not saved; inner supported probabilities are cross-run compared, not coefficient-replayed",
            "bootstrap is conditional on fixed OOF predictors, not retraining variance",
        ],
        "optimizer_invocations": 0,
        "oracle_calls": 0,
        "production_input_eligible": False,
        "search_superiority_accepted": False,
        "runtime": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "elapsed_seconds": time.monotonic() - started,
            "numpy": np.__version__,
            "python": os.sys.version,
        },
    }
    require(receipt["runtime"]["elapsed_seconds"] < 600, "audit wall budget exceeded")
    payload = sealed._canonical(receipt)
    binding = sealed._publish_claimed_tree(
        output,
        {"verification.json": payload},
        marker_artifact="oracle_activity_independent_reproduction_complete_v1",
        identity={"audit_commit": expected_commit, "receipt_sha256": digest(payload)},
        maximum_file_bytes=1024**2,
    )
    return {
        "output": str(output),
        "receipt_sha256": digest(payload),
        "completion_marker_sha256": binding.marker_sha256,
        "independent_reconstruction_passed": True,
        "reproduction_passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "first", "second", "projection", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-commit", "first-manifest-sha256", "second-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

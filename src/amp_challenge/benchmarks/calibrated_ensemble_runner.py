"""Pinned, budget-claimed controller-private oracle ensemble study runner.

Real execution requires root review and an explicit fit-mode Slurm submission.
The two exclusive execution slots cannot be retried or automatically replaced.
"""

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
from pathlib import Path

import numpy as np
import scipy

from amp_challenge.benchmarks.calibrated_ensemble_contract import (
    CONFIG,
    CONFIG_SHA,
    SEEDS,
    VIEWS,
    load_protocol,
    private_teacher_claims,
    require,
)
from amp_challenge.benchmarks.calibrated_ensemble_study import fit_study
from amp_challenge.data import generator_oracle_namespace_split as sealed
from amp_challenge.data.oracle_modeling_projection import read_projection
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.representations.peptide_esm import load_features

BUDGET_ROOT = Path("/lustre/scratch/users/yonghan.yang/amp_challenge/data/runs") / (
    "oracle-calibrated-ensemble-v1-budget-" + CONFIG_SHA[:12]
)
SOURCES = (
    CONFIG,
    "docs/benchmarks/oracle_calibrated_ensemble_v1.md",
    "src/amp_challenge/benchmarks/calibrated_ensemble_contract.py",
    "src/amp_challenge/benchmarks/calibrated_ensemble_models.py",
    "src/amp_challenge/benchmarks/calibrated_ensemble_evaluation.py",
    "src/amp_challenge/benchmarks/calibrated_ensemble_study.py",
    "src/amp_challenge/benchmarks/calibrated_ensemble_runner.py",
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


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def source_inventory(repository, expected_commit):
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "invalid exact source commit"
    )

    def git(*args):
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(repository), *args],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )

    require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit
        and not git("status", "--porcelain", "--untracked-files=all"),
        "ensemble requires the exact clean reviewed source",
    )
    inventory = {}
    for name in SOURCES:
        payload = git("show", f"{expected_commit}:{name}")
        require(
            (repository / name).read_bytes() == payload,
            "ensemble source differs from committed blob",
        )
        inventory[name] = digest(payload)
    return inventory


def _pinned(path, expected_sha):
    parent = sealed._open_root(path.parent)
    try:
        image = sealed._snapshot(parent, path.name, 32 * 1024**2, immutable=True)
    finally:
        os.close(parent)
    require(image.sha256 == expected_sha, "input audit receipt pin differs")
    return json.loads(image.payload)


def load_inputs(projection, features, feature_receipt, protocol):
    payloads = read_projection(
        projection, expected_manifest_sha256=protocol["projection_manifest_sha256"]
    )
    receipt = _pinned(feature_receipt, protocol["feature_verification_sha256"])
    sequences = [json.loads(line) for line in payloads["oracle_sequences.jsonl"].splitlines()]
    rows = [json.loads(line) for line in payloads["activity_contexts.jsonl"].splitlines()]
    feature_rows, arrays, feature_manifest = load_features(
        features, protocol["feature_manifest_sha256"]
    )
    index = {row["sequence_id"]: i for i, row in enumerate(feature_rows)}
    require(
        len(index) == len(feature_rows) and all(row["namespace"] == "oracle" for row in sequences),
        "feature identity or oracle namespace differs",
    )
    for row in sequences:
        require(
            row["sequence_id"] in index
            and feature_rows[index[row["sequence_id"]]]["sequence"] == row["sequence"],
            "oracle feature identity differs",
        )
    # Select the oracle inventory before descriptors, transforms or fitting.
    geometry = np.asarray(
        arrays["esm_length"][[index[row["sequence_id"]] for row in sequences]], dtype=np.float64
    )
    selected = {row["sequence_id"]: i for i, row in enumerate(sequences)}
    descriptor = np.asarray(
        [list(compute_descriptors(row["sequence"]).as_dict().values()) for row in sequences]
    )
    require(
        all(row["namespace"] == "oracle" and row["sequence_id"] in selected for row in rows),
        "non-oracle modeling row",
    )
    indices = [selected[row["sequence_id"]] for row in rows]
    matrices = {VIEWS[0]: descriptor[indices], VIEWS[1]: geometry[indices]}
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
    require(census == protocol["expected"], "frozen real oracle census differs")
    return (
        sequences,
        rows,
        matrices,
        geometry,
        {
            "projection_manifest_sha256": digest(payloads["manifest.json"]),
            "feature_manifest_sha256": digest(sealed._canonical(feature_manifest)),
            "feature_receipt": receipt,
            "census": census,
        },
    )


def claim_execution(budget_root, execution_index, identity):
    require(
        type(execution_index) is int and execution_index in (0, 1),
        "only two scientific execution slots exist",
    )
    budget_root.mkdir(mode=0o700, exist_ok=True)
    info = budget_root.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o700
        and info.st_uid == os.getuid(),
        "scientific budget root must be owner-private regular directory",
    )
    slot = budget_root / f"execution-{execution_index}"
    slot.mkdir(mode=0o700)  # Exclusive permanent claim; failed jobs cannot retry.
    payload = sealed._canonical(
        {
            "artifact": "oracle_ensemble_execution_claim_v1",
            "execution_index": execution_index,
            "maximum_fit_attempts": 1800,
            "failed_claims_cannot_be_reused": True,
            **identity,
        }
    )
    with (slot / "CLAIM.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    (slot / "CLAIM.json").chmod(0o444)
    return slot, digest(payload)


def encode_result(result, sequences, rows):
    payloads, array_inventory = {}, {}
    for name, values in result["arrays"].items():
        require(
            re.fullmatch(r"[a-z_]+", name) is not None
            and values.dtype in (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))
            and np.isfinite(values).all(),
            "unsafe saved array",
        )
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        payloads[f"arrays/{name}.npy"] = stream.getvalue()
        array_inventory[name] = {"shape": list(values.shape), "dtype": values.dtype.str}
    for name in (
        "models",
        "calibrators",
        "partitions",
        "outer_folds",
        "covariance_blocks",
        "pairs",
        "metrics",
    ):
        payloads[name + ".json"] = sealed._canonical(result[name])
    payloads["oracle_sequences.json"] = sealed._canonical(sequences)
    payloads["activity_contexts.json"] = sealed._canonical(rows)
    payloads["array_schema.json"] = sealed._canonical(array_inventory)
    require(
        sum(map(len, payloads.values())) <= 128 * 1024**2
        and max(map(len, payloads.values())) <= 32 * 1024**2,
        "study artifact budget exceeded",
    )
    return payloads


def run(
    *,
    repository,
    expected_commit,
    reviewed_config_sha256,
    execution_index,
    projection,
    features,
    feature_receipt,
):
    started = time.monotonic()
    require(
        re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", "")) is not None,
        "real study requires a reviewed Slurm allocation",
    )
    require(reviewed_config_sha256 == CONFIG_SHA, "caller configuration review pin differs")
    source = source_inventory(repository, expected_commit)
    protocol = load_protocol(repository)
    sequences, rows, matrices, geometry, inputs = load_inputs(
        projection, features, feature_receipt, protocol
    )
    identity = {
        "producer_commit": expected_commit,
        "config_sha256": CONFIG_SHA,
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    slot, claim_sha = claim_execution(BUDGET_ROOT, execution_index, identity)
    print(
        json.dumps(
            {
                "event": "scientific_execution_claimed",
                "slot": str(slot),
                "claim_sha256": claim_sha,
                **identity,
            }
        ),
        flush=True,
    )

    def progress(event):
        require(time.monotonic() - started < 1200, "execution wall budget exceeded before attempt")
        print(json.dumps(event, sort_keys=True), flush=True)

    result = fit_study(
        sequences, rows, matrices, geometry, seeds=SEEDS, members_per_seed=8, progress=progress
    )
    require(
        result["metrics"]["fit_attempts"] == {"base": 1600, "calibration": 200}
        and result["metrics"]["total_fit_attempts"] == 1800,
        "real scientific attempt envelope differs",
    )
    payloads = encode_result(result, sequences, rows)
    require(
        source_inventory(repository, expected_commit) == source, "source changed during fitting"
    )
    elapsed = time.monotonic() - started
    require(elapsed < 1200, "execution wall budget exceeded before publication")
    manifest = {
        "schema_version": 1,
        "artifact": protocol["artifact"],
        "scope": protocol["scope"],
        "source_inventory": source,
        "producer_commit": expected_commit,
        "config_sha256": CONFIG_SHA,
        "execution_index": execution_index,
        "claim_sha256": claim_sha,
        "inputs": inputs,
        "seeds": result["seeds"],
        "members_per_seed": 8,
        "pooled_members": 40,
        "fit_attempts": {"base": 1600, "calibration": 200},
        "actual_optimizer_invocations": {
            "base": sum(not record["constant_training_labels"] for record in result["models"]),
            "calibration": sum(record["fitted"] for record in result["calibrators"]),
        },
        "claims": private_teacher_claims(),
        "runtime": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": os.environ.get("SLURMD_NODENAME"),
            "elapsed_seconds": elapsed,
            "python": os.sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "files": {
            name: {"sha256": digest(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    payloads["manifest.json"] = sealed._canonical(manifest)
    manifest_sha = digest(payloads["manifest.json"])
    payloads["SHA256SUMS"] = b"".join(
        f"{digest(payload)}  {name}\n".encode() for name, payload in sorted(payloads.items())
    )
    require(
        sum(map(len, payloads.values())) <= 128 * 1024**2,
        "complete publication byte budget exceeded",
    )
    output = slot / "controller-private-bundle"
    binding = sealed._publish_claimed_tree(
        output,
        payloads,
        marker_artifact="oracle_calibrated_ensemble_study_complete_v1",
        identity={
            "producer_commit": expected_commit,
            "config_sha256": CONFIG_SHA,
            "manifest_sha256": manifest_sha,
            "execution_index": execution_index,
        },
        maximum_file_bytes=32 * 1024**2,
    )
    return {
        "output": str(output),
        "manifest_sha256": manifest_sha,
        "completion_marker_sha256": binding.marker_sha256,
        "claim_sha256": claim_sha,
        "fit_attempts": 1800,
        "claims": private_teacher_claims(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "projection", "features", "feature-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--reviewed-config-sha256", required=True)
    parser.add_argument("--execution-index", type=int, choices=(0, 1), required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Separately executed reconstruction; deliberately does not import the producer.

This same-account verifier authenticates immutable files and live terminal Slurm
allocations. It is not a separate author or isolation from a malicious account.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import stat
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from amp_challenge.generators.diffusion.categorical import (
    AbsorbingDiffusion,
    CosineMaskSchedule,
    PeptideVocabulary,
)
from amp_challenge.generators.diffusion.data import namespaced_seed
from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    canonical_model_logical_hash,
    configure_deterministic_runtime,
    load_safetensors_checkpoint,
    masked_token_objective,
)
from amp_challenge.generators.diffusion.namespace_checkpoint_contract import (
    ARTIFACT_FILES,
    TRIPLES,
    authenticate_generator_rows,
    canonical,
    learning_rate,
    load_contract,
    projection_bytes,
    read_regular,
    require,
    sha,
    training_distribution,
)
from amp_challenge.sequences import canonical_sequence_id

SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "Account",
    "Partition",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocCPUS",
    "AllocTRES",
    "Start",
    "End",
    "NodeList",
    "SubmitLine",
)


def parse_scheduler(payload: str) -> list[dict[str, str]]:
    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        cells = line.split("|")
        require(len(cells) == len(SACCT_FIELDS), "scheduler column count differs")
        row = dict(zip(SACCT_FIELDS, cells, strict=True))
        require(row["JobIDRaw"].isdigit(), "scheduler job identity differs")
        rows.append(row)
    require(len({row["JobIDRaw"] for row in rows}) == len(rows), "scheduler jobs repeat")
    return rows


def verify_scheduler(
    rows: list[dict[str, str]],
    manifests: list[dict],
    *,
    primary_array: str,
    duplicate_job: str,
    expected_commit: str,
    config_sha256: str,
    run_root: Path,
) -> dict:
    require(
        primary_array.isdigit() and duplicate_job.isdigit() and primary_array != duplicate_job,
        "external scheduler identities differ",
    )
    require(
        len(rows) == len(manifests) == 11,
        "exact eleven allocations required, including failed attempts",
    )
    by_job = {row["JobIDRaw"]: row for row in rows}
    require(
        set(by_job) == {item["job_id"] for item in manifests},
        "scheduler/manifest job inventory differs",
    )
    intervals = []
    elapsed_total = 0
    for index, item in enumerate(manifests):
        row = by_job[item["job_id"]]
        require(
            row["JobID"] == (f"{primary_array}_{index}" if index < 10 else duplicate_job),
            "fit scheduler array mapping differs",
        )
        require(
            item["array_job_id"] == (primary_array if index < 10 else None),
            "manifest array mapping differs",
        )
        require(
            row["JobName"] == "amp-ns-checkpoint-v1"
            and row["Account"] == "bio"
            and row["Partition"] == "gpumid",
            "scheduler allocation scope differs",
        )
        require(
            row["State"] == "COMPLETED" and row["ExitCode"] == "0:0",
            "scheduler allocation is not terminal successful",
        )
        tres = dict(part.split("=", 1) for part in row["AllocTRES"].split(","))
        require(
            row["AllocCPUS"] == "4"
            and tres.get("cpu") == "4"
            and tres.get("node") == "1"
            and tres.get("gres/gpu") == "1"
            and tres.get("mem") in {"16G", "16384M"},
            "scheduler resource allocation differs",
        )
        elapsed = int(row["ElapsedRaw"])
        require(0 < elapsed <= 900, "scheduler fit duration exceeds budget")
        started = datetime.fromisoformat(row["Start"])
        ended = datetime.fromisoformat(row["End"])
        require(
            (ended - started).total_seconds() == elapsed, "scheduler interval differs from elapsed"
        )
        require(row["NodeList"] == item["node"], "scheduler node differs from producer")
        command = shlex.split(row["SubmitLine"])
        launcher = "cluster/slurm/train_generator_namespace_checkpoints_v1.sbatch"
        positions = [
            offset
            for offset, value in enumerate(command)
            if value == launcher or value.endswith("/" + launcher)
        ]
        require(len(positions) == 1, "scheduler submission launcher differs")
        expected_args = [expected_commit, config_sha256, str(run_root)] + (
            ["10"] if index == 10 else []
        )
        require(
            command[positions[0] + 1 :] == expected_args,
            "scheduler submission source/config/output differs",
        )
        intervals.extend(((started, 1), (ended, -1)))
        elapsed_total += elapsed
    require(elapsed_total <= 9900, "aggregate scheduler GPU budget exceeded")
    active = peak = 0
    for _, change in sorted(intervals):
        active += change
        peak = max(peak, active)
    require(active == 0 and peak <= 2, "scheduler concurrency exceeds two GPUs")
    duplicate_start = datetime.fromisoformat(by_job[manifests[10]["job_id"]]["Start"])
    require(
        all(
            duplicate_start >= datetime.fromisoformat(by_job[item["job_id"]]["End"])
            for item in manifests[:10]
        ),
        "duplicate began before primary fits finished",
    )
    require(
        manifests[0]["node"] != manifests[10]["node"], "duplicate requires a different compute node"
    )
    return {
        "terminal_allocations": rows,
        "allocation_count": 11,
        "actual_gpu_seconds": elapsed_total,
        "actual_a100_hours": elapsed_total / 3600,
        "peak_concurrent_gpus": peak,
    }


def artifact_payloads(root: Path) -> dict[str, bytes]:
    entry = root.stat(follow_symlinks=False)
    require(
        stat.S_ISDIR(entry.st_mode) and stat.S_IMODE(entry.st_mode) == 0o555,
        "artifact root is not an immutable directory",
    )
    require(
        {path.name for path in root.iterdir()} == ARTIFACT_FILES,
        "exclusive artifact inventory differs",
    )
    payloads = {}
    for name in sorted(ARTIFACT_FILES):
        path = root / name
        require(
            stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444,
            "artifact file mode differs",
        )
        payloads[name] = read_regular(path)
    complete = json.loads(payloads["COMPLETE.json"])
    require(
        complete
        == {
            "artifact": "generator_namespace_checkpoint_complete_v1",
            "manifest_sha256": sha(payloads["manifest.json"]),
        },
        "completion manifest binding differs",
    )
    manifest = json.loads(payloads["manifest.json"])
    expected = {
        name: {"sha256": sha(payloads[name]), "bytes": len(payloads[name])}
        for name in sorted(ARTIFACT_FILES - {"manifest.json", "COMPLETE.json"})
    }
    require(manifest["artifacts"] == expected, "manifest file inventory differs")
    return payloads


def verify_trace(payload: bytes, distribution, seed: int) -> set[str]:
    records = [json.loads(line) for line in payload.splitlines()]
    require(len(records) == 1000, "training trace step count differs")
    schedule = CosineMaskSchedule(0.008)
    consumed = set()
    for step, record in enumerate(records, 1):
        require(
            set(record)
            == {
                "step",
                "sequence_ids",
                "levels",
                "loss",
                "gradient_norm",
                "learning_rate",
                "selected_tokens",
            },
            "training trace schema differs",
        )
        start = (step - 1) * 64
        drawn = distribution.draw(root_seed=seed, draw_start=start, draw_count=64)
        ids = [row.sequence_id for row in drawn]
        levels = [
            1 + namespaced_seed(seed, "timestep", ordinal, 0) % 64
            for ordinal in range(start, start + 64)
        ]
        require(
            record["step"] == step and record["sequence_ids"] == ids and record["levels"] == levels,
            "reconstructed training draws/levels differ",
        )
        require(record["learning_rate"] == learning_rate(step), "learning-rate trace differs")
        for field in ("loss", "gradient_norm"):
            require(
                type(record[field]) in (float, int)
                and math.isfinite(record[field])
                and record[field] >= 0,
                "nonfinite/negative training trace",
            )
        probabilities = schedule.probability(np.asarray(levels, dtype=np.float64) / 64)
        count = sum(
            max(1, min(len(row.sequence), math.ceil(float(probability) * len(row.sequence))))
            for row, probability in zip(drawn, probabilities, strict=True)
        )
        require(record["selected_tokens"] == count, "reconstructed corruption token count differs")
        consumed.update(ids)
    return consumed


def verify_samples(
    payload: bytes, distribution, *, triple: str, config_sha256: str, checkpoint_sha256: str
) -> dict:
    records = [json.loads(line) for line in payload.splitlines()]
    require(len(records) == 128, "raw sample count differs")
    seed = namespaced_seed(650000, "diagnostic-sampling", triple)
    lengths = distribution.length_prior.draw(root_seed=seed, draw_start=0, draw_count=128)
    identities = []
    valid = 0
    for ordinal, (record, length) in enumerate(zip(records, lengths, strict=True)):
        require(
            set(record)
            == {
                "schema_version",
                "ordinal",
                "seed",
                "sequence_id",
                "sequence",
                "length",
                "checkpoint_logical_sha256",
                "contract_sha256",
            },
            "sample schema differs",
        )
        require(
            type(record["ordinal"]) is int
            and record["ordinal"] == ordinal
            and record["schema_version"] == 1
            and record["seed"] == seed,
            "sample ordinal/seed differs",
        )
        require(
            record["contract_sha256"] == config_sha256
            and record["checkpoint_logical_sha256"] == checkpoint_sha256,
            "sample provenance differs",
        )
        sequence = record["sequence"]
        is_valid = (
            type(sequence) is str
            and 8 <= len(sequence) <= 50
            and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
        )
        valid += int(is_valid)
        require(
            is_valid
            and type(record["length"]) is int
            and record["length"] == len(sequence) == int(length),
            "sample sequence/length plan differs",
        )
        identity = canonical_sequence_id(sequence)
        require(record["sequence_id"] == identity, "sample sequence identity differs")
        identities.append(identity)
    training_ids = {row.sequence_id for row in distribution.rows}
    overlap = sum(identity in training_ids for identity in identities)
    return {
        "raw_sample_count": len(records),
        "valid_count": valid,
        "valid_rate": valid / len(records),
        "unique_valid_count": len(set(identities)),
        "unique_valid_rate": len(set(identities)) / len(records),
        "exact_training_overlap_count": overlap,
        "exact_training_overlap_rate": overlap / len(records),
    }


def verify_heldout(
    payload: bytes, rows: tuple[dict, ...], triple: str, model: NativeDenoiser
) -> dict[str, float]:
    records = [json.loads(line) for line in payload.splitlines()]
    expected_keys = [
        (fold, row["sequence_id"], level)
        for fold in range(5)
        if str(fold) not in triple
        for level in (8, 24, 40, 56)
        for row in rows
        if row["generator_fold"] == fold
    ]
    require(
        [(row["fold"], row["sequence_id"], row["level"]) for row in records] == expected_keys,
        "heldout record identities/count/order differ",
    )
    by_id = {row["sequence_id"]: row for row in rows}
    vocabulary = PeptideVocabulary("ACDEFGHIKLMNPQRSTVWY")
    diffusion = AbsorbingDiffusion(vocabulary, CosineMaskSchedule(0.008))
    model.eval()
    means = {}
    for fold in range(5):
        if str(fold) in triple:
            continue
        values = []
        for level in (8, 24, 40, 56):
            selected = [
                record for record in records if record["fold"] == fold and record["level"] == level
            ]
            for start in range(0, len(selected), 64):
                part = selected[start : start + 64]
                encoded = vocabulary.encode(
                    [by_id[record["sequence_id"]]["sequence"] for record in part], max_length=50
                )
                levels = np.full(len(part), level, dtype=np.int64)
                seeds = tuple(
                    namespaced_seed(640000, "holdout-corruption", record["sequence_id"], level)
                    for record in part
                )
                noisy, mask = diffusion.corrupt_fixed_count(
                    encoded.tokens, encoded.attention_mask, levels, total_levels=64, row_seeds=seeds
                )
                clean, tokens, selected_tokens, attention, timepoints = [
                    torch.from_numpy(value.copy())
                    for value in (encoded.tokens, noisy, mask, encoded.attention_mask, levels)
                ]
                with torch.inference_mode():
                    logits = model(
                        tokens, attention, timepoints, attention.sum(dim=1, dtype=torch.long)
                    )
                    objective = masked_token_objective(
                        logits, clean, selected_tokens, attention, corrupted_tokens=tokens
                    )
                for record, computed in zip(part, objective.row_losses.tolist(), strict=True):
                    require(
                        set(record) == {"fold", "sequence_id", "level", "loss"},
                        "heldout schema differs",
                    )
                    require(
                        math.isfinite(record["loss"])
                        and math.isclose(record["loss"], computed, rel_tol=1e-5, abs_tol=1e-5),
                        "CPU-reconstructed heldout loss differs",
                    )
                    values.append(record["loss"])
        means[str(fold)] = math.fsum(values) / len(values)
    return means


def verify_run(
    *,
    repository: Path,
    expected_commit: str,
    config_path: Path,
    config_sha256: str,
    run_root: Path,
    primary_array: str,
    duplicate_job: str,
) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repository), *args])

    require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit
        and not git("status", "--porcelain=v1", "--untracked-files=all"),
        "verifier requires clean frozen source",
    )
    contract = load_contract(config_path, config_sha256)
    rows = authenticate_generator_rows(contract)
    require(not run_root.is_symlink() and run_root.is_dir(), "run root differs")
    run_root = run_root.resolve()
    require(
        {entry.name for entry in run_root.iterdir()} == {f"{index:02d}" for index in range(11)},
        "run has extra/missing fit outputs",
    )
    paths = (
        git(
            "ls-files",
            "-z",
            "src/amp_challenge",
            "pyproject.toml",
            "uv.lock",
            "configs/diffusion/generator_namespace_checkpoints_v1.toml",
        )
        .decode()
        .split("\0")
    )
    expected_code = b"".join(
        f"{sha(git('show', expected_commit + ':' + path))}  {path}\n".encode()
        for path in sorted(paths)
        if path
    )
    manifests, snapshots = [], []
    for index in range(11):
        payloads = artifact_payloads(run_root / f"{index:02d}")
        manifest = json.loads(payloads["manifest.json"])
        require(
            manifest["fit_index"] == index and manifest["git_commit"] == expected_commit,
            "fit source/ordinal differs",
        )
        require(
            payloads["config.toml"] == contract.payload
            and payloads["CODE_SHA256SUMS"] == expected_code,
            "fit config/code bytes differ from frozen source",
        )
        manifests.append(manifest)
        snapshots.append(payloads)
    require(primary_array.isdigit() and duplicate_job.isdigit(), "scheduler input is not numeric")
    scheduler_raw = subprocess.check_output(
        [
            "/usr/bin/sacct",
            "-j",
            primary_array + "," + duplicate_job,
            "-X",
            "-n",
            "-P",
            "--format=" + ",".join(SACCT_FIELDS),
        ],
        env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
        text=True,
        timeout=60,
    )
    resources = verify_scheduler(
        parse_scheduler(scheduler_raw),
        manifests,
        primary_array=primary_array,
        duplicate_job=duplicate_job,
        expected_commit=expected_commit,
        config_sha256=config_sha256,
        run_root=run_root,
    )
    torch.set_num_threads(1)
    verified = []
    for index, (manifest, payloads) in enumerate(zip(manifests, snapshots, strict=True)):
        ordinal = 0 if index == 10 else index
        triple = TRIPLES[ordinal]
        seed = 530000 + int(triple)
        distribution = training_distribution(rows, triple)
        require(
            manifest["artifact"] == "generator_namespace_native_checkpoint_v1"
            and manifest["schema_version"] == 1
            and manifest["triple"] == triple
            and manifest["seed"] == seed
            and manifest["duplicate_of"] == (0 if index == 10 else None),
            "fit contract identity differs",
        )
        require(
            manifest["selected_generator_twin"] == (1 if index == 10 else 0),
            "generator twin differs",
        )
        for key, expected in {
            "config_sha256": contract.sha256,
            "namespace_receipt_sha256": contract.document["input"]["receipt_sha256"],
            "generator_corpus_sha256": contract.document["input"]["corpus_sha256"],
            "model_config": asdict(contract.model_config),
            "model_parameters": 354068,
            "final_step": 1000,
            "training_draw_count": 64000,
            "oracle_calls": 0,
            "production_eligible": False,
            "scientific_superiority_claim": False,
            "independent_verification_passed": False,
        }.items():
            require(
                manifest[key] == expected and type(manifest[key]) is type(expected),
                "fit metadata differs: " + key,
            )
        require(
            manifest["training_folds"] == list(map(int, triple))
            and manifest["training_sequence_ids"]
            == sorted(row.sequence_id for row in distribution.rows)
            and manifest["training_row_count"]
            == len(distribution.rows)
            == contract.document["fits"]["training_row_counts"][ordinal],
            "training fold projection differs",
        )
        require(
            manifest["training_union_component_ids"]
            == sorted(
                {row["union_component_id"] for row in rows if str(row["generator_fold"]) in triple}
            ),
            "training component identities differ",
        )
        require(
            payloads["training_projection.jsonl"] == projection_bytes(distribution),
            "training-only weighted projection differs",
        )
        consumed = verify_trace(payloads["training_trace.jsonl"], distribution, seed)
        require(
            manifest["consumed_sequence_ids"] == sorted(consumed), "consumed fold identities differ"
        )
        configure_deterministic_runtime(namespaced_seed(seed, "initialization", "model"))
        model = NativeDenoiser(contract.model_config)
        require(
            canonical_model_logical_hash(model) == manifest["initial_model_logical_sha256"],
            "initial seeded weights differ",
        )
        hashes = load_safetensors_checkpoint(
            model,
            run_root / f"{index:02d}" / "checkpoint.safetensors",
            expected_file_sha256=manifest["checkpoint_file_sha256"],
            expected_logical_state_sha256=manifest["checkpoint_logical_sha256"],
        )
        require(
            hashes.logical_state_sha256 != manifest["initial_model_logical_sha256"],
            "checkpoint weights unchanged",
        )
        metrics = json.loads(payloads["metrics.json"])
        require(
            metrics["diagnostic_only"] is True
            and metrics["checkpoint_selection_allowed"] is False
            and metrics["final_step"] == 1000
            and metrics["training_draw_count"] == 64000,
            "diagnostic scope differs",
        )
        require(
            metrics["final_training_loss"]
            == json.loads(payloads["training_trace.jsonl"].splitlines()[-1])["loss"],
            "final training loss differs",
        )
        samples = verify_samples(
            payloads["samples.jsonl"],
            distribution,
            triple=triple,
            config_sha256=config_sha256,
            checkpoint_sha256=hashes.logical_state_sha256,
        )
        require(
            all(metrics[key] == value for key, value in samples.items()),
            "independently reconstructed sample metrics differ",
        )
        heldout = verify_heldout(payloads["heldout_losses.jsonl"], rows, triple, model)
        require(metrics["heldout_fold_losses"] == heldout, "heldout aggregate differs")
        require(
            0 < metrics["worker_elapsed_seconds"] <= 720
            and 0 < metrics["peak_gpu_memory_reserved_bytes"] <= 12 * 1024**3,
            "worker diagnostic resource bound differs",
        )
        verified.append(
            {
                "fit_index": index,
                "triple": triple,
                "seed": seed,
                "job_id": manifest["job_id"],
                "node": manifest["node"],
                "manifest_sha256": sha(payloads["manifest.json"]),
                "checkpoint_file_sha256": hashes.file_sha256,
                "checkpoint_logical_sha256": hashes.logical_state_sha256,
                "training_rows": len(distribution.rows),
                "consumed_unique_ids": len(consumed),
                "heldout_rows": len(rows) - len(distribution.rows),
                "heldout_fold_losses": heldout,
                **samples,
            }
        )
        print(
            json.dumps(
                {
                    "verified_fit": index,
                    "triple": triple,
                    "raw_sample_count": samples["raw_sample_count"],
                }
            ),
            flush=True,
        )
    for name in (
        "training_projection.jsonl",
        "training_trace.jsonl",
        "checkpoint.safetensors",
        "samples.jsonl",
        "heldout_losses.jsonl",
    ):
        require(snapshots[0][name] == snapshots[10][name], "exact duplicate differs: " + name)
    for index, payloads in enumerate(snapshots):
        require(
            artifact_payloads(run_root / f"{index:02d}") == payloads,
            "artifacts changed during independent reconstruction",
        )
    require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit
        and not git("status", "--porcelain=v1", "--untracked-files=all"),
        "source changed during independent reconstruction",
    )
    return {
        "schema_version": 1,
        "artifact": "generator_namespace_checkpoint_verification_v1",
        "same_account_reconstruction": True,
        "independent_authoring_claim": False,
        "independent_execution_passed": True,
        "verification_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": expected_commit,
        "config_sha256": config_sha256,
        "namespace_receipt_sha256": contract.document["input"]["receipt_sha256"],
        "run_root": str(run_root),
        "fits": verified,
        "exact_duplicate_passed": True,
        "scheduler_stdout_sha256": sha(scheduler_raw.encode()),
        "resources": resources,
        "oracle_calls": 0,
        "scientific_superiority_claim": False,
        "production_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--primary-array-job-id", required=True)
    parser.add_argument("--duplicate-job-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.receipt.parent.is_dir() and not args.receipt.exists(),
        "receipt destination must be new with an existing parent",
    )
    result = verify_run(
        repository=args.repository,
        expected_commit=args.expected_git_commit,
        config_path=args.config,
        config_sha256=args.expected_config_sha256,
        run_root=args.run_root,
        primary_array=args.primary_array_job_id,
        duplicate_job=args.duplicate_job_id,
    )
    descriptor = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(result))
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    print(
        json.dumps(
            {"receipt": str(args.receipt), "receipt_sha256": sha(read_regular(args.receipt))}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

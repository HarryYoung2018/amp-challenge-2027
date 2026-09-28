"""Submit five shared-source jobs then 65 native arm jobs with fixed resources.

Manifest: artifact=native_screen_launch_v1, study_id, repository, source_commit,
interpreter/protocol/evaluator={path,sha256}, output_root, allocation_margin_seconds
(recommend 60), sources[5], runs[65]. Source rows: seed,run_id,provider={path,sha256},
output,seconds. Run rows additionally name arm_id and common_output. Outputs are
output_root/common/<seed> and output_root/runs/<run_id>. Recommended CLI seconds
are 840/7140 within hard 900/7200-second allocations; no additional GPU allowance.

Providers receive AMP_SCREEN_MANIFEST, AMP_SCREEN_MANIFEST_SHA256,
AMP_SCREEN_STAGE, AMP_SCREEN_SLOT, AMP_SCREEN_RUN_ID, AMP_SCREEN_COMMON_OUTPUT.
They must authenticate their declared slot and shared-source receipts themselves
(using load_common_initial_handoffs); manifest identity references grant no
scientific admission. Provider imports/provisioning stay inside existing CLI clocks.
An existing launch directory is never resubmitted, including unresolved intents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from amp_challenge.evaluation.evolutionary_kl_successor_protocol_v2 import (
    CONFIGURATION_IDS_V2,
    SCREEN_SEEDS_V2,
)

SELF = "src/amp_challenge/generators/search/native_screen_launcher.py"
SCRIPT = "cluster/slurm/native_screen_task.sh"
PREFIX = "amp_challenge.generators.search."
ROUTES = {
    "tuned_peptide_ga": "eligible_ga_campaign_cli",
    "categorical_diffusion_posthoc": "native_static_campaign_cli",
    "diffusion_reward_kl_no_search": "native_static_campaign_cli",
    "arcadiamp_style_iterative_d3pm": "native_static_campaign_cli",
    "mp2d_style_inference_search": "native_mp2d_campaign_cli",
    "ga_endpoint_distillation_no_kg": "ga_endpoint_campaign_cli",
    **{arm: "native_evolution_campaign_cli" for arm in CONFIGURATION_IDS_V2[7:]},
    "tr2d2_style_tree_offpolicy": "native_evolution_campaign_cli",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(
        type(value) is str and Path(value).is_absolute() and ".." not in Path(value).parts,
        "canonical absolute path required",
    )
    require(str(Path(value)) == value, "noncanonical path")
    return Path(value)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(reference):
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "pin schema differs")
    path = absolute(reference["path"])
    require(file_sha(path) == reference["sha256"], "pinned file bytes differ: " + str(path))
    return path


def write_once(path, value):
    payload = (
        value
        if type(value) is bytes
        else (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_manifest(path, expected_sha256, *, verify_providers=True):
    payload = absolute(str(path)).read_bytes()
    require(
        len(payload) <= 1024**2 and hashlib.sha256(payload).hexdigest() == expected_sha256,
        "manifest size/digest differs",
    )
    manifest = json.loads(payload)
    require(
        type(manifest) is dict
        and set(manifest)
        == {
            "artifact",
            "study_id",
            "repository",
            "source_commit",
            "interpreter",
            "protocol",
            "evaluator",
            "output_root",
            "allocation_margin_seconds",
            "sources",
            "runs",
        },
        "manifest fields differ",
    )
    require(
        manifest["artifact"] == "native_screen_launch_v1"
        and type(manifest["study_id"]) is str
        and re.fullmatch(r"[A-Za-z0-9_-]{1,48}", manifest["study_id"]) is not None,
        "study identity differs",
    )
    require(
        type(manifest["source_commit"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is not None,
        "full source commit required",
    )
    root, repository = absolute(manifest["output_root"]), absolute(manifest["repository"])
    require(root.resolve() == root and not root.is_relative_to(repository), "unsafe launch root")
    margin = manifest["allocation_margin_seconds"]
    require(type(margin) is int and 0 <= margin < 900, "allocation margin differs")
    for key in ("interpreter", "protocol", "evaluator"):
        pinned(manifest[key])
    require(
        type(manifest["sources"]) is list
        and len(manifest["sources"]) == 5
        and type(manifest["runs"]) is list
        and len(manifest["runs"]) == 65,
        "exact five source and 65 run rows required",
    )
    expected = [(None, seed) for seed in SCREEN_SEEDS_V2] + [
        (arm, seed) for arm in CONFIGURATION_IDS_V2 for seed in SCREEN_SEEDS_V2
    ]
    ids = set()
    for row, (arm, seed) in zip(manifest["sources"] + manifest["runs"], expected, strict=True):
        fields = {"seed", "run_id", "provider", "output", "seconds"}
        require(
            type(row) is dict
            and set(row) == fields | ({"arm_id", "common_output"} if arm else set()),
            "slot fields differ",
        )
        require(
            type(row["seed"]) is int
            and row["seed"] == seed
            and type(row["run_id"]) is str
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", row["run_id"]),
            "slot seed/run identity differs",
        )
        require(row["run_id"] not in ids, "duplicate run ID")
        ids.add(row["run_id"])
        if arm:
            require(
                row["arm_id"] == arm
                and row["run_id"] == f"screen.{arm}.seed-{seed}"
                and absolute(row["common_output"]) == root / "common" / str(seed),
                "arm/source slot differs",
            )
        output = root / "runs" / row["run_id"] if arm else root / "common" / str(seed)
        require(absolute(row["output"]) == output, "fixed output slot differs")
        seconds = row["seconds"]
        require(
            type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds == (7200 if arm else 900) - margin,
            "slot clock differs from matched allocation",
        )
        require(
            type(row["provider"]) is dict and set(row["provider"]) == {"path", "sha256"},
            "provider pin differs",
        )
        absolute(row["provider"]["path"])
        require(
            type(row["provider"]["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", row["provider"]["sha256"]),
            "provider digest differs",
        )
        if verify_providers:
            pinned(row["provider"])
    return manifest, payload


def check_source(manifest):
    repository = Path(manifest["repository"])

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])

    require(
        git("rev-parse", "HEAD").decode().strip() == manifest["source_commit"],
        "source HEAD differs",
    )
    require(
        not git("status", "--porcelain=v1", "--untracked-files=all"), "source checkout is not clean"
    )
    for name in (
        SELF,
        SCRIPT,
        "src/amp_challenge/evaluation/evolutionary_kl_successor_protocol_v2.py",
    ):
        require(
            git("show", manifest["source_commit"] + ":" + name) == (repository / name).read_bytes(),
            "frozen launcher bytes differ",
        )
    require(
        Path(__file__).read_bytes() == (repository / SELF).read_bytes(),
        "loaded launcher source differs",
    )


def submission_argv(manifest, stage, manifest_path, manifest_sha256, source_job=None):
    root, repository = Path(manifest["output_root"]), Path(manifest["repository"])
    command = [
        "sbatch",
        "--parsable",
        "--account=bio",
        "--partition=gpumid",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=32G",
        "--gres=gpu:1",
        "--no-requeue",
        "--export=NONE",
        "--array=" + ("0-4%4" if stage == "source" else "0-64%4"),
        "--time=" + ("00:15:00" if stage == "source" else "02:00:00"),
        "--job-name=amp-" + manifest["study_id"] + "-" + stage,
        "--output=" + str(root / "logs" / "%x-%A_%a.out"),
        "--error=" + str(root / "logs" / "%x-%A_%a.err"),
    ]
    if stage == "arm":
        require(type(source_job) is str and source_job.isdigit(), "source array job required")
        command += ["--dependency=afterok:" + source_job, "--kill-on-invalid-dep=yes"]
    return [
        *command,
        str(repository / SCRIPT),
        str(repository),
        manifest["interpreter"]["path"],
        manifest["interpreter"]["sha256"],
        file_sha(repository / SCRIPT),
        str(manifest_path),
        manifest_sha256,
        stage,
    ]


def submit(manifest_path, manifest_sha256):
    manifest, payload = load_manifest(manifest_path, manifest_sha256)
    check_source(manifest)
    root = Path(manifest["output_root"])
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_once(root / "manifest.json", payload)
    for name in ("common", "runs", "logs", "tasks"):
        (root / name).mkdir(mode=0o700)
    jobs = {}
    for stage in ("source", "arm"):
        command = submission_argv(
            manifest, stage, root / "manifest.json", manifest_sha256, jobs.get("source")
        )
        write_once(
            root / (stage + "-intent.json"), {"manifest_sha256": manifest_sha256, "argv": command}
        )
        # An exception after scheduler acceptance leaves this intent unresolved.
        # Never infer non-submission or retry; preserve any known source job ID.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        write_once(
            root / (stage + "-response.json"),
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
        require(
            result.returncode == 0
            and re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?\n?", result.stdout),
            "scheduler response failed or unresolved; do not resubmit",
        )
        jobs[stage] = result.stdout.strip().split(";")[0]
        write_once(
            root / (stage + "-job.json"),
            {"job_id": jobs[stage], "manifest_sha256": manifest_sha256},
        )
    return jobs


def dispatch(manifest_path, manifest_sha256, stage):
    manifest, _ = load_manifest(manifest_path, manifest_sha256, verify_providers=False)
    check_source(manifest)
    require(stage in ("source", "arm"), "task stage differs")
    require(
        os.environ.get("SLURM_JOB_ACCOUNT") == "bio"
        and os.environ.get("SLURM_JOB_PARTITION") == "gpumid"
        and os.environ.get("SLURM_CPUS_PER_TASK") == "8"
        and os.environ.get("SLURM_MEM_PER_NODE") == "32768",
        "task allocation differs",
    )
    slot = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    require(slot.isdigit(), "array slot required")
    rows = manifest["sources" if stage == "source" else "runs"]
    require(0 <= int(slot) < len(rows), "array slot out of range")
    row = rows[int(slot)]
    pinned(row["provider"])
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(gpu) and "," not in gpu, "one allocated GPU required")
    # CUDA ordinal zero can map to a different NVML index under Slurm cgroups.
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    require(bool(names) and all("A100" in name for name in names), "homogeneous A100 node required")
    output = Path(row["output"])
    require(not output.exists(), "task output exists; do not rerun")
    module = "common_initial_source_cli" if stage == "source" else ROUTES[row["arm_id"]]
    command = [
        manifest["interpreter"]["path"],
        "-m",
        PREFIX + module,
        "--provider",
        row["provider"]["path"],
        "--provider-sha256",
        row["provider"]["sha256"],
        "--output",
        str(output),
        "--seconds",
        str(row["seconds"]),
    ]
    write_once(
        Path(manifest["output_root"]) / "tasks" / f"{stage}-{slot}.json",
        {
            "manifest_sha256": manifest_sha256,
            "row": row,
            "argv": command,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        },
    )
    environment = dict(os.environ)
    environment.update(
        AMP_SCREEN_MANIFEST=str(manifest_path),
        AMP_SCREEN_MANIFEST_SHA256=manifest_sha256,
        AMP_SCREEN_STAGE=stage,
        AMP_SCREEN_SLOT=slot,
        AMP_SCREEN_RUN_ID=row["run_id"],
        AMP_SCREEN_COMMON_OUTPUT=row.get("common_output", row["output"]),
    )
    os.execve(command[0], command, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "task"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage", choices=("source", "arm"))
    args = parser.parse_args()
    if args.action == "submit":
        print(json.dumps(submit(args.manifest, args.manifest_sha256), sort_keys=True))
    else:
        dispatch(args.manifest, args.manifest_sha256, args.stage)


if __name__ == "__main__":
    main()

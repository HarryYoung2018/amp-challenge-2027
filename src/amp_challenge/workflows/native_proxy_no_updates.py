"""Run the explicit no-model-update control on the unchanged version-three inputs.

Launch only through cluster/submit.sh cpu. Thread settings deliberately match
the original CPU study and are installed before importing numerical packages.
The original arm identifier is retained for identical semantic random seeds;
intervention.json, not that identifier alone, declares this different treatment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def ensure_producer_source(repository, producer_commit):
    """Restore one exact checkpoint-source dependency, never substitute current code."""
    canonical = Path("/home/yonghan.yang/amp_challenge")
    command = ["git", "-C", str(repository), "cat-file", "-e", f"{producer_commit}^{{commit}}"]
    present = subprocess.run(command, check=False, stderr=subprocess.DEVNULL).returncode == 0
    if not present:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(canonical),
                producer_commit,
            ],
            check=True,
        )
        subprocess.run(command, check=True)
    return {
        "producer_commit": producer_commit,
        "source_repository": str(canonical),
        "restored_exact_git_object": not present,
        "checkpoint_source_checks_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/search/peptide_proxy_distance_v3.toml")
    )
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdigit()
        or os.environ.get("SLURM_JOB_PARTITION") != "standard"
        or os.environ.get("SLURM_JOB_ACCOUNT") != "bio"
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4
    ):
        raise RuntimeError("control requires the supported CPU runner with four allocated CPUs")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"

    from amp_challenge.generators.diffusion.native_initialization import PRODUCER_COMMIT
    from amp_challenge.workflows.native_proxy_study import execute

    if args.output.exists():
        raise ValueError("control output already exists; preserve previous attempts")
    dependency = ensure_producer_source(Path(__file__).resolve().parents[3], PRODUCER_COMMIT)
    try:
        result = execute(
            args.protocol,
            args.initial,
            args.output,
            seed=args.seed,
            arm_name="evolutionary_kl",
            max_seconds=7000,
            device="cpu",
            model_updates_disabled=True,
        )
    finally:
        if args.output.is_dir():
            with (args.output / "source-dependency.json").open("x") as stream:
                json.dump(dependency, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

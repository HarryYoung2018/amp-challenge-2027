"""Execute the recorded replay-teacher intervention on exact version-three inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from amp_challenge.workflows.native_proxy_no_updates import ensure_producer_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--arm",
        default="evolutionary_kl",
        choices=("evolutionary_kl", "evolutionary_wasserstein", "evolutionary_total_variation"),
    )
    parser.add_argument("--disable-model-updates", action="store_true")
    parser.add_argument("--precision-recheck", action="store_true")
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdigit()
        or os.environ.get("SLURM_JOB_PARTITION") != "standard"
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4
    ):
        raise RuntimeError("replay experiment requires the supported four-CPU runner")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    from amp_challenge.generators.diffusion.native_initialization import PRODUCER_COMMIT
    from amp_challenge.workflows.native_proxy_study import execute

    if args.output.exists():
        raise ValueError("preserve previous output; supply a new experiment directory")
    dependency = ensure_producer_source(Path(__file__).resolve().parents[3], PRODUCER_COMMIT)
    try:
        result = execute(
            Path("configs/search/peptide_proxy_distance_v3.toml"),
            args.initial,
            args.output,
            seed=args.seed,
            arm_name=args.arm,
            max_seconds=7000,
            device="cpu",
            model_updates_disabled=args.disable_model_updates,
            replay_teacher=True,
            precision_recheck=args.precision_recheck,
        )
    finally:
        if args.output.is_dir():
            with (args.output / "source-dependency.json").open("x") as stream:
                json.dump(dependency, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

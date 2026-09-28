"""Assemble a self-contained inference repository without publishing externally."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from amp_challenge.models.competition_oracle import CHECKPOINT
from amp_challenge.workflows.competition_train import BASE
from amp_challenge.workflows.peptide_proxy_initial import EXAMPLES


def package_submission(release, oracle, output):
    release, oracle, output = Path(release), Path(oracle), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[3]
    archive = output / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", "HEAD"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(output)], check=True)
    archive.unlink()
    bundle = output / "checkpoints/competition"
    for relative in ("campaign/result.json", "campaign/events.jsonl"):
        target = bundle / "evolutionary" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release / relative, target)
    shutil.copytree(
        release / "provider/native/policies", bundle / "evolutionary/provider/native/policies"
    )
    shutil.copytree(oracle, bundle / "oracle")
    (bundle / "embedding").mkdir()
    for name in (CHECKPOINT.name, "esm2_t6_8M_UR50D-contact-regression.pt"):
        shutil.copyfile(CHECKPOINT.parent / name, bundle / "embedding" / name)
    (bundle / "training").mkdir()
    shutil.copyfile(EXAMPLES, bundle / "training/oracle_examples.jsonl")
    for ordinal in range(10):
        shutil.copyfile(
            BASE / f"{ordinal:02d}/training_projection.jsonl",
            bundle / f"training/generator-{ordinal:02d}.jsonl",
        )
    manifest = {
        "release": str(release),
        "oracle": str(oracle),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "generation": "uv sync; uv run --no-sync generate",
        "hardware": "CUDA GPU by default; --device cpu supported but cross-device byte identity is not asserted",
        "publication": "local_package_only_not_uploaded_or_submitted",
        "training_data": "checkpoints/competition/training; original source and licensing documents retained under docs and data",
    }
    (output / "COMPETITION_PACKAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_submission(args.release, args.oracle, args.output)), flush=True)


if __name__ == "__main__":
    main()

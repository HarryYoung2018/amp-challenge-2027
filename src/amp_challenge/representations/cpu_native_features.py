"""Explicit CPU-only feature runtime identity; never an accepted CUDA receipt."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from amp_challenge.representations.peptide_esm import (
    CONFIG,
    CONTACT_SHA256,
    LOCK_SHA256,
    MODEL_SHA256,
    canonical_json,
    digest,
    file_digest,
)

SOURCE_FILES = (
    "src/amp_challenge/representations/cpu_native_features.py",
    "src/amp_challenge/representations/peptide_esm.py",
    "src/amp_challenge/representations/fixed_shape_esm.py",
    "src/amp_challenge/representations/laplacian.py",
    "src/amp_challenge/representations/candidate_features.py",
    "src/amp_challenge/representations/warm_candidate_features.py",
    "src/amp_challenge/representations/run_feature_cache_records.py",
    "integrations/ampdiffusion/esm2_cpu_proxy_worker.py",
    "integrations/ampdiffusion/esm2_warm_candidate_worker.py",
    "integrations/ampdiffusion/esm2_peptide_features_worker.py",
    "integrations/ampdiffusion/esm2_peptide_source_pins.json",
    "src/amp_challenge/workflows/cpu_proxy_features.py",
)


def cpu_allocation():
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdigit()
        or os.environ.get("SLURM_JOB_ACCOUNT") != "bio"
        or os.environ.get("SLURM_JOB_PARTITION") != "standard"
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4
    ):
        raise ValueError("CPU feature cohort requires bio/standard with at least four CPUs")
    return job


def cpu_source(repository, commit):
    repository = Path(repository)
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        != commit
    ):
        raise ValueError("CPU feature source commit changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository
    ):
        raise ValueError("CPU feature source must be clean and committed")
    return {
        "commit": commit,
        "files": {name: file_digest(repository / name) for name in SOURCE_FILES},
        "runtime_cohort": "cpu_float32_fixed128x52_not_cuda_equivalence_claim",
    }


def prepare_cpu_runtime(repository, bundle, private_esm):
    """Retain exact legacy packages/model/source pins; explicitly replace device contract."""
    cpu_allocation()
    expected = {"fair-esm": "2.0.0", "torch": "2.5.1+cu121", "numpy": "2.2.6"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if platform.python_version() != "3.10.19" or versions != expected:
        raise ValueError("CPU feature legacy Python/package pins differ")
    project = Path(bundle) / "source"
    if Path(sys.prefix).resolve() != (project / ".venv").resolve():
        raise ValueError("CPU feature worker must use the pinned legacy environment")
    checkpoint = Path(bundle) / "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
    pins = {
        str(checkpoint): MODEL_SHA256,
        str(checkpoint.with_name("esm2_t6_8M_UR50D-contact-regression.pt")): CONTACT_SHA256,
        str(project / "uv.lock"): LOCK_SHA256,
    }
    for path, expected_digest in pins.items():
        if file_digest(Path(path)) != expected_digest:
            raise ValueError("CPU feature checkpoint/contact/lock pin differs")
    original = project / ".venv/lib/python3.10/site-packages/esm"
    source_pins = json.loads(
        (Path(repository) / "integrations/ampdiffusion/esm2_peptide_source_pins.json").read_bytes()
    )
    actual = {
        str(path.relative_to(original)): file_digest(path)
        for path in sorted(original.rglob("*.py"))
    }
    if actual != source_pins:
        raise ValueError("CPU feature official ESM source inventory differs")
    private_esm.mkdir(mode=0o700)
    for relative, expected_digest in source_pins.items():
        destination = private_esm / "esm" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, destination)
        if file_digest(destination) != expected_digest:
            raise ValueError("CPU private official source copy differs")
    sys.path.insert(0, str(private_esm))
    import esm
    import numpy as np
    import torch

    if Path(esm.__file__).resolve() != private_esm / "esm/__init__.py":
        raise ValueError("CPU ESM import escaped private source")
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    runtime = {
        "device": "cpu",
        "dtype": "float32",
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_sha256": file_digest(Path(sys.executable)),
        "packages": versions,
        "checkpoint_assets": pins,
        "esm_source_sha256": actual,
        "esm_source_inventory_sha256": digest(canonical_json(actual)),
        "cpu_threads": 4,
        "node": platform.node(),
        "cpu_machine": platform.machine(),
        "torch_build": torch.__config__.show(),
        "cuda_numerical_equivalence_claimed": False,
    }
    return torch, esm, checkpoint, runtime


def recheck_cpu_runtime(bundle, private_esm, runtime):
    if runtime["device"] != "cpu" or runtime["cuda_numerical_equivalence_claimed"] is not False:
        raise ValueError("CPU runtime identity invalid")
    for name, expected in runtime["checkpoint_assets"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError("CPU checkpoint/lock changed")
    for root in (
        Path(bundle) / "source/.venv/lib/python3.10/site-packages/esm",
        private_esm / "esm",
    ):
        if {
            str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py"))
        } != runtime["esm_source_sha256"]:
            raise ValueError("CPU official ESM source changed")

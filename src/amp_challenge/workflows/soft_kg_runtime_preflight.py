"""Evidence-only dense soft-KG cluster timing preflight.

This synthetic benchmark freezes the largest currently proposed exhaustive
small-frontier shape.  It cannot authorize a production search configuration
or stand in for peptide-model, oracle, or end-to-end campaign measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from amp_challenge.acquisition.soft_kg import (
    EvaluationBatch,
    GaussianSoftKG,
    PreferenceMeasure,
    SoftKGProblem,
    UpperChanceConstraint,
)
from amp_challenge.models.posterior import JointGaussianPosterior

ARTIFACT: Final = "evolutionary_kl_dense_soft_kg_runtime_preflight_v1"
CONFIG_RELATIVE: Final = Path("configs/search/evolutionary_kl_soft_kg_runtime_preflight_v1.toml")
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_EXPECTED_PREFLIGHT: Final = {
    "profile": ARTIFACT,
    "seed": 420240908,
    "pool_size": 21,
    "decision_count": 21,
    "n_outputs": 2,
    "joint_size": 16,
    "n_fantasies": 512,
    "candidate_chunk_size": 256,
    "fantasy_chunk_size": 512,
    "max_combinations": 65536,
    "temperature": 0.25,
    "standard_error_multiplier": 1.0,
    "relative_eigenvalue_cutoff": 1e-10,
    "campaign_unique_call_budget": 512,
    "campaign_initial_design_unique_calls": 64,
    "campaign_adaptive_invocations": 28,
    "campaign_method_joint_q_cap": 14,
    "campaign_wall_budget_seconds": 7200,
    "kernel_wall_budget_fraction": 0.5,
    "kernel_wall_budget_seconds": 128.0,
    "kernel_peak_rss_budget_gib": 48,
    "requested_cpus": 4,
    "requested_memory_gib": 64,
    "measurement_repetitions": 1,
    "synthetic_only": True,
    "production_pins_modified": False,
}


@dataclass(frozen=True, slots=True)
class FrozenPreflightSpec:
    """Strictly decoded synthetic timing shape and resource gate."""

    profile: str
    seed: int
    pool_size: int
    decision_count: int
    n_outputs: int
    joint_size: int
    n_fantasies: int
    candidate_chunk_size: int
    fantasy_chunk_size: int
    max_combinations: int
    temperature: float
    standard_error_multiplier: float
    relative_eigenvalue_cutoff: float
    campaign_unique_call_budget: int
    campaign_initial_design_unique_calls: int
    campaign_adaptive_invocations: int
    campaign_method_joint_q_cap: int
    campaign_wall_budget_seconds: int
    kernel_wall_budget_fraction: float
    kernel_wall_budget_seconds: float
    kernel_peak_rss_budget_gib: int
    requested_cpus: int
    requested_memory_gib: int
    measurement_repetitions: int
    synthetic_only: bool
    production_pins_modified: bool

    @property
    def combination_count(self) -> int:
        return comb(self.pool_size, self.joint_size)

    @property
    def campaign_adaptive_call_count(self) -> int:
        return self.campaign_unique_call_budget - self.campaign_initial_design_unique_calls


def _load_frozen_spec(path: Path) -> tuple[FrozenPreflightSpec, str]:
    payload = path.read_bytes()
    document = tomllib.loads(payload.decode("utf-8"))
    if type(document) is not dict or set(document) != {"schema_version", "preflight"}:
        raise ValueError("runtime preflight config has an unexpected top-level schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("runtime preflight config schema_version must be exact integer one")
    values = document["preflight"]
    if type(values) is not dict or set(values) != set(_EXPECTED_PREFLIGHT):
        raise ValueError("runtime preflight config fields differ from the frozen schema")
    for key, expected in _EXPECTED_PREFLIGHT.items():
        value = values[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"runtime preflight field {key!r} differs from its frozen value")
    spec = FrozenPreflightSpec(**values)
    if spec.combination_count != 20_349:
        raise ValueError("runtime preflight combination count is not frozen at 20,349")
    if spec.combination_count > spec.max_combinations:
        raise ValueError("runtime preflight exceeds its exhaustive combination guard")
    if spec.campaign_adaptive_invocations != 28 or spec.campaign_adaptive_call_count != 448:
        raise ValueError("runtime preflight adaptive schedule is not frozen at 28 by 16 calls")
    if spec.campaign_adaptive_call_count != spec.campaign_adaptive_invocations * 16:
        raise ValueError("runtime preflight adaptive calls disagree with the protocol schedule")
    if spec.campaign_method_joint_q_cap != 14 or spec.joint_size != 16:
        raise ValueError("runtime preflight must distinguish method q=14 from cluster q=16")
    allocated_kernel_seconds = spec.kernel_wall_budget_seconds * spec.campaign_adaptive_invocations
    maximum_kernel_seconds = spec.campaign_wall_budget_seconds * spec.kernel_wall_budget_fraction
    if allocated_kernel_seconds > maximum_kernel_seconds:
        raise ValueError("runtime preflight kernel wall allocation exceeds its frozen fraction")
    return spec, hashlib.sha256(payload).hexdigest()


def _synthetic_belief(spec: FrozenPreflightSpec) -> JointGaussianPosterior:
    """Build the frozen, non-biological well-conditioned timing state."""

    if spec.decision_count != spec.pool_size or spec.n_outputs != 2:
        raise ValueError("synthetic timing state requires the frozen 21 by 2 shape")
    activity_mean = np.linspace(-0.75, 0.75, spec.decision_count, dtype=np.float64)
    constraint_mean = np.full(spec.decision_count, -0.75, dtype=np.float64)
    mean = np.column_stack((activity_mean, constraint_mean))
    flat_variance = np.tile(np.array([0.75, 0.04], dtype=np.float64), spec.decision_count)
    covariance = np.diag(flat_variance).reshape(
        spec.decision_count,
        spec.n_outputs,
        spec.decision_count,
        spec.n_outputs,
    )
    observation_noise = np.broadcast_to(
        np.diag(np.array([0.05, 0.01], dtype=np.float64)),
        (spec.decision_count, spec.n_outputs, spec.n_outputs),
    ).copy()
    return JointGaussianPosterior(mean, covariance, observation_noise)


def _synthetic_problem(spec: FrozenPreflightSpec) -> SoftKGProblem:
    decisions = tuple(range(spec.decision_count))
    return SoftKGProblem(
        decision_indices=decisions,
        objective_outputs=(0,),
        preferences=PreferenceMeasure(np.ones((1, 1), dtype=np.float64)),
        base_measure=np.ones(spec.decision_count, dtype=np.float64),
        constraints=(
            UpperChanceConstraint(
                output_index=1,
                upper_bound=0.0,
                max_violation_probability=0.05,
            ),
        ),
        always_safe_decisions=(0,),
    )


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    shape = ",".join(str(dimension) for dimension in canonical.shape).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"dtype=<f8;shape=")
    digest.update(shape)
    digest.update(b";bytes=")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _batch_inventory_sha256(pool_size: int, joint_size: int) -> str:
    digest = hashlib.sha256()
    for group in combinations(range(pool_size), joint_size):
        digest.update(",".join(str(index) for index in group).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _required_slurm_environment(spec: FrozenPreflightSpec) -> dict[str, object]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    node_list = os.environ.get("SLURM_JOB_NODELIST", "")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if not job_id.isdecimal() or int(job_id) <= 0 or not node_list:
        raise RuntimeError("runtime preflight must execute inside one Slurm allocation")
    if not cpus.isdecimal() or int(cpus) != spec.requested_cpus:
        raise RuntimeError("runtime preflight Slurm CPU allocation differs from the frozen spec")
    memory_mib = os.environ.get("SLURM_MEM_PER_NODE", "")
    if not memory_mib.isdecimal() or int(memory_mib) != spec.requested_memory_gib * 1024:
        raise RuntimeError("runtime preflight Slurm memory allocation differs from the frozen spec")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    account = os.environ.get("SLURM_JOB_ACCOUNT", "")
    if partition != "standard" or account != "bio":
        raise RuntimeError("runtime preflight must run on the frozen bio/standard allocation")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("runtime preflight peak-RSS units are frozen to Linux ru_maxrss KiB")
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "node_list": node_list,
        "partition": partition,
        "account": account,
        "cpus_per_task": int(cpus),
        "memory_per_node_mib": int(memory_mib),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
    }


def _write_exclusive(path: Path, document: dict[str, object]) -> None:
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def run_preflight(
    *,
    spec: FrozenPreflightSpec,
    config_sha256: str,
    expected_commit: str,
) -> dict[str, object]:
    """Measure one exact frozen synthetic kernel invocation."""

    if _COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be one lowercase 40-hex Git identity")
    slurm = _required_slurm_environment(spec)
    started = datetime.now(UTC).isoformat()
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    setup_wall_start = time.perf_counter()
    belief = _synthetic_belief(spec)
    problem = _synthetic_problem(spec)
    batch = EvaluationBatch(
        indices=tuple(range(spec.pool_size)),
        costs=np.ones(spec.pool_size, dtype=np.float64),
        eligible=np.ones(spec.pool_size, dtype=bool),
    )
    acquisition = GaussianSoftKG(
        problem,
        temperature=spec.temperature,
        observed_outputs=tuple(range(spec.n_outputs)),
        n_fantasies=spec.n_fantasies,
        standard_error_multiplier=spec.standard_error_multiplier,
        seed=spec.seed,
        relative_eigenvalue_cutoff=spec.relative_eigenvalue_cutoff,
        candidate_chunk_size=spec.candidate_chunk_size,
        fantasy_chunk_size=spec.fantasy_chunk_size,
    )
    setup_wall_seconds = time.perf_counter() - setup_wall_start

    kernel_cpu_start = time.process_time()
    kernel_wall_start = time.perf_counter()
    result = acquisition.select_joint(
        belief,
        batch,
        batch_size=spec.joint_size,
        max_combinations=spec.max_combinations,
    )
    kernel_wall_seconds = time.perf_counter() - kernel_wall_start
    kernel_cpu_seconds = time.process_time() - kernel_cpu_start
    total_wall_seconds = time.perf_counter() - total_wall_start
    total_cpu_seconds = time.process_time() - total_cpu_start

    if len(result.evaluation_batches) != spec.combination_count:
        raise RuntimeError("runtime preflight did not score the frozen exhaustive inventory")
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_budget_kib = spec.kernel_peak_rss_budget_gib * 1024 * 1024
    wall_pass = kernel_wall_seconds <= spec.kernel_wall_budget_seconds
    memory_pass = peak_rss_kib <= peak_rss_budget_kib
    output = {
        "batch_inventory_sha256": _batch_inventory_sha256(
            spec.pool_size,
            spec.joint_size,
        ),
        "estimate_sha256": _array_sha256(result.estimate),
        "score_sha256": _array_sha256(result.score),
        "selected_evaluation_indices": list(result.selected_evaluation_indices),
        "strictly_positive_score_count": int(np.count_nonzero(result.score > 0.0)),
    }
    return {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "completed",
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "git_commit": expected_commit,
        "config_sha256": config_sha256,
        "spec": {
            **asdict(spec),
            "combination_count": spec.combination_count,
            "campaign_adaptive_call_count": spec.campaign_adaptive_call_count,
            "kernel_total_wall_budget_seconds": (
                spec.kernel_wall_budget_seconds * spec.campaign_adaptive_invocations
            ),
            "flattened_decision_state": spec.decision_count * spec.n_outputs,
            "flattened_observation_state": spec.joint_size * spec.n_outputs,
        },
        "slurm": slurm,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "float64_itemsize": np.dtype(np.float64).itemsize,
            "longdouble_itemsize": np.dtype(np.longdouble).itemsize,
            "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
            "thread_controls": {
                key: os.environ.get(key, "")
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "timing_seconds": {
            "setup_wall": setup_wall_seconds,
            "kernel_wall": kernel_wall_seconds,
            "kernel_cpu": kernel_cpu_seconds,
            "total_wall": total_wall_seconds,
            "total_cpu": total_cpu_seconds,
        },
        "resources": {
            "peak_rss_kib_linux_ru_maxrss": peak_rss_kib,
            "peak_rss_budget_kib": peak_rss_budget_kib,
        },
        "synthetic_kernel_budget_gate": {
            "wall_pass": wall_pass,
            "memory_pass": memory_pass,
            "passed": wall_pass and memory_pass,
            "scope": "synthetic_dense_numeric_kernel_only",
            "production_feasibility_claim": "none",
        },
        "output": output,
        "limitations": [
            "synthetic_non_biological_diagonal_covariance",
            "single_node_single_invocation_timing",
            "no_peptide_model_or_oracle",
            "no_end_to_end_campaign_feasibility_claim",
            "no_production_pin_change",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve(strict=True)
    config_path = repo_root / CONFIG_RELATIVE
    spec, config_sha256 = _load_frozen_spec(config_path)
    document = run_preflight(
        spec=spec,
        config_sha256=config_sha256,
        expected_commit=args.expected_commit,
    )
    _write_exclusive(args.output, document)


if __name__ == "__main__":
    main()

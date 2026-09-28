"""Evidence-only bounded-beam soft-KG cluster timing preflight.

This synthetic benchmark freezes the research candidate's q14/pool20/F512
beam shape.  It cannot authorize production search or substitute for peptide,
oracle, or end-to-end scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import resource
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from amp_challenge.acquisition.soft_kg import (
    BeamJointSoftKGResult,
    EvaluationBatch,
    GaussianSoftKG,
    PreferenceMeasure,
    SoftKGProblem,
    UpperChanceConstraint,
)
from amp_challenge.models.posterior import JointGaussianPosterior
from amp_challenge.workflows.soft_kg_runtime_preflight import (
    _array_sha256,
    _required_slurm_environment,
    _write_exclusive,
)

ARTIFACT: Final = "evolutionary_kl_beam_soft_kg_runtime_preflight_v1"
CONFIG_RELATIVE: Final = Path(
    "configs/search/evolutionary_kl_soft_kg_beam_runtime_preflight_v1.toml"
)
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_EXPECTED_PREFLIGHT: Final = {
    "profile": ARTIFACT,
    "seed": 420240909,
    "pool_size": 20,
    "decision_count": 20,
    "n_outputs": 2,
    "joint_size": 14,
    "n_fantasies": 512,
    "candidate_chunk_size": 64,
    "fantasy_chunk_size": 512,
    "beam_width": 4,
    "max_groups_scored": 768,
    "temperature": 0.25,
    "standard_error_multiplier": 1.0,
    "relative_eigenvalue_cutoff": 1e-10,
    "campaign_adaptive_invocations": 28,
    "campaign_wall_budget_seconds": 7200,
    "kernel_wall_budget_fraction": 0.5,
    "kernel_wall_budget_seconds": 64.0,
    "kernel_peak_rss_budget_gib": 16,
    "requested_cpus": 4,
    "requested_memory_gib": 32,
    "measurement_repetitions": 1,
    "cross_candidate_covariance": True,
    "approximation_required": True,
    "expected_approximation_status": "beam_pruned",
    "dense_reference_evidence_status": "failed_no_go_unchanged",
    "synthetic_only": True,
    "production_pins_modified": False,
}


@dataclass(frozen=True, slots=True)
class FrozenBeamPreflightSpec:
    """Strictly decoded synthetic beam timing shape and resource gate."""

    profile: str
    seed: int
    pool_size: int
    decision_count: int
    n_outputs: int
    joint_size: int
    n_fantasies: int
    candidate_chunk_size: int
    fantasy_chunk_size: int
    beam_width: int
    max_groups_scored: int
    temperature: float
    standard_error_multiplier: float
    relative_eigenvalue_cutoff: float
    campaign_adaptive_invocations: int
    campaign_wall_budget_seconds: int
    kernel_wall_budget_fraction: float
    kernel_wall_budget_seconds: float
    kernel_peak_rss_budget_gib: int
    requested_cpus: int
    requested_memory_gib: int
    measurement_repetitions: int
    cross_candidate_covariance: bool
    approximation_required: bool
    expected_approximation_status: str
    dense_reference_evidence_status: str
    synthetic_only: bool
    production_pins_modified: bool

    @property
    def exhaustive_final_group_count(self) -> int:
        return comb(self.pool_size, self.joint_size)

    @property
    def worst_case_beam_groups_scored(self) -> int:
        first_depth = self.pool_size
        later_depths = sum(
            self.beam_width * (self.pool_size - depth + 1)
            for depth in range(2, self.joint_size + 1)
        )
        return first_depth + later_depths


def _load_frozen_beam_spec(path: Path) -> tuple[FrozenBeamPreflightSpec, str]:
    payload = path.read_bytes()
    document = tomllib.loads(payload.decode("utf-8"))
    if type(document) is not dict or set(document) != {"schema_version", "preflight"}:
        raise ValueError("beam runtime preflight config has an unexpected top-level schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("beam runtime preflight schema_version must be exact integer one")
    values = document["preflight"]
    if type(values) is not dict or set(values) != set(_EXPECTED_PREFLIGHT):
        raise ValueError("beam runtime preflight fields differ from the frozen schema")
    for key, expected in _EXPECTED_PREFLIGHT.items():
        value = values[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"beam runtime preflight field {key!r} differs from its frozen value")
    spec = FrozenBeamPreflightSpec(**values)
    if spec.exhaustive_final_group_count != 38_760:
        raise ValueError("beam runtime exhaustive reference count is not frozen at 38,760")
    if spec.worst_case_beam_groups_scored != 696:
        raise ValueError("beam runtime worst-case score bound is not frozen at 696")
    if spec.max_groups_scored < spec.worst_case_beam_groups_scored:
        raise ValueError("beam group cap cannot cover its proven worst-case expansion bound")
    allocated_kernel_seconds = spec.kernel_wall_budget_seconds * spec.campaign_adaptive_invocations
    maximum_kernel_seconds = spec.campaign_wall_budget_seconds * spec.kernel_wall_budget_fraction
    if allocated_kernel_seconds > maximum_kernel_seconds:
        raise ValueError("beam kernel wall allocation exceeds its frozen campaign fraction")
    return spec, hashlib.sha256(payload).hexdigest()


def _synthetic_beam_belief(spec: FrozenBeamPreflightSpec) -> JointGaussianPosterior:
    """Build a deterministic correlated, non-biological Gaussian state."""

    if spec.decision_count != spec.pool_size or spec.n_outputs != 2:
        raise ValueError(
            "synthetic beam timing state requires equal pool/decision size and 2 outputs"
        )
    locations = np.linspace(0.0, 1.0, spec.decision_count, dtype=np.float64)
    squared_distance = (locations[:, None] - locations[None, :]) ** 2
    point_covariance = np.exp(-squared_distance / 0.08) + 0.05 * np.eye(
        spec.decision_count,
        dtype=np.float64,
    )
    output_covariance = np.array([[0.75, 0.06], [0.06, 0.04]], dtype=np.float64)
    covariance = np.einsum(
        "ij,ab->iajb",
        point_covariance,
        output_covariance,
        optimize=False,
    )
    mean = np.column_stack(
        (
            np.linspace(-0.75, 0.75, spec.decision_count, dtype=np.float64),
            np.full(spec.decision_count, -0.75, dtype=np.float64),
        )
    )
    observation_noise = np.broadcast_to(
        np.diag(np.array([0.05, 0.01], dtype=np.float64)),
        (spec.decision_count, spec.n_outputs, spec.n_outputs),
    ).copy()
    return JointGaussianPosterior(mean, covariance, observation_noise)


def _synthetic_beam_problem(spec: FrozenBeamPreflightSpec) -> SoftKGProblem:
    decisions = tuple(range(spec.decision_count))
    return SoftKGProblem(
        decision_indices=decisions,
        objective_outputs=(0,),
        preferences=PreferenceMeasure(np.ones((1, 1), dtype=np.float64)),
        base_measure=np.ones(spec.decision_count, dtype=np.float64),
        constraints=(UpperChanceConstraint(1, 0.0, 0.05),),
        always_safe_decisions=(0,),
    )


def _group_inventory_sha256(groups: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(",".join(str(index) for index in group).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _depth_trace_document(result: BeamJointSoftKGResult) -> list[dict[str, object]]:
    depth_trace = result.depth_trace
    output: list[dict[str, object]] = []
    for trace in depth_trace:
        scored = trace.scored
        rows = [
            {
                "evaluation_indices": list(group),
                "estimate": float(scored.estimate[position]),
                "standard_error": float(scored.standard_error[position]),
                "standard_error_penalized_estimate": float(
                    scored.standard_error_penalized_estimate[position]
                ),
                "score": float(scored.score[position]),
                "total_cost": float(scored.total_cost[position]),
            }
            for position, group in enumerate(scored.evaluation_batches)
        ]
        output.append(
            {
                "depth": trace.depth,
                "generated_group_count": trace.generated_group_count,
                "completion_feasible_group_count": trace.completion_feasible_group_count,
                "scored_group_count": len(scored.evaluation_batches),
                "scored_group_inventory_sha256": _group_inventory_sha256(scored.evaluation_batches),
                "estimate_sha256": _array_sha256(scored.estimate),
                "score_sha256": _array_sha256(scored.score),
                "beam_pruned_group_count": trace.beam_pruned_group_count,
                "remaining_group_budget": trace.remaining_group_budget,
                "retained_evaluation_batches": [
                    list(group) for group in trace.retained_evaluation_batches
                ],
                "scored_groups": rows,
            }
        )
    return output


def run_beam_preflight(
    *,
    spec: FrozenBeamPreflightSpec,
    config_sha256: str,
    expected_commit: str,
) -> dict[str, object]:
    """Measure one exact frozen synthetic bounded-beam invocation."""

    if _COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be one lowercase 40-hex Git identity")
    slurm = _required_slurm_environment(spec)
    started = datetime.now(UTC).isoformat()
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    setup_wall_start = time.perf_counter()
    belief = _synthetic_beam_belief(spec)
    problem = _synthetic_beam_problem(spec)
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
    result = acquisition.select_joint_beam(
        belief,
        batch,
        batch_size=spec.joint_size,
        beam_width=spec.beam_width,
        max_groups_scored=spec.max_groups_scored,
    )
    kernel_wall_seconds = time.perf_counter() - kernel_wall_start
    kernel_cpu_seconds = time.process_time() - kernel_cpu_start
    total_wall_seconds = time.perf_counter() - total_wall_start
    total_cpu_seconds = time.process_time() - total_cpu_start

    if len(result.depth_trace) != spec.joint_size:
        raise RuntimeError("beam runtime preflight did not produce every frozen depth")
    if result.total_groups_scored > spec.worst_case_beam_groups_scored:
        raise RuntimeError("beam runtime preflight exceeded its proven expansion bound")
    if result.approximation_status != spec.expected_approximation_status:
        raise RuntimeError("beam runtime preflight approximation status changed")
    if any(len(group) != spec.joint_size for group in result.final_result.evaluation_batches):
        raise RuntimeError("beam runtime preflight final groups have the wrong size")

    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_budget_kib = spec.kernel_peak_rss_budget_gib * 1024 * 1024
    wall_pass = kernel_wall_seconds <= spec.kernel_wall_budget_seconds
    memory_pass = peak_rss_kib <= peak_rss_budget_kib
    trace_document = _depth_trace_document(result)
    covariance_sha256 = _array_sha256(
        belief.covariance.reshape(
            spec.decision_count * spec.n_outputs,
            spec.decision_count * spec.n_outputs,
        )
    )
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
            "exhaustive_final_group_count": spec.exhaustive_final_group_count,
            "worst_case_beam_groups_scored": spec.worst_case_beam_groups_scored,
            "kernel_total_wall_budget_seconds": (
                spec.kernel_wall_budget_seconds * spec.campaign_adaptive_invocations
            ),
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
        "synthetic_beam_kernel_budget_gate": {
            "wall_pass": wall_pass,
            "memory_pass": memory_pass,
            "trace_complete": True,
            "hard_group_cap_respected": result.total_groups_scored <= spec.max_groups_scored,
            "passed": wall_pass and memory_pass,
            "scope": "synthetic_correlated_bounded_beam_numeric_kernel_only",
            "production_feasibility_claim": "none",
        },
        "output": {
            "input_covariance_sha256": covariance_sha256,
            "approximation_status": result.approximation_status,
            "total_groups_scored": result.total_groups_scored,
            "selected_evaluation_indices": list(result.selected_evaluation_indices),
            "final_group_inventory_sha256": _group_inventory_sha256(
                result.final_result.evaluation_batches
            ),
            "final_estimate_sha256": _array_sha256(result.final_result.estimate),
            "final_score_sha256": _array_sha256(result.final_result.score),
            "depth_trace": trace_document,
        },
        "limitations": [
            "synthetic_non_biological_correlated_covariance",
            "single_node_single_invocation_timing",
            "bounded_beam_not_global_batch_optimum",
            "no_peptide_model_or_oracle",
            "no_end_to_end_campaign_feasibility_claim",
            "dense_exhaustive_no_go_evidence_unchanged",
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
    spec, config_sha256 = _load_frozen_beam_spec(repo_root / CONFIG_RELATIVE)
    document = run_beam_preflight(
        spec=spec,
        config_sha256=config_sha256,
        expected_commit=args.expected_commit,
    )
    _write_exclusive(args.output, document)


if __name__ == "__main__":
    main()

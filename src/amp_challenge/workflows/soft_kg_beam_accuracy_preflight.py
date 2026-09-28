"""Frozen synthetic accuracy gate for bounded-beam joint soft-KG.

This module compares the bounded selector with its exhaustive reference on a
balanced panel of deterministic, correlated small problems.  It is engineering
evidence only: it never reads peptide data, invokes an oracle, or changes a
production pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import time
import tomllib
from dataclasses import asdict, dataclass
from math import comb, sqrt
from pathlib import Path
from typing import Final

import numpy as np
import scipy
from numpy.typing import NDArray

from amp_challenge.acquisition.soft_kg import (
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

FloatArray = NDArray[np.float64]

ARTIFACT: Final = "evolutionary_kl_beam_soft_kg_accuracy_preflight_v1"
CONFIG_RELATIVE: Final = Path(
    "configs/search/evolutionary_kl_soft_kg_beam_accuracy_preflight_v1.toml"
)
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_EXPECTED_PREFLIGHT: Final = {
    "profile": ARTIFACT,
    "master_seed": 420240910,
    "bootstrap_seed": 20260908,
    "case_count": 40,
    "pool_size": 10,
    "decision_count": 10,
    "n_outputs": 2,
    "joint_size": 5,
    "n_fantasies": 512,
    "candidate_chunk_size": 64,
    "fantasy_chunk_size": 512,
    "beam_width": 4,
    "max_groups_scored": 160,
    "max_exact_combinations": 252,
    "temperature": 0.25,
    "standard_error_multiplier": 1.0,
    "relative_eigenvalue_cutoff": 1e-10,
    "constraint_upper_bound": 0.25,
    "constraint_max_violation_probability": 0.10,
    "bootstrap_replicates": 10_000,
    "minimum_positive_exact_cases": 24,
    "minimum_positive_exact_cases_per_regime": 4,
    "minimum_exact_match_fraction": 0.35,
    "minimum_optimum_reached_fraction": 0.50,
    "minimum_mean_score_capture": 0.90,
    "minimum_median_score_capture": 0.95,
    "minimum_p10_score_capture": 0.65,
    "minimum_worst_score_capture": 0.20,
    "minimum_regime_mean_score_capture": 0.80,
    "kernel_wall_budget_seconds": 300.0,
    "kernel_peak_rss_budget_gib": 16,
    "requested_cpus": 4,
    "requested_memory_gib": 32,
    "regimes": [
        "rbf_dense",
        "block_dense",
        "latent_rank_deficient",
        "mixed_sign_rank_deficient",
    ],
    "cost_modes": ["unit", "heterogeneous_capped"],
    "synthetic_only": True,
    "production_pins_modified": False,
    "scientific_claim": "none",
}


@dataclass(frozen=True, slots=True)
class FrozenBeamAccuracySpec:
    """Strictly decoded approximation panel and acceptance thresholds."""

    profile: str
    master_seed: int
    bootstrap_seed: int
    case_count: int
    pool_size: int
    decision_count: int
    n_outputs: int
    joint_size: int
    n_fantasies: int
    candidate_chunk_size: int
    fantasy_chunk_size: int
    beam_width: int
    max_groups_scored: int
    max_exact_combinations: int
    temperature: float
    standard_error_multiplier: float
    relative_eigenvalue_cutoff: float
    constraint_upper_bound: float
    constraint_max_violation_probability: float
    bootstrap_replicates: int
    minimum_positive_exact_cases: int
    minimum_positive_exact_cases_per_regime: int
    minimum_exact_match_fraction: float
    minimum_optimum_reached_fraction: float
    minimum_mean_score_capture: float
    minimum_median_score_capture: float
    minimum_p10_score_capture: float
    minimum_worst_score_capture: float
    minimum_regime_mean_score_capture: float
    kernel_wall_budget_seconds: float
    kernel_peak_rss_budget_gib: int
    requested_cpus: int
    requested_memory_gib: int
    regimes: tuple[str, ...]
    cost_modes: tuple[str, ...]
    synthetic_only: bool
    production_pins_modified: bool
    scientific_claim: str

    @property
    def exact_combination_count(self) -> int:
        return comb(self.pool_size, self.joint_size)

    @property
    def worst_case_beam_groups_scored(self) -> int:
        return self.pool_size + sum(
            self.beam_width * (self.pool_size - depth + 1)
            for depth in range(2, self.joint_size + 1)
        )


@dataclass(frozen=True, slots=True)
class SyntheticAccuracyCase:
    """One deterministic input shared by exhaustive and bounded selectors."""

    case_index: int
    case_id: str
    case_seed: int
    regime: str
    cost_mode: str
    belief: JointGaussianPosterior
    problem: SoftKGProblem
    batch: EvaluationBatch
    max_total_cost: float | None


def _load_frozen_accuracy_spec(path: Path) -> tuple[FrozenBeamAccuracySpec, str]:
    payload = path.read_bytes()
    document = tomllib.loads(payload.decode("utf-8"))
    if type(document) is not dict or set(document) != {"schema_version", "preflight"}:
        raise ValueError("beam accuracy config has an unexpected top-level schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("beam accuracy schema_version must be exact integer one")
    values = document["preflight"]
    if type(values) is not dict or set(values) != set(_EXPECTED_PREFLIGHT):
        raise ValueError("beam accuracy fields differ from the frozen schema")
    for key, expected in _EXPECTED_PREFLIGHT.items():
        value = values[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"beam accuracy field {key!r} differs from its frozen value")
    spec = FrozenBeamAccuracySpec(
        **{
            **values,
            "regimes": tuple(values["regimes"]),
            "cost_modes": tuple(values["cost_modes"]),
        }
    )
    factorial_size = len(spec.regimes) * len(spec.cost_modes)
    if spec.case_count % factorial_size != 0:
        raise ValueError("case_count must balance every regime-by-cost cell")
    if spec.decision_count != spec.pool_size or spec.n_outputs != 2:
        raise ValueError("frozen accuracy panel requires ten two-output decisions")
    if spec.exact_combination_count != spec.max_exact_combinations:
        raise ValueError("exact combination guard must cover exactly the frozen inventory")
    if spec.worst_case_beam_groups_scored > spec.max_groups_scored:
        raise ValueError("beam score cap cannot cover every generated depth")
    if not spec.synthetic_only or spec.production_pins_modified or spec.scientific_claim != "none":
        raise ValueError("beam accuracy preflight may carry no scientific or production claim")
    return spec, hashlib.sha256(payload).hexdigest()


def _case_seed(spec: FrozenBeamAccuracySpec, case_index: int) -> int:
    if case_index < 0 or case_index >= spec.case_count:
        raise ValueError("case_index is outside the frozen panel")
    return spec.master_seed + 1_000_003 * case_index


def _dense_kronecker_covariance(
    rng: np.random.Generator,
    *,
    regime: str,
    point_count: int,
) -> FloatArray:
    if regime == "rbf_dense":
        locations = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
        locations += rng.uniform(-0.015, 0.015, size=point_count)
        distances = locations[:, None] - locations[None, :]
        point_covariance = np.exp(-(distances**2) / 0.075)
        point_covariance += 0.035 * np.eye(point_count, dtype=np.float64)
        output_covariance = np.array([[0.70, 0.16], [0.16, 0.20]], dtype=np.float64)
    elif regime == "block_dense":
        blocks = np.arange(point_count) % 3
        point_covariance = 0.08 * np.ones((point_count, point_count), dtype=np.float64)
        point_covariance += 0.72 * (blocks[:, None] == blocks[None, :])
        point_covariance += 0.20 * np.eye(point_count, dtype=np.float64)
        output_covariance = np.array([[0.64, -0.17], [-0.17, 0.25]], dtype=np.float64)
    else:
        raise ValueError("unknown dense covariance regime")
    return np.asarray(np.kron(point_covariance, output_covariance), dtype=np.float64)


def _rank_deficient_covariance(
    rng: np.random.Generator,
    *,
    regime: str,
    point_count: int,
) -> FloatArray:
    flat_size = point_count * 2
    rank = 5 if regime == "latent_rank_deficient" else 4
    factors = rng.normal(size=(flat_size, rank))
    point_locations = np.linspace(-1.0, 1.0, point_count, dtype=np.float64)
    if regime == "latent_rank_deficient":
        for point in range(point_count):
            factors[2 * point, 0] += 1.6 * point_locations[point]
            factors[2 * point + 1, 1] += 1.2 * point_locations[point]
            factors[2 * point + 1, 0] += 0.55 * factors[2 * point, 0]
    elif regime == "mixed_sign_rank_deficient":
        signs = np.where(np.arange(point_count) % 2 == 0, 1.0, -1.0)
        for point in range(point_count):
            factors[2 * point, 0] += 1.4 * signs[point]
            factors[2 * point + 1, 0] -= 0.9 * signs[point]
            factors[2 * point + 1, 2] += point_locations[point]
    else:
        raise ValueError("unknown rank-deficient covariance regime")
    row_scales = np.tile(np.array([0.72, 0.34], dtype=np.float64), point_count)
    scaled = factors * row_scales[:, None] / sqrt(float(rank))
    return np.asarray(scaled @ scaled.T, dtype=np.float64)


def _build_case(spec: FrozenBeamAccuracySpec, case_index: int) -> SyntheticAccuracyCase:
    """Construct one content-deterministic correlated Gaussian case."""

    cell_count = len(spec.regimes) * len(spec.cost_modes)
    cell = case_index % cell_count
    regime = spec.regimes[cell % len(spec.regimes)]
    cost_mode = spec.cost_modes[cell // len(spec.regimes)]
    case_seed = _case_seed(spec, case_index)
    rng = np.random.Generator(np.random.PCG64(case_seed))

    if regime.endswith("dense"):
        flat_covariance = _dense_kronecker_covariance(
            rng,
            regime=regime,
            point_count=spec.decision_count,
        )
        observation_noise = np.broadcast_to(
            np.array([[0.030, 0.003], [0.003, 0.015]], dtype=np.float64),
            (spec.decision_count, 2, 2),
        ).copy()
    else:
        flat_covariance = _rank_deficient_covariance(
            rng,
            regime=regime,
            point_count=spec.decision_count,
        )
        observation_noise = np.zeros((spec.decision_count, 2, 2), dtype=np.float64)

    locations = np.linspace(-1.0, 1.0, spec.decision_count, dtype=np.float64)
    mean = np.column_stack(
        (
            0.55 * np.sin(np.pi * locations) + rng.normal(0.0, 0.16, spec.decision_count),
            -0.16 + 0.16 * np.cos(np.pi * locations) + rng.normal(0.0, 0.07, spec.decision_count),
        )
    )
    covariance = flat_covariance.reshape(
        spec.decision_count,
        spec.n_outputs,
        spec.decision_count,
        spec.n_outputs,
    )
    belief = JointGaussianPosterior(mean, covariance, observation_noise)
    problem = SoftKGProblem(
        decision_indices=tuple(range(spec.decision_count)),
        objective_outputs=(0,),
        preferences=PreferenceMeasure(np.ones((1, 1), dtype=np.float64)),
        base_measure=np.ones(spec.decision_count, dtype=np.float64),
        constraints=(
            UpperChanceConstraint(
                1,
                spec.constraint_upper_bound,
                spec.constraint_max_violation_probability,
            ),
        ),
        always_safe_decisions=(0,),
    )
    if cost_mode == "unit":
        costs = np.ones(spec.pool_size, dtype=np.float64)
        max_total_cost = None
    elif cost_mode == "heterogeneous_capped":
        costs = np.asarray(0.5 + 1.5 * rng.random(spec.pool_size), dtype=np.float64)
        cheapest = np.sort(costs)[: spec.joint_size]
        max_total_cost = float(np.sum(cheapest, dtype=np.float64) * 1.35)
    else:
        raise ValueError("unknown frozen cost mode")
    batch = EvaluationBatch(
        indices=tuple(range(spec.pool_size)),
        costs=costs,
        eligible=np.ones(spec.pool_size, dtype=bool),
    )
    case_identity = f"{ARTIFACT}:{case_index}:{case_seed}:{regime}:{cost_mode}".encode("ascii")
    return SyntheticAccuracyCase(
        case_index=case_index,
        case_id=hashlib.sha256(case_identity).hexdigest(),
        case_seed=case_seed,
        regime=regime,
        cost_mode=cost_mode,
        belief=belief,
        problem=problem,
        batch=batch,
        max_total_cost=max_total_cost,
    )


def _case_input_hashes(case: SyntheticAccuracyCase) -> dict[str, str]:
    belief = case.belief
    return {
        "mean_sha256": _array_sha256(belief.mean),
        "covariance_sha256": _array_sha256(
            belief.covariance.reshape(
                belief.n_points * belief.n_outputs,
                belief.n_points * belief.n_outputs,
            )
        ),
        "observation_noise_sha256": _array_sha256(belief.observation_noise),
        "costs_sha256": _array_sha256(case.batch.costs),
    }


def _evaluate_case(spec: FrozenBeamAccuracySpec, case_index: int) -> dict[str, object]:
    case = _build_case(spec, case_index)
    acquisition = GaussianSoftKG(
        case.problem,
        temperature=spec.temperature,
        observed_outputs=tuple(range(spec.n_outputs)),
        n_fantasies=spec.n_fantasies,
        standard_error_multiplier=spec.standard_error_multiplier,
        seed=case.case_seed,
        relative_eigenvalue_cutoff=spec.relative_eigenvalue_cutoff,
        candidate_chunk_size=spec.candidate_chunk_size,
        fantasy_chunk_size=spec.fantasy_chunk_size,
    )
    exact = acquisition.select_joint(
        case.belief,
        case.batch,
        batch_size=spec.joint_size,
        max_total_cost=case.max_total_cost,
        max_combinations=spec.max_exact_combinations,
    )
    beam = acquisition.select_joint_beam(
        case.belief,
        case.batch,
        batch_size=spec.joint_size,
        beam_width=spec.beam_width,
        max_total_cost=case.max_total_cost,
        max_groups_scored=spec.max_groups_scored,
    )

    exact_selected = exact.selected_evaluation_indices
    beam_selected = beam.selected_evaluation_indices
    exact_score_by_group = {
        group: float(exact.score[position])
        for position, group in enumerate(exact.evaluation_batches)
    }
    exact_best_score = 0.0 if not exact_selected else exact_score_by_group[exact_selected]
    beam_selected_exact_score = 0.0 if not beam_selected else exact_score_by_group[beam_selected]
    positive_exact = exact_best_score > 0.0
    score_capture = beam_selected_exact_score / exact_best_score if positive_exact else 1.0
    if not np.isfinite(score_capture) or score_capture < 0.0 or score_capture > 1.0 + 2e-14:
        raise RuntimeError("beam score capture falls outside the exhaustive reference")
    score_capture = min(1.0, float(score_capture))
    final_groups = beam.final_result.evaluation_batches
    exact_rank_by_group = {
        group: rank + 1
        for rank, group in enumerate(
            sorted(
                exact.evaluation_batches,
                key=lambda group: (-exact_score_by_group[group], group),
            )
        )
    }
    return {
        "case_index": case.case_index,
        "case_id": case.case_id,
        "case_seed": case.case_seed,
        "regime": case.regime,
        "cost_mode": case.cost_mode,
        "max_total_cost": case.max_total_cost,
        "input_sha256": _case_input_hashes(case),
        "exact_feasible_group_count": len(exact.evaluation_batches),
        "beam_total_groups_scored": beam.total_groups_scored,
        "beam_final_group_count": len(final_groups),
        "beam_approximation_status": beam.approximation_status,
        "exact_selected_evaluation_indices": list(exact_selected),
        "beam_selected_evaluation_indices": list(beam_selected),
        "exact_best_score": exact_best_score,
        "beam_selected_exact_score": beam_selected_exact_score,
        "beam_selected_exact_rank": (
            None if not beam_selected else exact_rank_by_group[beam_selected]
        ),
        "positive_exact_optimum": positive_exact,
        "exact_selection_match": beam_selected == exact_selected,
        "exact_optimum_reached_by_final_frontier": (
            beam_selected == exact_selected
            if not exact_selected
            else exact_selected in final_groups
        ),
        "score_capture": score_capture,
        "relative_score_regret": 1.0 - score_capture,
        "exact_group_inventory_sha256": _group_inventory_sha256(exact.evaluation_batches),
        "exact_score_sha256": _array_sha256(exact.score),
        "beam_final_group_inventory_sha256": _group_inventory_sha256(final_groups),
        "beam_final_score_sha256": _array_sha256(beam.final_result.score),
    }


def _group_inventory_sha256(groups: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(",".join(str(index) for index in group).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires valid binomial counts")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _summarize_cases(
    spec: FrozenBeamAccuracySpec,
    cases: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, bool]]:
    positive = [row for row in cases if row["positive_exact_optimum"] is True]
    positive_count = len(positive)
    if positive_count == 0:
        captures = np.empty(0, dtype=np.float64)
        matches = 0
        reached = 0
    else:
        captures = np.asarray([row["score_capture"] for row in positive], dtype=np.float64)
        matches = sum(row["exact_selection_match"] is True for row in positive)
        reached = sum(row["exact_optimum_reached_by_final_frontier"] is True for row in positive)

    positive_by_regime = {
        regime: [row for row in positive if row["regime"] == regime] for regime in spec.regimes
    }
    regime_summary = {
        regime: {
            "positive_exact_cases": len(rows),
            "mean_score_capture": (
                None if not rows else float(np.mean([float(row["score_capture"]) for row in rows]))
            ),
        }
        for regime, rows in positive_by_regime.items()
    }
    cost_summary = {
        mode: {
            "positive_exact_cases": len(rows),
            "mean_score_capture": (
                None if not rows else float(np.mean([float(row["score_capture"]) for row in rows]))
            ),
        }
        for mode in spec.cost_modes
        for rows in [[row for row in positive if row["cost_mode"] == mode]]
    }

    if positive_count:
        match_interval = _wilson_interval(matches, positive_count)
        reached_interval = _wilson_interval(reached, positive_count)
        bootstrap_rng = np.random.Generator(np.random.PCG64(spec.bootstrap_seed))
        indices = bootstrap_rng.integers(
            0,
            positive_count,
            size=(spec.bootstrap_replicates, positive_count),
        )
        bootstrap_means = np.mean(captures[indices], axis=1)
        bootstrap_interval = np.quantile(
            bootstrap_means,
            [0.025, 0.975],
            method="linear",
        )
        mean_capture = float(np.mean(captures))
        median_capture = float(np.quantile(captures, 0.5, method="linear"))
        p10_capture = float(np.quantile(captures, 0.1, method="linear"))
        worst_capture = float(np.min(captures))
        match_fraction = matches / positive_count
        reached_fraction = reached / positive_count
    else:
        match_interval = (0.0, 0.0)
        reached_interval = (0.0, 0.0)
        bootstrap_interval = np.array([0.0, 0.0], dtype=np.float64)
        mean_capture = median_capture = p10_capture = worst_capture = 0.0
        match_fraction = reached_fraction = 0.0

    checks = {
        "minimum_positive_exact_cases": positive_count >= spec.minimum_positive_exact_cases,
        "minimum_positive_exact_cases_per_regime": all(
            len(rows) >= spec.minimum_positive_exact_cases_per_regime
            for rows in positive_by_regime.values()
        ),
        "minimum_exact_match_fraction": match_fraction >= spec.minimum_exact_match_fraction,
        "minimum_optimum_reached_fraction": reached_fraction
        >= spec.minimum_optimum_reached_fraction,
        "minimum_mean_score_capture": mean_capture >= spec.minimum_mean_score_capture,
        "minimum_median_score_capture": median_capture >= spec.minimum_median_score_capture,
        "minimum_p10_score_capture": p10_capture >= spec.minimum_p10_score_capture,
        "minimum_worst_score_capture": worst_capture >= spec.minimum_worst_score_capture,
        "minimum_regime_mean_score_capture": all(
            rows
            and float(regime_summary[regime]["mean_score_capture"])
            >= spec.minimum_regime_mean_score_capture
            for regime, rows in positive_by_regime.items()
        ),
    }
    summary: dict[str, object] = {
        "case_count": len(cases),
        "positive_exact_cases": positive_count,
        "no_action_exact_cases": len(cases) - positive_count,
        "positive_case_exact_selection_matches": matches,
        "positive_case_exact_match_fraction": match_fraction,
        "positive_case_exact_match_wilson_95": list(match_interval),
        "positive_case_optimum_reached": reached,
        "positive_case_optimum_reached_fraction": reached_fraction,
        "positive_case_optimum_reached_wilson_95": list(reached_interval),
        "positive_case_mean_score_capture": mean_capture,
        "positive_case_mean_score_capture_bootstrap_type7_95": [
            float(bootstrap_interval[0]),
            float(bootstrap_interval[1]),
        ],
        "positive_case_median_score_capture": median_capture,
        "positive_case_p10_score_capture": p10_capture,
        "positive_case_worst_score_capture": worst_capture,
        "by_regime": regime_summary,
        "by_cost_mode": cost_summary,
    }
    return summary, checks


def _case_inventory_sha256(cases: list[dict[str, object]]) -> str:
    identity_rows = [
        {
            "case_id": row["case_id"],
            "case_index": row["case_index"],
            "case_seed": row["case_seed"],
            "cost_mode": row["cost_mode"],
            "input_sha256": row["input_sha256"],
            "max_total_cost": row["max_total_cost"],
            "regime": row["regime"],
        }
        for row in cases
    ]
    payload = json.dumps(
        identity_rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def run_accuracy_preflight(
    *,
    spec: FrozenBeamAccuracySpec,
    config_sha256: str,
    expected_commit: str,
) -> dict[str, object]:
    """Run the exact frozen comparison panel and construct its receipt."""

    if _COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be one lowercase 40-hex Git identity")
    slurm = _required_slurm_environment(spec)
    started = time.perf_counter()
    started_cpu = time.process_time()
    rows = [_evaluate_case(spec, case_index) for case_index in range(spec.case_count)]
    kernel_wall_seconds = time.perf_counter() - started
    kernel_cpu_seconds = time.process_time() - started_cpu
    summary, quality_checks = _summarize_cases(spec, rows)
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    resource_checks = {
        "kernel_wall_budget": kernel_wall_seconds <= spec.kernel_wall_budget_seconds,
        "kernel_peak_rss_budget": peak_rss_kib <= spec.kernel_peak_rss_budget_gib * 1024 * 1024,
    }
    all_checks = {**quality_checks, **resource_checks}
    return {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "accepted" if all(all_checks.values()) else "no_go",
        "git_commit": expected_commit,
        "config_sha256": config_sha256,
        "spec": {
            **asdict(spec),
            "exact_combination_count": spec.exact_combination_count,
            "worst_case_beam_groups_scored": spec.worst_case_beam_groups_scored,
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
            "kernel_wall": kernel_wall_seconds,
            "kernel_cpu": kernel_cpu_seconds,
        },
        "resources": {
            "peak_rss_kib_linux_ru_maxrss": peak_rss_kib,
            "peak_rss_budget_kib": spec.kernel_peak_rss_budget_gib * 1024 * 1024,
        },
        "case_inventory_sha256": _case_inventory_sha256(rows),
        "cases": rows,
        "summary": summary,
        "gate": {
            "checks": all_checks,
            "passed": all(all_checks.values()),
            "scope": "synthetic_small_problem_exact_vs_bounded_beam_only",
            "scientific_or_production_claim": "none",
        },
        "limitations": [
            "synthetic_non_biological_gaussian_panel",
            "small_pool_ten_batch_five_not_campaign_q14",
            "beam_width_four_only",
            "fixed_monte_carlo_fantasies",
            "no_peptide_model_or_oracle",
            "no_scientific_performance_claim",
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
    spec, config_sha256 = _load_frozen_accuracy_spec(repo_root / CONFIG_RELATIVE)
    document = run_accuracy_preflight(
        spec=spec,
        config_sha256=config_sha256,
        expected_commit=args.expected_commit,
    )
    _write_exclusive(args.output, document)


if __name__ == "__main__":
    main()

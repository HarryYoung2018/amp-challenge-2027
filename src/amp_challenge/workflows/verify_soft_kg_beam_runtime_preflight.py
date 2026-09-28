"""Stably replay one bounded-beam soft-KG timing producer receipt."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import tomllib
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from amp_challenge.acquisition.soft_kg import (
    BeamJointSoftKGResult,
    EvaluationBatch,
    GaussianSoftKG,
)
from amp_challenge.workflows.soft_kg_beam_runtime_preflight import (
    _EXPECTED_PREFLIGHT,
    ARTIFACT,
    CONFIG_RELATIVE,
    FrozenBeamPreflightSpec,
    _array_sha256,
    _group_inventory_sha256,
    _synthetic_beam_belief,
    _synthetic_beam_problem,
)
from amp_challenge.workflows.soft_kg_runtime_preflight import (
    _required_slurm_environment,
    _write_exclusive,
)
from amp_challenge.workflows.verify_soft_kg_beam_accuracy_preflight import (
    _capture_exact_artifact_inventory,
    _capture_path,
    _load_canonical_json_bytes,
    _parse_manifest_bytes,
    _revalidate_closed_snapshot_set,
    _snapshot_set_sha256,
    _verify_audit_checkout,
)

AUDIT_ARTIFACT: Final = "evolutionary_kl_beam_soft_kg_runtime_preflight_audit_v1"
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_INDEPENDENT_RELATIVE_TOLERANCE: Final = 5e-13
_INDEPENDENT_ABSOLUTE_TOLERANCE: Final = 5e-15
_PRODUCER_BASE: Final = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/"
    "evolutionary-kl-soft-kg-beam-runtime-preflight-v1"
)
_PRODUCER_REPO_ROOT: Final = Path("/home/yonghan.yang/amp_evo_softkg_optimizer_agent")
_PRODUCER_SOURCE_PATHS: Final = (
    CONFIG_RELATIVE,
    Path("src/amp_challenge/acquisition/soft_kg.py"),
    Path("src/amp_challenge/models/posterior.py"),
    Path("src/amp_challenge/workflows/soft_kg_beam_runtime_preflight.py"),
    Path("cluster/slurm/run_evolutionary_kl_soft_kg_beam_runtime_preflight.sbatch"),
)
_REPLAY_CORE_SOURCE_PATHS: Final = (
    Path("src/amp_challenge/acquisition/soft_kg.py"),
    Path("src/amp_challenge/models/posterior.py"),
    Path("src/amp_challenge/workflows/soft_kg_beam_runtime_preflight.py"),
)
_AUDIT_REPLAY_SOURCE_PATHS: Final = (
    *_REPLAY_CORE_SOURCE_PATHS,
    Path("src/amp_challenge/workflows/soft_kg_runtime_preflight.py"),
    Path("src/amp_challenge/workflows/verify_soft_kg_beam_accuracy_preflight.py"),
    Path("src/amp_challenge/workflows/verify_soft_kg_beam_runtime_preflight.py"),
    Path("cluster/slurm/audit_evolutionary_kl_soft_kg_beam_runtime_preflight.sbatch"),
)
_EXPECTED_TOP_LEVEL_KEYS: Final = {
    "artifact",
    "config_sha256",
    "environment",
    "finished_utc",
    "git_commit",
    "limitations",
    "output",
    "resources",
    "schema_version",
    "slurm",
    "spec",
    "started_utc",
    "status",
    "synthetic_beam_kernel_budget_gate",
    "timing_seconds",
}
_EXPECTED_OUTPUT_KEYS: Final = {
    "approximation_status",
    "depth_trace",
    "final_estimate_sha256",
    "final_group_inventory_sha256",
    "final_score_sha256",
    "input_covariance_sha256",
    "selected_evaluation_indices",
    "total_groups_scored",
}
_EXPECTED_DEPTH_KEYS: Final = {
    "beam_pruned_group_count",
    "completion_feasible_group_count",
    "depth",
    "estimate_sha256",
    "generated_group_count",
    "remaining_group_budget",
    "retained_evaluation_batches",
    "score_sha256",
    "scored_group_count",
    "scored_group_inventory_sha256",
    "scored_groups",
}
_EXPECTED_SCORED_GROUP_KEYS: Final = {
    "estimate",
    "evaluation_indices",
    "score",
    "standard_error",
    "standard_error_penalized_estimate",
    "total_cost",
}
_EXPECTED_ENVIRONMENT_KEYS: Final = {
    "float64_itemsize",
    "longdouble_itemsize",
    "longdouble_mantissa_bits",
    "numpy",
    "platform",
    "python",
    "scipy",
    "thread_controls",
}
_EXPECTED_THREAD_CONTROLS: Final = {
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
}
_EXPECTED_ENVIRONMENT: Final = {
    "float64_itemsize": 8,
    "longdouble_itemsize": 16,
    "longdouble_mantissa_bits": 63,
    "numpy": "2.4.6",
    "platform": "Linux-6.8.0-90-generic-x86_64-with-glibc2.39",
    "python": "3.11.14",
    "scipy": "1.16.3",
    "thread_controls": _EXPECTED_THREAD_CONTROLS,
}
_EXPECTED_LIMITATIONS: Final = [
    "synthetic_non_biological_correlated_covariance",
    "single_node_single_invocation_timing",
    "bounded_beam_not_global_batch_optimum",
    "no_peptide_model_or_oracle",
    "no_end_to_end_campaign_feasibility_claim",
    "dense_exhaustive_no_go_evidence_unchanged",
    "no_production_pin_change",
]


def _sha256(path: Path) -> str:
    return _capture_path(path, label=str(path)).sha256


def _load_canonical_json(path: Path) -> dict[str, object]:
    snapshot = _capture_path(path, label="producer receipt")
    return _load_canonical_json_bytes(snapshot.payload)


def _exact_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _exact_object(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} differs from the exact frozen schema")
    return value


def _strict_mapping_equal(
    observed: dict[str, object],
    expected: dict[str, object],
    *,
    name: str,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{name} differs from the exact frozen schema")
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if type(observed_value) is not type(expected_value) or observed_value != expected_value:
            raise ValueError(f"{name} field {key!r} differs from its frozen value")


def _integer_group(value: object, *, depth: int, name: str) -> tuple[int, ...]:
    if type(value) is not list or len(value) != depth:
        raise ValueError(f"{name} must be a depth-{depth} JSON list")
    group = tuple(_exact_integer(index, name=f"{name} index") for index in value)
    if group != tuple(sorted(set(group))) or any(index < 0 or index >= 20 for index in group):
        raise ValueError(f"{name} must be canonical, unique, and inside the frozen pool")
    return group


def _parse_manifest(path: Path) -> dict[Path, str]:
    snapshot = _capture_path(path, label="producer SHA256SUMS")
    return _parse_manifest_bytes(snapshot.payload)


def _verify_recorded_spec(document: dict[str, object], spec: FrozenBeamPreflightSpec) -> None:
    recorded = document.get("spec")
    field_names = tuple(field.name for field in fields(spec))
    expected_keys = {
        *field_names,
        "exhaustive_final_group_count",
        "worst_case_beam_groups_scored",
        "kernel_total_wall_budget_seconds",
    }
    if type(recorded) is not dict or set(recorded) != expected_keys:
        raise ValueError("producer recorded spec differs from the exact frozen schema")
    for name in field_names:
        expected = getattr(spec, name)
        if type(recorded[name]) is not type(expected) or recorded[name] != expected:
            raise ValueError(f"producer recorded spec field {name!r} differs from config")
    expected_derived: dict[str, object] = {
        "exhaustive_final_group_count": spec.exhaustive_final_group_count,
        "worst_case_beam_groups_scored": spec.worst_case_beam_groups_scored,
        "kernel_total_wall_budget_seconds": (
            spec.kernel_wall_budget_seconds * spec.campaign_adaptive_invocations
        ),
    }
    for name, expected in expected_derived.items():
        if type(recorded[name]) is not type(expected) or recorded[name] != expected:
            raise ValueError(f"producer derived spec field {name!r} does not reconstruct")


def _verify_git_bound_sources(
    repo_root: Path,
    commit: str,
    manifest: dict[Path, str],
) -> dict[Path, bytes]:
    source_blobs: dict[Path, bytes] = {}
    for relative_path in _PRODUCER_SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path.as_posix()}"],
            check=True,
            capture_output=True,
        )
        commit_digest = hashlib.sha256(completed.stdout).hexdigest()
        if commit_digest != manifest[_PRODUCER_REPO_ROOT / relative_path]:
            raise ValueError(f"producer manifest does not match commit blob {relative_path}")
        source_blobs[relative_path] = completed.stdout
    return source_blobs


def _decode_frozen_beam_spec(payload: bytes) -> tuple[FrozenBeamPreflightSpec, str]:
    """Decode exact producer-commit config bytes without reopening a checkout path."""

    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("beam runtime preflight config is not valid UTF-8 TOML") from error
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


def _source_blob_inventory_sha256(
    source_blobs: dict[Path, bytes],
    *,
    paths: tuple[Path, ...] = _PRODUCER_SOURCE_PATHS,
) -> str:
    digest = hashlib.sha256()
    for relative_path in paths:
        payload = source_blobs[relative_path]
        digest.update(relative_path.as_posix().encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _capture_audit_replay_sources(
    repo_root: Path,
    audit_commit: str,
) -> dict[Path, bytes]:
    source_blobs: dict[Path, bytes] = {}
    for relative_path in _AUDIT_REPLAY_SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{audit_commit}:{relative_path.as_posix()}"],
            check=True,
            capture_output=True,
        )
        snapshot = _capture_path(
            repo_root / relative_path,
            label=f"audit replay source {relative_path}",
        )
        if snapshot.payload != completed.stdout:
            raise ValueError(f"audit replay source differs from commit blob {relative_path}")
        source_blobs[relative_path] = snapshot.payload
    return source_blobs


def _source_sha256_by_path(
    source_blobs: dict[Path, bytes],
    *,
    paths: tuple[Path, ...],
) -> dict[str, str]:
    return {
        relative_path.as_posix(): hashlib.sha256(source_blobs[relative_path]).hexdigest()
        for relative_path in paths
    }


def _verify_depth_trace(
    trace_document: object,
    *,
    pool_size: int,
    joint_size: int,
    beam_width: int,
    max_groups_scored: int,
) -> tuple[int, tuple[int, ...]]:
    if type(trace_document) is not list or len(trace_document) != joint_size:
        raise ValueError("producer depth trace must contain every frozen depth")
    candidates = tuple(range(pool_size))
    previous_frontier: tuple[tuple[int, ...], ...] = ((),)
    total_scored = 0
    final_rows: list[tuple[tuple[int, ...], float]] = []

    for expected_depth, trace in enumerate(trace_document, start=1):
        trace = _exact_object(
            trace,
            keys=_EXPECTED_DEPTH_KEYS,
            name="producer depth trace row",
        )
        depth = _exact_integer(trace.get("depth"), name="trace depth")
        if depth != expected_depth:
            raise ValueError("producer depth trace is not consecutive")
        expected_groups = tuple(
            sorted(
                {
                    tuple(sorted((*group, candidate)))
                    for group in previous_frontier
                    for candidate in candidates
                    if candidate not in group
                }
            )
        )
        generated_count = _exact_integer(
            trace.get("generated_group_count"),
            name="generated_group_count",
        )
        feasible_count = _exact_integer(
            trace.get("completion_feasible_group_count"),
            name="completion_feasible_group_count",
        )
        scored_count = _exact_integer(
            trace.get("scored_group_count"),
            name="scored_group_count",
        )
        if generated_count != len(expected_groups) or feasible_count != len(expected_groups):
            raise ValueError("producer trace expansion counts do not reconstruct")
        if scored_count != len(expected_groups):
            raise ValueError("producer trace silently screened a generated depth")
        rows = trace.get("scored_groups")
        if type(rows) is not list or len(rows) != scored_count:
            raise ValueError("producer full scored-group trace is incomplete")

        groups: list[tuple[int, ...]] = []
        estimates: list[float] = []
        errors: list[float] = []
        penalized: list[float] = []
        scores: list[float] = []
        costs: list[float] = []
        for row_position, row in enumerate(rows):
            row = _exact_object(
                row,
                keys=_EXPECTED_SCORED_GROUP_KEYS,
                name="producer scored-group row",
            )
            group = _integer_group(
                row.get("evaluation_indices"),
                depth=depth,
                name="scored group",
            )
            if group != expected_groups[row_position]:
                raise ValueError("producer scored-group inventory is not complete canonical order")
            estimate = _finite_float(row.get("estimate"), name="estimate")
            error = _finite_float(row.get("standard_error"), name="standard_error")
            penalty = _finite_float(
                row.get("standard_error_penalized_estimate"),
                name="standard_error_penalized_estimate",
            )
            score = _finite_float(row.get("score"), name="score")
            cost = _finite_float(row.get("total_cost"), name="total_cost")
            if error < 0.0 or score < 0.0 or cost != float(depth):
                raise ValueError("producer score moments or frozen unit costs are invalid")
            if not np.isclose(penalty, estimate - error, rtol=4e-15, atol=0.0):
                raise ValueError("producer penalized estimate does not reconstruct")
            if not np.isclose(score, max(penalty, 0.0) / cost, rtol=4e-15, atol=0.0):
                raise ValueError("producer cost-normalized score does not reconstruct")
            groups.append(group)
            estimates.append(estimate)
            errors.append(error)
            penalized.append(penalty)
            scores.append(score)
            costs.append(cost)

        canonical_groups = tuple(groups)
        if trace.get("scored_group_inventory_sha256") != _group_inventory_sha256(canonical_groups):
            raise ValueError("producer scored-group inventory digest does not reconstruct")
        if trace.get("estimate_sha256") != _array_sha256(np.asarray(estimates)):
            raise ValueError("producer estimate digest does not reconstruct")
        if trace.get("score_sha256") != _array_sha256(np.asarray(scores)):
            raise ValueError("producer score digest does not reconstruct")

        ranked_positions = sorted(
            range(scored_count),
            key=lambda position: (-scores[position], canonical_groups[position]),
        )
        retained = tuple(
            _integer_group(group, depth=depth, name="retained group")
            for group in trace.get("retained_evaluation_batches", [])
        )
        expected_retained = tuple(
            canonical_groups[position] for position in ranked_positions[:beam_width]
        )
        if retained != expected_retained:
            raise ValueError("producer retained frontier does not reconstruct from scores")
        expected_pruned = max(0, scored_count - len(expected_retained)) if depth < joint_size else 0
        if (
            _exact_integer(
                trace.get("beam_pruned_group_count"),
                name="beam_pruned_group_count",
            )
            != expected_pruned
        ):
            raise ValueError("producer beam-pruning count does not reconstruct")
        total_scored += scored_count
        if (
            _exact_integer(
                trace.get("remaining_group_budget"),
                name="remaining_group_budget",
            )
            != max_groups_scored - total_scored
        ):
            raise ValueError("producer remaining group budget does not reconstruct")
        previous_frontier = retained
        final_rows = list(zip(canonical_groups, scores, strict=True))

    positive_final = [row for row in final_rows if row[1] > 0.0]
    selected = (
        ()
        if not positive_final
        else sorted(positive_final, key=lambda row: (-row[1], row[0]))[0][0]
    )
    return total_scored, selected


def _recompute_frozen_kernel(spec: FrozenBeamPreflightSpec) -> BeamJointSoftKGResult:
    belief = _synthetic_beam_belief(spec)
    acquisition = GaussianSoftKG(
        _synthetic_beam_problem(spec),
        temperature=spec.temperature,
        observed_outputs=tuple(range(spec.n_outputs)),
        n_fantasies=spec.n_fantasies,
        standard_error_multiplier=spec.standard_error_multiplier,
        seed=spec.seed,
        relative_eigenvalue_cutoff=spec.relative_eigenvalue_cutoff,
        candidate_chunk_size=spec.candidate_chunk_size,
        fantasy_chunk_size=spec.fantasy_chunk_size,
    )
    return acquisition.select_joint_beam(
        belief,
        EvaluationBatch(
            indices=tuple(range(spec.pool_size)),
            costs=np.ones(spec.pool_size, dtype=np.float64),
            eligible=np.ones(spec.pool_size, dtype=bool),
        ),
        batch_size=spec.joint_size,
        beam_width=spec.beam_width,
        max_groups_scored=spec.max_groups_scored,
    )


def _replay_numeric_comparison(
    trace_document: list[object],
    replay: BeamJointSoftKGResult,
) -> dict[str, object]:
    field_pairs = (
        ("estimate", "estimate"),
        ("standard_error", "standard_error"),
        ("standard_error_penalized_estimate", "standard_error_penalized_estimate"),
        ("score", "score"),
        ("total_cost", "total_cost"),
    )
    maximum_absolute_error = {receipt_field: 0.0 for receipt_field, _attr in field_pairs}
    maximum_scaled_relative_error = {receipt_field: 0.0 for receipt_field, _attr in field_pairs}
    replay_depth_hashes: list[dict[str, object]] = []

    for depth_position, replay_trace in enumerate(replay.depth_trace):
        raw_trace = trace_document[depth_position]
        if type(raw_trace) is not dict:
            raise ValueError("producer trace rows must remain objects during recomputation")
        raw_rows = raw_trace.get("scored_groups")
        if type(raw_rows) is not list:
            raise ValueError("producer scored-group rows are missing during recomputation")
        replay_scored = replay_trace.scored
        receipt_groups = tuple(
            _integer_group(
                row.get("evaluation_indices"),
                depth=replay_trace.depth,
                name="replayed group",
            )
            for row in raw_rows
        )
        if receipt_groups != replay_scored.evaluation_batches:
            raise ValueError("replayed kernel group inventory differs from the producer")
        if (
            replay_trace.generated_group_count
            != _exact_integer(raw_trace.get("generated_group_count"), name="generated count")
            or replay_trace.completion_feasible_group_count
            != _exact_integer(
                raw_trace.get("completion_feasible_group_count"), name="feasible count"
            )
            or replay_trace.beam_pruned_group_count
            != _exact_integer(raw_trace.get("beam_pruned_group_count"), name="pruned count")
            or replay_trace.remaining_group_budget
            != _exact_integer(raw_trace.get("remaining_group_budget"), name="remaining budget")
        ):
            raise ValueError("replayed kernel depth counters differ from the producer")
        receipt_retained = tuple(
            _integer_group(
                group,
                depth=replay_trace.depth,
                name="replayed retained group",
            )
            for group in raw_trace.get("retained_evaluation_batches", [])
        )
        if receipt_retained != replay_trace.retained_evaluation_batches:
            raise ValueError("replayed kernel retained frontier differs from the producer")

        for receipt_field, attribute in field_pairs:
            receipt_values = np.asarray(
                [_finite_float(row.get(receipt_field), name=receipt_field) for row in raw_rows],
                dtype=np.float64,
            )
            replay_values = np.asarray(
                getattr(replay_scored, attribute),
                dtype=np.float64,
            )
            differences = np.abs(receipt_values - replay_values)
            denominators = np.maximum(
                np.abs(replay_values),
                _INDEPENDENT_ABSOLUTE_TOLERANCE,
            )
            maximum_absolute_error[receipt_field] = max(
                maximum_absolute_error[receipt_field],
                float(np.max(differences, initial=0.0)),
            )
            maximum_scaled_relative_error[receipt_field] = max(
                maximum_scaled_relative_error[receipt_field],
                float(np.max(differences / denominators, initial=0.0)),
            )
            if not np.allclose(
                receipt_values,
                replay_values,
                rtol=_INDEPENDENT_RELATIVE_TOLERANCE,
                atol=_INDEPENDENT_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    f"replayed kernel {receipt_field} differs beyond the declared tolerance"
                )
        replay_depth_hashes.append(
            {
                "depth": replay_trace.depth,
                "group_inventory_sha256": _group_inventory_sha256(replay_scored.evaluation_batches),
                "estimate_sha256": _array_sha256(replay_scored.estimate),
                "standard_error_sha256": _array_sha256(replay_scored.standard_error),
                "score_sha256": _array_sha256(replay_scored.score),
            }
        )

    return {
        "relative_tolerance": _INDEPENDENT_RELATIVE_TOLERANCE,
        "absolute_tolerance": _INDEPENDENT_ABSOLUTE_TOLERANCE,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_scaled_relative_error": maximum_scaled_relative_error,
        "replay_depth_hashes": replay_depth_hashes,
    }


def _producer_accounting(job_id: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,JobName,State,ExitCode,NodeList,AllocCPUS,ReqMem,"
            "Account,Partition,ElapsedRaw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [line.split("|") for line in completed.stdout.splitlines() if line]
    parent = [record for record in records if record[0] == str(job_id)]
    if len(parent) != 1 or len(parent[0]) != 10:
        raise ValueError("Slurm accounting did not return one canonical producer job")
    record = parent[0]
    if (
        record[1] != "evo-softkg-beam-q14"
        or record[2] != "COMPLETED"
        or record[3] != "0:0"
        or not record[4]
        or record[5] != "4"
        or record[6] not in {"32G", "32768M"}
        or record[7] != "bio"
        or record[8] != "standard"
        or not record[9].isdecimal()
        or int(record[9]) <= 0
    ):
        raise ValueError("Slurm accounting does not authenticate the frozen producer allocation")
    return {
        "job_name": record[1],
        "state": record[2],
        "exit_code": record[3],
        "node_list": record[4],
        "allocated_cpus": int(record[5]),
        "requested_memory": record[6],
        "account": record[7],
        "partition": record[8],
        "elapsed_seconds": int(record[9]),
    }


def _utc_timestamp(value: object, *, name: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"{name} must carry an explicit UTC offset")
    return parsed


def _verify_producer_metadata(
    document: dict[str, object],
    *,
    spec: FrozenBeamPreflightSpec,
    expected_producer_job: int,
    accounting: dict[str, object],
) -> dict[str, object]:
    _exact_object(document, keys=_EXPECTED_TOP_LEVEL_KEYS, name="producer receipt")
    started = _utc_timestamp(document["started_utc"], name="producer started_utc")
    finished = _utc_timestamp(document["finished_utc"], name="producer finished_utc")
    if finished < started:
        raise ValueError("producer finished_utc precedes started_utc")

    limitations = document["limitations"]
    if type(limitations) is not list or limitations != _EXPECTED_LIMITATIONS:
        raise ValueError("producer limitations differ from the exact frozen list")

    environment = _exact_object(
        document["environment"],
        keys=_EXPECTED_ENVIRONMENT_KEYS,
        name="producer environment",
    )
    thread_controls = _exact_object(
        environment["thread_controls"],
        keys=set(_EXPECTED_THREAD_CONTROLS),
        name="producer thread controls",
    )
    _strict_mapping_equal(
        thread_controls,
        _EXPECTED_THREAD_CONTROLS,
        name="producer thread controls",
    )
    _strict_mapping_equal(
        environment,
        _EXPECTED_ENVIRONMENT,
        name="producer environment",
    )

    producer_slurm = _exact_object(
        document["slurm"],
        keys={
            "account",
            "cpu_affinity_count",
            "cpus_per_task",
            "job_id",
            "job_name",
            "memory_per_node_mib",
            "node_list",
            "partition",
        },
        name="producer Slurm receipt",
    )
    expected_slurm: dict[str, object] = {
        "account": "bio",
        "cpu_affinity_count": spec.requested_cpus,
        "cpus_per_task": spec.requested_cpus,
        "job_id": str(expected_producer_job),
        "job_name": "evo-softkg-beam-q14",
        "memory_per_node_mib": spec.requested_memory_gib * 1024,
        "node_list": accounting["node_list"],
        "partition": "standard",
    }
    _strict_mapping_equal(
        producer_slurm,
        expected_slurm,
        name="producer Slurm receipt",
    )
    return producer_slurm


def _verify_timing_resources_and_gate(
    document: dict[str, object],
    *,
    spec: FrozenBeamPreflightSpec,
    total_scored: int,
) -> tuple[float, int, bool, bool]:
    timing = _exact_object(
        document["timing_seconds"],
        keys={"kernel_cpu", "kernel_wall", "setup_wall", "total_cpu", "total_wall"},
        name="producer timing",
    )
    values = {key: _finite_float(value, name=key) for key, value in timing.items()}
    if (
        values["setup_wall"] < 0.0
        or values["kernel_wall"] <= 0.0
        or values["kernel_cpu"] <= 0.0
        or values["total_wall"] <= 0.0
        or values["total_cpu"] <= 0.0
        or values["total_wall"] < values["setup_wall"] + values["kernel_wall"]
        or values["total_cpu"] < values["kernel_cpu"]
    ):
        raise ValueError("producer timing values are non-positive or internally inconsistent")

    resources = _exact_object(
        document["resources"],
        keys={"peak_rss_budget_kib", "peak_rss_kib_linux_ru_maxrss"},
        name="producer resources",
    )
    peak_rss = _exact_integer(
        resources["peak_rss_kib_linux_ru_maxrss"],
        name="peak_rss_kib_linux_ru_maxrss",
    )
    peak_rss_budget = _exact_integer(
        resources["peak_rss_budget_kib"],
        name="peak_rss_budget_kib",
    )
    expected_rss_budget = spec.kernel_peak_rss_budget_gib * 1024 * 1024
    if peak_rss <= 0 or peak_rss_budget != expected_rss_budget:
        raise ValueError("producer peak RSS or its frozen budget is invalid")

    kernel_wall = values["kernel_wall"]
    wall_pass = kernel_wall <= spec.kernel_wall_budget_seconds
    memory_pass = peak_rss <= expected_rss_budget
    gate = _exact_object(
        document["synthetic_beam_kernel_budget_gate"],
        keys={
            "hard_group_cap_respected",
            "memory_pass",
            "passed",
            "production_feasibility_claim",
            "scope",
            "trace_complete",
            "wall_pass",
        },
        name="producer resource gate",
    )
    expected_gate: dict[str, object] = {
        "hard_group_cap_respected": total_scored <= spec.max_groups_scored,
        "memory_pass": memory_pass,
        "passed": wall_pass and memory_pass,
        "production_feasibility_claim": "none",
        "scope": "synthetic_correlated_bounded_beam_numeric_kernel_only",
        "trace_complete": True,
        "wall_pass": wall_pass,
    }
    _strict_mapping_equal(gate, expected_gate, name="producer resource gate")
    return kernel_wall, peak_rss, wall_pass, memory_pass


def _require_audit_exclusion(audit_job: str, producer_node: str) -> str:
    completed = subprocess.run(
        ["scontrol", "show", "job", "-o", audit_job],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(?:^| )ExcNodeList=([^ ]+)", completed.stdout)
    if match is None or match.group(1) != producer_node:
        raise RuntimeError("audit allocation does not record the exact producer-node exclusion")
    return match.group(1)


def verify_producer(
    *,
    repo_root: Path,
    producer_root: Path,
    expected_producer_job: int,
    expected_producer_commit: str,
    expected_receipt_sha256: str,
    audit_commit: str,
) -> dict[str, object]:
    """Return a stable excluded-node replay and content-hash audit receipt."""

    if expected_producer_job <= 0:
        raise ValueError("expected_producer_job must be positive")
    if _COMMIT_PATTERN.fullmatch(expected_producer_commit) is None:
        raise ValueError("expected_producer_commit must be lowercase 40-hex")
    if _COMMIT_PATTERN.fullmatch(audit_commit) is None:
        raise ValueError("audit_commit must be lowercase 40-hex")
    if _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None:
        raise ValueError("expected_receipt_sha256 must be lowercase 64-hex")
    _verify_audit_checkout(repo_root, audit_commit)
    audit_replay_source_blobs = _capture_audit_replay_sources(repo_root, audit_commit)
    expected_root = _PRODUCER_BASE / str(expected_producer_job)
    if producer_root != expected_root:
        raise ValueError("producer root does not match the independently supplied job identity")
    receipt_path = producer_root / "receipt.json"
    manifest_path = producer_root / "SHA256SUMS"
    worktree_path = producer_root / "worktree-status.txt"
    artifact_descriptor, artifact_root_snapshot, artifact_files_by_name = (
        _capture_exact_artifact_inventory(
            producer_root,
            (receipt_path, manifest_path, worktree_path),
            producer_base=_PRODUCER_BASE,
        )
    )
    os.close(artifact_descriptor)
    artifact_snapshots = {
        snapshot.relative_path: snapshot for snapshot in artifact_files_by_name.values()
    }
    receipt_snapshot = artifact_snapshots[Path("receipt.json")]
    manifest_snapshot = artifact_snapshots[Path("SHA256SUMS")]
    worktree_snapshot = artifact_snapshots[Path("worktree-status.txt")]
    if worktree_snapshot.payload != b"":
        raise ValueError("producer clean-worktree record is not empty")
    if receipt_snapshot.sha256 != expected_receipt_sha256:
        raise ValueError("producer receipt digest differs from the independently supplied value")

    expected_manifest_paths = (
        receipt_path,
        *(_PRODUCER_REPO_ROOT / path for path in _PRODUCER_SOURCE_PATHS),
        worktree_path,
    )
    manifest = _parse_manifest_bytes(manifest_snapshot.payload)
    if tuple(manifest) != expected_manifest_paths:
        raise ValueError("producer SHA256SUMS order or inventory differs from the frozen audit set")
    for snapshot in (receipt_snapshot, worktree_snapshot):
        if snapshot.sha256 != manifest[snapshot.path]:
            raise ValueError(f"producer manifest digest failed for {snapshot.path}")
    source_blobs = _verify_git_bound_sources(repo_root, expected_producer_commit, manifest)

    document = _load_canonical_json_bytes(receipt_snapshot.payload)
    _exact_object(document, keys=_EXPECTED_TOP_LEVEL_KEYS, name="producer receipt")
    spec, config_sha256 = _decode_frozen_beam_spec(source_blobs[CONFIG_RELATIVE])
    if (
        document.get("artifact") != ARTIFACT
        or document.get("schema_version") != 1
        or document.get("status") != "completed"
        or document.get("git_commit") != expected_producer_commit
        or document.get("config_sha256") != config_sha256
    ):
        raise ValueError("producer identity or frozen config binding failed")
    _verify_recorded_spec(document, spec)
    producer_accounting = _producer_accounting(expected_producer_job)
    producer_slurm = _verify_producer_metadata(
        document,
        spec=spec,
        expected_producer_job=expected_producer_job,
        accounting=producer_accounting,
    )
    audit_slurm = _required_slurm_environment(spec)
    audit_job = str(audit_slurm["job_id"])
    audit_node = str(audit_slurm["node_list"])
    producer_node = str(producer_slurm.get("node_list"))
    if audit_node == producer_node:
        raise RuntimeError("beam receipt audit must run on a different node from the producer")
    excluded_node = _require_audit_exclusion(audit_job, producer_node)

    output = _exact_object(
        document["output"],
        keys=_EXPECTED_OUTPUT_KEYS,
        name="producer output",
    )
    if output["approximation_status"] != "beam_pruned":
        raise ValueError("producer output or approximation status is invalid")
    total_scored, selected = _verify_depth_trace(
        output.get("depth_trace"),
        pool_size=spec.pool_size,
        joint_size=spec.joint_size,
        beam_width=spec.beam_width,
        max_groups_scored=spec.max_groups_scored,
    )
    if total_scored != _exact_integer(
        output.get("total_groups_scored"),
        name="total_groups_scored",
    ):
        raise ValueError("producer total scored groups does not reconstruct")
    raw_selected = output.get("selected_evaluation_indices")
    recorded_selected = (
        ()
        if raw_selected == []
        else _integer_group(raw_selected, depth=spec.joint_size, name="selected group")
    )
    if selected != recorded_selected:
        raise ValueError("producer selected group does not reconstruct")
    final_trace = output["depth_trace"][-1]
    if (
        output.get("final_group_inventory_sha256")
        != final_trace.get("scored_group_inventory_sha256")
        or output.get("final_estimate_sha256") != final_trace.get("estimate_sha256")
        or output.get("final_score_sha256") != final_trace.get("score_sha256")
    ):
        raise ValueError("producer final-output hashes do not match its complete final trace")
    synthetic_belief = _synthetic_beam_belief(spec)
    synthetic_covariance = synthetic_belief.covariance.reshape(
        spec.decision_count * spec.n_outputs,
        spec.decision_count * spec.n_outputs,
    )
    if output.get("input_covariance_sha256") != _array_sha256(synthetic_covariance):
        raise ValueError("producer synthetic input covariance does not reconstruct")

    replay = _recompute_frozen_kernel(spec)
    trace_document = output["depth_trace"]
    numeric_comparison = _replay_numeric_comparison(trace_document, replay)
    if (
        replay.approximation_status != output.get("approximation_status")
        or replay.total_groups_scored != total_scored
        or replay.selected_evaluation_indices != recorded_selected
    ):
        raise ValueError("replayed kernel final status or selection differs from the producer")

    kernel_wall, peak_rss, wall_pass, memory_pass = _verify_timing_resources_and_gate(
        document,
        spec=spec,
        total_scored=total_scored,
    )

    replayed_source_blobs = _verify_git_bound_sources(
        repo_root,
        expected_producer_commit,
        manifest,
    )
    if replayed_source_blobs != source_blobs:
        raise ValueError("producer commit source blobs changed during replay")
    replayed_audit_source_blobs = _capture_audit_replay_sources(repo_root, audit_commit)
    if replayed_audit_source_blobs != audit_replay_source_blobs:
        raise ValueError("audit replay source blobs changed during replay")
    _revalidate_closed_snapshot_set(
        root_path=producer_root,
        root_snapshot=artifact_root_snapshot,
        snapshots=artifact_snapshots,
        label="producer artifact",
    )
    _verify_audit_checkout(repo_root, audit_commit)
    artifact_snapshot_sha256 = _snapshot_set_sha256(
        artifact_root_snapshot,
        artifact_snapshots,
    )
    source_blob_inventory_sha256 = _source_blob_inventory_sha256(source_blobs)
    audit_replay_source_inventory_sha256 = _source_blob_inventory_sha256(
        audit_replay_source_blobs,
        paths=_AUDIT_REPLAY_SOURCE_PATHS,
    )
    historical_source_difference_paths = [
        relative_path.as_posix()
        for relative_path in _REPLAY_CORE_SOURCE_PATHS
        if audit_replay_source_blobs[relative_path] != source_blobs[relative_path]
    ]

    return {
        "artifact": AUDIT_ARTIFACT,
        "schema_version": 1,
        "status": "accepted",
        "finished_utc": datetime.now(UTC).isoformat(),
        "audit_git_commit": audit_commit,
        "producer": {
            "artifact": ARTIFACT,
            "job_id": str(expected_producer_job),
            "node_list": producer_slurm["node_list"],
            "git_commit": expected_producer_commit,
            "receipt_sha256": expected_receipt_sha256,
            "sha256sums_sha256": manifest_snapshot.sha256,
            "artifact_snapshot_sha256": artifact_snapshot_sha256,
            "source_blob_inventory_sha256": source_blob_inventory_sha256,
            "slurm_accounting": producer_accounting,
        },
        "audit_slurm": {
            **audit_slurm,
            "excluded_producer_node": excluded_node,
        },
        "checks": {
            "different_node": True,
            "producer_node_explicitly_excluded": True,
            "producer_slurm_completion_authenticated": True,
            "producer_top_level_schema_exactly_validated": True,
            "producer_environment_and_limitations_exactly_validated": True,
            "producer_slurm_metadata_exactly_validated": True,
            "producer_timing_resource_and_gate_schemas_exactly_validated": True,
            "producer_exact_commit_bound": True,
            "producer_clean_worktree_bound": True,
            "producer_root_exact_inventory_modes_and_links": True,
            "producer_root_and_path_identities_revalidated": True,
            "producer_artifact_contents_revalidated": True,
            "producer_receipt_hash_and_parse_use_same_stable_bytes": True,
            "producer_manifest_hash_and_parse_use_same_stable_bytes": True,
            "producer_receipt_canonical_duplicate_free_json": True,
            "producer_manifest_all_entries_bound": True,
            "producer_sources_rehydrated_from_exact_commit_blobs": True,
            "producer_commit_source_blobs_revalidated": True,
            "audit_replay_sources_match_exact_commit_blobs": True,
            "audit_replay_source_contents_revalidated": True,
            "historical_source_difference_inventory_recorded": True,
            "frozen_config_redecoded_from_producer_commit_blob": True,
            "every_depth_inventory_recomputed": True,
            "producer_score_arithmetic_and_ranking_recomputed": True,
            "current_repaired_kernel_implementation_replayed": True,
            "complete_numeric_trace_matches_declared_tolerance": True,
            "fail_closed_group_cap_respected": True,
            "resource_gate_recomputed": True,
        },
        "replay_implementation": {
            "kind": "current_audit_commit_repaired_code_lineage",
            "source_blob_inventory_sha256": audit_replay_source_inventory_sha256,
            "source_sha256_by_path": _source_sha256_by_path(
                audit_replay_source_blobs,
                paths=_AUDIT_REPLAY_SOURCE_PATHS,
            ),
            "historical_producer_core_sha256_by_path": _source_sha256_by_path(
                source_blobs,
                paths=_REPLAY_CORE_SOURCE_PATHS,
            ),
            "historical_producer_source_difference_paths": (historical_source_difference_paths),
        },
        "observed": {
            "kernel_wall_seconds": kernel_wall,
            "peak_rss_kib": peak_rss,
            "total_groups_scored": total_scored,
            "selected_evaluation_indices": list(selected),
            "operational_gate_passed": wall_pass and memory_pass,
        },
        "replay_numeric_comparison": numeric_comparison,
        "claim_scope": (
            "stable_excluded_node_behavioral_replay_with_current_repaired_code_lineage_for_"
            "synthetic_operational_evidence_only"
        ),
        "limitations": [
            "synthetic_non_biological_correlated_covariance",
            "replay_uses_current_repaired_code_lineage_not_exact_historical_producer_source",
            "replay_is_not_an_independent_algorithm_implementation",
            "same_account_preimport_source_mutation_is_outside_the_audit_threat_model",
            "historical_producer_source_path_identities_not_preserved",
            "producer_sources_rehydrated_from_exact_git_commit_blobs",
            "audit_replay_timing_is_not_producer_timing_evidence",
            "no_peptide_model_or_oracle",
            "no_scientific_performance_claim",
            "no_production_pin_change",
        ],
        "scientific_or_production_claim": "none",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-job", type=int, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    document = verify_producer(
        repo_root=args.repo_root.resolve(strict=True),
        producer_root=args.producer_root.resolve(strict=True),
        expected_producer_job=args.producer_job,
        expected_producer_commit=args.expected_producer_commit,
        expected_receipt_sha256=args.expected_receipt_sha256,
        audit_commit=args.audit_commit,
    )
    _write_exclusive(args.output, document)


if __name__ == "__main__":
    main()

"""CPU-only production-size profiler for the native-diffusion v1 scorer.

This diagnostic intentionally does not consume trained checkpoints or pilot-run
artifacts.  It rebuilds each outer-fold count prior from the accepted training
projection and substitutes production-shaped, all-zero float32 residual logits.
The resulting timings are operational evidence only: they are not scientific
model evidence and cannot authorize artifact promotion.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import os
import re
import resource
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import TypeVar
from unittest import mock

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1 import pilot_contract as pilot_contract_module
from amp_challenge.generators.diffusion.v1 import pilot_data as pilot_data_module
from amp_challenge.generators.diffusion.v1 import pilot_scoring as pilot_scoring_module
from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    RepositorySnapshot,
    build_repository_snapshot,
    canonical_json_bytes,
    validate_path_free_document,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    load_pilot_execution_v1_contract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    AuthenticatedCountPrior,
    PilotTrainingProjection,
    load_contract_fold_training_projection,
)
from amp_challenge.generators.diffusion.v1.pilot_records import (
    canonical_json_bytes as canonical_metric_json_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_records import (
    fold_method_record,
    fold_metrics_document,
)
from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    CHECKPOINT_STEPS,
    FoldMethodMetrics,
    ScoreCorruptionLedger,
    ScoringArchive,
    build_scoring_archive,
    load_contract_score_corruption_ledger,
    score_all_fold_methods,
    score_fold,
)

_ACCEPTED_PROJECTION_ROOT = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/diffusion/"
    "native-categorical-unconditional-v1/development-projections/224105/0"
)
_ACCEPTED_PROJECTION_RELATIVE_PATH = (
    "diffusion/native-categorical-unconditional-v1/development-projections/224105/0"
)
_ACCEPTED_PROJECTION_TOP_SHA256 = "4c0f627990f7f4fa43ebbe969a8422089dbc45a3317869a28035a0af24596657"
_ACCEPTED_PROJECTION_TREE_SHA256 = (
    "c693be7465c5ab2d227bc46378b10c5a407f5f4f5068e2a6046a5f13c1e3e75c"
)
_CHILD_CONTRACT_RELATIVE_PATH = Path("configs/diffusion/pilot_execution_v1.toml")
_PARENT_CONTRACT_RELATIVE_PATH = Path("configs/diffusion/unconditional_v1.toml")
_DEVELOPMENT_FOLDS = (0, 1, 2, 3)
_RESIDUE_CLASSES = 20
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CODE_INPUT_PATHS = {
    "launcher": "cluster/slurm/profile_native_diffusion_v1_scoring_cpu.sbatch",
    "pilot_artifacts_module": "src/amp_challenge/generators/diffusion/v1/pilot_artifacts.py",
    "pilot_contract_module": "src/amp_challenge/generators/diffusion/v1/pilot_contract.py",
    "pilot_data_module": "src/amp_challenge/generators/diffusion/v1/pilot_data.py",
    "pilot_records_module": "src/amp_challenge/generators/diffusion/v1/pilot_records.py",
    "pilot_scoring_module": "src/amp_challenge/generators/diffusion/v1/pilot_scoring.py",
    "scoring_profiler_module": "src/amp_challenge/generators/diffusion/v1/scoring_profiler.py",
}
_VALIDATION_TARGETS: tuple[tuple[object, str, str], ...] = (
    (
        pilot_contract_module.NativeDiffusionV1PilotContract,
        "revalidate",
        "pilot_contract_revalidate",
    ),
    (
        pilot_data_module.PilotTrainingProjection,
        "revalidate",
        "training_projection_revalidate",
    ),
    (pilot_data_module.CountPrior, "__post_init__", "count_prior_post_init"),
    (
        pilot_data_module.AuthenticatedCountPrior,
        "revalidate",
        "authenticated_count_prior_revalidate",
    ),
    (
        pilot_scoring_module.ScoreCorruptionLedger,
        "revalidate",
        "score_corruption_ledger_revalidate",
    ),
    (
        pilot_scoring_module,
        "_validate_corruption_ledger",
        "score_corruption_ledger_validate",
    ),
    (
        pilot_scoring_module.ScoringArchive,
        "revalidate",
        "scoring_archive_revalidate",
    ),
    (
        pilot_scoring_module,
        "_validate_scoring_archive",
        "scoring_archive_validate",
    ),
    (
        pilot_scoring_module,
        "_validated_scoring_view",
        "validated_scoring_view",
    ),
    (
        pilot_scoring_module,
        "gather_count_log_probability",
        "count_log_probability_gather",
    ),
    (
        pilot_scoring_module.MetricSummary,
        "__post_init__",
        "metric_summary_post_init",
    ),
    (
        pilot_scoring_module.MetricSummary,
        "revalidate",
        "metric_summary_revalidate",
    ),
    (pilot_scoring_module.RowNll, "__post_init__", "row_nll_post_init"),
    (pilot_scoring_module.RowNll, "revalidate", "row_nll_revalidate"),
    (pilot_scoring_module.LoucoPlan, "__post_init__", "louco_plan_post_init"),
    (pilot_scoring_module.LoucoPlan, "revalidate", "louco_plan_revalidate"),
    (
        pilot_scoring_module,
        "_validate_fold_method_metrics",
        "fold_method_metrics_validate",
    ),
)

_ResultT = TypeVar("_ResultT")


class _ValidationCounter:
    """Count selected validation entry points without changing their results."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter({label: 0 for _, _, label in _VALIDATION_TARGETS})

    def snapshot(self) -> dict[str, int]:
        return {label: int(self._counts[label]) for _, _, label in _VALIDATION_TARGETS}

    def counted(self, function: Callable[..., object], label: str) -> Callable[..., object]:
        @functools.wraps(function)
        def wrapper(*args: object, **kwargs: object) -> object:
            self._counts[label] += 1
            return function(*args, **kwargs)

        return wrapper


@contextmanager
def _installed_validation_counter() -> Iterator[_ValidationCounter]:
    counter = _ValidationCounter()
    with ExitStack() as stack:
        for owner, name, label in _VALIDATION_TARGETS:
            original = getattr(owner, name)
            stack.enter_context(mock.patch.object(owner, name, counter.counted(original, label)))
        yield counter


def _validation_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    if set(before) != set(after):  # pragma: no cover - fixed profiler invariant
        raise RuntimeError("validation counter inventory changed during a phase")
    return {name: after[name] - before[name] for name in before}


def _peak_rss_bytes() -> int:
    """Return Linux ``ru_maxrss`` in bytes (the source value is KiB)."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(value) not in {int, float} or value < 0:
        raise RuntimeError("process peak RSS is unavailable")
    return int(value) * 1024


def _measure(
    phase: str,
    counter: _ValidationCounter,
    action: Callable[[], _ResultT],
) -> tuple[_ResultT, dict[str, object]]:
    """Measure one callable after collecting unreachable prior-phase objects."""

    gc.collect()
    validations_before = counter.snapshot()
    peak_before = _peak_rss_bytes()
    process_before = process_time_ns()
    wall_before = perf_counter_ns()
    result = action()
    wall_after = perf_counter_ns()
    process_after = process_time_ns()
    peak_after = _peak_rss_bytes()
    validations_after = counter.snapshot()
    if wall_after < wall_before or process_after < process_before or peak_after < peak_before:
        raise RuntimeError("a monotonic profiling counter moved backwards")
    record: dict[str, object] = {
        "phase": phase,
        "wall_time_ns": wall_after - wall_before,
        "process_time_ns": process_after - process_before,
        "peak_rss_bytes": peak_after,
        "peak_rss_increase_bytes": peak_after - peak_before,
        "validation_call_counts": _validation_delta(
            validations_before,
            validations_after,
        ),
    }
    return result, record


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            observed = os.lstat(candidate)
        except OSError as error:
            raise ValueError("cannot inspect a required profiler input") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("a required profiler input traverses a symlink")


def _regular_file_sha256(path: Path) -> str:
    """Hash a stable, single-link regular file without following its leaf."""

    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError("cannot open a required profiler input") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("a profiler input is not a single-link regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.lstat(source)
    finally:
        os.close(descriptor)

    def fingerprint(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            stat.S_IMODE(item.st_mode),
            item.st_nlink,
        )

    if (
        fingerprint(before) != fingerprint(after)
        or fingerprint(before) != fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
    ):
        raise ValueError("a required profiler input changed while it was hashed")
    return digest.hexdigest()


def _validate_input_roots(
    repository_root: Path,
    projection_root: Path,
) -> tuple[Path, Path]:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    projection = Path(os.path.abspath(os.fspath(projection_root)))
    _reject_symlink_chain(repository)
    _reject_symlink_chain(projection)
    if repository.resolve(strict=True) != repository or not repository.is_dir():
        raise ValueError("repository root must be an existing canonical directory")
    if (
        projection != _ACCEPTED_PROJECTION_ROOT
        or projection.resolve(strict=True) != projection
        or not projection.is_dir()
    ):
        raise ValueError("only the accepted 224105/0 development projection is permitted")
    return repository, projection


def _load_exact_contract(repository_root: Path) -> NativeDiffusionV1PilotContract:
    return load_pilot_execution_v1_contract(
        repository_root / _CHILD_CONTRACT_RELATIVE_PATH,
        parent_path=repository_root / _PARENT_CONTRACT_RELATIVE_PATH,
    ).revalidate()


def _code_input_sha256s(snapshot: RepositorySnapshot) -> dict[str, str]:
    observed = {entry.relative_path: entry.sha256 for entry in snapshot.entries}
    if any(path not in observed for path in _CODE_INPUT_PATHS.values()):
        raise ValueError("repository snapshot is missing profiler code inputs")
    return {label: observed[path] for label, path in _CODE_INPUT_PATHS.items()}


def _validate_contract_projection(
    contract: NativeDiffusionV1PilotContract,
    projection_root: Path,
) -> None:
    projection = contract.table("projection")
    if (
        tuple(fold.outer_fold for fold in contract.folds) != _DEVELOPMENT_FOLDS
        or projection["producer_job_id"] != 224105
        or projection["canonical_twin_slot"] != 0
        or projection["canonical_bundle_relative_path"] != _ACCEPTED_PROJECTION_RELATIVE_PATH
        or contract.projection_top_sha256 != _ACCEPTED_PROJECTION_TOP_SHA256
        or contract.projection_tree_sha256 != _ACCEPTED_PROJECTION_TREE_SHA256
        or _regular_file_sha256(projection_root / "SHA256SUMS") != _ACCEPTED_PROJECTION_TOP_SHA256
    ):
        raise ValueError("contract and accepted development projection identity differ")


def _load_training_projection(
    contract: NativeDiffusionV1PilotContract,
    projection_root: Path,
    outer_fold: int,
) -> PilotTrainingProjection:
    fold = contract.fold(outer_fold)
    expected_path = f"folds/{outer_fold}/train.jsonl"
    if fold.train_path != expected_path:
        raise ValueError("training projection path differs from the exact fold contract")
    result = load_contract_fold_training_projection(
        contract,
        outer_fold,
        projection_root / expected_path,
    )
    result.revalidate()
    return result


def _load_score_ledger(
    contract: NativeDiffusionV1PilotContract,
    projection_root: Path,
    outer_fold: int,
) -> ScoreCorruptionLedger:
    fold = contract.fold(outer_fold)
    expected_path = f"folds/{outer_fold}/score.jsonl"
    if fold.score_path != expected_path:
        raise ValueError("score projection path differs from the exact fold contract")
    result = load_contract_score_corruption_ledger(
        contract,
        outer_fold,
        projection_root / expected_path,
    )
    result.revalidate()
    return result


def _gather_count_prior(
    ledger: ScoreCorruptionLedger,
    count_prior: AuthenticatedCountPrior,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.float64],
]:
    decoded = count_prior.revalidate()
    return pilot_scoring_module.gather_count_log_probability(
        ledger,
        decoded.log_relative_position_probability,
    )


def _build_archive(
    ledger: ScoreCorruptionLedger,
    count_prior: AuthenticatedCountPrior,
    gathered: tuple[
        NDArray[np.uint64],
        NDArray[np.uint8],
        NDArray[np.uint8],
        NDArray[np.float64],
    ],
    residual_logits: NDArray[np.float32],
) -> ScoringArchive:
    offsets, positions, targets, count_log_probability = gathered
    return build_scoring_archive(
        ledger,
        count_prior=count_prior,
        case_id=ledger.arrays()["case_id"],
        case_offsets=offsets,
        position=positions,
        target_token=targets,
        count_log_probability=count_log_probability,
        checkpoint_step=np.asarray(CHECKPOINT_STEPS, dtype="<u2"),
        residual_logit=residual_logits,
    )


def _method_key(value: FoldMethodMetrics) -> str:
    if value.checkpoint_step is None:
        return value.method
    return f"{value.method}-{value.checkpoint_step:06d}"


def _method_payload(value: FoldMethodMetrics) -> bytes:
    return canonical_metric_json_bytes(fold_method_record(value))


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _outer_fold(argument: int | None) -> int:
    slurm_rank = os.environ.get("SLURM_PROCID")
    if argument is None:
        if slurm_rank is None or not slurm_rank.isascii() or not slurm_rank.isdecimal():
            raise ValueError("outer fold must be passed explicitly or supplied by SLURM_PROCID")
        result = int(slurm_rank)
    else:
        result = argument
        if slurm_rank is not None and int(slurm_rank) != result:
            raise ValueError("explicit outer fold differs from SLURM_PROCID")
    if type(result) is not int or result not in _DEVELOPMENT_FOLDS:
        raise ValueError("outer fold must be exactly one of 0, 1, 2, 3")
    return result


def run_profile(
    *,
    repository_root: Path,
    projection_root: Path,
    outer_fold: int,
    expected_git_commit: str,
) -> dict[str, object]:
    """Run one diagnostic fold profile and return one path-free JSON value."""

    if type(outer_fold) is not int or outer_fold not in _DEVELOPMENT_FOLDS:
        raise ValueError("outer fold must be exactly one of 0, 1, 2, 3")
    if (
        type(expected_git_commit) is not str
        or _GIT_COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise ValueError("expected Git commit must be a lowercase forty-character object ID")
    repository, projection = _validate_input_roots(repository_root, projection_root)
    timings: list[dict[str, object]] = []
    method_sha256s: dict[str, str] = {}
    method_byte_counts: dict[str, int] = {}

    with _installed_validation_counter() as counter:
        repository_snapshot, timing = _measure(
            "repository_snapshot",
            counter,
            lambda: build_repository_snapshot(
                repository,
                expected_commit=expected_git_commit,
            ),
        )
        timings.append(timing)
        contract, timing = _measure(
            "exact_contract_load",
            counter,
            lambda: _load_exact_contract(repository),
        )
        timings.append(timing)
        _validate_contract_projection(contract, projection)
        fold = contract.fold(outer_fold)

        ledger, timing = _measure(
            "score_corruption_ledger_load",
            counter,
            lambda: _load_score_ledger(contract, projection, outer_fold),
        )
        timings.append(timing)
        training, timing = _measure(
            "training_projection_load",
            counter,
            lambda: _load_training_projection(contract, projection, outer_fold),
        )
        timings.append(timing)
        count_prior, timing = _measure(
            "count_prior_reconstruction",
            counter,
            lambda: AuthenticatedCountPrior.from_projection(training),
        )
        timings.append(timing)
        gathered, timing = _measure(
            "count_log_probability_gather",
            counter,
            lambda: _gather_count_prior(ledger, count_prior),
        )
        timings.append(timing)
        selected_token_count = len(gathered[2])
        residual_logits, timing = _measure(
            "diagnostic_zero_residual_logits_construction",
            counter,
            lambda: np.zeros(
                (len(CHECKPOINT_STEPS), selected_token_count, _RESIDUE_CLASSES),
                dtype="<f4",
                order="C",
            ),
        )
        timings.append(timing)
        residual_logits_sha256 = _sha256(memoryview(residual_logits).cast("B"))
        archive, timing = _measure(
            "scoring_archive_construction",
            counter,
            lambda: _build_archive(
                ledger,
                count_prior,
                gathered,
                residual_logits,
            ),
        )
        timings.append(timing)

        method_requests = (
            ("C0", None),
            ("C0T", None),
            *(("R128", step) for step in CHECKPOINT_STEPS),
        )
        for method, checkpoint_step in method_requests:
            label = method if checkpoint_step is None else f"{method}_{checkpoint_step:06d}"
            metric, timing = _measure(
                f"score_{label}",
                counter,
                lambda method=method, checkpoint_step=checkpoint_step: score_fold(
                    archive,
                    method=method,
                    checkpoint_step=checkpoint_step,
                ),
            )
            timings.append(timing)
            key = _method_key(metric)
            payload = _method_payload(metric)
            method_sha256s[key] = _sha256(payload)
            method_byte_counts[key] = len(payload)
            del metric, payload

        all_methods, timing = _measure(
            "score_all_fold_methods",
            counter,
            lambda: score_all_fold_methods(archive),
        )
        timings.append(timing)
        all_methods_document = fold_metrics_document(
            all_methods,
            child_contract_sha256=contract.config_sha256,
            parent_contract_sha256=contract.parent_config_sha256,
        )
        all_methods_payload = canonical_metric_json_bytes(all_methods_document)
        all_method_sha256s = {
            _method_key(method): _sha256(_method_payload(method)) for method in all_methods
        }
        if all_method_sha256s != method_sha256s:
            raise RuntimeError("individual and combined scoring calls produced different metrics")

        corruption_payload, timing = _measure(
            "score_corruption_npz_serialization",
            counter,
            ledger.npz_bytes,
        )
        timings.append(timing)
        archive_payload, timing = _measure(
            "score_residual_logits_npz_serialization",
            counter,
            archive.npz_bytes,
        )
        timings.append(timing)

    if (
        selected_token_count != fold.score_selected_tokens
        or len(ledger.rows) != fold.score_rows
        or len(training.rows) != fold.train_rows
        or residual_logits.dtype != np.dtype("<f4")
        or residual_logits.shape
        != (len(CHECKPOINT_STEPS), fold.score_selected_tokens, _RESIDUE_CLASSES)
        or not bool(np.count_nonzero(residual_logits) == 0)
    ):
        raise RuntimeError("diagnostic inputs differ from the production-size fold contract")

    report: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_diffusion_v1_production_size_cpu_scoring_profile",
        "status": "diagnostic_non_scientific",
        "git_commit": repository_snapshot.git_commit,
        "outer_fold": outer_fold,
        "repository_code_sha256": repository_snapshot.code_sha256,
        "code_input_sha256s": _code_input_sha256s(repository_snapshot),
        "input_sha256s": {
            "accepted_projection_top_manifest": contract.projection_top_sha256,
            "accepted_projection_tree": contract.projection_tree_sha256,
            "child_contract": contract.config_sha256,
            "parent_contract": contract.parent_config_sha256,
            "score_projection": fold.score_sha256,
            "training_projection": fold.train_sha256,
        },
        "input_census": {
            "checkpoint_count": len(CHECKPOINT_STEPS),
            "residue_classes": _RESIDUE_CLASSES,
            "score_cases": fold.score_cases,
            "score_rows": fold.score_rows,
            "selected_tokens": selected_token_count,
            "training_rows": fold.train_rows,
        },
        "diagnostic_residual_logits": {
            "construction": "all_zero",
            "dtype": "<f4",
            "shape": list(residual_logits.shape),
            "raw_bytes_sha256": residual_logits_sha256,
            "scientific_model_output": False,
        },
        "measurement_contract": {
            "clocks": ["perf_counter_ns", "process_time_ns"],
            "garbage_collection_before_each_phase": True,
            "peak_rss_source": "getrusage_rusage_self_ru_maxrss_linux_kib_to_bytes",
            "timed_scope": "named_callable_only",
            "validation_counts_scope": "named_callable_only",
            "validation_entry_points": [label for _, _, label in _VALIDATION_TARGETS],
        },
        "timings": timings,
        "output_sha256s": {
            "count_prior_npz": count_prior.sha256,
            "diagnostic_fold_metrics_json": _sha256(all_methods_payload),
            "diagnostic_method_json_by_method": method_sha256s,
            "score_corruption_arrays": ledger.arrays_sha256,
            "score_corruptions_npz": _sha256(corruption_payload),
            "scoring_archive_integrity_arrays": archive.arrays_sha256,
            "score_residual_logits_npz": _sha256(archive_payload),
        },
        "output_byte_counts": {
            "count_prior_npz": len(count_prior.payload),
            "diagnostic_fold_metrics_json": len(all_methods_payload),
            "diagnostic_method_json_by_method": method_byte_counts,
            "score_corruptions_npz": len(corruption_payload),
            "score_residual_logits_npz": len(archive_payload),
        },
        "checks": {
            "accepted_projection_224105_twin_0_only": True,
            "artifact_promotion_authorized": False,
            "development_folds_only": True,
            "fold4_read": False,
            "individual_and_combined_metrics_exact": True,
            "pilot_execution_artifacts_read": False,
            "scientific_evidence": False,
        },
    }
    validate_path_free_document(report, label="CPU scoring profile")
    canonical_json_bytes(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=_ACCEPTED_PROJECTION_ROOT,
    )
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--expected-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_profile(
        repository_root=arguments.repository_root,
        projection_root=arguments.projection_root,
        outer_fold=_outer_fold(arguments.outer_fold),
        expected_git_commit=arguments.expected_git_commit,
    )
    payload = canonical_json_bytes(report)
    written = sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    if written != len(payload):  # pragma: no cover - buffered stdout contract
        raise RuntimeError("profile output was truncated")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

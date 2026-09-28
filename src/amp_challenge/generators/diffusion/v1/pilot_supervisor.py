"""Production run control for the four-node native-diffusion v1 pilot.

One instance runs inside each task of the single four-node GPU ``srun``.  The
Slurm rank is the outer fold; no fold is accepted from a caller.  Training and
evaluation execute as separate child processes with deliberately scrubbed
environments, and the score projection is not opened or staged until the
four-fold checkpoint-ready release has been independently rebuilt by rank 0.

This module publishes operational producer evidence only.  It neither builds
the scientific pilot bundle nor makes a continuation/no-go decision; those
belong to the excluded-node verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import pwd
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    RepositorySnapshot,
    build_repository_snapshot,
    canonical_json_bytes,
    parse_canonical_json,
    parse_sha256sums,
    validate_path_free_document,
    verify_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    load_pilot_execution_v1_contract,
)
from amp_challenge.generators.diffusion.v1.pilot_control import (
    CheckpointDigest,
    ScoreReleaseReceipt,
    TrainerReadinessObservation,
    TrainerReadinessReceipt,
    build_score_release_receipt,
    build_trainer_readiness_receipt,
    parse_score_release_receipt,
    parse_trainer_readiness_receipt,
    publish_score_release_receipt,
    publish_trainer_readiness_receipt,
    verify_trainer_readiness_receipt,
)
from amp_challenge.generators.diffusion.v1.pilot_data import _read_regular_bytes
from amp_challenge.generators.diffusion.v1.pilot_inference import ReinferenceComparison
from amp_challenge.generators.diffusion.v1.pilot_progress import (
    EVALUATOR_PROGRESS_PHASES,
    EvaluatorProgressTrace,
    EvaluatorProgressWriter,
    verify_evaluator_progress,
)
from amp_challenge.generators.diffusion.v1.pilot_trainer_bundle import (
    AuthenticatedTrainerBundle,
    authenticate_trainer_bundle,
)

_FOLDS = (0, 1, 2, 3)
_FOLD_KEYS = ("0", "1", "2", "3")
_CHECKPOINT_STEPS = (250, 500, 1000, 2000, 4000)
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_JOB_RE = re.compile(r"[1-9][0-9]*")
_NODE_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,253}[A-Za-z0-9])?")
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_VISIBLE_GPU_RE = re.compile(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)")
_MAX_CONTRACT_BYTES = 262_144
_MAX_PROJECTION_BYTES = 64 * 1024 * 1024
_MAX_CONTROL_BYTES = 1 << 20
_MAX_CHILD_STDOUT_BYTES = 8 << 20
_MAX_CHILD_TELEMETRY_BYTES = 65_536
_ALLOCATION_TIME_LIMIT_SECONDS = 15 * 60
_CONTROLLER_CLEANUP_RESERVE_SECONDS = 60
_CHILD_TERMINATION_RESERVE_SECONDS = 120
_RUN_TIMEOUT_SECONDS = 12 * 60
_POLL_SECONDS = 2.0
_EXPECTED_GPU_MEMORY_MIB = 81_920
_MAX_TELEMETRY_SAMPLES_PER_PHASE = 2_048
_TELEMETRY_FRAME_LENGTH_BYTES = 4
_SUPERVISOR_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_INSTRUMENTED_CHILD_SOURCE = (
    "from amp_challenge.generators.diffusion.v1.pilot_supervisor "
    "import _instrumented_child_entry;raise SystemExit(_instrumented_child_entry())"
)
_TRAINER_RESULT_FIELDS = (
    "bundle_tree_sha256",
    "checkpoint_digest_by_step",
    "count_prior_file_sha256",
    "fit_identity_sha256",
    "outer_fold",
)
_EVALUATOR_RESULT_FIELDS = (
    "checkpoint_digest_by_step",
    "count_prior_file_sha256",
    "evaluator_bundle_sha256",
    "outer_fold",
    "reinference_by_step",
    "trainer_bundle_sha256",
)
_EVALUATOR_FROZEN_INPUT_PATHS = (
    "checkpoint-ready.json",
    "count_prior.npz",
    "pilot_execution_v1.toml",
    "score-release.json",
    "score.jsonl",
    "trainer_bundle.tree.json",
    "unconditional_v1.toml",
)
_WORKER_TELEMETRY_FIELDS = (
    "artifact",
    "checks",
    "child_contract_sha256",
    "child_process_pid_by_phase",
    "cuda_device_uuid",
    "evaluator_progress",
    "git_commit",
    "in_allocation_nvidia_smi_process_memory_samples",
    "node_name",
    "outer_fold",
    "parent_contract_sha256",
    "process_pid",
    "producer_job_id",
    "schema_version",
    "slurm_step_accounting",
    "torch_cuda_by_phase",
    "torch_cuda_max_memory_allocated",
    "torch_cuda_max_memory_reserved",
)
_CHILD_PASSTHROUGH_ENVIRONMENT = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "PATH",
    "USER",
)


class SupervisorInterrupted(RuntimeError):
    """Raised when Slurm or an operator interrupts one supervised worker."""


@dataclass(frozen=True, slots=True)
class SchedulerIdentity:
    """Authenticated rank, node, and sole visible A100 identity."""

    job_id: str
    step_id: int
    outer_fold: int
    node_name: str
    device_uuid: str
    cuda_visible_device: str

    def __post_init__(self) -> None:
        if type(self.job_id) is not str or _JOB_RE.fullmatch(self.job_id) is None:
            raise ValueError("job_id must be a positive decimal Slurm job ID")
        if type(self.step_id) is not int or self.step_id != 0:
            raise ValueError("pilot workers must run in exact Slurm step zero")
        _outer_fold(self.outer_fold)
        _node_name(self.node_name)
        if type(self.device_uuid) is not str or _GPU_UUID_RE.fullmatch(self.device_uuid) is None:
            raise ValueError("device_uuid must be one canonical NVIDIA GPU UUID")
        if (
            type(self.cuda_visible_device) is not str
            or _VISIBLE_GPU_RE.fullmatch(self.cuda_visible_device) is None
        ):
            raise ValueError("cuda_visible_device must identify exactly one full GPU")


@dataclass(frozen=True, slots=True)
class AllocationContainment:
    """Authenticated allocation epochs projected onto the worker monotonic clock."""

    job_id: str
    allocation_start_epoch_seconds: int
    allocation_end_epoch_seconds: int
    step_deadline_epoch_seconds: int
    worker_deadline_epoch_seconds: int
    authenticated_epoch_seconds: float
    allocation_remaining_seconds: float
    step_deadline_monotonic_seconds: float
    worker_deadline_monotonic_seconds: float

    def __post_init__(self) -> None:
        _job_id(self.job_id)
        integer_values = (
            self.allocation_start_epoch_seconds,
            self.allocation_end_epoch_seconds,
            self.step_deadline_epoch_seconds,
            self.worker_deadline_epoch_seconds,
        )
        if any(
            type(value) is not int or not 1_000_000_000 <= value <= 9_999_999_999
            for value in integer_values
        ):
            raise ValueError("allocation containment epochs must be exact ten-digit integers")
        float_values = (
            self.authenticated_epoch_seconds,
            self.allocation_remaining_seconds,
            self.step_deadline_monotonic_seconds,
            self.worker_deadline_monotonic_seconds,
        )
        if any(type(value) is not float for value in float_values):
            raise TypeError("allocation containment runtime values must be exact floats")
        if (
            self.allocation_end_epoch_seconds - self.allocation_start_epoch_seconds
            != _ALLOCATION_TIME_LIMIT_SECONDS
            or self.allocation_end_epoch_seconds - self.step_deadline_epoch_seconds
            != _CONTROLLER_CLEANUP_RESERVE_SECONDS
            or self.step_deadline_epoch_seconds - self.worker_deadline_epoch_seconds
            != _CHILD_TERMINATION_RESERVE_SECONDS
            or not self.allocation_start_epoch_seconds
            <= self.authenticated_epoch_seconds
            < self.worker_deadline_epoch_seconds
            or self.allocation_remaining_seconds
            != self.allocation_end_epoch_seconds - self.authenticated_epoch_seconds
            or not 0.0
            < self.worker_deadline_monotonic_seconds
            < self.step_deadline_monotonic_seconds
        ):
            raise ValueError("allocation containment hierarchy is inconsistent")


@dataclass(frozen=True, slots=True)
class PilotRunLayout:
    """Exact shared directories for one no-overwrite producer job."""

    root: Path
    trainers: Path
    evaluators: Path
    control: Path
    readiness: Path
    producer_results: Path
    operational_telemetry: Path
    evaluator_progress: Path
    release: Path
    producer_reinference: Path
    operational_telemetry_aggregate: Path


@dataclass(frozen=True, slots=True)
class ChildCudaTelemetry:
    """Allocator peaks and bounded external samples from one CUDA child."""

    role: str
    process_pid: int
    torch_cuda_max_memory_allocated: int
    torch_cuda_max_memory_reserved: int
    nvidia_smi_process_memory_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role not in {"trainer", "evaluator"}:
            raise ValueError("CUDA child role must be trainer or evaluator")
        if type(self.process_pid) is not int or self.process_pid <= 1:
            raise ValueError("CUDA child PID must be a positive non-init process ID")
        for label, value in (
            ("allocated peak", self.torch_cuda_max_memory_allocated),
            ("reserved peak", self.torch_cuda_max_memory_reserved),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"CUDA child {label} must be a positive byte count")
        if self.torch_cuda_max_memory_reserved < self.torch_cuda_max_memory_allocated:
            raise ValueError("CUDA reserved peak cannot be below allocated peak")
        samples = self.nvidia_smi_process_memory_samples
        if (
            type(samples) is not tuple
            or not 0 < len(samples) <= _MAX_TELEMETRY_SAMPLES_PER_PHASE
            or any(type(value) is not int or value <= 0 for value in samples)
        ):
            raise ValueError("CUDA child requires bounded positive nvidia-smi samples")

    def document(self) -> dict[str, object]:
        return {
            "process_pid": self.process_pid,
            "torch_cuda_max_memory_allocated": self.torch_cuda_max_memory_allocated,
            "torch_cuda_max_memory_reserved": self.torch_cuda_max_memory_reserved,
            "nvidia_smi_process_memory_samples": list(self.nvidia_smi_process_memory_samples),
        }


@dataclass(frozen=True, slots=True)
class ParsedEvaluatorResult:
    """Strict path-free form of one evaluator CLI result."""

    evaluator_bundle_sha256: str
    outer_fold: int
    trainer_bundle_sha256: str
    count_prior_file_sha256: str
    checkpoint_digest_by_step: Mapping[str, CheckpointDigest]
    reinference_by_step: Mapping[str, ReinferenceComparison]

    def __post_init__(self) -> None:
        _sha256(self.evaluator_bundle_sha256, label="evaluator bundle SHA-256")
        _outer_fold(self.outer_fold)
        _sha256(self.trainer_bundle_sha256, label="trainer bundle SHA-256")
        _sha256(self.count_prior_file_sha256, label="count-prior SHA-256")
        object.__setattr__(
            self,
            "checkpoint_digest_by_step",
            _freeze_checkpoint_map(self.checkpoint_digest_by_step),
        )
        object.__setattr__(
            self,
            "reinference_by_step",
            _freeze_reinference_map(self.reinference_by_step),
        )

    def document(self) -> dict[str, object]:
        """Return the exact canonical evaluator stdout document."""

        return {
            "checkpoint_digest_by_step": {
                key: self.checkpoint_digest_by_step[key].document() for key in _CHECKPOINT_KEYS
            },
            "count_prior_file_sha256": self.count_prior_file_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "outer_fold": self.outer_fold,
            "reinference_by_step": {
                key: self.reinference_by_step[key].canonical_record() for key in _CHECKPOINT_KEYS
            },
            "trainer_bundle_sha256": self.trainer_bundle_sha256,
        }


@dataclass(frozen=True, slots=True)
class SupervisorWorkerResult:
    """Path-free operational completion returned by one worker."""

    outer_fold: int
    producer_result_sha256: str
    score_release_sha256: str
    producer_reinference_sha256: str | None

    def __post_init__(self) -> None:
        _outer_fold(self.outer_fold)
        _sha256(self.producer_result_sha256, label="producer-result SHA-256")
        _sha256(self.score_release_sha256, label="score-release SHA-256")
        if self.producer_reinference_sha256 is not None:
            _sha256(
                self.producer_reinference_sha256,
                label="producer reinference SHA-256",
            )

    def document(self) -> dict[str, object]:
        """Return operational evidence without a scientific decision field."""

        return {
            "artifact": "native_categorical_diffusion_v1_r128_pilot_supervisor_result",
            "outer_fold": self.outer_fold,
            "producer_reinference_sha256": self.producer_reinference_sha256,
            "producer_result_sha256": self.producer_result_sha256,
            "score_release_sha256": self.score_release_sha256,
            "status": "execution_evidence_only_not_scientific_authorization",
        }


def allocation_containment_from_environment(
    *,
    environment: Mapping[str, str] | None = None,
    _test_only_wall_clock: Callable[[], float] | None = None,
    _test_only_monotonic_clock: Callable[[], float] | None = None,
) -> AllocationContainment:
    """Authenticate the shell-established allocation deadline before GPU/data use."""

    if environment is None:
        values: Mapping[str, str] = os.environ
    else:
        if not isinstance(environment, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise TypeError("allocation containment environment must be a string mapping")
        values = environment
    if any(name.startswith("SLURM_ARRAY_") for name in values):
        raise RuntimeError("allocation containment forbids Slurm arrays")
    wall_clock = time.time if _test_only_wall_clock is None else _test_only_wall_clock
    monotonic_clock = (
        time.monotonic if _test_only_monotonic_clock is None else _test_only_monotonic_clock
    )
    if not callable(wall_clock) or not callable(monotonic_clock):
        raise TypeError("allocation containment clock seams must be callable")
    job_id = _job_id(_required_environment(values, "SLURM_JOB_ID"))
    _decimal_environment(values, "SLURM_STEP_ID", expected=0)
    expected = {
        "AMP_CHILD_TERMINATION_RESERVE_SECONDS": _CHILD_TERMINATION_RESERVE_SECONDS,
        "AMP_CONTROLLER_CLEANUP_RESERVE_SECONDS": _CONTROLLER_CLEANUP_RESERVE_SECONDS,
    }
    for name, expected_value in expected.items():
        _decimal_environment(values, name, expected=expected_value)
    allocation_start = _decimal_environment(values, "AMP_ALLOCATION_START_EPOCH_SECONDS")
    allocation_end = _decimal_environment(values, "AMP_ALLOCATION_END_EPOCH_SECONDS")
    step_deadline = _decimal_environment(values, "AMP_STEP_DEADLINE_EPOCH_SECONDS")
    worker_deadline = _decimal_environment(values, "AMP_WORKER_DEADLINE_EPOCH_SECONDS")
    integer_values = (allocation_start, allocation_end, step_deadline, worker_deadline)
    if any(not 1_000_000_000 <= value <= 9_999_999_999 for value in integer_values):
        raise ValueError("allocation containment epochs must be ten-digit UTC seconds")
    if (
        allocation_end - allocation_start != _ALLOCATION_TIME_LIMIT_SECONDS
        or allocation_end - step_deadline != _CONTROLLER_CLEANUP_RESERVE_SECONDS
        or step_deadline - worker_deadline != _CHILD_TERMINATION_RESERVE_SECONDS
        or worker_deadline - allocation_start != _RUN_TIMEOUT_SECONDS
    ):
        raise RuntimeError("allocation containment environment violates the fixed hierarchy")

    # Take the monotonic sample first, making the wall-to-monotonic projection
    # conservative by the (tiny) time spent obtaining the subsequent wall sample.
    monotonic_now = monotonic_clock()
    wall_now = wall_clock()
    if (
        type(monotonic_now) is not float
        or monotonic_now <= 0.0
        or type(wall_now) is not float
        or not 1_000_000_000.0 <= wall_now <= 9_999_999_999.0
    ):
        raise RuntimeError("allocation containment clocks returned unsafe values")
    if not allocation_start <= wall_now < worker_deadline:
        raise TimeoutError("no contained Python worker interval remains")
    allocation_remaining = float(allocation_end) - wall_now
    return AllocationContainment(
        job_id=job_id,
        allocation_start_epoch_seconds=allocation_start,
        allocation_end_epoch_seconds=allocation_end,
        step_deadline_epoch_seconds=step_deadline,
        worker_deadline_epoch_seconds=worker_deadline,
        authenticated_epoch_seconds=wall_now,
        allocation_remaining_seconds=allocation_remaining,
        step_deadline_monotonic_seconds=monotonic_now + step_deadline - wall_now,
        worker_deadline_monotonic_seconds=monotonic_now + worker_deadline - wall_now,
    )


def run_pilot_supervisor_worker(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    projection_root: str | os.PathLike[str],
    run_root: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> SupervisorWorkerResult:
    """Run one rank-derived fold through train, release, and evaluate phases."""

    containment = allocation_containment_from_environment()
    deadline = containment.worker_deadline_monotonic_seconds
    commit = _git_commit(expected_git_commit)
    contract = load_pilot_execution_v1_contract(
        child_contract_path,
        parent_path=parent_contract_path,
    )
    _validate_contract_surface(contract)
    identity = scheduler_identity_from_environment(contract)
    repository = build_repository_snapshot(repository_root, expected_commit=commit)
    if repository.git_commit != commit:
        raise RuntimeError("repository snapshot changed the expected commit")
    layout = _open_run_layout(contract, run_root=run_root, job_id=identity.job_id)
    projection = _open_projection_root(contract, projection_root=projection_root)
    node_root = _node_local_root(identity)
    if Path(os.path.abspath(os.fspath(child_contract_path))).is_relative_to(projection):
        raise ValueError("child contract must not be sourced from the projection bundle")
    if Path(os.path.abspath(os.fspath(parent_contract_path))).is_relative_to(projection):
        raise ValueError("parent contract must not be sourced from the projection bundle")

    with _signal_boundary():
        train_payload = _read_projection_role(
            contract,
            projection,
            outer_fold=identity.outer_fold,
            role="train",
        )
        with _private_staged_input(
            node_root,
            role="train",
            payload=train_payload,
        ) as train_path:
            del train_payload
            trainer_stdout, trainer_telemetry = _run_child(
                _trainer_command(
                    child_contract_path=child_contract_path,
                    parent_contract_path=parent_contract_path,
                    train_path=train_path,
                    output_dir=layout.trainers / str(identity.outer_fold),
                    repository_root=repository_root,
                    expected_git_commit=commit,
                    outer_fold=identity.outer_fold,
                ),
                environment=_child_environment(node_root),
                deadline=deadline,
                label="pilot trainer",
                role="trainer",
                cuda_visible_device=identity.cuda_visible_device,
                expected_device_uuid=identity.device_uuid,
            )
            _remaining_seconds(deadline, label="pilot trainer post-child checks")
            _enforce_child_allocated_memory_gate(contract, trainer_telemetry)
            trainer = authenticate_trainer_bundle(
                contract,
                layout.trainers / str(identity.outer_fold),
                expected_outer_fold=identity.outer_fold,
                expected_code_sha256sums=repository.code_sha256sums,
            )
            _remaining_seconds(deadline, label="pilot trainer result validation")
            _validate_trainer_stdout(
                trainer_stdout,
                trainer=trainer,
                outer_fold=identity.outer_fold,
            )
            observation = trainer.make_readiness_observation(
                trusted_git_commit=commit,
                node_name=identity.node_name,
                device_uuid=identity.device_uuid,
            )
            readiness_receipt = build_trainer_readiness_receipt(
                contract=contract,
                observation=observation,
            )
            readiness_path = layout.readiness / f"{identity.outer_fold}.json"
            publish_trainer_readiness_receipt(
                readiness_path,
                readiness_receipt,
                contract=contract,
                observation=observation,
            )

            if identity.outer_fold == 0:
                _wait_for_sealed_files(
                    tuple(layout.readiness / f"{fold}.json" for fold in _FOLDS),
                    deadline=deadline,
                    label="four checkpoint-ready receipts",
                )
                coordinate_score_release(
                    contract=contract,
                    layout=layout,
                    repository=repository,
                    expected_git_commit=commit,
                    deadline=deadline,
                )
            release = _wait_and_verify_own_release(
                contract=contract,
                layout=layout,
                readiness_receipt=readiness_receipt,
                expected_git_commit=commit,
                deadline=deadline,
            )
            _remaining_seconds(deadline, label="released score staging")
        # The train-only parent and its sole file have been removed here.  No
        # score source was opened or staged before the verified release.

        release_sha256 = release.sha256
        progress = EvaluatorProgressWriter(
            layout.evaluator_progress / str(identity.outer_fold),
            child_contract_sha256=contract.config_sha256,
            parent_contract_sha256=contract.parent_config_sha256,
            git_commit=commit,
            outer_fold=identity.outer_fold,
            score_release_sha256=release_sha256,
        )
        progress.advance("score_projection_staging")
        score_payload = _read_projection_role(
            contract,
            projection,
            outer_fold=identity.outer_fold,
            role="score",
        )
        with _private_staged_input(
            node_root,
            role="score",
            payload=score_payload,
        ) as score_path:
            del score_payload
            progress.advance("evaluator_child_launch")
            evaluator_stdout, evaluator_telemetry = _run_child(
                _evaluator_command(
                    child_contract_path=child_contract_path,
                    parent_contract_path=parent_contract_path,
                    score_path=score_path,
                    trainer_bundle=layout.trainers / str(identity.outer_fold),
                    readiness_receipt=layout.readiness / f"{identity.outer_fold}.json",
                    score_release=layout.release,
                    progress_dir=layout.evaluator_progress / str(identity.outer_fold),
                    trace_monotonic_origin_ns=progress.monotonic_origin_ns,
                    supervisor_process_pid=progress.process_pid,
                    output_dir=layout.evaluators / str(identity.outer_fold),
                    repository_root=repository_root,
                    expected_git_commit=commit,
                    outer_fold=identity.outer_fold,
                ),
                environment=_child_environment(node_root),
                deadline=deadline,
                label="pilot evaluator",
                role="evaluator",
                cuda_visible_device=identity.cuda_visible_device,
                expected_device_uuid=identity.device_uuid,
            )
            _remaining_seconds(deadline, label="post-evaluator progress authentication")
            progress_trace = verify_evaluator_progress(
                layout.evaluator_progress / str(identity.outer_fold),
                expected_child_contract_sha256=contract.config_sha256,
                expected_parent_contract_sha256=contract.parent_config_sha256,
                expected_git_commit=commit,
                expected_outer_fold=identity.outer_fold,
                expected_score_release_sha256=release_sha256,
                expected_supervisor_process_pid=progress.process_pid,
                expected_evaluator_process_pid=evaluator_telemetry.process_pid,
                require_complete=True,
            )
            _remaining_seconds(deadline, label="post-evaluator result validation")
            _enforce_child_allocated_memory_gate(contract, evaluator_telemetry)
            evaluator_result = parse_evaluator_result(evaluator_stdout)
            _validate_evaluator_result(
                evaluator_result,
                contract=contract,
                trainer=trainer,
                layout=layout,
                repository=repository,
                expected_git_commit=commit,
                outer_fold=identity.outer_fold,
                readiness_receipt_sha256=readiness_receipt.sha256,
                score_release_sha256=release_sha256,
            )
            _remaining_seconds(deadline, label="fold producer-evidence publication")
        # The private score copy is gone before any cross-fold aggregation.

        result_path = layout.producer_results / f"{identity.outer_fold}.json"
        producer_result_payload = canonical_json_bytes(evaluator_result.document())
        _publish_canonical_record(result_path, producer_result_payload)
        telemetry_payload = canonical_json_bytes(
            build_worker_telemetry_document(
                contract=contract,
                identity=identity,
                expected_git_commit=commit,
                trainer=trainer_telemetry,
                evaluator=evaluator_telemetry,
                evaluator_progress=progress_trace,
            )
        )
        _publish_canonical_record(
            layout.operational_telemetry / f"{identity.outer_fold}.json",
            telemetry_payload,
        )
        _remaining_seconds(deadline, label="cross-fold producer aggregation")

        aggregate_sha256: str | None = None
        if identity.outer_fold == 0:
            result_paths = tuple(layout.producer_results / f"{fold}.json" for fold in _FOLDS)
            _wait_for_sealed_files(
                (
                    *result_paths,
                    *(layout.operational_telemetry / f"{fold}.json" for fold in _FOLDS),
                ),
                deadline=deadline,
                label="four evaluator results and operational telemetry records",
            )
            aggregate_payload = coordinate_producer_reinference(
                contract=contract,
                layout=layout,
                repository=repository,
                expected_git_commit=commit,
                deadline=deadline,
            )
            aggregate_sha256 = hashlib.sha256(aggregate_payload).hexdigest()
            coordinate_operational_telemetry(
                contract=contract,
                layout=layout,
                expected_git_commit=commit,
                deadline=deadline,
            )

        _remaining_seconds(deadline, label="pilot worker completion")

    return SupervisorWorkerResult(
        outer_fold=identity.outer_fold,
        producer_result_sha256=hashlib.sha256(producer_result_payload).hexdigest(),
        score_release_sha256=release_sha256,
        producer_reinference_sha256=aggregate_sha256,
    )


def scheduler_identity_from_environment(
    contract: NativeDiffusionV1PilotContract,
    *,
    environment: Mapping[str, str] | None = None,
    _test_only_gpu_query: Callable[[str], tuple[str, str, int]] | None = None,
) -> SchedulerIdentity:
    """Derive the fold solely from exact Slurm and sole-visible-GPU evidence."""

    _validate_contract_surface(contract)
    if environment is None:
        values: Mapping[str, str] = os.environ
    else:
        if not isinstance(environment, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in environment.items()
        ):
            raise TypeError("environment must be a string mapping")
        values = environment
    if _test_only_gpu_query is not None and not callable(_test_only_gpu_query):
        raise TypeError("_test_only_gpu_query must be callable or None")
    if any(name.startswith("SLURM_ARRAY_") for name in values):
        raise RuntimeError("pilot execution forbids Slurm arrays")

    job_id = _required_environment(values, "SLURM_JOB_ID")
    if _JOB_RE.fullmatch(job_id) is None:
        raise ValueError("SLURM_JOB_ID is not one positive decimal job ID")
    exact_integers = {
        "SLURM_STEP_ID": 0,
        "SLURM_JOB_NUM_NODES": 4,
        "SLURM_NTASKS": 4,
        "SLURM_STEP_NUM_NODES": 4,
        "SLURM_STEP_NUM_TASKS": 4,
        "SLURM_CPUS_PER_TASK": 8,
        "SLURM_MEM_PER_NODE": 32 * 1024,
        "SLURM_LOCALID": 0,
    }
    parsed = {
        name: _decimal_environment(values, name, expected=expected)
        for name, expected in exact_integers.items()
    }
    outer_fold = _decimal_environment(values, "SLURM_PROCID")
    node_id = _decimal_environment(values, "SLURM_NODEID")
    if outer_fold not in _FOLDS or node_id != outer_fold:
        raise RuntimeError("Slurm rank/node mapping is not exact fold 0..3 block order")
    if _required_environment(values, "SLURM_JOB_ACCOUNT") != "bio":
        raise RuntimeError("pilot allocation must use account bio")
    if _required_environment(values, "SLURM_JOB_PARTITION") != "gpumid":
        raise RuntimeError("pilot allocation must use partition gpumid")
    tres = _required_environment(values, "SLURM_TRES_PER_TASK")
    if tres != "cpu=8,gres/gpu=1":
        raise RuntimeError("Slurm TRES does not bind eight CPUs and one GPU per task")

    node_name = _node_name(_required_environment(values, "SLURMD_NODENAME"))
    host = socket.gethostname().split(".", maxsplit=1)[0]
    if host != node_name:
        raise RuntimeError("Slurm node identity differs from the executing hostname")
    visible = _required_environment(values, "CUDA_VISIBLE_DEVICES")
    if _VISIBLE_GPU_RE.fullmatch(visible) is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose one non-MIG GPU")
    query = _query_visible_gpu if _test_only_gpu_query is None else _test_only_gpu_query
    device_uuid, gpu_name, memory_mib = query(visible)
    expected_name = contract.parent_table("environment")["gpu_name"]
    if gpu_name != expected_name or memory_mib != _EXPECTED_GPU_MEMORY_MIB:
        raise RuntimeError("visible GPU is not the exact inherited A100 80GB device")
    if visible.startswith("GPU-") and visible.lower() != device_uuid.lower():
        raise RuntimeError("CUDA visibility UUID differs from nvidia-smi")
    return SchedulerIdentity(
        job_id=job_id,
        step_id=parsed["SLURM_STEP_ID"],
        outer_fold=outer_fold,
        node_name=node_name,
        device_uuid=device_uuid,
        cuda_visible_device=visible,
    )


def coordinate_score_release(
    *,
    contract: NativeDiffusionV1PilotContract,
    layout: PilotRunLayout,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    deadline: float,
) -> ScoreReleaseReceipt:
    """Rank-0 reauthenticate all trainers and publish the sole score release."""

    _remaining_seconds(deadline, label="score-release coordination")
    _validate_contract_surface(contract)
    commit = _git_commit(expected_git_commit)
    if type(repository) is not RepositorySnapshot or repository.git_commit != commit:
        raise TypeError("repository must be the exact expected RepositorySnapshot")
    receipt_bytes: dict[str, bytes] = {}
    observations: dict[str, TrainerReadinessObservation] = {}
    for fold in _FOLDS:
        _remaining_seconds(deadline, label=f"score-release fold {fold} authentication")
        key = str(fold)
        payload = _read_regular_bytes(
            layout.readiness / f"{fold}.json",
            maximum_bytes=_MAX_CONTROL_BYTES,
            label=f"readiness receipt {fold}",
            required_mode=0o444,
        )
        receipt = parse_trainer_readiness_receipt(payload, contract=contract)
        if receipt.outer_fold != fold or receipt.git_commit != commit:
            raise ValueError("readiness receipt occupies the wrong fold or commit")
        trainer = authenticate_trainer_bundle(
            contract,
            layout.trainers / key,
            expected_outer_fold=fold,
            expected_code_sha256sums=repository.code_sha256sums,
            expected_tree_sha256=receipt.trainer_bundle_sha256,
        )
        _remaining_seconds(deadline, label=f"score-release fold {fold} receipt validation")
        observation = trainer.make_readiness_observation(
            trusted_git_commit=commit,
            node_name=receipt.node_name,
            device_uuid=receipt.device_uuid,
        )
        verify_trainer_readiness_receipt(
            receipt,
            contract=contract,
            observation=observation,
        )
        receipt_bytes[key] = payload
        observations[key] = observation
        _remaining_seconds(deadline, label=f"score-release fold {fold} completion")
    _remaining_seconds(deadline, label="score-release publication")
    release = build_score_release_receipt(
        contract=contract,
        git_commit=commit,
        readiness_receipt_bytes_by_fold=receipt_bytes,
        observations_by_fold=observations,
    )
    publish_score_release_receipt(
        layout.release,
        release,
        contract=contract,
        git_commit=commit,
        readiness_receipt_bytes_by_fold=receipt_bytes,
        observations_by_fold=observations,
    )
    reopened = parse_score_release_receipt(
        _read_regular_bytes(
            layout.release,
            maximum_bytes=_MAX_CONTROL_BYTES,
            label="score release",
            required_mode=0o444,
        ),
        contract=contract,
    )
    if reopened != release or reopened.canonical_bytes() != release.canonical_bytes():
        raise RuntimeError("published score release differs after reopening")
    _seal_control_directory(layout.readiness, expected_files={f"{fold}.json" for fold in _FOLDS})
    _remaining_seconds(deadline, label="score-release completion")
    return release


def parse_evaluator_result(payload: bytes) -> ParsedEvaluatorResult:
    """Parse exact canonical path-free evaluator stdout."""

    document = _canonical_object(payload, fields=_EVALUATOR_RESULT_FIELDS, label="evaluator result")
    checkpoint_raw = _exact_object(
        document["checkpoint_digest_by_step"],
        fields=_CHECKPOINT_KEYS,
        label="evaluator checkpoint map",
    )
    reinference_raw = _exact_object(
        document["reinference_by_step"],
        fields=_CHECKPOINT_KEYS,
        label="evaluator reinference map",
    )
    checkpoints = {
        key: CheckpointDigest.from_document(
            checkpoint_raw[key],
            label=f"evaluator checkpoint map.{key}",
        )
        for key in _CHECKPOINT_KEYS
    }
    reinference: dict[str, ReinferenceComparison] = {}
    comparison_fields = (
        "archived_residual_logit_slice_sha256",
        "byte_equal",
        "reinferred_residual_logit_slice_sha256",
    )
    for key in _CHECKPOINT_KEYS:
        value = _exact_object(
            reinference_raw[key],
            fields=comparison_fields,
            label=f"evaluator reinference map.{key}",
        )
        reinference[key] = ReinferenceComparison(
            archived_residual_logit_slice_sha256=cast(
                str,
                value["archived_residual_logit_slice_sha256"],
            ),
            reinferred_residual_logit_slice_sha256=cast(
                str,
                value["reinferred_residual_logit_slice_sha256"],
            ),
            byte_equal=cast(bool, value["byte_equal"]),
        )
    result = ParsedEvaluatorResult(
        evaluator_bundle_sha256=cast(str, document["evaluator_bundle_sha256"]),
        outer_fold=cast(int, document["outer_fold"]),
        trainer_bundle_sha256=cast(str, document["trainer_bundle_sha256"]),
        count_prior_file_sha256=cast(str, document["count_prior_file_sha256"]),
        checkpoint_digest_by_step=checkpoints,
        reinference_by_step=reinference,
    )
    if canonical_json_bytes(result.document()) != payload:
        raise ValueError("evaluator result differs from its typed canonical reconstruction")
    validate_path_free_document(result.document(), label="evaluator result")
    return result


def build_producer_reinference_document(
    results_by_fold: Mapping[str, ParsedEvaluatorResult],
) -> dict[str, object]:
    """Build the verifier's exact four-fold/five-step byte-equality input."""

    if not isinstance(results_by_fold, Mapping) or set(results_by_fold) != set(_FOLD_KEYS):
        raise ValueError("producer results must contain exact fold keys 0..3")
    by_fold: dict[str, dict[str, dict[str, object]]] = {}
    for key in _FOLD_KEYS:
        result = results_by_fold[key]
        if type(result) is not ParsedEvaluatorResult or result.outer_fold != int(key):
            raise TypeError("producer result is missing or stored under the wrong fold key")
        by_fold[key] = {
            step: result.reinference_by_step[step].canonical_record() for step in _CHECKPOINT_KEYS
        }
    document: dict[str, object] = {
        "all_fold_step_pairs_byte_equal": True,
        "by_fold_and_step": by_fold,
    }
    canonical_json_bytes(document)
    validate_path_free_document(document, label="producer reinference aggregate")
    return document


def build_worker_telemetry_document(
    *,
    contract: NativeDiffusionV1PilotContract,
    identity: SchedulerIdentity,
    expected_git_commit: str,
    trainer: ChildCudaTelemetry,
    evaluator: ChildCudaTelemetry,
    evaluator_progress: EvaluatorProgressTrace,
) -> dict[str, object]:
    """Build one path-free operational record across both CUDA child phases."""

    _validate_contract_surface(contract)
    if type(identity) is not SchedulerIdentity:
        raise TypeError("identity must be an exact SchedulerIdentity")
    if type(trainer) is not ChildCudaTelemetry or type(evaluator) is not ChildCudaTelemetry:
        raise TypeError("worker phases must be exact ChildCudaTelemetry objects")
    if type(evaluator_progress) is not EvaluatorProgressTrace:
        raise TypeError("worker progress must be an exact EvaluatorProgressTrace")
    commit = _git_commit(expected_git_commit)
    if trainer.role != "trainer" or evaluator.role != "evaluator":
        raise ValueError("worker telemetry phases are not trainer then evaluator")
    if trainer.process_pid == evaluator.process_pid:
        raise ValueError("trainer and evaluator must be fresh distinct child processes")
    if (
        not evaluator_progress.complete
        or evaluator_progress.phases != EVALUATOR_PROGRESS_PHASES
        or evaluator_progress.supervisor_process_pid != os.getpid()
        or evaluator_progress.evaluator_process_pid != evaluator.process_pid
        or evaluator_progress.terminal_sha256 is None
    ):
        raise ValueError("worker progress is not the complete PID-bound evaluator trace")
    maximum_allocated = max(
        trainer.torch_cuda_max_memory_allocated,
        evaluator.torch_cuda_max_memory_allocated,
    )
    maximum_reserved = max(
        trainer.torch_cuda_max_memory_reserved,
        evaluator.torch_cuda_max_memory_reserved,
    )
    cap = cast(int, contract.table("resources")["maximum_peak_allocated_memory_bytes"])
    if maximum_allocated > cap:
        raise RuntimeError("worker exceeded the inherited peak allocated-memory gate")
    phase_values = (trainer, evaluator)
    samples = {
        item.role: [
            {
                "cuda_device_uuid": identity.device_uuid,
                "process_pid": item.process_pid,
                "used_memory_bytes": value,
            }
            for value in item.nvidia_smi_process_memory_samples
        ]
        for item in phase_values
    }
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_worker_operational_telemetry",
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": commit,
        "producer_job_id": identity.job_id,
        "outer_fold": identity.outer_fold,
        "node_name": identity.node_name,
        "cuda_device_uuid": identity.device_uuid,
        "process_pid": os.getpid(),
        "child_process_pid_by_phase": {item.role: item.process_pid for item in phase_values},
        "evaluator_progress": {
            "complete": True,
            "marker_count": len(evaluator_progress.phases),
            "terminal_marker_sha256": evaluator_progress.terminal_sha256,
        },
        "torch_cuda_max_memory_allocated": maximum_allocated,
        "torch_cuda_max_memory_reserved": maximum_reserved,
        "torch_cuda_by_phase": {
            item.role: {
                "max_memory_allocated": item.torch_cuda_max_memory_allocated,
                "max_memory_reserved": item.torch_cuda_max_memory_reserved,
            }
            for item in phase_values
        },
        "in_allocation_nvidia_smi_process_memory_samples": samples,
        "slurm_step_accounting": {
            "account": "bio",
            "partition": "gpumid",
            "job_id": identity.job_id,
            "step_id": identity.step_id,
            "task_rank": identity.outer_fold,
            "node_id": identity.outer_fold,
            "nodes": 4,
            "tasks": 4,
            "tasks_per_node": 1,
            "cpus_per_task": 8,
            "memory_per_node_mib": 32 * 1024,
            "gpus_per_task": 1,
        },
        "checks": {
            "allocator_peaks_bound_to_phase_child_pids": True,
            "allocator_peak_within_16_gib": True,
            "distinct_phase_processes": True,
            "evaluator_progress_complete_and_pid_bound": True,
            "external_process_memory_sampled_in_both_phases": True,
            "sole_cuda_device_bound": True,
            "supervisor_process_distinct_from_phase_children": True,
        },
    }
    _validate_worker_telemetry_document(
        document,
        contract=contract,
        expected_outer_fold=identity.outer_fold,
        expected_git_commit=commit,
        expected_producer_job_id=identity.job_id,
        expected_node_name=identity.node_name,
        expected_device_uuid=identity.device_uuid,
    )
    validate_path_free_document(document, label="worker operational telemetry")
    return document


def _enforce_child_allocated_memory_gate(
    contract: NativeDiffusionV1PilotContract,
    telemetry: ChildCudaTelemetry,
) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("memory gate requires the exact pilot contract")
    if type(telemetry) is not ChildCudaTelemetry:
        raise TypeError("memory gate requires exact child CUDA telemetry")
    cap = cast(int, contract.table("resources")["maximum_peak_allocated_memory_bytes"])
    if telemetry.torch_cuda_max_memory_allocated > cap:
        raise RuntimeError(f"{telemetry.role} exceeded the inherited allocated-memory gate")


def build_operational_telemetry_aggregate(
    *,
    contract: NativeDiffusionV1PilotContract,
    payloads_by_fold: Mapping[str, bytes],
    expected_git_commit: str,
    expected_producer_job_id: str,
    expected_node_and_device_by_fold: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    """Authenticate four worker records and bind their exact bytes and maxima."""

    _validate_contract_surface(contract)
    commit = _git_commit(expected_git_commit)
    producer_job_id = _job_id(expected_producer_job_id)
    if not isinstance(payloads_by_fold, Mapping) or set(payloads_by_fold) != set(_FOLD_KEYS):
        raise ValueError("worker telemetry payloads require exact fold keys 0..3")
    if not isinstance(expected_node_and_device_by_fold, Mapping) or set(
        expected_node_and_device_by_fold
    ) != set(_FOLD_KEYS):
        raise ValueError("worker telemetry identities require exact fold keys 0..3")
    summaries: dict[str, dict[str, object]] = {}
    nodes: list[str] = []
    devices: list[str] = []
    job_ids: list[str] = []
    maximum_allocated = 0
    maximum_reserved = 0
    for key in _FOLD_KEYS:
        payload = payloads_by_fold[key]
        if type(payload) is not bytes:
            raise TypeError("worker telemetry payload must be exact bytes")
        expected_identity = expected_node_and_device_by_fold[key]
        if (
            type(expected_identity) is not tuple
            or len(expected_identity) != 2
            or any(type(value) is not str for value in expected_identity)
        ):
            raise TypeError("worker telemetry expected identity must be a string pair")
        document = _canonical_object(
            payload,
            fields=_WORKER_TELEMETRY_FIELDS,
            label=f"worker operational telemetry {key}",
        )
        _validate_worker_telemetry_document(
            document,
            contract=contract,
            expected_outer_fold=int(key),
            expected_git_commit=commit,
            expected_producer_job_id=producer_job_id,
            expected_node_name=expected_identity[0],
            expected_device_uuid=expected_identity[1],
        )
        nodes.append(cast(str, document["node_name"]))
        devices.append(cast(str, document["cuda_device_uuid"]))
        job_ids.append(cast(str, document["producer_job_id"]))
        observed_allocated = cast(int, document["torch_cuda_max_memory_allocated"])
        maximum_allocated = max(maximum_allocated, observed_allocated)
        maximum_reserved = max(
            maximum_reserved,
            cast(int, document["torch_cuda_max_memory_reserved"]),
        )
        summaries[key] = {
            "cuda_device_uuid": document["cuda_device_uuid"],
            "evaluator_progress_terminal_sha256": cast(
                dict[str, object],
                document["evaluator_progress"],
            )["terminal_marker_sha256"],
            "node_name": document["node_name"],
            "outer_fold": int(key),
            "telemetry_record_sha256": hashlib.sha256(payload).hexdigest(),
            "torch_cuda_max_memory_allocated": observed_allocated,
            "torch_cuda_max_memory_reserved": document["torch_cuda_max_memory_reserved"],
        }
    if len(set(nodes)) != 4 or len(set(devices)) != 4 or len(set(job_ids)) != 1:
        raise ValueError(
            "worker telemetry requires one job and four distinct nodes and CUDA devices"
        )
    aggregate: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_operational_telemetry_aggregate",
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": commit,
        "producer_job_id": producer_job_id,
        "all_workers_within_allocated_memory_cap": True,
        "maximum_torch_cuda_memory_allocated": maximum_allocated,
        "maximum_torch_cuda_memory_reserved": maximum_reserved,
        "by_fold": summaries,
    }
    canonical_json_bytes(aggregate)
    validate_path_free_document(aggregate, label="operational telemetry aggregate")
    return aggregate


def coordinate_producer_reinference(
    *,
    contract: NativeDiffusionV1PilotContract,
    layout: PilotRunLayout,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    deadline: float,
) -> bytes:
    """Reopen all fold artifacts and atomically publish operational equality."""

    _remaining_seconds(deadline, label="producer reinference aggregation")
    _validate_contract_surface(contract)
    commit = _git_commit(expected_git_commit)
    if type(repository) is not RepositorySnapshot or repository.git_commit != commit:
        raise TypeError("repository must be the exact expected RepositorySnapshot")
    release_payload = _read_regular_bytes(
        layout.release,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="score release",
        required_mode=0o444,
    )
    release = parse_score_release_receipt(release_payload, contract=contract)
    if release.git_commit != commit:
        raise ValueError("score release differs from aggregate commit")
    results: dict[str, ParsedEvaluatorResult] = {}
    for fold in _FOLDS:
        _remaining_seconds(deadline, label=f"producer reinference fold {fold} authentication")
        key = str(fold)
        readiness_payload = _read_regular_bytes(
            layout.readiness / f"{fold}.json",
            maximum_bytes=_MAX_CONTROL_BYTES,
            label=f"readiness receipt {fold}",
            required_mode=0o444,
        )
        if (
            hashlib.sha256(readiness_payload).hexdigest()
            != (release.readiness_receipt_sha256_by_fold[key])
        ):
            raise ValueError("score release no longer binds a fold readiness receipt")
        readiness = parse_trainer_readiness_receipt(readiness_payload, contract=contract)
        if readiness.outer_fold != fold or readiness.git_commit != commit:
            raise ValueError("readiness receipt is under the wrong fold or commit")
        trainer = authenticate_trainer_bundle(
            contract,
            layout.trainers / key,
            expected_outer_fold=fold,
            expected_code_sha256sums=repository.code_sha256sums,
            expected_tree_sha256=readiness.trainer_bundle_sha256,
        )
        _remaining_seconds(deadline, label=f"producer reinference fold {fold} validation")
        result_payload = _read_regular_bytes(
            layout.producer_results / f"{fold}.json",
            maximum_bytes=_MAX_CONTROL_BYTES,
            label=f"evaluator producer result {fold}",
            required_mode=0o444,
        )
        result = parse_evaluator_result(result_payload)
        _validate_evaluator_result(
            result,
            contract=contract,
            trainer=trainer,
            layout=layout,
            repository=repository,
            expected_git_commit=commit,
            outer_fold=fold,
            readiness_receipt_sha256=hashlib.sha256(readiness_payload).hexdigest(),
            score_release_sha256=hashlib.sha256(release_payload).hexdigest(),
        )
        results[key] = result
        _remaining_seconds(deadline, label=f"producer reinference fold {fold} completion")
    payload = canonical_json_bytes(build_producer_reinference_document(results))
    _publish_canonical_record(layout.producer_reinference, payload)
    reopened = _read_regular_bytes(
        layout.producer_reinference,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="producer reinference aggregate",
        required_mode=0o444,
    )
    if reopened != payload:
        raise RuntimeError("producer reinference aggregate changed after publication")
    expected = {f"{fold}.json" for fold in _FOLDS}
    _seal_control_directory(layout.producer_results, expected_files=expected)
    _remaining_seconds(deadline, label="producer reinference publication")
    return payload


def coordinate_operational_telemetry(
    *,
    contract: NativeDiffusionV1PilotContract,
    layout: PilotRunLayout,
    expected_git_commit: str,
    deadline: float,
) -> bytes:
    """Rank 0 authenticate, aggregate, and seal all four worker telemetry records."""

    _remaining_seconds(deadline, label="operational telemetry aggregation")
    _validate_contract_surface(contract)
    commit = _git_commit(expected_git_commit)
    release_payload = _read_regular_bytes(
        layout.release,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="operational telemetry score release",
        required_mode=0o444,
    )
    release = parse_score_release_receipt(release_payload, contract=contract)
    if release.git_commit != commit:
        raise ValueError("operational telemetry score release has the wrong commit")
    release_sha256 = hashlib.sha256(release_payload).hexdigest()
    payloads: dict[str, bytes] = {}
    identities: dict[str, tuple[str, str]] = {}
    for fold in _FOLDS:
        _remaining_seconds(deadline, label=f"operational telemetry fold {fold} input")
        key = str(fold)
        readiness = parse_trainer_readiness_receipt(
            _read_regular_bytes(
                layout.readiness / f"{fold}.json",
                maximum_bytes=_MAX_CONTROL_BYTES,
                label=f"telemetry readiness receipt {fold}",
                required_mode=0o444,
            ),
            contract=contract,
        )
        if readiness.outer_fold != fold or readiness.git_commit != commit:
            raise ValueError("telemetry readiness receipt is under the wrong fold or commit")
        identities[key] = (readiness.node_name, readiness.device_uuid)
        payloads[key] = _read_regular_bytes(
            layout.operational_telemetry / f"{fold}.json",
            maximum_bytes=_MAX_CONTROL_BYTES,
            label=f"worker operational telemetry {fold}",
            required_mode=0o444,
        )
    aggregate = build_operational_telemetry_aggregate(
        contract=contract,
        payloads_by_fold=payloads,
        expected_git_commit=commit,
        expected_producer_job_id=_job_id(layout.root.name),
        expected_node_and_device_by_fold=identities,
    )
    for key in _FOLD_KEYS:
        _remaining_seconds(deadline, label=f"operational telemetry fold {key} progress")
        document = _canonical_object(
            payloads[key],
            fields=_WORKER_TELEMETRY_FIELDS,
            label=f"worker operational telemetry {key}",
        )
        child_pids = _exact_object(
            document["child_process_pid_by_phase"],
            fields=("evaluator", "trainer"),
            label=f"worker telemetry child PID map {key}",
        )
        progress_identity = _exact_object(
            document["evaluator_progress"],
            fields=("complete", "marker_count", "terminal_marker_sha256"),
            label=f"worker evaluator progress {key}",
        )
        verify_evaluator_progress(
            layout.evaluator_progress / key,
            expected_child_contract_sha256=contract.config_sha256,
            expected_parent_contract_sha256=contract.parent_config_sha256,
            expected_git_commit=commit,
            expected_outer_fold=int(key),
            expected_score_release_sha256=release_sha256,
            expected_supervisor_process_pid=cast(int, document["process_pid"]),
            expected_evaluator_process_pid=cast(int, child_pids["evaluator"]),
            expected_terminal_sha256=cast(
                str,
                progress_identity["terminal_marker_sha256"],
            ),
            require_complete=True,
        )
        _remaining_seconds(deadline, label=f"operational telemetry fold {key} completion")
    _remaining_seconds(deadline, label="operational telemetry publication")
    payload = canonical_json_bytes(aggregate)
    _publish_canonical_record(layout.operational_telemetry_aggregate, payload)
    reopened = _read_regular_bytes(
        layout.operational_telemetry_aggregate,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="operational telemetry aggregate",
        required_mode=0o444,
    )
    if reopened != payload:
        raise RuntimeError("operational telemetry aggregate changed after publication")
    _seal_evaluator_progress_root(layout.evaluator_progress)
    _seal_control_directory(
        layout.operational_telemetry,
        expected_files={f"{fold}.json" for fold in _FOLDS},
    )
    _remaining_seconds(deadline, label="operational telemetry publication")
    return payload


def _validate_contract_surface(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    resources = contract.table("resources")
    expected_resources: dict[str, object] = {
        "account": "bio",
        "partition": "gpumid",
        "gpu_type": "A100_80GB",
        "allocation_nodes": 4,
        "worker_tasks": 4,
        "maximum_concurrent_fit_tasks": 4,
        "nodes_per_fit": 1,
        "tasks_per_node": 1,
        "gpus_per_task": 1,
        "cpus_per_task": 8,
        "memory_gib_per_task": 32,
        "wall_minutes_per_fit": 60,
        "maximum_pilot_a100_hours": 4.0,
        "maximum_peak_allocated_memory_gib": 16.0,
        "maximum_total_allocated_gpu_seconds": 14_400,
        "maximum_peak_allocated_memory_bytes": 17_179_869_184,
        "bare_exclusive_allowed": False,
        "execution_command": "uv run --locked --no-sync",
        "run_subdir": "diffusion/native-categorical-unconditional-v1/pilot-executions",
        "require_clean_synchronized_commit": True,
    }
    for name, expected in expected_resources.items():
        value = resources.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"authenticated resources.{name} differs from supervisor")
    if contract.checkpoint_steps != _CHECKPOINT_STEPS:
        raise ValueError("authenticated checkpoint order differs from supervisor")
    barrier = contract.table("barrier")
    expected_barrier: dict[str, object] = {
        "execution_topology": "single_allocation_one_srun_four_gpu_tasks_on_four_distinct_nodes",
        "training_stage_receives_score_paths": False,
        "score_stage_inputs_materialized_after_release": True,
        "score_stage_opens_only_own_outer_fold": True,
        "score_stage_starts_after_release": True,
        "required_readiness_receipts": 4,
    }
    for name, expected in expected_barrier.items():
        value = barrier.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"authenticated barrier.{name} differs from supervisor")
    audit = contract.table("audit")
    expected_audit = {
        "account": "bio",
        "partition": "standard",
        "nodes": 1,
        "tasks": 1,
        "gpus": 0,
        "cpus_per_task": 8,
        "memory_gib_per_task": 32,
        "wall_minutes": 120,
        "third_node_required": True,
        "excluded_nodes": "all_four_gpu_producer_nodes",
    }
    for name, expected in expected_audit.items():
        value = audit.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"authenticated audit.{name} differs from supervisor")
    telemetry = contract.parent_table("telemetry")
    if telemetry.get("required_per_worker") != (
        "torch_cuda_max_memory_allocated",
        "torch_cuda_max_memory_reserved",
        "cuda_device_uuid",
        "process_pid",
        "in_allocation_nvidia_smi_process_memory_samples",
        "slurm_step_accounting",
    ):
        raise ValueError("inherited per-worker telemetry surface differs from supervisor")
    if (
        telemetry.get("allocated_memory_is_frozen_gate") is not True
        or telemetry.get("reserved_memory_is_report_only") is not True
        or telemetry.get("nvidia_smi_process_memory_is_report_only") is not True
        or telemetry.get("slurm_epilog_memory_is_report_only") is not True
    ):
        raise ValueError("inherited telemetry gate/report-only policy differs")


def _open_run_layout(
    contract: NativeDiffusionV1PilotContract,
    *,
    run_root: str | os.PathLike[str],
    job_id: str,
) -> PilotRunLayout:
    user = pwd.getpwuid(os.getuid()).pw_name
    scratch = Path("/lustre/scratch/users") / user / "amp_challenge"
    expected = scratch / cast(str, contract.table("resources")["run_subdir"]) / job_id
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root != expected:
        raise ValueError("pilot run root differs from exact current-user scratch job path")
    directories = {
        "root": root,
        "trainers": root / "trainers",
        "evaluators": root / "evaluators",
        "control": root / "control",
        "readiness": root / "control/checkpoint-ready",
        "producer_results": root / "control/producer-results",
        "operational_telemetry": root / "control/operational-telemetry",
        "evaluator_progress": root / "control/evaluator-progress",
    }
    for label, path in directories.items():
        _validate_private_directory(path, label=f"pilot {label} directory")
    return PilotRunLayout(
        **directories,
        release=root / "control/score-release.json",
        producer_reinference=root / "control/producer-reinference.json",
        operational_telemetry_aggregate=root / "control/operational-telemetry.json",
    )


def _open_projection_root(
    contract: NativeDiffusionV1PilotContract,
    *,
    projection_root: str | os.PathLike[str],
) -> Path:
    user = pwd.getpwuid(os.getuid()).pw_name
    scratch = Path("/lustre/scratch/users") / user / "amp_challenge"
    relative = cast(str, contract.table("projection")["canonical_bundle_relative_path"])
    root = Path(os.path.abspath(os.fspath(projection_root)))
    if root != scratch / PurePosixPath(relative):
        raise ValueError("projection root differs from the accepted canonical scratch bundle")
    _validate_sealed_directory(root, label="accepted projection root")
    top = _read_regular_bytes(
        root / "SHA256SUMS",
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="accepted projection top checksum",
        required_mode=0o444,
    )
    if hashlib.sha256(top).hexdigest() != contract.projection_top_sha256:
        raise ValueError("accepted projection top checksum changed")
    return root


def _node_local_root(identity: SchedulerIdentity) -> Path:
    raw = os.environ.get("TMPDIR")
    if not raw:
        raise ValueError("TMPDIR must be the worker's private node-local root")
    root = Path(os.path.abspath(raw))
    _validate_private_directory(root, label="node-local worker root")
    if root.is_relative_to(Path("/lustre")) or root.is_relative_to(Path("/home")):
        raise ValueError("worker TMPDIR must be node-local, not shared home or Lustre")
    expected_name = f"amp-native-diffusion-v1-{identity.job_id}-{identity.outer_fold}"
    if root.name != expected_name:
        raise ValueError("node-local worker root does not bind job and fold")
    cache = root / "pycache"
    _validate_private_directory(cache, label="node-local Python cache")
    return root


def _read_projection_role(
    contract: NativeDiffusionV1PilotContract,
    projection_root: Path,
    *,
    outer_fold: int,
    role: str,
) -> bytes:
    fold = contract.fold(_outer_fold(outer_fold))
    if role == "train":
        relative, expected = fold.train_path, fold.train_sha256
    elif role == "score":
        relative, expected = fold.score_path, fold.score_sha256
    else:
        raise ValueError("projection role must be exactly train or score")
    source = projection_root.joinpath(*PurePosixPath(relative).parts)
    payload = _read_regular_bytes(
        source,
        maximum_bytes=_MAX_PROJECTION_BYTES,
        label=f"fold {outer_fold} {role} projection",
        required_mode=0o444,
    )
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"fold {outer_fold} {role} projection differs from its contract pin")
    return payload


@contextmanager
def _private_staged_input(
    node_root: Path,
    *,
    role: str,
    payload: bytes,
):
    if role not in {"train", "score"}:
        raise ValueError("private staged role must be train or score")
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_PROJECTION_BYTES:
        raise ValueError("private staged payload must be non-empty bounded bytes")
    directory = node_root / f"{role}-input"
    if os.path.lexists(directory):
        raise FileExistsError(f"private {role} directory is not fresh")
    os.mkdir(directory, 0o700)
    destination = directory / f"{role}.jsonl"
    descriptor = -1
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short private projection write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if tuple(path.name for path in directory.iterdir()) != (f"{role}.jsonl",):
            raise RuntimeError("private projection directory gained another entry")
        reopened = _read_regular_bytes(
            destination,
            maximum_bytes=_MAX_PROJECTION_BYTES,
            label=f"private {role} projection",
            required_mode=0o400,
        )
        if reopened != payload:
            raise RuntimeError("private projection copy differs from source bytes")
        yield destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _remove_private_input(directory, expected_name=f"{role}.jsonl")


def _remove_private_input(directory: Path, *, expected_name: str) -> None:
    if not os.path.lexists(directory):
        return
    metadata = os.lstat(directory)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("refusing to clean a non-directory private input")
    entries = tuple(directory.iterdir())
    if len(entries) > 1 or any(path.name != expected_name for path in entries):
        raise RuntimeError("refusing to clean an unexpected private-input inventory")
    for path in entries:
        observed = os.lstat(path)
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise RuntimeError("refusing to clean an unsafe private-input entry")
        os.chmod(path, 0o600)
        os.unlink(path)
    os.rmdir(directory)


def _child_environment(node_root: Path) -> dict[str, str]:
    """Return the fixed trainer/evaluator environment without score authority."""

    environment = {
        name: os.environ[name] for name in _CHILD_PASSTHROUGH_ENVIRONMENT if name in os.environ
    }
    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LC_ALL": "C",
            "MKL_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": os.fspath(node_root / "pycache"),
            "PYTHONSAFEPATH": "1",
            "PYTORCH_ALLOC_CONF": "backend:native",
            "TMPDIR": os.fspath(node_root),
            "TZ": "UTC",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    if not environment.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("child environment lost the sole visible GPU")
    forbidden = sorted(
        name
        for name in environment
        if name.startswith(("AMP_", "GIT_", "SLURM_"))
        or name
        in {
            "CONDA_PREFIX",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTORCH_CUDA_ALLOC_CONF",
            "VIRTUAL_ENV",
        }
    )
    if forbidden or any("development-projections" in value for value in environment.values()):
        raise RuntimeError("child environment retained projection or repository authority")
    return environment


def _trainer_command(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    train_path: Path,
    output_dir: Path,
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
    outer_fold: int,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "amp_challenge.generators.diffusion.v1.pilot_trainer",
        "--child-contract",
        os.fspath(child_contract_path),
        "--parent-contract",
        os.fspath(parent_contract_path),
        "--train-jsonl",
        os.fspath(train_path),
        "--output-dir",
        os.fspath(output_dir),
        "--outer-fold",
        str(_outer_fold(outer_fold)),
        "--repository-root",
        os.fspath(repository_root),
        "--expected-git-commit",
        _git_commit(expected_git_commit),
    )


def _evaluator_command(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    score_path: Path,
    trainer_bundle: Path,
    readiness_receipt: Path,
    score_release: Path,
    progress_dir: Path,
    trace_monotonic_origin_ns: int,
    supervisor_process_pid: int,
    output_dir: Path,
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
    outer_fold: int,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "amp_challenge.generators.diffusion.v1.pilot_evaluator",
        "--child-contract",
        os.fspath(child_contract_path),
        "--parent-contract",
        os.fspath(parent_contract_path),
        "--score-jsonl",
        os.fspath(score_path),
        "--trainer-bundle",
        os.fspath(trainer_bundle),
        "--readiness-receipt",
        os.fspath(readiness_receipt),
        "--score-release",
        os.fspath(score_release),
        "--progress-dir",
        os.fspath(progress_dir),
        "--trace-monotonic-origin-ns",
        str(_nonnegative_integer(trace_monotonic_origin_ns, label="trace monotonic origin")),
        "--supervisor-process-pid",
        str(_positive_pid(supervisor_process_pid, label="supervisor process PID")),
        "--output-dir",
        os.fspath(output_dir),
        "--outer-fold",
        str(_outer_fold(outer_fold)),
        "--repository-root",
        os.fspath(repository_root),
        "--expected-git-commit",
        _git_commit(expected_git_commit),
    )


def _run_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    deadline: float,
    label: str,
    role: str,
    cuda_visible_device: str,
    expected_device_uuid: str,
) -> tuple[bytes, ChildCudaTelemetry]:
    values = tuple(command)
    if not values or any(type(value) is not str or not value for value in values):
        raise ValueError("child command must be a non-empty exact string sequence")
    expected_module_by_role = {
        "trainer": "amp_challenge.generators.diffusion.v1.pilot_trainer",
        "evaluator": "amp_challenge.generators.diffusion.v1.pilot_evaluator",
    }
    if (
        role not in expected_module_by_role
        or len(values) < 4
        or values[:2] != (sys.executable, "-m")
        or values[2] != expected_module_by_role[role]
    ):
        raise ValueError("instrumented child command differs from its exact role module")
    if _VISIBLE_GPU_RE.fullmatch(cuda_visible_device) is None:
        raise ValueError("instrumented child requires one safe visible GPU selector")
    if _GPU_UUID_RE.fullmatch(expected_device_uuid) is None:
        raise ValueError("instrumented child requires one canonical CUDA device UUID")
    if not isinstance(environment, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in environment.items()
    ):
        raise TypeError("child environment must be a string mapping")
    _remaining_seconds(deadline, label=label)
    read_descriptor, write_descriptor = os.pipe()
    os.set_blocking(read_descriptor, False)
    instrumented = (
        sys.executable,
        "-c",
        _INSTRUMENTED_CHILD_SOURCE,
        role,
        str(write_descriptor),
        *values[3:],
    )
    process: subprocess.Popen[bytes] | None = None
    samples: list[int] = []
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SUPERVISOR_SIGNALS)
        try:
            process = subprocess.Popen(
                instrumented,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=None,
                env=dict(environment),
                close_fds=True,
                start_new_session=True,
                pass_fds=(write_descriptor,),
            )
            os.close(write_descriptor)
            write_descriptor = -1
        finally:
            # Ownership is established before pending signals are unblocked,
            # so an immediately raised handler is caught by this outer try.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        while True:
            remaining = _remaining_seconds(deadline, label=label)
            try:
                stdout, _ = process.communicate(timeout=min(_POLL_SECONDS, remaining))
                break
            except subprocess.TimeoutExpired:
                if len(samples) < _MAX_TELEMETRY_SAMPLES_PER_PHASE:
                    observed = _query_process_gpu_memory(
                        cuda_visible_device,
                        process_pid=process.pid,
                        expected_device_uuid=expected_device_uuid,
                        timeout_seconds=min(
                            10.0,
                            _remaining_seconds(
                                deadline,
                                label=f"{label} nvidia-smi process sampling",
                            ),
                        ),
                    )
                    if observed is not None:
                        samples.append(observed)
    except BaseException:
        if process is not None:
            _terminate_child(process)
        with suppress(OSError):
            os.close(read_descriptor)
        if write_descriptor >= 0:
            with suppress(OSError):
                os.close(write_descriptor)
        raise
    if process is None:
        os.close(read_descriptor)
        raise RuntimeError("instrumented child was not launched")
    if process.returncode != 0:
        os.close(read_descriptor)
        raise RuntimeError(f"{label} exited with status {process.returncode}")
    if type(stdout) is not bytes or not 0 < len(stdout) <= _MAX_CHILD_STDOUT_BYTES:
        os.close(read_descriptor)
        raise RuntimeError(f"{label} produced invalid bounded stdout")
    telemetry_payload = _read_descriptor_bytes(
        read_descriptor,
        maximum_bytes=_MAX_CHILD_TELEMETRY_BYTES,
        label=f"{label} CUDA telemetry",
        deadline=deadline,
    )
    if not samples:
        raise RuntimeError(f"{label} had no in-allocation nvidia-smi process sample")
    telemetry = _parse_child_cuda_telemetry(
        telemetry_payload,
        role=role,
        expected_process_pid=process.pid,
        nvidia_smi_samples=tuple(samples),
    )
    return stdout, telemetry


def _instrumented_child_entry() -> int:
    """Run one exact child CLI and write allocator peaks to an inherited pipe."""

    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SUPERVISOR_SIGNALS)
    if len(sys.argv) < 4:
        raise ValueError("instrumented child invocation is incomplete")
    role = sys.argv[1]
    if role == "trainer":
        module_name = "amp_challenge.generators.diffusion.v1.pilot_trainer"
    elif role == "evaluator":
        module_name = "amp_challenge.generators.diffusion.v1.pilot_evaluator"
    else:
        raise ValueError("instrumented child role must be trainer or evaluator")
    raw_descriptor = sys.argv[2]
    if not raw_descriptor.isascii() or not raw_descriptor.isdecimal():
        raise ValueError("instrumented child telemetry descriptor is malformed")
    descriptor = int(raw_descriptor)
    if descriptor < 3:
        raise ValueError("instrumented child telemetry descriptor is unsafe")
    os.set_inheritable(descriptor, False)
    arguments = tuple(sys.argv[3:])
    if not arguments or any(type(value) is not str or not value for value in arguments):
        raise ValueError("instrumented child arguments are invalid")

    import torch

    device = torch.device("cuda:0")
    module = importlib.import_module(module_name)
    target = getattr(module, "main", None)
    if not callable(target):
        raise RuntimeError("instrumented child target lacks a callable main")
    status = target(arguments)
    if type(status) is not int or status != 0:
        raise RuntimeError("instrumented child target did not return exact success")
    # Each role runs in a fresh process, so its lifetime allocator peaks are
    # exactly its phase peaks.  The target must own first CUDA access so it can
    # attest the contract-bound runtime before CUDA initialization.
    if not torch.cuda.is_initialized():
        raise RuntimeError("instrumented child target did not initialize CUDA")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("instrumented child requires exactly one visible CUDA device")
    torch.cuda.synchronize(device)
    payload = canonical_json_bytes(
        {
            "process_pid": os.getpid(),
            "role": role,
            "torch_cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "torch_cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    )
    _write_descriptor_bytes(descriptor, payload)
    os.close(descriptor)
    return 0


def _parse_child_cuda_telemetry(
    payload: bytes,
    *,
    role: str,
    expected_process_pid: int,
    nvidia_smi_samples: tuple[int, ...],
) -> ChildCudaTelemetry:
    document = _canonical_object(
        payload,
        fields=(
            "process_pid",
            "role",
            "torch_cuda_max_memory_allocated",
            "torch_cuda_max_memory_reserved",
        ),
        label=f"{role} CUDA telemetry",
    )
    if document["role"] != role or document["process_pid"] != expected_process_pid:
        raise ValueError("CUDA child telemetry differs from the launched process")
    return ChildCudaTelemetry(
        role=cast(str, document["role"]),
        process_pid=cast(int, document["process_pid"]),
        torch_cuda_max_memory_allocated=cast(
            int,
            document["torch_cuda_max_memory_allocated"],
        ),
        torch_cuda_max_memory_reserved=cast(
            int,
            document["torch_cuda_max_memory_reserved"],
        ),
        nvidia_smi_process_memory_samples=nvidia_smi_samples,
    )


def _query_process_gpu_memory(
    visible: str,
    *,
    process_pid: int,
    expected_device_uuid: str,
    timeout_seconds: float = 10.0,
) -> int | None:
    if (
        _VISIBLE_GPU_RE.fullmatch(visible) is None
        or type(process_pid) is not int
        or _GPU_UUID_RE.fullmatch(expected_device_uuid) is None
        or type(timeout_seconds) is not float
        or not 0.0 < timeout_seconds <= 10.0
    ):
        raise ValueError("GPU process-memory query identity is malformed")
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={visible}",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0 or len(completed.stdout) > 65_536:
        raise RuntimeError("nvidia-smi process-memory sampling failed")
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("nvidia-smi process-memory output is not ASCII") from error
    matches: list[int] = []
    for line in lines:
        fields = tuple(item.strip() for item in line.split(","))
        if (
            len(fields) != 3
            or _GPU_UUID_RE.fullmatch(fields[0]) is None
            or not fields[1].isascii()
            or not fields[1].isdecimal()
        ):
            raise RuntimeError("nvidia-smi process-memory row is malformed")
        if int(fields[1]) != process_pid:
            continue
        if fields[0].lower() != expected_device_uuid.lower():
            raise RuntimeError("nvidia-smi process sample came from another CUDA device")
        if not fields[2].isascii() or not fields[2].isdecimal():
            raise RuntimeError("nvidia-smi process-memory value is malformed")
        used_mib = int(fields[2])
        if used_mib <= 0:
            raise RuntimeError("nvidia-smi process-memory sample must be positive")
        matches.append(used_mib * 1024 * 1024)
    if len(matches) > 1:
        raise RuntimeError("nvidia-smi returned duplicate rows for one CUDA child")
    return matches[0] if matches else None


def _write_descriptor_bytes(descriptor: int, payload: bytes) -> None:
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("write descriptor is unsafe")
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_CHILD_TELEMETRY_BYTES:
        raise ValueError("descriptor payload is not bounded bytes")
    framed = len(payload).to_bytes(_TELEMETRY_FRAME_LENGTH_BYTES, "big") + payload
    view = memoryview(framed)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short telemetry pipe write")
        written += count


def _read_descriptor_bytes(
    descriptor: int,
    *,
    maximum_bytes: int,
    label: str,
    deadline: float,
) -> bytes:
    if (
        type(descriptor) is not int
        or descriptor < 3
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise ValueError("read descriptor boundary is invalid")
    try:
        header = _read_exact_descriptor_bytes(
            descriptor,
            size=_TELEMETRY_FRAME_LENGTH_BYTES,
            deadline=deadline,
            label=label,
        )
        payload_size = int.from_bytes(header, "big")
        if not 0 < payload_size <= maximum_bytes:
            raise RuntimeError(f"{label} declares an invalid framed payload size")
        payload = _read_exact_descriptor_bytes(
            descriptor,
            size=payload_size,
            deadline=deadline,
            label=label,
        )
        readable, _, _ = select.select((descriptor,), (), (), 0.0)
        if readable:
            try:
                trailing = os.read(descriptor, 1)
            except BlockingIOError:
                trailing = b""
            if trailing:
                raise RuntimeError(f"{label} contains bytes after its exact frame")
    finally:
        os.close(descriptor)
    return payload


def _read_exact_descriptor_bytes(
    descriptor: int,
    *,
    size: int,
    deadline: float,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        remaining = _remaining_seconds(deadline, label=label)
        readable, _, _ = select.select(
            (descriptor,),
            (),
            (),
            min(_POLL_SECONDS, remaining),
        )
        if not readable:
            continue
        try:
            chunk = os.read(descriptor, size - received)
        except BlockingIOError:
            continue
        if not chunk:
            raise RuntimeError(f"{label} pipe closed before its exact frame")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def _validate_trainer_stdout(
    payload: bytes,
    *,
    trainer: AuthenticatedTrainerBundle,
    outer_fold: int,
) -> None:
    document = _canonical_object(payload, fields=_TRAINER_RESULT_FIELDS, label="trainer result")
    expected_checkpoints = {
        key: trainer.checkpoint_digest_by_step[key].document() for key in _CHECKPOINT_KEYS
    }
    expected = {
        "bundle_tree_sha256": trainer.bundle.tree_sha256,
        "checkpoint_digest_by_step": expected_checkpoints,
        "count_prior_file_sha256": trainer.count_prior.sha256,
        "fit_identity_sha256": trainer.fit_identity_sha256,
        "outer_fold": _outer_fold(outer_fold),
    }
    if document != expected:
        raise ValueError("trainer stdout differs from independently authenticated bundle bytes")


def _wait_and_verify_own_release(
    *,
    contract: NativeDiffusionV1PilotContract,
    layout: PilotRunLayout,
    readiness_receipt: TrainerReadinessReceipt,
    expected_git_commit: str,
    deadline: float,
) -> ScoreReleaseReceipt:
    _wait_for_sealed_files((layout.release,), deadline=deadline, label="score release")
    payload = _read_regular_bytes(
        layout.release,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="score release",
        required_mode=0o444,
    )
    release = parse_score_release_receipt(payload, contract=contract)
    if release.git_commit != _git_commit(expected_git_commit):
        raise ValueError("score release differs from the expected commit")
    own_key = str(readiness_receipt.outer_fold)
    if (
        release.readiness_receipt_sha256_by_fold[own_key]
        != hashlib.sha256(readiness_receipt.canonical_bytes()).hexdigest()
    ):
        raise ValueError("score release does not bind this worker's exact readiness bytes")
    return release


def _validate_evaluator_result(
    result: ParsedEvaluatorResult,
    *,
    contract: NativeDiffusionV1PilotContract,
    trainer: AuthenticatedTrainerBundle,
    layout: PilotRunLayout,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    outer_fold: int,
    readiness_receipt_sha256: str,
    score_release_sha256: str,
) -> None:
    fold = _outer_fold(outer_fold)
    commit = _git_commit(expected_git_commit)
    if type(repository) is not RepositorySnapshot or repository.git_commit != commit:
        raise TypeError("evaluator validation requires the exact repository snapshot")
    readiness_sha256 = _sha256(
        readiness_receipt_sha256,
        label="readiness receipt SHA-256",
    )
    release_sha256 = _sha256(
        score_release_sha256,
        label="score-release SHA-256",
    )
    if type(result) is not ParsedEvaluatorResult or result.outer_fold != fold:
        raise ValueError("evaluator result belongs to the wrong fold")
    if (
        result.trainer_bundle_sha256 != trainer.bundle.tree_sha256
        or result.count_prior_file_sha256 != trainer.count_prior.sha256
        or dict(result.checkpoint_digest_by_step) != dict(trainer.checkpoint_digest_by_step)
    ):
        raise ValueError("evaluator result differs from authenticated trainer evidence")
    bundle = verify_bundle(
        contract,
        bundle_kind="evaluator",
        root=layout.evaluators / str(fold),
        expected_tree_sha256=result.evaluator_bundle_sha256,
    )
    if bundle.read_bytes("CODE_SHA256SUMS", maximum_bytes=8 << 20) != repository.code_sha256sums:
        raise ValueError("evaluator CODE_SHA256SUMS differs from the trusted repository")
    frozen = parse_sha256sums(
        bundle.read_bytes("FROZEN_INPUT_SHA256SUMS", maximum_bytes=_MAX_CONTROL_BYTES),
        label="evaluator FROZEN_INPUT_SHA256SUMS",
    )
    expected_frozen = {
        "checkpoint-ready.json": readiness_sha256,
        "count_prior.npz": trainer.count_prior.sha256,
        "pilot_execution_v1.toml": contract.config_sha256,
        "score-release.json": release_sha256,
        "score.jsonl": contract.fold(fold).score_sha256,
        "trainer_bundle.tree.json": trainer.bundle.tree_sha256,
        "unconditional_v1.toml": contract.parent_config_sha256,
    }
    if tuple(frozen) != _EVALUATOR_FROZEN_INPUT_PATHS or frozen != expected_frozen:
        raise ValueError("evaluator bundle lacks the strict seven-input release chain")
    manifest = parse_canonical_json(
        bundle.read_bytes("manifest.json", maximum_bytes=_MAX_CONTROL_BYTES),
        label="evaluator manifest",
    )
    manifest_values = _exact_object(
        manifest,
        fields=cast(Sequence[str], contract.table("outputs")["evaluator_manifest_fields"]),
        label="evaluator manifest",
    )
    if (
        type(manifest_values["schema_version"]) is not int
        or manifest_values["schema_version"] != 1
        or manifest_values["artifact"] != contract.document["artifact"]
        or manifest_values["child_contract_sha256"] != contract.config_sha256
        or manifest_values["parent_contract_sha256"] != contract.parent_config_sha256
        or manifest_values["git_commit"] != commit
        or type(manifest_values["outer_fold"]) is not int
        or manifest_values["outer_fold"] != fold
        or manifest_values["fit_identity_sha256"] != contract.fit_identity_sha256(fold)
        or manifest_values["trainer_bundle_sha256"] != trainer.bundle.tree_sha256
    ):
        raise ValueError("evaluator manifest header differs from authenticated evidence")
    status = _exact_object(
        manifest_values["status"],
        fields=(
            "all_five_checkpoints_authenticated",
            "fold_metrics_complete",
            "post_publication_reinference_required",
            "readiness_receipt_sha256",
            "reinference_record_location",
            "score_release_bound_before_score_open",
            "score_release_sha256",
            "trainer_bundle_reopened",
        ),
        label="evaluator manifest status",
    )
    expected_status = {
        "all_five_checkpoints_authenticated": True,
        "fold_metrics_complete": True,
        "post_publication_reinference_required": True,
        "readiness_receipt_sha256": readiness_sha256,
        "reinference_record_location": "path_free_supervisor_result",
        "score_release_bound_before_score_open": True,
        "score_release_sha256": release_sha256,
        "trainer_bundle_reopened": True,
    }
    if status != expected_status:
        raise ValueError("evaluator manifest status does not bind release-gated reinference")


def _validate_worker_telemetry_document(
    document: object,
    *,
    contract: NativeDiffusionV1PilotContract,
    expected_outer_fold: int,
    expected_git_commit: str,
    expected_producer_job_id: str,
    expected_node_name: str,
    expected_device_uuid: str,
) -> dict[str, Any]:
    values = _exact_object(
        document,
        fields=_WORKER_TELEMETRY_FIELDS,
        label="worker operational telemetry",
    )
    fold = _outer_fold(expected_outer_fold)
    commit = _git_commit(expected_git_commit)
    producer_job_id = _job_id(expected_producer_job_id)
    node_name = _node_name(expected_node_name)
    if _GPU_UUID_RE.fullmatch(expected_device_uuid) is None:
        raise ValueError("expected worker telemetry CUDA UUID is malformed")
    if (
        values["schema_version"] != 1
        or type(values["schema_version"]) is not int
        or values["artifact"] != "native_categorical_diffusion_v1_r128_worker_operational_telemetry"
        or values["child_contract_sha256"] != contract.config_sha256
        or values["parent_contract_sha256"] != contract.parent_config_sha256
        or values["git_commit"] != commit
        or values["outer_fold"] != fold
        or type(values["outer_fold"]) is not int
        or values["node_name"] != node_name
        or values["cuda_device_uuid"] != expected_device_uuid
        or values["producer_job_id"] != producer_job_id
    ):
        raise ValueError("worker telemetry header differs from authenticated evidence")
    process_pid = values["process_pid"]
    if type(process_pid) is not int or process_pid <= 1:
        raise ValueError("worker telemetry supervisor PID is invalid")
    child_pids = _exact_object(
        values["child_process_pid_by_phase"],
        fields=("evaluator", "trainer"),
        label="worker telemetry child PID map",
    )
    if any(type(child_pids[key]) is not int or child_pids[key] <= 1 for key in child_pids):
        raise ValueError("worker telemetry child PID is invalid")
    if len({process_pid, *child_pids.values()}) != 3:
        raise ValueError("worker telemetry process identities are not distinct")
    evaluator_progress = _exact_object(
        values["evaluator_progress"],
        fields=("complete", "marker_count", "terminal_marker_sha256"),
        label="worker evaluator progress",
    )
    if (
        evaluator_progress["complete"] is not True
        or type(evaluator_progress["marker_count"]) is not int
        or evaluator_progress["marker_count"] != len(EVALUATOR_PROGRESS_PHASES)
    ):
        raise ValueError("worker evaluator progress is not the exact complete phase census")
    _sha256(
        evaluator_progress["terminal_marker_sha256"],
        label="worker evaluator progress terminal marker SHA-256",
    )

    phase_cuda = _exact_object(
        values["torch_cuda_by_phase"],
        fields=("evaluator", "trainer"),
        label="worker telemetry CUDA phase map",
    )
    allocated_by_phase: list[int] = []
    reserved_by_phase: list[int] = []
    for role in ("trainer", "evaluator"):
        item = _exact_object(
            phase_cuda[role],
            fields=("max_memory_allocated", "max_memory_reserved"),
            label=f"worker telemetry CUDA phase {role}",
        )
        allocated = item["max_memory_allocated"]
        reserved = item["max_memory_reserved"]
        if (
            type(allocated) is not int
            or type(reserved) is not int
            or allocated <= 0
            or reserved < allocated
        ):
            raise ValueError("worker telemetry CUDA phase peaks are invalid")
        allocated_by_phase.append(allocated)
        reserved_by_phase.append(reserved)
    maximum_allocated = values["torch_cuda_max_memory_allocated"]
    maximum_reserved = values["torch_cuda_max_memory_reserved"]
    cap = cast(int, contract.table("resources")["maximum_peak_allocated_memory_bytes"])
    if (
        type(maximum_allocated) is not int
        or type(maximum_reserved) is not int
        or maximum_allocated != max(allocated_by_phase)
        or maximum_reserved != max(reserved_by_phase)
        or maximum_allocated > cap
    ):
        raise ValueError("worker telemetry aggregate CUDA peaks are invalid")

    samples = _exact_object(
        values["in_allocation_nvidia_smi_process_memory_samples"],
        fields=("evaluator", "trainer"),
        label="worker telemetry external sample map",
    )
    for role in ("trainer", "evaluator"):
        entries = samples[role]
        if type(entries) is not list or not 0 < len(entries) <= _MAX_TELEMETRY_SAMPLES_PER_PHASE:
            raise ValueError("worker telemetry external samples are missing or unbounded")
        for raw in entries:
            item = _exact_object(
                raw,
                fields=("cuda_device_uuid", "process_pid", "used_memory_bytes"),
                label=f"worker telemetry external sample {role}",
            )
            if (
                item["cuda_device_uuid"] != expected_device_uuid
                or item["process_pid"] != child_pids[role]
                or type(item["process_pid"]) is not int
                or type(item["used_memory_bytes"]) is not int
                or item["used_memory_bytes"] <= 0
            ):
                raise ValueError("worker telemetry external sample is invalid")

    accounting = _exact_object(
        values["slurm_step_accounting"],
        fields=(
            "account",
            "cpus_per_task",
            "gpus_per_task",
            "job_id",
            "memory_per_node_mib",
            "node_id",
            "nodes",
            "partition",
            "step_id",
            "task_rank",
            "tasks",
            "tasks_per_node",
        ),
        label="worker telemetry Slurm step accounting",
    )
    expected_accounting = {
        "account": "bio",
        "cpus_per_task": 8,
        "gpus_per_task": 1,
        "job_id": values["producer_job_id"],
        "memory_per_node_mib": 32 * 1024,
        "node_id": fold,
        "nodes": 4,
        "partition": "gpumid",
        "step_id": 0,
        "task_rank": fold,
        "tasks": 4,
        "tasks_per_node": 1,
    }
    if accounting != expected_accounting:
        raise ValueError("worker telemetry Slurm step accounting differs from contract")
    checks = _exact_object(
        values["checks"],
        fields=(
            "allocator_peaks_bound_to_phase_child_pids",
            "allocator_peak_within_16_gib",
            "distinct_phase_processes",
            "evaluator_progress_complete_and_pid_bound",
            "external_process_memory_sampled_in_both_phases",
            "sole_cuda_device_bound",
            "supervisor_process_distinct_from_phase_children",
        ),
        label="worker telemetry checks",
    )
    if any(checks[key] is not True for key in checks):
        raise ValueError("worker telemetry contains an unpassed required check")
    validate_path_free_document(values, label="worker operational telemetry")
    return values


def _wait_for_sealed_files(
    paths: Sequence[Path],
    *,
    deadline: float,
    label: str,
) -> None:
    values = tuple(paths)
    if not values or len(values) != len(set(values)):
        raise ValueError("wait paths must be non-empty and unique")
    while True:
        pending = 0
        for path in values:
            if os.path.lexists(path):
                observed = os.lstat(path)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or stat.S_ISLNK(observed.st_mode)
                    or stat.S_IMODE(observed.st_mode) != 0o444
                    or not 0 < observed.st_size <= _MAX_CONTROL_BYTES
                ):
                    raise RuntimeError(f"{label} appeared as an unsafe entry")
                if observed.st_nlink == 2:
                    # Both control publishers commit with link(2) followed by
                    # unlink(2). On Lustre another rank can observe that tiny
                    # two-link commit window. Wait for the publisher to drop
                    # its private temporary name; persistent nlink=2 times out.
                    pending += 1
                elif observed.st_nlink != 1:
                    raise RuntimeError(f"{label} appeared as an unsafe entry")
            else:
                pending += 1
        if pending == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {label}")
        time.sleep(min(_POLL_SECONDS, remaining))


def _publish_canonical_record(destination: Path, payload: bytes) -> Path:
    document = parse_canonical_json(payload, label=destination.name)
    if type(document) is not dict:
        raise ValueError("operational record must be a canonical JSON object")
    validate_path_free_document(document, label=destination.name)
    if not 0 < len(payload) <= _MAX_CONTROL_BYTES:
        raise ValueError("operational record exceeds its bounded size")
    parent = destination.parent
    _validate_private_directory(parent, label="operational record parent")
    if os.path.lexists(destination):
        raise FileExistsError("operational record publication is no-overwrite")
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = Path(temporary_raw)
    linked = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short operational-record write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)
    if not linked:
        raise RuntimeError("operational record missed its atomic commit point")
    reopened = _read_regular_bytes(
        destination,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="published operational record",
        required_mode=0o444,
    )
    if reopened != payload:
        raise RuntimeError("published operational record changed bytes")
    return destination


def _seal_control_directory(path: Path, *, expected_files: set[str]) -> None:
    _validate_private_directory(path, label="control directory before sealing")
    observed = {entry.name for entry in path.iterdir()}
    if observed != expected_files:
        raise ValueError("control directory inventory differs before sealing")
    for entry in path.iterdir():
        _read_regular_bytes(
            entry,
            maximum_bytes=_MAX_CONTROL_BYTES,
            label="control record before directory seal",
            required_mode=0o444,
        )
    os.chmod(path, 0o555)
    _validate_sealed_directory(path, label="sealed control directory")


def _seal_evaluator_progress_root(path: Path) -> None:
    _validate_private_directory(path, label="evaluator progress root before sealing")
    entries = tuple(sorted(path.iterdir(), key=lambda value: value.name))
    if tuple(entry.name for entry in entries) != _FOLD_KEYS:
        raise ValueError("evaluator progress root does not contain exact fold directories")
    for entry in entries:
        _validate_sealed_directory(entry, label=f"sealed evaluator progress fold {entry.name}")
    os.chmod(path, 0o555)
    _fsync_directory(path)
    _fsync_directory(path.parent)
    _validate_sealed_directory(path, label="sealed evaluator progress root")


def _query_visible_gpu(visible: str) -> tuple[str, str, int]:
    if type(visible) is not str or _VISIBLE_GPU_RE.fullmatch(visible) is None:
        raise ValueError("visible GPU selector is unsafe")
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={visible}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=10.0,
    )
    if completed.returncode != 0 or len(completed.stdout) > 4096:
        raise RuntimeError("nvidia-smi failed to authenticate the visible GPU")
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("nvidia-smi returned non-ASCII GPU identity") from error
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one visible GPU")
    fields = tuple(part.strip() for part in lines[0].split(","))
    if len(fields) != 3 or _GPU_UUID_RE.fullmatch(fields[0]) is None:
        raise RuntimeError("nvidia-smi returned malformed GPU identity")
    try:
        memory_mib = int(fields[2])
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned malformed GPU memory") from error
    return fields[0], fields[1], memory_mib


def _child_deadline_signal(signum: int, _frame: object) -> None:
    raise SupervisorInterrupted(f"pilot supervisor interrupted by signal {signum}")


@contextmanager
def _signal_boundary():
    previous = {value: signal.getsignal(value) for value in _SUPERVISOR_SIGNALS}
    try:
        for value in _SUPERVISOR_SIGNALS:
            signal.signal(value, _child_deadline_signal)
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def _remaining_seconds(deadline: float, *, label: str) -> float:
    if type(deadline) is not float:
        raise TypeError("deadline must be an exact monotonic float")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"pilot run deadline expired before {label}")
    return remaining


def _validate_private_directory(path: Path, *, label: str) -> None:
    _reject_symlink_chain(path)
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_uid != os.getuid()
    ):
        raise ValueError(f"{label} must be an owned real mode-0700 directory")


def _validate_sealed_directory(path: Path, *, label: str) -> None:
    _reject_symlink_chain(path)
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o555
        or observed.st_uid != os.getuid()
    ):
        raise ValueError(f"{label} must be an owned real mode-0555 directory")


def _reject_symlink_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"path traverses a symbolic link: {current}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_object(
    payload: bytes,
    *,
    fields: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_CHILD_STDOUT_BYTES:
        raise ValueError(f"{label} must be non-empty bounded bytes")
    value = parse_canonical_json(payload, label=label)
    return _exact_object(value, fields=fields, label=label)


def _exact_object(value: object, *, fields: Sequence[str], label: str) -> dict[str, Any]:
    expected = tuple(fields)
    if type(value) is not dict or set(value) != set(expected) or len(value) != len(expected):
        raise ValueError(f"{label} does not have its exact object schema")
    return value


def _freeze_checkpoint_map(
    value: Mapping[str, CheckpointDigest],
) -> Mapping[str, CheckpointDigest]:
    if not isinstance(value, Mapping) or set(value) != set(_CHECKPOINT_KEYS):
        raise ValueError("checkpoint map must contain the exact five step keys")
    result: dict[str, CheckpointDigest] = {}
    for key in _CHECKPOINT_KEYS:
        item = value[key]
        if type(item) is not CheckpointDigest:
            raise TypeError("checkpoint map values must be exact CheckpointDigest objects")
        result[key] = CheckpointDigest(**item.document())
    return MappingProxyType(result)


def _freeze_reinference_map(
    value: Mapping[str, ReinferenceComparison],
) -> Mapping[str, ReinferenceComparison]:
    if not isinstance(value, Mapping) or set(value) != set(_CHECKPOINT_KEYS):
        raise ValueError("reinference map must contain the exact five step keys")
    result: dict[str, ReinferenceComparison] = {}
    for key in _CHECKPOINT_KEYS:
        item = value[key]
        if type(item) is not ReinferenceComparison:
            raise TypeError("reinference values must be exact comparisons")
        result[key] = ReinferenceComparison(**item.canonical_record())
    return MappingProxyType(result)


def _required_environment(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if type(value) is not str or not value:
        raise ValueError(f"required scheduler environment {name} is absent")
    return value


def _decimal_environment(
    environment: Mapping[str, str],
    name: str,
    *,
    expected: int | None = None,
) -> int:
    raw = _required_environment(environment, name)
    if not raw.isascii() or not raw.isdecimal() or (len(raw) > 1 and raw.startswith("0")):
        raise ValueError(f"scheduler environment {name} is not a canonical decimal")
    value = int(raw)
    if expected is not None and value != expected:
        raise RuntimeError(f"scheduler environment {name} differs from exact allocation")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        raise ValueError("expected Git commit must be a lowercase forty-character ID")
    return value


def _job_id(value: object) -> str:
    if type(value) is not str or _JOB_RE.fullmatch(value) is None:
        raise ValueError("producer job ID must be one positive canonical decimal")
    return value


def _outer_fold(value: object) -> int:
    if type(value) is not int or value not in _FOLDS:
        raise ValueError("outer fold must be one of exact integers 0..3")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return value


def _positive_pid(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 1:
        raise ValueError(f"{label} must be an exact positive non-init integer")
    return value


def _node_name(value: object) -> str:
    if type(value) is not str or _NODE_RE.fullmatch(value) is None:
        raise ValueError("node name must be bounded printable scheduler identity")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker", help="run the fold derived from SLURM_PROCID")
    worker.add_argument("--child-contract", required=True)
    worker.add_argument("--parent-contract", required=True)
    worker.add_argument("--projection-root", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--repository-root", required=True)
    worker.add_argument("--expected-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "worker":  # pragma: no cover - argparse exhausts choices
        raise RuntimeError("unknown supervisor command")
    result = run_pilot_supervisor_worker(
        child_contract_path=args.child_contract,
        parent_contract_path=args.parent_contract,
        projection_root=args.projection_root,
        run_root=args.run_root,
        repository_root=args.repository_root,
        expected_git_commit=args.expected_git_commit,
    )
    print(canonical_json_bytes(result.document()).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

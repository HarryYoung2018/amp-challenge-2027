"""Bounded local-pipe feature service; no oracle, network, or campaign authority."""

from __future__ import annotations

import json
import math
import os
import select
import signal
import stat
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from amp_challenge.representations.candidate_features import (
    CandidateFeatureRequest,
    _exclusive_write,
    source_identity,
    validate_candidate_features,
)
from amp_challenge.representations.fixed_shape_esm import FIXED_LAYOUT, FIXED_LAYOUT_SHA256
from amp_challenge.representations.peptide_esm import canonical_json, digest, file_digest
from amp_challenge.representations.run_feature_cache_records import (
    TR2_CAPACITY_SOURCE_FILES,
    check_tr2_capacity,
    check_tr2_capacity_request,
)

WORKER = "integrations/ampdiffusion/esm2_warm_candidate_worker.py"
SELF = "src/amp_challenge/representations/warm_candidate_features.py"
FIXED_SOURCE = "src/amp_challenge/representations/fixed_shape_esm.py"
MAX_REQUESTS = 128
MAX_SECONDS = 7200
OPERATION_SECONDS = 120
MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_WIRE_BYTES = 8192


def warm_source_identity(repository: Path, expected_commit: str, tr2_grouped_capacity=None) -> dict:
    identity = source_identity(repository, expected_commit)
    identity["files"].update(
        {name: file_digest(repository / name) for name in (SELF, WORKER, FIXED_SOURCE)}
    )
    if check_tr2_capacity(tr2_grouped_capacity, repository) is not None:
        identity["files"].update(
            {name: file_digest(repository / name) for name in TR2_CAPACITY_SOURCE_FILES}
        )
    return identity


def decode_wire(payload: bytes) -> dict:
    if not 0 < len(payload) <= MAX_WIRE_BYTES:
        raise ValueError("warm feature wire size differs")
    value = json.loads(payload)
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise ValueError("warm feature wire must be a canonical object")
    return value


def allocation_job() -> str:
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdigit()
        or os.environ.get("SLURM_JOB_ACCOUNT") != "bio"
        or os.environ.get("SLURM_JOB_PARTITION") != "gpumid"
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 4
    ):
        raise ValueError("warm features require an existing bio/gpumid four-CPU allocation")
    return job


def artifact_bytes(root: Path) -> int:
    """Count bounded retained regular files; no links or hidden directory traversal."""
    total, entries = 0, 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            path = Path(parent) / name
            metadata = path.lstat()
            entries += 1
            if (
                entries > 10000
                or stat.S_ISLNK(metadata.st_mode)
                or not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode))
                or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
            ):
                raise ValueError("warm session artifact inventory is unsafe or unbounded")
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
                if total > MAX_OUTPUT_BYTES:
                    raise ValueError("warm session output byte cap exceeded")
    return total


def legacy_environment(bundle: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_") or key in {"CUDA_VISIBLE_DEVICES", "HOME", "USER", "LOGNAME"}
    }
    environment.update(
        PATH="/usr/bin:/bin:/home/yonghan.yang/.local/bin",
        LC_ALL="C.UTF-8",
        LANG="C.UTF-8",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        UV_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        UV_PROJECT_ENVIRONMENT=str(bundle / "source/.venv"),
        UV_CACHE_DIR="/lustre/scratch/users/yonghan.yang/amp_challenge/uv-cache",
    )
    return environment


def verify_invocation_pins(root: Path, complete: dict) -> None:
    """Validate local timing bytes against an externally pinned final receipt.

    Consistent hashes protect the saved measurements; they do not make the
    measurements independent external timing attestations.
    """
    profile = check_tr2_capacity(
        json.loads((root / "session.json").read_bytes()).get("tr2_grouped_capacity")
    )
    count = complete.get("completed_requests")
    pins = complete.get("invocation_sha256s")
    if (
        type(count) is not int
        or not 0 <= count <= (281 if profile else MAX_REQUESTS)
        or type(pins) is not list
        or len(pins) != count
    ):
        raise ValueError("warm invocation pin inventory differs")
    for ordinal, expected in enumerate(pins):
        if file_digest(root / f"batch-{ordinal:04d}/invocation.json") != expected:
            raise ValueError("warm invocation timing/provenance bytes differ from final seal")


class WarmFeatureFailure(RuntimeError):
    """The session is permanently closed and partial artifacts are preserved."""


class WarmCandidateFeatureSession:
    """One sequential, non-restarting child in an existing GPU allocation.

    Use as a context manager. Setup, validation and teardown consume the original
    deadline. The outer controller must enforce its own hard wall interruption.
    Local receipts authenticate file consistency, not independent oracle truth.
    """

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        repository: Path,
        expected_commit: str,
        bundle: Path,
        output_root: Path,
        timeout_seconds: float,
        maximum_requests: int = MAX_REQUESTS,
        fixed_shape: bool = False,
        tr2_grouped_capacity: dict | None = None,
    ) -> None:
        self.start = time.monotonic()
        profile = check_tr2_capacity(tr2_grouped_capacity, repository)
        if profile and not fixed_shape:
            raise ValueError("TR2 capacity requires fixed numerical shape")
        if (
            type(timeout_seconds) not in (float, int)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= MAX_SECONDS
            or type(maximum_requests) is not int
            or not 1 <= maximum_requests <= (281 if profile else MAX_REQUESTS)
            or (profile is not None and maximum_requests != 281)
            or type(fixed_shape) is not bool
        ):
            raise ValueError("warm session resource bounds differ")
        CandidateFeatureRequest(run_id, session_id, ("ACDEFGHI",))
        self.deadline = self.start + timeout_seconds
        self.repository = repository.resolve(strict=True)
        self.bundle = bundle.resolve(strict=True)
        self.output_root = output_root.absolute()
        self.expected_commit = expected_commit
        self.job_id = allocation_job()
        self.source = warm_source_identity(self.repository, expected_commit, profile)
        self.output_root.mkdir(mode=0o700)
        self.config = {
            "artifact": "warm_candidate_feature_session_v1",
            "run_id": run_id,
            "session_id": session_id,
            "source": self.source,
            "job_id": self.job_id,
            "maximum_requests": maximum_requests,
            "timeout_seconds": timeout_seconds,
            "deadline_monotonic": self.deadline,
            "maximum_output_bytes": MAX_OUTPUT_BYTES,
        }
        if profile is not None:
            self.config["tr2_grouped_capacity"] = profile
        if fixed_shape:
            self.config["feature_layout"] = FIXED_LAYOUT
            self.config["feature_layout_sha256"] = FIXED_LAYOUT_SHA256
        payload = canonical_json(self.config)
        self.config_sha256 = digest(payload)
        self.head = self.config_sha256
        self.ordinal = 0
        self.invocation_sha256s: list[str] = []
        self.batch_ids: set[str] = set()
        self.closed = False
        self.process = None
        self.stderr = None
        self.buffer = b""
        _exclusive_write(self.output_root / "session.json", payload)
        try:
            self._check_deadline(min(self.deadline, self.start + OPERATION_SECONDS))
            self.stderr = (self.output_root / "stderr.log").open("xb")
            self.process = subprocess.Popen(
                self._command(),
                cwd=self.repository,
                env=legacy_environment(self.bundle),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                start_new_session=True,
                bufsize=0,
            )
            operation_deadline = min(self.deadline, self.start + OPERATION_SECONDS)
            ready = self._receive(operation_deadline)
            if ready != {
                "artifact": "warm_candidate_feature_ready_v1",
                "session_sha256": self.config_sha256,
            }:
                raise ValueError("warm feature readiness differs")
            self._check_source()
            self._check_deadline(operation_deadline)
            _exclusive_write(self.output_root / "ready.json", canonical_json(ready))
            self.ready_seconds = time.monotonic() - self.start
        except BaseException as error:
            self._fail(error)
            raise

    def _command(self) -> list[str]:
        return [
            "/home/yonghan.yang/.local/bin/uv",
            "run",
            "--project",
            str(self.bundle / "source"),
            "--locked",
            "--no-sync",
            "python",
            "-u",
            str(self.repository / WORKER),
            "--bundle",
            str(self.bundle),
            "--root",
            str(self.output_root),
            "--session-sha256",
            self.config_sha256,
            "--expected-commit",
            self.expected_commit,
        ]

    def _check_source(self) -> None:
        check_tr2_capacity(self.config.get("tr2_grouped_capacity"), self.repository)
        if warm_source_identity(
            self.repository, self.expected_commit, self.config.get("tr2_grouped_capacity")
        ) != self.source or (self.output_root / "session.json").read_bytes() != canonical_json(
            self.config
        ):
            raise ValueError("warm session source/config changed")

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("warm feature original wall deadline exceeded")

    def _receive(self, deadline: float) -> dict:
        while b"\n" not in self.buffer:
            self._check_deadline(deadline)
            remaining = deadline - time.monotonic()
            if not select.select([self.process.stdout], [], [], max(0.0, remaining))[0]:
                raise TimeoutError("warm feature response deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), MAX_WIRE_BYTES + 1 - len(self.buffer))
            if not chunk:
                raise WarmFeatureFailure("warm child exited before a complete response")
            self.buffer += chunk
            if len(self.buffer) > MAX_WIRE_BYTES:
                raise ValueError("warm feature response is oversized")
        payload, self.buffer = self.buffer.split(b"\n", 1)
        if self.buffer:
            raise ValueError("warm child emitted unsolicited response bytes")
        return decode_wire(payload + b"\n")

    def _send(self, message: dict, deadline: float) -> None:
        self._check_deadline(deadline)
        payload = canonical_json(message)
        decode_wire(payload)
        if not select.select([], [self.process.stdin], [], max(0.0, deadline - time.monotonic()))[
            1
        ]:
            raise TimeoutError("warm feature command deadline exceeded")
        # Commands are shorter than PIPE_BUF; only one may be outstanding.
        if len(payload) > 4096 or os.write(self.process.stdin.fileno(), payload) != len(payload):
            raise WarmFeatureFailure("warm feature command was not fully delivered")

    def request(self, request: CandidateFeatureRequest) -> dict:
        if self.closed:
            raise WarmFeatureFailure("warm session is permanently closed")
        started = time.monotonic()
        operation_deadline = min(self.deadline, started + OPERATION_SECONDS)
        try:
            self._check_deadline(operation_deadline)
            if (
                type(request) is not CandidateFeatureRequest
                or request.run_id != self.config["run_id"]
                or request.batch_id in self.batch_ids
                or self.ordinal >= self.config["maximum_requests"]
            ):
                raise ValueError("warm request run/batch/ordinal bound differs")
            request.__post_init__()
            check_tr2_capacity_request(
                self.config.get("tr2_grouped_capacity"), self.ordinal, request.sequences
            )
            self._check_source()
            batch = self.output_root / f"batch-{self.ordinal:04d}"
            batch.mkdir(mode=0o700)
            _exclusive_write(batch / "request.json", request.payload)
            command = {
                "operation": "features",
                "ordinal": self.ordinal,
                "previous_sha256": self.head,
                "request_sha256": digest(request.payload),
            }
            _exclusive_write(batch / "command.json", canonical_json(command))
            self._send(command, operation_deadline)
            response = self._receive(operation_deadline)
            expected = {
                "operation": "features",
                "ordinal": self.ordinal,
                "previous_sha256": self.head,
                "request_sha256": digest(request.payload),
                "manifest_sha256": file_digest(batch / "features/manifest.json"),
            }
            if response != expected:
                raise ValueError("warm response chain/manifest differs")
            _, _, manifest = validate_candidate_features(
                batch / "features", response["manifest_sha256"], request, self.source, self.job_id
            )
            if manifest["identity"].get("warm_session") != {
                "session_sha256": self.config_sha256,
                "ordinal": self.ordinal,
                "previous_sha256": self.head,
            }:
                raise ValueError("warm feature session identity differs")
            expected_layout = {
                key: self.config[key]
                for key in ("feature_layout", "feature_layout_sha256")
                if key in self.config
            }
            actual_layout = {
                key: manifest["identity"][key]
                for key in ("feature_layout", "feature_layout_sha256")
                if key in manifest["identity"]
            }
            if actual_layout != expected_layout:
                raise ValueError("warm feature numerical layout identity differs")
            if (batch / "request.json").read_bytes() != request.payload:
                raise ValueError("warm feature request changed")
            self._check_source()
            artifact_bytes(self.output_root)
            self._check_deadline(operation_deadline)
            _exclusive_write(batch / "response.json", canonical_json(response))
            receipt = {
                "artifact": "warm_candidate_feature_invocation_v1",
                "response_sha256": digest(canonical_json(response)),
                "elapsed_seconds": time.monotonic() - started,
                "rows": len(request.sequences),
                "oracle_calls": 0,
                "production_input_eligible": False,
                "scientific_evidence_accepted": False,
                "timing_scope": "local_measurement_not_independent_external_timing_authority",
            }
            _exclusive_write(batch / "invocation.json", canonical_json(receipt))
            self._check_deadline(operation_deadline)
            self.head = receipt["response_sha256"]
            self.ordinal += 1
            self.invocation_sha256s.append(digest(canonical_json(receipt)))
            self.batch_ids.add(request.batch_id)
            return receipt
        except BaseException as error:
            self._fail(error)
            raise

    def _kill_and_reap(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()
        if self.stderr is not None:
            self.stderr.close()

    def _fail(self, error: BaseException) -> None:
        self.closed = True
        self._kill_and_reap()
        failure_path = self.output_root / "FAILED.json"
        if not failure_path.exists():
            _exclusive_write(
                failure_path,
                canonical_json(
                    {
                        "artifact": "warm_candidate_feature_failure_v1",
                        "session_sha256": self.config_sha256,
                        "completed_requests": self.ordinal,
                        "previous_sha256": self.head,
                        "error": f"{type(error).__name__}: {error}",
                        "elapsed_seconds": time.monotonic() - self.start,
                        "oracle_calls": 0,
                    }
                ),
            )

    def close(self) -> None:
        if self.closed:
            return
        deadline = min(self.deadline, time.monotonic() + OPERATION_SECONDS)
        try:
            command = {"operation": "close", "ordinal": self.ordinal, "previous_sha256": self.head}
            self._send(command, deadline)
            if self._receive(deadline) != command:
                raise ValueError("warm close receipt differs")
            self.process.stdin.close()
            if self.process.wait(timeout=max(0.001, deadline - time.monotonic())) != 0:
                raise WarmFeatureFailure("warm child exit status differs")
            self._check_source()
            verify_invocation_pins(
                self.output_root,
                {
                    "completed_requests": self.ordinal,
                    "invocation_sha256s": self.invocation_sha256s,
                },
            )
            total = artifact_bytes(self.output_root)
            self._check_deadline(deadline)
            _exclusive_write(
                self.output_root / "COMPLETE.json",
                canonical_json(
                    {
                        "artifact": "warm_candidate_feature_session_complete_v1",
                        "session_sha256": self.config_sha256,
                        "completed_requests": self.ordinal,
                        "invocation_sha256s": self.invocation_sha256s,
                        "previous_sha256": self.head,
                        "startup_seconds": self.ready_seconds,
                        "elapsed_seconds": time.monotonic() - self.start,
                        "retained_bytes_before_receipt": total,
                        "stderr_sha256": file_digest(self.output_root / "stderr.log"),
                        "oracle_calls": 0,
                        "production_input_eligible": False,
                        "scientific_evidence_accepted": False,
                    }
                ),
            )
            self._check_deadline(deadline)
            self.closed = True
            self._kill_and_reap()
        except BaseException as error:
            self._fail(error)
            raise

    def __enter__(self) -> WarmCandidateFeatureSession:
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        if error is not None:
            if not self.closed:
                self._fail(error)
        else:
            self.close()

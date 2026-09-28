"""Bounded, label-free feature requests for fresh generated peptides.

The worker shares an existing Slurm GPU allocation. It neither submits jobs nor
queries an oracle. Its timings are local measurements, not independent receipts.
Compatible with both the current controller and the pinned Python 3.10 ESM worker.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.representations.peptide_esm import (
    canonical_json,
    digest,
    file_digest,
    load_features,
    read_sequences,
)

MAXIMUM_ROWS = 128
MAXIMUM_REQUEST_BYTES = 64 * 1024
MAXIMUM_WORKER_SECONDS = 120.0
WORKER = "integrations/ampdiffusion/esm2_candidate_features_worker.py"
SOURCE_FILES = (
    "src/amp_challenge/representations/candidate_features.py",
    "src/amp_challenge/representations/peptide_esm.py",
    "src/amp_challenge/representations/laplacian.py",
    WORKER,
    "integrations/ampdiffusion/esm2_peptide_features_worker.py",
    "integrations/ampdiffusion/esm2_peptide_source_pins.json",
)
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CandidateFeatureRequest:
    run_id: str
    batch_id: str
    sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (self.run_id, self.batch_id):
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError("feature run/batch identity is invalid")
        if not isinstance(self.sequences, tuple) or not 1 <= len(self.sequences) <= MAXIMUM_ROWS:
            raise ValueError("feature request requires 1..128 explicit unique peptides")
        read_sequences(self.sequence_payload)

    @property
    def sequence_payload(self) -> bytes:
        rows = []
        for sequence in self.sequences:
            if not isinstance(sequence, str) or not sequence.isascii():
                raise ValueError("feature peptide must be canonical ASCII text")
            rows.append({"sequence_id": digest(sequence.encode("ascii")), "sequence": sequence})
        return b"".join(canonical_json(row) for row in rows)

    @property
    def payload(self) -> bytes:
        return canonical_json(
            {
                "artifact": "generated_peptide_feature_request_v1",
                "run_id": self.run_id,
                "batch_id": self.batch_id,
                "sequence_input_sha256": digest(self.sequence_payload),
                "rows": read_sequences(self.sequence_payload),
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes, expected_sha256: str) -> CandidateFeatureRequest:
        if (
            not isinstance(expected_sha256, str)
            or not _SHA.fullmatch(expected_sha256)
            or len(payload) > MAXIMUM_REQUEST_BYTES
            or digest(payload) != expected_sha256
        ):
            raise ValueError("feature request size/digest mismatch")
        document = json.loads(payload)
        if not isinstance(document, dict) or set(document) != {
            "artifact",
            "run_id",
            "batch_id",
            "sequence_input_sha256",
            "rows",
        }:
            raise ValueError("feature request must contain only label-free request fields")
        rows = document["rows"]
        if not isinstance(rows, list):
            raise ValueError("feature request rows must be an array")
        canonical_rows = b"".join(canonical_json(row) for row in rows)
        validated = read_sequences(canonical_rows)
        request = cls(
            document["run_id"], document["batch_id"], tuple(row["sequence"] for row in validated)
        )
        if request.payload != payload:
            raise ValueError("feature request schema, identity or canonical encoding mismatch")
        return request


def source_identity(repository: Path, expected_commit: str) -> dict:
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("feature source commit must be explicit")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository
    )
    if actual != expected_commit or dirty:
        raise ValueError("feature source must be the exact clean requested commit")
    return {
        "git_commit": actual,
        "files": {relative: file_digest(repository / relative) for relative in SOURCE_FILES},
    }


def validate_candidate_features(
    output: Path, manifest_sha256: str, request: CandidateFeatureRequest, source: dict, job_id: str
) -> tuple[list[dict], dict, dict]:
    rows, arrays, manifest = load_features(output, manifest_sha256)
    if rows != read_sequences(request.sequence_payload):
        raise ValueError("generated feature rows/order differ from the request")
    identity = manifest["identity"]
    expected = {
        "candidate_request_sha256": digest(request.payload),
        "run_id": request.run_id,
        "batch_id": request.batch_id,
        "source": source,
        "job_id": job_id,
        "input_scope": "generated_peptides_label_free_not_namespace_membership_or_oracle_truth",
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("generated feature request/source/allocation identity mismatch")
    return rows, arrays, manifest


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)


class CandidateFeatureWorkerFailure(RuntimeError):
    """Failure artifacts are retained; the caller must stop, never silently retry."""


def run_candidate_feature_worker(
    request: CandidateFeatureRequest,
    *,
    repository: Path,
    expected_commit: str,
    bundle: Path,
    output_root: Path,
    timeout_seconds: float,
) -> dict:
    """Execute one fresh legacy-runtime process in the EXISTING Slurm allocation.

    The caller must include this entire operation in its arm wall clock and
    resource ledger. The 120-second per-batch cap is a bound, not free overhead.
    No namespace/exclusion decision, scientific acceptance or oracle-call credit
    is inferred here. No requests are deduplicated or retried behind the caller.
    """
    start = time.monotonic()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not 0 < timeout_seconds <= MAXIMUM_WORKER_SECONDS
    ):
        raise ValueError("feature timeout must be positive and no more than 120 seconds")
    deadline = start + timeout_seconds
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job_id.isdigit()
        or os.environ.get("SLURM_JOB_ACCOUNT") != "bio"
        or os.environ.get("SLURM_JOB_PARTITION") != "gpumid"
    ):
        raise ValueError("fresh features require an existing bio/gpumid Slurm allocation")
    # GPU count/type is checked by the pinned worker before model loading.
    repository, bundle = repository.resolve(strict=True), bundle.resolve(strict=True)
    output_root = output_root.absolute()
    source = source_identity(repository, expected_commit)
    output_root.mkdir(mode=0o700)  # deliberately no overwrite, cleanup or retry
    request_path = output_root / "request.json"
    _exclusive_write(request_path, request.payload)
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
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        UV_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        UV_PROJECT_ENVIRONMENT=str(bundle / "source/.venv"),
        UV_CACHE_DIR="/lustre/scratch/users/yonghan.yang/amp_challenge/uv-cache",
    )
    command = [
        "/home/yonghan.yang/.local/bin/uv",
        "run",
        "--project",
        str(bundle / "source"),
        "--locked",
        "--no-sync",
        "python",
        str(repository / WORKER),
        "--bundle",
        str(bundle),
        "--request",
        str(request_path),
        "--request-sha256",
        digest(request.payload),
        "--expected-commit",
        expected_commit,
        "--output",
        str(output_root / "features"),
        "--private-esm",
        str(output_root / "official-source"),
    ]
    status, failure, manifest_sha256, returncode = "failed", None, None, None
    try:
        with (
            (output_root / "stdout.log").open("xb") as stdout,
            (output_root / "stderr.log").open("xb") as stderr,
        ):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise CandidateFeatureWorkerFailure("feature deadline expired during setup")
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=remaining_seconds)
            except BaseException as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The child may have exited between wait and cleanup.
                finally:
                    returncode = process.wait()
                if isinstance(error, subprocess.TimeoutExpired):
                    raise CandidateFeatureWorkerFailure(
                        "feature worker exceeded its remaining wall-time cap"
                    ) from None
                raise
        if returncode != 0:
            raise CandidateFeatureWorkerFailure("feature worker exited unsuccessfully")
        manifest_sha256 = file_digest(output_root / "features/manifest.json")
        validate_candidate_features(
            output_root / "features", manifest_sha256, request, source, job_id
        )
        if (
            source_identity(repository, expected_commit) != source
            or request_path.read_bytes() != request.payload
        ):
            raise CandidateFeatureWorkerFailure("source/request changed during feature execution")
        if time.monotonic() > deadline:
            raise CandidateFeatureWorkerFailure("feature deadline expired during output validation")
        status = "completed_data_adapter_only"
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        failure = f"{type(error).__name__}: {error}"
    receipt = {
        "artifact": "generated_peptide_feature_invocation_v1",
        "status": status,
        "failure": failure,
        "request_sha256": digest(request.payload),
        "source": source,
        "job_id": job_id,
        "manifest_sha256": manifest_sha256,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - start,
        "timeout_seconds": timeout_seconds,
        "stdout_sha256": file_digest(output_root / "stdout.log")
        if (output_root / "stdout.log").exists()
        else None,
        "stderr_sha256": file_digest(output_root / "stderr.log")
        if (output_root / "stderr.log").exists()
        else None,
        "timing_scope": "local_measurement_not_independent_external_timing_authority",
        "oracle_calls": 0,
        "production_input_eligible": False,
        "scientific_evidence_accepted": False,
    }
    _exclusive_write(output_root / "invocation.json", canonical_json(receipt))
    if failure is not None:
        raise CandidateFeatureWorkerFailure(
            f"{failure}; retained receipt: {output_root / 'invocation.json'}"
        )
    return receipt

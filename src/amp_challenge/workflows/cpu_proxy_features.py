"""CPU feature provider retaining exact ESM assets and full fixed-shape extraction."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
from time import monotonic

import numpy as np

from amp_challenge.representations.candidate_features import CandidateFeatureRequest
from amp_challenge.representations.cpu_native_features import cpu_allocation, cpu_source
from amp_challenge.representations.fixed_shape_esm import FIXED_LAYOUT_SHA256
from amp_challenge.representations.peptide_esm import (
    canonical_json,
    digest,
    file_digest,
    load_features,
)
from amp_challenge.workflows.native_proxy_evolution import WarmEvolutionFeatures


class CpuEvolutionFeatures(WarmEvolutionFeatures):
    """One paid persistent CPU worker, with no CUDA receipt or equivalence claim."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source = cpu_source(self.repository, self.commit)
        self.job_id = cpu_allocation()
        self.closed = False
        self.process = None
        self.error_stream = None
        self.runtime = self.model_sha256 = None
        self.ordinal = 0
        self.config = {
            "run_id": self.run_id,
            "deadline": self.deadline,
            "source": self.source,
            "job_id": self.job_id,
            "maximum_requests": 512,
        }
        if not 0 < self.deadline - monotonic() <= 7000:
            raise ValueError("CPU features require the original at-most7000-second clock")
        payload = canonical_json(self.config)
        with (self.root / "session.json").open("xb") as stream:
            stream.write(payload)
        self.head = digest(payload)
        self.session = self  # Shared transform code binds this actual CPU source.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_") or key in {"HOME", "USER", "LOGNAME"}
        }
        environment.update(
            PATH="/usr/bin:/bin",
            LC_ALL="C.UTF-8",
            LANG="C.UTF-8",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        try:
            self.error_stream = (self.root / "worker.stderr").open("xb")
            self.process = subprocess.Popen(
                [
                    str(self.bundle / "source/.venv/bin/python"),
                    "-u",
                    str(self.repository / "integrations/ampdiffusion/esm2_cpu_proxy_worker.py"),
                    "--root",
                    str(self.root),
                    "--bundle",
                    str(self.bundle),
                    "--commit",
                    self.commit,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.error_stream,
                env=environment,
                start_new_session=True,
            )
            ready = self._receive()
            if set(ready) != {"ready", "runtime", "model_sha256"} or ready["ready"] != self.head:
                raise ValueError("CPU worker readiness differs")
            if (
                ready["runtime"]["device"] != "cpu"
                or ready["runtime"]["cuda_numerical_equivalence_claimed"] is not False
            ):
                raise ValueError("CPU runtime claims differ")
            self.runtime, self.model_sha256 = ready["runtime"], ready["model_sha256"]
            with (self.root / "ready.json").open("xb") as stream:
                stream.write(canonical_json(ready))
        except BaseException:
            self._abort()
            raise

    def _receive(self):
        self.check()
        if not select.select([self.process.stdout], [], [], self.deadline - monotonic())[0]:
            raise TimeoutError("CPU feature original deadline exhausted waiting for worker")
        payload = self.process.stdout.readline(1048577)
        if not 0 < len(payload) <= 1048576:
            raise ValueError("CPU worker exited or exceeded bounded response size")
        result = json.loads(payload)
        if canonical_json(result) != payload:
            raise ValueError("CPU response must be canonical JSON")
        self.check()
        return result

    def _send(self, value):
        self.check()
        self.process.stdin.write(canonical_json(value))
        self.process.stdin.flush()

    def _raw(self, sequences):
        if self.closed or self.requests >= 512:
            raise ValueError("CPU feature provider closed or maximum requests exhausted")
        self.check()
        if cpu_source(self.repository, self.commit) != self.source:
            raise ValueError("CPU source mutated")
        request = CandidateFeatureRequest(self.run_id, f"b{self.requests:05d}", tuple(sequences))
        batch = self.root / f"batch-{self.requests:04d}"
        batch.mkdir(mode=0o700)
        with (batch / "request.json").open("xb") as stream:
            stream.write(request.payload)
        command = {
            "operation": "features",
            "ordinal": self.requests,
            "previous_sha256": self.head,
            "request_sha256": digest(request.payload),
        }
        try:
            self._send(command)
            response = self._receive()
            expected = {**command, "manifest_sha256": file_digest(batch / "features/manifest.json")}
            if response != expected:
                raise ValueError("CPU feature response receipt differs")
            rows, arrays, manifest = load_features(batch / "features", response["manifest_sha256"])
            identity = manifest["identity"]
            expected_identity = {
                "runtime_cohort": "cpu_float32_fixed128x52_not_cuda_equivalence_claim",
                "source": self.source,
                "runtime": self.runtime,
                "job_id": self.job_id,
                "run_id": self.run_id,
                "batch_id": request.batch_id,
                "request_sha256": digest(request.payload),
                "feature_layout_sha256": FIXED_LAYOUT_SHA256,
                "model_sha256": self.model_sha256,
                "previous_sha256": self.head,
                "ordinal": self.requests,
            }
            if (
                any(identity.get(key) != value for key, value in expected_identity.items())
                or tuple(row["sequence"] for row in rows) != tuple(sequences)
                or (batch / "request.json").read_bytes() != request.payload
            ):
                raise ValueError("CPU feature identity, sequence order or request changed")
            with (batch / "response.json").open("xb") as stream:
                stream.write(canonical_json(response))
            self.head = digest(canonical_json(response))
            self.requests += 1
            self.ordinal = self.requests
            self.check()
            return np.asarray(arrays["esm_length_spectral"], dtype=np.float64), response[
                "manifest_sha256"
            ]
        except BaseException:
            self._abort()
            raise

    def _abort(self):
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()
        if self.error_stream is not None:
            self.error_stream.close()

    def close(self):
        if self.closed:
            return
        try:
            command = {"operation": "close", "ordinal": self.requests, "previous_sha256": self.head}
            self._send(command)
            if self._receive() != command:
                raise ValueError("CPU feature close receipt differs")
            self.process.wait(timeout=max(0.01, self.deadline - monotonic()))
            if self.process.returncode != 0:
                raise ValueError("CPU feature worker failed during close")
            with (self.root / "COMPLETE").open("xb") as stream:
                stream.write(
                    canonical_json(
                        {
                            "requests": self.requests,
                            "head_sha256": self.head,
                            "runtime_cohort": self.source["runtime_cohort"],
                        }
                    )
                )
        finally:
            self._abort()

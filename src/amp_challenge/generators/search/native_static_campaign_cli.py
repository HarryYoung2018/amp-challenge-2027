"""Slurm entry point for a provisioned native baseline 512-charge lifecycle.

The provider must supply approved receipt/eligibility authorities and assets;
there is deliberately no bundled failed-oracle or synthetic scientific default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--provider-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("a Slurm allocation is required")
    if not 0 < args.seconds <= 7200:
        raise ValueError("original clock must be at most 7200 seconds")
    if not args.output.is_absolute() or not args.provider.is_absolute():
        raise ValueError("absolute provider/output paths required")
    epoch = time.monotonic()
    deadline = epoch + args.seconds
    epoch_id = f"slurm-{os.environ['SLURM_JOB_ID']}-{time.time_ns()}"
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)

    def save(name, document):
        with (args.output / name).open("x") as stream:
            json.dump(document, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("original provisioning/execution deadline exceeded")

    def expired(_signum, _frame):
        raise TimeoutError("original provisioning/execution alarm")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    inputs = None
    try:
        payload = args.provider.read_bytes()
        if hashlib.sha256(payload).hexdigest() != args.provider_sha256:
            raise ValueError("provisioning source differs from external pin")
        save(
            "provisioning.json",
            {
                "original_epoch": epoch,
                "original_deadline": deadline,
                "clock_epoch_id": epoch_id,
                "provider": str(args.provider),
                "provider_sha256": args.provider_sha256,
                "job_id": os.environ["SLURM_JOB_ID"],
                "scientific_evidence_accepted": False,
            },
        )
        # Import and all model loading occur after the original epoch. Compile
        # the exact pinned bytes, avoiding stale or alternate provider bytecode.
        spec = importlib.util.spec_from_file_location("static_campaign_provider", args.provider)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(payload, str(args.provider), "exec"), module.__dict__)
        check()
        inputs = module.provision(
            output_root=args.output,
            original_epoch=epoch,
            original_deadline=deadline,
            clock_epoch_id=epoch_id,
            monotonic=time.monotonic,
        )
        check()
        from amp_challenge.generators.search.native_static_campaign import execute_baseline_campaign

        result = execute_baseline_campaign(
            inputs,
            output_root=args.output,
            original_epoch=epoch,
            original_deadline=deadline,
            clock_epoch_id=epoch_id,
            monotonic=time.monotonic,
        )
        check()
        save("returned.json", result)
        check()
        return 0
    except BaseException as error:
        if inputs is not None:
            with suppress(BaseException):
                inputs.bridge.abort(error)
        save(
            "failed.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "original_deadline": deadline,
                "observed_at": time.monotonic(),
                "scientific_evidence_accepted": False,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


if __name__ == "__main__":
    raise SystemExit(main())

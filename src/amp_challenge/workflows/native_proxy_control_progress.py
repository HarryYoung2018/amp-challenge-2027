"""Read-only incremental control progress monitor, run through the audit launcher."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 10 <= args.interval <= 300:
        raise ValueError("progress interval must be between 10 and 300 seconds")
    rows = {
        seed: {"seed": seed, "evaluations": 0, "status": "provisioning"}
        for seed in range(73121, 73126)
    }
    positions = dict.fromkeys(rows, 0)
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        for seed, row in rows.items():
            directory = args.root / f"evolutionary_kl-{seed}"
            journal = directory / "campaign/events.jsonl"
            if journal.exists():
                with journal.open() as stream:
                    stream.seek(positions[seed])
                    while True:
                        line = stream.readline()
                        if not line or not line.endswith("\n"):
                            break
                        event = json.loads(line)
                        positions[seed] = stream.tell()
                        if event["kind"] == "outcome":
                            row.update(
                                evaluations=event["evaluation_index"] + 1,
                                status=event["status"],
                            )
                        elif event["kind"] == "terminal":
                            row.update(status=event["result"]["status"], terminal=True)
            failure = directory / "setup_or_runner_failure.json"
            if failure.exists():
                row.update(status="runner_failed", terminal=True)
        print(json.dumps({"rows": list(rows.values()), "audit_claimed": False}), flush=True)
        if all(row.get("terminal") for row in rows.values()):
            return
        time.sleep(args.interval)
    raise TimeoutError("control progress monitor deadline")


if __name__ == "__main__":
    main()

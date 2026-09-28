"""Fail early when project code is executed outside a CSCC compute allocation.

Only standard-library imports belong here: the package calls this before
importing numerical libraries or loading models. Non-cluster installations
remain usable without Slurm.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path


def _cgroup_job_id() -> str:
    """Recover kernel-assigned membership for workers with a scrubbed environment."""
    try:
        membership = Path("/proc/self/cgroup").read_text()
    except OSError:
        return ""
    jobs = set(re.findall(r"(?:^|/)job_([1-9][0-9]*)(?=/|$)", membership, flags=re.MULTILINE))
    return jobs.pop() if len(jobs) == 1 else ""


def _scheduler(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["scontrol", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot verify the Slurm allocation; refusing compute work.") from error
    return result.stdout


def require_compute_node() -> None:
    """Require a running job owned by this user on the current non-login host."""
    host = socket.gethostname().split(".")[0].lower()
    if "login" in host:
        raise RuntimeError("Project execution is forbidden on login nodes. Use cluster/submit.sh.")
    job_id = os.environ.get("SLURM_JOB_ID", "") or _cgroup_job_id()
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise RuntimeError("A Slurm compute allocation is required. Use cluster/submit.sh.")
    try:
        record = json.loads(_scheduler("--json", "show", "job", job_id))
    except (ValueError, TypeError) as error:
        raise RuntimeError("Cannot read structured Slurm allocation metadata.") from error
    if (
        not isinstance(record, dict)
        or record.get("errors") != []
        or not isinstance(record.get("jobs"), list)
        or len(record["jobs"]) != 1
        or not isinstance(record["jobs"][0], dict)
    ):
        raise RuntimeError("Cannot read structured Slurm allocation metadata.")
    fields = record["jobs"][0]
    if (
        type(fields.get("job_id")) is not int
        or fields["job_id"] != int(job_id)
        or fields.get("job_state") != ["RUNNING"]
    ):
        raise RuntimeError("The Slurm job is not the requested running allocation.")
    if type(fields.get("user_id")) is not int or fields["user_id"] != os.getuid():
        raise RuntimeError("The Slurm allocation belongs to a different user.")
    nodes = fields.get("nodes", "")
    if not isinstance(nodes, str) or not nodes or nodes == "(null)":
        raise RuntimeError("The Slurm allocation has no compute nodes.")
    hosts = {line.split(".")[0].lower() for line in _scheduler("show", "hostnames", nodes).split()}
    if host not in hosts:
        raise RuntimeError("The current host is outside the Slurm allocation.")


def enforce_cluster_execution() -> None:
    """Automatically guard all package imports on CSCC and in Slurm sessions."""
    host = socket.gethostname().split(".")[0].lower()
    on_cluster = (
        "cscc" in host
        or "login" in host
        or re.match(r"^(?:cn|gpu[a-z]*)-[0-9]+$", host) is not None
        or bool(os.environ.get("SLURM_CLUSTER_NAME"))
        or bool(os.environ.get("SLURM_JOB_ID"))
    )
    if on_cluster:
        require_compute_node()

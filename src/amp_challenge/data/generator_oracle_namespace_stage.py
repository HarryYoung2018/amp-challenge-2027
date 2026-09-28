"""Authenticate and stage the accepted namespace-split inputs without manual copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.data.generator_oracle_namespace_split import (
    _AUTHORITY_BINDINGS,
    _CLAIMS,
    _INPUT_PATHS,
    _MAX_CONFIG_BYTES,
    _STAGING_MANIFEST,
    _STAGING_SUMS,
    _canonical,
    _open_root,
    _publish_claimed_tree,
    _sha,
    _snapshot,
    accepted_source_paths,
    load_config,
)

_GIT = re.compile(r"[0-9a-f]{40}")
_JOB = re.compile(r"[1-9][0-9]{0,19}")
_SHA = re.compile(r"[0-9a-f]{64}")
_SOURCE_PATH = "src/amp_challenge/data/generator_oracle_namespace_split.py"
_STAGE_SOURCE_PATH = "src/amp_challenge/data/generator_oracle_namespace_stage.py"


class StagingError(RuntimeError):
    """Fail-closed staging error."""


@dataclass(frozen=True)
class StagingExecution:
    output_dir: Path
    manifest_sha256: str
    producer_source_sha256: str
    twin_id: int
    root_dev: int
    root_ino: int
    completion_marker_sha256: str


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode:
        raise StagingError("could not authenticate producer Git identity")
    return completed.stdout


def stage_accepted_inputs(
    *,
    config_path: Path,
    expected_config_sha256: str,
    scratch_root: Path,
    repository_root: Path,
    expected_producer_source_sha256: str,
    producer_git_commit: str,
    producer_job_id: str,
    twin_id: int,
    runtime_environment_sha256: str,
    output_dir: Path,
) -> StagingExecution:
    cfg = load_config(config_path, expected_config_sha256)
    if not _SHA.fullmatch(expected_producer_source_sha256):
        raise StagingError("producer source SHA-256 is malformed")
    if not _GIT.fullmatch(producer_git_commit) or not _JOB.fullmatch(producer_job_id):
        raise StagingError("producer commit/job identity is malformed")
    if not _SHA.fullmatch(runtime_environment_sha256):
        raise StagingError("runtime environment SHA-256 is malformed")
    source_paths = accepted_source_paths(twin_id)
    if not repository_root.is_absolute() or not scratch_root.is_absolute():
        raise StagingError("repository and scratch roots must be absolute")
    actual_head = _git_bytes(repository_root, "rev-parse", "HEAD").decode().strip()
    status = _git_bytes(repository_root, "status", "--porcelain", "--untracked-files=all")
    if actual_head != producer_git_commit or status:
        raise StagingError("producer repository is not the exact clean commit")
    source_root_fd = _open_root(repository_root)
    try:
        code_snapshots = {
            path: _snapshot(
                source_root_fd,
                path,
                _MAX_CONFIG_BYTES * 32,
                immutable=False,
            )
            for path in (_SOURCE_PATH, _STAGE_SOURCE_PATH)
        }
    finally:
        os.close(source_root_fd)
    committed_sources = {
        path: _git_bytes(repository_root, "show", f"{producer_git_commit}:{path}")
        for path in code_snapshots
    }
    if (
        code_snapshots[_SOURCE_PATH].sha256 != expected_producer_source_sha256
        or _sha(committed_sources[_SOURCE_PATH]) != expected_producer_source_sha256
        or any(
            committed_sources[path] != snapshot.payload for path, snapshot in code_snapshots.items()
        )
    ):
        raise StagingError("producer source does not match the committed source pin")

    scratch_fd = _open_root(scratch_root)
    try:
        snapshots = {
            name: _snapshot(
                scratch_fd,
                source_paths[name],
                cfg.max_input_bytes,
                immutable=False,
            )
            for name in _INPUT_PATHS
        }
    finally:
        os.close(scratch_fd)
    for name, snapshot in snapshots.items():
        _, expected_bytes = cfg.inputs[name]
        if snapshot.sha256 != cfg.raw["inputs"][name]["sha256"] or snapshot.size != expected_bytes:
            raise StagingError(f"accepted source content pin mismatch: {name}")

    evidence = {
        name: {
            "bytes": snapshots[name].size,
            "logical_role": name,
            "sha256": snapshots[name].sha256,
            "source_dev": snapshots[name].fingerprint[0],
            "source_ino": snapshots[name].fingerprint[1],
            "source_mode": stat.S_IMODE(snapshots[name].fingerprint[5]),
            "source_nlink": snapshots[name].fingerprint[6],
            "source_relative_path": source_paths[name],
            "source_size": snapshots[name].fingerprint[2],
            "staged_relative_path": _INPUT_PATHS[name],
            "upstream_job_id": _AUTHORITY_BINDINGS[name][0],
            "upstream_receipt_logical": _AUTHORITY_BINDINGS[name][1],
            "upstream_receipt_sha256": cfg.raw["inputs"][_AUTHORITY_BINDINGS[name][1]]["sha256"],
        }
        for name in sorted(_INPUT_PATHS)
    }
    manifest = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_staging_v1",
        "status": "authenticated_non_authorizing_staging_only",
        "claims": _CLAIMS,
        "producer_identity": {
            "config_sha256": cfg.sha256,
            "git_commit": producer_git_commit,
            "job_id": producer_job_id,
            "source_path": _SOURCE_PATH,
            "source_sha256": expected_producer_source_sha256,
            "twin_id": twin_id,
            "runtime_environment_sha256": runtime_environment_sha256,
        },
        "producer_inventory": {
            path: snapshot.sha256 for path, snapshot in sorted(code_snapshots.items())
        },
        "inputs": evidence,
    }
    manifest_payload = _canonical(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    sums_payload = (
        b"".join(
            f"{snapshots[name].sha256}  {_INPUT_PATHS[name]}\n".encode()
            for name in sorted(_INPUT_PATHS, key=_INPUT_PATHS.get)
        )
        + f"{manifest_sha256}  {_STAGING_MANIFEST}\n".encode()
    )

    payloads = {_INPUT_PATHS[name]: snapshots[name].payload for name in sorted(_INPUT_PATHS)}
    payloads[_STAGING_MANIFEST] = manifest_payload
    payloads[_STAGING_SUMS] = sums_payload
    try:
        binding = _publish_claimed_tree(
            output_dir,
            payloads,
            marker_artifact="generator_oracle_namespace_staging_complete_v2",
            identity={
                "config_sha256": cfg.sha256,
                "producer_git_commit": producer_git_commit,
                "producer_job_id": producer_job_id,
                "producer_source_sha256": expected_producer_source_sha256,
                "runtime_environment_sha256": runtime_environment_sha256,
                "stager_source_sha256": code_snapshots[_STAGE_SOURCE_PATH].sha256,
                "twin_id": twin_id,
            },
            maximum_file_bytes=cfg.max_input_bytes,
        )
    except Exception as error:
        if isinstance(error, StagingError):
            raise
        raise StagingError(str(error)) from error
    return StagingExecution(
        output_dir,
        manifest_sha256,
        expected_producer_source_sha256,
        twin_id,
        binding.root_dev,
        binding.root_ino,
        binding.marker_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-producer-source-sha256", required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--twin-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--runtime-environment-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    execution = stage_accepted_inputs(
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        scratch_root=args.scratch_root,
        repository_root=args.repository_root,
        expected_producer_source_sha256=args.expected_producer_source_sha256,
        producer_git_commit=args.producer_git_commit,
        producer_job_id=args.producer_job_id,
        twin_id=args.twin_id,
        runtime_environment_sha256=args.runtime_environment_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "completion_marker_sha256": execution.completion_marker_sha256,
                "manifest_sha256": execution.manifest_sha256,
                "root_dev": execution.root_dev,
                "root_ino": execution.root_ino,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

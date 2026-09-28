"""Capture and verify a fresh job-scoped UV environment by complete inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from amp_challenge.data.generator_oracle_namespace_split import (
    _canonical,
    _open_root,
    _publish_claimed_tree,
    _read_committed_tree,
    _sha,
    _validate_json_shape,
)

_MAX_ENTRIES = 20_000
_MAX_DEPTH = 16
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_ATTESTATION_FILE = "environment_inventory.json"
_MARKER_ARTIFACT = "generator_oracle_namespace_runtime_inventory_complete_v1"
_GIT = re.compile(r"[0-9a-f]{40}")
_JOB = re.compile(r"[1-9][0-9]{0,19}")
_ROLES = frozenset({"producer", "audit", "finalizer"})


class RuntimeInventoryError(RuntimeError):
    """Fail-closed runtime inventory error."""


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", os.fspath(repository), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode:
        raise RuntimeInventoryError("runtime repository authentication failed")
    return completed.stdout


def _hash_regular(directory_fd: int, name: str) -> tuple[str, os.stat_result]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_FILE_BYTES:
            raise RuntimeInventoryError("runtime inventory has an unsafe or oversized file")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)

        def fields(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                value.st_mode,
                value.st_nlink,
            )

        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            fields(before) != fields(after)
            or observed != before.st_size
            or fields(after) != fields(entry)
        ):
            raise RuntimeInventoryError("runtime file changed or was substituted during inventory")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def inventory_environment(environment: Path) -> dict[str, Any]:
    root_fd = _open_root(environment)
    entries: dict[str, dict[str, int | str]] = {}
    total_bytes = 0

    def visit(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal total_bytes
        if depth > _MAX_DEPTH:
            raise RuntimeInventoryError("runtime environment exceeds depth cap")
        names: list[str] = []
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                if (
                    entry.name in {"", ".", ".."}
                    or "/" in entry.name
                    or len(entries) + len(names) >= _MAX_ENTRIES
                ):
                    raise RuntimeInventoryError(
                        "runtime environment inventory is unsafe or too large"
                    )
                names.append(entry.name)
        for name in sorted(names):
            if len(entries) >= _MAX_ENTRIES:
                raise RuntimeInventoryError("runtime environment inventory is unsafe or too large")
            relative = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            common: dict[str, int | str] = {
                "dev": info.st_dev,
                "ino": info.st_ino,
                "mode": stat.S_IMODE(info.st_mode),
                "nlink": info.st_nlink,
            }
            if stat.S_ISDIR(info.st_mode):
                entries[relative] = {"type": "directory", **common}
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(child)
                    if (held.st_dev, held.st_ino, held.st_mode) != (
                        info.st_dev,
                        info.st_ino,
                        info.st_mode,
                    ):
                        raise RuntimeInventoryError("runtime directory was substituted")
                    visit(child, relative, depth + 1)
                    rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if (
                        rebound.st_dev,
                        rebound.st_ino,
                        rebound.st_mode,
                        rebound.st_nlink,
                    ) != (
                        held.st_dev,
                        held.st_ino,
                        held.st_mode,
                        held.st_nlink,
                    ):
                        raise RuntimeInventoryError("runtime directory changed during inventory")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                digest, held = _hash_regular(directory_fd, name)
                total_bytes += held.st_size
                if total_bytes > _MAX_TOTAL_BYTES:
                    raise RuntimeInventoryError("runtime environment exceeds total byte cap")
                entries[relative] = {
                    "type": "file",
                    **common,
                    "bytes": held.st_size,
                    "sha256": digest,
                }
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(name, dir_fd=directory_fd)
                if not target or len(os.fsencode(target)) > 4_096:
                    raise RuntimeInventoryError("runtime symlink target is invalid")
                rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (info.st_dev, info.st_ino, info.st_mode) != (
                    rebound.st_dev,
                    rebound.st_ino,
                    rebound.st_mode,
                ):
                    raise RuntimeInventoryError("runtime symlink changed during inventory")
                entries[relative] = {"type": "symlink", **common, "target": target}
            else:
                raise RuntimeInventoryError("runtime environment contains an unsafe entry type")

    try:
        root_before = os.fstat(root_fd)
        visit(root_fd, "", 0)
        root_after = os.fstat(root_fd)
        if (
            root_before.st_dev,
            root_before.st_ino,
            root_before.st_mode,
            root_before.st_nlink,
        ) != (
            root_after.st_dev,
            root_after.st_ino,
            root_after.st_mode,
            root_after.st_nlink,
        ):
            raise RuntimeInventoryError("runtime environment root changed during inventory")
        return {
            "schema_version": 1,
            "artifact": "generator_oracle_namespace_runtime_inventory_v1",
            "root": {
                "dev": root_after.st_dev,
                "ino": root_after.st_ino,
            },
            "entries": entries,
            "entry_count": len(entries),
            "total_regular_file_bytes": total_bytes,
        }
    finally:
        os.close(root_fd)


def capture_environment(
    *,
    environment: Path,
    repository: Path,
    expected_commit: str,
    output_dir: Path,
    job_id: str,
    role: str,
) -> tuple[str, str]:
    if (
        not _GIT.fullmatch(expected_commit)
        or not _JOB.fullmatch(job_id)
        or role not in _ROLES
        or not repository.is_absolute()
    ):
        raise RuntimeInventoryError("runtime capture identity is malformed")
    actual_commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if actual_commit != expected_commit or status:
        raise RuntimeInventoryError("runtime capture repository is not the exact clean commit")
    inventory = inventory_environment(environment)
    inventory.update(
        {
            "git_commit": expected_commit,
            "job_id": job_id,
            "role": role,
            "pyproject_sha256": _sha((repository / "pyproject.toml").read_bytes()),
            "uv_lock_sha256": _sha((repository / "uv.lock").read_bytes()),
        }
    )
    _validate_json_shape(
        inventory,
        maximum_depth=20,
        maximum_containers=25_000,
        maximum_string_bytes=65_536,
        label="runtime environment inventory",
    )
    payload = _canonical(inventory)
    binding = _publish_claimed_tree(
        output_dir,
        {_ATTESTATION_FILE: payload},
        marker_artifact=_MARKER_ARTIFACT,
        identity={
            "environment_inventory_sha256": _sha(payload),
            "git_commit": expected_commit,
            "job_id": job_id,
            "role": role,
        },
        maximum_file_bytes=16 * 1024 * 1024,
    )
    return _sha(payload), binding.marker_sha256


def verify_environment(
    *,
    environment: Path,
    attestation_dir: Path,
    expected_inventory_sha256: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_inventory_sha256):
        raise RuntimeInventoryError("runtime inventory SHA-256 is malformed")
    snapshots, marker, _ = _read_committed_tree(
        attestation_dir,
        expected_marker_artifact=_MARKER_ARTIFACT,
        expected_files={_ATTESTATION_FILE},
        maximum_file_bytes=16 * 1024 * 1024,
        maximum_json_depth=16,
        maximum_json_containers=20_000,
        maximum_json_string_bytes=65_536,
    )
    payload = snapshots[_ATTESTATION_FILE].payload
    if _sha(payload) != expected_inventory_sha256:
        raise RuntimeInventoryError("runtime attestation content pin mismatch")
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeInventoryError("runtime inventory is invalid JSON") from error
    _validate_json_shape(
        document,
        maximum_depth=20,
        maximum_containers=25_000,
        maximum_string_bytes=65_536,
        label="runtime environment inventory",
    )
    if (
        not isinstance(document, dict)
        or _canonical(document) != payload
        or set(document)
        != {
            "schema_version",
            "artifact",
            "root",
            "entries",
            "entry_count",
            "total_regular_file_bytes",
            "git_commit",
            "job_id",
            "role",
            "pyproject_sha256",
            "uv_lock_sha256",
        }
        or document.get("schema_version") != 1
        or isinstance(document.get("schema_version"), bool)
        or document.get("artifact") != "generator_oracle_namespace_runtime_inventory_v1"
        or not isinstance(document.get("root"), dict)
        or set(document["root"]) != {"dev", "ino"}
        or not isinstance(document.get("entries"), dict)
        or document.get("entry_count") != len(document["entries"])
        or not isinstance(document.get("total_regular_file_bytes"), int)
        or isinstance(document.get("total_regular_file_bytes"), bool)
        or not _GIT.fullmatch(str(document.get("git_commit", "")))
        or not _JOB.fullmatch(str(document.get("job_id", "")))
        or document.get("role") not in _ROLES
        or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("pyproject_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("uv_lock_sha256", "")))
    ):
        raise RuntimeInventoryError("runtime inventory schema or identity is malformed")
    if marker.get("identity") != {
        "environment_inventory_sha256": _sha(payload),
        "git_commit": document["git_commit"],
        "job_id": document["job_id"],
        "role": document["role"],
    }:
        raise RuntimeInventoryError("runtime marker does not bind the exact inventory identity")
    current = inventory_environment(environment)
    for key in ("git_commit", "job_id", "role", "pyproject_sha256", "uv_lock_sha256"):
        current[key] = document[key]
    if _canonical(current) != payload:
        raise RuntimeInventoryError("runtime environment differs from authenticated inventory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--environment", type=Path, required=True)
    capture.add_argument("--repository", type=Path, required=True)
    capture.add_argument("--expected-commit", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--job-id", required=True)
    capture.add_argument("--role", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--environment", type=Path, required=True)
    verify.add_argument("--attestation-dir", type=Path, required=True)
    verify.add_argument("--expected-inventory-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        inventory_sha256, marker_sha256 = capture_environment(
            environment=args.environment,
            repository=args.repository,
            expected_commit=args.expected_commit,
            output_dir=args.output_dir,
            job_id=args.job_id,
            role=args.role,
        )
        print(
            json.dumps(
                {
                    "environment_inventory_sha256": inventory_sha256,
                    "completion_marker_sha256": marker_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    verify_environment(
        environment=args.environment,
        attestation_dir=args.attestation_dir,
        expected_inventory_sha256=args.expected_inventory_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

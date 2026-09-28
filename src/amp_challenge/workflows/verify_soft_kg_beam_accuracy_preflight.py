"""Stably replay one exact-versus-beam soft-KG accuracy receipt.

The verifier runs on a node explicitly excluding the producer node.  It binds
the immutable producer artifact to independently supplied identities, then
replays all frozen cases with the reviewed producer Python implementation.
Producer timing is checked only as producer-side operational evidence; replay
wall time is deliberately not used as scientific or production evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tomllib
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np

from amp_challenge.workflows.soft_kg_beam_accuracy_preflight import (
    _EXPECTED_PREFLIGHT,
    ARTIFACT,
    CONFIG_RELATIVE,
    FrozenBeamAccuracySpec,
    _case_inventory_sha256,
    _evaluate_case,
    _summarize_cases,
)
from amp_challenge.workflows.soft_kg_runtime_preflight import (
    _required_slurm_environment,
    _write_exclusive,
)

AUDIT_ARTIFACT: Final = "evolutionary_kl_beam_soft_kg_accuracy_preflight_audit_v1"
_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_INDEPENDENT_RELATIVE_TOLERANCE: Final = 5e-13
_INDEPENDENT_ABSOLUTE_TOLERANCE: Final = 5e-15
_PRODUCER_BASE: Final = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/"
    "evolutionary-kl-soft-kg-beam-accuracy-preflight-v1"
)
_PRODUCER_REPO_ROOT: Final = Path("/home/yonghan.yang/amp_evolutionary_validation_20260908")
_PRODUCER_SOURCE_PATHS: Final = (
    CONFIG_RELATIVE,
    Path("src/amp_challenge/acquisition/soft_kg.py"),
    Path("src/amp_challenge/models/posterior.py"),
    Path("src/amp_challenge/workflows/soft_kg_beam_accuracy_preflight.py"),
    Path("cluster/slurm/run_evolutionary_kl_soft_kg_beam_accuracy_preflight.sbatch"),
)
_EXPECTED_TOP_LEVEL_KEYS: Final = {
    "artifact",
    "case_inventory_sha256",
    "cases",
    "config_sha256",
    "environment",
    "gate",
    "git_commit",
    "limitations",
    "resources",
    "schema_version",
    "slurm",
    "spec",
    "status",
    "summary",
    "timing_seconds",
}
_EXPECTED_LIMITATIONS: Final = [
    "synthetic_non_biological_gaussian_panel",
    "small_pool_ten_batch_five_not_campaign_q14",
    "beam_width_four_only",
    "fixed_monte_carlo_fantasies",
    "no_peptide_model_or_oracle",
    "no_scientific_performance_claim",
    "no_production_pin_change",
]

_Identity: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _DirectorySnapshot:
    """Identity of a directory held open across one authentication pass."""

    path: Path
    identity: _Identity


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """Bytes and identity captured from one no-follow file descriptor."""

    path: Path
    relative_path: Path
    payload: bytes
    sha256: str
    identity: _Identity


def _stat_identity(metadata: os.stat_result) -> _Identity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _safe_relative_path(relative_path: Path, *, label: str) -> tuple[str, ...]:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return relative_path.parts


def _open_stable_directory(path: Path, *, label: str) -> tuple[int, _DirectorySnapshot]:
    before = os.lstat(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened without following links") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or _stat_identity(before) != _stat_identity(opened)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while it was opened")
    return descriptor, _DirectorySnapshot(path=path, identity=_stat_identity(opened))


def _open_beneath(root_descriptor: int, relative_path: Path, *, label: str) -> int:
    parts = _safe_relative_path(relative_path, label=label)
    directory_descriptor = os.dup(root_descriptor)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened beneath its authenticated root") from error
    finally:
        os.close(directory_descriptor)


def _read_descriptor_bytes(
    descriptor: int,
    *,
    expected_size: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError(f"{label} was truncated while its stable descriptor was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError(f"{label} grew while its stable descriptor was read")
    return b"".join(chunks)


def _capture_file_beneath(
    *,
    root_descriptor: int,
    root_path: Path,
    relative_path: Path,
    label: str,
) -> _FileSnapshot:
    descriptor = _open_beneath(root_descriptor, relative_path, label=label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        payload = _read_descriptor_bytes(
            descriptor,
            expected_size=before.st_size,
            label=label,
        )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"{label} changed while its stable descriptor was read")
    return _FileSnapshot(
        path=root_path / relative_path,
        relative_path=relative_path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=_stat_identity(before),
    )


def _assert_directory_unchanged(
    descriptor: int,
    snapshot: _DirectorySnapshot,
    *,
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.lstat(snapshot.path)
    except OSError as error:
        raise ValueError(f"{label} disappeared after authentication") from error
    if _stat_identity(opened) != snapshot.identity or _stat_identity(current) != snapshot.identity:
        raise ValueError(f"{label} identity changed after authentication")


def _assert_file_unchanged(
    *,
    root_descriptor: int,
    root_path: Path,
    snapshot: _FileSnapshot,
    label: str,
) -> None:
    current = _capture_file_beneath(
        root_descriptor=root_descriptor,
        root_path=root_path,
        relative_path=snapshot.relative_path,
        label=label,
    )
    if current.identity != snapshot.identity or current.sha256 != snapshot.sha256:
        raise ValueError(f"{label} changed after authentication")


def _capture_path(path: Path, *, label: str) -> _FileSnapshot:
    root_descriptor, root_snapshot = _open_stable_directory(path.parent, label=f"{label} parent")
    try:
        snapshot = _capture_file_beneath(
            root_descriptor=root_descriptor,
            root_path=path.parent,
            relative_path=Path(path.name),
            label=label,
        )
        _assert_file_unchanged(
            root_descriptor=root_descriptor,
            root_path=path.parent,
            snapshot=snapshot,
            label=label,
        )
        _assert_directory_unchanged(root_descriptor, root_snapshot, label=f"{label} parent")
        return snapshot
    finally:
        os.close(root_descriptor)


def _sha256(path: Path) -> str:
    return _capture_path(path, label=str(path)).sha256


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        output[key] = value
    return output


def _load_canonical_json_bytes(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("producer receipt must be canonical ASCII JSON") from error
    document = json.loads(
        decoded,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if type(document) is not dict:
        raise ValueError("producer receipt must be one JSON object")
    canonical = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if payload != canonical:
        raise ValueError("producer receipt is not canonical JSON")
    return document


def _load_canonical_json(path: Path) -> dict[str, object]:
    snapshot = _capture_path(path, label="producer receipt")
    return _load_canonical_json_bytes(snapshot.payload)


def _exact_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _parse_manifest_bytes(payload: bytes) -> dict[Path, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("producer SHA256SUMS must be non-empty canonical LF text")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("producer SHA256SUMS must be ASCII") from error
    entries: dict[Path, str] = {}
    for line in lines:
        manifest_fields = line.split("  ", maxsplit=1)
        if len(manifest_fields) != 2 or _SHA256_PATTERN.fullmatch(manifest_fields[0]) is None:
            raise ValueError("producer SHA256SUMS has a non-canonical row")
        target = Path(manifest_fields[1])
        if not target.is_absolute() or target in entries:
            raise ValueError("producer SHA256SUMS targets must be unique absolute paths")
        entries[target] = manifest_fields[0]
    return entries


def _parse_manifest(path: Path) -> dict[Path, str]:
    snapshot = _capture_path(path, label="producer SHA256SUMS")
    return _parse_manifest_bytes(snapshot.payload)


def _decode_frozen_accuracy_spec(payload: bytes) -> tuple[FrozenBeamAccuracySpec, str]:
    """Decode config bytes already captured from a stable producer-source FD."""

    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("beam accuracy config is not valid UTF-8 TOML") from error
    if type(document) is not dict or set(document) != {"schema_version", "preflight"}:
        raise ValueError("beam accuracy config has an unexpected top-level schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("beam accuracy schema_version must be exact integer one")
    values = document["preflight"]
    if type(values) is not dict or set(values) != set(_EXPECTED_PREFLIGHT):
        raise ValueError("beam accuracy fields differ from the frozen schema")
    for key, expected in _EXPECTED_PREFLIGHT.items():
        value = values[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"beam accuracy field {key!r} differs from its frozen value")
    spec = FrozenBeamAccuracySpec(
        **{
            **values,
            "regimes": tuple(values["regimes"]),
            "cost_modes": tuple(values["cost_modes"]),
        }
    )
    if spec.case_count % (len(spec.regimes) * len(spec.cost_modes)) != 0:
        raise ValueError("case_count must balance every regime-by-cost cell")
    if spec.decision_count != spec.pool_size or spec.n_outputs != 2:
        raise ValueError("frozen accuracy panel requires ten two-output decisions")
    if spec.exact_combination_count != spec.max_exact_combinations:
        raise ValueError("exact combination guard must cover exactly the frozen inventory")
    if spec.worst_case_beam_groups_scored > spec.max_groups_scored:
        raise ValueError("beam score cap cannot cover every generated depth")
    if not spec.synthetic_only or spec.production_pins_modified or spec.scientific_claim != "none":
        raise ValueError("beam accuracy preflight may carry no scientific or production claim")
    return spec, hashlib.sha256(payload).hexdigest()


def _verify_exact_artifact_inventory(
    producer_root: Path,
    expected_files: tuple[Path, ...],
) -> None:
    root_descriptor, root_snapshot, snapshots = _capture_exact_artifact_inventory(
        producer_root,
        expected_files,
    )
    try:
        for name, snapshot in snapshots.items():
            _assert_file_unchanged(
                root_descriptor=root_descriptor,
                root_path=producer_root,
                snapshot=snapshot,
                label=f"producer artifact {name}",
            )
        _assert_directory_unchanged(
            root_descriptor,
            root_snapshot,
            label="producer root",
        )
    finally:
        os.close(root_descriptor)


def _capture_exact_artifact_inventory(
    producer_root: Path,
    expected_files: tuple[Path, ...],
    *,
    producer_base: Path | None = None,
) -> tuple[int, _DirectorySnapshot, dict[str, _FileSnapshot]]:
    frozen_base = _PRODUCER_BASE if producer_base is None else producer_base
    if producer_root.parent != frozen_base:
        raise ValueError("producer root is outside the frozen artifact namespace")
    if any(path.parent != producer_root for path in expected_files):
        raise ValueError("producer expected files must be direct children of its root")
    expected_names = tuple(path.name for path in expected_files)
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("producer expected file names must be unique")
    root_descriptor, root_snapshot = _open_stable_directory(
        producer_root,
        label="producer root",
    )
    root_mode = stat.S_IMODE(root_snapshot.identity[3])
    if root_mode != 0o555 or root_snapshot.identity[4] != 2:
        os.close(root_descriptor)
        raise ValueError("producer root must have exact mode 0555 and link count two")
    observed_names = tuple(sorted(os.listdir(root_descriptor)))
    if observed_names != tuple(sorted(expected_names)):
        os.close(root_descriptor)
        raise ValueError("producer root contains an unexpected or missing entry")
    snapshots: dict[str, _FileSnapshot] = {}
    try:
        for name in observed_names:
            snapshot = _capture_file_beneath(
                root_descriptor=root_descriptor,
                root_path=producer_root,
                relative_path=Path(name),
                label=f"producer artifact {name}",
            )
            if stat.S_IMODE(snapshot.identity[3]) != 0o444 or snapshot.identity[4] != 1:
                raise ValueError("producer root files must have exact mode 0444 and link count one")
            snapshots[name] = snapshot
    except Exception:
        os.close(root_descriptor)
        raise
    return root_descriptor, root_snapshot, snapshots


def _verify_recorded_spec(
    document: dict[str, object],
    spec: FrozenBeamAccuracySpec,
) -> None:
    recorded = document.get("spec")
    field_names = tuple(field.name for field in fields(spec))
    expected_keys = {
        *field_names,
        "exact_combination_count",
        "worst_case_beam_groups_scored",
    }
    if type(recorded) is not dict or set(recorded) != expected_keys:
        raise ValueError("producer recorded spec differs from the exact frozen schema")
    for name in field_names:
        expected = getattr(spec, name)
        if type(expected) is tuple:
            expected = list(expected)
        if type(recorded[name]) is not type(expected) or recorded[name] != expected:
            raise ValueError(f"producer recorded spec field {name!r} differs from config")
    if (
        recorded["exact_combination_count"] != spec.exact_combination_count
        or recorded["worst_case_beam_groups_scored"] != spec.worst_case_beam_groups_scored
    ):
        raise ValueError("producer derived spec fields do not reconstruct")


def _verify_git_bound_sources(
    repo_root: Path,
    commit: str,
    manifest: dict[Path, str],
    source_snapshots: dict[Path, _FileSnapshot],
) -> None:
    for relative_path in _PRODUCER_SOURCE_PATHS:
        producer_path = _PRODUCER_REPO_ROOT / relative_path
        snapshot = source_snapshots.get(relative_path)
        if snapshot is None or snapshot.path != producer_path:
            raise ValueError(f"producer source snapshot is missing for {relative_path}")
        if snapshot.identity[4] != 1:
            raise ValueError("producer manifest source entries must be regular single-link files")
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path.as_posix()}"],
            check=True,
            capture_output=True,
        )
        commit_digest = hashlib.sha256(completed.stdout).hexdigest()
        if commit_digest != manifest[producer_path] or snapshot.sha256 != commit_digest:
            raise ValueError(f"producer manifest does not match commit blob {relative_path}")


def _capture_producer_sources() -> tuple[
    int,
    _DirectorySnapshot,
    dict[Path, _FileSnapshot],
]:
    root_descriptor, root_snapshot = _open_stable_directory(
        _PRODUCER_REPO_ROOT,
        label="producer source root",
    )
    snapshots: dict[Path, _FileSnapshot] = {}
    try:
        for relative_path in _PRODUCER_SOURCE_PATHS:
            snapshot = _capture_file_beneath(
                root_descriptor=root_descriptor,
                root_path=_PRODUCER_REPO_ROOT,
                relative_path=relative_path,
                label=f"producer source {relative_path}",
            )
            if snapshot.identity[4] != 1:
                raise ValueError(
                    "producer manifest source entries must be regular single-link files"
                )
            snapshots[relative_path] = snapshot
    except Exception:
        os.close(root_descriptor)
        raise
    return root_descriptor, root_snapshot, snapshots


def _revalidate_snapshot_set(
    *,
    root_descriptor: int,
    root_path: Path,
    root_snapshot: _DirectorySnapshot,
    snapshots: dict[Path, _FileSnapshot],
    label: str,
) -> None:
    for relative_path, snapshot in snapshots.items():
        _assert_file_unchanged(
            root_descriptor=root_descriptor,
            root_path=root_path,
            snapshot=snapshot,
            label=f"{label} {relative_path}",
        )
    _assert_directory_unchanged(root_descriptor, root_snapshot, label=f"{label} root")


def _revalidate_closed_snapshot_set(
    *,
    root_path: Path,
    root_snapshot: _DirectorySnapshot,
    snapshots: dict[Path, _FileSnapshot],
    label: str,
) -> None:
    root_descriptor, current_root = _open_stable_directory(
        root_path,
        label=f"{label} root",
    )
    try:
        if current_root.identity != root_snapshot.identity:
            raise ValueError(f"{label} root identity changed after authentication")
        _revalidate_snapshot_set(
            root_descriptor=root_descriptor,
            root_path=root_path,
            root_snapshot=root_snapshot,
            snapshots=snapshots,
            label=label,
        )
    finally:
        os.close(root_descriptor)


def _snapshot_set_sha256(
    root_snapshot: _DirectorySnapshot,
    snapshots: dict[Path, _FileSnapshot],
) -> str:
    return _canonical_sha256(
        {
            "root_identity": list(root_snapshot.identity),
            "files": {
                relative_path.as_posix(): {
                    "identity": list(snapshot.identity),
                    "sha256": snapshot.sha256,
                }
                for relative_path, snapshot in sorted(
                    snapshots.items(),
                    key=lambda item: item[0].as_posix(),
                )
            },
        }
    )


def _verify_audit_checkout(repo_root: Path, audit_commit: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != audit_commit or status:
        raise ValueError("audit commit must identify the exact clean checkout")


def _new_numeric_stats() -> dict[str, object]:
    return {
        "float_comparison_count": 0,
        "maximum_absolute_error": 0.0,
        "maximum_absolute_error_path": None,
        "maximum_scaled_relative_error": 0.0,
        "maximum_scaled_relative_error_path": None,
    }


def _compare_tree(
    observed: object,
    expected: object,
    *,
    path: str,
    numeric_stats: dict[str, object],
) -> None:
    """Compare typed JSON trees, allowing only declared finite-float tolerance."""

    if type(expected) is float:
        actual = _finite_float(observed, name=path)
        if not np.isfinite(expected):
            raise ValueError(f"replayed value at {path} is unexpectedly non-finite")
        absolute_error = abs(actual - expected)
        scaled_relative_error = absolute_error / max(
            abs(expected),
            _INDEPENDENT_ABSOLUTE_TOLERANCE,
        )
        numeric_stats["float_comparison_count"] = (
            _exact_integer(
                numeric_stats["float_comparison_count"],
                name="float comparison count",
            )
            + 1
        )
        if absolute_error > float(numeric_stats["maximum_absolute_error"]):
            numeric_stats["maximum_absolute_error"] = absolute_error
            numeric_stats["maximum_absolute_error_path"] = path
        if scaled_relative_error > float(numeric_stats["maximum_scaled_relative_error"]):
            numeric_stats["maximum_scaled_relative_error"] = scaled_relative_error
            numeric_stats["maximum_scaled_relative_error_path"] = path
        if not np.isclose(
            actual,
            expected,
            rtol=_INDEPENDENT_RELATIVE_TOLERANCE,
            atol=_INDEPENDENT_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"producer numeric value at {path} exceeds declared tolerance")
        return
    if type(expected) is dict:
        if type(observed) is not dict or set(observed) != set(expected):
            raise ValueError(f"producer object schema at {path} differs from regeneration")
        for key, expected_value in expected.items():
            _compare_tree(
                observed[key],
                expected_value,
                path=f"{path}.{key}",
                numeric_stats=numeric_stats,
            )
        return
    if type(expected) is list:
        if type(observed) is not list or len(observed) != len(expected):
            raise ValueError(f"producer list shape at {path} differs from regeneration")
        for index, (actual_value, expected_value) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _compare_tree(
                actual_value,
                expected_value,
                path=f"{path}[{index}]",
                numeric_stats=numeric_stats,
            )
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f"producer value at {path} differs exactly from regeneration")


def _producer_accounting(job_id: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,JobName,State,ExitCode,NodeList,AllocCPUS,ReqMem,"
            "Account,Partition,ElapsedRaw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [line.split("|") for line in completed.stdout.splitlines() if line]
    parent = [record for record in records if record[0] == str(job_id)]
    if len(parent) != 1 or len(parent[0]) != 10:
        raise ValueError("Slurm accounting did not return one canonical producer job")
    record = parent[0]
    if (
        record[1] != "evo-softkg-beam-accuracy"
        or record[2] != "COMPLETED"
        or record[3] != "0:0"
        or not record[4]
        or record[5] != "4"
        or record[6] not in {"32G", "32768M"}
        or record[7] != "bio"
        or record[8] != "standard"
        or not record[9].isdecimal()
        or int(record[9]) <= 0
    ):
        raise ValueError("Slurm accounting does not authenticate the frozen producer allocation")
    return {
        "job_name": record[1],
        "state": record[2],
        "exit_code": record[3],
        "node_list": record[4],
        "allocated_cpus": int(record[5]),
        "requested_memory": record[6],
        "account": record[7],
        "partition": record[8],
        "elapsed_seconds": int(record[9]),
    }


def _audit_job_control(
    audit_job: str,
    *,
    audit_node: str,
    producer_node: str,
) -> dict[str, object]:
    completed = subprocess.run(
        ["scontrol", "show", "job", "-o", audit_job],
        check=True,
        capture_output=True,
        text=True,
    )

    def field(name: str) -> str:
        match = re.search(rf"(?:^| ){re.escape(name)}=(\S+)", completed.stdout)
        if match is None:
            raise RuntimeError(f"audit allocation omits {name}")
        return match.group(1)

    observed = {
        "job_id": field("JobId"),
        "state": field("JobState"),
        "node_list": field("NodeList"),
        "excluded_node_list": field("ExcNodeList"),
        "partition": field("Partition"),
        "account": field("Account"),
        "num_nodes": field("NumNodes"),
        "num_cpus": field("NumCPUs"),
    }
    if (
        observed["job_id"] != audit_job
        or observed["state"] != "RUNNING"
        or observed["node_list"] != audit_node
        or observed["excluded_node_list"] != producer_node
        or observed["partition"] != "standard"
        or observed["account"] != "bio"
        or observed["num_nodes"] != "1"
        or observed["num_cpus"] != "4"
    ):
        raise RuntimeError("audit allocation or exact producer-node exclusion is invalid")
    return observed


def _verify_producer_slurm(
    document: dict[str, object],
    *,
    expected_job: int,
    accounting: dict[str, object],
) -> dict[str, object]:
    recorded = document.get("slurm")
    expected = {
        "job_id": str(expected_job),
        "job_name": "evo-softkg-beam-accuracy",
        "node_list": accounting["node_list"],
        "partition": "standard",
        "account": "bio",
        "cpus_per_task": 4,
        "memory_per_node_mib": 32768,
        "cpu_affinity_count": 4,
    }
    _compare_tree(
        recorded,
        expected,
        path="slurm",
        numeric_stats=_new_numeric_stats(),
    )
    if type(recorded) is not dict:
        raise AssertionError("typed comparison accepted a non-object Slurm record")
    return recorded


def _verify_environment(document: dict[str, object]) -> None:
    environment = document.get("environment")
    expected_keys = {
        "python",
        "numpy",
        "scipy",
        "platform",
        "float64_itemsize",
        "longdouble_itemsize",
        "longdouble_mantissa_bits",
        "thread_controls",
    }
    if type(environment) is not dict or set(environment) != expected_keys:
        raise ValueError("producer environment has an unexpected schema")
    for name in ("python", "numpy", "scipy", "platform"):
        if type(environment[name]) is not str or not environment[name]:
            raise ValueError(f"producer environment field {name!r} must be non-empty text")
    if (
        environment["float64_itemsize"] != 8
        or type(environment["longdouble_itemsize"]) is not int
        or environment["longdouble_itemsize"] < 8
        or type(environment["longdouble_mantissa_bits"]) is not int
        or environment["longdouble_mantissa_bits"] < 52
        or environment["thread_controls"]
        != {
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
    ):
        raise ValueError("producer numeric ABI or thread controls are invalid")


def _reconstruct_gate(
    document: dict[str, object],
    *,
    spec: FrozenBeamAccuracySpec,
    independent_cases: list[dict[str, object]],
    summary_numeric_stats: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool], str, float, int]:
    summary, quality_checks = _summarize_cases(spec, independent_cases)
    _compare_tree(
        document.get("summary"),
        summary,
        path="summary",
        numeric_stats=summary_numeric_stats,
    )
    timing = document.get("timing_seconds")
    resources = document.get("resources")
    if type(timing) is not dict or set(timing) != {"kernel_cpu", "kernel_wall"}:
        raise ValueError("producer timing has an unexpected schema")
    if type(resources) is not dict or set(resources) != {
        "peak_rss_budget_kib",
        "peak_rss_kib_linux_ru_maxrss",
    }:
        raise ValueError("producer resources have an unexpected schema")
    kernel_wall = _finite_float(timing["kernel_wall"], name="kernel_wall")
    kernel_cpu = _finite_float(timing["kernel_cpu"], name="kernel_cpu")
    peak_rss = _exact_integer(
        resources["peak_rss_kib_linux_ru_maxrss"],
        name="peak_rss_kib_linux_ru_maxrss",
    )
    expected_rss_budget = spec.kernel_peak_rss_budget_gib * 1024 * 1024
    if (
        kernel_wall <= 0.0
        or kernel_cpu <= 0.0
        or peak_rss <= 0
        or resources["peak_rss_budget_kib"] != expected_rss_budget
    ):
        raise ValueError("producer timing or resource values are invalid")
    resource_checks = {
        "kernel_wall_budget": kernel_wall <= spec.kernel_wall_budget_seconds,
        "kernel_peak_rss_budget": peak_rss <= expected_rss_budget,
    }
    all_checks = {**quality_checks, **resource_checks}
    disposition = "accepted" if all(all_checks.values()) else "no_go"
    expected_gate = {
        "checks": all_checks,
        "passed": all(all_checks.values()),
        "scope": "synthetic_small_problem_exact_vs_bounded_beam_only",
        "scientific_or_production_claim": "none",
    }
    _compare_tree(
        document.get("gate"),
        expected_gate,
        path="gate",
        numeric_stats=_new_numeric_stats(),
    )
    if type(document.get("status")) is not str or document["status"] != disposition:
        raise ValueError("producer gate or accepted/no-go disposition does not reconstruct")
    return summary, all_checks, disposition, kernel_wall, peak_rss


def verify_producer(
    *,
    repo_root: Path,
    producer_root: Path,
    expected_producer_job: int,
    expected_producer_commit: str,
    expected_receipt_sha256: str,
    audit_commit: str,
) -> dict[str, object]:
    """Return a stable excluded-node semantic replay receipt."""

    if expected_producer_job <= 0:
        raise ValueError("expected_producer_job must be positive")
    if _COMMIT_PATTERN.fullmatch(expected_producer_commit) is None:
        raise ValueError("expected_producer_commit must be lowercase 40-hex")
    if _COMMIT_PATTERN.fullmatch(audit_commit) is None:
        raise ValueError("audit_commit must be lowercase 40-hex")
    if _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None:
        raise ValueError("expected_receipt_sha256 must be lowercase 64-hex")
    _verify_audit_checkout(repo_root, audit_commit)

    expected_root = _PRODUCER_BASE / str(expected_producer_job)
    if producer_root != expected_root:
        raise ValueError("producer root does not match the independently supplied job identity")
    receipt_path = producer_root / "receipt.json"
    manifest_path = producer_root / "SHA256SUMS"
    worktree_path = producer_root / "worktree-status.txt"
    artifact_descriptor, artifact_root_snapshot, artifact_files_by_name = (
        _capture_exact_artifact_inventory(
            producer_root,
            (receipt_path, manifest_path, worktree_path),
        )
    )
    os.close(artifact_descriptor)
    artifact_snapshots = {
        snapshot.relative_path: snapshot for snapshot in artifact_files_by_name.values()
    }
    receipt_snapshot = artifact_snapshots[Path("receipt.json")]
    manifest_snapshot = artifact_snapshots[Path("SHA256SUMS")]
    worktree_snapshot = artifact_snapshots[Path("worktree-status.txt")]
    if worktree_snapshot.payload != b"":
        raise ValueError("producer clean-worktree record is not empty")
    if receipt_snapshot.sha256 != expected_receipt_sha256:
        raise ValueError("producer receipt digest differs from the independently supplied value")

    source_descriptor, source_root_snapshot, source_snapshots = _capture_producer_sources()
    os.close(source_descriptor)

    expected_manifest_paths = (
        receipt_path,
        *(_PRODUCER_REPO_ROOT / path for path in _PRODUCER_SOURCE_PATHS),
        worktree_path,
    )
    manifest = _parse_manifest_bytes(manifest_snapshot.payload)
    if tuple(manifest) != expected_manifest_paths:
        raise ValueError("producer SHA256SUMS order or inventory differs from the frozen audit set")
    captured_by_path = {
        **{snapshot.path: snapshot for snapshot in artifact_snapshots.values()},
        **{snapshot.path: snapshot for snapshot in source_snapshots.values()},
    }
    for path, expected_digest in manifest.items():
        snapshot = captured_by_path.get(path)
        if snapshot is None or snapshot.sha256 != expected_digest:
            raise ValueError(f"producer manifest digest failed for {path}")
    _verify_git_bound_sources(
        repo_root,
        expected_producer_commit,
        manifest,
        source_snapshots,
    )

    document = _load_canonical_json_bytes(receipt_snapshot.payload)
    if set(document) != _EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("producer receipt has an unexpected top-level schema")
    spec, config_sha256 = _decode_frozen_accuracy_spec(source_snapshots[CONFIG_RELATIVE].payload)
    if (
        type(document["artifact"]) is not str
        or document["artifact"] != ARTIFACT
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or type(document["git_commit"]) is not str
        or document["git_commit"] != expected_producer_commit
        or type(document["config_sha256"]) is not str
        or document["config_sha256"] != config_sha256
        or type(document["limitations"]) is not list
        or document["limitations"] != _EXPECTED_LIMITATIONS
    ):
        raise ValueError("producer identity, config binding, or claim limits failed")
    _verify_recorded_spec(document, spec)
    _verify_environment(document)

    producer_accounting = _producer_accounting(expected_producer_job)
    producer_slurm = _verify_producer_slurm(
        document,
        expected_job=expected_producer_job,
        accounting=producer_accounting,
    )
    audit_slurm = _required_slurm_environment(spec)
    audit_job = str(audit_slurm["job_id"])
    audit_node = str(audit_slurm["node_list"])
    producer_node = str(producer_slurm["node_list"])
    if audit_node == producer_node:
        raise RuntimeError("accuracy audit must run on a different node from the producer")
    audit_control = _audit_job_control(
        audit_job,
        audit_node=audit_node,
        producer_node=producer_node,
    )

    producer_cases = document.get("cases")
    if type(producer_cases) is not list or len(producer_cases) != spec.case_count:
        raise ValueError("producer receipt does not contain the exact frozen case count")
    independent_cases = [_evaluate_case(spec, case_index) for case_index in range(spec.case_count)]
    case_numeric_stats = _new_numeric_stats()
    _compare_tree(
        producer_cases,
        independent_cases,
        path="cases",
        numeric_stats=case_numeric_stats,
    )
    producer_inventory = _case_inventory_sha256(producer_cases)
    independent_inventory = _case_inventory_sha256(independent_cases)
    if (
        type(document["case_inventory_sha256"]) is not str
        or _SHA256_PATTERN.fullmatch(document["case_inventory_sha256"]) is None
        or document["case_inventory_sha256"] != producer_inventory
        or producer_inventory != independent_inventory
    ):
        raise ValueError("case identity/input inventory does not match the stable replay")

    summary_numeric_stats = _new_numeric_stats()
    summary, checks, disposition, kernel_wall, peak_rss = _reconstruct_gate(
        document,
        spec=spec,
        independent_cases=independent_cases,
        summary_numeric_stats=summary_numeric_stats,
    )

    _revalidate_closed_snapshot_set(
        root_path=_PRODUCER_REPO_ROOT,
        root_snapshot=source_root_snapshot,
        snapshots=source_snapshots,
        label="producer source",
    )
    _revalidate_closed_snapshot_set(
        root_path=producer_root,
        root_snapshot=artifact_root_snapshot,
        snapshots=artifact_snapshots,
        label="producer artifact",
    )
    _verify_audit_checkout(repo_root, audit_commit)
    artifact_snapshot_sha256 = _snapshot_set_sha256(
        artifact_root_snapshot,
        artifact_snapshots,
    )
    source_snapshot_sha256 = _snapshot_set_sha256(
        source_root_snapshot,
        source_snapshots,
    )

    return {
        "artifact": AUDIT_ARTIFACT,
        "schema_version": 1,
        "status": "accepted",
        "finished_utc": datetime.now(UTC).isoformat(),
        "audit_git_commit": audit_commit,
        "producer": {
            "artifact": ARTIFACT,
            "job_id": str(expected_producer_job),
            "node_list": producer_node,
            "git_commit": expected_producer_commit,
            "receipt_sha256": expected_receipt_sha256,
            "sha256sums_sha256": manifest_snapshot.sha256,
            "artifact_snapshot_sha256": artifact_snapshot_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "slurm_accounting": producer_accounting,
        },
        "audit_slurm": {
            **audit_slurm,
            "authenticated_job_control": audit_control,
        },
        "checks": {
            "audit_exact_clean_commit_bound": True,
            "different_node": True,
            "producer_node_explicitly_excluded": True,
            "audit_allocation_authenticated": True,
            "producer_slurm_completion_authenticated": True,
            "producer_exact_commit_bound": True,
            "producer_clean_worktree_bound": True,
            "producer_root_exact_inventory_modes_and_links": True,
            "producer_root_and_path_identities_revalidated": True,
            "producer_artifact_contents_revalidated": True,
            "producer_source_root_and_path_identities_revalidated": True,
            "producer_source_contents_revalidated": True,
            "producer_receipt_hash_and_parse_use_same_stable_bytes": True,
            "producer_manifest_hash_and_parse_use_same_stable_bytes": True,
            "producer_receipt_canonical_duplicate_free_json": True,
            "producer_manifest_exact_order_inventory_and_rehash": True,
            "producer_sources_match_exact_commit_blobs": True,
            "frozen_config_redecoded_without_mutation": True,
            "all_forty_cases_stably_replayed_on_excluded_node": True,
            "reviewed_exhaustive_and_beam_implementation_replayed": True,
            "case_identities_inputs_hashes_and_selections_match": True,
            "full_exact_and_beam_score_hashes_match": True,
            "case_numerics_match_declared_tolerance": True,
            "summary_recomputed_by_reviewed_implementation": True,
            "frozen_quality_and_resource_gates_reconstructed": True,
            "accepted_or_no_go_disposition_reconstructed": True,
            "producer_timing_used_only_for_producer_resource_gate": True,
        },
        "observed": {
            "producer_disposition": disposition,
            "producer_gate_passed": all(checks.values()),
            "case_count": len(independent_cases),
            "positive_exact_cases": summary["positive_exact_cases"],
            "exact_selection_match_fraction": summary["positive_case_exact_match_fraction"],
            "optimum_reached_fraction": summary["positive_case_optimum_reached_fraction"],
            "mean_score_capture": summary["positive_case_mean_score_capture"],
            "p10_score_capture": summary["positive_case_p10_score_capture"],
            "worst_score_capture": summary["positive_case_worst_score_capture"],
            "producer_kernel_wall_seconds": kernel_wall,
            "producer_peak_rss_kib": peak_rss,
        },
        "reconstruction": {
            "relative_tolerance": _INDEPENDENT_RELATIVE_TOLERANCE,
            "absolute_tolerance": _INDEPENDENT_ABSOLUTE_TOLERANCE,
            "case_numeric_comparison": case_numeric_stats,
            "summary_numeric_comparison": summary_numeric_stats,
            "producer_case_inventory_sha256": producer_inventory,
            "replay_case_inventory_sha256": independent_inventory,
            "producer_case_results_sha256": _canonical_sha256(producer_cases),
            "replay_case_results_sha256": _canonical_sha256(independent_cases),
            "producer_summary_sha256": _canonical_sha256(document["summary"]),
            "replay_summary_sha256": _canonical_sha256(summary),
            "exact_canonical_case_results_match": producer_cases == independent_cases,
            "exact_canonical_summary_match": document["summary"] == summary,
        },
        "timing_evidence_scope": "producer_measurement_only_audit_replay_not_timed",
        "claim_scope": (
            "stable_excluded_node_replay_of_reviewed_exact_vs_beam_implementation_for_"
            "frozen_synthetic_small_problem_engineering_evidence_only"
        ),
        "limitations": [
            "synthetic_non_biological_gaussian_panel",
            "small_pool_ten_batch_five_not_campaign_q14",
            "audit_replay_timing_is_not_producer_timing_evidence",
            "excluded_node_replay_reuses_reviewed_producer_python_implementation",
            "no_peptide_model_or_oracle",
            "no_scientific_performance_claim",
            "no_production_pin_change",
        ],
        "scientific_or_production_claim": "none",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-job", type=int, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    document = verify_producer(
        repo_root=args.repo_root.resolve(strict=True),
        producer_root=args.producer_root.resolve(strict=True),
        expected_producer_job=args.producer_job,
        expected_producer_commit=args.expected_producer_commit,
        expected_receipt_sha256=args.expected_receipt_sha256,
        audit_commit=args.audit_commit,
    )
    _write_exclusive(args.output, document)


if __name__ == "__main__":
    main()

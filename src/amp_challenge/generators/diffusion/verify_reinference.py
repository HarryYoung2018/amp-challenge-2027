"""Verify exact native-v0 checkpoint re-inference against accepted evidence.

This is a current-code, clean-room *comparator*, not a second model
implementation.  It consumes two complete evaluation bundles produced by an
exact checkout of the accepted legacy code and requires every byte and mode to
match the immutable accepted bundle.  It intentionally imports none of the
native diffusion inference, evaluation driver, sampling, or bundle producer
modules.

The scientific receipt is path-free and records what the byte comparison
establishes.  A separate path-free operational receipt records the Slurm
topology supplied through a strictly validated preflight document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

_ARTIFACT = "native_categorical_diffusion_unconditional_v0"
_SCIENTIFIC_ARTIFACT = f"{_ARTIFACT}_checkpoint_reinference_v1_independent_verification"
_OPERATIONAL_ARTIFACT = f"{_ARTIFACT}_checkpoint_reinference_v1_operational_verification"
_STATUS = "passed_exact_accepted_code_reexecution"
_PROVENANCE_STATUS = "checkpoint_origin_exactly_reproduced_v1"
_DECISION_STATUS = "reproducible_no_go_v0"
_LEGACY_GIT_COMMIT = "3e81822a1e2fa2c5b0bfcbd6527d1e9ea31a8474"
_LEGACY_TREE_SHA1 = "a407b726724a96853909e4d8bd90787f9e7592b0"
_PACKAGING_VERSION = "26.3"
_PACKAGING_WHEEL_SHA256 = "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"
_PACKAGING_REQUIREMENT_SHA256 = "baf6a8f5ae8296f8449019a6e611c67633ff83d8cfb34321accf15d33a2ca268"
_PACKAGING_COMPATIBILITY_SOURCE = "legacy_uv_lock_dev_edge_hashed_wheel"
_PACKAGING_COMPATIBILITY_INSTALL = "uv_pip_offline_no_deps_only_binary_require_hashes"
_V0_CONFIG_SHA256 = "8242e589db8f35710e44fd63343444c3d5ce0f30626933be37561c4cbb49d764"
_ACCEPTED_MANIFEST_SHA256 = "cf53ef4b23f17ca6e38b9f7a7d491ca9ef6aac754647be8881a58dc743f521b5"
_ACCEPTED_TREE_SHA256 = "e5d9d2a65a4d4d87584e5b3941eb4feb1b89e70bfadf3809576a47b482702a1a"
_TRAINING_BINDING_SHA256 = "58cee997dc5249eaf6ad5e9545595f8770585afa7d62b3992333df5af4db7d79"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_LUSTRE_FID_RE = re.compile(r"\[0x[0-9a-f]+:0x[0-9a-f]+:0x[0-9a-f]+\]")
_GIT_REPOSITORY_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)

_EVALUATION_FILES = (
    "contract.toml",
    "training_bundle.sha256",
    "validation_corruptions.npz",
    "validation_token_stats.npz",
    "validation_metrics.json",
    "baseline_metrics.json",
    "length_plan.json",
    "raw_proposals.fasta",
    "candidate_ledger.jsonl",
    "sampling_metrics.json",
    "manifest.json",
)
_EVALUATION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "config_sha256",
        "git_commit",
        "seeds",
        "training_bundle_sha256",
        "validation",
        "baselines",
        "sampling",
        "gates",
        "decision_status",
        "artifacts",
    }
)
_ACCEPTED_ARTIFACT_SHA256 = {
    "baseline_metrics.json": "f78fcf761b79c76248d6f4a5d4f7ea53e7c56a4ce2df04e12cd7dac2c48ea2ed",
    "candidate_ledger.jsonl": "cd26499d827b755eefaa6895a49b0e55a5a31452b9b057f0cb92a0fce7764e58",
    "contract.toml": _V0_CONFIG_SHA256,
    "length_plan.json": "1e211d84326338b9f91fd3537c1054a3994e18da98f1ffad208b244e14e4b6d8",
    "raw_proposals.fasta": "95215a042ed53676cf2c75f923eef11bd8bc03f60e4cceb1fde5dfa9176b7d28",
    "sampling_metrics.json": "939302d943671022a5c8d3b037fd0add021241f40c02da9d823757085effccc1",
    "training_bundle.sha256": _TRAINING_BINDING_SHA256,
    "validation_corruptions.npz": "0072a3d1f8e057592af3dab7908e5632374ad55dec2cbcd37c33ea68194c0881",
    "validation_metrics.json": "190c4bac252d3cdd8bacfafdb614cbc1cc4121a3619610997e8122b9825a3e11",
    "validation_token_stats.npz": "ac4053ed82a0473646cf71bad2620a586a2fed74be397238d23bb92e9ef4b8b7",
}

_TRAINING_FILES = (
    "CODE_SHA256SUMS",
    "INPUT_SHA256SUMS",
    "contract.toml",
    "environment.json",
    "rng.json",
    "training_schedule.sha256",
    "model_final.safetensors",
    "training_trace.jsonl",
    "train_metrics.json",
    "manifest.json",
)
_TRAINING_LABELS = (
    "seed-42-primary",
    "seed-42-twin",
    "seed-43",
    "seed-44",
)
_TRAINING_SEEDS = {
    "seed-42-primary": 42,
    "seed-42-twin": 42,
    "seed-43": 43,
    "seed-44": 44,
}
_TRAINING_MANIFEST_SHA256 = {
    "seed-42-primary": "0dbc65c0af94847120e29ac4134adc17a4277c17214cc0474ab69f9f3d2d23bb",
    "seed-42-twin": "0dbc65c0af94847120e29ac4134adc17a4277c17214cc0474ab69f9f3d2d23bb",
    "seed-43": "e1d9dce1a1db38e5ad26b28bc3e6d01d9d8279fda96b728b8e1f7964294fc669",
    "seed-44": "ef4648947909ede6471bbd7c38c53e261ab4bf0d76cf8deb4fb414b0d15a6e8b",
}
_CHECKPOINT_FILE_SHA256 = {
    "seed-42-primary": "1468704c71999982d769cd09abfb2775e99ca830b53c57fefd9f3c2aa9ae4405",
    "seed-42-twin": "1468704c71999982d769cd09abfb2775e99ca830b53c57fefd9f3c2aa9ae4405",
    "seed-43": "0c9f4b8d3954d0f83f48f6379fdcd47a4fb36281822a3500d2d373f388a29fac",
    "seed-44": "32fbd8bd9bbfbbc7bd4dd1c030e7e00e9a42981e0c1d2d28a14d6fe9152cf719",
}
_CHECKPOINT_MODEL_LOGICAL_SHA256 = {
    "seed-42-primary": "fa439e72ed6dba2b6545adb111f76d77922b8c484ae6c67f43e43747e4d3e7b4",
    "seed-42-twin": "fa439e72ed6dba2b6545adb111f76d77922b8c484ae6c67f43e43747e4d3e7b4",
    "seed-43": "897e761499d94e24655d4a94f95576c588100436b8cd81336f9d5ee6b860d7ed",
    "seed-44": "7ee9417347e74f1240258de922919f708c628c0724647e3ebc990deda67cde36",
}
_CHECKPOINT_BOUND_SHA256 = {
    "seed-42-primary": "21effb759ac6c104d7ad5c612d68cd070328fac10807a19cd5780e0ffb76b6f7",
    "seed-42-twin": "21effb759ac6c104d7ad5c612d68cd070328fac10807a19cd5780e0ffb76b6f7",
    "seed-43": "3909cf358dbd87ed923e52ea786d1e0bffa183cd48b6a45062e45d5c07cb7740",
    "seed-44": "d518e9b7632b8c1e0b8f37c5489d0ca9851d299e4660ad24f0f5b6dcd37d63b4",
}
_TRAINING_BINDING_PAYLOAD = b"".join(
    f"{_TRAINING_MANIFEST_SHA256[label]}  {label}\n".encode("ascii") for label in _TRAINING_LABELS
)
_PRODUCER_EVIDENCE_FILES = (
    "0.ack",
    "0.receipt",
    "0.result",
    "1.ack",
    "1.receipt",
    "1.result",
)
_CHECKPOINT_BINDING_DOMAIN = (
    b"amp-challenge/native-categorical-diffusion/checkpoint-contract-binding/v1\0"
)
_EVALUATION_BUNDLE_DOMAIN = b"amp-native-diffusion-evaluation-bundle-v1\0"
_REINFERENCE_CONTRACT_ARTIFACT = "native_v0_checkpoint_origin_reinference_v1"
_ACCEPTED_LOGICAL_SHA256 = "744f73d7fbe000687148c91f49e48d6a726f2fb7f819dcf14766f482448aa70e"
_LEGACY_CODE_SHA256SUMS_SHA256 = "a2510fdc489f622016d44a39bcc755b19b8a1934e5f2868ea17cbf7813f4fd0c"
_LEGACY_UV_LOCK_SHA256 = "75d80f72a22dfc74b1da083615cebbe7c87d1438ba21d8967787c8c0b8bf36bb"
_INPUT_SHA256 = {
    "accepted_corpus_sha256": "c03595a8650b732307ada7e030f996d65345238efaea55d25925ac433bec8bc7",
    "training_projection_sha256": (
        "127a0eb88c5dc10c94904dcc5a3e98ff75a55890dc29af807e99f3f93c61ae46"
    ),
    "organizer_reference_sha256": (
        "cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9"
    ),
    "seed42_audit_receipt_sha256": (
        "9484a2047924ac720fceda2fb1ba005fa452b531ad33be429f07314c574b4639"
    ),
    "accepted_evaluation_independent_receipt_sha256": (
        "02f81ef97e52237f39035f91c2151b50fc79e2698e777065e67a8ed19dbe3e7b"
    ),
    "accepted_evaluation_operational_receipt_sha256": (
        "1a105b6a4e3c1e0bfa0bb5ed98426c21dcadd0f83aba9973c2e84a86d582267e"
    ),
}


class VerificationError(ValueError):
    """Raised when immutable, semantic, or operational evidence differs."""


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    return cast(str, value)


def _require_commit(value: object, *, label: str) -> str:
    _require(type(value) is str and _GIT_RE.fullmatch(value) is not None, f"{label} invalid")
    return cast(str, value)


def _exact_dict(value: object, fields: Iterable[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be an object")
    document = cast(dict[str, object], value)
    expected = set(fields)
    _require(set(document) == expected, f"{label} fields differ: expected {sorted(expected)}")
    return document


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _json_exact(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _strict_json(payload: bytes, *, label: str, pretty: bool = False) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} is not one LF-terminated JSON document",
    )

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            _require(key not in result, f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        _fail(f"{label} contains non-finite value {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not UTF-8 JSON") from error
    _require(type(value) is dict, f"{label} must be one object")
    _require(_canonical_json_bytes(value, pretty=pretty) == payload, f"{label} is not canonical")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]
    mode: int


@dataclass(frozen=True, slots=True)
class BundleAudit:
    root: Path
    snapshots: Mapping[str, Snapshot]
    tree_sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TrainingAudit:
    label: str
    seed: int
    root: Path
    snapshots: Mapping[str, Snapshot]
    manifest: Mapping[str, object]
    checkpoint_file_sha256: str
    checkpoint_model_logical_sha256: str
    checkpoint_bound_logical_sha256: str


@dataclass(frozen=True, slots=True)
class ReinferenceContract:
    snapshot: Snapshot
    document: Mapping[str, object]
    current_git_commit: str


@dataclass(frozen=True, slots=True)
class ProducerEvidenceAudit:
    root: Path
    snapshots: Mapping[str, Snapshot]
    job_id: int
    nodes: tuple[str, str]
    resources: Mapping[str, object]
    environment: Mapping[str, object]
    workers: tuple[Mapping[str, object], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class VerificationExecution:
    receipt_dir: Path
    independent_receipt: Path
    operational_receipt: Path
    independent_sha256: str
    operational_sha256: str
    decision_status: str


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path, *, label: str, include_leaf: bool = True) -> None:
    target = _absolute(path)
    candidates = (
        [*reversed(target.parents), target] if include_leaf else list(reversed(target.parents))
    )
    for candidate in candidates:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise VerificationError(f"cannot inspect {label}: {candidate}") from error
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symbolic link")


def _fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )


def _snapshot(
    path: str | Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> Snapshot:
    source = _absolute(path)
    _reject_symlink_chain(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise VerificationError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.lstat(source)
    finally:
        os.close(descriptor)
    _require(
        _fingerprint(before) == _fingerprint(after) == _fingerprint(named),
        f"{label} changed while read",
    )
    mode = stat.S_IMODE(before.st_mode)
    if required_mode is not None:
        _require(mode == required_mode, f"{label} mode must be {required_mode:04o}")
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} size changed while read")
    return Snapshot(
        path=source,
        payload=payload,
        sha256=_sha256(payload),
        fingerprint=_fingerprint(before),
        mode=mode,
    )


def _unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _snapshot(snapshot.path, label=label)
    _require(
        current.sha256 == snapshot.sha256
        and current.fingerprint == snapshot.fingerprint
        and current.mode == snapshot.mode,
        f"{label} changed during verification",
    )


def _read_bundle(
    path: str | Path,
    *,
    expected_files: Sequence[str],
    label: str,
) -> tuple[Path, dict[str, Snapshot]]:
    root = _absolute(path)
    _reject_symlink_chain(root, label=label)
    try:
        before = os.lstat(root)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}") from error
    _require(stat.S_ISDIR(before.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(before.st_mode) == 0o555, f"{label} mode must be 0555")
    children = tuple(sorted(root.iterdir(), key=lambda item: item.name))
    _require(
        tuple(item.name for item in children) == tuple(sorted(expected_files)),
        f"{label} inventory differs",
    )
    snapshots = {
        child.name: _snapshot(child, label=f"{label} artifact", required_mode=0o444)
        for child in children
    }
    after = os.lstat(root)
    _require(_fingerprint(before) == _fingerprint(after), f"{label} changed while read")
    return root, snapshots


def _tree_sha256(snapshots: Mapping[str, Snapshot]) -> str:
    payload = b"".join(
        f"{snapshots[name].mode:03o} {snapshots[name].sha256} {name}\n".encode("ascii")
        for name in sorted(snapshots)
    )
    return _sha256(payload)


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} is not canonical checksum text",
    )
    try:
        rows = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    prior: str | None = None
    for row in rows:
        _require(len(row) >= 67 and row[64:66] == "  ", f"{label} row malformed")
        digest, name = row[:64], row[66:]
        _require_sha256(digest, label=f"{label} digest")
        pure = PurePosixPath(name)
        _require(
            bool(name)
            and not name.startswith("/")
            and "\\" not in name
            and not pure.is_absolute()
            and all(part not in {"", ".", ".."} for part in pure.parts),
            f"{label} contains unsafe path",
        )
        _require(name not in result, f"{label} contains duplicate path")
        _require(prior is None or name > prior, f"{label} rows are not strictly ordered")
        result[name] = digest
        prior = name
    _require(bool(result), f"{label} cannot be empty")
    return result


def _run_git(repository: Path, *arguments: str) -> bytes:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Git verification failed: {' '.join(arguments)}") from error


def _verified_repository_root(repository_root: str | Path) -> Path:
    overrides = tuple(
        sorted(
            name
            for name in os.environ
            if name in _GIT_REPOSITORY_ENVIRONMENT or name.startswith("GIT_CONFIG_")
        )
    )
    _require(not overrides, f"Git repository-selection environment is forbidden: {overrides}")
    repository = _absolute(repository_root)
    _reject_symlink_chain(repository, label="repository")
    _require(repository.is_dir(), "repository_root must be a directory")
    top = _absolute(_run_git(repository, "rev-parse", "--show-toplevel").decode().strip())
    _require(top == repository, "repository_root must be the exact Git worktree top level")
    _require(
        not _run_git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"),
        "Git replacement refs are forbidden",
    )
    return repository


def _verify_repository_state(
    repository_root: str | Path,
    *,
    current_git_commit: str,
    legacy_git_commit: str,
) -> Path:
    current = _require_commit(current_git_commit, label="current_git_commit")
    legacy = _require_commit(legacy_git_commit, label="legacy_git_commit")
    _require(legacy == _LEGACY_GIT_COMMIT, "legacy Git commit differs from accepted execution")
    repository = _verified_repository_root(repository_root)
    _require(
        _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode().strip() == current,
        "repository HEAD differs from current_git_commit",
    )
    _require(
        _run_git(repository, "rev-parse", "--verify", "@{upstream}^{commit}").decode().strip()
        == current,
        "repository upstream differs from current_git_commit",
    )
    _require(
        _run_git(
            repository,
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main^{commit}",
        )
        .decode()
        .strip()
        == current,
        "cached origin/main differs from current_git_commit",
    )
    _require(
        _run_git(repository, "rev-parse", "--symbolic-full-name", "@{upstream}").decode().strip()
        == "refs/remotes/origin/main",
        "current branch upstream is not origin/main",
    )
    _require(
        not _run_git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        "current repository is not exactly clean",
    )
    resolved_legacy = (
        _run_git(repository, "rev-parse", "--verify", f"{legacy}^{{commit}}").decode().strip()
    )
    _require(resolved_legacy == legacy, "legacy Git commit cannot be resolved exactly")
    legacy_tree = (
        _run_git(repository, "rev-parse", "--verify", f"{legacy}^{{tree}}").decode().strip()
    )
    _require(legacy_tree == _LEGACY_TREE_SHA1, "legacy Git tree differs from accepted execution")
    _run_git(repository, "merge-base", "--is-ancestor", legacy, current)
    return repository


def _verify_legacy_code_manifest(
    entries: Mapping[str, str],
    *,
    repository: Path,
    legacy_git_commit: str,
) -> None:
    raw = _run_git(repository, "ls-tree", "-rz", "--full-tree", "-r", legacy_git_commit)
    tree: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, name_bytes = record.split(b"\t", maxsplit=1)
        mode, kind, object_id = metadata.decode("ascii").split(" ")
        name = name_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"} and kind == "blob",
            "legacy Git tree contains a non-regular entry",
        )
        tree[name] = (object_id, mode, kind)
    _require(set(tree) == set(entries), "training CODE_SHA256SUMS differs from legacy Git tree")
    for name, expected in entries.items():
        object_id = tree[name][0]
        payload = _run_git(repository, "cat-file", "blob", object_id)
        _require(_sha256(payload) == expected, f"legacy Git blob differs for {name}")


def _framed_update(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _bound_checkpoint_sha256(model_logical_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_CHECKPOINT_BINDING_DOMAIN)
    _framed_update(digest, bytes.fromhex(_V0_CONFIG_SHA256))
    _framed_update(digest, bytes.fromhex(model_logical_sha256))
    return digest.hexdigest()


def _expected_contract_document() -> dict[str, object]:
    checkpoint_key_by_label = {
        "seed-42-primary": "seed42_primary",
        "seed-42-twin": "seed42_twin",
        "seed-43": "seed43",
        "seed-44": "seed44",
    }
    files = dict(_ACCEPTED_ARTIFACT_SHA256)
    files["manifest.json"] = _ACCEPTED_MANIFEST_SHA256
    return {
        "schema_version": 1,
        "artifact": _REINFERENCE_CONTRACT_ARTIFACT,
        "evidence_doc": "docs/benchmarks/native_v0_checkpoint_reinference_v1.md",
        "status": {
            "before_execution": "predeclared_not_run",
            "evidence_invalid": "invalid_reinference",
            "exact_reproduction": "checkpoint_origin_exactly_reproduced_v1",
        },
        "scope": {
            "purpose": "close_native_v0_checkpoint_and_proposal_origin_limitations",
            "changes_native_v0_scientific_decision": False,
            "native_v0_decision": _DECISION_STATUS,
            "automatic_production_eligible": False,
            "independent_model_implementation": False,
            "comparison": "byte_and_mode_exact",
            "numeric_tolerance": 0.0,
        },
        "code": {
            "current_supervisor_branch": "main",
            "legacy_git_commit": _LEGACY_GIT_COMMIT,
            "legacy_tracked_files": 299,
            "legacy_code_sha256sums_sha256": _LEGACY_CODE_SHA256SUMS_SHA256,
            "legacy_uv_lock_sha256": _LEGACY_UV_LOCK_SHA256,
            "historical_tracking_ref": "private_capsule_only",
            "live_remote_ref_mutation": False,
            "loader_weakening_allowed": False,
        },
        "environment": {
            "base_extra": "diffusion",
            "base_install": "uv_sync_locked_offline_no_editable_legacy_capsule",
            "compatibility_install": _PACKAGING_COMPATIBILITY_INSTALL,
            "compatibility_package": "packaging",
            "compatibility_requirement_sha256": _PACKAGING_REQUIREMENT_SHA256,
            "compatibility_source": _PACKAGING_COMPATIBILITY_SOURCE,
            "compatibility_version": _PACKAGING_VERSION,
            "compatibility_wheel_sha256": _PACKAGING_WHEEL_SHA256,
        },
        "inputs": {"v0_contract_sha256": _V0_CONFIG_SHA256, **_INPUT_SHA256},
        "checkpoints": {
            checkpoint_key_by_label[label]: {
                "seed": _TRAINING_SEEDS[label],
                "manifest_sha256": _TRAINING_MANIFEST_SHA256[label],
                "file_sha256": _CHECKPOINT_FILE_SHA256[label],
                "logical_state_sha256": _CHECKPOINT_MODEL_LOGICAL_SHA256[label],
                "contract_bound_logical_sha256": _CHECKPOINT_BOUND_SHA256[label],
            }
            for label in _TRAINING_LABELS
        },
        "accepted_evaluation": {
            "manifest_sha256": _ACCEPTED_MANIFEST_SHA256,
            "logical_sha256": _ACCEPTED_LOGICAL_SHA256,
            "tree_sha256": _ACCEPTED_TREE_SHA256,
            "decision_status": _DECISION_STATUS,
            "validation_sequences": 199,
            "corruption_cases": 50944,
            "selected_tokens": 607976,
            "raw_proposals_per_method_seed": 2048,
            "total_raw_proposals": 18432,
            "files": files,
        },
        "execution": {
            "producer_nodes": 2,
            "producer_tasks": 2,
            "tasks_per_node": 1,
            "cpus_per_task": 8,
            "memory_gib_per_node": 32,
            "gpus_per_task": 1,
            "gpu": "NVIDIA A100-SXM4-80GB",
            "evaluation_batch_sequences": 256,
            "worker0_seed42_primary": "seed42_primary",
            "worker0_seed42_twin": "seed42_twin",
            "worker1_seed42_primary": "seed42_twin",
            "worker1_seed42_twin": "seed42_primary",
            "seed42_physical_bundle_roots_distinct": True,
            "seed42_physical_checkpoint_lustre_fids_distinct": True,
            "accepted_evaluation_supplied_to_producer": False,
            "legacy_capsule_import_probe_before_gpu": True,
            "single_exact_srun": True,
        },
        "audit": {
            "cpu_partition": "standard",
            "cpus": 8,
            "memory_gib": 32,
            "third_node_distinct_from_producers": True,
            "rerun_twins_byte_identical": True,
            "rerun_twins_mode_identical": True,
            "accepted_bundle_byte_identical": True,
            "accepted_bundle_mode_identical": True,
            "receipt_files": ["independent-verification.json", "operational-receipt.json"],
        },
        "decision": {
            "pass_requires_all_checks": True,
            "pass_closes_checkpoint_statistics_origin_limitation": True,
            "pass_closes_native_proposal_origin_limitation": True,
            "pass_closes_count_control_reexecution": True,
            "pass_promotes_native_v0": False,
            "pass_authorizes_fold4_tuning": False,
            "pass_authorizes_final_library_share": False,
        },
    }


def _load_contract(path: str | Path, *, current_git_commit: str) -> ReinferenceContract:
    snapshot = _snapshot(path, label="reinference contract")
    try:
        value = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("reinference contract is not valid UTF-8 TOML") from error
    _require(value == _expected_contract_document(), "reinference contract semantics differ")
    return ReinferenceContract(
        snapshot=snapshot,
        document=value,
        current_git_commit=_require_commit(current_git_commit, label="current_git_commit"),
    )


def _evaluation_logical_sha256(manifest: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_EVALUATION_BUNDLE_DOMAIN)
    _framed_update(digest, _canonical_json_bytes(dict(manifest)))
    return digest.hexdigest()


def _verify_training_bundle(
    path: str | Path,
    *,
    label: str,
    v0_contract: Snapshot,
) -> TrainingAudit:
    _require(label in _TRAINING_LABELS, "training bundle label is not frozen")
    root, snapshots = _read_bundle(
        path,
        expected_files=_TRAINING_FILES,
        label=f"training bundle {label}",
    )
    _require(
        snapshots["contract.toml"].payload == v0_contract.payload,
        f"training bundle {label} contract differs",
    )
    _require(
        snapshots["manifest.json"].sha256 == _TRAINING_MANIFEST_SHA256[label],
        f"training bundle {label} manifest hash differs",
    )
    manifest = _strict_json(
        snapshots["manifest.json"].payload,
        label=f"training bundle {label} manifest",
    )
    _exact_dict(
        manifest,
        {
            "schema_version",
            "artifact",
            "config_sha256",
            "git_commit",
            "seed",
            "corpus",
            "model",
            "rng",
            "training",
            "artifacts",
        },
        label=f"training bundle {label} manifest",
    )
    _require(manifest["schema_version"] == 1, f"training bundle {label} schema differs")
    _require(manifest["artifact"] == _ARTIFACT, f"training bundle {label} artifact differs")
    _require(
        manifest["config_sha256"] == _V0_CONFIG_SHA256,
        f"training bundle {label} config binding differs",
    )
    _require(
        manifest["git_commit"] == _LEGACY_GIT_COMMIT,
        f"training bundle {label} Git commit differs",
    )
    _require(manifest["seed"] == _TRAINING_SEEDS[label], f"training bundle {label} seed differs")
    artifacts = _exact_dict(
        manifest["artifacts"],
        set(_TRAINING_FILES) - {"manifest.json"},
        label=f"training bundle {label} artifacts",
    )
    for name in _TRAINING_FILES[:-1]:
        _require(
            artifacts[name] == snapshots[name].sha256,
            f"training bundle {label} artifact hash differs for {name}",
        )
    model = cast(dict[str, object], manifest["model"])
    _require(type(model) is dict, f"training bundle {label} model record invalid")
    checkpoint_file = _require_sha256(
        model.get("checkpoint_file_sha256"),
        label=f"training bundle {label} checkpoint file hash",
    )
    checkpoint_logical = _require_sha256(
        model.get("checkpoint_logical_state_sha256"),
        label=f"training bundle {label} checkpoint logical hash",
    )
    _require(
        model.get("checkpoint_format") == "safetensors",
        f"training bundle {label} checkpoint format differs",
    )
    _require(
        checkpoint_file
        == snapshots["model_final.safetensors"].sha256
        == _CHECKPOINT_FILE_SHA256[label],
        f"training bundle {label} checkpoint bytes differ",
    )
    _require(
        checkpoint_logical == _CHECKPOINT_MODEL_LOGICAL_SHA256[label],
        f"training bundle {label} checkpoint logical state differs",
    )
    checkpoint_bound = _bound_checkpoint_sha256(checkpoint_logical)
    _require(
        checkpoint_bound == _CHECKPOINT_BOUND_SHA256[label],
        f"training bundle {label} checkpoint contract binding differs",
    )
    return TrainingAudit(
        label=label,
        seed=_TRAINING_SEEDS[label],
        root=root,
        snapshots=snapshots,
        manifest=manifest,
        checkpoint_file_sha256=checkpoint_file,
        checkpoint_model_logical_sha256=checkpoint_logical,
        checkpoint_bound_logical_sha256=checkpoint_bound,
    )


def _verify_validation_checkpoint_bindings(payload: bytes) -> None:
    document = _strict_json(payload, label="accepted validation metrics")
    _exact_dict(
        document,
        {
            "schema_version",
            "config_sha256",
            "seeds",
            "method_order",
            "models",
            "bootstrap",
            "timestep_bins",
        },
        label="accepted validation metrics",
    )
    _require(document["schema_version"] == 1, "validation metrics schema differs")
    _require(document["config_sha256"] == _V0_CONFIG_SHA256, "validation config differs")
    _require(document["seeds"] == [42, 43, 44], "validation seed order differs")
    models = document["models"]
    _require(type(models) is list and len(models) == 3, "validation model records differ")
    expected = (
        (42, _CHECKPOINT_BOUND_SHA256["seed-42-primary"]),
        (43, _CHECKPOINT_BOUND_SHA256["seed-43"]),
        (44, _CHECKPOINT_BOUND_SHA256["seed-44"]),
    )
    for raw, (seed, binding) in zip(cast(list[object], models), expected, strict=True):
        model = _exact_dict(
            raw,
            {"seed", "checkpoint_contract_logical_sha256", "metrics"},
            label=f"validation model seed {seed}",
        )
        _require(
            model["seed"] == seed and model["checkpoint_contract_logical_sha256"] == binding,
            f"validation checkpoint binding differs for seed {seed}",
        )


def _verify_evaluation_bundle(
    path: str | Path,
    *,
    label: str,
    v0_contract: Snapshot,
) -> BundleAudit:
    root, snapshots = _read_bundle(path, expected_files=_EVALUATION_FILES, label=label)
    _require(snapshots["contract.toml"].payload == v0_contract.payload, f"{label} contract differs")
    expected_all = {**_ACCEPTED_ARTIFACT_SHA256, "manifest.json": _ACCEPTED_MANIFEST_SHA256}
    observed = {name: snapshots[name].sha256 for name in _EVALUATION_FILES}
    _require(observed == expected_all, f"{label} frozen artifact hashes differ")
    tree_sha256 = _tree_sha256(snapshots)
    _require(tree_sha256 == _ACCEPTED_TREE_SHA256, f"{label} tree hash differs")
    manifest = _strict_json(snapshots["manifest.json"].payload, label=f"{label} manifest")
    _exact_dict(manifest, _EVALUATION_MANIFEST_FIELDS, label=f"{label} manifest")
    _require(manifest["schema_version"] == 1, f"{label} schema differs")
    _require(manifest["artifact"] == _ARTIFACT, f"{label} artifact differs")
    _require(manifest["config_sha256"] == _V0_CONFIG_SHA256, f"{label} config binding differs")
    _require(manifest["git_commit"] == _LEGACY_GIT_COMMIT, f"{label} legacy commit differs")
    _require(manifest["seeds"] == [42, 43, 44], f"{label} seed cohort differs")
    _require(
        manifest["training_bundle_sha256"] == _TRAINING_BINDING_SHA256,
        f"{label} training binding differs",
    )
    _require(manifest["decision_status"] == _DECISION_STATUS, f"{label} decision differs")
    _require(
        manifest["artifacts"] == _ACCEPTED_ARTIFACT_SHA256,
        f"{label} manifest artifact map differs",
    )
    _require(
        snapshots["training_bundle.sha256"].payload == _TRAINING_BINDING_PAYLOAD,
        f"{label} training manifest binding payload differs",
    )
    _require(
        _evaluation_logical_sha256(manifest) == _ACCEPTED_LOGICAL_SHA256,
        f"{label} logical identity differs",
    )
    validation = cast(dict[str, object], manifest["validation"])
    sampling = cast(dict[str, object], manifest["sampling"])
    _require(type(validation) is dict and type(sampling) is dict, f"{label} census invalid")
    _require(
        validation.get("validation_sequences") == 199
        and validation.get("corruption_cases") == 50944
        and validation.get("selected_tokens") == 607976,
        f"{label} validation census differs",
    )
    _require(
        sampling.get("raw_proposals_per_method_seed") == 2048
        and sampling.get("total_raw_proposals") == 18432,
        f"{label} sampling census differs",
    )
    _verify_validation_checkpoint_bindings(snapshots["validation_metrics.json"].payload)
    return BundleAudit(root=root, snapshots=snapshots, tree_sha256=tree_sha256, manifest=manifest)


def _reject_path_values(value: object, *, label: str = "path-free document") -> None:
    if isinstance(value, Path):
        _fail(f"{label} contains a Path value")
    if type(value) is str:
        text = cast(str, value)
        _require(
            "/" not in text and "\\" not in text and not text.startswith("~") and "://" not in text,
            f"{label} contains a path-like string",
        )
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            _require(type(key) is str, f"{label} has a non-string key")
            _reject_path_values(item, label=label)
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _reject_path_values(item, label=label)


def _all_true(value: Mapping[str, object], *, label: str) -> None:
    _require(bool(value), f"{label} cannot be empty")
    _require(
        all(type(item) is bool and item for item in value.values()),
        f"{label} contains a false or non-boolean check",
    )


def _expected_production_resources() -> dict[str, object]:
    return {
        "account": "bio",
        "cpus_per_task": 8,
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpus_per_task": 1,
        "memory_per_node_mib": 32768,
        "nodes": 2,
        "partition": "gpumid",
        "tasks": 2,
        "tasks_per_node": 1,
    }


def _expected_runtime_compatibility() -> dict[str, object]:
    return {
        "install": _PACKAGING_COMPATIBILITY_INSTALL,
        "package": "packaging",
        "requirement_sha256": _PACKAGING_REQUIREMENT_SHA256,
        "source": _PACKAGING_COMPATIBILITY_SOURCE,
        "version": _PACKAGING_VERSION,
        "wheel_sha256": _PACKAGING_WHEEL_SHA256,
    }


def _expected_audit_resources() -> dict[str, object]:
    return {
        "account": "bio",
        "cpus_per_task": 8,
        "gpus": 0,
        "memory_per_node_mib": 32768,
        "nodes": 1,
        "partition": "standard",
        "tasks": 1,
    }


def _expected_preflight_checks() -> dict[str, bool]:
    return {
        "accepted_bundle_not_supplied_to_producers": True,
        "audit_job_and_node_distinct_from_production": True,
        "audit_runtime_on_slurm_compute": True,
        "current_repository_clean_synchronized_exact_tree_verified": True,
        "fresh_locked_offline_no_editable_environments_verified": True,
        "legacy_checkout_exact_tree_verified": True,
        "producer_resources_verified_from_slurm_accounting": True,
        "seed42_checkpoint_role_swap_verified": True,
        "spooled_launchers_attested": True,
        "two_full_evaluations_completed": True,
    }


def _lustre_fid(path: Path, *, label: str, directory: bool) -> str:
    _reject_symlink_chain(path, label=label)
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}") from error
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    _require(expected_kind(metadata.st_mode), f"{label} has the wrong file type")
    environment = dict(os.environ)
    environment.update({"LANG": "C", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            ("/usr/bin/lfs", "path2fid", os.fspath(path)),
            check=True,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"cannot derive {label} Lustre FID") from error
    try:
        output = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} Lustre FID output is not ASCII") from error
    matches = _LUSTRE_FID_RE.findall(output)
    _require(len(matches) == 1, f"{label} Lustre FID output is invalid")
    return matches[0]


def _verify_producer_evidence(
    reruns: Sequence[BundleAudit],
    training: Mapping[str, TrainingAudit],
    *,
    contract_sha256: str,
    current_git_commit: str,
    legacy_git_commit: str,
) -> ProducerEvidenceAudit:
    _require(len(reruns) == 2, "producer evidence requires exactly two reruns")
    job_root = reruns[0].root.parent
    _require(
        reruns[1].root.parent == job_root
        and reruns[0].root == job_root / "0"
        and reruns[1].root == job_root / "1",
        "rerun bundle roots must be ranks 0 and 1 under one production job",
    )
    _reject_symlink_chain(job_root, label="producer job root")
    try:
        job_before = os.lstat(job_root)
    except OSError as error:
        raise VerificationError("cannot inspect producer job root") from error
    _require(
        stat.S_ISDIR(job_before.st_mode) and stat.S_IMODE(job_before.st_mode) == 0o555,
        "producer job root mode must be 0555",
    )
    _require(
        tuple(sorted(item.name for item in job_root.iterdir())) == ("0", "1", "node-receipts"),
        "producer job root inventory differs",
    )
    _require(job_root.name.isascii() and job_root.name.isdecimal(), "producer job id invalid")
    job_id = int(job_root.name)
    _require(job_id > 0 and str(job_id) == job_root.name, "producer job id invalid")

    evidence_root, snapshots = _read_bundle(
        job_root / "node-receipts",
        expected_files=_PRODUCER_EVIDENCE_FILES,
        label="producer node evidence",
    )
    documents = {
        name: _strict_json(
            snapshot.payload,
            label=f"producer evidence {name}",
            pretty=True,
        )
        for name, snapshot in snapshots.items()
    }
    for name, document in documents.items():
        _reject_path_values(document, label=f"producer evidence {name}")

    expected_environment: dict[str, object] = {
        "fresh_job_scoped": True,
        "install": "uv_sync_locked_offline_no_editable_legacy_capsule",
        "legacy_code_sha256sums_sha256": _LEGACY_CODE_SHA256SUMS_SHA256,
        "runtime_compatibility": _expected_runtime_compatibility(),
        "scope": "diffusion",
        "uv_lock_sha256": _LEGACY_UV_LOCK_SHA256,
    }
    expected_inputs: dict[str, object] = {
        "accepted_corpus": _INPUT_SHA256["accepted_corpus_sha256"],
        "organizer_reference": _INPUT_SHA256["organizer_reference_sha256"],
        "reinference_contract": contract_sha256,
        "training_projection": _INPUT_SHA256["training_projection_sha256"],
    }
    resources = _expected_production_resources()
    training_fids = {
        label: {
            "root": _lustre_fid(
                training[label].root,
                label=f"training bundle {label} root",
                directory=True,
            ),
            "checkpoint": _lustre_fid(
                training[label].snapshots["model_final.safetensors"].path,
                label=f"training bundle {label} checkpoint",
                directory=False,
            ),
        }
        for label in ("seed-42-primary", "seed-42-twin")
    }
    _require(
        all(
            _LUSTRE_FID_RE.fullmatch(fid) is not None
            for identities in training_fids.values()
            for fid in identities.values()
        ),
        "seed-42 training Lustre FID format differs",
    )
    _require(
        training_fids["seed-42-primary"]["root"] != training_fids["seed-42-twin"]["root"],
        "seed-42 training roots must have distinct Lustre FIDs",
    )
    _require(
        training_fids["seed-42-primary"]["checkpoint"]
        != training_fids["seed-42-twin"]["checkpoint"],
        "seed-42 checkpoints must have distinct Lustre FIDs",
    )

    receipt_fields = {
        "accepted_evaluation_supplied_to_producer",
        "artifact",
        "bundle_index",
        "current_git_commit",
        "environment",
        "frozen_input_sha256",
        "legacy_git_commit",
        "node_name",
        "producer_job_id",
        "resources",
        "schema_version",
        "seed42_evaluation_checkpoint_lustre_fid",
        "seed42_evaluation_root_lustre_fid",
        "seed42_evaluation_source",
        "seed42_twin_binding_checkpoint_lustre_fid",
        "seed42_twin_binding_root_lustre_fid",
        "seed42_twin_binding_source",
    }
    result_fields = {
        "artifact",
        "bundle_index",
        "bundle_manifest_sha256",
        "bundle_tree_sha256",
        "current_git_commit",
        "decision_status",
        "legacy_git_commit",
        "logical_sha256",
        "node_name",
        "peak_gpu_memory_bytes",
        "producer_job_id",
        "schema_version",
        "seed42_evaluation_source",
        "status",
        "training_bundle_sha256",
    }
    expected_roles = (
        ("seed42_primary", "seed-42-primary", "seed42_twin", "seed-42-twin"),
        ("seed42_twin", "seed-42-twin", "seed42_primary", "seed-42-primary"),
    )
    nodes: list[str] = []
    workers: list[Mapping[str, object]] = []
    receipts: list[dict[str, object]] = []
    for rank, (source, source_label, twin, twin_label) in enumerate(expected_roles):
        receipt = _exact_dict(
            documents[f"{rank}.receipt"], receipt_fields, label=f"producer receipt {rank}"
        )
        _require(
            type(receipt["schema_version"]) is int and receipt["schema_version"] == 1,
            f"producer receipt {rank} schema differs",
        )
        _require(
            receipt["artifact"] == "native_v0_reinference_worker_receipt"
            and type(receipt["bundle_index"]) is int
            and receipt["bundle_index"] == rank
            and type(receipt["producer_job_id"]) is int
            and receipt["producer_job_id"] == job_id,
            f"producer receipt {rank} identity differs",
        )
        _require(
            receipt["current_git_commit"] == current_git_commit
            and receipt["legacy_git_commit"] == legacy_git_commit,
            f"producer receipt {rank} Git binding differs",
        )
        _require(
            type(receipt["accepted_evaluation_supplied_to_producer"]) is bool
            and receipt["accepted_evaluation_supplied_to_producer"] is False,
            f"producer receipt {rank} accepted-evaluation exclusion differs",
        )
        _require(
            _json_exact(receipt["environment"], expected_environment)
            and _json_exact(receipt["frozen_input_sha256"], expected_inputs)
            and _json_exact(receipt["resources"], resources),
            f"producer receipt {rank} frozen runtime or input binding differs",
        )
        node = receipt["node_name"]
        _require(
            type(node) is str and _NODE_RE.fullmatch(cast(str, node)) is not None,
            f"producer receipt {rank} node invalid",
        )
        nodes.append(cast(str, node))
        expected_identity_fields = {
            "seed42_evaluation_checkpoint_lustre_fid": training_fids[source_label]["checkpoint"],
            "seed42_evaluation_root_lustre_fid": training_fids[source_label]["root"],
            "seed42_evaluation_source": source,
            "seed42_twin_binding_checkpoint_lustre_fid": training_fids[twin_label]["checkpoint"],
            "seed42_twin_binding_root_lustre_fid": training_fids[twin_label]["root"],
            "seed42_twin_binding_source": twin,
        }
        _require(
            all(receipt[key] == value for key, value in expected_identity_fields.items()),
            f"producer receipt {rank} seed-42 physical role binding differs",
        )
        receipts.append(receipt)

        result = _exact_dict(
            documents[f"{rank}.result"], result_fields, label=f"producer result {rank}"
        )
        _require(
            type(result["schema_version"]) is int
            and result["schema_version"] == 1
            and result["artifact"] == "native_v0_reinference_worker_result"
            and type(result["bundle_index"]) is int
            and result["bundle_index"] == rank
            and type(result["producer_job_id"]) is int
            and result["producer_job_id"] == job_id,
            f"producer result {rank} identity differs",
        )
        _require(
            result["current_git_commit"] == current_git_commit
            and result["legacy_git_commit"] == legacy_git_commit
            and result["node_name"] == node
            and result["seed42_evaluation_source"] == source,
            f"producer result {rank} receipt binding differs",
        )
        _require(
            result["bundle_manifest_sha256"] == _ACCEPTED_MANIFEST_SHA256
            and result["bundle_tree_sha256"] == _ACCEPTED_TREE_SHA256
            and result["logical_sha256"] == _ACCEPTED_LOGICAL_SHA256
            and result["training_bundle_sha256"] == _TRAINING_BINDING_SHA256
            and result["decision_status"] == _DECISION_STATUS
            and result["status"] == "completed_exact_accepted_code_reexecution",
            f"producer result {rank} scientific identity differs",
        )
        peak = result["peak_gpu_memory_bytes"]
        _require(
            type(peak) is int and 0 <= cast(int, peak) <= 17179869184,
            f"producer result {rank} peak GPU memory invalid",
        )
        workers.append(
            {
                "bundle_index": rank,
                "bundle_manifest_sha256": result["bundle_manifest_sha256"],
                "bundle_tree_sha256": result["bundle_tree_sha256"],
                "node_name": node,
                **expected_identity_fields,
            }
        )
    _require(nodes[0] != nodes[1], "producer receipts must attest distinct nodes")

    ack_fields = {
        "artifact",
        "bundle_index",
        "observed_sibling_receipt_sha256",
        "producer_job_id",
        "schema_version",
    }
    for rank in (0, 1):
        ack = _exact_dict(documents[f"{rank}.ack"], ack_fields, label=f"producer ack {rank}")
        _require(
            type(ack["schema_version"]) is int
            and ack["schema_version"] == 1
            and ack["artifact"] == "native_v0_reinference_worker_ack"
            and type(ack["bundle_index"]) is int
            and ack["bundle_index"] == rank
            and type(ack["producer_job_id"]) is int
            and ack["producer_job_id"] == job_id
            and ack["observed_sibling_receipt_sha256"] == snapshots[f"{1 - rank}.receipt"].sha256,
            f"producer ack {rank} reciprocal receipt binding differs",
        )

    job_after = os.lstat(job_root)
    _require(_fingerprint(job_before) == _fingerprint(job_after), "producer job root changed")
    return ProducerEvidenceAudit(
        root=evidence_root,
        snapshots=snapshots,
        job_id=job_id,
        nodes=(nodes[0], nodes[1]),
        resources=resources,
        environment=expected_environment,
        workers=(workers[0], workers[1]),
    )


def _load_runtime_preflight(
    path: str | Path,
    *,
    current_git_commit: str,
    producer_evidence: ProducerEvidenceAudit,
) -> tuple[Snapshot, Mapping[str, object]]:
    snapshot = _snapshot(
        path,
        label="operational runtime preflight",
        required_mode=0o444,
    )
    value = _strict_json(
        snapshot.payload,
        label="operational runtime preflight",
        pretty=True,
    )
    _reject_path_values(value, label="operational runtime preflight")
    top = _exact_dict(
        value,
        {"schema_version", "production", "audit", "checks"},
        label="operational runtime preflight",
    )
    _require(
        type(top["schema_version"]) is int and top["schema_version"] == 1,
        "operational preflight schema differs",
    )
    production = _exact_dict(
        top["production"],
        {"job_id", "nodes", "resources", "environment", "legacy_checkout", "workers"},
        label="operational preflight production",
    )
    audit = _exact_dict(
        top["audit"],
        {"job_id", "node_name", "current_git_commit", "resources", "environment"},
        label="operational preflight audit",
    )
    _require(
        type(production["job_id"]) is int and cast(int, production["job_id"]) > 0,
        "production job_id invalid",
    )
    _require(
        production["job_id"] == producer_evidence.job_id,
        "production job_id differs from validated producer evidence",
    )
    _require(
        type(audit["job_id"]) is int and cast(int, audit["job_id"]) > 0,
        "audit job_id invalid",
    )
    nodes = production["nodes"]
    _require(
        type(nodes) is list
        and len(nodes) == 2
        and all(type(node) is str and _NODE_RE.fullmatch(cast(str, node)) for node in nodes)
        and nodes[0] != nodes[1],
        "production nodes must be two distinct safe names",
    )
    _require(
        nodes == list(producer_evidence.nodes),
        "production nodes differ from validated producer evidence",
    )
    _require(
        type(audit["node_name"]) is str
        and _NODE_RE.fullmatch(cast(str, audit["node_name"])) is not None
        and audit["node_name"] not in nodes,
        "audit node must be safe and distinct from production",
    )
    _require(audit["job_id"] != production["job_id"], "audit and production jobs must differ")
    _require(
        _json_exact(production["resources"], _expected_production_resources()),
        "production resources differ from the contract",
    )
    _require(
        _json_exact(production["resources"], producer_evidence.resources),
        "production resources differ from validated producer evidence",
    )
    _require(
        _json_exact(audit["resources"], _expected_audit_resources()),
        "audit resources differ from the contract",
    )
    _require(
        _json_exact(
            production["environment"],
            {
                "fresh_job_scoped": True,
                "install": "uv_sync_locked_offline_no_editable_legacy_capsule",
                "runtime_compatibility": _expected_runtime_compatibility(),
                "scope": "diffusion",
            },
        ),
        "production environment differs from the contract",
    )
    _require(
        _json_exact(
            production["environment"],
            {
                "fresh_job_scoped": producer_evidence.environment["fresh_job_scoped"],
                "install": producer_evidence.environment["install"],
                "runtime_compatibility": producer_evidence.environment["runtime_compatibility"],
                "scope": producer_evidence.environment["scope"],
            },
        ),
        "production environment differs from validated producer evidence",
    )
    _require(
        _json_exact(
            audit["environment"],
            {
                "fresh_job_scoped": True,
                "install": "uv_sync_locked_offline_no_editable_refreshed_project",
                "scope": "core",
            },
        ),
        "audit environment differs from the contract",
    )
    _require(
        _json_exact(
            production["legacy_checkout"],
            {
                "git_commit": _LEGACY_GIT_COMMIT,
                "git_tree_sha1": _LEGACY_TREE_SHA1,
                "tracked_files": 299,
                "uv_lock_sha256": _LEGACY_UV_LOCK_SHA256,
            },
        ),
        "legacy checkout attestation differs",
    )
    _require(
        audit["current_git_commit"] == current_git_commit,
        "audit current Git commit differs",
    )
    workers = production["workers"]
    _require(type(workers) is list and len(workers) == 2, "production worker records differ")
    expected_sources = (
        ("seed42_primary", "seed42_twin"),
        ("seed42_twin", "seed42_primary"),
    )
    observed_nodes: list[str] = []
    for index, (raw, sources) in enumerate(
        zip(cast(list[object], workers), expected_sources, strict=True)
    ):
        worker = _exact_dict(
            raw,
            {
                "bundle_index",
                "node_name",
                "bundle_manifest_sha256",
                "bundle_tree_sha256",
                "seed42_evaluation_source",
                "seed42_twin_binding_source",
                "seed42_evaluation_checkpoint_lustre_fid",
                "seed42_evaluation_root_lustre_fid",
                "seed42_twin_binding_checkpoint_lustre_fid",
                "seed42_twin_binding_root_lustre_fid",
            },
            label=f"production worker {index}",
        )
        _require(
            type(worker["bundle_index"]) is int and worker["bundle_index"] == index,
            f"production worker {index} index differs",
        )
        _require(
            type(worker["node_name"]) is str
            and _NODE_RE.fullmatch(cast(str, worker["node_name"])) is not None,
            f"production worker {index} node invalid",
        )
        observed_nodes.append(cast(str, worker["node_name"]))
        _require(
            worker["bundle_manifest_sha256"] == _ACCEPTED_MANIFEST_SHA256
            and worker["bundle_tree_sha256"] == _ACCEPTED_TREE_SHA256,
            f"production worker {index} output identity differs",
        )
        _require(
            (
                worker["seed42_evaluation_source"],
                worker["seed42_twin_binding_source"],
            )
            == sources,
            f"production worker {index} seed-42 role mapping differs",
        )
        _require(
            _json_exact(worker, producer_evidence.workers[index]),
            f"production worker {index} differs from validated producer evidence",
        )
    _require(observed_nodes == nodes, "worker nodes differ from production node order")
    checks = _exact_dict(
        top["checks"],
        _expected_preflight_checks(),
        label="operational preflight checks",
    )
    _require(checks == _expected_preflight_checks(), "operational preflight checks differ")
    _all_true(checks, label="operational preflight checks")
    return snapshot, value


def _bundle_bytes_and_modes_equal(left: BundleAudit, right: BundleAudit) -> bool:
    return all(
        left.snapshots[name].payload == right.snapshots[name].payload
        and left.snapshots[name].mode == right.snapshots[name].mode
        for name in _EVALUATION_FILES
    )


def _write_receipt(path: Path, value: Mapping[str, object]) -> Snapshot:
    _reject_path_values(value, label="verification receipt")
    payload = _canonical_json_bytes(dict(value), pretty=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as error:
        raise VerificationError(f"cannot publish verification receipt {path.name}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    snapshot = _snapshot(path, label="published verification receipt", required_mode=0o444)
    _require(snapshot.payload == payload, "published verification receipt bytes differ")
    return snapshot


def _prepare_receipt_dir(
    path: str | Path,
    *,
    protected_roots: Sequence[Path],
) -> Path:
    output = _absolute(path)
    _require(output.name not in {"", ".", ".."}, "receipt_dir must have a concrete name")
    _reject_symlink_chain(output, label="receipt output", include_leaf=False)
    _require(output.parent.is_dir(), "receipt_dir parent must already exist")
    _require(not os.path.lexists(output), "refusing to replace receipt_dir")
    for raw in protected_roots:
        root = raw.resolve(strict=True)
        _require(
            output != root and not output.is_relative_to(root) and not root.is_relative_to(output),
            "receipt_dir overlaps protected evidence",
        )
    try:
        os.mkdir(output, 0o700)
    except OSError as error:
        raise VerificationError("cannot create receipt_dir") from error
    return output


def _parse_training_bundle_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        _require(separator == "=" and bool(raw_path), "training bundle must be LABEL=PATH")
        _require(_LABEL_RE.fullmatch(label) is not None, "training bundle label is invalid")
        _require(label not in result, f"duplicate training bundle label {label}")
        result[label] = Path(raw_path)
    _require(
        set(result) == set(_TRAINING_LABELS),
        f"training bundle labels must be {list(_TRAINING_LABELS)}",
    )
    return result


def verify_native_diffusion_reinference(
    accepted_bundle: str | Path,
    rerun_bundles: Sequence[str | Path],
    training_bundles: Mapping[str, str | Path],
    *,
    contract_path: str | Path,
    repository_root: str | Path,
    current_git_commit: str,
    legacy_git_commit: str,
    operational_preflight: str | Path,
    receipt_dir: str | Path,
) -> VerificationExecution:
    """Verify two exact accepted-code reruns and publish immutable receipts."""

    current = _require_commit(current_git_commit, label="current_git_commit")
    legacy = _require_commit(legacy_git_commit, label="legacy_git_commit")
    _require(current != legacy, "current verifier commit must differ from the legacy commit")
    _require(len(rerun_bundles) == 2, "exactly two rerun bundles are required")
    _require(
        set(training_bundles) == set(_TRAINING_LABELS),
        f"training bundle labels must be {list(_TRAINING_LABELS)}",
    )
    repository = _verify_repository_state(
        repository_root,
        current_git_commit=current,
        legacy_git_commit=legacy,
    )

    expected_contract_path = (
        repository / "configs/diffusion/native_v0_checkpoint_reinference_v1.toml"
    )
    _require(
        _absolute(contract_path) == expected_contract_path,
        "contract must be the reviewed repository reinference contract",
    )
    contract = _load_contract(contract_path, current_git_commit=current)
    _require(
        _run_git(
            repository,
            "cat-file",
            "blob",
            f"{current}:configs/diffusion/native_v0_checkpoint_reinference_v1.toml",
        )
        == contract.snapshot.payload,
        "reinference contract differs from the current committed blob",
    )

    v0_contract = _snapshot(
        repository / "configs/diffusion/unconditional_v0.toml",
        label="native-v0 contract",
    )
    _require(v0_contract.sha256 == _V0_CONFIG_SHA256, "native-v0 contract hash differs")
    for commit in (legacy, current):
        _require(
            _run_git(
                repository,
                "cat-file",
                "blob",
                f"{commit}:configs/diffusion/unconditional_v0.toml",
            )
            == v0_contract.payload,
            f"native-v0 contract differs at Git commit {commit}",
        )
    _require(
        _sha256(_run_git(repository, "cat-file", "blob", f"{legacy}:uv.lock"))
        == _LEGACY_UV_LOCK_SHA256,
        "legacy uv.lock differs from the reinference contract",
    )

    training: dict[str, TrainingAudit] = {}
    for label in _TRAINING_LABELS:
        training[label] = _verify_training_bundle(
            training_bundles[label],
            label=label,
            v0_contract=v0_contract,
        )
    training_roots = [training[label].root.resolve(strict=True) for label in _TRAINING_LABELS]
    _require(
        len(set(training_roots)) == 4, "training bundle roots must be four distinct directories"
    )
    seed42_primary = training["seed-42-primary"]
    seed42_twin = training["seed-42-twin"]
    _require(
        all(
            seed42_primary.snapshots[name].payload == seed42_twin.snapshots[name].payload
            and seed42_primary.snapshots[name].mode == seed42_twin.snapshots[name].mode
            for name in _TRAINING_FILES
        ),
        "seed-42 primary and twin training bundles differ",
    )
    binding_payload = b"".join(
        f"{training[label].snapshots['manifest.json'].sha256}  {label}\n".encode("ascii")
        for label in _TRAINING_LABELS
    )
    _require(
        binding_payload == _TRAINING_BINDING_PAYLOAD
        and _sha256(binding_payload) == _TRAINING_BINDING_SHA256,
        "training cohort binding differs",
    )
    code_manifest_snapshot = training["seed-42-primary"].snapshots["CODE_SHA256SUMS"]
    _require(
        code_manifest_snapshot.sha256 == _LEGACY_CODE_SHA256SUMS_SHA256,
        "legacy code checksum manifest hash differs",
    )
    code_manifest = _parse_sha_manifest(
        code_manifest_snapshot.payload,
        label="legacy training CODE_SHA256SUMS",
    )
    _require(len(code_manifest) == 299, "legacy code checksum manifest census differs")
    _verify_legacy_code_manifest(
        code_manifest,
        repository=repository,
        legacy_git_commit=legacy,
    )
    _require(
        all(
            training[label].snapshots["CODE_SHA256SUMS"].payload == code_manifest_snapshot.payload
            for label in _TRAINING_LABELS
        ),
        "training bundles do not share the accepted legacy code manifest",
    )

    accepted = _verify_evaluation_bundle(
        accepted_bundle,
        label="accepted evaluation bundle",
        v0_contract=v0_contract,
    )
    reruns = tuple(
        _verify_evaluation_bundle(
            path,
            label=f"reinference bundle {index}",
            v0_contract=v0_contract,
        )
        for index, path in enumerate(rerun_bundles)
    )
    _require(
        _bundle_bytes_and_modes_equal(accepted, reruns[0])
        and _bundle_bytes_and_modes_equal(accepted, reruns[1])
        and _bundle_bytes_and_modes_equal(reruns[0], reruns[1]),
        "accepted and rerun evaluation bundles are not byte-and-mode identical",
    )
    all_evaluation_roots = [
        accepted.root.resolve(strict=True),
        *(item.root.resolve(strict=True) for item in reruns),
    ]
    _require(
        len(set(all_evaluation_roots)) == 3,
        "accepted and rerun bundles must be three distinct directories",
    )

    producer_evidence = _verify_producer_evidence(
        reruns,
        training,
        contract_sha256=contract.snapshot.sha256,
        current_git_commit=current,
        legacy_git_commit=legacy,
    )
    preflight_snapshot, preflight = _load_runtime_preflight(
        operational_preflight,
        current_git_commit=current,
        producer_evidence=producer_evidence,
    )
    production = cast(dict[str, object], preflight["production"])
    production_job = cast(int, production["job_id"])
    _require(
        reruns[0].root.parent == reruns[1].root.parent
        and reruns[0].root.name == "0"
        and reruns[1].root.name == "1"
        and reruns[0].root.parent.name == str(production_job),
        "rerun bundle paths do not bind the attested production job and ranks",
    )

    protected_roots = [repository, *training_roots, *all_evaluation_roots]
    output = _prepare_receipt_dir(receipt_dir, protected_roots=protected_roots)
    checkpoint_records = [
        {
            "checkpoint_contract_logical_sha256": training[label].checkpoint_bound_logical_sha256,
            "checkpoint_file_sha256": training[label].checkpoint_file_sha256,
            "checkpoint_logical_state_sha256": training[label].checkpoint_model_logical_sha256,
            "label": label,
            "manifest_sha256": training[label].snapshots["manifest.json"].sha256,
            "seed": training[label].seed,
        }
        for label in _TRAINING_LABELS
    ]
    accepted_file_hashes = {name: accepted.snapshots[name].sha256 for name in _EVALUATION_FILES}
    scientific_checks: dict[str, object] = {
        "accepted_bundle_exact_frozen_hashes_inventory_and_modes": True,
        "accepted_code_full_checkpoint_inference_reexecuted_twice": True,
        "accepted_legacy_code_manifest_matches_exact_git_tree": True,
        "checkpoint_files_manifests_and_logical_bindings_verified": True,
        "count_control_outputs_reproduced_byte_for_byte": True,
        "native_proposals_reproduced_byte_for_byte": True,
        "native_validation_token_statistics_reproduced_byte_for_byte": True,
        "reinference_contract_semantics_and_current_blob_verified": True,
        "rerun_twins_and_accepted_bundle_byte_and_mode_identical": True,
        "seed42_primary_twin_role_swap_exercised": True,
        "seed42_root_and_checkpoint_lustre_fids_verified": True,
        "training_bundle_cohort_and_seed42_exact_twin_verified": True,
        "v0_no_go_decision_preserved": True,
    }
    _all_true(scientific_checks, label="scientific checks")
    scientific_receipt: dict[str, object] = {
        "artifact": _SCIENTIFIC_ARTIFACT,
        "automatic_production_eligible": False,
        "bundle_identity": {
            "files": accepted_file_hashes,
            "logical_sha256": _ACCEPTED_LOGICAL_SHA256,
            "manifest_sha256": _ACCEPTED_MANIFEST_SHA256,
            "tree_sha256": _ACCEPTED_TREE_SHA256,
        },
        "checks": scientific_checks,
        "checkpoint_bindings": checkpoint_records,
        "conclusions": {
            "checkpoint_statistics_origin_limitation_closed": True,
            "count_control_reexecution_limitation_closed": True,
            "native_proposal_origin_limitation_closed": True,
            "native_v0_promoted": False,
        },
        "current_git_commit": current,
        "decision_status": _DECISION_STATUS,
        "legacy_git_commit": legacy,
        "provenance_status": _PROVENANCE_STATUS,
        "limitations": [
            "the comparator is independent of producer modules but is not an independent model implementation",
            "the reruns execute the exact accepted legacy evaluator and therefore test reproducibility rather than implementation diversity",
            "the accepted evaluation is absent from declared producer inputs and commands; the shared filesystem is not a security boundary",
            "the native v0 scientific decision remains no go and does not authorize production use or fold four tuning",
        ],
        "reinference_contract_sha256": contract.snapshot.sha256,
        "reproduction": {
            "accepted_bundle_count": 1,
            "comparison": "byte_and_mode_exact",
            "numeric_tolerance": 0.0,
            "rerun_bundle_count": 2,
            "seed42_checkpoint_role_swap": True,
        },
        "schema_version": 1,
        "status": _STATUS,
    }
    independent_snapshot = _write_receipt(
        output / "independent-verification.json",
        scientific_receipt,
    )

    operational_checks: dict[str, object] = {
        "accepted_evaluation_was_not_supplied_to_producers": True,
        "audit_job_and_node_distinct_from_producers": True,
        "current_repository_clean_synchronized_and_exact": True,
        "exact_two_node_two_task_gpu_execution_attested": True,
        "fresh_locked_noneditable_environments_attested": True,
        "legacy_private_capsule_commit_tree_and_lock_attested": True,
        "preflight_and_worker_output_bindings_verified": True,
        "producer_receipts_acks_and_results_exactly_verified": True,
        "receipt_directory_and_files_published_without_replacement": True,
        "spooled_producer_and_audit_launchers_attested": True,
    }
    _all_true(operational_checks, label="operational checks")
    operational_receipt: dict[str, object] = {
        "artifact": _OPERATIONAL_ARTIFACT,
        "automatic_production_eligible": False,
        "bundle_identity": {
            "manifest_sha256": _ACCEPTED_MANIFEST_SHA256,
            "tree_sha256": _ACCEPTED_TREE_SHA256,
        },
        "checks": operational_checks,
        "current_git_commit": current,
        "decision_status": _DECISION_STATUS,
        "independent_receipt_sha256": independent_snapshot.sha256,
        "legacy_git_commit": legacy,
        "provenance_status": _PROVENANCE_STATUS,
        "reinference_contract_sha256": contract.snapshot.sha256,
        "runtime": {
            "audit": preflight["audit"],
            "production": preflight["production"],
        },
        "runtime_preflight_sha256": preflight_snapshot.sha256,
        "producer_evidence_sha256": {
            name: producer_evidence.snapshots[name].sha256 for name in _PRODUCER_EVIDENCE_FILES
        },
        "schema_version": 1,
        "status": _STATUS,
    }
    operational_snapshot = _write_receipt(
        output / "operational-receipt.json",
        operational_receipt,
    )
    os.chmod(output, 0o555, follow_symlinks=False)

    input_snapshots = [
        contract.snapshot,
        v0_contract,
        preflight_snapshot,
        *(snapshot for snapshot in producer_evidence.snapshots.values()),
        *(snapshot for item in training.values() for snapshot in item.snapshots.values()),
        *(snapshot for snapshot in accepted.snapshots.values()),
        *(snapshot for item in reruns for snapshot in item.snapshots.values()),
    ]
    for snapshot in input_snapshots:
        _unchanged(snapshot, label="verification input")
    _unchanged(independent_snapshot, label="independent receipt")
    _unchanged(operational_snapshot, label="operational receipt")
    output_children = tuple(sorted(output.iterdir(), key=lambda item: item.name))
    _require(
        tuple(item.name for item in output_children)
        == ("independent-verification.json", "operational-receipt.json")
        and stat.S_IMODE(output.stat(follow_symlinks=False).st_mode) == 0o555
        and all(
            not item.is_symlink()
            and stat.S_ISREG(item.stat(follow_symlinks=False).st_mode)
            and stat.S_IMODE(item.stat(follow_symlinks=False).st_mode) == 0o444
            for item in output_children
        ),
        "receipt publication inventory or modes differ",
    )
    return VerificationExecution(
        receipt_dir=output.resolve(strict=True),
        independent_receipt=independent_snapshot.path,
        operational_receipt=operational_snapshot.path,
        independent_sha256=independent_snapshot.sha256,
        operational_sha256=operational_snapshot.sha256,
        decision_status=_DECISION_STATUS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-bundle", type=Path, required=True)
    parser.add_argument("--rerun-bundle", action="append", type=Path, required=True)
    parser.add_argument("--training-bundle", action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--legacy-git-commit", required=True)
    parser.add_argument("--operational-preflight", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        training_bundles = _parse_training_bundle_arguments(arguments.training_bundle)
        execution = verify_native_diffusion_reinference(
            arguments.accepted_bundle,
            arguments.rerun_bundle,
            training_bundles,
            contract_path=arguments.contract,
            repository_root=arguments.repository_root,
            current_git_commit=arguments.current_git_commit,
            legacy_git_commit=arguments.legacy_git_commit,
            operational_preflight=arguments.operational_preflight,
            receipt_dir=arguments.receipt_dir,
        )
    except (OSError, UnicodeError, ValueError, RuntimeError, tomllib.TOMLDecodeError) as error:
        print(f"AMP native diffusion reinference verification error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "automatic_production_eligible": False,
                "decision_status": execution.decision_status,
                "independent_receipt_sha256": execution.independent_sha256,
                "operational_receipt_sha256": execution.operational_sha256,
                "provenance_status": _PROVENANCE_STATUS,
                "status": _STATUS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

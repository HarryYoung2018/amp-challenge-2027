"""Independent verifier for the native-diffusion v0 cohort evaluation.

This module is a clean-room consumer of the frozen contract and published
artifacts.  In particular, it does not import any training, inference,
evaluation, sampling, or evaluation-bundle producer module.  Stored model
token statistics are sufficient to reconstruct the validation metrics and
gates, while the three count baselines are independently refit and rescored.
The final checkpoints are schema- and hash-verified but are deliberately not
run for a second inference pass, and native proposals are not regenerated from
them; those explicit limitations are recorded in the independent receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance.Indel import normalized_similarity as indel_ratio
from torch import nn

from amp_challenge.descriptors import compute_descriptors
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import global_sequence_identity

from .categorical import AbsorbingDiffusion, CosineMaskSchedule, PeptideVocabulary
from .contract import (
    CONFIG_SHA256,
    NativeDiffusionContract,
    load_unconditional_v0_contract,
)
from .data import (
    DiffusionCorpusRow,
    TrainingDistribution,
    TrainingRow,
    load_native_diffusion_corpus,
    load_training_projection,
    namespaced_seed,
)
from .model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    load_safetensors_checkpoint,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_DRIVER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
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
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_PAD = 20
_MASK = 21
_CALIBRATION_BINS = 15
_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
_CASE_DOMAIN = b"amp-challenge/native-categorical-diffusion/validation-case/v1\0"
_BOOTSTRAP_DOMAIN = b"amp-native-diffusion-cohort-bootstrap-draws-v1\0"
_SCHEDULE_DOMAIN = b"amp-native-diffusion-learning-rate-schedule-v1\0"
_CHECKPOINT_BINDING_DOMAIN = (
    b"amp-challenge/native-categorical-diffusion/checkpoint-contract-binding/v1\0"
)
_CONTROL_LOGICAL_DOMAIN = b"amp-challenge/native-categorical-diffusion/control-logical/v1\0"
_CONTROL_DRAW_DOMAIN = b"amp-challenge/native-categorical-diffusion/control-draw/v1\0"
_PROBABILITY_TABLE_DOMAIN = b"amp-challenge/native-categorical-diffusion/count-tables/v1\0"
_TRAIN_IDS_DOMAIN = b"amp-challenge/native-categorical-diffusion/baseline-train-ids/v1\0"
_SEQUENCE_COLLECTION_DOMAIN = b"amp-challenge/native-categorical-diffusion/sequence-collection/v1\0"
_CLUSTER_PREFILTER_MARGIN = 1e-12
_DESCRIPTOR_FEATURES = (
    "length",
    "molecular_weight_da",
    "net_charge",
    "charge_density",
    "isoelectric_point",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "hydrophobic_fraction",
    "aromatic_fraction",
    "basic_fraction",
    "acidic_fraction",
    "shannon_entropy",
    "max_residue_fraction",
)
_TWIN_ENVIRONMENT_FIELDS = (
    "python",
    "numpy",
    "torch",
    "torch_cuda",
    "safetensors",
    "triton",
    "nvidia_cudnn_cu13",
    "gpu_name",
    "compute_capability",
    "driver",
    "allocator",
)
_TRAIN_METRIC_FIELDS = frozenset(
    {
        "schema_version",
        "steps",
        "drawn_sequences",
        "total_selected_tokens",
        "final_loss",
        "mean_loss",
        "final_mean_row_accuracy",
        "mean_row_accuracy",
        "mean_gradient_norm_before_clipping",
        "final_learning_rate",
        "loss_reduction",
        "sampling_weight_application",
        "validation_consulted",
        "peak_gpu_memory_bytes",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "schema_version",
        "step",
        "loss",
        "mean_row_accuracy",
        "gradient_norm_before_clipping",
        "learning_rate",
        "selected_tokens",
        "dropout_seed",
        "batch_sha256",
    }
)


class VerificationError(ValueError):
    """Raised when any immutable or scientific verification invariant fails."""


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA-256"
    )
    return cast(str, value)


def _require_git(value: object, *, label: str = "git_commit") -> str:
    _require(type(value) is str and _GIT_RE.fullmatch(value) is not None, f"{label} is invalid")
    return cast(str, value)


def _finite(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    return cast(float, value)


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return cast(int, value)


def _exact_keys(value: object, expected: Iterable[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a JSON object")
    result = cast(dict[str, object], value)
    expected_set = set(expected)
    _require(
        set(result) == expected_set, f"{label} schema differs: expected {sorted(expected_set)}"
    )
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(payload: bytes, *, label: str) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} is not canonical LF-terminated JSON",
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
    _require(type(value) is dict, f"{label} must be one JSON object")
    _require(_canonical_json_bytes(value) == payload, f"{label} bytes are not canonical")
    return cast(dict[str, object], value)


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} is not canonical JSONL",
    )
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload[:-1].split(b"\n"), start=1):
        rows.append(_strict_json(line + b"\n", label=f"{label} row {index}"))
    _require(bool(rows), f"{label} cannot be empty")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path, *, label: str, leaf: bool = True) -> None:
    target = _absolute(path)
    values = [*reversed(target.parents), target] if leaf else list(reversed(target.parents))
    for value in values:
        try:
            mode = os.lstat(value).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise VerificationError(f"cannot inspect {label}: {value}") from error
        _require(not stat.S_ISLNK(mode), f"{label} traverses a symbolic link: {value}")


def _read_snapshot(path: str | Path, *, label: str) -> Snapshot:
    source = _absolute(path)
    _reject_symlink_chain(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise VerificationError(f"cannot open {label}: {source}") from error
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
    identity = lambda item: (  # noqa: E731 - compact immutable identity helper
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        stat.S_IMODE(item.st_mode),
    )
    _require(identity(before) == identity(after) == identity(named), f"{label} changed while read")
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} size changed while read")
    return Snapshot(
        path=source,
        payload=payload,
        sha256=_sha256(payload),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        mode=stat.S_IMODE(before.st_mode),
    )


def _snapshot_unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _read_snapshot(snapshot.path, label=label)
    _require(
        (
            current.sha256,
            current.device,
            current.inode,
            current.size,
            current.mtime_ns,
            current.ctime_ns,
            current.mode,
        )
        == (
            snapshot.sha256,
            snapshot.device,
            snapshot.inode,
            snapshot.size,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
            snapshot.mode,
        ),
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
        root_before = os.lstat(root)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}: {root}") from error
    _require(stat.S_ISDIR(root_before.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(root_before.st_mode) == 0o555, f"{label} directory mode must be 0555")
    children = tuple(sorted(root.iterdir(), key=lambda item: item.name))
    _require(
        tuple(item.name for item in children) == tuple(sorted(expected_files)),
        f"{label} inventory differs from the contract",
    )
    snapshots: dict[str, Snapshot] = {}
    for child in children:
        snapshot = _read_snapshot(child, label=f"{label}/{child.name}")
        _require(snapshot.mode == 0o444, f"{label}/{child.name} mode must be 0444")
        snapshots[child.name] = snapshot
    root_after = os.lstat(root)
    _require(
        (
            root_before.st_dev,
            root_before.st_ino,
            root_before.st_mtime_ns,
            root_before.st_ctime_ns,
            root_before.st_mode,
        )
        == (
            root_after.st_dev,
            root_after.st_ino,
            root_after.st_mtime_ns,
            root_after.st_ctime_ns,
            root_after.st_mode,
        ),
        f"{label} directory changed while read",
    )
    return root, snapshots


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} is not canonical checksum text",
    )
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    prior: str | None = None
    for line in lines:
        _require(len(line) >= 67 and line[64:66] == "  ", f"{label} row is malformed")
        digest, name = line[:64], line[66:]
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
        raise VerificationError(f"Git verification failed: git {' '.join(arguments)}") from error


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
    top_level = _absolute(
        _run_git(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    )
    _require(top_level == repository, "repository_root must be the exact Git worktree top level")
    if _run_git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise VerificationError("repository-local Git replacement refs are forbidden")
    return repository


def _verify_code_manifest(
    entries: Mapping[str, str],
    *,
    git_commit: str,
    repository_root: str | Path,
) -> None:
    repository = _verified_repository_root(repository_root)
    resolved = (
        _run_git(repository, "rev-parse", "--verify", f"{git_commit}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    _require(resolved == git_commit, "training Git commit cannot be resolved exactly")
    raw = _run_git(repository, "ls-tree", "-rz", "--full-tree", "-r", git_commit)
    tree: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, name_bytes = record.split(b"\t", maxsplit=1)
        mode, kind, object_id = metadata.decode("ascii").split(" ")
        name = name_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"} and kind == "blob", "Git tree contains non-regular entry"
        )
        tree[name] = (object_id, mode)
    _require(set(tree) == set(entries), "CODE_SHA256SUMS inventory differs from the Git tree")
    for name in sorted(tree):
        payload = _run_git(repository, "cat-file", "blob", tree[name][0])
        _require(_sha256(payload) == entries[name], f"CODE_SHA256SUMS mismatch for {name}")


def _verify_repository_state(repository_root: str | Path, git_commit: str) -> None:
    _require(
        type(git_commit) is str and _GIT_RE.fullmatch(git_commit) is not None, "invalid Git commit"
    )
    repository = _verified_repository_root(repository_root)
    observed = {
        "HEAD": _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}")
        .decode("ascii")
        .strip(),
        "upstream": _run_git(repository, "rev-parse", "--verify", "@{upstream}^{commit}")
        .decode("ascii")
        .strip(),
        "origin/main": _run_git(
            repository,
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main^{commit}",
        )
        .decode("ascii")
        .strip(),
    }
    _require(
        set(observed.values()) == {git_commit},
        "repository HEAD, upstream, and cached origin/main differ from the evaluated Git commit",
    )
    _require(
        _run_git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        == b"",
        "repository worktree is not clean",
    )


def _learning_rate(contract: NativeDiffusionContract, step: int) -> float:
    training = contract.training
    if training.warmup_steps and step <= training.warmup_steps:
        return training.learning_rate * step / training.warmup_steps
    progress = (step - training.warmup_steps) / (training.max_steps - training.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        training.final_learning_rate
        + (training.learning_rate - training.final_learning_rate) * cosine
    )


def _schedule_sha256(contract: NativeDiffusionContract) -> str:
    digest = hashlib.sha256()
    digest.update(_SCHEDULE_DOMAIN)
    for step in range(1, contract.training.max_steps + 1):
        digest.update(step.to_bytes(8, "big"))
        encoded = _learning_rate(contract, step).hex().encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _model_config(contract: NativeDiffusionContract) -> NativeDenoiserConfig:
    return NativeDenoiserConfig(
        layers=contract.model.layers,
        hidden_dim=contract.model.hidden_dim,
        attention_heads=contract.model.attention_heads,
        ffn_dim=contract.model.ffn_dim,
        dropout=contract.model.dropout,
        layer_norm_epsilon=contract.model.layer_norm_epsilon,
        min_length=contract.model.min_length,
        max_length=contract.model.max_length,
        levels=contract.diffusion.levels,
    )


def _parameter_groups(model: NativeDenoiser) -> tuple[tuple[str, ...], tuple[str, ...]]:
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding | nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    decay: list[str] = []
    no_decay: list[str] = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        (no_decay if name.endswith("bias") or id(parameter) in no_decay_ids else decay).append(name)
    _require(bool(decay) and bool(no_decay), "model parameter partition is empty")
    return tuple(decay), tuple(no_decay)


@dataclass(frozen=True, slots=True)
class TrainingBundleAudit:
    label: str
    seed: int
    git_commit: str
    manifest_sha256: str
    checkpoint_file_sha256: str
    checkpoint_logical_sha256: str
    environment_identity: tuple[tuple[str, object], ...]
    file_hashes: tuple[tuple[str, str], ...]
    snapshots: tuple[Snapshot, ...]


def _verify_training_environment(
    document: dict[str, object],
    *,
    contract: NativeDiffusionContract,
    seed: int,
) -> tuple[tuple[str, object], ...]:
    expected_keys = {
        "schema_version",
        *_TWIN_ENVIRONMENT_FIELDS,
        "device_type",
        "twin_environment_equal_fields",
        "runtime",
        "amp",
        "tf32",
        "torch_compile",
    }
    value = _exact_keys(document, expected_keys, label="environment.json")
    expected_identity: dict[str, object] = {
        "python": contract.environment.python,
        "numpy": contract.environment.numpy,
        "torch": contract.environment.torch,
        "torch_cuda": contract.environment.torch_cuda,
        "safetensors": contract.environment.safetensors,
        "triton": contract.environment.triton,
        "nvidia_cudnn_cu13": contract.environment.nvidia_cudnn_cu13,
        "gpu_name": contract.environment.gpu_name,
        "compute_capability": list(contract.environment.compute_capability),
        "allocator": contract.determinism.pytorch_allocator,
    }
    for name, expected in expected_identity.items():
        _require(value[name] == expected, f"training environment {name} differs from contract")
    _require(
        type(value["driver"]) is str
        and _DRIVER_RE.fullmatch(cast(str, value["driver"])) is not None,
        "training driver version is invalid",
    )
    _require(value["schema_version"] == 1, "training environment schema_version differs")
    _require(value["device_type"] == "cuda", "production training device must be cuda")
    _require(
        value["twin_environment_equal_fields"] == list(_TWIN_ENVIRONMENT_FIELDS),
        "twin environment field declaration differs",
    )
    _require(
        value["amp"] is False and value["tf32"] is False and value["torch_compile"] is False,
        "prohibited training acceleration is enabled",
    )
    initialization_seed = namespaced_seed(seed, "initialization", "model")
    expected_runtime = {
        "cublas_workspace_config": contract.determinism.cublas_workspace_config,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "default_dtype": "float32",
        "deterministic_algorithms": True,
        "flash_sdpa": False,
        "math_sdpa": True,
        "memory_efficient_sdpa": False,
        "mha_fastpath": False,
        "numpy_legacy_seed": initialization_seed % 2**32,
        "seed": initialization_seed,
        "matmul_tf32": False,
    }
    _require(value["runtime"] == expected_runtime, "training deterministic runtime differs")
    return tuple((name, value[name]) for name in _TWIN_ENVIRONMENT_FIELDS)


def _verify_training_rng(
    document: dict[str, object],
    *,
    contract: NativeDiffusionContract,
    seed: int,
) -> None:
    value = _exact_keys(
        document,
        {
            "schema_version",
            "derivation",
            "root_seed",
            "initialization_seed_uint64",
            "namespaces",
            "minibatch_key",
            "timestep_key",
            "corruption_key",
            "dropout_key",
        },
        label="rng.json",
    )
    expected = {
        "schema_version": 1,
        "derivation": contract.determinism.rng_derivation,
        "root_seed": seed,
        "initialization_seed_uint64": namespaced_seed(seed, "initialization", "model"),
        "namespaces": list(contract.determinism.rng_namespaces),
        "minibatch_key": ["global_draw_ordinal"],
        "timestep_key": ["global_draw_ordinal", "rejection_counter"],
        "corruption_key": ["global_draw_ordinal", "sequence_id", "level"],
        "dropout_key": ["optimizer_step"],
    }
    _require(value == expected, "rng.json differs from the frozen derivation")


def _verify_training_metrics(
    document: dict[str, object],
    *,
    contract: NativeDiffusionContract,
) -> None:
    value = _exact_keys(document, _TRAIN_METRIC_FIELDS, label="train_metrics.json")
    _require(value["schema_version"] == 1, "train metric schema_version differs")
    _require(value["steps"] == contract.training.max_steps, "train metric steps differ")
    _require(
        value["drawn_sequences"] == contract.training.max_steps * contract.training.batch_sequences,
        "drawn sequence count differs",
    )
    _integer(value["total_selected_tokens"], label="total_selected_tokens", minimum=1)
    for name in (
        "final_loss",
        "mean_loss",
        "final_mean_row_accuracy",
        "mean_row_accuracy",
        "mean_gradient_norm_before_clipping",
        "final_learning_rate",
    ):
        number = _finite(value[name], label=name)
        _require(number >= 0.0, f"{name} cannot be negative")
    _require(
        0.0 <= cast(float, value["final_mean_row_accuracy"]) <= 1.0, "final accuracy is invalid"
    )
    _require(0.0 <= cast(float, value["mean_row_accuracy"]) <= 1.0, "mean accuracy is invalid")
    _require(
        value["final_learning_rate"] == _learning_rate(contract, contract.training.max_steps),
        "final learning rate differs",
    )
    _require(value["loss_reduction"] == contract.diffusion.loss_reduction, "loss reduction differs")
    _require(
        value["sampling_weight_application"] == contract.training.sampling_weight_application,
        "sampling weight application differs",
    )
    _require(value["validation_consulted"] is False, "training consulted validation")
    peak = _integer(value["peak_gpu_memory_bytes"], label="peak_gpu_memory_bytes")
    _require(
        peak <= int(contract.compute.maximum_peak_gpu_memory_gib * 1024**3),
        "training exceeded GPU-memory cap",
    )


def _verify_training_trace(
    rows: tuple[dict[str, object], ...],
    *,
    contract: NativeDiffusionContract,
    seed: int,
    metrics: Mapping[str, object],
) -> None:
    expected_steps = tuple(
        step
        for step in range(1, contract.training.max_steps + 1)
        if step == 1
        or step % contract.training.training_log_interval_steps == 0
        or step == contract.training.max_steps
    )
    _require(
        tuple(row.get("step") for row in rows) == expected_steps,
        "training trace step ledger differs",
    )
    for row, step in zip(rows, expected_steps, strict=True):
        value = _exact_keys(row, _TRACE_FIELDS, label=f"training trace step {step}")
        _require(value["schema_version"] == 1, "training trace schema_version differs")
        for name in ("loss", "mean_row_accuracy", "gradient_norm_before_clipping", "learning_rate"):
            number = _finite(value[name], label=f"trace {name}")
            _require(number >= 0.0, f"trace {name} cannot be negative")
        _require(0.0 <= cast(float, value["mean_row_accuracy"]) <= 1.0, "trace accuracy is invalid")
        _require(
            value["learning_rate"] == _learning_rate(contract, step), "trace learning rate differs"
        )
        _integer(value["selected_tokens"], label="trace selected_tokens", minimum=1)
        _require(
            value["dropout_seed"] == namespaced_seed(seed, "dropout", step),
            "trace dropout seed differs",
        )
        _require_sha256(value["batch_sha256"], label="trace batch_sha256")
    final = rows[-1]
    _require(final["loss"] == metrics["final_loss"], "final trace loss differs from metrics")
    _require(
        final["mean_row_accuracy"] == metrics["final_mean_row_accuracy"],
        "final trace accuracy differs from metrics",
    )


def _verify_training_bundle(
    path: str | Path,
    *,
    label: str,
    expected_seed: int,
    contract: NativeDiffusionContract,
    contract_payload: bytes,
    repository_root: str | Path,
) -> TrainingBundleAudit:
    _, snapshots = _read_bundle(
        path,
        expected_files=contract.artifacts.training_bundle_files,
        label=f"training bundle {label}",
    )
    _require(
        snapshots["contract.toml"].payload == contract_payload,
        f"training bundle {label} contract differs",
    )
    manifest = _strict_json(
        snapshots["manifest.json"].payload, label=f"training bundle {label} manifest"
    )
    _exact_keys(
        manifest,
        contract.artifacts.training_manifest_fields,
        label=f"training bundle {label} manifest",
    )
    _require(
        manifest["schema_version"] == contract.artifacts.schema_version,
        "training manifest schema_version differs",
    )
    _require(manifest["artifact"] == contract.artifact, "training artifact name differs")
    _require(manifest["config_sha256"] == CONFIG_SHA256, "training config SHA differs")
    git_commit = _require_git(manifest["git_commit"])
    _require(manifest["seed"] == expected_seed, f"training bundle {label} seed differs")

    artifact_hashes = _exact_keys(
        manifest["artifacts"],
        set(contract.artifacts.training_bundle_files) - {"manifest.json"},
        label="training artifact hashes",
    )
    for name, snapshot in snapshots.items():
        if name != "manifest.json":
            _require(
                artifact_hashes[name] == snapshot.sha256,
                f"training artifact hash differs for {label}/{name}",
            )

    input_entries = _parse_sha_manifest(
        snapshots["INPUT_SHA256SUMS"].payload, label="INPUT_SHA256SUMS"
    )
    _require(
        input_entries
        == {
            "contract.toml": contract.config_sha256,
            "training_projection.jsonl": contract.input.training_projection_sha256,
        },
        "training input checksum manifest differs",
    )
    code_entries = _parse_sha_manifest(
        snapshots["CODE_SHA256SUMS"].payload, label="CODE_SHA256SUMS"
    )
    _verify_code_manifest(code_entries, git_commit=git_commit, repository_root=repository_root)

    corpus = _exact_keys(
        manifest["corpus"],
        {
            "accepted_parent_sha256",
            "training_projection_sha256",
            "trainer_visible_sequences",
            "trainer_visible_fields",
            "roles",
        },
        label="training manifest corpus",
    )
    _require(
        corpus
        == {
            "accepted_parent_sha256": contract.input.corpus_sha256,
            "training_projection_sha256": contract.input.training_projection_sha256,
            "trainer_visible_sequences": contract.input.expected_train_sequences,
            "trainer_visible_fields": list(contract.leakage.trainer_allowed_fields),
            "roles": list(contract.leakage.trainer_allowed_roles),
        },
        "training corpus boundary differs",
    )

    model_config = _model_config(contract)
    model = NativeDenoiser(model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(
        parameter_count == contract.model.expected_trainable_parameters,
        "independent model parameter count differs",
    )
    model_manifest = _exact_keys(
        manifest["model"],
        {
            "config",
            "trainable_parameters",
            "checkpoint_file_sha256",
            "checkpoint_logical_state_sha256",
            "checkpoint_format",
        },
        label="training model manifest",
    )
    _require(model_manifest["config"] == asdict(model_config), "training model config differs")
    _require(
        model_manifest["trainable_parameters"] == parameter_count,
        "training parameter count differs",
    )
    checkpoint_file = _require_sha256(
        model_manifest["checkpoint_file_sha256"], label="checkpoint file SHA"
    )
    checkpoint_logical = _require_sha256(
        model_manifest["checkpoint_logical_state_sha256"], label="checkpoint logical SHA"
    )
    _require(
        model_manifest["checkpoint_format"] == contract.environment.checkpoint_format,
        "checkpoint format differs",
    )
    _require(
        checkpoint_file == snapshots["model_final.safetensors"].sha256,
        "checkpoint physical hash differs",
    )
    observed_checkpoint = load_safetensors_checkpoint(
        model,
        snapshots["model_final.safetensors"].path,
        expected_file_sha256=checkpoint_file,
        expected_logical_state_sha256=checkpoint_logical,
    )
    _require(
        observed_checkpoint.logical_state_sha256 == checkpoint_logical,
        "checkpoint logical hash differs",
    )

    environment = _verify_training_environment(
        _strict_json(snapshots["environment.json"].payload, label="environment.json"),
        contract=contract,
        seed=expected_seed,
    )
    rng = _strict_json(snapshots["rng.json"].payload, label="rng.json")
    _verify_training_rng(rng, contract=contract, seed=expected_seed)
    rng_manifest = _exact_keys(
        manifest["rng"], {"filename", "sha256"}, label="training rng manifest"
    )
    _require(
        rng_manifest == {"filename": "rng.json", "sha256": snapshots["rng.json"].sha256},
        "training RNG binding differs",
    )

    schedule = _schedule_sha256(contract)
    _require(
        snapshots["training_schedule.sha256"].payload == f"{schedule}\n".encode("ascii"),
        "training schedule digest differs",
    )
    metrics = _strict_json(snapshots["train_metrics.json"].payload, label="train_metrics.json")
    _verify_training_metrics(metrics, contract=contract)
    trace = _strict_jsonl(snapshots["training_trace.jsonl"].payload, label="training_trace.jsonl")
    _verify_training_trace(trace, contract=contract, seed=expected_seed, metrics=metrics)
    decay_names, no_decay_names = _parameter_groups(model)
    training = _exact_keys(
        manifest["training"],
        {
            "steps",
            "batch_sequences",
            "optimizer",
            "parameter_decay_names",
            "parameter_no_decay_names",
            "schedule_sha256",
            "checkpoint_selection",
            "validation_during_training",
            "early_stopping",
            "resume_supported",
            "metrics_sha256",
        },
        label="training manifest training",
    )
    expected_training = {
        "steps": contract.training.max_steps,
        "batch_sequences": contract.training.batch_sequences,
        "optimizer": contract.training.optimizer,
        "parameter_decay_names": list(decay_names),
        "parameter_no_decay_names": list(no_decay_names),
        "schedule_sha256": schedule,
        "checkpoint_selection": contract.training.checkpoint_selection,
        "validation_during_training": False,
        "early_stopping": False,
        "resume_supported": False,
        "metrics_sha256": snapshots["train_metrics.json"].sha256,
    }
    _require(training == expected_training, "training manifest protocol differs")

    for snapshot in snapshots.values():
        _snapshot_unchanged(snapshot, label=f"training bundle {label}/{snapshot.path.name}")
    return TrainingBundleAudit(
        label=label,
        seed=expected_seed,
        git_commit=git_commit,
        manifest_sha256=snapshots["manifest.json"].sha256,
        checkpoint_file_sha256=checkpoint_file,
        checkpoint_logical_sha256=checkpoint_logical,
        environment_identity=environment,
        file_hashes=tuple(sorted((name, snapshot.sha256) for name, snapshot in snapshots.items())),
        snapshots=tuple(snapshots[name] for name in sorted(snapshots)),
    )


@dataclass(frozen=True, slots=True)
class ArraySpec:
    name: str
    dtype: np.dtype[Any]
    dimensions: tuple[int | str, ...]


def _parse_array_specs(values: Sequence[str], *, label: str) -> tuple[ArraySpec, ...]:
    aliases = {"S64": "|S64", "u1": "|u1", "b1": "|b1"}
    specs: list[ArraySpec] = []
    _require(type(values) is tuple and bool(values), f"{label} schema must be a tuple")
    for encoded in values:
        _require(type(encoded) is str, f"{label} schema row must be a string")
        parts = cast(str, encoded).split("|")
        _require(len(parts) == 3, f"{label} schema row is malformed")
        name, dtype_name, shape_text = parts
        _require(
            re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"{label} member name is invalid"
        )
        _require(
            dtype_name in {"S64", "u1", "b1", "<u2", "<u8", "<f8"}, f"{label} dtype is invalid"
        )
        dimensions: list[int | str] = []
        for raw_dimension in shape_text.split(","):
            if raw_dimension == "N_selected":
                dimensions.append(raw_dimension)
            else:
                _require(
                    raw_dimension.isascii()
                    and raw_dimension.isdigit()
                    and not raw_dimension.startswith("0"),
                    f"{label} dimension is invalid",
                )
                dimensions.append(int(raw_dimension))
        specs.append(
            ArraySpec(name, np.dtype(aliases.get(dtype_name, dtype_name)), tuple(dimensions))
        )
    _require(len({spec.name for spec in specs}) == len(specs), f"{label} names are duplicated")
    return tuple(specs)


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _canonical_npz_bytes(
    specs: Sequence[ArraySpec],
    arrays: Mapping[str, NDArray[np.generic]],
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for spec in specs:
            info = zipfile.ZipInfo(f"{spec.name}.npy", date_time=_DOS_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.internal_attr = 0
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                _npy_bytes(np.ascontiguousarray(arrays[spec.name])),
                compress_type=zipfile.ZIP_STORED,
            )
    return output.getvalue()


def _read_npy(payload: bytes, *, spec: ArraySpec, label: str) -> NDArray[np.generic]:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        _require(version == (1, 0), f"{label} must use NPY format 1.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    except (EOFError, ValueError) as error:
        raise VerificationError(f"{label} has an invalid NPY header") from error
    _require(not fortran_order, f"{label} must be C-contiguous")
    _require(dtype == spec.dtype and not dtype.hasobject, f"{label} dtype differs from contract")
    expected_bytes = math.prod(shape) * dtype.itemsize
    raw = stream.read()
    _require(len(raw) == expected_bytes, f"{label} payload size differs from its header")
    values = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
    _require(values.flags.c_contiguous, f"{label} is not contiguous")
    if dtype.kind == "f":
        _require(bool(np.all(np.isfinite(values))), f"{label} contains non-finite values")
    values.flags.writeable = False
    return values


def _read_npz(
    snapshot: Snapshot,
    *,
    specs: Sequence[ArraySpec],
    label: str,
) -> dict[str, NDArray[np.generic]]:
    expected_names = tuple(f"{spec.name}.npy" for spec in specs)
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.payload), mode="r") as archive:
            _require(archive.comment == b"", f"{label} ZIP comment is forbidden")
            infos = archive.infolist()
            _require(
                tuple(info.filename for info in infos) == expected_names,
                f"{label} member order differs from contract",
            )
            arrays: dict[str, NDArray[np.generic]] = {}
            symbolic: dict[str, int] = {}
            for spec, info in zip(specs, infos, strict=True):
                _require(info.date_time == _DOS_EPOCH, f"{label}/{info.filename} timestamp differs")
                _require(
                    info.compress_type == zipfile.ZIP_STORED,
                    f"{label}/{info.filename} is compressed",
                )
                _require(info.create_system == 3, f"{label}/{info.filename} creator differs")
                _require(
                    info.external_attr == (stat.S_IFREG | 0o444) << 16,
                    f"{label}/{info.filename} mode differs",
                )
                _require(
                    info.internal_attr == 0 and info.flag_bits == 0,
                    f"{label}/{info.filename} flags differ",
                )
                _require(
                    info.extra == b"" and info.comment == b"",
                    f"{label}/{info.filename} metadata extras are forbidden",
                )
                _require(
                    info.file_size == info.compress_size,
                    f"{label}/{info.filename} stored size differs",
                )
                values = _read_npy(archive.read(info), spec=spec, label=f"{label}/{info.filename}")
                _require(
                    values.ndim == len(spec.dimensions), f"{label}/{info.filename} rank differs"
                )
                for observed, expected in zip(values.shape, spec.dimensions, strict=True):
                    if type(expected) is int:
                        _require(observed == expected, f"{label}/{info.filename} shape differs")
                    else:
                        prior = symbolic.setdefault(expected, observed)
                        _require(
                            observed == prior, f"{label} symbolic dimension {expected} differs"
                        )
                arrays[spec.name] = values
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise VerificationError(f"{label} is not a valid deterministic NPZ") from error
    _require(
        _canonical_npz_bytes(specs, arrays) == snapshot.payload,
        f"{label} bytes are not canonical deterministic NPZ",
    )
    return arrays


def _decode_fixed_ascii(values: NDArray[np.generic], *, label: str) -> tuple[str, ...]:
    _require(values.ndim == 1 and values.dtype == np.dtype("|S64"), f"{label} must be S64 vector")
    result: list[str] = []
    for raw in values:
        payload = bytes(raw)
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as error:
            raise VerificationError(f"{label} contains non-ASCII text") from error
        _require("\x00" not in text, f"{label} contains an embedded NUL")
        result.append(text)
    return tuple(result)


def _case_id(sequence_id: str, level: int, replicate: int, row_seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(_CASE_DOMAIN)
    for value in (
        CONFIG_SHA256.encode("ascii"),
        sequence_id.encode("ascii"),
        level.to_bytes(2, "big"),
        replicate.to_bytes(2, "big"),
        row_seed.to_bytes(8, "big"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class InputAudit:
    contract: NativeDiffusionContract
    contract_snapshot: Snapshot
    corpus_snapshot: Snapshot
    projection_snapshot: Snapshot
    reference_snapshot: Snapshot
    corpus_rows: tuple[DiffusionCorpusRow, ...]
    training: TrainingDistribution
    reference_sequences: tuple[str, ...]


def _read_reference_fasta(
    snapshot: Snapshot,
    *,
    contract: NativeDiffusionContract,
) -> tuple[str, ...]:
    _require(
        snapshot.sha256 == contract.input.organizer_reference_sha256,
        "organizer reference SHA-256 differs",
    )
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("organizer reference FASTA is not UTF-8") from error
    sequences: list[str] = []
    active = False
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if active:
                sequences.append("".join(parts))
            active = True
            parts = []
        elif active:
            parts.append(line.upper())
    if active:
        sequences.append("".join(parts))
    _require(
        len(sequences) == contract.input.organizer_reference_records,
        "organizer reference FASTA record census differs",
    )
    _require(
        all(sequence and not (set(sequence) - set(_ALPHABET)) for sequence in sequences),
        "organizer reference FASTA contains a noncanonical sequence",
    )
    return tuple(sequences)


def _verify_inputs(
    *,
    contract_path: str | Path,
    corpus_path: str | Path,
    training_projection_path: str | Path,
    reference_fasta: str | Path,
) -> InputAudit:
    contract_snapshot = _read_snapshot(contract_path, label="contract")
    _require(
        contract_snapshot.sha256 == CONFIG_SHA256, "contract SHA-256 differs from imported hard pin"
    )
    contract = load_unconditional_v0_contract(contract_snapshot.path)
    _require(contract.config_sha256 == CONFIG_SHA256, "parsed contract identity differs")
    corpus_snapshot = _read_snapshot(corpus_path, label="accepted corpus")
    _require(
        corpus_snapshot.sha256 == contract.input.corpus_sha256, "accepted corpus SHA-256 differs"
    )
    corpus = load_native_diffusion_corpus(
        corpus_snapshot.path, expected_sha256=contract.input.corpus_sha256
    )
    _require(
        len(corpus.rows)
        == contract.input.expected_train_sequences + contract.input.expected_validation_sequences,
        "accepted corpus row census differs",
    )
    projection_snapshot = _read_snapshot(training_projection_path, label="training projection")
    _require(
        projection_snapshot.sha256 == contract.input.training_projection_sha256,
        "training projection SHA-256 differs",
    )
    training = load_training_projection(
        projection_snapshot.path,
        expected_sha256=contract.input.training_projection_sha256,
        expected_rows=contract.input.expected_train_sequences,
    )
    expected_projection = tuple(
        TrainingRow(row.sequence_id, row.sequence, row.sampling_weight)
        for row in corpus.rows
        if row.role == "train"
    )
    _require(
        training.rows == expected_projection, "training projection does not equal corpus train rows"
    )
    reference_snapshot = _read_snapshot(reference_fasta, label="organizer reference")
    references = _read_reference_fasta(reference_snapshot, contract=contract)
    for snapshot, label in (
        (contract_snapshot, "contract"),
        (corpus_snapshot, "accepted corpus"),
        (projection_snapshot, "training projection"),
        (reference_snapshot, "organizer reference"),
    ):
        _snapshot_unchanged(snapshot, label=label)
    return InputAudit(
        contract=contract,
        contract_snapshot=contract_snapshot,
        corpus_snapshot=corpus_snapshot,
        projection_snapshot=projection_snapshot,
        reference_snapshot=reference_snapshot,
        corpus_rows=corpus.rows,
        training=training,
        reference_sequences=references,
    )


@dataclass(frozen=True, slots=True)
class CorruptionAudit:
    arrays: Mapping[str, NDArray[np.generic]]
    validation_rows: tuple[DiffusionCorpusRow, ...]
    case_sequence_ids: tuple[str, ...]
    case_union_ids: tuple[str, ...]


def _verify_corruptions(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    inputs: InputAudit,
) -> CorruptionAudit:
    contract = inputs.contract
    validation_rows = tuple(
        sorted(
            (row for row in inputs.corpus_rows if row.role == "validation"),
            key=lambda row: row.sequence_id,
        )
    )
    _require(
        len(validation_rows) == contract.evaluation.expected_validation_sequences,
        "validation row census differs",
    )
    sequence_ids = _decode_fixed_ascii(arrays["sequence_id"], label="validation sequence_id")
    homology_ids = _decode_fixed_ascii(
        arrays["homology_component_id"], label="validation homology_component_id"
    )
    union_ids = _decode_fixed_ascii(
        arrays["union_component_id"], label="validation union_component_id"
    )
    _require(
        sequence_ids == tuple(row.sequence_id for row in validation_rows),
        "validation sequence ID order differs",
    )
    _require(
        homology_ids == tuple(row.homology_component_id for row in validation_rows),
        "validation homology IDs differ",
    )
    _require(
        union_ids == tuple(row.union_component_id for row in validation_rows),
        "validation union IDs differ",
    )
    lengths = cast(NDArray[np.uint8], arrays["length"])
    weights = cast(FloatArray, arrays["sampling_weight"])
    _require(
        np.array_equal(
            lengths, np.asarray([len(row.sequence) for row in validation_rows], dtype=np.uint8)
        ),
        "validation lengths differ",
    )
    _require(
        np.array_equal(
            weights, np.asarray([row.sampling_weight for row in validation_rows], dtype="<f8")
        ),
        "validation weights differ",
    )

    vocabulary = PeptideVocabulary(_ALPHABET)
    encoded = vocabulary.encode(
        [row.sequence for row in validation_rows], max_length=contract.model.max_length
    )
    expected_clean = encoded.tokens.astype(np.uint8)
    expected_attention = encoded.attention_mask
    _require(
        np.array_equal(arrays["clean_tokens"], expected_clean),
        "validation clean token matrix differs",
    )
    _require(
        np.array_equal(arrays["attention_mask"], expected_attention),
        "validation attention mask differs",
    )

    case_ids = _decode_fixed_ascii(arrays["case_id"], label="validation case_id")
    row_indices = cast(NDArray[np.uint16], arrays["row_index"])
    levels = cast(NDArray[np.uint8], arrays["level"])
    replicates = cast(NDArray[np.uint8], arrays["replicate"])
    seeds = cast(NDArray[np.uint64], arrays["row_seed"])
    mask_counts = cast(NDArray[np.uint8], arrays["mask_count"])
    corrupted = cast(NDArray[np.uint8], arrays["corrupted_tokens"])
    selected = cast(NDArray[np.bool_], arrays["selected_mask"])
    expected_case_ids: list[str] = []
    expected_case_sequences: list[str] = []
    expected_case_unions: list[str] = []
    expected_row_indices: list[int] = []
    expected_levels: list[int] = []
    expected_replicates: list[int] = []
    expected_seeds: list[int] = []
    expected_counts: list[int] = []
    schedule = CosineMaskSchedule(contract.diffusion.cosine_offset)
    for row_index, row in enumerate(validation_rows):
        for level in range(1, contract.evaluation.levels + 1):
            count = int(
                schedule.mask_counts(
                    np.asarray([len(row.sequence)], dtype=np.int64),
                    level,
                    total_levels=contract.evaluation.levels,
                )[0]
            )
            for replicate in range(contract.evaluation.replicates_per_sequence_level):
                row_seed = namespaced_seed(
                    contract.evaluation.evaluation_seed,
                    "validation",
                    CONFIG_SHA256,
                    row.sequence_id,
                    level,
                    replicate,
                )
                expected_case_ids.append(_case_id(row.sequence_id, level, replicate, row_seed))
                expected_case_sequences.append(row.sequence_id)
                expected_case_unions.append(row.union_component_id)
                expected_row_indices.append(row_index)
                expected_levels.append(level)
                expected_replicates.append(replicate)
                expected_seeds.append(row_seed)
                expected_counts.append(count)
    _require(case_ids == tuple(expected_case_ids), "validation case ID ledger differs")
    _require(
        np.array_equal(row_indices, np.asarray(expected_row_indices, dtype="<u2")),
        "validation row-index ledger differs",
    )
    _require(
        np.array_equal(levels, np.asarray(expected_levels, dtype=np.uint8)),
        "validation level ledger differs",
    )
    _require(
        np.array_equal(replicates, np.asarray(expected_replicates, dtype=np.uint8)),
        "validation replicate ledger differs",
    )
    _require(
        np.array_equal(seeds, np.asarray(expected_seeds, dtype="<u8")),
        "validation corruption seeds differ",
    )
    _require(
        np.array_equal(mask_counts, np.asarray(expected_counts, dtype=np.uint8)),
        "validation mask counts differ",
    )

    diffusion = AbsorbingDiffusion(vocabulary, schedule)
    expected_corrupted = np.empty_like(corrupted)
    expected_selected = np.empty_like(selected)
    batch_size = contract.evaluation.batch_sequences
    for start in range(0, len(case_ids), batch_size):
        stop = min(len(case_ids), start + batch_size)
        case_rows = row_indices[start:stop].astype(np.int64)
        clean = encoded.tokens[case_rows]
        attention = encoded.attention_mask[case_rows]
        batch_corrupted, batch_selected = diffusion.corrupt_fixed_count(
            clean,
            attention,
            levels[start:stop].astype(np.int64),
            total_levels=contract.evaluation.levels,
            row_seeds=[int(value) for value in seeds[start:stop]],
        )
        expected_corrupted[start:stop] = batch_corrupted.astype(np.uint8)
        expected_selected[start:stop] = batch_selected
    _require(np.array_equal(corrupted, expected_corrupted), "validation corrupted tokens differ")
    _require(np.array_equal(selected, expected_selected), "validation selected masks differ")
    return CorruptionAudit(
        arrays=arrays,
        validation_rows=validation_rows,
        case_sequence_ids=tuple(expected_case_sequences),
        case_union_ids=tuple(expected_case_unions),
    )


@dataclass(frozen=True, slots=True)
class TokenStatistics:
    arrays: Mapping[str, NDArray[np.generic]]
    methods: tuple[str, ...]
    case_offsets: NDArray[np.uint64]


def _verify_token_statistics(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    corruptions: CorruptionAudit,
    contract: NativeDiffusionContract,
) -> TokenStatistics:
    case_ids = _decode_fixed_ascii(arrays["case_id"], label="token-stat case_id")
    expected_case_ids = _decode_fixed_ascii(
        corruptions.arrays["case_id"], label="corruption case_id"
    )
    _require(case_ids == expected_case_ids, "token-stat case IDs differ from corruptions")
    methods = _decode_fixed_ascii(arrays["method"], label="token-stat method")
    _require(
        methods == contract.artifacts.validation_token_stats_method_order,
        "token-stat method order differs",
    )
    offsets = cast(NDArray[np.uint64], arrays["case_offsets"])
    _require(
        int(offsets[0]) == 0 and bool(np.all(offsets[1:] > offsets[:-1])),
        "case offsets must increase from zero",
    )
    expected_counts = cast(NDArray[np.uint8], corruptions.arrays["mask_count"]).astype(np.uint64)
    _require(
        np.array_equal(np.diff(offsets), expected_counts), "case offsets differ from mask counts"
    )
    selected_count = int(offsets[-1])
    for name in ("position", "target_token"):
        _require(arrays[name].shape == (selected_count,), f"token-stat {name} length differs")
    for name in (
        "target_log_probability",
        "top1_confidence",
        "top1_correct",
        "top3_correct",
        "multiclass_brier",
    ):
        _require(
            arrays[name].shape == (len(methods), selected_count), f"token-stat {name} shape differs"
        )
    target = cast(NDArray[np.uint8], arrays["target_token"])
    confidence = cast(FloatArray, arrays["top1_confidence"])
    log_probability = cast(FloatArray, arrays["target_log_probability"])
    top1 = cast(NDArray[np.bool_], arrays["top1_correct"])
    top3 = cast(NDArray[np.bool_], arrays["top3_correct"])
    brier = cast(FloatArray, arrays["multiclass_brier"])
    _require(bool(np.all(target < 20)), "token-stat targets contain special tokens")
    _require(bool(np.all(log_probability <= 1e-15)), "target log probability exceeds zero")
    _require(
        bool(np.all((confidence >= 0.0) & (confidence <= 1.0))), "top1 confidence is out of range"
    )
    _require(not bool(np.any(top1 & ~top3)), "top3 correctness contradicts top1")
    _require(bool(np.all((brier >= 0.0) & (brier <= 2.0 + 1e-12))), "Brier score is out of range")
    target_probability = np.exp(log_probability)
    _require(
        bool(np.all(target_probability <= confidence + 1e-12)),
        "target probability exceeds top1 confidence",
    )
    _require(
        bool(np.all(~top1 | np.isclose(target_probability, confidence, rtol=0.0, atol=1e-12))),
        "correct top1 target probability differs from confidence",
    )

    positions = cast(NDArray[np.uint8], arrays["position"])
    clean = cast(NDArray[np.uint8], corruptions.arrays["clean_tokens"])
    selected = cast(NDArray[np.bool_], corruptions.arrays["selected_mask"])
    row_indices = cast(NDArray[np.uint16], corruptions.arrays["row_index"])
    cursor = 0
    for case_index in range(len(case_ids)):
        expected_positions = np.flatnonzero(selected[case_index]).astype(np.uint8)
        stop = cursor + len(expected_positions)
        _require(
            np.array_equal(positions[cursor:stop], expected_positions),
            "token positions differ from selected masks",
        )
        row = int(row_indices[case_index])
        _require(
            np.array_equal(target[cursor:stop], clean[row, expected_positions]),
            "token targets differ from clean tokens",
        )
        cursor = stop
    _require(cursor == selected_count, "token-stat cursor did not consume all selected tokens")
    return TokenStatistics(arrays=arrays, methods=methods, case_offsets=offsets)


@dataclass(frozen=True, slots=True)
class BaselineTables:
    unigram: FloatArray
    forward: FloatArray
    reverse: FloatArray
    relative: FloatArray
    total_effective_residue_evidence: float
    total_effective_forward_transition_evidence: float
    total_effective_reverse_transition_evidence: float


def _fit_baseline_tables(
    training: TrainingDistribution,
    *,
    contract: NativeDiffusionContract,
) -> BaselineTables:
    scale = contract.input.expected_train_sequences
    unigram_counts = np.zeros(20, dtype=np.float64)
    forward_counts = np.zeros((20, 20), dtype=np.float64)
    reverse_counts = np.zeros((20, 20), dtype=np.float64)
    relative_counts = np.zeros(
        (
            len(contract.baselines.relative_position_length_bins) - 1,
            contract.baselines.relative_position_bins,
            20,
        ),
        dtype=np.float64,
    )
    vocabulary = PeptideVocabulary(_ALPHABET)
    edges = contract.baselines.relative_position_length_bins
    for row in training.rows:
        encoded = vocabulary.encode([row.sequence]).tokens[0, : len(row.sequence)]
        residue_mass = scale * row.sampling_weight / len(row.sequence)
        length_bin = int(np.searchsorted(edges, len(row.sequence), side="right") - 1)
        for position, residue_raw in enumerate(encoded):
            residue = int(residue_raw)
            unigram_counts[residue] += residue_mass
            position_bin = min(
                contract.baselines.relative_position_bins - 1,
                contract.baselines.relative_position_bins * position // len(row.sequence),
            )
            relative_counts[length_bin, position_bin, residue] += residue_mass
        transition_mass = scale * row.sampling_weight / (len(row.sequence) - 1)
        for left_raw, right_raw in pairwise(encoded):
            left, right = int(left_raw), int(right_raw)
            forward_counts[left, right] += transition_mass
            reverse_counts[right, left] += transition_mass
    residue_evidence = float(np.sum(unigram_counts))
    forward_evidence = float(np.sum(forward_counts))
    reverse_evidence = float(np.sum(reverse_counts))
    _require(
        math.isclose(residue_evidence, float(scale), rel_tol=0.0, abs_tol=1e-10),
        "unigram evidence total differs",
    )
    _require(
        math.isclose(forward_evidence, float(scale), rel_tol=0.0, abs_tol=1e-10),
        "forward evidence total differs",
    )
    _require(
        math.isclose(reverse_evidence, float(scale), rel_tol=0.0, abs_tol=1e-10),
        "reverse evidence total differs",
    )
    pseudocount = contract.baselines.count_pseudocount
    unigram = unigram_counts + pseudocount
    unigram /= np.sum(unigram)
    forward = forward_counts + pseudocount
    forward /= np.sum(forward, axis=1, keepdims=True)
    reverse = reverse_counts + pseudocount
    reverse /= np.sum(reverse, axis=1, keepdims=True)
    relative = (
        relative_counts + contract.baselines.relative_position_prior_mass * unigram[None, None, :]
    )
    relative /= np.sum(relative, axis=2, keepdims=True)
    for value in (unigram, forward, reverse, relative):
        value.flags.writeable = False
    return BaselineTables(
        unigram=unigram,
        forward=forward,
        reverse=reverse,
        relative=relative,
        total_effective_residue_evidence=residue_evidence,
        total_effective_forward_transition_evidence=forward_evidence,
        total_effective_reverse_transition_evidence=reverse_evidence,
    )


def _baseline_distribution(
    method: str,
    *,
    tables: BaselineTables,
    corrupted_row: NDArray[np.uint8],
    position: int,
    length: int,
    contract: NativeDiffusionContract,
) -> FloatArray:
    if method == "component_weighted_unigram":
        return tables.unigram
    if method == "length_relative_position_frequency":
        edges = contract.baselines.relative_position_length_bins
        length_bin = int(np.searchsorted(edges, length, side="right") - 1)
        position_bin = min(
            contract.baselines.relative_position_bins - 1,
            contract.baselines.relative_position_bins * position // length,
        )
        return tables.relative[length_bin, position_bin]
    _require(method == "component_weighted_bidirectional_markov", "unknown baseline method")
    score = np.log(tables.unigram)
    if position > 0 and int(corrupted_row[position - 1]) < 20:
        left = int(corrupted_row[position - 1])
        score = score + np.log(tables.forward[left]) - np.log(tables.unigram)
    if position + 1 < length and int(corrupted_row[position + 1]) < 20:
        right = int(corrupted_row[position + 1])
        score = score + np.log(tables.reverse[right]) - np.log(tables.unigram)
    score -= np.max(score)
    probability = np.exp(score)
    probability /= math.fsum(probability.tolist())
    return probability


def _probability_statistics(
    probability: FloatArray, target: int
) -> tuple[float, float, bool, bool, float]:
    _require(
        probability.shape == (20,) and bool(np.all(np.isfinite(probability))),
        "baseline probability vector is invalid",
    )
    ranking = np.argsort(-probability, kind="stable")
    prediction = int(ranking[0])
    target_probability = float(probability[target])
    one_hot = np.zeros(20, dtype=np.float64)
    one_hot[target] = 1.0
    return (
        math.log(target_probability),
        float(probability[prediction]),
        prediction == target,
        bool(target in ranking[:3]),
        float(np.sum(np.square(probability - one_hot))),
    )


def _verify_baseline_token_statistics(
    statistics: TokenStatistics,
    *,
    corruptions: CorruptionAudit,
    inputs: InputAudit,
) -> None:
    contract = inputs.contract
    tables = _fit_baseline_tables(inputs.training, contract=contract)
    row_indices = cast(NDArray[np.uint16], corruptions.arrays["row_index"])
    corrupted = cast(NDArray[np.uint8], corruptions.arrays["corrupted_tokens"])
    lengths = cast(NDArray[np.uint8], corruptions.arrays["length"])
    targets = cast(NDArray[np.uint8], statistics.arrays["target_token"])
    positions = cast(NDArray[np.uint8], statistics.arrays["position"])
    expected_log = np.empty((3, len(targets)), dtype=np.float64)
    expected_confidence = np.empty_like(expected_log)
    expected_top1 = np.empty((3, len(targets)), dtype=np.bool_)
    expected_top3 = np.empty_like(expected_top1)
    expected_brier = np.empty_like(expected_log)
    for case_index in range(len(row_indices)):
        start = int(statistics.case_offsets[case_index])
        stop = int(statistics.case_offsets[case_index + 1])
        row = int(row_indices[case_index])
        length = int(lengths[row])
        for token_index in range(start, stop):
            position = int(positions[token_index])
            target = int(targets[token_index])
            for method_index, method in enumerate(contract.baselines.names):
                probability = _baseline_distribution(
                    method,
                    tables=tables,
                    corrupted_row=corrupted[case_index],
                    position=position,
                    length=length,
                    contract=contract,
                )
                logp, confidence, top1, top3, brier = _probability_statistics(probability, target)
                expected_log[method_index, token_index] = logp
                expected_confidence[method_index, token_index] = confidence
                expected_top1[method_index, token_index] = top1
                expected_top3[method_index, token_index] = top3
                expected_brier[method_index, token_index] = brier
    comparisons = (
        ("target_log_probability", expected_log, 1e-14),
        ("top1_confidence", expected_confidence, 1e-14),
        ("multiclass_brier", expected_brier, 1e-14),
    )
    for name, expected, tolerance in comparisons:
        observed = cast(FloatArray, statistics.arrays[name])[:3]
        _require(
            bool(np.allclose(observed, expected, rtol=0.0, atol=tolerance)),
            f"stored baseline {name} differs from independent fit",
        )
    _require(
        np.array_equal(statistics.arrays["top1_correct"][:3], expected_top1),
        "stored baseline top1 flags differ",
    )
    _require(
        np.array_equal(statistics.arrays["top3_correct"][:3], expected_top3),
        "stored baseline top3 flags differ",
    )


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    method: str
    primary_nll: float
    perplexity: float
    top1_accuracy: float
    top3_accuracy: float
    mean_brier: float
    ece: float
    row_nll: tuple[float, ...]
    row_ece: tuple[float, ...]
    timestep_bins: Mapping[str, float]
    summary: Mapping[str, object]

    def summary_record(self) -> dict[str, object]:
        return dict(self.summary)


def _ece_from_bins(
    counts: Sequence[float],
    confidence: Sequence[float],
    correct: Sequence[float],
) -> float:
    total = math.fsum(counts)
    _require(total > 0.0, "ECE calibration mass is empty")
    result = 0.0
    for count, confidence_sum, correct_sum in zip(counts, confidence, correct, strict=True):
        if count > 0.0:
            result += count / total * abs(correct_sum / count - confidence_sum / count)
    return result


def _level_ranges(names: Sequence[str]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for name in names:
        if name == "64_fully_masked":
            result[name] = (64, 64)
            continue
        match = re.fullmatch(r"([0-9]+)_to_([0-9]+)", name)
        _require(match is not None, f"invalid timestep-bin name {name}")
        lower, upper = (int(value) for value in cast(re.Match[str], match).groups())
        _require(1 <= lower <= upper <= 64, f"invalid timestep-bin bounds {name}")
        result[name] = (lower, upper)
    return result


def _reconstruct_method_metrics(
    statistics: TokenStatistics,
    *,
    corruptions: CorruptionAudit,
    contract: NativeDiffusionContract,
) -> tuple[MethodMetrics, ...]:
    row_indices = cast(NDArray[np.uint16], corruptions.arrays["row_index"])
    levels = cast(NDArray[np.uint8], corruptions.arrays["level"])
    weights_by_row = cast(FloatArray, corruptions.arrays["sampling_weight"])
    case_count_by_row = (
        contract.evaluation.levels * contract.evaluation.replicates_per_sequence_level
    )
    _require(
        math.isclose(math.fsum(weights_by_row.tolist()), 1.0, rel_tol=0.0, abs_tol=1e-15),
        "validation row weights do not sum to one",
    )
    bins = _level_ranges(contract.evaluation.timestep_bins)
    case_ids = _decode_fixed_ascii(corruptions.arrays["case_id"], label="metric case_id")
    homology_ids = _decode_fixed_ascii(
        corruptions.arrays["homology_component_id"], label="metric homology IDs"
    )
    union_ids = _decode_fixed_ascii(
        corruptions.arrays["union_component_id"], label="metric union IDs"
    )
    replicates = cast(NDArray[np.uint8], corruptions.arrays["replicate"])
    mask_counts = cast(NDArray[np.uint8], corruptions.arrays["mask_count"])
    sequence_ids = _decode_fixed_ascii(
        corruptions.arrays["sequence_id"], label="metric sequence IDs"
    )
    ledger_payload = b"".join(
        _canonical_json_bytes(
            {
                "case_id": case_ids[case],
                "corruption_seed_hex": f"{int(cast(NDArray[np.uint64], corruptions.arrays['row_seed'])[case]):016x}",
                "homology_component_id": homology_ids[int(row_indices[case])],
                "level": int(levels[case]),
                "mask_count": int(mask_counts[case]),
                "replicate": int(replicates[case]),
                "sampling_weight": float(weights_by_row[int(row_indices[case])]),
                "schema_version": 1,
                "sequence_id": sequence_ids[int(row_indices[case])],
                "union_component_id": union_ids[int(row_indices[case])],
            }
        )
        for case in range(len(case_ids))
    )
    ledger_sha256 = _sha256(ledger_payload)
    output: list[MethodMetrics] = []
    logp_all = cast(FloatArray, statistics.arrays["target_log_probability"])
    confidence_all = cast(FloatArray, statistics.arrays["top1_confidence"])
    top1_all = cast(NDArray[np.bool_], statistics.arrays["top1_correct"])
    top3_all = cast(NDArray[np.bool_], statistics.arrays["top3_correct"])
    brier_all = cast(FloatArray, statistics.arrays["multiclass_brier"])
    for method_index, method in enumerate(statistics.methods):
        nll = -logp_all[method_index]
        confidence = confidence_all[method_index]
        top1 = top1_all[method_index]
        top3 = top3_all[method_index]
        brier = brier_all[method_index]
        case_records: list[dict[str, object]] = []
        case_nll: list[float] = []
        case_top1: list[float] = []
        case_top3: list[float] = []
        case_brier: list[float] = []
        case_calibration: list[tuple[list[int], list[float], list[int]]] = []
        for case in range(len(case_ids)):
            start = int(statistics.case_offsets[case])
            stop = int(statistics.case_offsets[case + 1])
            count = stop - start
            counts = [0] * _CALIBRATION_BINS
            confidence_sums = [0.0] * _CALIBRATION_BINS
            correct_sums = [0] * _CALIBRATION_BINS
            for confidence_value, correct_value in zip(
                confidence[start:stop], top1[start:stop], strict=True
            ):
                value = float(confidence_value)
                bin_index = min(_CALIBRATION_BINS - 1, int(value * _CALIBRATION_BINS))
                counts[bin_index] += 1
                confidence_sums[bin_index] += value
                correct_sums[bin_index] += int(correct_value)
            mean_nll = math.fsum(float(value) for value in nll[start:stop]) / count
            top1_mean = int(np.sum(top1[start:stop], dtype=np.int64)) / count
            top3_mean = int(np.sum(top3[start:stop], dtype=np.int64)) / count
            brier_mean = math.fsum(float(value) for value in brier[start:stop]) / count
            row = int(row_indices[case])
            case_records.append(
                {
                    "calibration_confidence_sums": confidence_sums,
                    "calibration_correct_sums": correct_sums,
                    "calibration_counts": counts,
                    "case_id": case_ids[case],
                    "homology_component_id": homology_ids[row],
                    "level": int(levels[case]),
                    "masked_tokens": count,
                    "mean_brier": brier_mean,
                    "mean_nll": mean_nll,
                    "method": method,
                    "replicate": int(replicates[case]),
                    "sampling_weight": float(weights_by_row[row]),
                    "schema_version": 1,
                    "sequence_id": sequence_ids[row],
                    "top1_accuracy": top1_mean,
                    "top3_accuracy": top3_mean,
                    "union_component_id": union_ids[row],
                }
            )
            case_nll.append(mean_nll)
            case_top1.append(top1_mean)
            case_top3.append(top3_mean)
            case_brier.append(brier_mean)
            case_calibration.append((counts, confidence_sums, correct_sums))
        row_nll: list[float] = []
        row_ece: list[float] = []
        row_records: list[dict[str, object]] = []
        global_counts = [0.0] * _CALIBRATION_BINS
        global_confidence = [0.0] * _CALIBRATION_BINS
        global_correct = [0.0] * _CALIBRATION_BINS
        for row in range(len(weights_by_row)):
            case_start = row * case_count_by_row
            case_stop = case_start + case_count_by_row
            row_counts = [0.0] * _CALIBRATION_BINS
            row_confidence = [0.0] * _CALIBRATION_BINS
            row_correct = [0.0] * _CALIBRATION_BINS
            for case in range(case_start, case_stop):
                scale = 1.0 / (case_count_by_row * int(mask_counts[case]))
                counts, confidence_sums, correct_sums = case_calibration[case]
                for bin_index in range(_CALIBRATION_BINS):
                    row_counts[bin_index] += counts[bin_index] * scale
                    row_confidence[bin_index] += confidence_sums[bin_index] * scale
                    row_correct[bin_index] += correct_sums[bin_index] * scale
            for bin_index in range(_CALIBRATION_BINS):
                global_counts[bin_index] += float(weights_by_row[row]) * row_counts[bin_index]
                global_confidence[bin_index] += (
                    float(weights_by_row[row]) * row_confidence[bin_index]
                )
                global_correct[bin_index] += float(weights_by_row[row]) * row_correct[bin_index]
            nll_mean = math.fsum(case_nll[case_start:case_stop]) / case_count_by_row
            top1_mean = math.fsum(case_top1[case_start:case_stop]) / case_count_by_row
            top3_mean = math.fsum(case_top3[case_start:case_stop]) / case_count_by_row
            brier_mean = math.fsum(case_brier[case_start:case_stop]) / case_count_by_row
            ece_value = _ece_from_bins(row_counts, row_confidence, row_correct)
            row_nll.append(nll_mean)
            row_ece.append(ece_value)
            row_records.append(
                {
                    "case_count": case_count_by_row,
                    "ece": ece_value,
                    "homology_component_id": homology_ids[row],
                    "masked_tokens": sum(
                        int(mask_counts[case]) for case in range(case_start, case_stop)
                    ),
                    "mean_brier": brier_mean,
                    "mean_nll": nll_mean,
                    "method": method,
                    "sampling_weight": float(weights_by_row[row]),
                    "schema_version": 1,
                    "sequence_id": sequence_ids[row],
                    "top1_accuracy": top1_mean,
                    "top3_accuracy": top3_mean,
                    "union_component_id": union_ids[row],
                }
            )
        primary = math.fsum(
            float(weight * value) for weight, value in zip(weights_by_row, row_nll, strict=True)
        )
        overall_top1 = math.fsum(
            float(weights_by_row[row] * cast(float, row_records[row]["top1_accuracy"]))
            for row in range(len(row_records))
        )
        overall_top3 = math.fsum(
            float(weights_by_row[row] * cast(float, row_records[row]["top3_accuracy"]))
            for row in range(len(row_records))
        )
        overall_brier = math.fsum(
            float(weights_by_row[row] * cast(float, row_records[row]["mean_brier"]))
            for row in range(len(row_records))
        )
        overall_ece = _ece_from_bins(global_counts, global_confidence, global_correct)
        timestep: dict[str, float] = {}
        for name, (lower, upper) in bins.items():
            row_values: list[float] = []
            for row in range(len(weights_by_row)):
                case_start = row * case_count_by_row
                chosen_values = [
                    case_nll[case]
                    for case in range(case_start, case_start + case_count_by_row)
                    if lower <= int(levels[case]) <= upper
                ]
                _require(bool(chosen_values), f"timestep bin {name} omits a validation row")
                row_values.append(math.fsum(chosen_values) / len(chosen_values))
            timestep[name] = math.fsum(
                float(weight * value)
                for weight, value in zip(weights_by_row, row_values, strict=True)
            )
        case_payload = b"".join(_canonical_json_bytes(item) for item in case_records)
        row_payload = b"".join(_canonical_json_bytes(item) for item in row_records)
        summary = {
            "case_count": len(case_records),
            "case_metrics_sha256": _sha256(case_payload),
            "ece": overall_ece,
            "ledger_sha256": ledger_sha256,
            "mean_brier": overall_brier,
            "method": method,
            "perplexity": math.exp(primary),
            "primary_nll": primary,
            "row_count": len(row_records),
            "row_metrics_sha256": _sha256(row_payload),
            "schema_version": 1,
            "top1_accuracy": overall_top1,
            "top3_accuracy": overall_top3,
        }
        output.append(
            MethodMetrics(
                method=method,
                primary_nll=primary,
                perplexity=math.exp(primary),
                top1_accuracy=overall_top1,
                top3_accuracy=overall_top3,
                mean_brier=overall_brier,
                ece=overall_ece,
                row_nll=tuple(row_nll),
                row_ece=tuple(row_ece),
                timestep_bins=timestep,
                summary=summary,
            )
        )
    return tuple(output)


def _relative_improvement(model: float, control: float, *, label: str) -> float:
    _require(
        math.isfinite(model) and math.isfinite(control) and control > 0.0,
        f"{label} values are invalid",
    )
    value = (control - model) / control
    _require(math.isfinite(value), f"{label} relative improvement is non-finite")
    return value


def _cohort_bootstrap(
    models: Sequence[MethodMetrics],
    baseline: MethodMetrics,
    *,
    corruptions: CorruptionAudit,
    contract: NativeDiffusionContract,
) -> dict[str, object]:
    model_values = tuple(models)
    rows = corruptions.validation_rows
    _require(len(model_values) == len(contract.training.seeds), "bootstrap model cohort differs")
    _require(
        all(len(item.row_nll) == len(rows) for item in (*model_values, baseline)),
        "bootstrap row metrics differ",
    )
    union_ids = tuple(sorted({row.union_component_id for row in rows}))
    _require(len(union_ids) == 57, "bootstrap requires exactly 57 validation union components")
    union_index = {value: index for index, value in enumerate(union_ids)}
    masses = np.zeros(len(union_ids), dtype=np.float64)
    baseline_sums = np.zeros(len(union_ids), dtype=np.float64)
    model_sums = np.zeros((len(model_values), len(union_ids)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        index = union_index[row.union_component_id]
        masses[index] += row.sampling_weight
        baseline_sums[index] += row.sampling_weight * baseline.row_nll[row_index]
        for model_index, model in enumerate(model_values):
            model_sums[model_index, index] += row.sampling_weight * model.row_nll[row_index]

    def estimate(multiplicity: NDArray[np.int64]) -> float:
        mass = math.fsum(float(multiplicity[index] * masses[index]) for index in range(len(masses)))
        control_sum = math.fsum(
            float(multiplicity[index] * baseline_sums[index]) for index in range(len(masses))
        )
        _require(mass > 0.0 and control_sum > 0.0, "bootstrap draw has invalid mass")
        control_nll = control_sum / mass
        improvements: list[float] = []
        for model_index in range(len(model_values)):
            model_sum = math.fsum(
                float(multiplicity[index] * model_sums[model_index, index])
                for index in range(len(masses))
            )
            improvements.append((control_nll - model_sum / mass) / control_nll)
        return math.fsum(improvements) / len(improvements)

    replicates = contract.evaluation.bootstrap_replicates
    samples = np.empty(replicates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_BOOTSTRAP_DOMAIN)
    for replicate in range(replicates):
        rng = np.random.Generator(
            np.random.PCG64DXSM(
                namespaced_seed(contract.evaluation.bootstrap_seed, "bootstrap", replicate)
            )
        )
        draws = rng.integers(0, len(union_ids), size=len(union_ids))
        multiplicity = np.bincount(draws, minlength=len(union_ids)).astype(np.int64)
        for slot, draw in enumerate(draws):
            digest.update(f"{replicate}\t{slot}\t{union_ids[int(draw)]}\n".encode("ascii"))
        samples[replicate] = estimate(multiplicity)
    _require(bool(np.all(np.isfinite(samples))), "bootstrap statistics are non-finite")
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "schema_version": 1,
        "baseline_method": baseline.method,
        "model_methods": [item.method for item in model_values],
        "unit": contract.evaluation.bootstrap_unit,
        "unit_count": len(union_ids),
        "replicates": replicates,
        "seed": contract.evaluation.bootstrap_seed,
        "statistic": "arithmetic_mean_relative_improvement_across_seeds",
        "point_mean_relative_nll_improvement": estimate(np.ones(len(union_ids), dtype=np.int64)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "draws_sha256": digest.hexdigest(),
    }


def _denoising_gate(
    metrics: Sequence[MethodMetrics],
    *,
    corruptions: CorruptionAudit,
    contract: NativeDiffusionContract,
) -> tuple[dict[str, object], MethodMetrics]:
    values = tuple(metrics)
    baseline_count = len(contract.baselines.names)
    _require(
        tuple(item.method for item in values)
        == contract.artifacts.validation_token_stats_method_order,
        "denoising metric method order differs",
    )
    baselines = values[:baseline_count]
    models = values[baseline_count:]
    strongest = min(baselines, key=lambda item: (item.primary_nll, item.method))
    per_seed: list[dict[str, object]] = []
    improvements: list[float] = []
    every_seed_beats = True
    every_seed_ece = True
    gate = contract.denoising_gates
    for seed, model in zip(contract.training.seeds, models, strict=True):
        improvement = _relative_improvement(
            model.primary_nll, strongest.primary_nll, label=f"seed {seed}"
        )
        regression = model.ece - strongest.ece
        beats = model.primary_nll < strongest.primary_nll
        absolute_ece = model.ece <= gate.maximum_ece
        relative_ece = regression <= gate.maximum_ece_regression
        improvements.append(improvement)
        every_seed_beats &= beats
        every_seed_ece &= absolute_ece and relative_ece
        per_seed.append(
            {
                "seed": seed,
                "method": model.method,
                "primary_nll": model.primary_nll,
                "relative_nll_improvement": improvement,
                "beats_strongest_baseline": beats,
                "ece": model.ece,
                "ece_regression": regression,
                "maximum_ece_passed": absolute_ece,
                "maximum_ece_regression_passed": relative_ece,
            }
        )
    mean_improvement = math.fsum(improvements) / len(improvements)
    bootstrap = _cohort_bootstrap(models, strongest, corruptions=corruptions, contract=contract)
    _require(
        math.isclose(
            cast(float, bootstrap["point_mean_relative_nll_improvement"]),
            mean_improvement,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "bootstrap point differs from primary improvement",
    )
    bin_records: list[dict[str, object]] = []
    high_noise_passed = False
    regression_passed = True
    for name in contract.evaluation.timestep_bins:
        baseline_nll = strongest.timestep_bins[name]
        seed_nlls = tuple(model.timestep_bins[name] for model in models)
        model_mean = math.fsum(seed_nlls) / len(seed_nlls)
        improvement = _relative_improvement(model_mean, baseline_nll, label=f"timestep bin {name}")
        regression = -improvement
        within_regression = regression <= gate.maximum_timestep_bin_relative_nll_regression
        regression_passed &= within_regression
        high_pass: bool | None = None
        if name == gate.high_noise_timestep_bin:
            high_pass = improvement >= gate.minimum_high_noise_relative_nll_improvement
            high_noise_passed = high_pass
        bin_records.append(
            {
                "name": name,
                "baseline_nll": baseline_nll,
                "model_seed_nll": list(seed_nlls),
                "model_arithmetic_mean_nll": model_mean,
                "relative_nll_improvement": improvement,
                "relative_nll_regression": regression,
                "maximum_regression_passed": within_regression,
                "high_noise_minimum_passed": high_pass,
            }
        )
    checks = {
        "every_seed_beats_strongest_baseline": every_seed_beats,
        "minimum_mean_relative_nll_improvement": mean_improvement
        >= gate.minimum_mean_relative_nll_improvement,
        "strict_bootstrap_lower_bound": cast(float, bootstrap["lower_95"])
        > gate.minimum_bootstrap_lower_bound_improvement,
        "minimum_high_noise_relative_nll_improvement": high_noise_passed,
        "maximum_timestep_bin_relative_nll_regression": regression_passed,
        "every_seed_ece": every_seed_ece,
    }
    record = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "strongest_baseline_method": strongest.method,
        "per_seed": per_seed,
        "mean_relative_nll_improvement": mean_improvement,
        "bootstrap": bootstrap,
        "timestep_bins": bin_records,
        "checks": checks,
    }
    return record, strongest


def _framed_update(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _bound_checkpoint(config_sha256: str, model_logical_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_CHECKPOINT_BINDING_DOMAIN)
    _framed_update(digest, bytes.fromhex(_require_sha256(config_sha256, label="config SHA")))
    _framed_update(
        digest, bytes.fromhex(_require_sha256(model_logical_sha256, label="model logical SHA"))
    )
    return digest.hexdigest()


def _training_projection_sha256(training: TrainingDistribution) -> str:
    return _sha256(
        b"".join(
            _canonical_json_bytes(
                {
                    "sampling_weight": row.sampling_weight,
                    "sequence": row.sequence,
                    "sequence_id": row.sequence_id,
                }
            )
            for row in training.rows
        )
    )


def _count_control_logical_sha256(
    tables: BaselineTables,
    *,
    training: TrainingDistribution,
    contract: NativeDiffusionContract,
) -> str:
    table_digest = hashlib.sha256()
    table_digest.update(_PROBABILITY_TABLE_DOMAIN)
    for name, values in (
        ("unigram", tables.unigram),
        ("forward_markov", tables.forward),
        ("reverse_markov", tables.reverse),
        ("relative_position", tables.relative),
    ):
        array = np.asarray(values, dtype="<f8", order="C")
        _framed_update(
            table_digest,
            _canonical_json_bytes(
                {
                    "dtype": "little_endian_float64",
                    "name": name,
                    "shape": list(array.shape),
                }
            ),
        )
        _framed_update(table_digest, array.tobytes(order="C"))
    ids_digest = hashlib.sha256()
    ids_digest.update(_TRAIN_IDS_DOMAIN)
    for row in training.rows:
        ids_digest.update(_canonical_json_bytes({"sequence_id": row.sequence_id}))
    record = {
        "config_sha256": CONFIG_SHA256,
        "effective_count_scale": contract.input.expected_train_sequences,
        "names": list(contract.baselines.names),
        "probability_tables_sha256": table_digest.hexdigest(),
        "schema_version": 1,
        "total_effective_forward_transition_evidence": (
            tables.total_effective_forward_transition_evidence
        ),
        "total_effective_residue_evidence": tables.total_effective_residue_evidence,
        "total_effective_reverse_transition_evidence": (
            tables.total_effective_reverse_transition_evidence
        ),
        "training_projection_sha256": _training_projection_sha256(training),
        "training_sequence_ids_sha256": ids_digest.hexdigest(),
    }
    digest = hashlib.sha256()
    digest.update(_CONTROL_LOGICAL_DOMAIN)
    _framed_update(digest, _canonical_json_bytes(record))
    return digest.hexdigest()


def _length_plan_sha256(lengths: Sequence[int]) -> str:
    payload = b"".join(
        _canonical_json_bytes({"length": int(length), "ordinal": ordinal})
        for ordinal, length in enumerate(lengths)
    )
    return _sha256(payload)


def _control_draw_seed(method: str, seed: int, ordinal: int, position: int) -> int:
    digest = hashlib.sha256()
    digest.update(_CONTROL_DRAW_DOMAIN)
    for payload in (
        method.encode("ascii"),
        seed.to_bytes(8, "big"),
        ordinal.to_bytes(8, "big"),
        position.to_bytes(2, "big"),
    ):
        _framed_update(digest, payload)
    return int.from_bytes(digest.digest()[:8], "big")


def _expected_control_sequence(
    *,
    method: str,
    seed: int,
    ordinal: int,
    length: int,
    tables: BaselineTables,
) -> str:
    _require(
        method in {"component_weighted_unigram", "component_weighted_forward_markov"},
        "unknown count-generator control",
    )
    residues: list[int] = []
    for position in range(length):
        probabilities = (
            tables.unigram
            if method == "component_weighted_unigram" or position == 0
            else tables.forward[residues[-1]]
        )
        uniform = float(
            np.random.Generator(
                np.random.PCG64DXSM(_control_draw_seed(method, seed, ordinal, position))
            ).random()
        )
        cumulative = np.cumsum(probabilities, dtype=np.float64)
        cumulative[-1] = 1.0
        residues.append(min(int(np.searchsorted(cumulative, uniform, side="right")), 19))
    return "".join(_ALPHABET[index] for index in residues)


@dataclass(frozen=True, slots=True)
class Proposal:
    method: str
    seed: int
    ordinal: int
    sequence_id: str
    sequence: str
    length: int
    generator_binding_kind: str
    generator_binding_sha256: str
    config_sha256: str
    training_projection_sha256: str
    length_plan_sha256: str


@dataclass(frozen=True, slots=True)
class ProposalAudit:
    records: tuple[Proposal, ...]
    batches: Mapping[tuple[str, int], tuple[Proposal, ...]]
    length_plans: Mapping[int, tuple[int, ...]]
    length_plan_hashes: Mapping[int, str]
    count_control_logical_sha256: str


_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "seed",
        "ordinal",
        "sequence_id",
        "sequence",
        "length",
        "generator_binding_kind",
        "generator_binding_sha256",
        "config_sha256",
        "training_projection_sha256",
        "length_plan_sha256",
    }
)


def _proposal_from_record(value: dict[str, object], *, label: str) -> Proposal:
    row = _exact_keys(value, _CANDIDATE_FIELDS, label=label)
    _require(row["schema_version"] == 1, f"{label} schema_version differs")
    method = row["method"]
    _require(
        type(method) is str and _METHOD_RE.fullmatch(cast(str, method)) is not None,
        f"{label} method is invalid",
    )
    seed = _integer(row["seed"], label=f"{label} seed")
    ordinal = _integer(row["ordinal"], label=f"{label} ordinal")
    _require(seed < 2**64, f"{label} seed must fit uint64")
    _require(ordinal < 2**64, f"{label} ordinal must fit uint64")
    length = _integer(row["length"], label=f"{label} length")
    _require(8 <= length <= 50, f"{label} length must be in 8..50")
    sequence = row["sequence"]
    _require(type(sequence) is str, f"{label} sequence must be a string")
    canonical = canonicalize_sequence(cast(str, sequence), min_length=8, max_length=50)
    _require(canonical == sequence, f"{label} sequence is not canonical")
    _require(length == len(canonical), f"{label} length differs")
    sequence_id = _require_sha256(row["sequence_id"], label=f"{label} sequence_id")
    _require(sequence_id == canonical_sequence_id(canonical), f"{label} sequence ID differs")
    binding_kind = row["generator_binding_kind"]
    _require(
        binding_kind in {"checkpoint_contract_logical_sha256", "count_control_logical_sha256"},
        f"{label} binding kind differs",
    )
    return Proposal(
        method=cast(str, method),
        seed=seed,
        ordinal=ordinal,
        sequence_id=sequence_id,
        sequence=canonical,
        length=length,
        generator_binding_kind=cast(str, binding_kind),
        generator_binding_sha256=_require_sha256(
            row["generator_binding_sha256"], label=f"{label} binding SHA"
        ),
        config_sha256=_require_sha256(row["config_sha256"], label=f"{label} config SHA"),
        training_projection_sha256=_require_sha256(
            row["training_projection_sha256"], label=f"{label} training SHA"
        ),
        length_plan_sha256=_require_sha256(
            row["length_plan_sha256"], label=f"{label} length-plan SHA"
        ),
    )


def _verify_proposals(
    ledger_payload: bytes,
    fasta_payload: bytes,
    *,
    inputs: InputAudit,
    training_audits: Mapping[str, TrainingBundleAudit],
) -> ProposalAudit:
    contract = inputs.contract
    rows = _strict_jsonl(ledger_payload, label="candidate_ledger.jsonl")
    expected_count = (
        len(contract.artifacts.proposal_method_order)
        * len(contract.training.seeds)
        * contract.sampling.raw_proposals_per_seed
    )
    _require(len(rows) == expected_count, "candidate ledger census differs")
    proposals = tuple(
        _proposal_from_record(row, label=f"candidate ledger row {index}")
        for index, row in enumerate(rows, start=1)
    )
    expected_keys = tuple(
        (method, seed, ordinal)
        for method in contract.artifacts.proposal_method_order
        for seed in contract.training.seeds
        for ordinal in range(contract.sampling.raw_proposals_per_seed)
    )
    _require(
        tuple((item.method, item.seed, item.ordinal) for item in proposals) == expected_keys,
        "candidate ledger method/seed/ordinal order differs",
    )
    _require(
        all(item.config_sha256 == CONFIG_SHA256 for item in proposals),
        "candidate ledger config binding differs",
    )
    _require(
        all(
            item.training_projection_sha256 == contract.input.training_projection_sha256
            for item in proposals
        ),
        "candidate ledger training binding differs",
    )

    tables = _fit_baseline_tables(inputs.training, contract=contract)
    control_logical = _count_control_logical_sha256(
        tables, training=inputs.training, contract=contract
    )
    primary_training = {
        42: training_audits["seed-42-primary"],
        43: training_audits["seed-43"],
        44: training_audits["seed-44"],
    }
    batches: dict[tuple[str, int], tuple[Proposal, ...]] = {}
    length_plans: dict[int, tuple[int, ...]] = {}
    length_hashes: dict[int, str] = {}
    cursor = 0
    for method in contract.artifacts.proposal_method_order:
        for seed in contract.training.seeds:
            stop = cursor + contract.sampling.raw_proposals_per_seed
            batch = proposals[cursor:stop]
            batches[(method, seed)] = batch
            lengths = tuple(item.length for item in batch)
            digest = _length_plan_sha256(lengths)
            if seed in length_plans:
                _require(length_plans[seed] == lengths, "methods do not share a seed length plan")
            else:
                length_plans[seed] = lengths
                length_hashes[seed] = digest
            _require(
                all(item.length_plan_sha256 == digest for item in batch),
                "candidate length-plan binding differs",
            )
            if method == "native_categorical_diffusion":
                expected_binding = _bound_checkpoint(
                    CONFIG_SHA256, primary_training[seed].checkpoint_logical_sha256
                )
                _require(
                    all(
                        item.generator_binding_kind == "checkpoint_contract_logical_sha256"
                        and item.generator_binding_sha256 == expected_binding
                        for item in batch
                    ),
                    "native candidate checkpoint binding differs",
                )
            else:
                _require(
                    all(
                        item.generator_binding_kind == "count_control_logical_sha256"
                        and item.generator_binding_sha256 == control_logical
                        for item in batch
                    ),
                    "control candidate logical binding differs",
                )
                expected_sequences = tuple(
                    _expected_control_sequence(
                        method=method,
                        seed=seed,
                        ordinal=item.ordinal,
                        length=item.length,
                        tables=tables,
                    )
                    for item in batch
                )
                _require(
                    tuple(item.sequence for item in batch) == expected_sequences,
                    "count-control proposals differ from independent deterministic regeneration",
                )
            cursor = stop

    for seed, lengths in length_plans.items():
        expected = inputs.training.length_prior.draw(
            root_seed=seed,
            draw_start=0,
            draw_count=contract.sampling.raw_proposals_per_seed,
            namespace="proposal",
        )
        _require(lengths == expected, f"seed {seed} length plan differs from training-only prior")

    expected_fasta = b"".join(
        (
            f">{item.method}|seed={item.seed}|ordinal={item.ordinal}|sequence_id={item.sequence_id}\n"
            f"{item.sequence}\n"
        ).encode("ascii")
        for item in proposals
    )
    _require(fasta_payload == expected_fasta, "raw proposal FASTA differs from candidate ledger")
    return ProposalAudit(
        records=proposals,
        batches=batches,
        length_plans=length_plans,
        length_plan_hashes=length_hashes,
        count_control_logical_sha256=control_logical,
    )


def _sequence_collection_sha256(sequences: Sequence[str], *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(_SEQUENCE_COLLECTION_DOMAIN)
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    for sequence in sequences:
        digest.update(_canonical_json_bytes({"sequence": sequence}))
    return digest.hexdigest()


def _maximum_ratio(sequence: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    match = process.extractOne(sequence, references, scorer=indel_ratio)
    _require(match is not None, "nonempty reference search returned no match")
    return float(cast(tuple[object, float, object], match)[1])


def _cluster_sequences_exact(
    sequences: Sequence[str],
    *,
    identity_threshold: float,
) -> tuple[tuple[str, ...], ...]:
    """Reconstruct exact single-link components after a necessary Indel filter.

    The matches in any global alignment are a common subsequence, so Indel/LCS
    similarity cannot be lower than that alignment's matches divided by its
    alignment length.  Pairs rejected here cannot meet the exact frozen
    identity threshold.  A conservative boundary margin is followed by the
    unchanged exact global-alignment calculation for every surviving pair.
    """

    _require(
        0.0 <= identity_threshold <= 1.0,
        "identity threshold must lie in [0, 1]",
    )
    canonical = sorted(
        {canonicalize_sequence(sequence, min_length=1, max_length=10**9) for sequence in sequences}
    )
    parent = list(range(len(canonical)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    indel_cutoff = max(0.0, identity_threshold - _CLUSTER_PREFILTER_MARGIN)
    for left_index, left in enumerate(canonical):
        for right_index in range(left_index + 1, len(canonical)):
            right = canonical[right_index]
            if min(len(left), len(right)) / max(len(left), len(right)) < identity_threshold:
                continue
            if float(indel_ratio(left, right)) < indel_cutoff:
                continue
            if find(left_index) == find(right_index):
                continue
            if global_sequence_identity(left, right) >= identity_threshold:
                union(left_index, right_index)

    components: dict[int, list[str]] = {}
    for index, sequence in enumerate(canonical):
        components.setdefault(find(index), []).append(sequence)
    output = [tuple(sorted(component)) for component in components.values()]
    return tuple(sorted(output, key=lambda component: component[0]))


def _candidate_diagnostics(
    candidates: Sequence[str],
    *,
    training: Sequence[str],
    references: Sequence[str],
    contract: NativeDiffusionContract,
) -> tuple[dict[str, object], tuple[str, ...]]:
    raw = tuple(candidates)
    valid: list[str] = []
    for sequence in raw:
        try:
            canonical = canonicalize_sequence(sequence)
        except (TypeError, ValueError):
            continue
        if canonical == sequence:
            valid.append(sequence)
    unique = tuple(sorted(set(valid)))
    train_values = tuple(training)
    _require(
        len(train_values) == len(set(train_values)),
        "training projection contains duplicate sequences",
    )
    reference_values = tuple(sorted(set(references)))
    train_set = set(train_values)
    reference_set = set(reference_values)
    common = tuple(sequence for sequence in unique if sequence not in reference_set)
    exact_train = sum(sequence in train_set for sequence in valid)
    exact_reference = sum(sequence in reference_set for sequence in unique)
    top_safe = sum(
        _maximum_ratio(sequence, reference_values) <= contract.sampling.top_reference_ratio
        for sequence in common
    )
    nearest = np.asarray(
        [_maximum_ratio(sequence, train_values) for sequence in common], dtype=np.float64
    )
    if len(nearest):
        q50, q90, q99 = np.quantile(nearest, [0.50, 0.90, 0.99], method="linear")
    else:
        q50 = q90 = q99 = 0.0
    components = (
        _cluster_sequences_exact(common, identity_threshold=contract.sampling.identity_threshold)
        if common
        else ()
    )
    sizes = np.asarray([len(component) for component in components], dtype=np.float64)
    if len(sizes):
        proportions = sizes / np.sum(sizes)
        hill2 = float(1.0 / np.sum(np.square(proportions)))
        largest = float(np.max(proportions))
    else:
        hill2 = largest = 0.0
    residue_counts = Counter("".join(common))
    total = sum(residue_counts.values())
    if total:
        probabilities = np.asarray(
            [count / total for count in residue_counts.values()], dtype=np.float64
        )
        entropy = float(-np.sum(probabilities * np.log2(probabilities)) / math.log2(20))
        maximum_fraction = max(residue_counts.values()) / total
    else:
        entropy = maximum_fraction = 0.0
    raw_count = len(raw)
    record = {
        "schema_version": 1,
        "raw_count": raw_count,
        "canonical_valid_count": len(valid),
        "canonical_valid_fraction": len(valid) / raw_count,
        "unique_valid_count": len(unique),
        "raw_unique_fraction": len(unique) / raw_count,
        "exact_train_overlap_count": exact_train,
        "exact_train_overlap_fraction": exact_train / raw_count,
        "exact_reference_overlap_count": exact_reference,
        "common_funnel_count": len(common),
        "common_funnel_yield_fraction": len(common) / raw_count,
        "top_reference_safe_count": top_safe,
        "identity_70_cluster_count": len(components),
        "hill2_effective_70pct_clusters": hill2,
        "largest_70pct_cluster_fraction": largest,
        "nearest_train_indel_ratio_q50": float(q50),
        "nearest_train_indel_ratio_q90": float(q90),
        "nearest_train_indel_ratio_q99": float(q99),
        "residue_entropy_fraction": entropy,
        "maximum_residue_fraction": maximum_fraction,
        "top_reference_ratio_threshold": contract.sampling.top_reference_ratio,
        "identity_threshold": contract.sampling.identity_threshold,
        "training_sequences_sha256": _sequence_collection_sha256(train_values, label="training"),
        "reference_sequences_sha256": _sequence_collection_sha256(
            reference_values, label="reference"
        ),
    }
    pool = tuple(sequence for sequence in unique if sequence not in reference_set)
    _require(bool(pool), "distribution candidate pool is empty")
    return record, pool


def _ngram_jsd(
    candidates: Sequence[str],
    training: TrainingDistribution,
    *,
    order: int,
) -> float:
    def masses(sequences: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
        contributions: dict[str, list[float]] = defaultdict(list)
        for sequence, weight in sorted(zip(sequences, weights, strict=True)):
            available = len(sequence) - order + 1
            contribution = weight / available
            for position in range(available):
                contributions[sequence[position : position + order]].append(contribution)
        result = {key: math.fsum(values) for key, values in contributions.items()}
        _require(
            math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12),
            "n-gram mass does not sum to one",
        )
        return result

    candidate_weights = (1.0 / len(candidates),) * len(candidates)
    training_sequences = tuple(row.sequence for row in training.rows)
    training_weights = tuple(row.sampling_weight for row in training.rows)
    left = masses(candidates, candidate_weights)
    right = masses(training_sequences, training_weights)
    support = tuple(sorted(set(left) | set(right)))
    p = np.asarray([left.get(item, 0.0) for item in support], dtype=np.float64)
    q = np.asarray([right.get(item, 0.0) for item in support], dtype=np.float64)
    midpoint = 0.5 * (p + q)

    def kl(values: FloatArray) -> float:
        selected = values > 0.0
        return float(np.sum(values[selected] * np.log2(values[selected] / midpoint[selected])))

    result = 0.5 * (kl(p) + kl(q))
    _require(math.isfinite(result) and -1e-12 <= result <= 1.0 + 1e-12, "n-gram JSD is invalid")
    return min(max(result, 0.0), 1.0)


def _descriptor_matrix(sequences: Sequence[str]) -> FloatArray:
    matrix = np.empty((len(sequences), len(_DESCRIPTOR_FEATURES)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        descriptors = compute_descriptors(sequence)
        matrix[row] = tuple(float(getattr(descriptors, name)) for name in _DESCRIPTOR_FEATURES)
    _require(bool(np.all(np.isfinite(matrix))), "descriptor matrix contains non-finite values")
    return matrix


def _descriptor_expectation(
    left: FloatArray,
    right: FloatArray,
    left_weights: FloatArray,
    right_weights: FloatArray,
) -> float:
    contributions: list[float] = []
    chunk = 256
    for left_start in range(0, len(left), chunk):
        for right_start in range(0, len(right), chunk):
            left_values = left[left_start : left_start + chunk]
            right_values = right[right_start : right_start + chunk]
            difference = left_values[:, None, :] - right_values[None, :, :]
            distances = np.sqrt(np.sum(np.square(difference), axis=2))
            pair_weights = (
                left_weights[left_start : left_start + chunk, None]
                * right_weights[None, right_start : right_start + chunk]
            )
            contributions.append(float(np.sum(distances * pair_weights)))
    value = math.fsum(contributions)
    _require(math.isfinite(value) and value >= 0.0, "descriptor expectation is invalid")
    return value


def _descriptor_distance(
    candidates: Sequence[str],
    training: TrainingDistribution,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    candidate = _descriptor_matrix(candidates)
    reference = _descriptor_matrix(tuple(row.sequence for row in training.rows))
    reference_weights = np.asarray([row.sampling_weight for row in training.rows], dtype=np.float64)
    candidate_weights = np.full(len(candidate), 1.0 / len(candidate), dtype=np.float64)
    mean = np.asarray(
        [
            math.fsum(
                float(reference_weights[row] * reference[row, column])
                for row in range(len(reference))
            )
            for column in range(reference.shape[1])
        ],
        dtype=np.float64,
    )
    variance = np.asarray(
        [
            math.fsum(
                float(reference_weights[row] * (reference[row, column] - mean[column]) ** 2)
                for row in range(len(reference))
            )
            for column in range(reference.shape[1])
        ],
        dtype=np.float64,
    )
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-12)
    candidate = (candidate - mean) / scale
    reference = (reference - mean) / scale
    distance = (
        2.0 * _descriptor_expectation(candidate, reference, candidate_weights, reference_weights)
        - _descriptor_expectation(candidate, candidate, candidate_weights, candidate_weights)
        - _descriptor_expectation(reference, reference, reference_weights, reference_weights)
    )
    _require(
        math.isfinite(distance) and distance >= -1e-10, "descriptor energy distance is negative"
    )
    return (
        max(0.0, distance),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
    )


@dataclass(frozen=True, slots=True)
class SamplingMetric:
    method: str
    seed: int
    length_plan_sha256: str
    diagnostics: Mapping[str, object]
    ngram_1: Mapping[str, object]
    ngram_3: Mapping[str, object]
    descriptor: Mapping[str, object]

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": self.method,
            "seed": self.seed,
            "length_plan_sha256": self.length_plan_sha256,
            "diagnostics": dict(self.diagnostics),
            "ngram_jsd": {"1": dict(self.ngram_1), "3": dict(self.ngram_3)},
            "descriptor_energy_distance": dict(self.descriptor),
        }


def _sampling_metrics(
    proposals: ProposalAudit,
    *,
    inputs: InputAudit,
) -> tuple[SamplingMetric, ...]:
    contract = inputs.contract
    training_sequences = tuple(row.sequence for row in inputs.training.rows)
    output: list[SamplingMetric] = []
    for method in contract.artifacts.proposal_method_order:
        for seed in contract.training.seeds:
            batch = proposals.batches[(method, seed)]
            candidates = tuple(item.sequence for item in batch)
            diagnostics, pool = _candidate_diagnostics(
                candidates,
                training=training_sequences,
                references=inputs.reference_sequences,
                contract=contract,
            )
            pool_sha = _sequence_collection_sha256(pool, label="candidate_pool")
            ngram_records: list[dict[str, object]] = []
            for order in (1, 3):
                ngram_records.append(
                    {
                        "bits": _ngram_jsd(pool, inputs.training, order=order),
                        "candidate_count": len(pool),
                        "candidate_pool_sha256": pool_sha,
                        "candidate_weighting": "equal_weight_divided_by_available_ngrams",
                        "order": order,
                        "reference_count": len(inputs.training.rows),
                        "reference_weighting": "sampling_weight_divided_by_available_ngrams",
                        "schema_version": 1,
                        "training_projection_sha256": contract.input.training_projection_sha256,
                    }
                )
            distance, mean, scale = _descriptor_distance(pool, inputs.training)
            descriptor = {
                "candidate_count": len(pool),
                "candidate_pool_sha256": pool_sha,
                "distance": distance,
                "estimator": contract.sampling.descriptor_energy_distance,
                "feature_names": list(contract.sampling.descriptor_features),
                "reference_count": len(inputs.training.rows),
                "reference_population": contract.sampling.distribution_reference,
                "scale_floor": 1e-12,
                "schema_version": 1,
                "standardization": contract.sampling.descriptor_standardization,
                "training_mean": list(mean),
                "training_projection_sha256": contract.input.training_projection_sha256,
                "training_scale": list(scale),
            }
            output.append(
                SamplingMetric(
                    method=method,
                    seed=seed,
                    length_plan_sha256=proposals.length_plan_hashes[seed],
                    diagnostics=diagnostics,
                    ngram_1=ngram_records[0],
                    ngram_3=ngram_records[1],
                    descriptor=descriptor,
                )
            )
    return tuple(output)


def _distance_improvement(candidate: float, control: float) -> float:
    if control == 0.0:
        return 0.0 if candidate == 0.0 else -math.inf
    return (control - candidate) / control


def _sampling_gate(
    metrics: Sequence[SamplingMetric],
    *,
    contract: NativeDiffusionContract,
) -> dict[str, object]:
    values = tuple(metrics)
    by_key = {(item.method, item.seed): item for item in values}
    native = contract.artifacts.proposal_method_order[0]
    controls = contract.artifacts.proposal_method_order[1:]
    gate = contract.sampling_gates
    per_seed: list[dict[str, object]] = []
    absolute_passed = True
    for seed in contract.training.seeds:
        diagnostics = by_key[(native, seed)].diagnostics
        checks = {
            "canonical_valid_fraction": cast(float, diagnostics["canonical_valid_fraction"])
            >= gate.minimum_canonical_valid_fraction_each_seed,
            "raw_unique_fraction": cast(float, diagnostics["raw_unique_fraction"])
            >= gate.minimum_raw_unique_fraction_each_seed,
            "exact_train_overlap_fraction": cast(float, diagnostics["exact_train_overlap_fraction"])
            <= gate.maximum_exact_train_overlap_fraction_each_seed,
            "common_funnel_yield_fraction": cast(float, diagnostics["common_funnel_yield_fraction"])
            >= gate.minimum_common_funnel_yield_fraction_each_seed,
            "top_reference_safe_count": cast(int, diagnostics["top_reference_safe_count"])
            >= gate.minimum_top_reference_safe_count_each_seed,
            "hill2_effective_70pct_clusters": cast(
                float, diagnostics["hill2_effective_70pct_clusters"]
            )
            >= gate.minimum_hill2_effective_70pct_clusters_each_seed,
            "largest_70pct_cluster_fraction": cast(
                float, diagnostics["largest_70pct_cluster_fraction"]
            )
            <= gate.maximum_largest_70pct_cluster_fraction_each_seed,
        }
        passed = all(checks.values())
        absolute_passed &= passed
        per_seed.append(
            {"seed": seed, "passed": passed, "checks": checks, "diagnostics": dict(diagnostics)}
        )

    def get(item: SamplingMetric, name: str) -> float:
        if name == "common_funnel_yield_fraction":
            return cast(float, item.diagnostics[name])
        if name == "ngram_1_jsd_bits":
            return cast(float, item.ngram_1["bits"])
        if name == "ngram_3_jsd_bits":
            return cast(float, item.ngram_3["bits"])
        return cast(float, item.descriptor["distance"])

    names = (
        "common_funnel_yield_fraction",
        "ngram_1_jsd_bits",
        "ngram_3_jsd_bits",
        "descriptor_energy_distance",
    )
    medians: dict[str, dict[str, float]] = {}
    method_records: list[dict[str, object]] = []
    for method in contract.artifacts.proposal_method_order:
        record = {
            name: float(
                median(get(by_key[(method, seed)], name) for seed in contract.training.seeds)
            )
            for name in names
        }
        medians[method] = record
        method_records.append({"method": method, **record})
    native_values = medians[native]
    best_yield = max(medians[item][names[0]] for item in controls)
    best_one = min(medians[item][names[1]] for item in controls)
    best_three = min(medians[item][names[2]] for item in controls)
    best_descriptor = min(medians[item][names[3]] for item in controls)
    yield_deficit = best_yield - native_values[names[0]]
    one_regression = native_values[names[1]] - best_one
    three_regression = native_values[names[2]] - best_three
    descriptor_ratio = (
        (1.0 if native_values[names[3]] == 0.0 else math.inf)
        if best_descriptor == 0.0
        else native_values[names[3]] / best_descriptor
    )
    three_improvement = _distance_improvement(native_values[names[2]], best_three)
    descriptor_improvement = _distance_improvement(native_values[names[3]], best_descriptor)
    comparisons: dict[str, object] = {
        "best_control_common_funnel_yield_fraction": best_yield,
        "native_common_funnel_yield_deficit": yield_deficit,
        "common_funnel_yield_passed": yield_deficit
        <= gate.maximum_common_funnel_yield_deficit_vs_best_control,
        "best_control_ngram_1_jsd_bits": best_one,
        "native_ngram_1_jsd_regression_bits": one_regression,
        "ngram_1_jsd_passed": one_regression
        <= gate.maximum_ngram_jsd_regression_bits_vs_best_control,
        "best_control_ngram_3_jsd_bits": best_three,
        "native_ngram_3_jsd_regression_bits": three_regression,
        "ngram_3_jsd_passed": three_regression
        <= gate.maximum_ngram_jsd_regression_bits_vs_best_control,
        "best_control_descriptor_energy_distance": best_descriptor,
        "native_descriptor_energy_distance_ratio": descriptor_ratio,
        "descriptor_ratio_passed": descriptor_ratio
        <= gate.maximum_descriptor_energy_distance_ratio_vs_best_control,
        "native_ngram_3_relative_improvement": three_improvement,
        "native_descriptor_relative_improvement": descriptor_improvement,
        "minimum_one_distance_improvement_passed": max(three_improvement, descriptor_improvement)
        >= gate.minimum_improvement_in_3mer_jsd_or_descriptor_distance,
    }
    relative_passed = all(
        cast(bool, comparisons[name])
        for name in (
            "common_funnel_yield_passed",
            "ngram_1_jsd_passed",
            "ngram_3_jsd_passed",
            "descriptor_ratio_passed",
            "minimum_one_distance_improvement_passed",
        )
    )
    if not math.isfinite(descriptor_ratio):
        comparisons["native_descriptor_energy_distance_ratio"] = float(np.finfo(np.float64).max)
    if not math.isfinite(three_improvement):
        comparisons["native_ngram_3_relative_improvement"] = -float(np.finfo(np.float64).max)
    if not math.isfinite(descriptor_improvement):
        comparisons["native_descriptor_relative_improvement"] = -float(np.finfo(np.float64).max)
    return {
        "schema_version": 1,
        "passed": absolute_passed and relative_passed,
        "absolute_each_seed_passed": absolute_passed,
        "relative_median_passed": relative_passed,
        "per_seed": per_seed,
        "method_medians": method_records,
        "relative_comparisons": comparisons,
    }


def _assert_close(observed: object, expected: object, *, label: str) -> None:
    if type(expected) is float:
        _require(
            type(observed) is float and math.isfinite(cast(float, observed)),
            f"{label} must be a finite float",
        )
        _require(
            math.isclose(cast(float, observed), expected, rel_tol=1e-13, abs_tol=1e-13),
            f"{label} differs: {observed!r} != {expected!r}",
        )
        return
    _require(type(observed) is type(expected), f"{label} type differs")
    if type(expected) is dict:
        observed_dict = cast(dict[str, object], observed)
        expected_dict = cast(dict[str, object], expected)
        _require(set(observed_dict) == set(expected_dict), f"{label} keys differ")
        for key in expected_dict:
            _assert_close(observed_dict[key], expected_dict[key], label=f"{label}.{key}")
        return
    if type(expected) is list:
        observed_list = cast(list[object], observed)
        expected_list = cast(list[object], expected)
        _require(len(observed_list) == len(expected_list), f"{label} length differs")
        for index, (left, right) in enumerate(zip(observed_list, expected_list, strict=True)):
            _assert_close(left, right, label=f"{label}[{index}]")
        return
    _require(observed == expected, f"{label} differs: {observed!r} != {expected!r}")


def _parse_training_binding(
    payload: bytes,
    *,
    audits: Mapping[str, TrainingBundleAudit],
    contract: NativeDiffusionContract,
) -> str:
    labels = contract.artifacts.training_bundle_binding_labels
    expected = b"".join(
        f"{audits[label].manifest_sha256}  {label}\n".encode("ascii") for label in labels
    )
    _require(payload == expected, "training_bundle.sha256 differs from verified manifests")
    _require(
        audits[labels[0]].manifest_sha256 == audits[labels[1]].manifest_sha256,
        "seed-42 twin manifests differ",
    )
    _require(
        audits[labels[0]].file_hashes == audits[labels[1]].file_hashes,
        "seed-42 twin bundle bytes differ",
    )
    _require(
        audits[labels[0]].environment_identity == audits[labels[1]].environment_identity,
        "seed-42 twin environments differ",
    )
    return _sha256(payload)


def _length_plan_document(
    proposals: ProposalAudit,
    *,
    contract: NativeDiffusionContract,
) -> dict[str, object]:
    return {
        "schema_version": contract.artifacts.schema_version,
        "config_sha256": CONFIG_SHA256,
        "distribution": contract.sampling.length_distribution,
        "shared_across_methods": contract.sampling.shared_length_plan_across_methods,
        "seeds": [
            {
                "seed": seed,
                "count": len(proposals.length_plans[seed]),
                "length_plan_sha256": proposals.length_plan_hashes[seed],
                "items": [
                    {"ordinal": ordinal, "length": length}
                    for ordinal, length in enumerate(proposals.length_plans[seed])
                ],
            }
            for seed in contract.training.seeds
        ],
    }


def _producer_evidence(value: object) -> tuple[dict[str, object], bool]:
    document = _exact_keys(value, {"schema_version", "passed", "checks"}, label="producer evidence")
    _require(document["schema_version"] == 1, "producer evidence schema_version differs")
    checks = document["checks"]
    _require(
        type(checks) is list and len(cast(list[object], checks)) == 8,
        "producer evidence check ledger differs",
    )
    names = (
        "accepted_input_hashes",
        "clean_synchronized_git",
        "training_bundle_integrity",
        "runtime_environment",
        "deterministic_execution",
        "validation_execution",
        "sampling_execution",
        "finite_values",
    )
    observed: list[dict[str, object]] = []
    for raw, name in zip(cast(list[object], checks), names, strict=True):
        row = _exact_keys(raw, {"name", "passed"}, label=f"producer evidence {name}")
        _require(
            row["name"] == name and type(row["passed"]) is bool, f"producer evidence {name} differs"
        )
        observed.append(row)
    passed = all(cast(bool, row["passed"]) for row in observed)
    _require(document["passed"] is passed, "producer evidence aggregate differs")
    return document, passed


@dataclass(frozen=True, slots=True)
class EvaluationAudit:
    root: Path
    snapshots: Mapping[str, Snapshot]
    manifest: Mapping[str, object]
    manifest_sha256: str
    logical_sha256: str
    metrics: tuple[MethodMetrics, ...]
    denoising_gate: Mapping[str, object]
    sampling_metrics: tuple[SamplingMetric, ...]
    sampling_gate: Mapping[str, object]
    decision_status: str
    producer_evidence_passed: bool


def _evaluation_logical_sha256(manifest: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(b"amp-native-diffusion-evaluation-bundle-v1\0")
    _framed_update(digest, _canonical_json_bytes(dict(manifest)))
    return digest.hexdigest()


def _verify_evaluation_bundle(
    path: str | Path,
    *,
    inputs: InputAudit,
    training_audits: Mapping[str, TrainingBundleAudit],
) -> EvaluationAudit:
    contract = inputs.contract
    root, snapshots = _read_bundle(
        path, expected_files=contract.artifacts.evaluation_bundle_files, label="evaluation bundle"
    )
    _require(
        snapshots["contract.toml"].payload == inputs.contract_snapshot.payload,
        "evaluation contract copy differs",
    )
    manifest = _strict_json(snapshots["manifest.json"].payload, label="evaluation manifest")
    _exact_keys(
        manifest, contract.artifacts.evaluation_manifest_fields, label="evaluation manifest"
    )
    _require(
        manifest["schema_version"] == contract.artifacts.schema_version,
        "evaluation manifest schema_version differs",
    )
    _require(manifest["artifact"] == contract.artifact, "evaluation manifest artifact differs")
    _require(manifest["config_sha256"] == CONFIG_SHA256, "evaluation config binding differs")
    git_commit = _require_git(manifest["git_commit"], label="evaluation git_commit")
    _require(
        all(item.git_commit == git_commit for item in training_audits.values()),
        "training/evaluation Git commits differ",
    )
    _require(
        manifest["seeds"] == list(contract.artifacts.evaluation_seed_order),
        "evaluation seed cohort differs",
    )
    artifact_hashes = _exact_keys(
        manifest["artifacts"],
        set(contract.artifacts.evaluation_bundle_files) - {"manifest.json"},
        label="evaluation artifact hashes",
    )
    for name, snapshot in snapshots.items():
        if name != "manifest.json":
            _require(
                artifact_hashes[name] == snapshot.sha256,
                f"evaluation artifact hash differs for {name}",
            )
    binding_sha = _parse_training_binding(
        snapshots["training_bundle.sha256"].payload,
        audits=training_audits,
        contract=contract,
    )
    _require(
        manifest["training_bundle_sha256"] == binding_sha,
        "evaluation training-bundle binding differs",
    )

    corruption_specs = _parse_array_specs(
        contract.artifacts.validation_corruptions_npz_schema, label="corruption NPZ"
    )
    token_specs = _parse_array_specs(
        contract.artifacts.validation_token_stats_npz_schema, label="token-stat NPZ"
    )
    corruption_arrays = _read_npz(
        snapshots["validation_corruptions.npz"],
        specs=corruption_specs,
        label="validation_corruptions.npz",
    )
    corruptions = _verify_corruptions(corruption_arrays, inputs=inputs)
    token_arrays = _read_npz(
        snapshots["validation_token_stats.npz"],
        specs=token_specs,
        label="validation_token_stats.npz",
    )
    statistics = _verify_token_statistics(token_arrays, corruptions=corruptions, contract=contract)
    _verify_baseline_token_statistics(statistics, corruptions=corruptions, inputs=inputs)
    metrics = _reconstruct_method_metrics(statistics, corruptions=corruptions, contract=contract)
    denoising, strongest = _denoising_gate(metrics, corruptions=corruptions, contract=contract)

    proposals = _verify_proposals(
        snapshots["candidate_ledger.jsonl"].payload,
        snapshots["raw_proposals.fasta"].payload,
        inputs=inputs,
        training_audits=training_audits,
    )
    length_document = _strict_json(snapshots["length_plan.json"].payload, label="length_plan.json")
    _exact_keys(
        length_document,
        {"schema_version", "config_sha256", "distribution", "shared_across_methods", "seeds"},
        label="length plan",
    )
    _assert_close(
        length_document, _length_plan_document(proposals, contract=contract), label="length plan"
    )
    sampling_metrics = _sampling_metrics(proposals, inputs=inputs)
    sampling_gate = _sampling_gate(sampling_metrics, contract=contract)

    baseline_expected = {
        "schema_version": contract.artifacts.schema_version,
        "config_sha256": CONFIG_SHA256,
        "method_order": list(contract.baselines.names),
        "methods": [item.summary_record() for item in metrics[: len(contract.baselines.names)]],
        "strongest_method": strongest.method,
    }
    baseline_document = _strict_json(
        snapshots["baseline_metrics.json"].payload, label="baseline_metrics.json"
    )
    _exact_keys(
        baseline_document,
        {"schema_version", "config_sha256", "method_order", "methods", "strongest_method"},
        label="baseline metrics",
    )
    _assert_close(baseline_document, baseline_expected, label="baseline metrics")
    primary_training = {
        42: training_audits["seed-42-primary"],
        43: training_audits["seed-43"],
        44: training_audits["seed-44"],
    }
    validation_expected = {
        "schema_version": contract.artifacts.schema_version,
        "config_sha256": CONFIG_SHA256,
        "seeds": list(contract.training.seeds),
        "method_order": list(contract.artifacts.validation_token_stats_method_order),
        "models": [
            {
                "seed": seed,
                "checkpoint_contract_logical_sha256": _bound_checkpoint(
                    CONFIG_SHA256, primary_training[seed].checkpoint_logical_sha256
                ),
                "metrics": model.summary_record(),
            }
            for seed, model in zip(
                contract.training.seeds, metrics[len(contract.baselines.names) :], strict=True
            )
        ],
        "bootstrap": denoising["bootstrap"],
        "timestep_bins": denoising["timestep_bins"],
    }
    validation_document = _strict_json(
        snapshots["validation_metrics.json"].payload, label="validation_metrics.json"
    )
    _exact_keys(
        validation_document,
        {
            "schema_version",
            "config_sha256",
            "seeds",
            "method_order",
            "models",
            "bootstrap",
            "timestep_bins",
        },
        label="validation metrics",
    )
    _assert_close(validation_document, validation_expected, label="validation metrics")
    sampling_expected = {
        "schema_version": contract.artifacts.schema_version,
        "config_sha256": CONFIG_SHA256,
        "seeds": list(contract.training.seeds),
        "method_order": list(contract.artifacts.proposal_method_order),
        "methods": [item.canonical_record() for item in sampling_metrics],
        "aggregates": sampling_gate["method_medians"],
        "gates": sampling_gate,
    }
    sampling_document = _strict_json(
        snapshots["sampling_metrics.json"].payload, label="sampling_metrics.json"
    )
    _exact_keys(
        sampling_document,
        {
            "schema_version",
            "config_sha256",
            "seeds",
            "method_order",
            "methods",
            "aggregates",
            "gates",
        },
        label="sampling metrics",
    )
    _assert_close(sampling_document, sampling_expected, label="sampling metrics")

    gates = _exact_keys(
        manifest["gates"],
        {
            "schema_version",
            "decision_precedence",
            "evidence",
            "denoising",
            "sampling",
            "decision_status",
        },
        label="evaluation gates",
    )
    evidence, evidence_passed = _producer_evidence(gates["evidence"])
    expected_status = (
        contract.status.evidence_invalid
        if not evidence_passed
        else contract.status.candidate_generator_only
        if cast(bool, denoising["passed"]) and cast(bool, sampling_gate["passed"])
        else contract.status.reproducible_no_go
    )
    expected_gates = {
        "schema_version": 1,
        "decision_precedence": [
            "evidence_invalid",
            "reproducible_no_go",
            "candidate_generator_only",
        ],
        "evidence": evidence,
        "denoising": denoising,
        "sampling": sampling_gate,
        "decision_status": expected_status,
    }
    _assert_close(gates, expected_gates, label="evaluation gates")
    _require(manifest["decision_status"] == expected_status, "evaluation decision status differs")

    validation_manifest = _exact_keys(
        manifest["validation"],
        {
            "accepted_corpus_sha256",
            "validation_fold",
            "validation_sequences",
            "corruption_cases",
            "selected_tokens",
            "ledger_sha256",
            "method_order",
            "corruptions_sha256",
            "token_stats_sha256",
            "metrics_sha256",
        },
        label="evaluation validation manifest",
    )
    _require(
        validation_manifest
        == {
            "accepted_corpus_sha256": contract.input.corpus_sha256,
            "validation_fold": contract.input.validation_fold,
            "validation_sequences": contract.evaluation.expected_validation_sequences,
            "corruption_cases": contract.evaluation.expected_corruption_cases,
            "selected_tokens": int(statistics.case_offsets[-1]),
            "ledger_sha256": metrics[0].summary["ledger_sha256"],
            "method_order": list(contract.artifacts.validation_token_stats_method_order),
            "corruptions_sha256": snapshots["validation_corruptions.npz"].sha256,
            "token_stats_sha256": snapshots["validation_token_stats.npz"].sha256,
            "metrics_sha256": snapshots["validation_metrics.json"].sha256,
        },
        "evaluation validation manifest differs",
    )
    baseline_manifest = _exact_keys(
        manifest["baselines"],
        {"method_order", "strongest_method", "metrics_sha256"},
        label="evaluation baseline manifest",
    )
    _require(
        baseline_manifest
        == {
            "method_order": list(contract.baselines.names),
            "strongest_method": strongest.method,
            "metrics_sha256": snapshots["baseline_metrics.json"].sha256,
        },
        "evaluation baseline manifest differs",
    )
    sampling_manifest = _exact_keys(
        manifest["sampling"],
        {
            "method_order",
            "raw_proposals_per_method_seed",
            "total_raw_proposals",
            "length_plan_sha256",
            "fasta_sha256",
            "candidate_ledger_sha256",
            "metrics_sha256",
            "training_projection_sha256",
            "organizer_reference_sha256",
            "organizer_reference_records",
        },
        label="evaluation sampling manifest",
    )
    _require(
        sampling_manifest
        == {
            "method_order": list(contract.artifacts.proposal_method_order),
            "raw_proposals_per_method_seed": contract.sampling.raw_proposals_per_seed,
            "total_raw_proposals": len(proposals.records),
            "length_plan_sha256": snapshots["length_plan.json"].sha256,
            "fasta_sha256": snapshots["raw_proposals.fasta"].sha256,
            "candidate_ledger_sha256": snapshots["candidate_ledger.jsonl"].sha256,
            "metrics_sha256": snapshots["sampling_metrics.json"].sha256,
            "training_projection_sha256": contract.input.training_projection_sha256,
            "organizer_reference_sha256": contract.input.organizer_reference_sha256,
            "organizer_reference_records": contract.input.organizer_reference_records,
        },
        "evaluation sampling manifest differs",
    )
    for snapshot in snapshots.values():
        _snapshot_unchanged(snapshot, label=f"evaluation bundle/{snapshot.path.name}")
    return EvaluationAudit(
        root=root,
        snapshots=snapshots,
        manifest=manifest,
        manifest_sha256=snapshots["manifest.json"].sha256,
        logical_sha256=_evaluation_logical_sha256(manifest),
        metrics=metrics,
        denoising_gate=denoising,
        sampling_metrics=sampling_metrics,
        sampling_gate=sampling_gate,
        decision_status=expected_status,
        producer_evidence_passed=evidence_passed,
    )


@dataclass(frozen=True, slots=True)
class IndependentVerificationResult:
    """Identity and decision of a successfully published verification receipt."""

    receipt_path: Path
    receipt_sha256: str
    decision_status: str
    document: Mapping[str, object]


def _write_receipt(
    path: str | Path,
    document: Mapping[str, object],
    *,
    protected_roots: Sequence[Path],
) -> Snapshot:
    target = _absolute(path)
    _require(target.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require(target.parent.is_dir(), "verification receipt parent must already exist")
    _reject_symlink_chain(target.parent, label="verification receipt parent")
    parent = target.parent.resolve(strict=True)
    target = parent / target.name
    for raw_root in protected_roots:
        root = raw_root.resolve(strict=True)
        _require(
            target != root and not target.is_relative_to(root),
            "verification receipt must be outside every protected input tree",
        )
    _require(not os.path.lexists(target), f"refusing to overwrite verification receipt: {target}")
    payload = _canonical_json_bytes(dict(document))
    descriptor: int | None = None
    staging: Path | None = None
    staged_identity: tuple[int, int] | None = None
    committed = False
    try:
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{target.name}.staging-",
            dir=parent,
        )
        staging = Path(staging_name)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "verification receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        staged_snapshot = _read_snapshot(staging, label="staged verification receipt")
        _require(
            staged_snapshot.payload == payload and staged_snapshot.mode == 0o444,
            "staged verification receipt changed",
        )
        staged_identity = (staged_snapshot.device, staged_snapshot.inode)
        try:
            os.link(staging, target, follow_symlinks=False)
        except FileExistsError as error:
            raise VerificationError(
                f"refusing to overwrite verification receipt: {target}"
            ) from error
        except OSError as error:
            raise VerificationError(f"cannot publish verification receipt: {target}") from error
        committed = True
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staging is not None:
            staging.unlink(missing_ok=True)
            if committed:
                directory_descriptor = os.open(
                    parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
    snapshot = _read_snapshot(target, label="independent verification receipt")
    _require(
        staged_identity is not None and (snapshot.device, snapshot.inode) == staged_identity,
        "published receipt identity differs from staged receipt",
    )
    _require(
        snapshot.payload == payload and snapshot.mode == 0o444,
        "published receipt changed",
    )
    return snapshot


def verify_native_diffusion_evaluation(
    *,
    contract_path: str | Path,
    accepted_corpus: str | Path,
    training_projection: str | Path,
    reference_fasta: str | Path,
    training_bundles: Mapping[str, str | Path],
    evaluation_bundle: str | Path,
    repository_root: str | Path,
    receipt_path: str | Path,
) -> IndependentVerificationResult:
    """Verify the complete frozen cohort and publish one immutable receipt.

    Model predictions are checked as lossless sufficient statistics bound into
    the same evaluation manifest as the verified checkpoint identities.  This
    verifier deliberately does not rerun the checkpoints, so the receipt says
    explicitly that prediction and native-proposal provenance are co-binding,
    not cryptographic proof of checkpoint execution.
    """

    _require(type(training_bundles) is dict, "training_bundles must be a plain dict")
    inputs = _verify_inputs(
        contract_path=contract_path,
        corpus_path=accepted_corpus,
        training_projection_path=training_projection,
        reference_fasta=reference_fasta,
    )
    contract = inputs.contract
    _require(
        _absolute(receipt_path).name == contract.artifacts.independent_receipt_file,
        "verification receipt filename differs from the contract",
    )
    labels = contract.artifacts.training_bundle_binding_labels
    _require(tuple(training_bundles) == labels, "training bundle mapping order differs")
    expected_seeds = (42, 42, 43, 44)
    audits: dict[str, TrainingBundleAudit] = {}
    for label, seed in zip(labels, expected_seeds, strict=True):
        audits[label] = _verify_training_bundle(
            training_bundles[label],
            label=label,
            expected_seed=seed,
            contract=contract,
            contract_payload=inputs.contract_snapshot.payload,
            repository_root=repository_root,
        )
    git_commits = {audit.git_commit for audit in audits.values()}
    _require(len(git_commits) == 1, "training bundles use different Git commits")
    git_commit = next(iter(git_commits))
    _verify_repository_state(repository_root, git_commit)
    evaluation = _verify_evaluation_bundle(
        evaluation_bundle,
        inputs=inputs,
        training_audits=audits,
    )

    primary = audits[labels[0]]
    twin = audits[labels[1]]
    twin_record = {
        "seed": 42,
        "manifest_byte_identical": primary.manifest_sha256 == twin.manifest_sha256,
        "checkpoint_file_byte_identical": (
            primary.checkpoint_file_sha256 == twin.checkpoint_file_sha256
        ),
        "checkpoint_logical_state_identical": (
            primary.checkpoint_logical_sha256 == twin.checkpoint_logical_sha256
        ),
        "environment_identity_identical": (
            primary.environment_identity == twin.environment_identity
        ),
        "all_bundle_artifacts_byte_identical": primary.file_hashes == twin.file_hashes,
    }
    _require(
        all(value is True for key, value in twin_record.items() if key != "seed"),
        "seed-42 twin equality failed",
    )
    manifest_artifacts = cast(dict[str, object], evaluation.manifest["artifacts"])
    receipt: dict[str, object] = {
        "schema_version": contract.artifacts.schema_version,
        "artifact": f"{contract.artifact}_independent_verification",
        "decision_status": evaluation.decision_status,
        "config_sha256": CONFIG_SHA256,
        "git_commit": git_commit,
        "input_hashes": {
            "contract.toml": inputs.contract_snapshot.sha256,
            "accepted_corpus.jsonl": inputs.corpus_snapshot.sha256,
            "training_projection.jsonl": inputs.projection_snapshot.sha256,
            "organizer_reference.fasta": inputs.reference_snapshot.sha256,
        },
        "training_bundle_hashes": [
            {
                "label": label,
                "seed": audits[label].seed,
                "manifest_sha256": audits[label].manifest_sha256,
                "checkpoint_file_sha256": audits[label].checkpoint_file_sha256,
                "checkpoint_logical_state_sha256": audits[label].checkpoint_logical_sha256,
                "checkpoint_contract_logical_sha256": _bound_checkpoint(
                    CONFIG_SHA256,
                    audits[label].checkpoint_logical_sha256,
                ),
                "artifacts": dict(audits[label].file_hashes),
            }
            for label in labels
        ],
        "evaluation_bundle_hashes": {
            "manifest_sha256": evaluation.manifest_sha256,
            "logical_sha256": evaluation.logical_sha256,
            "artifacts": manifest_artifacts,
        },
        "seed42_twin": twin_record,
        "validation": {
            "validation_sequences": contract.evaluation.expected_validation_sequences,
            "corruption_cases": contract.evaluation.expected_corruption_cases,
            "method_order": list(contract.artifacts.validation_token_stats_method_order),
            "method_summaries": [item.summary_record() for item in evaluation.metrics],
            "baseline_token_statistics_independently_refit": True,
            "model_token_statistics_reconstructed_without_checkpoint_reinference": True,
        },
        "gates": {
            "denoising": dict(evaluation.denoising_gate),
            "sampling": dict(evaluation.sampling_gate),
            "producer_evidence_passed": evaluation.producer_evidence_passed,
            "decision_status": evaluation.decision_status,
        },
        "checks": {
            "contract_and_input_hashes": True,
            "training_bundle_inventory_modes_and_hashes": True,
            "training_bundle_semantics_and_checkpoint_states": True,
            "seed42_exact_twin": True,
            "repository_commit_upstream_and_cleanliness": True,
            "evaluation_bundle_inventory_modes_and_hashes": True,
            "deterministic_npz_bytes_and_schemas": True,
            "corruption_ledger_rederived": True,
            "baseline_statistics_independently_refit": True,
            "validation_metrics_and_bootstrap_reconstructed": True,
            "candidate_ledgers_and_fasta_reconstructed": True,
            "count_control_proposals_independently_regenerated": True,
            "sampling_metrics_and_all_gates_reconstructed": True,
            "producer_documents_match_independent_reconstruction": True,
        },
        "limitations": [
            "checkpoint_reinference_is_not_part_of_this_verifier",
            "model_token_statistics_and_checkpoint_identities_are_cryptographically_co_bound_by_the_evaluation_manifest_but_checkpoint_origin_of_predictions_is_not_proven_without_reinference",
            "native_diffusion_proposals_are_checkpoint_bound_but_are_not_regenerated_from_the_checkpoint",
            "distinct_physical_node_execution_and_evaluation_runtime_attestation_require_the_separate_operational_receipt",
        ],
    }
    _require(
        tuple(receipt) == contract.artifacts.independent_receipt_fields,
        "independent receipt top-level schema differs from the contract",
    )

    for snapshot, label in (
        (inputs.contract_snapshot, "contract"),
        (inputs.corpus_snapshot, "accepted corpus"),
        (inputs.projection_snapshot, "training projection"),
        (inputs.reference_snapshot, "organizer reference"),
    ):
        _snapshot_unchanged(snapshot, label=label)
    for audit in audits.values():
        for snapshot in audit.snapshots:
            _snapshot_unchanged(
                snapshot, label=f"training bundle {audit.label}/{snapshot.path.name}"
            )
    for snapshot in evaluation.snapshots.values():
        _snapshot_unchanged(snapshot, label=f"evaluation bundle/{snapshot.path.name}")
    _verify_repository_state(repository_root, git_commit)

    protected = [
        _absolute(repository_root),
        *(_absolute(training_bundles[label]) for label in labels),
        evaluation.root,
    ]
    published = _write_receipt(receipt_path, receipt, protected_roots=protected)
    return IndependentVerificationResult(
        receipt_path=published.path,
        receipt_sha256=published.sha256,
        decision_status=evaluation.decision_status,
        document=receipt,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--accepted-corpus", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument(
        "--training-bundle",
        action="append",
        metavar="LABEL=PATH",
        required=True,
    )
    parser.add_argument("--evaluation-bundle", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _training_bundle_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, path = value.partition("=")
        _require(bool(separator and label and path), "--training-bundle must be LABEL=PATH")
        _require(label not in result, f"duplicate training bundle label: {label}")
        result[label] = Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = verify_native_diffusion_evaluation(
            contract_path=arguments.contract,
            accepted_corpus=arguments.accepted_corpus,
            training_projection=arguments.training_projection,
            reference_fasta=arguments.reference_fasta,
            training_bundles=_training_bundle_arguments(arguments.training_bundle),
            evaluation_bundle=arguments.evaluation_bundle,
            repository_root=arguments.repository_root,
            receipt_path=arguments.receipt,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"native diffusion evaluation verification failed: {error}", file=sys.stderr)
        return 1
    print(_canonical_json_bytes(dict(result.document)).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the cluster CLI
    raise SystemExit(main())

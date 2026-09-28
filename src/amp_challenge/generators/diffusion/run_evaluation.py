"""Production GPU driver for the frozen native categorical-diffusion v0 cohort.

The public boundary accepts paths only.  It does not accept models, callbacks,
devices, seeds, methods, metrics, or gate decisions.  Every scientific result
is reconstructed by :mod:`evaluation_bundle` from the shared corruption panel,
selected-token sufficient statistics, and raw proposals produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from .categorical import AbsorbingDiffusion, CosineMaskSchedule, PeptideVocabulary
from .contract import NativeDiffusionContract, load_unconditional_v0_contract
from .data import (
    DiffusionCorpusRow,
    LengthPrior,
    NativeDiffusionCorpus,
    TrainingDistribution,
    load_native_diffusion_corpus,
    load_training_projection,
)
from .evaluation import (
    CountBaselineSuite,
    ValidationCaseLedger,
    build_validation_case_ledger,
    fit_count_baselines,
    sample_count_generator_control_v0,
)
from .evaluation_bundle import (
    PRODUCER_CHECK_NAMES,
    EvaluationBundleResult,
    EvaluationProtocol,
    OrganizerReference,
    ProducerEvidenceChecks,
    ProposalBatch,
    TrainingBundleBinding,
    TrainingManifestRecord,
    ValidationCorruptions,
    ValidationTokenStatistics,
    load_organizer_reference,
    load_training_bundle_binding,
    protocol_from_contract,
    publish_unconditional_v0_evaluation_bundle,
)
from .inference import NativeV0LogitProvider, load_native_v0_logit_provider
from .sampling import canonical_length_plan, sample_unconditional_v0

_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_WIDTH = 50
_BATCH_SIZE = 256
_PROPOSALS_PER_SEED = 2_048
_SEEDS = (42, 43, 44)
_MAXIMUM_PEAK_GPU_MEMORY_GIB = 16.0
_MAXIMUM_PEAK_GPU_MEMORY_BYTES = 16 * 1024**3
_BASELINE_METHODS = (
    "component_weighted_unigram",
    "component_weighted_bidirectional_markov",
    "length_relative_position_frequency",
)
_PROPOSAL_METHODS = (
    "native_categorical_diffusion",
    "component_weighted_unigram",
    "component_weighted_forward_markov",
)
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


@dataclass(frozen=True, slots=True)
class EvaluationRunResult(EvaluationBundleResult):
    """Published semantic result plus this producer's operational GPU peak."""

    peak_gpu_memory_bytes: int

    def __post_init__(self) -> None:
        EvaluationBundleResult.__post_init__(self)
        if (
            type(self.peak_gpu_memory_bytes) is not int
            or not 0 <= self.peak_gpu_memory_bytes <= _MAXIMUM_PEAK_GPU_MEMORY_BYTES
        ):
            raise ValueError("evaluation peak GPU memory must lie within the frozen 16 GiB cap")


@dataclass(frozen=True, slots=True)
class _SelectedTokenLayout:
    case_ids: tuple[str, ...]
    case_offsets: NDArray[np.uint64]
    position: NDArray[np.uint8]
    target_token: NDArray[np.uint8]

    def __post_init__(self) -> None:
        cases = len(self.case_ids)
        if cases == 0 or len(set(self.case_ids)) != cases:
            raise ValueError("selected-token layout requires unique case IDs")
        offsets = _exact_array(
            self.case_offsets,
            dtype=np.dtype("<u8"),
            shape=(cases + 1,),
            label="case_offsets",
        )
        if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError("case offsets must start at zero and increase strictly")
        selected = int(offsets[-1])
        positions = _exact_array(
            self.position,
            dtype=np.dtype("|u1"),
            shape=(selected,),
            label="position",
        )
        targets = _exact_array(
            self.target_token,
            dtype=np.dtype("|u1"),
            shape=(selected,),
            label="target_token",
        )
        if np.any(positions >= _WIDTH) or np.any(targets >= 20):
            raise ValueError("selected-token layout contains an invalid position or target")
        object.__setattr__(self, "case_offsets", offsets)
        object.__setattr__(self, "position", positions)
        object.__setattr__(self, "target_token", targets)

    @property
    def selected_count(self) -> int:
        return int(self.case_offsets[-1])


@dataclass(slots=True)
class _TokenStatisticBuffer:
    target_log_probability: NDArray[np.float64]
    top1_confidence: NDArray[np.float64]
    top1_correct: NDArray[np.bool_]
    top3_correct: NDArray[np.bool_]
    multiclass_brier: NDArray[np.float64]
    filled: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _SelectedBatchStatistics:
    target_log_probability: NDArray[np.float64]
    top1_confidence: NDArray[np.float64]
    top1_correct: NDArray[np.bool_]
    top3_correct: NDArray[np.bool_]
    multiclass_brier: NDArray[np.float64]

    def __post_init__(self) -> None:
        size = len(self.target_log_probability)
        values = (
            ("target_log_probability", self.target_log_probability, np.dtype("<f8")),
            ("top1_confidence", self.top1_confidence, np.dtype("<f8")),
            ("top1_correct", self.top1_correct, np.dtype("|b1")),
            ("top3_correct", self.top3_correct, np.dtype("|b1")),
            ("multiclass_brier", self.multiclass_brier, np.dtype("<f8")),
        )
        for name, value, dtype in values:
            normalized = _exact_array(value, dtype=dtype, shape=(size,), label=name)
            object.__setattr__(self, name, normalized)
        if size == 0:
            raise ValueError("selected batch statistics cannot be empty")
        if np.any(self.target_log_probability > 1e-15):
            raise ValueError("target log probabilities cannot exceed log(1)")
        if np.any((self.top1_confidence < 0.0) | (self.top1_confidence > 1.0)):
            raise ValueError("top-1 confidence must lie in [0, 1]")
        if np.any(self.top1_correct & ~self.top3_correct):
            raise ValueError("top-3 correctness cannot exclude a top-1 hit")
        if np.any((self.multiclass_brier < 0.0) | (self.multiclass_brier > 2.0 + 1e-12)):
            raise ValueError("multiclass Brier values must lie in [0, 2]")


def _exact_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    label: str,
) -> NDArray[np.generic]:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    array = cast(np.ndarray, value)
    if array.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype.str}")
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"path ancestor does not exist: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"path traverses a symbolic link: {current}")
        if current.parent == current:
            return
        current = current.parent


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = source.stat(follow_symlinks=False)
    fingerprints = {
        (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            stat.S_IMODE(item.st_mode),
        )
        for item in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return payload


def _git(repository: Path, *arguments: str) -> bytes:
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
        completed = subprocess.run(
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
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Git command failed: git {' '.join(arguments)}") from error
    return completed.stdout


def _validate_deterministic_environment_before_cuda(
    contract: NativeDiffusionContract,
) -> None:
    """Reject allocator/cuBLAS drift without making any CUDA API call."""

    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
    if contract.compute.maximum_peak_gpu_memory_gib != _MAXIMUM_PEAK_GPU_MEMORY_GIB:
        raise ValueError("production evaluation requires the frozen 16 GiB GPU-memory cap")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract.determinism.cublas_workspace_config:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exported before evaluation CUDA inspection"
        )
    if os.environ.get("PYTORCH_ALLOC_CONF") != contract.determinism.pytorch_allocator:
        raise RuntimeError("the PyTorch allocator must be pinned before evaluation CUDA inspection")
    legacy_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_allocator not in (None, contract.determinism.pytorch_allocator):
        raise RuntimeError("legacy PyTorch allocator configuration conflicts with native v0")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"):
        raise RuntimeError("native v0 prohibits disabling CUDA memory caching")


def _reset_peak_gpu_memory_observation(contract: NativeDiffusionContract) -> None:
    """Start one integrated producer peak ledger before any model operation."""

    _validate_deterministic_environment_before_cuda(contract)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)


def _synchronized_peak_gpu_memory_bytes() -> int:
    """Synchronize all generated work before reading the integrated CUDA peak."""

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    if type(peak) is not int or peak < 0:
        raise RuntimeError("CUDA returned an invalid peak-memory observation")
    return peak


def _publish_post_generation_evidence(
    contract: NativeDiffusionContract,
    *,
    organizer_reference_path: Path,
    protocol: EvaluationProtocol,
    proposal_batches: tuple[ProposalBatch, ...],
    publisher: Callable[[OrganizerReference], EvaluationBundleResult],
) -> EvaluationRunResult:
    """Gate the GPU peak, then admit the reference, then permit publication."""

    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
    if not callable(publisher):
        raise TypeError("publisher must be callable")
    _validate_complete_proposal_grid(protocol, proposal_batches)
    if contract.compute.maximum_peak_gpu_memory_gib != _MAXIMUM_PEAK_GPU_MEMORY_GIB:
        raise ValueError("production evaluation requires the frozen 16 GiB GPU-memory cap")
    peak = _synchronized_peak_gpu_memory_bytes()
    if peak > _MAXIMUM_PEAK_GPU_MEMORY_BYTES:
        raise RuntimeError(
            "evaluation exceeded the preregistered 16 GiB peak GPU-memory cap before publication"
        )
    organizer_reference = load_organizer_reference(
        organizer_reference_path,
        expected_sha256=protocol.organizer_reference_sha256,
        expected_records=protocol.organizer_reference_records,
    )
    published = publisher(organizer_reference)
    if type(published) is not EvaluationBundleResult:
        raise TypeError("evaluation publisher returned an invalid result")
    return EvaluationRunResult(
        output_dir=published.output_dir,
        decision_status=published.decision_status,
        manifest_sha256=published.manifest_sha256,
        logical_sha256=published.logical_sha256,
        training_bundle_sha256=published.training_bundle_sha256,
        peak_gpu_memory_bytes=peak,
    )


def _verify_repository_snapshot(
    repository_root: str | Path,
    *,
    expected_git_commit: str,
) -> Path:
    if (
        type(expected_git_commit) is not str
        or _GIT_COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise ValueError("expected_git_commit must be a lowercase forty-character object ID")
    overrides = tuple(
        sorted(
            name
            for name in os.environ
            if name in _GIT_REPOSITORY_ENVIRONMENT or name.startswith("GIT_CONFIG_")
        )
    )
    if overrides:
        raise ValueError(f"Git repository-selection environment is forbidden: {overrides}")
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    _reject_symlink_chain(repository)
    if not repository.is_dir():
        raise ValueError("repository_root must be a real directory")
    top_level = Path(
        os.path.abspath(_git(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip())
    )
    if top_level != repository:
        raise ValueError("repository_root must be the exact Git worktree top level")
    if _git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("repository-local Git replacement refs are forbidden")
    observed = {
        "HEAD": _git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip(),
        "upstream": _git(repository, "rev-parse", "--verify", "@{upstream}^{commit}")
        .decode("ascii")
        .strip(),
        "origin/main": _git(
            repository,
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main^{commit}",
        )
        .decode("ascii")
        .strip(),
    }
    if set(observed.values()) != {expected_git_commit}:
        raise ValueError(
            "repository HEAD, upstream, and cached origin/main must equal expected_git_commit"
        )
    if _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError("production evaluation requires a clean repository worktree")
    return repository


def _validated_output_path(
    output_dir: str | Path,
    *,
    contract: NativeDiffusionContract,
    protected_paths: Sequence[str | Path],
) -> Path:
    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
    scratch_variable = contract.compute.scratch_root_env
    scratch_raw = os.environ.get(scratch_variable)
    if not scratch_raw:
        raise ValueError(f"required scratch environment variable is unset: {scratch_variable}")
    scratch = Path(os.path.abspath(scratch_raw))
    if not Path(scratch_raw).is_absolute():
        raise ValueError(f"{scratch_variable} must contain an absolute path")
    _reject_symlink_chain(scratch)
    if not stat.S_ISDIR(scratch.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"{scratch_variable} must name a real directory")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if output.name in {"", ".", ".."}:
        raise ValueError("evaluation output must have a concrete run name")
    _reject_symlink_chain(output.parent)
    if not stat.S_ISDIR(output.parent.stat(follow_symlinks=False).st_mode):
        raise ValueError("evaluation output parent must be a real directory")
    base = scratch / contract.compute.run_subdir
    if output == base or base not in output.parents:
        raise ValueError("evaluation output must be a named run below the contract scratch root")
    for raw in protected_paths:
        protected = Path(os.path.abspath(os.fspath(raw)))
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("evaluation output overlaps a protected input or repository path")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite or resume evaluation output: {output}")
    return output


def _materialize_validation_corruptions(
    rows: Sequence[DiffusionCorpusRow],
    ledger: ValidationCaseLedger,
) -> ValidationCorruptions:
    """Materialize one stateless width-50 panel; private for small CPU tests."""

    values = tuple(rows)
    if not values or any(type(row) is not DiffusionCorpusRow for row in values):
        raise TypeError("validation rows must be DiffusionCorpusRow values")
    if type(ledger) is not ValidationCaseLedger:
        raise TypeError("ledger must be a ValidationCaseLedger")
    row_ids = tuple(row.sequence_id for row in values)
    if row_ids != tuple(sorted(row_ids)) or len(set(row_ids)) != len(row_ids):
        raise ValueError("validation rows must be uniquely ordered by sequence ID")
    row_lookup = {row.sequence_id: index for index, row in enumerate(values)}
    if set(row_lookup) != {case.sequence_id for case in ledger.cases}:
        raise ValueError("validation rows do not exactly cover the corruption ledger")
    vocabulary = PeptideVocabulary()
    encoded = vocabulary.encode([row.sequence for row in values], max_length=_WIDTH)
    clean = np.asarray(encoded.tokens, dtype=np.uint8, order="C")
    attention = np.asarray(encoded.attention_mask, dtype=np.bool_, order="C")
    corrupted = np.empty((len(ledger.cases), _WIDTH), dtype=np.uint8)
    selected = np.empty((len(ledger.cases), _WIDTH), dtype=np.bool_)
    diffusion = AbsorbingDiffusion(vocabulary, CosineMaskSchedule(offset=0.008))
    for start in range(0, len(ledger.cases), _BATCH_SIZE):
        cases = ledger.cases[start : start + _BATCH_SIZE]
        indices = np.asarray([row_lookup[case.sequence_id] for case in cases], dtype=np.int64)
        batch_corrupted, batch_selected = diffusion.corrupt_fixed_count(
            encoded.tokens[indices],
            encoded.attention_mask[indices],
            np.asarray([case.level for case in cases], dtype=np.int64),
            total_levels=ledger.levels,
            row_seeds=tuple(case.row_seed for case in cases),
        )
        stop = start + len(cases)
        corrupted[start:stop] = batch_corrupted.astype(np.uint8)
        selected[start:stop] = batch_selected
        if not np.array_equal(
            np.sum(batch_selected, axis=1, dtype=np.int64),
            np.asarray([case.mask_count for case in cases], dtype=np.int64),
        ):
            raise RuntimeError("materialized corruption mask counts differ from the ledger")
    return ValidationCorruptions(
        rows=values,
        ledger=ledger,
        clean_tokens=clean,
        attention_mask=attention,
        corrupted_tokens=corrupted,
        selected_mask=selected,
    )


def _selected_token_layout(corruptions: ValidationCorruptions) -> _SelectedTokenLayout:
    """Flatten cases then ascending positions without a lossy aggregate."""

    if type(corruptions) is not ValidationCorruptions:
        raise TypeError("corruptions must be ValidationCorruptions")
    counts = np.sum(corruptions.selected_mask, axis=1, dtype=np.uint64)
    if np.any(counts == 0):
        raise ValueError("every corruption case must select at least one token")
    offsets = np.zeros(len(counts) + 1, dtype="<u8")
    np.cumsum(counts, dtype=np.uint64, out=offsets[1:])
    positions = np.empty(int(offsets[-1]), dtype="|u1")
    targets = np.empty(int(offsets[-1]), dtype="|u1")
    row_lookup = {row.sequence_id: index for index, row in enumerate(corruptions.rows)}
    for case_index, case in enumerate(corruptions.ledger.cases):
        start = int(offsets[case_index])
        stop = int(offsets[case_index + 1])
        chosen = np.flatnonzero(corruptions.selected_mask[case_index]).astype(np.uint8)
        if len(chosen) != stop - start:
            raise RuntimeError("selected-token layout count drifted from its offsets")
        positions[start:stop] = chosen
        targets[start:stop] = corruptions.clean_tokens[row_lookup[case.sequence_id], chosen]
    return _SelectedTokenLayout(
        case_ids=tuple(case.case_id for case in corruptions.ledger.cases),
        case_offsets=offsets,
        position=positions,
        target_token=targets,
    )


def _probabilities_from_native_logits(
    logits: object,
    *,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the frozen float64 log-softmax to strict provider FP32 output."""

    if type(logits) is not np.ndarray:
        raise TypeError("native provider logits must be an exact numpy.ndarray")
    raw = cast(np.ndarray, logits)
    expected_shape = (batch_size, _WIDTH, 20)
    if raw.dtype != np.dtype(np.float32):
        raise TypeError("native provider logits must have dtype float32")
    if raw.shape != expected_shape:
        raise ValueError(f"native provider logits must have shape {expected_shape}")
    if np.any(~np.isfinite(raw)):
        raise ValueError("native provider logits must be finite")
    values = raw.astype(np.float64, copy=True)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    totals = np.sum(exponentials, axis=-1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("native logits produced invalid categorical mass")
    log_probabilities = shifted - np.log(totals)
    probabilities = np.exp(log_probabilities)
    return (
        cast(NDArray[np.float64], probabilities),
        cast(NDArray[np.float64], log_probabilities),
    )


def _selected_batch_statistics(
    probabilities: object,
    log_probabilities: object,
    clean_tokens: object,
    selected_mask: object,
) -> _SelectedBatchStatistics:
    """Extract sufficient statistics with stable residue-index tie breaking."""

    if type(probabilities) is not np.ndarray or type(log_probabilities) is not np.ndarray:
        raise TypeError("probability inputs must be exact numpy.ndarray values")
    probability_values = cast(np.ndarray, probabilities)
    log_values = cast(np.ndarray, log_probabilities)
    if probability_values.dtype != np.dtype("<f8") or log_values.dtype != np.dtype("<f8"):
        raise TypeError("probability inputs must have little-endian float64 dtype")
    if probability_values.ndim != 3 or probability_values.shape[-2:] != (_WIDTH, 20):
        raise ValueError("probabilities must have shape [batch, 50, 20]")
    if log_values.shape != probability_values.shape:
        raise ValueError("probabilities and log probabilities must have identical shapes")
    batch = len(probability_values)
    if type(clean_tokens) is not np.ndarray or type(selected_mask) is not np.ndarray:
        raise TypeError("clean tokens and selected mask must be exact numpy.ndarray values")
    clean = cast(np.ndarray, clean_tokens)
    selected = cast(np.ndarray, selected_mask)
    if clean.dtype != np.dtype("|u1") or clean.shape != (batch, _WIDTH):
        raise TypeError("clean_tokens must be a [batch, 50] uint8 array")
    if selected.dtype != np.dtype("|b1") or selected.shape != clean.shape:
        raise TypeError("selected_mask must be a matching Boolean array")
    if np.any(np.sum(selected, axis=1, dtype=np.int64) <= 0):
        raise ValueError("every batch row must select at least one token")
    if np.any(~np.isfinite(probability_values)) or np.any(probability_values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if np.any(~np.isfinite(log_values)) or np.any(log_values > 1e-15):
        raise ValueError("log probabilities must be finite and no greater than zero")
    if not np.allclose(
        np.sum(probability_values, axis=-1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("probabilities must sum to one")
    selected_probabilities = probability_values[selected]
    selected_logs = log_values[selected]
    targets = clean[selected].astype(np.int64)
    if np.any(targets >= 20):
        raise ValueError("selected clean targets must be residue tokens")
    rows = np.arange(len(targets), dtype=np.int64)
    # Log probabilities preserve the exact logit ordering even when converting
    # a very small probability to float64 underflows to zero.  Stable sorting
    # retains residue-index order only for genuine equal-logit ties.
    rankings = np.argsort(-selected_logs, axis=1, kind="stable")
    predictions = rankings[:, 0]
    target_probability = selected_probabilities[rows, targets]
    target_log_probability = selected_logs[rows, targets]
    top1_confidence = selected_probabilities[rows, predictions]
    top1_correct = predictions == targets
    top3_correct = np.any(rankings[:, :3] == targets[:, None], axis=1)
    brier = np.sum(np.square(selected_probabilities), axis=1) - 2.0 * target_probability + 1.0
    return _SelectedBatchStatistics(
        target_log_probability=np.asarray(target_log_probability, dtype="<f8"),
        top1_confidence=np.asarray(top1_confidence, dtype="<f8"),
        top1_correct=np.asarray(top1_correct, dtype="|b1"),
        top3_correct=np.asarray(top3_correct, dtype="|b1"),
        multiclass_brier=np.asarray(brier, dtype="<f8"),
    )


def _allocate_statistic_buffer(methods: int, selected: int) -> _TokenStatisticBuffer:
    if type(methods) is not int or methods <= 0 or type(selected) is not int or selected <= 0:
        raise ValueError("statistic buffer dimensions must be positive integers")
    shape = (methods, selected)
    return _TokenStatisticBuffer(
        target_log_probability=np.full(shape, np.nan, dtype="<f8"),
        top1_confidence=np.full(shape, np.nan, dtype="<f8"),
        top1_correct=np.zeros(shape, dtype="|b1"),
        top3_correct=np.zeros(shape, dtype="|b1"),
        multiclass_brier=np.full(shape, np.nan, dtype="<f8"),
        filled=np.zeros(shape, dtype="|b1"),
    )


def _store_batch_statistics(
    buffer: _TokenStatisticBuffer,
    *,
    method_index: int,
    start: int,
    stop: int,
    statistics: _SelectedBatchStatistics,
) -> None:
    if type(buffer) is not _TokenStatisticBuffer:
        raise TypeError("buffer must be a _TokenStatisticBuffer")
    if type(method_index) is not int or not 0 <= method_index < len(buffer.filled):
        raise ValueError("method index is outside the statistic buffer")
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= buffer.filled.shape[1]
    ):
        raise ValueError("selected-token slice is invalid")
    if len(statistics.target_log_probability) != stop - start:
        raise ValueError("batch statistics do not fill the declared selected-token slice")
    if np.any(buffer.filled[method_index, start:stop]):
        raise RuntimeError("selected-token statistics cannot be overwritten")
    for name in (
        "target_log_probability",
        "top1_confidence",
        "top1_correct",
        "top3_correct",
        "multiclass_brier",
    ):
        getattr(buffer, name)[method_index, start:stop] = getattr(statistics, name)
    buffer.filled[method_index, start:stop] = True


def _case_row_indices(corruptions: ValidationCorruptions) -> NDArray[np.int64]:
    lookup = {row.sequence_id: index for index, row in enumerate(corruptions.rows)}
    result = np.asarray(
        [lookup[case.sequence_id] for case in corruptions.ledger.cases],
        dtype=np.int64,
    )
    result.flags.writeable = False
    return result


def _score_baselines(
    corruptions: ValidationCorruptions,
    suite: CountBaselineSuite,
    layout: _SelectedTokenLayout,
    buffer: _TokenStatisticBuffer,
) -> None:
    if type(suite) is not CountBaselineSuite:
        raise TypeError("suite must be a CountBaselineSuite")
    row_indices = _case_row_indices(corruptions)
    lengths = np.asarray([len(row.sequence) for row in corruptions.rows], dtype=np.int64)
    for method_index, method in enumerate(_BASELINE_METHODS):
        for case_start in range(0, len(corruptions.ledger.cases), _BATCH_SIZE):
            case_stop = min(case_start + _BATCH_SIZE, len(corruptions.ledger.cases))
            source = row_indices[case_start:case_stop]
            probabilities = suite.probabilities(
                method,
                corruptions.corrupted_tokens[case_start:case_stop].astype(np.int64),
                corruptions.attention_mask[source],
                lengths[source],
            )
            statistics = _selected_batch_statistics(
                probabilities,
                np.log(probabilities),
                corruptions.clean_tokens[source],
                corruptions.selected_mask[case_start:case_stop],
            )
            _store_batch_statistics(
                buffer,
                method_index=method_index,
                start=int(layout.case_offsets[case_start]),
                stop=int(layout.case_offsets[case_stop]),
                statistics=statistics,
            )


def _native_provider_inputs(
    corruptions: ValidationCorruptions,
    row_indices: NDArray[np.int64],
    *,
    start: int,
    stop: int,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]]:
    source = row_indices[start:stop]
    values = (
        np.asarray(corruptions.corrupted_tokens[start:stop], dtype=np.int64, order="C"),
        np.asarray(corruptions.attention_mask[source], dtype=np.bool_, order="C"),
        np.asarray(
            [case.level for case in corruptions.ledger.cases[start:stop]],
            dtype=np.int64,
        ),
        np.asarray([len(corruptions.rows[index].sequence) for index in source], dtype=np.int64),
    )
    for value in values:
        value.flags.writeable = False
    return values


def _verify_provider_repeatability(
    provider: NativeV0LogitProvider,
    corruptions: ValidationCorruptions,
) -> None:
    if type(provider) is not NativeV0LogitProvider:
        raise TypeError("native evaluation requires an exact sealed NativeV0LogitProvider")
    row_indices = _case_row_indices(corruptions)
    stop = min(_BATCH_SIZE, len(corruptions.ledger.cases))
    inputs = _native_provider_inputs(corruptions, row_indices, start=0, stop=stop)
    first = provider(*inputs)
    second = provider(*inputs)
    if any(
        item.dtype != np.dtype(np.float32) or item.shape != (stop, _WIDTH, 20)
        for item in (first, second)
    ):
        raise RuntimeError("sealed provider repeatability probe returned invalid logits")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise RuntimeError("sealed provider repeatability probe returned non-finite logits")
    if not np.array_equal(first, second):
        raise RuntimeError("sealed provider is not byte-repeatable on an identical fixed batch")


def _score_native_provider(
    provider: NativeV0LogitProvider,
    corruptions: ValidationCorruptions,
    layout: _SelectedTokenLayout,
    buffer: _TokenStatisticBuffer,
    *,
    method_index: int,
) -> None:
    if type(provider) is not NativeV0LogitProvider:
        raise TypeError("native evaluation requires an exact sealed NativeV0LogitProvider")
    row_indices = _case_row_indices(corruptions)
    for case_start in range(0, len(corruptions.ledger.cases), _BATCH_SIZE):
        case_stop = min(case_start + _BATCH_SIZE, len(corruptions.ledger.cases))
        inputs = _native_provider_inputs(
            corruptions,
            row_indices,
            start=case_start,
            stop=case_stop,
        )
        logits = provider(*inputs)
        probabilities, log_probabilities = _probabilities_from_native_logits(
            logits,
            batch_size=case_stop - case_start,
        )
        statistics = _selected_batch_statistics(
            probabilities,
            log_probabilities,
            corruptions.clean_tokens[row_indices[case_start:case_stop]],
            corruptions.selected_mask[case_start:case_stop],
        )
        _store_batch_statistics(
            buffer,
            method_index=method_index,
            start=int(layout.case_offsets[case_start]),
            stop=int(layout.case_offsets[case_stop]),
            statistics=statistics,
        )


def _finalize_token_statistics(
    layout: _SelectedTokenLayout,
    buffer: _TokenStatisticBuffer,
    *,
    methods: tuple[str, ...],
) -> ValidationTokenStatistics:
    if type(methods) is not tuple or len(methods) != buffer.filled.shape[0]:
        raise ValueError("method axis differs from the statistic buffer")
    if not np.all(buffer.filled):
        raise RuntimeError("selected-token statistic matrix is incomplete")
    for name in (
        "target_log_probability",
        "top1_confidence",
        "multiclass_brier",
    ):
        if np.any(~np.isfinite(getattr(buffer, name))):
            raise FloatingPointError(f"token statistic {name} contains a non-finite value")
    return ValidationTokenStatistics(
        case_ids=layout.case_ids,
        case_offsets=layout.case_offsets.copy(),
        position=layout.position.copy(),
        target_token=layout.target_token.copy(),
        methods=methods,
        target_log_probability=buffer.target_log_probability.copy(),
        top1_confidence=buffer.top1_confidence.copy(),
        top1_correct=buffer.top1_correct.copy(),
        top3_correct=buffer.top3_correct.copy(),
        multiclass_brier=buffer.multiclass_brier.copy(),
    )


def _draw_shared_length_plan(
    prior: LengthPrior,
    *,
    seed: int,
    count: int,
    require_locked_count: bool,
) -> tuple[tuple[int, ...], str]:
    """Draw and replay one train-only length plan; private count aids CPU tests."""

    if type(prior) is not LengthPrior:
        raise TypeError("length prior must be a LengthPrior")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("length-plan seed must be an unsigned 64-bit integer")
    if type(count) is not int or count <= 0:
        raise ValueError("length-plan count must be a positive integer")
    if type(require_locked_count) is not bool:
        raise TypeError("require_locked_count must be Boolean")
    lengths = prior.draw(
        root_seed=seed,
        draw_start=0,
        draw_count=count,
        namespace="proposal",
    )
    replay = prior.draw(
        root_seed=seed,
        draw_start=0,
        draw_count=count,
        namespace="proposal",
    )
    if lengths != replay:
        raise RuntimeError("train-only length prior did not replay exactly")
    plan, digest = canonical_length_plan(
        lengths,
        require_locked_count=require_locked_count,
    )
    if tuple(ordinal for ordinal, _ in plan) != tuple(range(count)):
        raise RuntimeError("length-plan ordinals differ from the frozen zero-based order")
    if tuple(length for _, length in plan) != lengths:
        raise RuntimeError("canonical length plan changed the ordered length draws")
    return lengths, digest


def _verify_seed42_twin_binding(
    binding: TrainingBundleBinding,
    protocol: EvaluationProtocol,
) -> None:
    if type(binding) is not TrainingBundleBinding:
        raise TypeError("binding must be a TrainingBundleBinding")
    if type(protocol) is not EvaluationProtocol:
        raise TypeError("protocol must be an EvaluationProtocol")
    if tuple(record.label for record in binding.records) != protocol.training_bundle_labels:
        raise ValueError("training bundle labels differ from the frozen protocol")
    primary, twin = binding.records[:2]
    if primary.label != "seed-42-primary" or twin.label != "seed-42-twin":
        raise ValueError("the first two training bundles must be the seed-42 twins")
    if primary.seed != 42 or twin.seed != 42:
        raise ValueError("seed-42 twin bindings declare the wrong seed")
    attributes = (
        "manifest_sha256",
        "checkpoint_file_sha256",
        "checkpoint_model_logical_sha256",
        "checkpoint_bound_logical_sha256",
    )
    if any(getattr(primary, name) != getattr(twin, name) for name in attributes):
        raise ValueError("seed-42 twin manifest and checkpoint bindings are not exact")
    if tuple(sorted(binding.primary_by_seed)) != protocol.seeds:
        raise ValueError("training bundle binding does not contain seeds 42, 43, and 44")


def _verify_provider_binding(
    provider: NativeV0LogitProvider,
    record: TrainingManifestRecord,
    *,
    protocol: EvaluationProtocol,
    expected_git_commit: str,
) -> None:
    if type(provider) is not NativeV0LogitProvider:
        raise TypeError("provider must be an exact sealed NativeV0LogitProvider")
    if type(record) is not TrainingManifestRecord:
        raise TypeError("record must be a TrainingManifestRecord")
    observed = {
        "seed": provider.seed,
        "git_commit": provider.git_commit,
        "config_sha256": provider.config_sha256,
        "checkpoint_file_sha256": provider.checkpoint_file_sha256,
        "checkpoint_model_logical_sha256": provider.checkpoint_model_logical_sha256,
        "checkpoint_bound_logical_sha256": provider.checkpoint_logical_sha256,
    }
    expected = {
        "seed": record.seed,
        "git_commit": expected_git_commit,
        "config_sha256": protocol.config_sha256,
        "checkpoint_file_sha256": record.checkpoint_file_sha256,
        "checkpoint_model_logical_sha256": record.checkpoint_model_logical_sha256,
        "checkpoint_bound_logical_sha256": record.checkpoint_bound_logical_sha256,
    }
    if observed != expected:
        raise ValueError("sealed provider identity differs from its training-bundle binding")


def _validate_proposal_batches(
    batches: Mapping[tuple[str, int], ProposalBatch],
    *,
    protocol: EvaluationProtocol,
    binding: TrainingBundleBinding,
    training: TrainingDistribution,
    suite: CountBaselineSuite,
) -> tuple[ProposalBatch, ...]:
    if type(batches) is not dict:
        raise TypeError("proposal batches must be a plain dict")
    if type(binding) is not TrainingBundleBinding:
        raise TypeError("binding must be an exact TrainingBundleBinding")
    if type(training) is not TrainingDistribution:
        raise TypeError("training must be an exact TrainingDistribution")
    if len(training.rows) != protocol.training_sequences:
        raise ValueError("training projection census differs from the frozen protocol")
    if type(suite) is not CountBaselineSuite:
        raise TypeError("suite must be an exact CountBaselineSuite")
    if suite.training_projection_sha256 != protocol.training_projection_sha256:
        raise ValueError("count-control suite uses a different training projection")
    recomputed_suite = fit_count_baselines(
        training,
        require_locked_census=False,
        effective_count_scale=protocol.training_sequences,
    )
    if suite.logical_sha256 != recomputed_suite.logical_sha256:
        raise ValueError("count-control suite differs from the frozen training-only fit")
    expected_keys = tuple(
        (method, seed) for method in protocol.proposal_methods for seed in protocol.seeds
    )
    if set(batches) != set(expected_keys):
        raise ValueError("proposal batches do not cover the frozen method/seed grid")
    ordered = tuple(batches[key] for key in expected_keys)
    primary = binding.primary_by_seed
    if binding.config_sha256 != protocol.config_sha256 or tuple(sorted(primary)) != protocol.seeds:
        raise ValueError("training binding differs from the frozen evaluation cohort")
    for seed in protocol.seeds:
        expected_lengths = training.length_prior.draw(
            root_seed=seed,
            draw_start=0,
            draw_count=protocol.raw_proposals_per_seed,
            namespace="proposal",
        )
        expected_plan = tuple(enumerate(expected_lengths))
        _, expected_plan_sha256 = canonical_length_plan(
            expected_lengths,
            require_locked_count=False,
        )
        for method in protocol.proposal_methods:
            batch = batches[(method, seed)]
            if type(batch) is not ProposalBatch or (batch.method, batch.seed) != (method, seed):
                raise ValueError("proposal batch identity differs from the frozen method/seed axis")
            if len(batch.records) != protocol.raw_proposals_per_seed:
                raise ValueError("proposal batch does not contain the frozen record census")
            if (
                batch.length_plan_sha256 != expected_plan_sha256
                or tuple((record.ordinal, record.length) for record in batch.records)
                != expected_plan
            ):
                raise ValueError(
                    "proposal batch differs from the exact train-derived ordinal/length plan"
                )
            if any(
                record.config_sha256 != protocol.config_sha256
                or record.training_projection_sha256 != protocol.training_projection_sha256
                for record in batch.records
            ):
                raise ValueError("proposal records do not bind the frozen contract inputs")
            if method == _PROPOSAL_METHODS[0]:
                if any(
                    record.generator_binding_kind != "checkpoint_contract_logical_sha256"
                    or record.generator_binding_sha256
                    != primary[seed].checkpoint_bound_logical_sha256
                    for record in batch.records
                ):
                    raise ValueError("native proposal batch is not bound to its seed checkpoint")
                continue
            if any(
                record.generator_binding_kind != "count_control_logical_sha256"
                or record.generator_binding_sha256 != suite.logical_sha256
                for record in batch.records
            ):
                raise ValueError("count-control proposal batch has the wrong logical binding")
            expected_control = sample_count_generator_control_v0(
                suite,
                expected_lengths,
                method=method,
                seed=seed,
                ordinals=tuple(range(protocol.raw_proposals_per_seed)),
                batch_size=_BATCH_SIZE,
                require_locked_count=False,
            )
            observed_candidates = tuple(
                (record.ordinal, record.sequence_id, record.sequence, record.length)
                for record in batch.records
            )
            expected_candidates = tuple(
                (item.ordinal, item.sequence_id, item.sequence, item.length)
                for item in expected_control.candidates
            )
            if observed_candidates != expected_candidates:
                raise ValueError("count-control proposal batch differs from stateless regeneration")
    return ordered


def _validate_complete_proposal_grid(
    protocol: EvaluationProtocol,
    proposal_batches: tuple[ProposalBatch, ...],
) -> None:
    if type(protocol) is not EvaluationProtocol:
        raise TypeError("protocol must be an EvaluationProtocol")
    if type(proposal_batches) is not tuple or any(
        type(batch) is not ProposalBatch for batch in proposal_batches
    ):
        raise TypeError("proposal batches must be a sealed tuple of ProposalBatch values")
    expected_axis = tuple(
        (method, seed) for method in protocol.proposal_methods for seed in protocol.seeds
    )
    observed_axis = tuple((batch.method, batch.seed) for batch in proposal_batches)
    if observed_axis != expected_axis or any(
        len(batch.records) != protocol.raw_proposals_per_seed for batch in proposal_batches
    ):
        raise ValueError(
            "organizer reference cannot be opened before the complete proposal grid is sealed"
        )


def _load_post_generation_organizer_reference(
    organizer_reference_path: Path,
    *,
    protocol: EvaluationProtocol,
    proposal_batches: tuple[ProposalBatch, ...],
) -> OrganizerReference:
    """Open the compliance FASTA only after the complete proposal grid exists."""

    _validate_complete_proposal_grid(protocol, proposal_batches)
    return load_organizer_reference(
        organizer_reference_path,
        expected_sha256=protocol.organizer_reference_sha256,
        expected_records=protocol.organizer_reference_records,
    )


def _producer_checks_from_completed(completed: set[str]) -> ProducerEvidenceChecks:
    """Create true evidence flags only after every named validation completed."""

    if type(completed) is not set or any(type(name) is not str for name in completed):
        raise TypeError("completed checks must be a plain set of names")
    required = set(PRODUCER_CHECK_NAMES)
    if completed != required:
        missing = sorted(required - completed)
        extra = sorted(completed - required)
        raise ValueError(
            f"producer checks are incomplete or unknown: missing={missing}, extra={extra}"
        )
    return ProducerEvidenceChecks(**{name: True for name in PRODUCER_CHECK_NAMES})


def _load_inputs(
    *,
    contract_path: Path,
    accepted_corpus_path: Path,
    training_projection_path: Path,
    bundle_directories: dict[str, Path],
    expected_git_commit: str,
) -> tuple[
    NativeDiffusionContract,
    bytes,
    EvaluationProtocol,
    NativeDiffusionCorpus,
    TrainingDistribution,
    TrainingBundleBinding,
]:
    contract = load_unconditional_v0_contract(contract_path)
    contract_payload = _read_regular_bytes(contract_path, label="diffusion contract")
    protocol = protocol_from_contract(contract)
    if hashlib.sha256(contract_payload).hexdigest() != protocol.config_sha256:
        raise ValueError("contract changed after strict parsing")
    corpus = load_native_diffusion_corpus(
        accepted_corpus_path,
        expected_sha256=protocol.corpus_sha256,
    )
    training = load_training_projection(
        training_projection_path,
        expected_sha256=protocol.training_projection_sha256,
        expected_rows=protocol.training_sequences,
    )
    expected_projection = tuple(
        (row.sequence_id, row.sequence, row.sampling_weight) for row in corpus.train_rows
    )
    observed_projection = tuple(
        (row.sequence_id, row.sequence, row.sampling_weight) for row in training.rows
    )
    if observed_projection != expected_projection:
        raise ValueError("training projection is not the exact train-only corpus projection")
    binding = load_training_bundle_binding(
        bundle_directories,
        protocol=protocol,
        contract_payload=contract_payload,
        git_commit=expected_git_commit,
    )
    _verify_seed42_twin_binding(binding, protocol)
    return contract, contract_payload, protocol, corpus, training, binding


def run_unconditional_v0_evaluation(
    *,
    output_dir: str | Path,
    contract_path: str | Path,
    accepted_corpus_path: str | Path,
    training_projection_path: str | Path,
    organizer_reference_path: str | Path,
    seed42_primary_bundle: str | Path,
    seed42_twin_bundle: str | Path,
    seed43_bundle: str | Path,
    seed44_bundle: str | Path,
    repository_root: str | Path,
    expected_git_commit: str,
) -> EvaluationRunResult:
    """Execute and immutably publish the exact production v0 cohort evaluation."""

    repository = _verify_repository_snapshot(
        repository_root,
        expected_git_commit=expected_git_commit,
    )
    contract_source = Path(os.path.abspath(os.fspath(contract_path)))
    corpus_source = Path(os.path.abspath(os.fspath(accepted_corpus_path)))
    projection_source = Path(os.path.abspath(os.fspath(training_projection_path)))
    reference_source = Path(os.path.abspath(os.fspath(organizer_reference_path)))
    bundle_directories = {
        "seed-42-primary": Path(os.path.abspath(os.fspath(seed42_primary_bundle))),
        "seed-42-twin": Path(os.path.abspath(os.fspath(seed42_twin_bundle))),
        "seed-43": Path(os.path.abspath(os.fspath(seed43_bundle))),
        "seed-44": Path(os.path.abspath(os.fspath(seed44_bundle))),
    }
    contract, contract_payload, protocol, corpus, training, binding = _load_inputs(
        contract_path=contract_source,
        accepted_corpus_path=corpus_source,
        training_projection_path=projection_source,
        bundle_directories=bundle_directories,
        expected_git_commit=expected_git_commit,
    )
    if (
        protocol.seeds != _SEEDS
        or protocol.width != _WIDTH
        or protocol.evaluation_batch_sequences != _BATCH_SIZE
        or protocol.raw_proposals_per_seed != _PROPOSALS_PER_SEED
        or contract.sampling.batch_sequences != _BATCH_SIZE
        or contract.compute.maximum_peak_gpu_memory_gib != _MAXIMUM_PEAK_GPU_MEMORY_GIB
        or protocol.baseline_methods != _BASELINE_METHODS
        or protocol.proposal_methods != _PROPOSAL_METHODS
    ):
        raise ValueError("loaded contract differs from the frozen production driver constants")
    output = _validated_output_path(
        output_dir,
        contract=contract,
        protected_paths=(
            repository,
            contract_source,
            corpus_source,
            projection_source,
            reference_source,
            *bundle_directories.values(),
        ),
    )
    completed: set[str] = {"clean_synchronized_git"}

    ledger = build_validation_case_ledger(
        corpus,
        levels=protocol.levels,
        replicates=protocol.replicates,
        evaluation_seed=contract.evaluation.evaluation_seed,
        require_locked_census=True,
    )
    corruptions = _materialize_validation_corruptions(corpus.validation_rows, ledger)
    if (
        corruptions.width != _WIDTH
        or len(corruptions.rows) != protocol.validation_sequences
        or len(corruptions.ledger.cases) != protocol.corruption_cases
    ):
        raise RuntimeError("materialized validation panel differs from the frozen census")
    layout = _selected_token_layout(corruptions)
    buffer = _allocate_statistic_buffer(len(protocol.token_stats_methods), layout.selected_count)
    suite = fit_count_baselines(training, require_locked_census=True)
    if suite.training_projection_sha256 != protocol.training_projection_sha256:
        raise RuntimeError("count baselines did not bind the accepted training projection")
    _reset_peak_gpu_memory_observation(contract)
    _score_baselines(corruptions, suite, layout, buffer)

    proposals: dict[tuple[str, int], ProposalBatch] = {}
    primary_records = binding.primary_by_seed
    deterministic_seeds: set[int] = set()
    runtime_seeds: set[int] = set()
    integrity_seeds: set[int] = set()
    for seed_index, seed in enumerate(protocol.seeds):
        lengths, length_plan_sha256 = _draw_shared_length_plan(
            training.length_prior,
            seed=seed,
            count=protocol.raw_proposals_per_seed,
            require_locked_count=True,
        )
        for method in protocol.proposal_methods[1:]:
            control = sample_count_generator_control_v0(
                suite,
                lengths,
                method=method,
                seed=seed,
                batch_size=_BATCH_SIZE,
                require_locked_count=True,
            )
            if control.length_plan_sha256 != length_plan_sha256:
                raise RuntimeError("control sampler changed the shared length plan")
            proposals[(method, seed)] = ProposalBatch.from_control(control, protocol=protocol)

        label = "seed-42-primary" if seed == 42 else f"seed-{seed}"
        provider = load_native_v0_logit_provider(
            training_bundle=bundle_directories[label],
            contract_path=contract_source,
            repository_root=repository,
            expected_git_commit=expected_git_commit,
        )
        record = primary_records[seed]
        _verify_provider_binding(
            provider,
            record,
            protocol=protocol,
            expected_git_commit=expected_git_commit,
        )
        integrity_seeds.add(seed)
        runtime_seeds.add(seed)
        _verify_provider_repeatability(provider, corruptions)
        deterministic_seeds.add(seed)
        _score_native_provider(
            provider,
            corruptions,
            layout,
            buffer,
            method_index=len(protocol.baseline_methods) + seed_index,
        )
        sampled = sample_unconditional_v0(
            provider,
            lengths,
            checkpoint_logical_sha256=provider.checkpoint_logical_sha256,
            seed=seed,
            contract_sha256=protocol.config_sha256,
            batch_size=_BATCH_SIZE,
            require_locked_count=True,
        )
        if sampled.length_plan_sha256 != length_plan_sha256:
            raise RuntimeError("native sampler changed the shared length plan")
        proposals[(_PROPOSAL_METHODS[0], seed)] = ProposalBatch.from_diffusion(
            sampled,
            protocol=protocol,
        )
        del provider

    if integrity_seeds != set(_SEEDS):
        raise RuntimeError("not every primary training bundle was verified by its provider")
    completed.add("training_bundle_integrity")
    if runtime_seeds != set(_SEEDS):
        raise RuntimeError("not every primary provider passed the exact runtime preflight")
    completed.add("runtime_environment")
    if deterministic_seeds != set(_SEEDS):
        raise RuntimeError("not every primary provider passed deterministic replay")
    completed.add("deterministic_execution")

    token_statistics = _finalize_token_statistics(
        layout,
        buffer,
        methods=protocol.token_stats_methods,
    )
    token_statistics.validate_against(corruptions)
    completed.add("validation_execution")
    proposal_batches = _validate_proposal_batches(
        proposals,
        protocol=protocol,
        binding=binding,
        training=training,
        suite=suite,
    )
    completed.add("sampling_execution")
    if any(
        np.any(~np.isfinite(value))
        for value in (
            token_statistics.target_log_probability,
            token_statistics.top1_confidence,
            token_statistics.multiclass_brier,
        )
    ):
        raise FloatingPointError("final selected-token statistics contain non-finite values")
    if any(
        len(batch.records) != _PROPOSALS_PER_SEED
        or any(not math.isfinite(float(record.length)) for record in batch.records)
        for batch in proposal_batches
    ):
        raise RuntimeError("final proposal census or values are invalid")
    completed.add("finite_values")

    def publish_generated_evidence(
        organizer_reference: OrganizerReference,
    ) -> EvaluationBundleResult:
        # The orchestration boundary supplies this only after the proposal-grid
        # and synchronized peak-memory gates have passed.
        completed.add("accepted_input_hashes")
        producer_checks = _producer_checks_from_completed(completed)

        def prepublish_check() -> None:
            _verify_repository_snapshot(
                repository,
                expected_git_commit=expected_git_commit,
            )
            current = _load_inputs(
                contract_path=contract_source,
                accepted_corpus_path=corpus_source,
                training_projection_path=projection_source,
                bundle_directories=bundle_directories,
                expected_git_commit=expected_git_commit,
            )
            (
                current_contract,
                current_payload,
                current_protocol,
                current_corpus,
                current_training,
                current_binding,
            ) = current
            current_reference = _load_post_generation_organizer_reference(
                reference_source,
                protocol=current_protocol,
                proposal_batches=proposal_batches,
            )
            if (
                current_contract != contract
                or current_payload != contract_payload
                or current_protocol != protocol
                or current_corpus != corpus
                or current_training != training
                or current_binding != binding
                or current_reference != organizer_reference
            ):
                raise RuntimeError("evaluation inputs changed before immutable publication")

        return publish_unconditional_v0_evaluation_bundle(
            output_dir=output,
            contract_path=contract_source,
            accepted_corpus_path=corpus_source,
            training_projection_path=projection_source,
            organizer_reference_path=reference_source,
            training_bundle_directories=bundle_directories,
            corruptions=corruptions,
            token_statistics=token_statistics,
            proposal_batches=proposal_batches,
            producer_checks=producer_checks,
            expected_git_commit=expected_git_commit,
            prepublish_check=prepublish_check,
        )

    return _publish_post_generation_evidence(
        contract,
        organizer_reference_path=reference_source,
        protocol=protocol,
        proposal_batches=proposal_batches,
        publisher=publish_generated_evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the byte-pinned native categorical-diffusion v0 cohort evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--accepted-corpus", required=True, type=Path)
    parser.add_argument("--training-projection", required=True, type=Path)
    parser.add_argument("--organizer-reference", required=True, type=Path)
    parser.add_argument("--seed-42-primary-bundle", required=True, type=Path)
    parser.add_argument("--seed-42-twin-bundle", required=True, type=Path)
    parser.add_argument("--seed-43-bundle", required=True, type=Path)
    parser.add_argument("--seed-44-bundle", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_unconditional_v0_evaluation(
        output_dir=arguments.output,
        contract_path=arguments.contract,
        accepted_corpus_path=arguments.accepted_corpus,
        training_projection_path=arguments.training_projection,
        organizer_reference_path=arguments.organizer_reference,
        seed42_primary_bundle=arguments.seed_42_primary_bundle,
        seed42_twin_bundle=arguments.seed_42_twin_bundle,
        seed43_bundle=arguments.seed_43_bundle,
        seed44_bundle=arguments.seed_44_bundle,
        repository_root=arguments.repository_root,
        expected_git_commit=arguments.expected_git_commit,
    )
    print(
        json.dumps(
            {
                "decision_status": result.decision_status,
                "logical_sha256": result.logical_sha256,
                "manifest_sha256": result.manifest_sha256,
                "output_dir": str(result.output_dir),
                "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                "training_bundle_sha256": result.training_bundle_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the production CLI
    raise SystemExit(main())

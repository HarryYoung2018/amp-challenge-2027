"""Strict training entry point for native unconditional categorical diffusion v0.

The public entry point accepts only the byte-pinned preregistration and
train-only projection.
The smaller :class:`TrainingPlan` and :func:`_execute_training_plan` boundary is
deliberately kept separate so unit tests can exercise a tiny immutable model on
CPU without creating a second production contract or weakening the CLI pins.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch import nn

from amp_challenge.generators.diffusion.categorical import (
    DEFAULT_ALPHABET,
    AbsorbingDiffusion,
    CosineMaskSchedule,
    PeptideVocabulary,
)
from amp_challenge.generators.diffusion.contract import (
    NativeDiffusionContract,
    load_unconditional_v0_contract,
)
from amp_challenge.generators.diffusion.data import (
    TrainingDistribution,
    load_training_projection,
    namespaced_seed,
)
from amp_challenge.generators.diffusion.model import (
    CheckpointHashes,
    NativeDenoiser,
    NativeDenoiserConfig,
    configure_deterministic_runtime,
    masked_token_objective,
    save_safetensors_checkpoint,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

TRAINING_BUNDLE_FILES = (
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
TRAINING_MANIFEST_FIELDS = (
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
)
RNG_NAMESPACES = (
    "initialization",
    "minibatch",
    "timestep",
    "corruption",
    "dropout",
    "proposal",
)
TWIN_ENVIRONMENT_EQUAL_FIELDS = (
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

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DRIVER_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")
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
_SCHEDULE_DOMAIN = b"amp-native-diffusion-learning-rate-schedule-v1\x00"
_BATCH_DOMAIN = b"amp-native-diffusion-training-batch-v1\x00"
_UINT64_RANGE = 1 << 64


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Immutable, fully resolved values used by the training loop.

    Production plans are created only by :func:`training_plan_from_contract`.
    Tests may construct a smaller instance and call the private execution core;
    the public trainer still reloads the exact byte-pinned v0 contract.
    """

    artifact: str
    config_sha256: str
    corpus_sha256: str
    training_projection_sha256: str
    expected_train_sequences: int
    model_config: NativeDenoiserConfig
    expected_trainable_parameters: int
    seeds: tuple[int, ...]
    batch_sequences: int
    max_steps: int
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    warmup_steps: int
    final_learning_rate: float
    gradient_clip_norm: float
    log_interval_steps: int
    cosine_offset: float
    rng_namespaces: tuple[str, ...]
    maximum_peak_gpu_memory_gib: float
    bundle_files: tuple[str, ...]
    manifest_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.artifact) is not str or not self.artifact:
            raise ValueError("artifact must be a non-empty string")
        _require_sha256(self.config_sha256, label="config_sha256")
        _require_sha256(self.corpus_sha256, label="corpus_sha256")
        _require_sha256(
            self.training_projection_sha256,
            label="training_projection_sha256",
        )
        if not isinstance(self.model_config, NativeDenoiserConfig):
            raise TypeError("model_config must be a NativeDenoiserConfig")
        positive_integers = {
            "expected_train_sequences": self.expected_train_sequences,
            "expected_trainable_parameters": self.expected_trainable_parameters,
            "batch_sequences": self.batch_sequences,
            "max_steps": self.max_steps,
            "log_interval_steps": self.log_interval_steps,
        }
        for label, value in positive_integers.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
        ):
            raise ValueError("seeds must be unique uint32 integers")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be an integer in [0, max_steps)")
        if (
            type(self.betas) is not tuple
            or len(self.betas) != 2
            or any(type(value) is not float or not 0.0 < value < 1.0 for value in self.betas)
        ):
            raise ValueError("betas must contain two floats in (0, 1)")
        positive_floats = {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
            "maximum_peak_gpu_memory_gib": self.maximum_peak_gpu_memory_gib,
        }
        for label, value in positive_floats.items():
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be a positive finite float")
        if (
            type(self.weight_decay) is not float
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("weight_decay must be a non-negative finite float")
        if (
            type(self.final_learning_rate) is not float
            or not math.isfinite(self.final_learning_rate)
            or not 0.0 < self.final_learning_rate <= self.learning_rate
        ):
            raise ValueError("final_learning_rate must lie in (0, learning_rate]")
        if (
            type(self.cosine_offset) is not float
            or not math.isfinite(self.cosine_offset)
            or not 0.0 <= self.cosine_offset < 1.0
        ):
            raise ValueError("cosine_offset must be a finite float in [0, 1)")
        if self.rng_namespaces != RNG_NAMESPACES:
            raise ValueError("rng_namespaces differ from the v0 namespace contract")
        if self.bundle_files != TRAINING_BUNDLE_FILES:
            raise ValueError("bundle_files differ from the v0 training inventory")
        if self.manifest_fields != TRAINING_MANIFEST_FIELDS:
            raise ValueError("manifest_fields differ from the v0 manifest schema")


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    """Path-free bytes copied into, or used to construct, a training bundle."""

    contract_payload: bytes
    code_manifest_payload: bytes
    input_manifest_payload: bytes
    git_commit: str

    def __post_init__(self) -> None:
        for label, payload in (
            ("contract_payload", self.contract_payload),
            ("code_manifest_payload", self.code_manifest_payload),
            ("input_manifest_payload", self.input_manifest_payload),
        ):
            if type(payload) is not bytes or not payload:
                raise ValueError(f"{label} must be non-empty bytes")
        if _GIT_COMMIT_RE.fullmatch(self.git_commit) is None:
            raise ValueError("git_commit must be a lowercase forty-character Git object ID")


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Operational return value for one successfully published final run."""

    output_dir: Path
    checkpoint_hashes: CheckpointHashes
    final_loss: float
    mean_loss: float
    steps: int


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _expect(label: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f"{label} must equal {expected!r}, got {observed!r}")


def training_plan_from_contract(contract: NativeDiffusionContract) -> TrainingPlan:
    """Resolve the already byte-pinned production contract into loop values."""

    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
    required = {
        "input.component_weighting": (
            contract.input.component_weighting,
            "homology_component_equal_within_role_v1",
        ),
        "model.kind": (contract.model.kind, "bidirectional_pre_layer_norm_transformer"),
        "model.alphabet": (contract.model.alphabet, DEFAULT_ALPHABET),
        "model.special_tokens": (contract.model.special_tokens, ("PAD", "MASK")),
        "model.activation": (contract.model.activation, "gelu"),
        "model.token_embedding": (contract.model.token_embedding, "learned"),
        "model.timestep_embedding": (
            contract.model.timestep_embedding,
            "learned_0_to_64_row_zero_reserved",
        ),
        "model.tie_residue_input_output_weights": (
            contract.model.tie_residue_input_output_weights,
            True,
        ),
        "model.attention_projection_bias": (
            contract.model.attention_projection_bias,
            True,
        ),
        "model.ffn_bias": (contract.model.ffn_bias, True),
        "model.final_layer_norm": (contract.model.final_layer_norm, True),
        "model.output_bias": (contract.model.output_bias, True),
        "model.prediction_classes": (contract.model.prediction_classes, 20),
        "model.initialization": (
            contract.model.initialization,
            "normal_std_0.02_bias_zero_norm_scale_one",
        ),
        "diffusion.kind": (contract.diffusion.kind, "absorbing_mask_fixed_count"),
        "diffusion.schedule": (contract.diffusion.schedule, "cosine_alpha_bar"),
        "diffusion.timestep_sampling": (
            contract.diffusion.timestep_sampling,
            "uniform_integer_1_to_64",
        ),
        "diffusion.mask_count": (
            contract.diffusion.mask_count,
            "ceil_length_times_mask_probability",
        ),
        "diffusion.mask_position_sampling": (
            contract.diffusion.mask_position_sampling,
            "uniform_without_replacement",
        ),
        "diffusion.prediction_target": (contract.diffusion.prediction_target, "clean_residue"),
        "diffusion.loss_positions": (
            contract.diffusion.loss_positions,
            "masked_valid_positions_only",
        ),
        "diffusion.loss_reduction": (
            contract.diffusion.loss_reduction,
            "masked_mean_per_sequence_then_batch_mean",
        ),
        "training.sample_with_replacement": (contract.training.sample_with_replacement, True),
        "training.sampling_weight_application": (
            contract.training.sampling_weight_application,
            "weighted_draw_only",
        ),
        "training.optimizer": (contract.training.optimizer, "adamw_unfused"),
        "training.exclude_from_weight_decay": (
            contract.training.exclude_from_weight_decay,
            ("bias", "normalization", "embedding"),
        ),
        "training.lr_schedule": (
            contract.training.lr_schedule,
            "linear_warmup_cosine_decay",
        ),
        "training.precision": (contract.training.precision, "float32"),
        "training.gradient_accumulation_steps": (
            contract.training.gradient_accumulation_steps,
            1,
        ),
        "training.early_stopping": (contract.training.early_stopping, False),
        "training.checkpoint_selection": (
            contract.training.checkpoint_selection,
            "final_step_10000_only",
        ),
        "training.validation_during_training": (
            contract.training.validation_during_training,
            False,
        ),
        "training.ema": (contract.training.ema, False),
        "training.amp": (contract.training.amp, False),
        "training.tf32": (contract.training.tf32, False),
        "training.torch_compile": (contract.training.torch_compile, False),
        "determinism.rng_derivation": (
            contract.determinism.rng_derivation,
            "sha256_namespace_to_uint64_v1",
        ),
        "determinism.rng_namespaces": (
            contract.determinism.rng_namespaces,
            RNG_NAMESPACES,
        ),
        "determinism.corruption_rng": (
            contract.determinism.corruption_rng,
            "numpy_pcg64_2.4.6",
        ),
        "determinism.deterministic_algorithms": (
            contract.determinism.deterministic_algorithms,
            True,
        ),
        "determinism.math_sdpa_only": (contract.determinism.math_sdpa_only, True),
        "determinism.mha_fastpath": (contract.determinism.mha_fastpath, False),
        "determinism.cublas_workspace_config": (
            contract.determinism.cublas_workspace_config,
            ":4096:8",
        ),
        "determinism.cudnn_benchmark": (contract.determinism.cudnn_benchmark, False),
        "determinism.dataloader_workers": (contract.determinism.dataloader_workers, 0),
        "determinism.pytorch_allocator": (
            contract.determinism.pytorch_allocator,
            "backend:native",
        ),
        "determinism.twin_environment_equal_fields": (
            contract.determinism.twin_environment_equal_fields,
            TWIN_ENVIRONMENT_EQUAL_FIELDS,
        ),
        "environment.python": (contract.environment.python, "3.11.14"),
        "environment.numpy": (contract.environment.numpy, "2.4.6"),
        "environment.torch": (contract.environment.torch, "2.14.0"),
        "environment.torch_cuda": (contract.environment.torch_cuda, "13.0"),
        "environment.safetensors": (contract.environment.safetensors, "0.6.2"),
        "environment.triton": (contract.environment.triton, "3.8.0"),
        "environment.nvidia_cudnn_cu13": (
            contract.environment.nvidia_cudnn_cu13,
            "9.24.0.43",
        ),
        "environment.gpu_name": (
            contract.environment.gpu_name,
            "NVIDIA A100-SXM4-80GB",
        ),
        "environment.compute_capability": (
            contract.environment.compute_capability,
            (8, 0),
        ),
        "environment.checkpoint_format": (contract.environment.checkpoint_format, "safetensors"),
        "environment.allow_pickle": (contract.environment.allow_pickle, False),
        "artifacts.schema_version": (contract.artifacts.schema_version, 1),
        "artifacts.canonical_json": (contract.artifacts.canonical_json, True),
        "artifacts.reject_nonfinite_json": (contract.artifacts.reject_nonfinite_json, True),
        "artifacts.file_mode": (contract.artifacts.file_mode, "0444"),
        "artifacts.directory_mode": (contract.artifacts.directory_mode, "0555"),
        "artifacts.manifest_published_last": (
            contract.artifacts.manifest_published_last,
            True,
        ),
        "artifacts.semantic_manifests_path_free": (
            contract.artifacts.semantic_manifests_path_free,
            True,
        ),
        "artifacts.training_bundle_files": (
            contract.artifacts.training_bundle_files,
            TRAINING_BUNDLE_FILES,
        ),
        "artifacts.training_manifest_fields": (
            contract.artifacts.training_manifest_fields,
            TRAINING_MANIFEST_FIELDS,
        ),
        "leakage.trainer_allowed_roles": (contract.leakage.trainer_allowed_roles, ("train",)),
        "leakage.trainer_allowed_fields": (
            contract.leakage.trainer_allowed_fields,
            ("sequence_id", "sequence", "sampling_weight"),
        ),
        "leakage.validation_visible_to_trainer": (
            contract.leakage.validation_visible_to_trainer,
            False,
        ),
        "leakage.validation_used_for_early_stopping": (
            contract.leakage.validation_used_for_early_stopping,
            False,
        ),
    }
    for label, (observed, expected) in required.items():
        _expect(label, observed, expected)
    if len(contract.training.betas) != 2:
        raise ValueError("training.betas must contain exactly two values")
    if any(
        (
            contract.model.property_conditioning,
            contract.model.classifier_free_guidance,
            contract.model.self_conditioning,
            contract.leakage.labels_allowed,
            contract.leakage.provenance_allowed,
            contract.leakage.study_keys_allowed,
            contract.leakage.oracle_predictions_allowed,
            contract.leakage.structures_allowed,
            contract.leakage.organizer_reference_allowed_during_training,
        )
    ):
        raise ValueError("production contract exposes conditioning or prohibited trainer data")
    model_config = NativeDenoiserConfig(
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
    return TrainingPlan(
        artifact=contract.artifact,
        config_sha256=contract.config_sha256,
        corpus_sha256=contract.input.corpus_sha256,
        training_projection_sha256=contract.input.training_projection_sha256,
        expected_train_sequences=contract.input.expected_train_sequences,
        model_config=model_config,
        expected_trainable_parameters=contract.model.expected_trainable_parameters,
        seeds=contract.training.seeds,
        batch_sequences=contract.training.batch_sequences,
        max_steps=contract.training.max_steps,
        learning_rate=contract.training.learning_rate,
        betas=(contract.training.betas[0], contract.training.betas[1]),
        epsilon=contract.training.epsilon,
        weight_decay=contract.training.weight_decay,
        warmup_steps=contract.training.warmup_steps,
        final_learning_rate=contract.training.final_learning_rate,
        gradient_clip_norm=contract.training.gradient_clip_norm,
        log_interval_steps=contract.training.training_log_interval_steps,
        cosine_offset=contract.diffusion.cosine_offset,
        rng_namespaces=contract.determinism.rng_namespaces,
        maximum_peak_gpu_memory_gib=contract.compute.maximum_peak_gpu_memory_gib,
        bundle_files=contract.artifacts.training_bundle_files,
        manifest_fields=contract.artifacts.training_manifest_fields,
    )


def learning_rate_for_step(plan: TrainingPlan, step: int) -> float:
    """Return the preregistered one-indexed warmup/cosine update rate."""

    if not isinstance(plan, TrainingPlan):
        raise TypeError("plan must be a TrainingPlan")
    if type(step) is not int or not 1 <= step <= plan.max_steps:
        raise ValueError("step must be an integer in 1..max_steps")
    if plan.warmup_steps and step <= plan.warmup_steps:
        return plan.learning_rate * step / plan.warmup_steps
    decay_steps = plan.max_steps - plan.warmup_steps
    progress = (step - plan.warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return plan.final_learning_rate + (plan.learning_rate - plan.final_learning_rate) * cosine


def _schedule_digest(plan: TrainingPlan) -> str:
    digest = hashlib.sha256()
    digest.update(_SCHEDULE_DOMAIN)
    for step in range(1, plan.max_steps + 1):
        digest.update(step.to_bytes(8, "big"))
        encoded = learning_rate_for_step(plan, step).hex().encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _bounded_uint64(root_seed: int, namespace: str, upper_bound: int, *parts: str | int) -> int:
    if type(upper_bound) is not int or upper_bound <= 0 or upper_bound > _UINT64_RANGE:
        raise ValueError("upper_bound must be an integer in 1..2**64")
    acceptance_limit = _UINT64_RANGE - (_UINT64_RANGE % upper_bound)
    retry = 0
    while True:
        value = namespaced_seed(root_seed, namespace, *parts, retry)
        if value < acceptance_limit:
            return value % upper_bound
        retry += 1


def _levels_for_ordinals(
    *,
    root_seed: int,
    start: int,
    count: int,
    levels: int,
) -> np.ndarray:
    return np.asarray(
        [
            1 + _bounded_uint64(root_seed, "timestep", levels, ordinal)
            for ordinal in range(start, start + count)
        ],
        dtype=np.int64,
    )


def _optimizer_parameter_groups(
    model: NativeDenoiser,
    *,
    weight_decay: float,
) -> tuple[list[dict[str, object]], tuple[str, ...], tuple[str, ...]]:
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding | nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))

    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    named = sorted(model.named_parameters(), key=lambda item: item[0])
    for name, parameter in named:
        excluded = name.endswith("bias") or id(parameter) in no_decay_ids
        if excluded:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if not decay or not no_decay or len(decay) + len(no_decay) != len(named):
        raise RuntimeError("AdamW parameter partition is incomplete")
    if {id(parameter) for parameter in decay} & {id(parameter) for parameter in no_decay}:
        raise RuntimeError("AdamW parameter partition overlaps")
    groups: list[dict[str, object]] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return groups, tuple(decay_names), tuple(no_decay_names)


def _build_optimizer(
    model: NativeDenoiser,
    plan: TrainingPlan,
) -> tuple[torch.optim.AdamW, tuple[str, ...], tuple[str, ...]]:
    groups, decay_names, no_decay_names = _optimizer_parameter_groups(
        model,
        weight_decay=plan.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=plan.learning_rate,
        betas=plan.betas,
        eps=plan.epsilon,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    return optimizer, decay_names, no_decay_names


def _validate_training_distribution(
    distribution: TrainingDistribution,
    plan: TrainingPlan,
) -> None:
    if not isinstance(distribution, TrainingDistribution):
        raise TypeError("distribution must be a TrainingDistribution")
    if len(distribution.rows) != plan.expected_train_sequences:
        raise ValueError("trainer-visible row count differs from the immutable plan")
    if len({row.sequence_id for row in distribution.rows}) != len(distribution.rows):
        raise ValueError("trainer-visible sequence IDs must be unique")
    for row in distribution.rows:
        sequence = canonicalize_sequence(
            row.sequence,
            min_length=plan.model_config.min_length,
            max_length=plan.model_config.max_length,
        )
        if sequence != row.sequence or canonical_sequence_id(sequence) != row.sequence_id:
            raise ValueError("trainer-visible row has a noncanonical sequence identity")
    expected_length_mass: dict[int, list[float]] = {}
    for row in distribution.rows:
        expected_length_mass.setdefault(len(row.sequence), []).append(row.sampling_weight)
    lengths = tuple(sorted(expected_length_mass))
    masses = tuple(math.fsum(expected_length_mass[length]) for length in lengths)
    total = math.fsum(masses)
    expected_prior = tuple(value / total for value in masses)
    if (
        distribution.length_prior.lengths != lengths
        or distribution.length_prior.probabilities != expected_prior
    ):
        raise ValueError("length prior is not the train-only weighted empirical distribution")


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


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("training trace cannot be empty")
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-terminated text")
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"{label} has a malformed checksum row")
        digest, name = line[:64], line[66:]
        _require_sha256(digest, label=f"{label} digest")
        pure = PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"{label} contains an unsafe path")
        if previous is not None and name <= previous:
            raise ValueError(f"{label} rows must be strictly name-ordered")
        if name in result:
            raise ValueError(f"{label} contains a duplicate path")
        result[name] = digest
        previous = name
    return result


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    if not entries:
        raise ValueError("checksum manifest cannot be empty")
    for name, digest in entries.items():
        _require_sha256(digest, label=f"checksum for {name}")
    payload = "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))
    parsed = _parse_sha_manifest(payload.encode("ascii"), label="checksum manifest")
    if parsed != dict(entries):
        raise ValueError("checksum manifest round trip changed")
    return payload.encode("ascii")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"artifact {path.name} must be non-empty bytes")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_dropout(root_seed: int, step: int, device: torch.device) -> int:
    seed = namespaced_seed(root_seed, "dropout", step)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return seed


def _batch_digest(
    *,
    draw_start: int,
    sequence_ids: Sequence[str],
    levels: np.ndarray,
    selected: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(_BATCH_DOMAIN)
    for offset, (sequence_id, level, mask) in enumerate(
        zip(sequence_ids, levels, selected, strict=True)
    ):
        digest.update((draw_start + offset).to_bytes(8, "big"))
        digest.update(bytes.fromhex(sequence_id))
        digest.update(int(level).to_bytes(8, "big"))
        digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def _environment_document(
    *,
    device: torch.device,
    runtime: Mapping[str, object],
    production_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    if production_identity is None:
        if device.type != "cpu":
            raise ValueError("CUDA environment identity must come from strict preflight")
        identity: dict[str, object] = {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
            "torch": torch.__version__.split("+", maxsplit=1)[0],
            "torch_cuda": torch.version.cuda,
            "safetensors": _optional_distribution_version("safetensors"),
            "triton": _optional_distribution_version("triton"),
            "nvidia_cudnn_cu13": _optional_distribution_version("nvidia-cudnn-cu13"),
            "gpu_name": None,
            "compute_capability": [],
            "driver": None,
            "allocator": None,
        }
    else:
        identity = dict(production_identity)
        if tuple(identity) != TWIN_ENVIRONMENT_EQUAL_FIELDS:
            raise ValueError("production environment identity schema differs from the contract")
    return {
        "schema_version": 1,
        **identity,
        "device_type": device.type,
        "twin_environment_equal_fields": list(TWIN_ENVIRONMENT_EQUAL_FIELDS),
        "runtime": dict(runtime),
        "amp": False,
        "tf32": False,
        "torch_compile": False,
    }


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_runtime_controls(runtime: Mapping[str, object], *, seed: int) -> None:
    expected = {
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "default_dtype": "float32",
        "deterministic_algorithms": True,
        "flash_sdpa": False,
        "math_sdpa": True,
        "memory_efficient_sdpa": False,
        "mha_fastpath": False,
        "numpy_legacy_seed": seed % 2**32,
        "seed": seed,
        "matmul_tf32": False,
    }
    if dict(runtime) != expected:
        raise RuntimeError("deterministic runtime controls differ from the v0 contract")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("float32 matmul precision differs from the v0 contract")
    cudnn_sdp_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if callable(cudnn_sdp_enabled) and cudnn_sdp_enabled():
        raise RuntimeError("cuDNN scaled-dot-product attention must remain disabled")


def _execute_training_plan(
    *,
    plan: TrainingPlan,
    distribution: TrainingDistribution,
    provenance: TrainingProvenance,
    output_dir: str | Path,
    seed: int,
    device: str | torch.device,
    require_cuda: bool,
    production_environment_identity: Mapping[str, object] | None = None,
    prepublish_check: Callable[[], None] | None = None,
) -> TrainingRunResult:
    """Execute a resolved plan; production callers must set ``require_cuda``."""

    if not isinstance(plan, TrainingPlan):
        raise TypeError("plan must be a TrainingPlan")
    if not isinstance(provenance, TrainingProvenance):
        raise TypeError("provenance must be TrainingProvenance")
    if type(seed) is not int or seed not in plan.seeds:
        raise ValueError("seed must be one of the immutable plan seeds")
    if not isinstance(require_cuda, bool):
        raise TypeError("require_cuda must be boolean")
    if not require_cuda and production_environment_identity is not None:
        raise ValueError("test runs cannot inject a production environment identity")
    if prepublish_check is not None and not callable(prepublish_check):
        raise TypeError("prepublish_check must be callable")
    if hashlib.sha256(provenance.contract_payload).hexdigest() != plan.config_sha256:
        raise ValueError("contract payload does not match the plan config SHA-256")
    _parse_sha_manifest(provenance.code_manifest_payload, label="CODE_SHA256SUMS")
    input_entries = _parse_sha_manifest(
        provenance.input_manifest_payload,
        label="INPUT_SHA256SUMS",
    )
    if input_entries != {
        "contract.toml": plan.config_sha256,
        "training_projection.jsonl": plan.training_projection_sha256,
    }:
        raise ValueError("INPUT_SHA256SUMS does not bind the plan inputs")
    _validate_training_distribution(distribution, plan)

    target_device = torch.device(device)
    if require_cuda and target_device.type != "cuda":
        raise ValueError("production v0 training requires a CUDA device")
    if require_cuda and target_device != torch.device("cuda:0"):
        raise ValueError("production v0 training requires the bound CUDA device cuda:0")
    if require_cuda and production_environment_identity is None:
        raise ValueError("CUDA production runs require the strict preflight environment identity")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    output = _validated_new_output(output_dir)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=str(output.parent),
        )
    )
    published = False
    checkpoint_hashes: CheckpointHashes | None = None
    losses: list[float] = []
    accuracies: list[float] = []
    gradient_norms: list[float] = []
    total_selected = 0
    trace: list[dict[str, object]] = []
    try:
        _write_new_bytes(staging / "CODE_SHA256SUMS", provenance.code_manifest_payload)
        _write_new_bytes(staging / "INPUT_SHA256SUMS", provenance.input_manifest_payload)
        _write_new_bytes(staging / "contract.toml", provenance.contract_payload)

        initialization_seed = namespaced_seed(seed, "initialization", "model")
        runtime = configure_deterministic_runtime(initialization_seed)
        _validate_runtime_controls(runtime, seed=initialization_seed)
        if target_device.type == "cuda":
            torch.cuda.set_device(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
        model = NativeDenoiser(plan.model_config).to(device=target_device, dtype=torch.float32)
        parameters = tuple(model.parameters())
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("every native-v0 model parameter must remain trainable")
        parameter_count = sum(parameter.numel() for parameter in parameters)
        if parameter_count != plan.expected_trainable_parameters:
            raise ValueError(
                "model parameter count differs from the immutable plan: "
                f"expected {plan.expected_trainable_parameters}, got {parameter_count}"
            )
        optimizer, decay_names, no_decay_names = _build_optimizer(model, plan)
        vocabulary = PeptideVocabulary(DEFAULT_ALPHABET)
        diffusion = AbsorbingDiffusion(
            vocabulary=vocabulary,
            schedule=CosineMaskSchedule(plan.cosine_offset),
        )
        model.train()

        rng_document = {
            "schema_version": 1,
            "derivation": "sha256_namespace_to_uint64_v1",
            "root_seed": seed,
            "initialization_seed_uint64": initialization_seed,
            "namespaces": list(plan.rng_namespaces),
            "minibatch_key": ["global_draw_ordinal"],
            "timestep_key": ["global_draw_ordinal", "rejection_counter"],
            "corruption_key": ["global_draw_ordinal", "sequence_id", "level"],
            "dropout_key": ["optimizer_step"],
        }
        _write_new_bytes(
            staging / "environment.json",
            _canonical_json_bytes(
                _environment_document(
                    device=target_device,
                    runtime=runtime,
                    production_identity=production_environment_identity,
                )
            ),
        )
        _write_new_bytes(staging / "rng.json", _canonical_json_bytes(rng_document))
        schedule_digest = _schedule_digest(plan)
        _write_new_bytes(
            staging / "training_schedule.sha256",
            f"{schedule_digest}\n".encode("ascii"),
        )

        for step in range(1, plan.max_steps + 1):
            draw_start = (step - 1) * plan.batch_sequences
            rows = distribution.draw(
                root_seed=seed,
                draw_start=draw_start,
                draw_count=plan.batch_sequences,
                namespace="minibatch",
            )
            encoded = vocabulary.encode(
                [row.sequence for row in rows],
                max_length=plan.model_config.max_length,
            )
            levels = _levels_for_ordinals(
                root_seed=seed,
                start=draw_start,
                count=plan.batch_sequences,
                levels=plan.model_config.levels,
            )
            corruption_seeds = tuple(
                namespaced_seed(
                    seed,
                    "corruption",
                    draw_start + offset,
                    row.sequence_id,
                    int(levels[offset]),
                )
                for offset, row in enumerate(rows)
            )
            corrupted, selected = diffusion.corrupt_fixed_count(
                encoded.tokens,
                encoded.attention_mask,
                levels,
                total_levels=plan.model_config.levels,
                row_seeds=corruption_seeds,
            )
            lengths = encoded.attention_mask.sum(axis=1, dtype=np.int64)
            expected_mask_counts = diffusion.schedule.mask_counts(
                lengths,
                levels,
                total_levels=plan.model_config.levels,
            )
            observed_mask_counts = selected.sum(axis=1, dtype=np.int64)
            if not np.array_equal(observed_mask_counts, expected_mask_counts):
                raise RuntimeError("fixed-count corruption violated the timestep mask schedule")
            dropout_seed = _seed_dropout(seed, step, target_device)
            clean_tensor = torch.from_numpy(encoded.tokens).to(target_device)
            corrupted_tensor = torch.from_numpy(corrupted).to(target_device)
            attention_tensor = torch.from_numpy(encoded.attention_mask).to(target_device)
            selected_tensor = torch.from_numpy(selected).to(target_device)
            levels_tensor = torch.from_numpy(levels).to(target_device)
            lengths_tensor = attention_tensor.sum(dim=1, dtype=torch.long)

            learning_rate = learning_rate_for_step(plan, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                corrupted_tensor,
                attention_tensor,
                levels_tensor,
                lengths_tensor,
            )
            objective = masked_token_objective(
                logits,
                clean_tensor,
                selected_tensor,
                attention_tensor,
                corrupted_tokens=corrupted_tensor,
            )
            objective.loss.backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                plan.gradient_clip_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()

            loss = float(objective.loss.detach().cpu().item())
            accuracy = float(objective.row_accuracies.detach().mean().cpu().item())
            gradient_norm = float(gradient_norm_tensor.detach().cpu().item())
            if not all(math.isfinite(value) for value in (loss, accuracy, gradient_norm)):
                raise FloatingPointError("training produced a non-finite metric")
            selected_count = int(objective.selected_counts.detach().sum().cpu().item())
            losses.append(loss)
            accuracies.append(accuracy)
            gradient_norms.append(gradient_norm)
            total_selected += selected_count
            if step == 1 or step % plan.log_interval_steps == 0 or step == plan.max_steps:
                trace.append(
                    {
                        "schema_version": 1,
                        "step": step,
                        "loss": loss,
                        "mean_row_accuracy": accuracy,
                        "gradient_norm_before_clipping": gradient_norm,
                        "learning_rate": learning_rate,
                        "selected_tokens": selected_count,
                        "dropout_seed": dropout_seed,
                        "batch_sha256": _batch_digest(
                            draw_start=draw_start,
                            sequence_ids=[row.sequence_id for row in rows],
                            levels=levels,
                            selected=selected,
                        ),
                    }
                )

        peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        )
        if peak_memory_bytes > int(plan.maximum_peak_gpu_memory_gib * 1024**3):
            raise RuntimeError("training exceeded the preregistered peak GPU-memory cap")
        metrics = {
            "schema_version": 1,
            "steps": plan.max_steps,
            "drawn_sequences": plan.max_steps * plan.batch_sequences,
            "total_selected_tokens": total_selected,
            "final_loss": losses[-1],
            "mean_loss": math.fsum(losses) / len(losses),
            "final_mean_row_accuracy": accuracies[-1],
            "mean_row_accuracy": math.fsum(accuracies) / len(accuracies),
            "mean_gradient_norm_before_clipping": math.fsum(gradient_norms) / len(gradient_norms),
            "final_learning_rate": learning_rate_for_step(plan, plan.max_steps),
            "loss_reduction": "masked_mean_per_sequence_then_batch_mean",
            "sampling_weight_application": "weighted_draw_only",
            "validation_consulted": False,
            "peak_gpu_memory_bytes": peak_memory_bytes,
        }
        _write_new_bytes(
            staging / "training_trace.jsonl",
            _canonical_jsonl_bytes(trace),
        )
        _write_new_bytes(staging / "train_metrics.json", _canonical_json_bytes(metrics))
        checkpoint_hashes = save_safetensors_checkpoint(
            model,
            staging / "model_final.safetensors",
        )
        _require_sha256(
            checkpoint_hashes.logical_state_sha256,
            label="checkpoint logical-state SHA-256",
        )

        artifact_hashes = {
            name: _file_sha256(staging / name)
            for name in TRAINING_BUNDLE_FILES
            if name != "manifest.json"
        }
        if artifact_hashes["model_final.safetensors"] != checkpoint_hashes.file_sha256:
            raise RuntimeError("final checkpoint changed after safetensors publication")
        manifest = {
            "schema_version": 1,
            "artifact": plan.artifact,
            "config_sha256": plan.config_sha256,
            "git_commit": provenance.git_commit,
            "seed": seed,
            "corpus": {
                "accepted_parent_sha256": plan.corpus_sha256,
                "training_projection_sha256": plan.training_projection_sha256,
                "trainer_visible_sequences": len(distribution.rows),
                "trainer_visible_fields": [
                    "sequence_id",
                    "sequence",
                    "sampling_weight",
                ],
                "roles": ["train"],
            },
            "model": {
                "config": asdict(plan.model_config),
                "trainable_parameters": parameter_count,
                "checkpoint_file_sha256": checkpoint_hashes.file_sha256,
                "checkpoint_logical_state_sha256": checkpoint_hashes.logical_state_sha256,
                "checkpoint_format": "safetensors",
            },
            "rng": {
                "filename": "rng.json",
                "sha256": artifact_hashes["rng.json"],
            },
            "training": {
                "steps": plan.max_steps,
                "batch_sequences": plan.batch_sequences,
                "optimizer": "adamw_unfused",
                "parameter_decay_names": list(decay_names),
                "parameter_no_decay_names": list(no_decay_names),
                "schedule_sha256": schedule_digest,
                "checkpoint_selection": f"final_step_{plan.max_steps}_only",
                "validation_during_training": False,
                "early_stopping": False,
                "resume_supported": False,
                "metrics_sha256": artifact_hashes["train_metrics.json"],
            },
            "artifacts": artifact_hashes,
        }
        if tuple(manifest) != TRAINING_MANIFEST_FIELDS:
            raise AssertionError("training manifest construction order/schema changed")
        # The manifest is intentionally the final artifact created in staging.
        _write_new_bytes(staging / "manifest.json", _canonical_json_bytes(manifest))
        if prepublish_check is not None:
            prepublish_check()
        _publish_bundle_noreplace(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            _remove_staging(staging)

    if checkpoint_hashes is None:
        raise AssertionError("published training run has no final checkpoint identity")
    return TrainingRunResult(
        output_dir=output,
        checkpoint_hashes=checkpoint_hashes,
        final_loss=losses[-1],
        mean_loss=math.fsum(losses) / len(losses),
        steps=plan.max_steps,
    )


def _validated_new_output(path: str | Path) -> Path:
    output = Path(os.path.abspath(os.fspath(path)))
    if output.name in {"", ".", ".."}:
        raise ValueError("output directory must have a concrete final name")
    parent = output.parent
    _reject_symlink_chain(parent)
    try:
        parent_stat = parent.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("output parent must already exist") from error
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("output parent must be a directory")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite or resume training output: {output}")
    return output


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


def _bundle_identity(root: Path) -> tuple[tuple[str, int, str], ...]:
    result: list[tuple[str, int, str]] = []
    for child in sorted(root.iterdir(), key=lambda value: value.name):
        metadata = child.stat(follow_symlinks=False)
        if child.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("training bundle must contain only regular non-symlink files")
        result.append((child.name, metadata.st_size, _file_sha256(child)))
    return tuple(result)


def _prepare_staging(staging: Path) -> tuple[tuple[tuple[str, int, str], ...], tuple[int, int]]:
    if {child.name for child in staging.iterdir()} != set(TRAINING_BUNDLE_FILES):
        raise ValueError("training staging inventory differs from the contract")
    for child in staging.iterdir():
        if stat.S_IMODE(child.stat(follow_symlinks=False).st_mode) != 0o444:
            raise ValueError("training artifacts must be read-only before publication")
        with child.open("rb") as handle:
            os.fsync(handle.fileno())
    identity = _bundle_identity(staging)
    root_stat = staging.stat(follow_symlinks=False)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    directory_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.chmod(staging, 0o555)
    return identity, root_identity


def _renameat2_noreplace(staging: Path, output: Path) -> int:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    return 0 if result == 0 else (ctypes.get_errno() or errno.EIO)


def _publish_by_links(
    staging: Path,
    output: Path,
    *,
    expected_identity: tuple[tuple[str, int, str], ...],
) -> None:
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to overwrite training output: {output}") from error
    claim = output.stat(follow_symlinks=False)
    claim_identity = (claim.st_dev, claim.st_ino)
    try:
        if not stat.S_ISDIR(claim.st_mode) or output.is_symlink():
            raise RuntimeError("training publication claim is not a real directory")
        if staging.stat(follow_symlinks=False).st_dev != claim.st_dev:
            raise RuntimeError("training staging and output are on different filesystems")
        expected = {name: (size, digest) for name, size, digest in expected_identity}
        for name in sorted(set(TRAINING_BUNDLE_FILES) - {"manifest.json"}):
            os.link(staging / name, output / name, follow_symlinks=False)
        manifest_source = staging / "manifest.json"
        os.chmod(manifest_source, 0o000)
        os.link(manifest_source, output / "manifest.json", follow_symlinks=False)
        if {child.name for child in output.iterdir()} != set(TRAINING_BUNDLE_FILES):
            raise RuntimeError("linked training publication inventory changed")
        for name in set(TRAINING_BUNDLE_FILES) - {"manifest.json"}:
            observed = output / name
            observed_stat = observed.stat(follow_symlinks=False)
            expected_size, expected_digest = expected[name]
            if (
                not stat.S_ISREG(observed_stat.st_mode)
                or observed_stat.st_size != expected_size
                or _file_sha256(observed) != expected_digest
            ):
                raise RuntimeError(f"linked training publication changed for {name}")
        manifest_destination = output / "manifest.json"
        source_stat = manifest_source.stat(follow_symlinks=False)
        destination_stat = manifest_destination.stat(follow_symlinks=False)
        if (
            (source_stat.st_dev, source_stat.st_ino)
            != (destination_stat.st_dev, destination_stat.st_ino)
            or destination_stat.st_size != expected["manifest.json"][0]
            or stat.S_IMODE(destination_stat.st_mode) != 0
        ):
            raise RuntimeError("linked training manifest changed before commit")
        current = output.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != claim_identity:
            raise RuntimeError("training publication claim changed")
        directory_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(output, 0o555)
        _fsync_directory(output.parent)
        # The final chmod is the fallback publication's atomic commit point.
        os.chmod(manifest_destination, 0o444)
    except BaseException:
        try:
            current = output.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == claim_identity and not output.is_symlink():
                os.chmod(output, 0o555)
        finally:
            os.chmod(staging, 0o755)
        raise
    try:
        os.chmod(staging, 0o755)
        _remove_staging(staging)
    except OSError:
        pass


def _publish_bundle_noreplace(staging: Path, output: Path) -> None:
    expected_identity, root_identity = _prepare_staging(staging)
    error_number = _renameat2_noreplace(staging, output)
    if error_number == 0:
        observed_root = output.stat(follow_symlinks=False)
        if (
            staging.exists()
            or output.is_symlink()
            or (observed_root.st_dev, observed_root.st_ino) != root_identity
            or stat.S_IMODE(observed_root.st_mode) != 0o555
            or _bundle_identity(output) != expected_identity
        ):
            raise RuntimeError("atomic training publication postcondition failed")
        _fsync_directory(output.parent)
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        _publish_by_links(staging, output, expected_identity=expected_identity)
        return
    os.chmod(staging, 0o755)
    raise OSError(error_number, os.strerror(error_number), str(output))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except FileNotFoundError:
        return
    shutil.rmtree(path)


def _read_regular_file_snapshot(
    path: Path,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular file: {source}")
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
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            stat.S_IMODE(value.st_mode),
        )
        for value in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError(f"input changed while being read: {source}")
    return payload, fingerprints.pop()


def _read_regular_bytes(path: Path) -> bytes:
    return _read_regular_file_snapshot(path)[0]


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


def _git_commit_tree(
    repository: Path,
    commit: str,
) -> tuple[tuple[str, str, str], ...]:
    raw = _run_git(repository, "ls-tree", "-rz", "--full-tree", "-r", commit)
    entries: list[tuple[str, str, str]] = []
    prior: str | None = None
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, name_bytes = record.split(b"\t", maxsplit=1)
            mode, kind, object_id = metadata.decode("ascii").split(" ")
            name = name_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git commit tree contains a malformed entry") from error
        pure = PurePosixPath(name)
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or _GIT_COMMIT_RE.fullmatch(object_id) is None
        ):
            raise ValueError("Git commit tree contains a non-regular entry")
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or "\n" in name
            or "\r" in name
            or pure.is_absolute()
            or pure.as_posix() != name
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError("Git commit tree contains an unsafe path")
        if prior is not None and name <= prior:
            raise ValueError("Git commit tree is unordered or duplicated")
        entries.append((name, mode, object_id))
        prior = name
    if not entries:
        raise ValueError("Git commit tree cannot be empty")
    return tuple(entries)


def _tracked_worktree_manifest(
    repository: Path,
    *,
    expected_commit: str,
) -> bytes:
    tree = _git_commit_tree(repository, expected_commit)
    raw_names = _run_git(repository, "ls-files", "-z")
    try:
        names = tuple(name.decode("utf-8") for name in raw_names.split(b"\0") if name)
    except UnicodeDecodeError as error:
        raise ValueError("tracked repository inventory is not UTF-8") from error
    tree_names = tuple(name for name, _mode, _object_id in tree)
    if names != tree_names:
        raise ValueError("tracked repository inventory differs from the expected commit tree")
    ignored_source_paths = _run_git(
        repository,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "src",
    )
    if _forbidden_ignored_source_paths(ignored_source_paths):
        raise ValueError("ignored source shadow artifacts are forbidden")

    entries: dict[str, str] = {}
    snapshots: dict[str, tuple[bytes, tuple[int, int, int, int, int, int]]] = {}
    for name, mode, object_id in tree:
        path = repository.joinpath(*PurePosixPath(name).parts)
        payload, fingerprint = _read_regular_file_snapshot(path)
        observed_mode = "100755" if fingerprint[-1] & stat.S_IXUSR else "100644"
        if observed_mode != mode:
            raise ValueError(f"tracked file mode differs from the expected commit: {name}")
        committed_payload = _run_git(repository, "cat-file", "blob", object_id)
        if payload != committed_payload:
            raise ValueError(f"tracked file bytes differ from the expected commit: {name}")
        entries[name] = hashlib.sha256(payload).hexdigest()
        snapshots[name] = (payload, fingerprint)

    if _run_git(repository, "ls-files", "-z") != raw_names:
        raise ValueError("tracked repository inventory changed during attestation")
    for name, _mode, _object_id in tree:
        current = _read_regular_file_snapshot(repository.joinpath(*PurePosixPath(name).parts))
        if current != snapshots[name]:
            raise ValueError(f"tracked file changed during repository attestation: {name}")
    ignored_source_paths_after = _run_git(
        repository,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "src",
    )
    if ignored_source_paths_after != ignored_source_paths:
        raise ValueError("ignored source inventory changed during repository attestation")
    return _sha_manifest_bytes(entries)


def _forbidden_ignored_source_paths(payload: bytes) -> tuple[str, ...]:
    forbidden: list[str] = []
    for encoded in payload.split(b"\0"):
        if not encoded:
            continue
        try:
            name = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ignored source inventory is not UTF-8") from error
        pure = PurePosixPath(name)
        is_cache = (
            len(pure.parts) >= 4
            and pure.parts[:2] == ("src", "amp_challenge")
            and pure.parent.name == "__pycache__"
            and pure.suffix in {".pyc", ".pyo"}
        )
        if not is_cache:
            forbidden.append(name)
    return tuple(forbidden)


def _repository_snapshot(
    repository_root: str | Path,
    *,
    expected_commit: str,
) -> tuple[str, bytes]:
    if type(expected_commit) is not str or _GIT_COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected_git_commit must be a lowercase Git object ID")
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
        raise ValueError("repository_root must be a directory")
    top_level = Path(
        os.path.abspath(
            _run_git(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        )
    )
    if top_level != repository:
        raise ValueError("repository_root must be the exact Git worktree top level")
    if _run_git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("repository-local Git replacement refs are forbidden")
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
    if set(observed.values()) != {expected_commit}:
        raise ValueError(
            "repository HEAD, upstream, and cached origin/main must equal expected_git_commit"
        )
    if _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError("repository worktree must be clean before training")
    code_manifest = _tracked_worktree_manifest(repository, expected_commit=expected_commit)
    if _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError("repository changed while its code manifest was computed")
    return observed["HEAD"], code_manifest


def _validate_production_environment(
    contract: NativeDiffusionContract,
    device: torch.device,
) -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract.determinism.cublas_workspace_config:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be exported before any CUDA inspection"
        )
    if os.environ.get("PYTORCH_ALLOC_CONF") != contract.determinism.pytorch_allocator:
        raise RuntimeError(
            "PYTORCH_ALLOC_CONF=backend:native must be exported before any CUDA inspection"
        )
    legacy_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_allocator not in (None, contract.determinism.pytorch_allocator):
        raise RuntimeError("legacy PyTorch allocator configuration conflicts with the contract")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"):
        raise RuntimeError("the production contract prohibits disabling CUDA memory caching")
    if device != torch.device("cuda:0"):
        raise RuntimeError("the production contract requires the bound CUDA device cuda:0")

    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != contract.environment.python:
        raise RuntimeError("Python version differs from the production contract")
    if np.__version__ != contract.environment.numpy:
        raise RuntimeError("NumPy version differs from the production contract")
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if torch_version != contract.environment.torch:
        raise RuntimeError("PyTorch version differs from the production contract")
    if torch.version.cuda != contract.environment.torch_cuda:
        raise RuntimeError("PyTorch CUDA runtime differs from the production contract")
    package_versions = {
        "safetensors": _optional_distribution_version("safetensors"),
        "triton": _optional_distribution_version("triton"),
        "nvidia_cudnn_cu13": _optional_distribution_version("nvidia-cudnn-cu13"),
    }
    required_packages = {
        "safetensors": contract.environment.safetensors,
        "triton": contract.environment.triton,
        "nvidia_cudnn_cu13": contract.environment.nvidia_cudnn_cu13,
    }
    if package_versions != required_packages:
        raise RuntimeError("installed CUDA-library package versions differ from the contract")

    # All environment and package checks above are intentionally CUDA-free.
    if not torch.cuda.is_available():
        raise RuntimeError("the production contract requires one visible CUDA GPU")
    if torch.cuda.device_count() != contract.compute.gpus_per_job:
        raise RuntimeError("visible CUDA device count differs from the production contract")
    index = device.index
    if index is None:  # Defensive: the exact cuda:0 check above makes this unreachable.
        raise RuntimeError("the production CUDA device index is unavailable")
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    if properties.name != contract.environment.gpu_name:
        raise RuntimeError("CUDA device name differs from the production contract")
    if capability != contract.environment.compute_capability:
        raise RuntimeError("CUDA compute capability differs from the production contract")
    try:
        allocator_backend = torch.cuda.memory.get_allocator_backend()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("cannot verify the active PyTorch CUDA allocator") from error
    allocator = f"backend:{allocator_backend}"
    if allocator != contract.determinism.pytorch_allocator:
        raise RuntimeError("active PyTorch CUDA allocator differs from the production contract")
    driver = _cuda_driver_version()
    return {
        "python": python_version,
        "numpy": np.__version__,
        "torch": torch_version,
        "torch_cuda": torch.version.cuda,
        "safetensors": package_versions["safetensors"],
        "triton": package_versions["triton"],
        "nvidia_cudnn_cu13": package_versions["nvidia_cudnn_cu13"],
        "gpu_name": properties.name,
        "compute_capability": list(capability),
        "driver": driver,
        "allocator": allocator,
    }


def _cuda_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        lines = completed.stdout.decode("ascii").strip().splitlines()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise RuntimeError("cannot obtain the CUDA driver version from nvidia-smi") from error
    if len(lines) != 1 or _DRIVER_VERSION_RE.fullmatch(lines[0]) is None:
        raise RuntimeError("nvidia-smi did not report one dotted-numeric CUDA driver version")
    return lines[0]


def _validate_production_output(
    output_dir: str | Path,
    contract: NativeDiffusionContract,
    *,
    repository: Path,
    contract_path: Path,
    training_projection_path: Path,
) -> Path:
    variable = contract.compute.scratch_root_env
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"required scratch environment variable is unset: {variable}")
    scratch = Path(value)
    if not scratch.is_absolute():
        raise ValueError(f"{variable} must name an absolute path")
    scratch = Path(os.path.abspath(os.fspath(scratch)))
    _reject_symlink_chain(scratch)
    base = scratch / contract.compute.run_subdir
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if output == base or base not in output.parents:
        raise ValueError("production output must be a named run below the contract scratch root")
    protected = (repository, contract_path, training_projection_path)
    if any(
        output == path or output in path.parents or path in output.parents for path in protected
    ):
        raise ValueError("training output overlaps a protected input or repository path")
    return output


def train_unconditional_v0(
    *,
    contract_path: str | Path,
    training_projection_path: str | Path,
    output_dir: str | Path,
    seed: int,
    repository_root: str | Path,
    expected_git_commit: str | None = None,
) -> TrainingRunResult:
    """Train exactly one production v0 seed from the two pinned inputs."""

    expected = (
        os.environ.get("AMP_EXPECTED_GIT_COMMIT")
        if expected_git_commit is None
        else expected_git_commit
    )
    if type(expected) is not str or _GIT_COMMIT_RE.fullmatch(expected) is None:
        raise ValueError(
            "AMP_EXPECTED_GIT_COMMIT or expected_git_commit must be a lowercase Git object ID"
        )
    contract_source = Path(os.path.abspath(os.fspath(contract_path)))
    projection_source = Path(os.path.abspath(os.fspath(training_projection_path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    contract = load_unconditional_v0_contract(contract_source)
    plan = training_plan_from_contract(contract)
    if type(seed) is not int or seed not in plan.seeds:
        raise ValueError("seed is not one of the three preregistered training seeds")
    output = _validate_production_output(
        output_dir,
        contract,
        repository=repository,
        contract_path=contract_source,
        training_projection_path=projection_source,
    )
    git_commit, code_manifest = _repository_snapshot(
        repository,
        expected_commit=expected,
    )
    target_device = torch.device("cuda:0")
    environment_identity = _validate_production_environment(contract, target_device)
    contract_payload = _read_regular_bytes(contract_source)
    if hashlib.sha256(contract_payload).hexdigest() != plan.config_sha256:
        raise ValueError("contract changed after strict parsing")
    distribution = load_training_projection(
        projection_source,
        expected_sha256=plan.training_projection_sha256,
        expected_rows=plan.expected_train_sequences,
    )
    input_manifest = _sha_manifest_bytes(
        {
            "contract.toml": plan.config_sha256,
            "training_projection.jsonl": plan.training_projection_sha256,
        }
    )
    provenance = TrainingProvenance(
        contract_payload=contract_payload,
        code_manifest_payload=code_manifest,
        input_manifest_payload=input_manifest,
        git_commit=git_commit,
    )

    def prepublish_check() -> None:
        final_commit, final_code_manifest = _repository_snapshot(
            repository,
            expected_commit=expected,
        )
        if final_commit != git_commit or final_code_manifest != code_manifest:
            raise RuntimeError("repository changed while training")
        if hashlib.sha256(_read_regular_bytes(contract_source)).hexdigest() != plan.config_sha256:
            raise RuntimeError("contract changed while training")
        if (
            hashlib.sha256(_read_regular_bytes(projection_source)).hexdigest()
            != plan.training_projection_sha256
        ):
            raise RuntimeError("training projection changed while training")

    return _execute_training_plan(
        plan=plan,
        distribution=distribution,
        provenance=provenance,
        output_dir=output,
        seed=seed,
        device=target_device,
        require_cuda=True,
        production_environment_identity=environment_identity,
        prepublish_check=prepublish_check,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one byte-pinned native unconditional categorical diffusion v0 seed.",
        allow_abbrev=False,
    )
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--training-projection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--expected-git-commit",
        default=None,
        help="defaults to AMP_EXPECTED_GIT_COMMIT",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = train_unconditional_v0(
        contract_path=arguments.contract,
        training_projection_path=arguments.training_projection,
        output_dir=arguments.output,
        seed=arguments.seed,
        repository_root=arguments.repository_root,
        expected_git_commit=arguments.expected_git_commit,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "checkpoint_file_sha256": result.checkpoint_hashes.file_sha256,
                "checkpoint_logical_state_sha256": (result.checkpoint_hashes.logical_state_sha256),
                "final_loss": result.final_loss,
                "mean_loss": result.mean_loss,
                "steps": result.steps,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())

"""Checkpoint-bound PyTorch inference for native categorical diffusion v0.

The public loader accepts a sealed training bundle, the byte-pinned v0
contract, and a synchronized repository checkout.  It never accepts a model,
callback, device override, or caller-asserted checkpoint digest.  A small
private plan and loader exist solely so unit tests can exercise the same bundle
and tensor boundary on CPU without weakening the production API.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .contract import NativeDiffusionContract, load_unconditional_v0_contract
from .data import namespaced_seed
from .model import (
    MASK_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
    CheckpointHashes,
    NativeDenoiser,
    NativeDenoiserConfig,
    configure_deterministic_runtime,
    load_safetensors_checkpoint,
)

MAX_INFERENCE_BATCH_SEQUENCES = 256
PRODUCTION_SEQUENCE_WIDTH = 50

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
_MODEL_LOGICAL_BINDING_DOMAIN = (
    b"amp-challenge/native-categorical-diffusion/checkpoint-contract-binding/v1\0"
)
_MAX_CHECKPOINT_BYTES = 1 << 30
_MAX_METADATA_BYTES = 128 << 20
_CONSTRUCTION_TOKEN = object()
_NATIVE_FORWARD = NativeDenoiser.forward
_TWIN_ENVIRONMENT_EQUAL_FIELDS = (
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
_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        *_TWIN_ENVIRONMENT_EQUAL_FIELDS,
        "device_type",
        "twin_environment_equal_fields",
        "runtime",
        "amp",
        "tf32",
        "torch_compile",
    }
)

_CORPUS_FIELDS = frozenset(
    {
        "accepted_parent_sha256",
        "training_projection_sha256",
        "trainer_visible_sequences",
        "trainer_visible_fields",
        "roles",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "config",
        "trainable_parameters",
        "checkpoint_file_sha256",
        "checkpoint_logical_state_sha256",
        "checkpoint_format",
    }
)
_RNG_LINK_FIELDS = frozenset({"filename", "sha256"})
_TRAINING_FIELDS = frozenset(
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
    }
)
_RNG_FIELDS = frozenset(
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
    }
)
_METRIC_FIELDS = frozenset(
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


@dataclass(frozen=True, slots=True)
class _RuntimePins:
    python: str
    numpy: str
    torch: str
    torch_cuda: str
    safetensors: str
    triton: str
    nvidia_cudnn_cu13: str
    gpu_name: str
    compute_capability: tuple[int, int]
    allocator: str
    cublas_workspace_config: str
    gpus_per_job: int


@dataclass(frozen=True, slots=True)
class _ResolvedInferencePlan:
    artifact: str
    config_sha256: str
    contract_payload: bytes
    corpus_sha256: str
    training_projection_sha256: str
    expected_train_sequences: int
    model_config: NativeDenoiserConfig
    expected_trainable_parameters: int
    seeds: tuple[int, ...]
    max_steps: int
    training_batch_sequences: int
    learning_rate: float
    warmup_steps: int
    final_learning_rate: float
    training_log_interval_steps: int
    maximum_peak_gpu_memory_gib: float
    schedule_sha256: str
    rng_namespaces: tuple[str, ...]
    bundle_files: tuple[str, ...]
    manifest_fields: tuple[str, ...]
    runtime_pins: _RuntimePins | None
    production: bool


@dataclass(frozen=True, slots=True)
class _TestInferencePlan:
    """Explicit test-only injection contract for a tiny CPU bundle."""

    artifact: str
    contract_payload: bytes
    corpus_sha256: str
    training_projection_sha256: str
    expected_train_sequences: int
    model_config: NativeDenoiserConfig
    expected_trainable_parameters: int
    seeds: tuple[int, ...]
    max_steps: int
    training_batch_sequences: int
    learning_rate: float
    warmup_steps: int
    final_learning_rate: float
    training_log_interval_steps: int
    maximum_peak_gpu_memory_gib: float
    schedule_sha256: str
    rng_namespaces: tuple[str, ...]
    bundle_files: tuple[str, ...]
    manifest_fields: tuple[str, ...]
    expected_git_commit: str

    def resolved(self) -> _ResolvedInferencePlan:
        if type(self.contract_payload) is not bytes or not self.contract_payload:
            raise ValueError("test contract_payload must be non-empty bytes")
        return _ResolvedInferencePlan(
            artifact=self.artifact,
            config_sha256=hashlib.sha256(self.contract_payload).hexdigest(),
            contract_payload=self.contract_payload,
            corpus_sha256=_sha256(self.corpus_sha256, label="test corpus_sha256"),
            training_projection_sha256=_sha256(
                self.training_projection_sha256,
                label="test training_projection_sha256",
            ),
            expected_train_sequences=self.expected_train_sequences,
            model_config=self.model_config,
            expected_trainable_parameters=self.expected_trainable_parameters,
            seeds=self.seeds,
            max_steps=self.max_steps,
            training_batch_sequences=self.training_batch_sequences,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            final_learning_rate=self.final_learning_rate,
            training_log_interval_steps=self.training_log_interval_steps,
            maximum_peak_gpu_memory_gib=self.maximum_peak_gpu_memory_gib,
            schedule_sha256=_sha256(self.schedule_sha256, label="test schedule_sha256"),
            rng_namespaces=self.rng_namespaces,
            bundle_files=self.bundle_files,
            manifest_fields=self.manifest_fields,
            runtime_pins=None,
            production=False,
        )


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _BundleSnapshot:
    root: Path
    root_fingerprint: tuple[int, int, int, int, int, int]
    files: Mapping[str, _FileSnapshot]


@dataclass(frozen=True, slots=True)
class _VerifiedManifest:
    seed: int
    git_commit: str
    checkpoint_file_sha256: str
    checkpoint_model_logical_sha256: str
    initialization_seed: int
    training_driver: str | None


@dataclass(frozen=True, slots=True)
class _VerifiedProviderState:
    model: NativeDenoiser
    device: torch.device
    config_sha256: str
    checkpoint_hashes: CheckpointHashes
    seed: int
    git_commit: str


class NativeV0LogitProvider:
    """A sealed checkpoint's only production NumPy inference boundary."""

    __slots__ = (
        "__checkpoint_file_sha256",
        "__checkpoint_logical_sha256",
        "__checkpoint_model_logical_sha256",
        "__config_sha256",
        "__device",
        "__git_commit",
        "__model",
        "__model_config",
        "__parameter_guard",
        "__seed",
    )

    def __init__(self) -> None:
        raise TypeError("NativeV0LogitProvider must be constructed by a verified bundle loader")

    def __initialize(
        self,
        construction_token: object,
        *,
        state: _VerifiedProviderState,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("NativeV0LogitProvider must be constructed by a verified bundle loader")
        if type(state) is not _VerifiedProviderState:
            raise TypeError("provider state must originate from verified bundle loading")
        model = state.model
        device = state.device
        if type(model) is not NativeDenoiser or type(model).forward is not _NATIVE_FORWARD:
            raise TypeError("inference requires the exact NativeDenoiser implementation")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        config_digest = _sha256(state.config_sha256, label="config_sha256")
        file_digest = _sha256(
            state.checkpoint_hashes.file_sha256,
            label="checkpoint file SHA-256",
        )
        model_digest = _sha256(
            state.checkpoint_hashes.logical_state_sha256,
            label="checkpoint model logical SHA-256",
        )
        if type(state.seed) is not int or state.seed < 0 or state.seed >= 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if _GIT_COMMIT_RE.fullmatch(state.git_commit) is None:
            raise ValueError("git_commit must be a lowercase forty-character object ID")
        model.eval()
        model.requires_grad_(False)
        self.__model = model
        self.__device = device
        self.__model_config = model.config
        self.__config_sha256 = config_digest
        self.__checkpoint_file_sha256 = file_digest
        self.__checkpoint_model_logical_sha256 = model_digest
        self.__checkpoint_logical_sha256 = _bind_checkpoint_to_contract(
            config_digest,
            model_digest,
        )
        self.__seed = state.seed
        self.__git_commit = state.git_commit
        self.__parameter_guard = _parameter_guard(model)
        self._assert_model_is_inference_only()

    @property
    def checkpoint_logical_sha256(self) -> str:
        """Config-bound logical identity to record in sampling artifacts."""

        return self.__checkpoint_logical_sha256

    @property
    def checkpoint_model_logical_sha256(self) -> str:
        """Logical model/state identity stored by the training manifest."""

        return self.__checkpoint_model_logical_sha256

    @property
    def checkpoint_file_sha256(self) -> str:
        return self.__checkpoint_file_sha256

    @property
    def config_sha256(self) -> str:
        return self.__config_sha256

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def git_commit(self) -> str:
        return self.__git_commit

    def _assert_model_is_inference_only(self) -> None:
        model = self.__model
        if type(model) is not NativeDenoiser or type(model).forward is not _NATIVE_FORWARD:
            raise RuntimeError("verified native denoiser implementation was substituted")
        for module in model.modules():
            if "forward" in module.__dict__:
                raise RuntimeError("verified native denoiser forward method was overridden")
            if any(
                getattr(module, name)
                for name in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                )
            ):
                raise RuntimeError("callbacks cannot be attached to the verified native denoiser")
        if model.config != self.__model_config:
            raise RuntimeError("verified native denoiser configuration changed")
        if model.training:
            raise RuntimeError("verified native denoiser must remain in eval mode")
        parameters = tuple(model.parameters())
        if not parameters or any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("verified native denoiser parameters must remain frozen")
        if any(
            parameter.dtype != torch.float32 or parameter.device != self.__device
            for parameter in parameters
        ):
            raise RuntimeError("verified native denoiser dtype or device changed")
        if _parameter_guard(model) != self.__parameter_guard:
            raise RuntimeError("verified checkpoint parameters were replaced or modified")
        _validate_active_runtime_controls()

    def __call__(
        self,
        tokens: NDArray[np.int64],
        attention_mask: NDArray[np.bool_],
        levels: NDArray[np.int64],
        lengths: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Return finite CPU FP32 logits for one canonical width-50 batch."""

        self._assert_model_is_inference_only()
        token_values, mask_values, level_values, length_values = _validated_numpy_inputs(
            tokens,
            attention_mask,
            levels,
            lengths,
            config=self.__model_config,
        )
        token_tensor = torch.from_numpy(token_values).to(self.__device)
        mask_tensor = torch.from_numpy(mask_values).to(self.__device)
        level_tensor = torch.from_numpy(level_values).to(self.__device)
        length_tensor = torch.from_numpy(length_values).to(self.__device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.__device.type,
                enabled=False,
            ),
        ):
            logits = self.__model(
                token_tensor,
                mask_tensor,
                level_tensor,
                length_tensor,
            )
        if logits.dtype != torch.float32 or logits.device != self.__device or logits.requires_grad:
            raise RuntimeError(
                "native denoiser returned logits with an invalid dtype, device, or graph"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("native denoiser returned non-finite logits")
        result = logits.detach().cpu().contiguous().numpy().copy()
        expected_shape = (len(token_values), PRODUCTION_SEQUENCE_WIDTH, 20)
        if result.dtype != np.float32 or result.shape != expected_shape:
            raise RuntimeError("native denoiser returned a noncanonical NumPy result")
        result.flags.writeable = False
        self._assert_model_is_inference_only()
        return cast(NDArray[np.float32], result)


def load_native_v0_logit_provider(
    *,
    training_bundle: str | Path,
    contract_path: str | Path,
    repository_root: str | Path,
    expected_git_commit: str,
) -> NativeV0LogitProvider:
    """Load the exact production v0 checkpoint on the sole visible CUDA GPU.

    There is intentionally no device, model, callback, or checkpoint-hash
    argument.  All checkpoint identities originate inside the verified sealed
    bundle, and production CPU inference cannot be requested through this API.
    """

    if (
        type(expected_git_commit) is not str
        or _GIT_COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise ValueError("expected_git_commit must be a lowercase Git object ID")
    contract_source = Path(os.path.abspath(os.fspath(contract_path)))
    contract = load_unconditional_v0_contract(contract_source)
    contract_payload = _read_regular_file(contract_source, label="external contract")
    if hashlib.sha256(contract_payload).hexdigest() != contract.config_sha256:
        raise ValueError("external contract changed after strict parsing")
    plan = _plan_from_contract(contract, contract_payload)
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    code_manifest = _repository_code_manifest(
        repository,
        expected_git_commit=expected_git_commit,
        contract_path=contract_source,
        config_sha256=plan.config_sha256,
    )
    return _load_verified_provider(
        training_bundle=training_bundle,
        plan=plan,
        device=torch.device("cuda:0"),
        expected_git_commit=expected_git_commit,
        expected_code_manifest=code_manifest,
        require_cuda=True,
    )


def _load_native_v0_logit_provider_for_test(
    *,
    training_bundle: str | Path,
    plan: _TestInferencePlan,
) -> NativeV0LogitProvider:
    """Private CPU-capable loader for tiny immutable unit-test plans only."""

    if type(plan) is not _TestInferencePlan:
        raise TypeError("test loader requires an explicit _TestInferencePlan")
    expected_git_commit = _git_commit(
        plan.expected_git_commit,
        label="test expected_git_commit",
    )
    return _load_verified_provider(
        training_bundle=training_bundle,
        plan=plan.resolved(),
        device=torch.device("cpu"),
        expected_git_commit=expected_git_commit,
        expected_code_manifest=None,
        require_cuda=False,
    )


def _plan_from_contract(
    contract: NativeDiffusionContract,
    contract_payload: bytes,
) -> _ResolvedInferencePlan:
    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
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
    if model_config.max_length != PRODUCTION_SEQUENCE_WIDTH:
        raise ValueError("production native-v0 inference requires width 50")
    if len(contract.environment.compute_capability) != 2:
        raise ValueError("production compute capability must contain major and minor")
    schedule_sha256 = _schedule_digest(
        max_steps=contract.training.max_steps,
        warmup_steps=contract.training.warmup_steps,
        learning_rate=contract.training.learning_rate,
        final_learning_rate=contract.training.final_learning_rate,
    )
    return _ResolvedInferencePlan(
        artifact=contract.artifact,
        config_sha256=contract.config_sha256,
        contract_payload=contract_payload,
        corpus_sha256=contract.input.corpus_sha256,
        training_projection_sha256=contract.input.training_projection_sha256,
        expected_train_sequences=contract.input.expected_train_sequences,
        model_config=model_config,
        expected_trainable_parameters=contract.model.expected_trainable_parameters,
        seeds=contract.training.seeds,
        max_steps=contract.training.max_steps,
        training_batch_sequences=contract.training.batch_sequences,
        learning_rate=contract.training.learning_rate,
        warmup_steps=contract.training.warmup_steps,
        final_learning_rate=contract.training.final_learning_rate,
        training_log_interval_steps=contract.training.training_log_interval_steps,
        maximum_peak_gpu_memory_gib=contract.compute.maximum_peak_gpu_memory_gib,
        schedule_sha256=schedule_sha256,
        rng_namespaces=contract.determinism.rng_namespaces,
        bundle_files=contract.artifacts.training_bundle_files,
        manifest_fields=contract.artifacts.training_manifest_fields,
        runtime_pins=_RuntimePins(
            python=contract.environment.python,
            numpy=contract.environment.numpy,
            torch=contract.environment.torch,
            torch_cuda=contract.environment.torch_cuda,
            safetensors=contract.environment.safetensors,
            triton=contract.environment.triton,
            nvidia_cudnn_cu13=contract.environment.nvidia_cudnn_cu13,
            gpu_name=contract.environment.gpu_name,
            compute_capability=(
                contract.environment.compute_capability[0],
                contract.environment.compute_capability[1],
            ),
            allocator=contract.determinism.pytorch_allocator,
            cublas_workspace_config=contract.determinism.cublas_workspace_config,
            gpus_per_job=contract.compute.gpus_per_job,
        ),
        production=True,
    )


def _load_verified_provider(
    *,
    training_bundle: str | Path,
    plan: _ResolvedInferencePlan,
    device: torch.device,
    expected_git_commit: str | None,
    expected_code_manifest: bytes | None,
    require_cuda: bool,
) -> NativeV0LogitProvider:
    if not isinstance(plan, _ResolvedInferencePlan):
        raise TypeError("plan must be a resolved inference plan")
    if require_cuda != plan.production:
        raise ValueError("production and CUDA requirements cannot be weakened")
    if require_cuda and device != torch.device("cuda:0"):
        raise ValueError("production inference requires the sole visible CUDA device")
    if not require_cuda and device.type != "cpu":
        raise ValueError("test inference requires CPU")

    snapshot = _snapshot_bundle(training_bundle, plan.bundle_files)
    verified = _verify_bundle(
        snapshot,
        plan,
        expected_git_commit=expected_git_commit,
        expected_code_manifest=expected_code_manifest,
    )

    if plan.production:
        pins = cast(_RuntimePins, plan.runtime_pins)
        _validate_runtime_environment_before_cuda(pins)
    runtime = configure_deterministic_runtime(verified.initialization_seed)
    _validate_runtime_controls(runtime, seed=verified.initialization_seed)
    _validate_active_runtime_controls()
    if require_cuda:
        _validate_cuda_environment(
            cast(_RuntimePins, plan.runtime_pins),
            device,
            expected_driver=cast(str, verified.training_driver),
        )

    model = NativeDenoiser(plan.model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != plan.expected_trainable_parameters:
        raise ValueError(
            "native denoiser parameter count differs from the sealed training manifest"
        )
    checkpoint_path = snapshot.root / "model_final.safetensors"
    checkpoint_hashes = load_safetensors_checkpoint(
        model,
        checkpoint_path,
        expected_file_sha256=verified.checkpoint_file_sha256,
        expected_logical_state_sha256=verified.checkpoint_model_logical_sha256,
    )
    if checkpoint_hashes != CheckpointHashes(
        file_sha256=verified.checkpoint_file_sha256,
        logical_state_sha256=verified.checkpoint_model_logical_sha256,
    ):
        raise RuntimeError("checkpoint loader returned an identity different from the manifest")
    _assert_snapshot_unchanged(snapshot, plan.bundle_files)
    model.to(device=device, dtype=torch.float32)
    provider_state = _VerifiedProviderState(
        model=model,
        device=device,
        config_sha256=plan.config_sha256,
        checkpoint_hashes=checkpoint_hashes,
        seed=verified.seed,
        git_commit=verified.git_commit,
    )
    return _construct_native_v0_provider(provider_state)


def _construct_native_v0_provider(state: _VerifiedProviderState) -> NativeV0LogitProvider:
    provider = object.__new__(NativeV0LogitProvider)
    provider._NativeV0LogitProvider__initialize(
        _CONSTRUCTION_TOKEN,
        state=state,
    )
    return provider


def _verify_bundle(
    snapshot: _BundleSnapshot,
    plan: _ResolvedInferencePlan,
    *,
    expected_git_commit: str | None,
    expected_code_manifest: bytes | None,
) -> _VerifiedManifest:
    files = snapshot.files
    if files["contract.toml"].payload != plan.contract_payload:
        raise ValueError("training bundle contract differs from the exact external contract")
    if files["contract.toml"].sha256 != plan.config_sha256:
        raise ValueError("training bundle contract SHA-256 differs from the plan")

    manifest = _json_object(files["manifest.json"].payload, label="training manifest")
    if set(manifest) != set(plan.manifest_fields):
        raise ValueError("training manifest top-level schema differs from the contract")
    _expect(manifest, "schema_version", 1, label="training manifest")
    _expect(manifest, "artifact", plan.artifact, label="training manifest")
    _expect(manifest, "config_sha256", plan.config_sha256, label="training manifest")
    git_commit = _git_commit(manifest.get("git_commit"), label="training manifest git_commit")
    if expected_git_commit is not None and git_commit != expected_git_commit:
        raise ValueError("training manifest git_commit differs from the synchronized checkout")
    seed = _integer(manifest.get("seed"), label="training manifest seed", minimum=0)
    if seed not in plan.seeds:
        raise ValueError("training manifest seed is not preregistered")

    artifact_hashes = _mapping(manifest.get("artifacts"), label="training manifest artifacts")
    expected_artifact_names = set(plan.bundle_files) - {"manifest.json"}
    if set(artifact_hashes) != expected_artifact_names:
        raise ValueError("training manifest artifact inventory differs from the contract")
    for name in sorted(expected_artifact_names):
        digest = _sha256(artifact_hashes[name], label=f"artifact hash for {name}")
        if digest != files[name].sha256:
            raise ValueError(f"training artifact hash mismatch for {name}")

    if expected_code_manifest is not None and files["CODE_SHA256SUMS"].payload != (
        expected_code_manifest
    ):
        raise ValueError("training CODE_SHA256SUMS differs from the synchronized checkout")
    _parse_sha_manifest(files["CODE_SHA256SUMS"].payload, label="CODE_SHA256SUMS")
    input_entries = _parse_sha_manifest(
        files["INPUT_SHA256SUMS"].payload,
        label="INPUT_SHA256SUMS",
    )
    if input_entries != {
        "contract.toml": plan.config_sha256,
        "training_projection.jsonl": plan.training_projection_sha256,
    }:
        raise ValueError("INPUT_SHA256SUMS does not bind the exact trainer inputs")

    corpus = _exact_mapping(manifest.get("corpus"), _CORPUS_FIELDS, label="manifest corpus")
    _expect(corpus, "accepted_parent_sha256", plan.corpus_sha256, label="manifest corpus")
    _expect(
        corpus,
        "training_projection_sha256",
        plan.training_projection_sha256,
        label="manifest corpus",
    )
    _expect(
        corpus,
        "trainer_visible_sequences",
        plan.expected_train_sequences,
        label="manifest corpus",
    )
    _expect(
        corpus,
        "trainer_visible_fields",
        ["sequence_id", "sequence", "sampling_weight"],
        label="manifest corpus",
    )
    _expect(corpus, "roles", ["train"], label="manifest corpus")

    model = _exact_mapping(manifest.get("model"), _MODEL_FIELDS, label="manifest model")
    expected_model_config = asdict(plan.model_config)
    _expect(model, "config", expected_model_config, label="manifest model")
    _expect(
        model,
        "trainable_parameters",
        plan.expected_trainable_parameters,
        label="manifest model",
    )
    _expect(model, "checkpoint_format", "safetensors", label="manifest model")
    checkpoint_file_sha256 = _sha256(
        model.get("checkpoint_file_sha256"),
        label="manifest checkpoint file SHA-256",
    )
    checkpoint_model_logical_sha256 = _sha256(
        model.get("checkpoint_logical_state_sha256"),
        label="manifest checkpoint logical SHA-256",
    )
    if checkpoint_file_sha256 != files["model_final.safetensors"].sha256:
        raise ValueError("checkpoint physical SHA-256 is not linked to its bundle artifact")

    rng_link = _exact_mapping(manifest.get("rng"), _RNG_LINK_FIELDS, label="manifest rng")
    _expect(rng_link, "filename", "rng.json", label="manifest rng")
    _expect(rng_link, "sha256", files["rng.json"].sha256, label="manifest rng")
    initialization_seed = _verify_rng(files["rng.json"].payload, plan, seed=seed)

    training = _exact_mapping(
        manifest.get("training"),
        _TRAINING_FIELDS,
        label="manifest training",
    )
    _expect(training, "steps", plan.max_steps, label="manifest training")
    _expect(
        training,
        "batch_sequences",
        plan.training_batch_sequences,
        label="manifest training",
    )
    _expect(training, "optimizer", "adamw_unfused", label="manifest training")
    _expect(training, "schedule_sha256", plan.schedule_sha256, label="manifest training")
    _expect(
        training,
        "checkpoint_selection",
        f"final_step_{plan.max_steps}_only",
        label="manifest training",
    )
    for name in ("validation_during_training", "early_stopping", "resume_supported"):
        _expect(training, name, False, label="manifest training")
    _expect(
        training,
        "metrics_sha256",
        files["train_metrics.json"].sha256,
        label="manifest training",
    )
    _verify_parameter_partition(training, plan)

    schedule_payload = files["training_schedule.sha256"].payload
    if schedule_payload != f"{plan.schedule_sha256}\n".encode("ascii"):
        raise ValueError("training_schedule.sha256 differs from the contract-derived schedule")
    metrics = _verify_training_metrics(files["train_metrics.json"].payload, plan)
    training_driver = _verify_environment(
        files["environment.json"].payload,
        plan,
        initialization_seed=initialization_seed,
    )
    _verify_training_trace(
        files["training_trace.jsonl"].payload,
        plan,
        seed=seed,
        metrics=metrics,
    )
    return _VerifiedManifest(
        seed=seed,
        git_commit=git_commit,
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_model_logical_sha256=checkpoint_model_logical_sha256,
        initialization_seed=initialization_seed,
        training_driver=training_driver,
    )


def _verify_rng(payload: bytes, plan: _ResolvedInferencePlan, *, seed: int) -> int:
    document = _json_object(payload, label="rng.json")
    if set(document) != _RNG_FIELDS:
        raise ValueError("rng.json schema differs from the v0 contract")
    initialization_seed = namespaced_seed(seed, "initialization", "model")
    expected = {
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
    if document != expected:
        raise ValueError("rng.json differs from the deterministic training contract")
    return initialization_seed


def _verify_training_metrics(
    payload: bytes,
    plan: _ResolvedInferencePlan,
) -> dict[str, object]:
    document = _json_object(payload, label="train_metrics.json")
    if set(document) != _METRIC_FIELDS:
        raise ValueError("train_metrics.json schema differs from the v0 producer")
    _expect(document, "schema_version", 1, label="train_metrics.json")
    _expect(document, "steps", plan.max_steps, label="train_metrics.json")
    _expect(
        document,
        "drawn_sequences",
        plan.max_steps * plan.training_batch_sequences,
        label="train_metrics.json",
    )
    _expect(
        document,
        "loss_reduction",
        "masked_mean_per_sequence_then_batch_mean",
        label="train_metrics.json",
    )
    _expect(
        document,
        "sampling_weight_application",
        "weighted_draw_only",
        label="train_metrics.json",
    )
    _expect(document, "validation_consulted", False, label="train_metrics.json")
    for name in (
        "final_loss",
        "mean_loss",
        "final_mean_row_accuracy",
        "mean_row_accuracy",
        "mean_gradient_norm_before_clipping",
        "final_learning_rate",
    ):
        value = _finite_float(document.get(name), label=f"train_metrics.json {name}")
        if value < 0.0:
            raise ValueError(f"train_metrics.json {name} cannot be negative")
    for name in ("final_mean_row_accuracy", "mean_row_accuracy"):
        value = cast(float, document[name])
        if value > 1.0:
            raise ValueError(f"train_metrics.json {name} must lie in [0, 1]")
    _expect(
        document,
        "final_learning_rate",
        _learning_rate_for_step(plan, plan.max_steps),
        label="train_metrics.json",
    )
    _integer(
        document.get("total_selected_tokens"),
        label="train_metrics.json total_selected_tokens",
        minimum=1,
    )
    peak = _integer(
        document.get("peak_gpu_memory_bytes"),
        label="train_metrics.json peak_gpu_memory_bytes",
        minimum=0,
    )
    if peak > int(plan.maximum_peak_gpu_memory_gib * 1024**3):
        raise ValueError("train_metrics.json exceeds the preregistered GPU-memory cap")
    return document


def _verify_training_trace(
    payload: bytes,
    plan: _ResolvedInferencePlan,
    *,
    seed: int,
    metrics: Mapping[str, object],
) -> None:
    records = _canonical_jsonl(payload, label="training_trace.jsonl")
    expected_steps = tuple(
        step
        for step in range(1, plan.max_steps + 1)
        if step == 1 or step % plan.training_log_interval_steps == 0 or step == plan.max_steps
    )
    if tuple(record.get("step") for record in records) != expected_steps:
        raise ValueError("training trace step ledger differs from the v0 producer")
    for number, (record, step) in enumerate(
        zip(records, expected_steps, strict=True),
        1,
    ):
        expected_fields = {
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
        if set(record) != expected_fields:
            raise ValueError(f"training trace row {number} has the wrong schema")
        _expect(record, "schema_version", 1, label=f"training trace row {number}")
        _expect(record, "step", step, label=f"training trace row {number}")
        _integer(
            record.get("selected_tokens"),
            label=f"training trace row {number} selected_tokens",
            minimum=1,
        )
        dropout_seed = _integer(
            record.get("dropout_seed"),
            label=f"training trace row {number} dropout_seed",
            minimum=0,
        )
        if dropout_seed >= 2**64:
            raise ValueError("training trace dropout_seed exceeds uint64")
        if dropout_seed != namespaced_seed(seed, "dropout", step):
            raise ValueError("training trace dropout_seed differs from the v0 derivation")
        _sha256(record.get("batch_sha256"), label=f"training trace row {number} batch")
        for name in (
            "loss",
            "mean_row_accuracy",
            "gradient_norm_before_clipping",
            "learning_rate",
        ):
            value = _finite_float(
                record.get(name),
                label=f"training trace row {number} {name}",
            )
            if value < 0.0:
                raise ValueError(f"training trace row {number} {name} cannot be negative")
        if cast(float, record["mean_row_accuracy"]) > 1.0:
            raise ValueError(f"training trace row {number} accuracy must lie in [0, 1]")
        _expect(
            record,
            "learning_rate",
            _learning_rate_for_step(plan, step),
            label=f"training trace row {number}",
        )
    final = records[-1]
    _expect(final, "loss", metrics["final_loss"], label="final training trace row")
    _expect(
        final,
        "mean_row_accuracy",
        metrics["final_mean_row_accuracy"],
        label="final training trace row",
    )


def _learning_rate_for_step(plan: _ResolvedInferencePlan, step: int) -> float:
    if type(step) is not int or not 1 <= step <= plan.max_steps:
        raise ValueError("step must be an integer in 1..max_steps")
    if plan.warmup_steps and step <= plan.warmup_steps:
        return plan.learning_rate * step / plan.warmup_steps
    decay_steps = plan.max_steps - plan.warmup_steps
    progress = (step - plan.warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return plan.final_learning_rate + (plan.learning_rate - plan.final_learning_rate) * cosine


def _verify_parameter_partition(
    training: Mapping[str, object],
    plan: _ResolvedInferencePlan,
) -> None:
    decay = _string_list(training.get("parameter_decay_names"), label="decay names")
    no_decay = _string_list(training.get("parameter_no_decay_names"), label="no-decay names")
    if not decay or not no_decay or set(decay) & set(no_decay):
        raise ValueError("manifest parameter partition is empty or overlapping")
    if decay != sorted(decay) or no_decay != sorted(no_decay):
        raise ValueError("manifest parameter partition must be sorted")
    probe = NativeDenoiser(plan.model_config)
    expected_names = {name for name, _ in probe.named_parameters()}
    if set(decay) | set(no_decay) != expected_names:
        raise ValueError("manifest parameter partition does not cover the exact model schema")
    modules = dict(probe.named_modules())
    expected_no_decay: set[str] = set()
    for name, _ in probe.named_parameters():
        if name.endswith("bias"):
            expected_no_decay.add(name)
            continue
        module_name = name.rsplit(".", maxsplit=1)[0] if "." in name else ""
        module = modules.get(module_name)
        if isinstance(module, torch.nn.Embedding | torch.nn.LayerNorm):
            expected_no_decay.add(name)
    if set(no_decay) != expected_no_decay:
        raise ValueError("manifest no-decay partition differs from the optimizer contract")


def _verify_environment(
    payload: bytes,
    plan: _ResolvedInferencePlan,
    *,
    initialization_seed: int,
) -> str | None:
    document = _json_object(payload, label="environment.json")
    if set(document) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment.json schema differs from the training producer")
    _expect(document, "schema_version", 1, label="environment.json")
    _expect(
        document,
        "twin_environment_equal_fields",
        list(_TWIN_ENVIRONMENT_EQUAL_FIELDS),
        label="environment.json",
    )
    for name in ("amp", "tf32", "torch_compile"):
        _expect(document, name, False, label="environment.json")
    runtime = _mapping(document.get("runtime"), label="environment runtime")
    _validate_runtime_controls(runtime, seed=initialization_seed)
    if plan.production:
        pins = cast(_RuntimePins, plan.runtime_pins)
        expected_values: dict[str, object] = {
            "python": pins.python,
            "numpy": pins.numpy,
            "torch": pins.torch,
            "torch_cuda": pins.torch_cuda,
            "safetensors": pins.safetensors,
            "triton": pins.triton,
            "nvidia_cudnn_cu13": pins.nvidia_cudnn_cu13,
            "gpu_name": pins.gpu_name,
            "compute_capability": list(pins.compute_capability),
            "allocator": pins.allocator,
            "device_type": "cuda",
        }
        for name, expected in expected_values.items():
            _expect(document, name, expected, label="environment.json")
        driver = document.get("driver")
        if type(driver) is not str or _DRIVER_VERSION_RE.fullmatch(driver) is None:
            raise ValueError("environment.json driver must be one dotted-numeric version")
        return driver

    expected_cpu_identity: dict[str, object] = {
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
        "device_type": "cpu",
    }
    for name, expected in expected_cpu_identity.items():
        _expect(document, name, expected, label="test environment.json")
    return None


def _validate_runtime_environment_before_cuda(pins: _RuntimePins) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != pins.cublas_workspace_config:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exported before production CUDA inference"
        )
    if os.environ.get("PYTORCH_ALLOC_CONF") != pins.allocator:
        raise RuntimeError("the PyTorch allocator must be pinned before CUDA inference")
    legacy_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_allocator not in (None, pins.allocator):
        raise RuntimeError("legacy PyTorch allocator configuration conflicts with native v0")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"):
        raise RuntimeError("native v0 prohibits disabling CUDA memory caching")
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__.split("+", maxsplit=1)[0],
        "torch_cuda": torch.version.cuda,
        "safetensors": _distribution_version("safetensors"),
        "triton": _distribution_version("triton"),
        "nvidia_cudnn_cu13": _distribution_version("nvidia-cudnn-cu13"),
    }
    expected = {
        "python": pins.python,
        "numpy": pins.numpy,
        "torch": pins.torch,
        "torch_cuda": pins.torch_cuda,
        "safetensors": pins.safetensors,
        "triton": pins.triton,
        "nvidia_cudnn_cu13": pins.nvidia_cudnn_cu13,
    }
    if observed != expected:
        raise RuntimeError(
            f"production runtime versions differ: expected {expected}, got {observed}"
        )


def _validate_cuda_environment(
    pins: _RuntimePins,
    device: torch.device,
    *,
    expected_driver: str,
) -> None:
    if device != torch.device("cuda:0") or not torch.cuda.is_available():
        raise RuntimeError("production inference requires CUDA device zero")
    if torch.cuda.device_count() != pins.gpus_per_job:
        raise RuntimeError("visible GPU count differs from the contract")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != pins.gpu_name:
        raise RuntimeError("GPU name differs from the exact A100 contract")
    if tuple(torch.cuda.get_device_capability(device)) != pins.compute_capability:
        raise RuntimeError("GPU compute capability differs from the contract")
    allocator_getter = getattr(torch.cuda.memory, "get_allocator_backend", None)
    if allocator_getter is None:
        raise RuntimeError("cannot inspect the active PyTorch CUDA allocator")
    allocator = f"backend:{allocator_getter()}"
    if allocator != pins.allocator:
        raise RuntimeError("active PyTorch CUDA allocator is not native")
    if _cuda_driver_version() != expected_driver:
        raise RuntimeError("CUDA driver differs from the sealed training environment")


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
        raise ValueError("deterministic runtime controls differ from native v0")


def _validate_active_runtime_controls() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("cuBLAS workspace configuration drifted after provider loading")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("default dtype drifted after provider loading")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms were disabled after provider loading")
    observed = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdpa": torch.backends.cuda.math_sdp_enabled(),
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
    }
    expected = {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "matmul_tf32": False,
        "flash_sdpa": False,
        "memory_efficient_sdpa": False,
        "math_sdpa": True,
        "mha_fastpath": False,
    }
    if observed != expected:
        raise RuntimeError("deterministic runtime controls drifted after provider loading")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("float32 matmul precision differs from native v0")
    cudnn_sdp_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if callable(cudnn_sdp_enabled) and cudnn_sdp_enabled():
        raise RuntimeError("cuDNN scaled-dot-product attention must remain disabled")


def _validated_numpy_inputs(
    tokens: object,
    attention_mask: object,
    levels: object,
    lengths: object,
    *,
    config: NativeDenoiserConfig,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    token_values = _numpy_array(tokens, label="tokens", dtype=np.dtype(np.int64), ndim=2)
    mask_values = _numpy_array(
        attention_mask,
        label="attention_mask",
        dtype=np.dtype(np.bool_),
        ndim=2,
    )
    level_values = _numpy_array(levels, label="levels", dtype=np.dtype(np.int64), ndim=1)
    length_values = _numpy_array(lengths, label="lengths", dtype=np.dtype(np.int64), ndim=1)
    if token_values.shape != mask_values.shape:
        raise ValueError("tokens and attention_mask must have identical shapes")
    batch, width = token_values.shape
    if not 1 <= batch <= MAX_INFERENCE_BATCH_SEQUENCES:
        raise ValueError("inference batch size must lie in 1..256")
    if width != PRODUCTION_SEQUENCE_WIDTH or config.max_length != PRODUCTION_SEQUENCE_WIDTH:
        raise ValueError("native-v0 inference requires a fixed sequence width of 50")
    if level_values.shape != (batch,) or length_values.shape != (batch,):
        raise ValueError("levels and lengths must contain exactly one value per sequence")
    if np.any((length_values < config.min_length) | (length_values > config.max_length)):
        raise ValueError("lengths lie outside the native denoiser contract")
    if np.any((level_values < 1) | (level_values > config.levels)):
        raise ValueError("levels lie outside the native denoiser contract")
    expected_mask = np.arange(width, dtype=np.int64)[None, :] < length_values[:, None]
    if not np.array_equal(mask_values, expected_mask):
        raise ValueError("attention_mask must equal the declared non-empty prefix")
    if np.any((token_values < 0) | (token_values > MASK_TOKEN_INDEX)):
        raise ValueError("tokens contain an index outside the 22-token vocabulary")
    if np.any(mask_values & (token_values == PAD_TOKEN_INDEX)):
        raise ValueError("valid positions cannot contain PAD")
    if np.any(~mask_values & (token_values != PAD_TOKEN_INDEX)):
        raise ValueError("positions outside the valid prefix must contain PAD")
    return (
        cast(NDArray[np.int64], np.array(token_values, dtype=np.int64, order="C", copy=True)),
        cast(NDArray[np.bool_], np.array(mask_values, dtype=np.bool_, order="C", copy=True)),
        cast(NDArray[np.int64], np.array(level_values, dtype=np.int64, order="C", copy=True)),
        cast(NDArray[np.int64], np.array(length_values, dtype=np.int64, order="C", copy=True)),
    )


def _numpy_array(value: object, *, label: str, dtype: np.dtype[Any], ndim: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    array = cast(np.ndarray, value)
    if array.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype.name}")
    if array.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}")
    return np.array(array, dtype=dtype, order="C", copy=True)


def _parameter_guard(model: NativeDenoiser) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            parameter.data_ptr(),
            parameter._version,
            tuple(parameter.shape),
        )
        for name, parameter in model.named_parameters()
    )


def _bind_checkpoint_to_contract(config_sha256: str, model_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_MODEL_LOGICAL_BINDING_DOMAIN)
    for value in (config_sha256, model_sha256):
        encoded = bytes.fromhex(_sha256(value, label="checkpoint binding digest"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _schedule_digest(
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    final_learning_rate: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"amp-native-diffusion-learning-rate-schedule-v1\0")
    for step in range(1, max_steps + 1):
        if warmup_steps and step <= warmup_steps:
            rate = learning_rate * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            rate = final_learning_rate + (learning_rate - final_learning_rate) * cosine
        encoded = rate.hex().encode("ascii")
        digest.update(step.to_bytes(8, "big"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _snapshot_bundle(
    root_value: str | Path,
    inventory: Sequence[str],
) -> _BundleSnapshot:
    root = Path(os.path.abspath(os.fspath(root_value)))
    _reject_symlink_chain(root)
    before = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o555:
        raise ValueError("training bundle must be a sealed mode-0555 directory")
    expected = tuple(inventory)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("training bundle inventory contract is empty or duplicated")
    names = tuple(sorted(entry.name for entry in os.scandir(root)))
    if set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("training bundle inventory differs from the contract")
    files: dict[str, _FileSnapshot] = {}
    for name in sorted(expected):
        pure = PurePosixPath(name)
        if pure.name != name or pure.is_absolute() or name in {"", ".", ".."}:
            raise ValueError("training bundle inventory contains an unsafe filename")
        maximum = (
            _MAX_CHECKPOINT_BYTES if name == "model_final.safetensors" else _MAX_METADATA_BYTES
        )
        payload, fingerprint = _read_regular_file_snapshot(
            root / name,
            label=f"training artifact {name}",
            maximum_bytes=maximum,
        )
        files[name] = _FileSnapshot(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            fingerprint=fingerprint,
        )
    after = root.stat(follow_symlinks=False)
    if _fingerprint(before) != _fingerprint(after):
        raise ValueError("training bundle directory changed while it was read")
    return _BundleSnapshot(
        root=root,
        root_fingerprint=_fingerprint(before),
        files=files,
    )


def _assert_snapshot_unchanged(snapshot: _BundleSnapshot, inventory: Sequence[str]) -> None:
    current = _snapshot_bundle(snapshot.root, inventory)
    if current.root_fingerprint != snapshot.root_fingerprint:
        raise ValueError("training bundle directory changed during checkpoint loading")
    if {name: (item.sha256, item.fingerprint) for name, item in current.files.items()} != {
        name: (item.sha256, item.fingerprint) for name, item in snapshot.files.items()
    }:
        raise ValueError("training bundle artifact changed during checkpoint loading")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    return _read_regular_file_snapshot(
        path,
        label=label,
        maximum_bytes=_MAX_METADATA_BYTES,
        required_mode=None,
    )[0]


def _read_regular_file_snapshot(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    required_mode: int | None = 0o444,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    _reject_symlink_chain(path)
    try:
        named_before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}") from error
    invalid_mode = required_mode is not None and stat.S_IMODE(named_before.st_mode) != required_mode
    if (
        not stat.S_ISREG(named_before.st_mode)
        or invalid_mode
        or named_before.st_size <= 0
        or named_before.st_size > maximum_bytes
    ):
        mode_requirement = " mode-0444" if required_mode == 0o444 else ""
        raise ValueError(f"{label} must be a non-empty{mode_requirement} regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened_before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.stat(follow_symlinks=False)
    fingerprints = {
        _fingerprint(item) for item in (named_before, opened_before, opened_after, named_after)
    }
    if len(fingerprints) != 1:
        raise ValueError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != named_before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return payload, fingerprints.pop()


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


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _repository_code_manifest(
    repository: Path,
    *,
    expected_git_commit: str,
    contract_path: Path,
    config_sha256: str,
) -> bytes:
    if (
        type(expected_git_commit) is not str
        or _GIT_COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
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
    _reject_symlink_chain(repository)
    if not repository.is_dir():
        raise ValueError("repository_root must be a directory")
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
        raise ValueError("repository worktree must be clean for production inference")
    entries = _tracked_worktree_entries(repository, expected_commit=expected_git_commit)
    try:
        relative_contract = contract_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError(
            "production contract must be inside the synchronized repository"
        ) from error
    if entries.get(relative_contract) != config_sha256:
        raise ValueError("tracked contract is absent or differs from the exact config SHA-256")
    if _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError("repository changed while its code manifest was computed")
    return _sha_manifest_bytes(entries)


def _git_commit_tree(
    repository: Path,
    commit: str,
) -> tuple[tuple[str, str, str], ...]:
    raw = _git(repository, "ls-tree", "-rz", "--full-tree", "-r", commit)
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


def _read_git_regular_file_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    _reject_symlink_chain(path)
    try:
        named_before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.stat(follow_symlinks=False)
    fingerprints = {
        _fingerprint(item) for item in (named_before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != named_before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return payload, fingerprints.pop()


def _tracked_worktree_entries(
    repository: Path,
    *,
    expected_commit: str,
) -> dict[str, str]:
    tree = _git_commit_tree(repository, expected_commit)
    raw_names = _git(repository, "ls-files", "-z")
    try:
        names = tuple(value.decode("utf-8") for value in raw_names.split(b"\0") if value)
    except UnicodeDecodeError as error:
        raise ValueError("tracked repository inventory is not UTF-8") from error
    tree_names = tuple(name for name, _mode, _object_id in tree)
    if names != tree_names:
        raise ValueError("tracked repository inventory differs from the expected commit tree")
    ignored_source_paths = _git(
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
        payload, fingerprint = _read_git_regular_file_snapshot(
            path,
            label=f"tracked file {name}",
        )
        observed_mode = "100755" if fingerprint[-1] & stat.S_IXUSR else "100644"
        if observed_mode != mode:
            raise ValueError(f"tracked file mode differs from the expected commit: {name}")
        committed_payload = _git(repository, "cat-file", "blob", object_id)
        if payload != committed_payload:
            raise ValueError(f"tracked file bytes differ from the expected commit: {name}")
        entries[name] = hashlib.sha256(payload).hexdigest()
        snapshots[name] = (payload, fingerprint)

    if _git(repository, "ls-files", "-z") != raw_names:
        raise ValueError("tracked repository inventory changed during attestation")
    for name, _mode, _object_id in tree:
        current = _read_git_regular_file_snapshot(
            repository.joinpath(*PurePosixPath(name).parts),
            label=f"tracked file {name}",
        )
        if current != snapshots[name]:
            raise ValueError(f"tracked file changed during repository attestation: {name}")
    ignored_source_paths_after = _git(
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
    return entries


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


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    if not entries:
        raise ValueError("checksum manifest cannot be empty")
    payload = "".join(
        f"{_sha256(entries[name], label=f'checksum for {name}')}  {name}\n"
        for name in sorted(entries)
    ).encode("ascii")
    if _parse_sha_manifest(payload, label="checksum manifest") != dict(entries):
        raise ValueError("checksum manifest round trip changed")
    return payload


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be canonical LF-terminated text")
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"{label} contains a malformed checksum row")
        digest, name = line[:64], line[66:]
        _sha256(digest, label=f"{label} digest")
        pure = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError(f"{label} contains an unsafe path")
        if previous is not None and name <= previous:
            raise ValueError(f"{label} rows must be strictly name-ordered")
        result[name] = digest
        previous = name
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_nonfinite(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)


def _json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _reject_nonfinite(value, label=label)
    if payload != _canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, object], value)


def _canonical_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty canonical JSONL")
    records = tuple(
        _json_object(line + b"\n", label=f"{label} row {number}")
        for number, line in enumerate(payload[:-1].split(b"\n"), 1)
    )
    return records


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        raise ValueError(f"{label} schema differs from the training producer")
    return result


def _expect(document: Mapping[str, object], name: str, expected: object, *, label: str) -> None:
    observed = document.get(name)
    if not _strictly_equal(observed, expected):
        raise ValueError(f"{label}.{name} must equal {expected!r}")


def _strictly_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        observed_mapping = cast(dict[object, object], observed)
        return set(observed_mapping) == set(expected) and all(
            _strictly_equal(observed_mapping[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        observed_list = cast(list[object], observed)
        return len(observed_list) == len(expected) and all(
            _strictly_equal(left, right)
            for left, right in zip(observed_list, expected, strict=True)
        )
    return observed == expected


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object, *, label: str) -> str:
    if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return value


def _integer(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    return value


def _string_list(value: object, *, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return cast(list[str], value)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required production distribution is missing: {name}") from error


def _optional_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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

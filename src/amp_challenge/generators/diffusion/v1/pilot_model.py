"""Exact R128 residual denoiser and logical checkpoint schema for the v1 pilot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import struct
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import module as torch_module
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_rng import (
    PAD_TOKEN_INDEX,
    TRAINING_ROOT_SEED,
    reseed_torch_for_initialization,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
INPUT_VOCABULARY_SIZE = 22
RESIDUE_VOCABULARY_SIZE = 20
EXPECTED_TRAINABLE_PARAMETERS = 354_068
INITIALIZATION_STD = 0.02
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_PYTORCH_ALLOC_CONF = "backend:native"

_STATE_DOMAIN = b"amp-native-denoiser-logical-state-v1\0"
_MODEL_STATE_DOMAIN = b"amp-native-denoiser-model-logical-v1\0"

R128_TENSOR_SCHEMA = (
    ("encoder.layers.0.linear1.bias", "F32", (384,)),
    ("encoder.layers.0.linear1.weight", "F32", (384, 128)),
    ("encoder.layers.0.linear2.bias", "F32", (128,)),
    ("encoder.layers.0.linear2.weight", "F32", (128, 384)),
    ("encoder.layers.0.norm1.bias", "F32", (128,)),
    ("encoder.layers.0.norm1.weight", "F32", (128,)),
    ("encoder.layers.0.norm2.bias", "F32", (128,)),
    ("encoder.layers.0.norm2.weight", "F32", (128,)),
    ("encoder.layers.0.self_attn.in_proj_bias", "F32", (384,)),
    ("encoder.layers.0.self_attn.in_proj_weight", "F32", (384, 128)),
    ("encoder.layers.0.self_attn.out_proj.bias", "F32", (128,)),
    ("encoder.layers.0.self_attn.out_proj.weight", "F32", (128, 128)),
    ("encoder.layers.1.linear1.bias", "F32", (384,)),
    ("encoder.layers.1.linear1.weight", "F32", (384, 128)),
    ("encoder.layers.1.linear2.bias", "F32", (128,)),
    ("encoder.layers.1.linear2.weight", "F32", (128, 384)),
    ("encoder.layers.1.norm1.bias", "F32", (128,)),
    ("encoder.layers.1.norm1.weight", "F32", (128,)),
    ("encoder.layers.1.norm2.bias", "F32", (128,)),
    ("encoder.layers.1.norm2.weight", "F32", (128,)),
    ("encoder.layers.1.self_attn.in_proj_bias", "F32", (384,)),
    ("encoder.layers.1.self_attn.in_proj_weight", "F32", (384, 128)),
    ("encoder.layers.1.self_attn.out_proj.bias", "F32", (128,)),
    ("encoder.layers.1.self_attn.out_proj.weight", "F32", (128, 128)),
    ("encoder.norm.bias", "F32", (128,)),
    ("encoder.norm.weight", "F32", (128,)),
    ("length_embedding.weight", "F32", (43, 128)),
    ("position_embedding.weight", "F32", (50, 128)),
    ("residue_output_bias", "F32", (20,)),
    ("timestep_embedding.weight", "F32", (65, 128)),
    ("token_embedding.weight", "F32", (22, 128)),
)

R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD = (
    "065aa5e571d8a830caa8226ed84b90a82b7b6da2f1e6d4e79f0f92cbffcb4bc0",
    "20fd386333cadd73dddcdb2312068c1fbd3be81ca6594e5f036df073aac443e8",
    "fca8c7f3fc8fa3757ac4c9a0169a948cdc7121771aa6ff81f1714e5394bc0864",
    "d76671d4af2a20d185ff42482bc528a451bbd8436485519dba388a8f9436e8b1",
)


def establish_r128_deterministic_runtime(
    contract: NativeDiffusionV1PilotContract | None = None,
) -> dict[str, object]:
    """Establish the inherited parent runtime before any R128 tensor exists.

    Environment variables whose meaning is fixed at CUDA initialization are
    never repaired after CUDA has initialized.  A conflicting value, or a
    missing value discovered too late, is a hard failure.
    """

    if contract is not None:
        _validate_parent_runtime_contract(contract)
    _establish_pre_cuda_environment()
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return assert_r128_deterministic_runtime(contract)


def assert_r128_deterministic_runtime(
    contract: NativeDiffusionV1PilotContract | None = None,
) -> dict[str, object]:
    """Reject any drift from the exact inherited parent numerical runtime."""

    if contract is not None:
        _validate_parent_runtime_contract(contract)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the parent contract")
    if os.environ.get("PYTORCH_ALLOC_CONF") != _PYTORCH_ALLOC_CONF:
        raise RuntimeError("PYTORCH_ALLOC_CONF differs from the parent contract")
    legacy_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_allocator not in (None, _PYTORCH_ALLOC_CONF):
        raise RuntimeError("legacy CUDA allocator configuration conflicts with the parent")
    observed: dict[str, object] = {
        "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdpa": torch.backends.cuda.math_sdp_enabled(),
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
    }
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        observed["cudnn_sdpa"] = torch.backends.cuda.cudnn_sdp_enabled()
    expected: dict[str, object] = {
        "default_dtype": "float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "matmul_tf32": False,
        "flash_sdpa": False,
        "memory_efficient_sdpa": False,
        "math_sdpa": True,
        "mha_fastpath": False,
    }
    if "cudnn_sdpa" in observed:
        expected["cudnn_sdpa"] = False
    failed = tuple(
        name
        for name, expected_value in expected.items()
        if type(observed[name]) is not type(expected_value) or observed[name] != expected_value
    )
    if failed:
        raise RuntimeError(f"R128 deterministic runtime drifted: {', '.join(failed)}")
    return {
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
        "pytorch_allocator": _PYTORCH_ALLOC_CONF,
        **observed,
    }


def _establish_pre_cuda_environment() -> None:
    required = {
        "CUBLAS_WORKSPACE_CONFIG": _CUBLAS_WORKSPACE_CONFIG,
        "PYTORCH_ALLOC_CONF": _PYTORCH_ALLOC_CONF,
    }
    for name, expected in required.items():
        observed = os.environ.get(name)
        if observed is not None and observed != expected:
            raise RuntimeError(f"{name} conflicts with the inherited parent runtime")
        if observed is None:
            if torch.cuda.is_initialized():
                raise RuntimeError(f"{name} was not established before CUDA initialization")
            os.environ[name] = expected
    legacy_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_allocator not in (None, _PYTORCH_ALLOC_CONF):
        raise RuntimeError("legacy CUDA allocator configuration conflicts with the parent")


def _validate_parent_runtime_contract(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    determinism = contract.parent_table("determinism")
    required: dict[str, object] = {
        "deterministic_algorithms": True,
        "math_sdpa_only": True,
        "mha_fastpath": False,
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
        "cudnn_benchmark": False,
        "pytorch_allocator": _PYTORCH_ALLOC_CONF,
    }
    for name, expected in required.items():
        observed = determinism.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"authenticated parent determinism.{name} changed")
    training_tf32 = contract.parent_table("training").get("tf32")
    if training_tf32 is not False:
        raise ValueError("authenticated parent training.tf32 must be exact false")


def verify_r128_production_environment(
    contract: NativeDiffusionV1PilotContract,
    device: str | torch.device = "cuda:0",
) -> dict[str, object]:
    """Verify the complete authenticated A100 runtime before production work."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    bound_device = torch.device(device)
    if bound_device != torch.device("cuda:0"):
        raise RuntimeError("production R128 fitting requires exact device cuda:0")
    assert_r128_deterministic_runtime(contract)
    environment = contract.parent_table("environment")
    expected_environment: dict[str, object] = {
        "python": "3.11.14",
        "numpy": "2.4.6",
        "torch": "2.14.0",
        "torch_cuda": "13.0",
        "safetensors": "0.6.2",
        "packaging": "26.3",
        "triton": "3.8.0",
        "nvidia_cudnn_cu13": "9.24.0.43",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "compute_capability": (8, 0),
        "checkpoint_format": "safetensors",
        "allow_pickle": False,
    }
    for name, expected in expected_environment.items():
        observed = environment.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"authenticated parent environment.{name} changed")
    compute = contract.parent_table("compute")
    if type(compute.get("gpus_per_task")) is not int or compute["gpus_per_task"] != 1:
        raise ValueError("authenticated parent compute.gpus_per_task must be exact one")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"):
        raise RuntimeError("production forbids disabling CUDA memory caching")

    package_names = {
        "safetensors": "safetensors",
        "packaging": "packaging",
        "triton": "triton",
        "nvidia_cudnn_cu13": "nvidia-cudnn-cu13",
    }
    installed: dict[str, str] = {}
    for field, distribution in package_names.items():
        try:
            installed[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required distribution {distribution} is not installed") from error
    observed_versions: dict[str, object] = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "torch": torch.__version__.split("+", maxsplit=1)[0],
        "torch_cuda": torch.version.cuda,
        **installed,
    }
    for name, observed in observed_versions.items():
        if observed != environment[name]:
            raise RuntimeError(f"installed {name} runtime differs from the parent contract")

    # Package/version checks above do not inspect CUDA.  Environment pins were
    # established before reaching the first CUDA query below.
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if properties.name != environment["gpu_name"]:
        raise RuntimeError("visible CUDA GPU name differs from the parent contract")
    if capability != environment["compute_capability"]:
        raise RuntimeError("visible CUDA compute capability differs from the parent contract")
    try:
        allocator_backend = torch.cuda.memory.get_allocator_backend()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("cannot verify the active PyTorch CUDA allocator") from error
    allocator = f"backend:{allocator_backend}"
    if allocator != contract.parent_table("determinism")["pytorch_allocator"]:
        raise RuntimeError("active PyTorch CUDA allocator differs from the parent contract")
    cudnn_parts = installed["nvidia_cudnn_cu13"].split(".")
    if len(cudnn_parts) < 3 or any(not part.isdigit() for part in cudnn_parts[:3]):
        raise RuntimeError("installed cuDNN distribution version is not numeric")
    expected_cudnn_runtime = (
        int(cudnn_parts[0]) * 10_000 + int(cudnn_parts[1]) * 100 + int(cudnn_parts[2])
    )
    observed_cudnn_runtime = torch.backends.cudnn.version()
    if observed_cudnn_runtime != expected_cudnn_runtime:
        raise RuntimeError("loaded cuDNN runtime differs from the pinned distribution")
    return {
        **observed_versions,
        "gpu_name": properties.name,
        "compute_capability": capability,
        "visible_cuda_devices": 1,
        "allocator": allocator,
        "cudnn_runtime_version": observed_cudnn_runtime,
    }


@dataclass(frozen=True, slots=True)
class R128Config:
    """The single architecture selected by the frozen pilot child contract."""

    layers: int = 2
    hidden_dim: int = 128
    attention_heads: int = 4
    ffn_dim: int = 384
    dropout: float = 0.20
    layer_norm_epsilon: float = 1e-5
    min_length: int = 8
    max_length: int = 50
    levels: int = 64

    def __post_init__(self) -> None:
        expected = (2, 128, 4, 384, 0.20, 1e-5, 8, 50, 64)
        observed = (
            self.layers,
            self.hidden_dim,
            self.attention_heads,
            self.ffn_dim,
            self.dropout,
            self.layer_norm_epsilon,
            self.min_length,
            self.max_length,
            self.levels,
        )
        if any(
            type(left) is not type(right) or left != right
            for left, right in zip(observed, expected, strict=True)
        ):
            raise ValueError("R128Config may contain only the exact pilot architecture values")


class R128Denoiser(nn.Module):
    """Two-layer pre-LN bidirectional Transformer producing raw residual logits."""

    def __init__(self, config: R128Config | None = None) -> None:
        establish_r128_deterministic_runtime()
        super().__init__()
        if config is not None and type(config) is not R128Config:
            raise TypeError("config must be an R128Config")
        self.config = config or R128Config()
        width = self.config.hidden_dim
        self.token_embedding = nn.Embedding(
            INPUT_VOCABULARY_SIZE,
            width,
            padding_idx=PAD_TOKEN_INDEX,
        )
        self.position_embedding = nn.Embedding(self.config.max_length, width)
        self.length_embedding = nn.Embedding(
            self.config.max_length - self.config.min_length + 1,
            width,
        )
        self.timestep_embedding = nn.Embedding(self.config.levels + 1, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout,
            activation="gelu",
            layer_norm_eps=self.config.layer_norm_epsilon,
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.layers,
            norm=nn.LayerNorm(width, eps=self.config.layer_norm_epsilon),
            enable_nested_tensor=False,
        )
        self.residue_output_bias = nn.Parameter(torch.zeros(RESIDUE_VOCABULARY_SIZE))
        self.float()
        self.reset_parameters()

    @property
    def residue_output_weight(self) -> Tensor:
        """The output table tied to residue rows 0..19 of the input embedding."""

        return self.token_embedding.weight[:RESIDUE_VOCABULARY_SIZE]

    def reset_parameters(self) -> None:
        """Apply normal(0,.02), zero biases, unit norms, and a zero PAD row."""

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=INITIALIZATION_STD)
                    if module.padding_idx is not None:
                        module.weight[module.padding_idx].zero_()
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.normal_(module.in_proj_weight, mean=0.0, std=INITIALIZATION_STD)
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.zero_()
                    if module.bias_k is not None:
                        module.bias_k.zero_()
                    if module.bias_v is not None:
                        module.bias_v.zero_()
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=INITIALIZATION_STD)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.LayerNorm):
                    if module.weight is not None:
                        module.weight.fill_(1.0)
                    if module.bias is not None:
                        module.bias.zero_()
            self.residue_output_bias.zero_()

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor,
        levels: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Return finite float32 residual logits with shape ``[B,W,20]``."""

        assert_r128_deterministic_runtime()
        assert_r128_model_execution_surface(self)
        _validate_inputs(
            tokens=tokens,
            attention_mask=attention_mask,
            levels=levels,
            lengths=lengths,
            expected_device=self.token_embedding.weight.device,
        )
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("R128 parameters must remain torch.float32")
        batch, width = tokens.shape
        positions = torch.arange(width, device=tokens.device, dtype=torch.long)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + self.length_embedding(lengths - self.config.min_length).reshape(
            batch, 1, -1
        )
        hidden = hidden + self.timestep_embedding(levels).reshape(batch, 1, -1)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        residual = F.linear(hidden, self.residue_output_weight, self.residue_output_bias)
        residual = residual.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        if residual.dtype != torch.float32 or not bool(torch.isfinite(residual).all().item()):
            raise FloatingPointError("R128 forward pass produced invalid residual logits")
        return residual


_R128_FORWARD_METHOD = R128Denoiser.forward
_R128_RESET_PARAMETERS_METHOD = R128Denoiser.reset_parameters
_R128_OUTPUT_WEIGHT_DESCRIPTOR = R128Denoiser.residue_output_weight
_MODULE_CALL_METHOD = nn.Module.__call__
_MODULE_WRAPPED_CALL_IMPL_METHOD = nn.Module._wrapped_call_impl
_MODULE_CALL_IMPL_METHOD = nn.Module._call_impl
_MODULE_MODULES_METHOD = nn.Module.modules
_MODULE_NAMED_MODULES_METHOD = nn.Module.named_modules
_MODULE_PARAMETERS_METHOD = nn.Module.parameters
_MODULE_NAMED_PARAMETERS_METHOD = nn.Module.named_parameters
_MODULE_TRAIN_METHOD = nn.Module.train
_MODULE_EVAL_METHOD = nn.Module.eval
_MODULE_STATE_DICT_METHOD = nn.Module.state_dict
_MODULE_LOAD_STATE_DICT_METHOD = nn.Module.load_state_dict
_TENSOR_BACKWARD_METHOD = Tensor.backward
_EMBEDDING_FORWARD_METHOD = nn.Embedding.forward
_TRANSFORMER_ENCODER_FORWARD_METHOD = nn.TransformerEncoder.forward
_TRANSFORMER_LAYER_FORWARD_METHOD = nn.TransformerEncoderLayer.forward
_MULTIHEAD_ATTENTION_FORWARD_METHOD = nn.MultiheadAttention.forward
_LINEAR_FORWARD_METHOD = nn.Linear.forward
_LAYER_NORM_FORWARD_METHOD = nn.LayerNorm.forward
_DROPOUT_FORWARD_METHOD = nn.Dropout.forward
_FUNCTIONAL_EMBEDDING = F.embedding
_FUNCTIONAL_LINEAR = F.linear
_FUNCTIONAL_DROPOUT = F.dropout
_FUNCTIONAL_LAYER_NORM = F.layer_norm
_FUNCTIONAL_MULTI_HEAD_ATTENTION_FORWARD = F.multi_head_attention_forward
_FUNCTIONAL_GELU = F.gelu


def assert_r128_model_execution_surface(model: R128Denoiser) -> None:
    """Reject method/config replacement and every forward/backward/state hook."""

    if type(model) is not R128Denoiser:
        raise TypeError("model must be an exact R128Denoiser")
    if R128Denoiser.forward is not _R128_FORWARD_METHOD:
        raise RuntimeError("R128Denoiser.forward was overridden")
    if R128Denoiser.reset_parameters is not _R128_RESET_PARAMETERS_METHOD:
        raise RuntimeError("R128Denoiser.reset_parameters was overridden")
    if R128Denoiser.residue_output_weight is not _R128_OUTPUT_WEIGHT_DESCRIPTOR:
        raise RuntimeError("R128 tied output-weight descriptor was overridden")
    if nn.Module.__call__ is not _MODULE_CALL_METHOD:
        raise RuntimeError("torch Module call method was overridden")
    if nn.Module._wrapped_call_impl is not _MODULE_WRAPPED_CALL_IMPL_METHOD:
        raise RuntimeError("torch Module wrapped-call dispatch was overridden")
    if nn.Module._call_impl is not _MODULE_CALL_IMPL_METHOD:
        raise RuntimeError("torch Module call dispatch was overridden")
    base_methods = (
        (nn.Module.modules, _MODULE_MODULES_METHOD, "modules"),
        (nn.Module.named_modules, _MODULE_NAMED_MODULES_METHOD, "named_modules"),
        (nn.Module.parameters, _MODULE_PARAMETERS_METHOD, "parameters"),
        (nn.Module.named_parameters, _MODULE_NAMED_PARAMETERS_METHOD, "named_parameters"),
        (nn.Module.train, _MODULE_TRAIN_METHOD, "train"),
        (nn.Module.eval, _MODULE_EVAL_METHOD, "eval"),
    )
    for observed, expected, name in base_methods:
        if observed is not expected:
            raise RuntimeError(f"torch Module {name} method was overridden")
    if nn.Module.state_dict is not _MODULE_STATE_DICT_METHOD:
        raise RuntimeError("torch Module state_dict was overridden")
    if nn.Module.load_state_dict is not _MODULE_LOAD_STATE_DICT_METHOD:
        raise RuntimeError("torch Module load_state_dict was overridden")
    if Tensor.backward is not _TENSOR_BACKWARD_METHOD:
        raise RuntimeError("torch Tensor backward was overridden")
    forward_methods = (
        (nn.Embedding.forward, _EMBEDDING_FORWARD_METHOD, "Embedding"),
        (
            nn.TransformerEncoder.forward,
            _TRANSFORMER_ENCODER_FORWARD_METHOD,
            "TransformerEncoder",
        ),
        (
            nn.TransformerEncoderLayer.forward,
            _TRANSFORMER_LAYER_FORWARD_METHOD,
            "TransformerEncoderLayer",
        ),
        (
            nn.MultiheadAttention.forward,
            _MULTIHEAD_ATTENTION_FORWARD_METHOD,
            "MultiheadAttention",
        ),
        (nn.Linear.forward, _LINEAR_FORWARD_METHOD, "Linear"),
        (nn.LayerNorm.forward, _LAYER_NORM_FORWARD_METHOD, "LayerNorm"),
        (nn.Dropout.forward, _DROPOUT_FORWARD_METHOD, "Dropout"),
    )
    for observed, expected, name in forward_methods:
        if observed is not expected:
            raise RuntimeError(f"torch {name}.forward was overridden")
    functional_methods = (
        (F.embedding, _FUNCTIONAL_EMBEDDING, "embedding"),
        (F.linear, _FUNCTIONAL_LINEAR, "linear"),
        (F.dropout, _FUNCTIONAL_DROPOUT, "dropout"),
        (F.layer_norm, _FUNCTIONAL_LAYER_NORM, "layer_norm"),
        (
            F.multi_head_attention_forward,
            _FUNCTIONAL_MULTI_HEAD_ATTENTION_FORWARD,
            "multi_head_attention_forward",
        ),
        (F.gelu, _FUNCTIONAL_GELU, "gelu"),
    )
    for observed, expected, name in functional_methods:
        if observed is not expected:
            raise RuntimeError(f"torch functional {name} was overridden")
    forbidden_instance_methods = frozenset(
        {
            "__call__",
            "_wrapped_call_impl",
            "_call_impl",
            "modules",
            "named_modules",
            "parameters",
            "named_parameters",
            "train",
            "eval",
            "forward",
            "state_dict",
            "load_state_dict",
        }
    )
    hook_fields = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks_with_kwargs",
    )
    modules = _r128_exact_module_graph(model)
    for module in modules:
        if forbidden_instance_methods & set(module.__dict__):
            raise RuntimeError("R128 module method was overridden on an instance")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError("R128 torch-compiled module dispatch is forbidden")
        for field in hook_fields:
            hooks = getattr(module, field, None)
            if hooks:
                raise RuntimeError(f"R128 model hook registry {field} must be empty")
    for field in (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
    ):
        if getattr(torch_module, field, None):
            raise RuntimeError(f"R128 global model hook registry {field} must be empty")
    for parameter in model.parameters():
        if getattr(parameter, "_backward_hooks", None):
            raise RuntimeError("R128 parameter gradient hooks are forbidden")
        if getattr(parameter, "_post_accumulate_grad_hooks", None):
            raise RuntimeError("R128 post-accumulate gradient hooks are forbidden")
        gradient = parameter.grad
        if gradient is not None and getattr(gradient, "_backward_hooks", None):
            raise RuntimeError("R128 gradient-tensor hooks are forbidden")


def _r128_exact_module_graph(model: R128Denoiser) -> tuple[nn.Module, ...]:
    """Validate the complete state-preserving execution configuration."""

    if type(model.config) is not R128Config or model.config != R128Config():
        raise ValueError("R128 model config differs from the frozen architecture")
    expected_root_children = (
        "token_embedding",
        "position_embedding",
        "length_embedding",
        "timestep_embedding",
        "encoder",
    )
    if type(model._modules) is not dict or tuple(model._modules) != expected_root_children:
        raise ValueError("R128 root module graph differs from the frozen architecture")

    embeddings = (
        (model.token_embedding, 22, 20, "token"),
        (model.position_embedding, 50, None, "position"),
        (model.length_embedding, 43, None, "length"),
        (model.timestep_embedding, 65, None, "timestep"),
    )
    modules: list[nn.Module] = [model]
    for embedding, count, padding, label in embeddings:
        if type(embedding) is not nn.Embedding or (
            embedding.num_embeddings,
            embedding.embedding_dim,
            embedding.padding_idx,
            embedding.max_norm,
            embedding.norm_type,
            embedding.scale_grad_by_freq,
            embedding.sparse,
        ) != (count, 128, padding, None, 2.0, False, False):
            raise ValueError(f"R128 {label} embedding configuration changed")
        if type(embedding._modules) is not dict or embedding._modules:
            raise ValueError(f"R128 {label} embedding gained a child module")
        modules.append(embedding)

    encoder = model.encoder
    if type(encoder) is not nn.TransformerEncoder or (
        encoder.num_layers,
        encoder.enable_nested_tensor,
        encoder.use_nested_tensor,
        encoder.mask_check,
    ) != (2, False, False, True):
        raise ValueError("R128 TransformerEncoder configuration changed")
    if type(encoder._modules) is not dict or tuple(encoder._modules) != ("layers", "norm"):
        raise ValueError("R128 TransformerEncoder module graph changed")
    layers = encoder.layers
    if type(layers) is not nn.ModuleList or tuple(layers._modules) != ("0", "1"):
        raise ValueError("R128 TransformerEncoder layer inventory changed")
    modules.extend((encoder, layers))
    for layer_index, layer in enumerate(layers):
        if type(layer) is not nn.TransformerEncoderLayer:
            raise ValueError(f"R128 encoder layer {layer_index} type changed")
        expected_layer_children = (
            "self_attn",
            "linear1",
            "dropout",
            "linear2",
            "norm1",
            "norm2",
            "dropout1",
            "dropout2",
        )
        if type(layer._modules) is not dict or tuple(layer._modules) != expected_layer_children:
            raise ValueError(f"R128 encoder layer {layer_index} module graph changed")
        if (
            layer.norm_first is not True
            or layer.activation is not _FUNCTIONAL_GELU
            or layer.activation_relu_or_gelu != 2
        ):
            raise ValueError(f"R128 encoder layer {layer_index} execution config changed")
        attention = layer.self_attn
        if type(attention) is not nn.MultiheadAttention or (
            attention.embed_dim,
            attention.kdim,
            attention.vdim,
            attention.num_heads,
            attention.dropout,
            attention.batch_first,
            attention.add_zero_attn,
            attention._qkv_same_embed_dim,
        ) != (128, 128, 128, 4, 0.2, True, False, True):
            raise ValueError(f"R128 encoder layer {layer_index} attention config changed")
        if attention.bias_k is not None or attention.bias_v is not None:
            raise ValueError(f"R128 encoder layer {layer_index} attention bias config changed")
        if type(attention._modules) is not dict or tuple(attention._modules) != ("out_proj",):
            raise ValueError(f"R128 encoder layer {layer_index} attention graph changed")
        if type(attention.out_proj) is not NonDynamicallyQuantizableLinear or (
            attention.out_proj.in_features,
            attention.out_proj.out_features,
            attention.out_proj.bias is not None,
        ) != (128, 128, True):
            raise ValueError(f"R128 encoder layer {layer_index} output projection changed")
        if type(layer.linear1) is not nn.Linear or (
            layer.linear1.in_features,
            layer.linear1.out_features,
            layer.linear1.bias is not None,
        ) != (128, 384, True):
            raise ValueError(f"R128 encoder layer {layer_index} linear1 config changed")
        if type(layer.linear2) is not nn.Linear or (
            layer.linear2.in_features,
            layer.linear2.out_features,
            layer.linear2.bias is not None,
        ) != (384, 128, True):
            raise ValueError(f"R128 encoder layer {layer_index} linear2 config changed")
        for name in ("dropout", "dropout1", "dropout2"):
            dropout = getattr(layer, name)
            if type(dropout) is not nn.Dropout or (dropout.p, dropout.inplace) != (0.2, False):
                raise ValueError(f"R128 encoder layer {layer_index} {name} config changed")
        for name in ("norm1", "norm2"):
            _assert_r128_layer_norm(getattr(layer, name), label=f"layer {layer_index} {name}")
        modules.extend(
            (
                layer,
                attention,
                attention.out_proj,
                layer.linear1,
                layer.dropout,
                layer.linear2,
                layer.norm1,
                layer.norm2,
                layer.dropout1,
                layer.dropout2,
            )
        )
    _assert_r128_layer_norm(encoder.norm, label="encoder norm")
    modules.append(encoder.norm)

    named_modules = tuple(_MODULE_NAMED_MODULES_METHOD(model))
    if tuple(module for _, module in named_modules) != tuple(modules):
        raise ValueError("R128 named module traversal differs from its exact graph")
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("R128 module graph contains duplicate module identities")
    named_parameters = tuple(
        sorted(_MODULE_NAMED_PARAMETERS_METHOD(model), key=lambda item: item[0])
    )
    expected_parameters = tuple(sorted((name, shape) for name, _, shape in R128_TENSOR_SCHEMA))
    if tuple(name for name, _ in named_parameters) != tuple(
        name for name, _ in expected_parameters
    ):
        raise ValueError("R128 live parameter inventory differs from its tensor schema")
    storage_ranges: list[tuple[int, int, str]] = []
    for (name, parameter), (_, shape) in zip(
        named_parameters,
        expected_parameters,
        strict=True,
    ):
        expected_bytes = math.prod(shape) * torch.finfo(torch.float32).bits // 8
        if (
            type(parameter) is not nn.Parameter
            or parameter.dtype != torch.float32
            or tuple(parameter.shape) != shape
            or parameter.layout != torch.strided
            or parameter.storage_offset() != 0
            or parameter.stride() != _contiguous_stride(shape)
            or not parameter.is_contiguous()
            or parameter.untyped_storage().nbytes() != expected_bytes
            or parameter.data_ptr() != parameter.untyped_storage().data_ptr()
        ):
            raise ValueError(f"R128 parameter {name} storage layout changed")
        start = parameter.data_ptr()
        stop = start + expected_bytes
        if any(
            start < prior_stop and prior_start < stop
            for prior_start, prior_stop, _ in storage_ranges
        ):
            raise ValueError(f"R128 parameter {name} overlaps another parameter storage")
        storage_ranges.append((start, stop, name))
    training_mode = model.training
    if type(training_mode) is not bool or any(
        module.training is not training_mode for module in modules
    ):
        raise ValueError("R128 module training modes are inconsistent")
    output_weight = model.residue_output_weight
    token_weight = model.token_embedding.weight
    if (
        output_weight.shape != (20, 128)
        or output_weight._base is not token_weight
        or output_weight.data_ptr() != token_weight.data_ptr()
        or output_weight.stride() != token_weight.stride()
    ):
        raise ValueError("R128 residue output weight is not the exact tied embedding view")
    return tuple(modules)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _assert_r128_layer_norm(module: nn.Module, *, label: str) -> None:
    if type(module) is not nn.LayerNorm or (
        module.normalized_shape,
        module.eps,
        module.elementwise_affine,
        module.weight is not None,
        module.bias is not None,
    ) != ((128,), 1e-5, True, True, True):
        raise ValueError(f"R128 {label} configuration changed")
    if type(module._modules) is not dict or module._modules:
        raise ValueError(f"R128 {label} gained a child module")


def build_r128_model_for_fit(
    fit_identity_sha256: str,
    *,
    device: str | torch.device = "cpu",
    root_seed: int = TRAINING_ROOT_SEED,
) -> tuple[R128Denoiser, int]:
    """Seed immediately before construction and return the exact R128 model."""

    establish_r128_deterministic_runtime()
    seed = reseed_torch_for_initialization(fit_identity_sha256, root_seed=root_seed)
    model = R128Denoiser().to(device=torch.device(device), dtype=torch.float32)
    assert_r128_model_execution_surface(model)
    validate_r128_state(model.state_dict())
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"R128 parameter census expected {EXPECTED_TRAINABLE_PARAMETERS}, got {count}"
        )
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("every R128 parameter must remain trainable")
    return model, seed


def build_r128_model_from_contract(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[R128Denoiser, int]:
    """Bind construction to one authenticated child-contract fit identity."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    establish_r128_deterministic_runtime(contract)
    _validate_contract_model(contract)
    fit_identity = contract.fit_identity_sha256(outer_fold)
    if fit_identity != contract.fold(outer_fold).fit_identity_sha256:
        raise ValueError("derived fit identity differs from the frozen fold identity")
    model, seed = build_r128_model_for_fit(
        fit_identity,
        device=device,
        root_seed=contract.seed,
    )
    expected_model_sha256 = R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD[outer_fold]
    if canonical_r128_model_sha256(model.state_dict()) != expected_model_sha256:
        raise RuntimeError("R128 initial model state differs from its frozen golden hash")
    return model, seed


def r128_model_config_document() -> dict[str, object]:
    """Return the exact child-contract model table as a fresh plain mapping."""

    return {
        "kind": "bidirectional_pre_layer_norm_transformer",
        "alphabet": ALPHABET,
        "min_length": 8,
        "max_length": 50,
        "special_tokens": ["PAD", "MASK"],
        "layers": 2,
        "hidden_dim": 128,
        "attention_heads": 4,
        "ffn_dim": 384,
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "dropout": 0.20,
        "layer_norm_epsilon": 1e-5,
        "activation": "gelu",
        "tie_residue_input_output_weights": True,
        "prediction_classes": 20,
        "initialization": "normal_std_0.02_bias_zero_norm_scale_one",
        "residual_training_logits": ("log(clipped_renormalized_fold_local_C0)+raw_residual_logits"),
        "residual_training_lambda": 1.0,
        "residual_training_temperature": 1.0,
        "property_conditioning": False,
        "self_conditioning": False,
        "geometry_conditioning": False,
    }


def validate_r128_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return a detached CPU copy after enforcing the exact 31-tensor schema."""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a tensor mapping")
    if any(type(name) is not str for name in state):
        raise TypeError("R128 state names must be exact strings")
    expected_names = tuple(name for name, _, _ in R128_TENSOR_SCHEMA)
    if set(state) != set(expected_names) or len(state) != len(expected_names):
        missing = sorted(set(expected_names) - set(state))
        extra = sorted(set(state) - set(expected_names))
        raise ValueError(f"R128 state schema mismatch: missing={missing}, extra={extra}")
    result: dict[str, Tensor] = {}
    for name, dtype, shape in R128_TENSOR_SCHEMA:
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"R128 state member {name} must be a tensor")
        if dtype != "F32" or tensor.dtype != torch.float32:
            raise TypeError(f"R128 state member {name} must be F32")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"R128 state member {name} must have shape {shape}")
        if tensor.layout != torch.strided:
            raise TypeError(f"R128 state member {name} must be a strided tensor")
        value = tensor.detach().cpu().contiguous().clone()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"R128 state member {name} contains a non-finite value")
        result[name] = value
    if tuple(result) != expected_names:  # pragma: no cover - loop invariant
        raise RuntimeError("R128 state validation changed tensor order")
    return result


def canonical_r128_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Apply the established native-denoiser logical-state hash to R128."""

    validated = validate_r128_state(state)
    digest = hashlib.sha256()
    digest.update(_STATE_DOMAIN)
    digest.update(struct.pack("<Q", len(validated)))
    for name, _, shape in R128_TENSOR_SCHEMA:
        tensor = validated[name]
        raw = np.asarray(tensor.numpy(), dtype="<f4", order="C").tobytes(order="C")
        _framed_update(digest, name.encode("utf-8"))
        _framed_update(digest, b"float32")
        digest.update(struct.pack("<Q", len(shape)))
        for dimension in shape:
            digest.update(struct.pack("<Q", dimension))
        _framed_update(digest, raw)
    return digest.hexdigest()


def canonical_r128_model_sha256(state: Mapping[str, Tensor]) -> str:
    """Bind logical state to the established nine-field architecture config."""

    config_bytes = json.dumps(
        asdict(R128Config()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    state_sha256 = canonical_r128_state_sha256(state).encode("ascii")
    digest = hashlib.sha256()
    digest.update(_MODEL_STATE_DOMAIN)
    _framed_update(digest, config_bytes)
    _framed_update(digest, state_sha256)
    return digest.hexdigest()


def checkpoint_tensor_records(state: Mapping[str, Tensor]) -> list[dict[str, object]]:
    """Return the exact ordered metadata schema after validating tensor values."""

    validate_r128_state(state)
    return [
        {"name": name, "dtype": dtype, "shape": list(shape)}
        for name, dtype, shape in R128_TENSOR_SCHEMA
    ]


def _validate_contract_model(contract: NativeDiffusionV1PilotContract) -> None:
    observed = contract.table("model")
    expected = r128_model_config_document()
    if set(observed) != set(expected):
        raise ValueError("child model table does not have the exact R128 fields")
    for key, value in expected.items():
        candidate = observed[key]
        if isinstance(value, list):
            if not isinstance(candidate, tuple) or tuple(value) != candidate:
                raise ValueError(f"child model field {key} differs from R128")
        elif type(candidate) is not type(value) or candidate != value:
            raise ValueError(f"child model field {key} differs from R128")
    tensors = tuple((item.name, item.dtype, item.shape) for item in contract.checkpoint_tensors)
    if tensors != R128_TENSOR_SCHEMA:
        raise ValueError("child checkpoint tensor schema differs from R128")


def _validate_inputs(
    *,
    tokens: Tensor,
    attention_mask: Tensor,
    levels: Tensor,
    lengths: Tensor,
    expected_device: torch.device,
) -> None:
    _tensor(tokens, label="tokens", dtype=torch.int64, rank=2)
    _tensor(attention_mask, label="attention_mask", dtype=torch.bool, rank=2)
    _tensor(levels, label="levels", dtype=torch.int64, rank=1)
    _tensor(lengths, label="lengths", dtype=torch.int64, rank=1)
    if tokens.shape != attention_mask.shape:
        raise ValueError("tokens and attention_mask must have identical shapes")
    batch, width = tokens.shape
    if batch <= 0 or width <= 0 or width > 50:
        raise ValueError("model input axes are empty or too wide")
    if levels.shape != (batch,) or lengths.shape != (batch,):
        raise ValueError("levels and lengths must contain one value per row")
    if any(value.device != expected_device for value in (tokens, attention_mask, levels, lengths)):
        raise ValueError("all model inputs must share the model device")
    if bool(torch.any((tokens < 0) | (tokens >= INPUT_VOCABULARY_SIZE)).item()):
        raise ValueError("tokens contain an index outside the 22-token vocabulary")
    if bool(torch.any((levels < 1) | (levels > 64)).item()):
        raise ValueError("levels must lie in 1..64")
    if bool(torch.any((lengths < 8) | (lengths > width)).item()):
        raise ValueError("lengths must lie in 8..input_width")
    positions = torch.arange(width, device=expected_device).unsqueeze(0)
    if not torch.equal(attention_mask, positions < lengths.unsqueeze(1)):
        raise ValueError("attention_mask must be a contiguous prefix matching lengths")
    if bool(torch.any(attention_mask & tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("valid input positions cannot contain PAD")
    if bool(torch.any(~attention_mask & ~tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("padding input positions must contain PAD")


def _tensor(value: object, *, label: str, dtype: torch.dtype, rank: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    if value.ndim != rank:
        raise ValueError(f"{label} must have rank {rank}")
    return value


def _framed_update(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little"))
    digest.update(value)

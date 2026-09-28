"""PyTorch denoiser and checkpoint contract for native categorical diffusion.

Torch is deliberately isolated in this module.  Importing
``amp_challenge.generators.diffusion`` therefore remains lightweight for the
competition entry point and for CPU-only data tooling.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import struct
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_VOCABULARY_SIZE = 22
RESIDUE_VOCABULARY_SIZE = 20
PAD_TOKEN_INDEX = 20
MASK_TOKEN_INDEX = 21
_INITIALIZATION_STD = 0.02
_LOGICAL_STATE_DOMAIN = b"amp-native-denoiser-logical-state-v1\x00"
_MODEL_LOGICAL_DOMAIN = b"amp-native-denoiser-model-logical-v1\x00"
_MAX_CHECKPOINT_BYTES = 1 << 30


@dataclass(frozen=True, slots=True)
class NativeDenoiserConfig:
    """Architecture contract for the first native unconditional denoiser."""

    layers: int = 4
    hidden_dim: int = 256
    attention_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.10
    layer_norm_epsilon: float = 1e-5
    min_length: int = 8
    max_length: int = 50
    levels: int = 64

    def __post_init__(self) -> None:
        integer_fields = {
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "attention_heads": self.attention_heads,
            "ffn_dim": self.ffn_dim,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "levels": self.levels,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot exceed max_length")
        if type(self.dropout) is not float or not np.isfinite(self.dropout):
            raise ValueError("dropout must be a finite float")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if type(self.layer_norm_epsilon) is not float or not np.isfinite(self.layer_norm_epsilon):
            raise ValueError("layer_norm_epsilon must be a finite float")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class MaskedTokenObjective:
    """A sequence-balanced masked-token objective and its row diagnostics."""

    loss: Tensor
    row_losses: Tensor
    row_accuracies: Tensor
    selected_counts: Tensor


@dataclass(frozen=True, slots=True)
class CheckpointHashes:
    """Physical and logical identities of one safe tensor checkpoint."""

    file_sha256: str
    logical_state_sha256: str


class NativeDenoiser(nn.Module):
    """Pre-layer-normalized Transformer for absorbing-mask denoising."""

    def __init__(self, config: NativeDenoiserConfig | None = None) -> None:
        super().__init__()
        if config is not None and not isinstance(config, NativeDenoiserConfig):
            raise TypeError("config must be a NativeDenoiserConfig")
        self.config = config or NativeDenoiserConfig()
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
        """The output weight tied to the twenty residue-token embeddings."""

        return self.token_embedding.weight[:RESIDUE_VOCABULARY_SIZE]

    def reset_parameters(self) -> None:
        """Apply the fully explicit v0 initialization policy."""

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=_INITIALIZATION_STD)
                    if module.padding_idx is not None:
                        module.weight[module.padding_idx].zero_()
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.normal_(module.in_proj_weight, mean=0.0, std=_INITIALIZATION_STD)
                    if module.in_proj_bias is not None:
                        module.in_proj_bias.zero_()
                    if module.bias_k is not None:
                        module.bias_k.zero_()
                    if module.bias_v is not None:
                        module.bias_v.zero_()
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=_INITIALIZATION_STD)
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
        """Predict clean residue logits for a batch of corrupted peptides."""

        _validate_model_inputs(
            tokens=tokens,
            attention_mask=attention_mask,
            levels=levels,
            lengths=lengths,
            config=self.config,
            expected_device=self.token_embedding.weight.device,
        )
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("NativeDenoiser parameters must remain float32")

        batch_size, sequence_width = tokens.shape
        positions = torch.arange(sequence_width, device=tokens.device, dtype=torch.long)
        hidden = self.token_embedding(tokens)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        length_indices = lengths - self.config.min_length
        hidden = hidden + self.length_embedding(length_indices).reshape(batch_size, 1, -1)
        hidden = hidden + self.timestep_embedding(levels).reshape(batch_size, 1, -1)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        logits = F.linear(hidden, self.residue_output_weight, self.residue_output_bias)
        return logits.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


def configure_deterministic_runtime(seed: int) -> dict[str, Any]:
    """Configure the strict FP32, math-attention runtime used by v0.

    The cuBLAS workspace setting is established before any CUDA seeding.  A
    conflicting caller-provided setting is rejected instead of silently
    weakening the reproducibility contract.
    """

    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with the v0 contract")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    numpy_legacy_seed = seed % 2**32
    np.random.seed(numpy_legacy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    return {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdpa": torch.backends.cuda.math_sdp_enabled(),
        "memory_efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        "numpy_legacy_seed": numpy_legacy_seed,
        "seed": seed,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def masked_token_objective(
    logits: Tensor,
    clean_tokens: Tensor,
    selected_mask: Tensor,
    attention_mask: Tensor,
    *,
    corrupted_tokens: Tensor,
) -> MaskedTokenObjective:
    """Compute equal-per-sequence cross entropy over selected masked tokens."""

    _validate_masked_objective_inputs(
        logits=logits,
        clean_tokens=clean_tokens,
        selected_mask=selected_mask,
        attention_mask=attention_mask,
        corrupted_tokens=corrupted_tokens,
    )
    batch_size, sequence_width = clean_tokens.shape
    safe_targets = torch.where(selected_mask, clean_tokens, torch.zeros_like(clean_tokens))
    token_losses = F.cross_entropy(
        logits.reshape(batch_size * sequence_width, RESIDUE_VOCABULARY_SIZE),
        safe_targets.reshape(batch_size * sequence_width),
        reduction="none",
    ).reshape(batch_size, sequence_width)
    selected_float = selected_mask.to(dtype=logits.dtype)
    selected_counts = selected_mask.sum(dim=1)
    row_losses = (token_losses * selected_float).sum(dim=1) / selected_counts.to(logits.dtype)
    correct = logits.argmax(dim=-1).eq(clean_tokens) & selected_mask
    row_accuracies = correct.sum(dim=1).to(logits.dtype) / selected_counts.to(logits.dtype)
    return MaskedTokenObjective(
        loss=row_losses.mean(),
        row_losses=row_losses,
        row_accuracies=row_accuracies,
        selected_counts=selected_counts,
    )


def canonical_logical_state_hash(state: Mapping[str, Tensor]) -> str:
    """Hash sorted tensor names, dtypes, shapes, and little-endian bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a non-empty tensor mapping")
    names = list(state)
    if any(type(name) is not str or not name for name in names):
        raise ValueError("state tensor names must be non-empty strings")
    names.sort()
    digest = hashlib.sha256()
    digest.update(_LOGICAL_STATE_DOMAIN)
    digest.update(struct.pack("<Q", len(names)))
    for name in names:
        tensor = state[name]
        _validate_hashable_tensor(tensor, name=name)
        dtype_name = _canonical_dtype_name(tensor.dtype)
        raw = _little_endian_tensor_bytes(tensor)
        _update_framed(digest, name.encode("utf-8"))
        _update_framed(digest, dtype_name.encode("ascii"))
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        _update_framed(digest, raw)
    return digest.hexdigest()


def canonical_model_logical_hash(model: NativeDenoiser) -> str:
    """Bind tensor state to every architecture value that gives it meaning."""

    if not isinstance(model, NativeDenoiser):
        raise TypeError("model must be a NativeDenoiser")
    return _model_and_state_hash(model.config, _canonical_model_state(model))


def _model_and_state_hash(
    config: NativeDenoiserConfig,
    state: Mapping[str, Tensor],
) -> str:
    config_payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    state_hash = canonical_logical_state_hash(state).encode("ascii")
    digest = hashlib.sha256()
    digest.update(_MODEL_LOGICAL_DOMAIN)
    _update_framed(digest, config_payload)
    _update_framed(digest, state_hash)
    return digest.hexdigest()


def save_safetensors_checkpoint(
    model: NativeDenoiser,
    path: str | os.PathLike[str],
) -> CheckpointHashes:
    """Atomically publish a new, read-only safetensors model checkpoint."""

    if not isinstance(model, NativeDenoiser):
        raise TypeError("model must be a NativeDenoiser")
    state = _canonical_model_state(model)
    logical_hash = _model_and_state_hash(model.config, state)
    try:
        from safetensors.torch import save
    except ImportError as error:  # pragma: no cover - exercised without the optional extra
        raise RuntimeError("safetensors is required to save a checkpoint") from error
    payload = save(state, metadata=None)
    if not payload or len(payload) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("serialized checkpoint has an invalid size")
    destination = Path(path)
    _atomic_write_new(destination, payload)
    return CheckpointHashes(
        file_sha256=hashlib.sha256(payload).hexdigest(),
        logical_state_sha256=logical_hash,
    )


def load_safetensors_checkpoint(
    model: NativeDenoiser,
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str | None = None,
    expected_logical_state_sha256: str | None = None,
) -> CheckpointHashes:
    """Load a safe checkpoint after exact schema, hash, and finiteness checks."""

    if not isinstance(model, NativeDenoiser):
        raise TypeError("model must be a NativeDenoiser")
    _validate_optional_sha256(expected_file_sha256, label="expected_file_sha256")
    _validate_optional_sha256(
        expected_logical_state_sha256,
        label="expected_logical_state_sha256",
    )
    payload = _read_regular_file(Path(path))
    file_hash = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and file_hash != expected_file_sha256:
        raise ValueError("checkpoint file SHA-256 does not match the expected digest")
    try:
        from safetensors.torch import load

        loaded = load(payload)
    except Exception as error:
        raise ValueError("checkpoint is not valid safetensors data") from error
    state = _validated_loaded_state(model, loaded)
    logical_hash = _model_and_state_hash(model.config, state)
    if expected_logical_state_sha256 is not None and logical_hash != expected_logical_state_sha256:
        raise ValueError("checkpoint logical-state SHA-256 does not match the expected digest")
    model.load_state_dict(state, strict=True)
    return CheckpointHashes(
        file_sha256=file_hash,
        logical_state_sha256=logical_hash,
    )


def _validate_model_inputs(
    *,
    tokens: Tensor,
    attention_mask: Tensor,
    levels: Tensor,
    lengths: Tensor,
    config: NativeDenoiserConfig,
    expected_device: torch.device,
) -> None:
    _require_tensor(tokens, name="tokens", dtype=torch.long, ndim=2)
    _require_tensor(attention_mask, name="attention_mask", dtype=torch.bool, ndim=2)
    _require_tensor(levels, name="levels", dtype=torch.long, ndim=1)
    _require_tensor(lengths, name="lengths", dtype=torch.long, ndim=1)
    if tokens.shape != attention_mask.shape:
        raise ValueError("tokens and attention_mask must have identical shapes")
    batch_size, sequence_width = tokens.shape
    if batch_size <= 0 or sequence_width <= 0:
        raise ValueError("tokens must contain a non-empty batch and sequence axis")
    if sequence_width > config.max_length:
        raise ValueError("token width exceeds configured max_length")
    if levels.shape != (batch_size,) or lengths.shape != (batch_size,):
        raise ValueError("levels and lengths must contain exactly one value per row")
    tensors = (tokens, attention_mask, levels, lengths)
    if any(tensor.device != expected_device for tensor in tensors):
        raise ValueError("all inputs must be on the same device as the model")
    if bool(torch.any((tokens < 0) | (tokens >= INPUT_VOCABULARY_SIZE)).item()):
        raise ValueError("tokens contain an index outside the 22-token vocabulary")
    if bool(torch.any((levels < 1) | (levels > config.levels)).item()):
        raise ValueError("diffusion levels must lie in [1, levels]")
    if bool(torch.any((lengths < config.min_length) | (lengths > sequence_width)).item()):
        raise ValueError("lengths must lie in [min_length, token width]")
    expected_mask = torch.arange(sequence_width, device=tokens.device).unsqueeze(0)
    expected_mask = expected_mask < lengths.unsqueeze(1)
    if not torch.equal(attention_mask, expected_mask):
        raise ValueError("attention_mask must be a non-empty prefix matching lengths")
    if bool(torch.any(attention_mask & tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("valid prefix positions cannot contain PAD")
    if bool(torch.any(~attention_mask & ~tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("positions outside the valid prefix must contain PAD")


def _validate_masked_objective_inputs(
    *,
    logits: Tensor,
    clean_tokens: Tensor,
    selected_mask: Tensor,
    attention_mask: Tensor,
    corrupted_tokens: Tensor,
) -> None:
    _require_tensor(logits, name="logits", dtype=torch.float32, ndim=3)
    _require_tensor(clean_tokens, name="clean_tokens", dtype=torch.long, ndim=2)
    _require_tensor(selected_mask, name="selected_mask", dtype=torch.bool, ndim=2)
    _require_tensor(attention_mask, name="attention_mask", dtype=torch.bool, ndim=2)
    batch_size, sequence_width = clean_tokens.shape
    if batch_size <= 0 or sequence_width <= 0:
        raise ValueError("objective inputs must have non-empty batch and sequence axes")
    if logits.shape != (batch_size, sequence_width, RESIDUE_VOCABULARY_SIZE):
        raise ValueError("logits must have shape [batch, width, 20]")
    if selected_mask.shape != clean_tokens.shape or attention_mask.shape != clean_tokens.shape:
        raise ValueError("clean tokens and both masks must have identical shapes")
    inputs = [clean_tokens, selected_mask, attention_mask]
    _require_tensor(corrupted_tokens, name="corrupted_tokens", dtype=torch.long, ndim=2)
    if corrupted_tokens.shape != clean_tokens.shape:
        raise ValueError("corrupted_tokens must match clean_tokens")
    inputs.append(corrupted_tokens)
    if any(tensor.device != logits.device for tensor in inputs):
        raise ValueError("objective tensors must be on one device")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    lengths = attention_mask.sum(dim=1)
    if bool(torch.any(lengths < 1).item()):
        raise ValueError("every objective row must contain a valid token")
    expected_mask = torch.arange(sequence_width, device=logits.device).unsqueeze(0)
    if not torch.equal(attention_mask, expected_mask < lengths.unsqueeze(1)):
        raise ValueError("attention_mask must be a contiguous non-empty prefix")
    if bool(
        torch.any(
            attention_mask & ((clean_tokens < 0) | (clean_tokens >= RESIDUE_VOCABULARY_SIZE))
        ).item()
    ):
        raise ValueError("valid clean positions must contain residue tokens")
    if bool(torch.any(~attention_mask & ~clean_tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("clean positions outside the valid prefix must contain PAD")
    if bool(torch.any(selected_mask & ~attention_mask).item()):
        raise ValueError("selected_mask cannot select padding")
    selected_counts = selected_mask.sum(dim=1)
    if bool(torch.any(selected_counts < 1).item()):
        raise ValueError("selected_mask must select at least one token in every row")
    if bool(
        torch.any(
            attention_mask
            & (
                (corrupted_tokens < 0)
                | (corrupted_tokens >= INPUT_VOCABULARY_SIZE)
                | corrupted_tokens.eq(PAD_TOKEN_INDEX)
            )
        ).item()
    ):
        raise ValueError("valid corrupted positions must contain residue or MASK tokens")
    if bool(torch.any(~attention_mask & ~corrupted_tokens.eq(PAD_TOKEN_INDEX)).item()):
        raise ValueError("corrupted positions outside the prefix must contain PAD")
    if not torch.equal(corrupted_tokens.eq(MASK_TOKEN_INDEX), selected_mask):
        raise ValueError("selected_mask must exactly identify corrupted MASK positions")
    if bool(
        torch.any((~selected_mask & attention_mask) & corrupted_tokens.ne(clean_tokens)).item()
    ):
        raise ValueError("unselected corrupted positions must equal the clean tokens")


def _require_tensor(tensor: object, *, name: str, dtype: torch.dtype, ndim: int) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")


def _canonical_model_state(model: NativeDenoiser) -> dict[str, Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    return _validated_loaded_state(model, state)


def _validated_loaded_state(
    model: NativeDenoiser,
    loaded: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint state must be a tensor mapping")
    expected = model.state_dict()
    if set(loaded) != set(expected):
        missing = sorted(set(expected) - set(loaded))
        extra = sorted(set(loaded) - set(expected))
        raise ValueError(f"checkpoint state schema mismatch: missing={missing}, extra={extra}")
    validated: dict[str, Tensor] = {}
    for name in sorted(expected):
        tensor = loaded[name]
        if not isinstance(tensor, Tensor):
            raise ValueError(f"checkpoint entry {name!r} is not a tensor")
        if tensor.shape != expected[name].shape:
            raise ValueError(f"checkpoint tensor {name!r} has an unexpected shape")
        if tensor.dtype != torch.float32 or expected[name].dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name!r} must be float32")
        if tensor.layout != torch.strided:
            raise ValueError(f"checkpoint tensor {name!r} must use strided layout")
        canonical = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(canonical).all().item()):
            raise ValueError(f"checkpoint tensor {name!r} contains a non-finite value")
        validated[name] = canonical
    return validated


def _validate_hashable_tensor(tensor: object, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"state entry {name!r} must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"state entry {name!r} must use strided layout")
    _canonical_dtype_name(tensor.dtype)
    if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"state entry {name!r} contains a non-finite value")


def _canonical_dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.bool: "bool",
        torch.uint8: "uint8",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported state tensor dtype: {dtype}") from error


def _little_endian_tensor_bytes(tensor: Tensor) -> bytes:
    canonical = tensor.detach().cpu().contiguous()
    raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    element_size = canonical.element_size()
    if sys.byteorder == "little" or element_size == 1:
        return raw
    swapped = bytearray(len(raw))  # pragma: no cover - cluster and CI are little-endian
    for offset in range(0, len(raw), element_size):
        swapped[offset : offset + element_size] = raw[offset : offset + element_size][::-1]
    return bytes(swapped)


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def _atomic_write_new(destination: Path, payload: bytes) -> None:
    destination = destination.absolute()
    parent = destination.parent
    _reject_symlink_chain(parent)
    if not parent.is_dir():
        raise ValueError("checkpoint parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    linked = False
    linked_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        source_stat = temporary.stat(follow_symlinks=False)
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        linked_identity = (source_stat.st_dev, source_stat.st_ino)
        published_stat = destination.stat(follow_symlinks=False)
        if (published_stat.st_dev, published_stat.st_ino) != linked_identity:
            raise RuntimeError("checkpoint publication changed inode identity")
        temporary.unlink()
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if linked and linked_identity is not None:
            try:
                target_stat = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (target_stat.st_dev, target_stat.st_ino) == linked_identity:
                    destination.unlink()
        raise
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_regular_file(path: Path) -> bytes:
    absolute = path.absolute()
    _reject_symlink_chain(absolute)
    try:
        named_before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot inspect checkpoint: {absolute}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError("checkpoint must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"cannot open checkpoint: {absolute}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint must be a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint has an invalid size")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("checkpoint ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("checkpoint grew while it was read")
        after = os.fstat(descriptor)
        try:
            named_after = absolute.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("checkpoint changed while it was read") from error
        _reject_symlink_chain(absolute)
        fingerprints = {
            _file_fingerprint(value) for value in (named_before, before, after, named_after)
        }
        if len(fingerprints) != 1:
            raise ValueError("checkpoint changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_symlink_chain(path: Path) -> None:
    candidate = path
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            raise ValueError(f"checkpoint path ancestor does not exist: {candidate}") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"checkpoint path traverses a symbolic link: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _validate_optional_sha256(value: str | None, *, label: str) -> None:
    if value is None:
        return
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")

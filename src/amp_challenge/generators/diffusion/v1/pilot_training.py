"""One-step/few-step training kernel for the frozen native v1 R128 pilot.

This is deliberately not a command-line producer or publication layer.  The
kernel exposes the exact numerical operations needed by that later layer while
remaining small enough to exercise deterministically on CPU fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim.adam as torch_adam_module
from numpy.typing import NDArray
from safetensors.torch import load as load_safetensors
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import optimizer as torch_optimizer_module

from amp_challenge.generators.diffusion.v1.pilot_bridge import (
    count_prior_training_bridge,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    PilotFoldContract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    ALPHABET,
    AuthenticatedCountPrior,
    PilotTrainingProjection,
    PilotTrainingRow,
    SealedCountPrior,
    _read_regular_bytes,
    seal_count_prior_npz,
)
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD,
    RESIDUE_VOCABULARY_SIZE,
    R128Denoiser,
    assert_r128_deterministic_runtime,
    assert_r128_model_execution_surface,
    build_r128_model_from_contract,
    canonical_r128_model_sha256,
    checkpoint_tensor_records,
    establish_r128_deterministic_runtime,
    r128_model_config_document,
    validate_r128_state,
    verify_r128_production_environment,
)
from amp_challenge.generators.diffusion.v1.pilot_rng import (
    MASK_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
    TRAINING_ROOT_SEED,
    CorruptedBatch,
    assert_pilot_rng_execution_surface,
    corrupt_training_batch,
    initialization_seed,
    reseed_torch_for_model_dropout,
    timestep_levels,
    weighted_minibatch_rows,
)

_SCHEDULE_DOMAIN = b"amp-native-diffusion-learning-rate-schedule-v1\0"
_BATCH_DOMAIN = b"amp-challenge/native-categorical-diffusion/r128-training-batch/v1\0"
_LENGTH_EDGES = (8, 15, 20, 25, 33, 51)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OPTIMIZER_STATE_DOMAIN = b"amp-native-diffusion-r128-optimizer-state-v1\0"
_COUNT_TENSOR_DOMAIN = b"amp-native-diffusion-r128-count-tensor-v1\0"
_SESSION_CONSTRUCTION_TOKEN = object()
_CHECKPOINT_SEAL_CAPABILITY = object()
_ADAMW_IMPORTED_STEP_METHOD = torch.optim.AdamW.step
_ADAMW_STEP_METHOD = (
    _ADAMW_IMPORTED_STEP_METHOD
    if getattr(_ADAMW_IMPORTED_STEP_METHOD, "hooked", None) is True
    else None
)
_ADAMW_RAW_STEP_METHOD = (
    getattr(_ADAMW_IMPORTED_STEP_METHOD, "__wrapped__", None)
    if _ADAMW_STEP_METHOD is not None
    else _ADAMW_IMPORTED_STEP_METHOD
)
_ADAMW_RAW_STEP_CODE = getattr(_ADAMW_RAW_STEP_METHOD, "__code__", None)
_ADAMW_CORE_STEP_METHOD = getattr(_ADAMW_RAW_STEP_METHOD, "__wrapped__", None)
_ADAMW_CORE_STEP_CODE = getattr(_ADAMW_CORE_STEP_METHOD, "__code__", None)
_ADAMW_CORE_STEP_DEFAULTS = getattr(_ADAMW_CORE_STEP_METHOD, "__defaults__", None)
_ADAMW_CORE_STEP_KWDEFAULTS = (
    None
    if getattr(_ADAMW_CORE_STEP_METHOD, "__kwdefaults__", None) is None
    else tuple(_ADAMW_CORE_STEP_METHOD.__kwdefaults__.items())
)
_ADAMW_RAW_STEP_CLOSURE = tuple(
    cell.cell_contents for cell in (_ADAMW_RAW_STEP_METHOD.__closure__ or ())
)
_ADAMW_RAW_STEP_CLOSURE_CODES = tuple(
    getattr(value, "__code__", None) for value in _ADAMW_RAW_STEP_CLOSURE
)
_OPTIMIZER_PROFILE_HOOK_STEP_METHOD = torch.optim.Optimizer.profile_hook_step
_ADAMW_PROFILE_STEP_CODE = torch.optim.Optimizer.profile_hook_step(_ADAMW_RAW_STEP_METHOD).__code__
_ADAMW_ZERO_GRAD_METHOD = torch.optim.AdamW.zero_grad
_ADAMW_STATE_DICT_METHOD = torch.optim.AdamW.state_dict
_ADAMW_LOAD_STATE_DICT_METHOD = torch.optim.AdamW.load_state_dict
_OPTIMIZER_ZERO_GRAD_METHOD = torch.optim.Optimizer.zero_grad
_OPTIMIZER_STATE_DICT_METHOD = torch.optim.Optimizer.state_dict
_OPTIMIZER_LOAD_STATE_DICT_METHOD = torch.optim.Optimizer.load_state_dict
_COUNT_PRIOR_TRAINING_BRIDGE_FUNCTION = count_prior_training_bridge
_COUNT_PRIOR_TRAINING_BRIDGE_CODE = count_prior_training_bridge.__code__
_BUILD_R128_MODEL_FROM_CONTRACT_FUNCTION = build_r128_model_from_contract
_BUILD_R128_MODEL_FROM_CONTRACT_CODE = build_r128_model_from_contract.__code__
_SEAL_COUNT_PRIOR_NPZ_FUNCTION = seal_count_prior_npz
_SEAL_COUNT_PRIOR_NPZ_CODE = seal_count_prior_npz.__code__
_COUNT_PRIOR_FROM_PROJECTION_FUNCTION = AuthenticatedCountPrior.from_projection.__func__
_COUNT_PRIOR_FROM_PROJECTION_CODE = _COUNT_PRIOR_FROM_PROJECTION_FUNCTION.__code__
_WEIGHTED_MINIBATCH_ROWS_FUNCTION = weighted_minibatch_rows
_WEIGHTED_MINIBATCH_ROWS_CODE = weighted_minibatch_rows.__code__
_TIMESTEP_LEVELS_FUNCTION = timestep_levels
_TIMESTEP_LEVELS_CODE = timestep_levels.__code__
_CORRUPT_TRAINING_BATCH_FUNCTION = corrupt_training_batch
_CORRUPT_TRAINING_BATCH_CODE = corrupt_training_batch.__code__
_RESEED_MODEL_DROPOUT_FUNCTION = reseed_torch_for_model_dropout
_RESEED_MODEL_DROPOUT_CODE = reseed_torch_for_model_dropout.__code__
_INITIALIZATION_SEED_FUNCTION = initialization_seed
_INITIALIZATION_SEED_CODE = initialization_seed.__code__
_FUNCTIONAL_CROSS_ENTROPY = F.cross_entropy
_FUNCTIONAL_CROSS_ENTROPY_CODE = getattr(F.cross_entropy, "__code__", None)
_CLIP_GRAD_NORM_FUNCTION = torch.nn.utils.clip_grad_norm_
_CLIP_GRAD_NORM_CODE = getattr(torch.nn.utils.clip_grad_norm_, "__code__", None)
_ADAM_FUNCTIONAL = torch_adam_module.adam
_ADAM_FUNCTIONAL_CODE = getattr(torch_adam_module.adam, "__code__", None)
_ADAM_FUNCTIONAL_CLOSURE = tuple(
    cell.cell_contents for cell in (torch_adam_module.adam.__closure__ or ())
)
_ADAM_FUNCTIONAL_CLOSURE_CODES = tuple(
    getattr(value, "__code__", None) for value in _ADAM_FUNCTIONAL_CLOSURE
)
_SINGLE_TENSOR_ADAM_FUNCTION = torch_adam_module._single_tensor_adam
_SINGLE_TENSOR_ADAM_CODE = getattr(torch_adam_module._single_tensor_adam, "__code__", None)
_R128_INITIAL_MODEL_SHA256 = tuple(R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD)
_ASSERT_PILOT_RNG_SURFACE_FUNCTION = assert_pilot_rng_execution_surface
_ASSERT_PILOT_RNG_SURFACE_CODE = assert_pilot_rng_execution_surface.__code__
_LOAD_SAFETENSORS_FUNCTION = load_safetensors
_LOAD_SAFETENSORS_CODE = getattr(load_safetensors, "__code__", None)
_CHECKPOINT_TENSOR_RECORDS_FUNCTION = checkpoint_tensor_records
_CHECKPOINT_TENSOR_RECORDS_CODE = checkpoint_tensor_records.__code__
_R128_MODEL_CONFIG_DOCUMENT_FUNCTION = r128_model_config_document
_R128_MODEL_CONFIG_DOCUMENT_CODE = r128_model_config_document.__code__
_VALIDATE_R128_STATE_FUNCTION = validate_r128_state
_VALIDATE_R128_STATE_CODE = validate_r128_state.__code__


@dataclass(frozen=True, slots=True)
class PilotTrainingRecipe:
    """Resolved training values; defaults are the exact R128 pilot recipe."""

    batch_sequences: int = 128
    max_steps: int = 4000
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.90, 0.95)
    epsilon: float = 1e-8
    weight_decay: float = 0.05
    warmup_steps: int = 200
    final_learning_rate: float = 2e-5
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    context_dropout: float = 0.15
    checkpoint_steps: tuple[int, ...] = (250, 500, 1000, 2000, 4000)

    def __post_init__(self) -> None:
        for label, value in (
            ("batch_sequences", self.batch_sequences),
            ("max_steps", self.max_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive exact integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be an exact integer in [0,max_steps)")
        if (
            type(self.betas) is not tuple
            or len(self.betas) != 2
            or any(type(value) is not float or not 0.0 < value < 1.0 for value in self.betas)
        ):
            raise ValueError("betas must contain two exact floats in (0,1)")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("epsilon", self.epsilon),
            ("weight_decay", self.weight_decay),
            ("final_learning_rate", self.final_learning_rate),
            ("gradient_clip_norm", self.gradient_clip_norm),
        ):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be a positive finite exact float")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final_learning_rate cannot exceed learning_rate")
        for label, value in (
            ("label_smoothing", self.label_smoothing),
            ("context_dropout", self.context_dropout),
        ):
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{label} must be a finite exact float in [0,1)")
        if (
            type(self.checkpoint_steps) is not tuple
            or not self.checkpoint_steps
            or any(
                type(step) is not int or not 1 <= step <= self.max_steps
                for step in self.checkpoint_steps
            )
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
        ):
            raise ValueError("checkpoint_steps must be unique ascending steps within max_steps")


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    """A complete deterministic NumPy minibatch before its device transfer."""

    rows: tuple[PilotTrainingRow, ...]
    global_draw_ordinals: tuple[int, ...]
    clean_tokens: NDArray[np.int64]
    attention_mask: NDArray[np.bool_]
    levels: NDArray[np.int64]
    corruption: CorruptedBatch

    def __post_init__(self) -> None:
        batch = len(self.rows)
        if batch <= 0 or len(self.global_draw_ordinals) != batch:
            raise ValueError("prepared batch rows and ordinals must be non-empty and aligned")
        if any(type(row) is not PilotTrainingRow for row in self.rows):
            raise TypeError("prepared batch may contain only PilotTrainingRow values")
        if any(type(value) is not int or value < 0 for value in self.global_draw_ordinals):
            raise ValueError("prepared batch ordinals must be non-negative exact integers")
        if (
            type(self.clean_tokens) is not np.ndarray
            or self.clean_tokens.dtype != np.dtype("<i8")
            or self.clean_tokens.shape != (batch, 50)
            or not self.clean_tokens.flags.c_contiguous
        ):
            raise TypeError("clean_tokens must be exact int64 [batch,50]")
        if (
            type(self.attention_mask) is not np.ndarray
            or self.attention_mask.dtype != np.dtype("|b1")
            or self.attention_mask.shape != (batch, 50)
            or not self.attention_mask.flags.c_contiguous
        ):
            raise TypeError("attention_mask must be exact bool [batch,50]")
        if (
            type(self.levels) is not np.ndarray
            or self.levels.dtype != np.dtype("<i8")
            or self.levels.shape != (batch,)
            or not self.levels.flags.c_contiguous
        ):
            raise TypeError("levels must be exact int64 [batch]")
        if type(self.corruption) is not CorruptedBatch:
            raise TypeError("corruption must be a CorruptedBatch")
        if self.corruption.tokens.shape != self.clean_tokens.shape:
            raise ValueError("corruption and clean token arrays must align")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_BATCH_DOMAIN)
        for offset, row in enumerate(self.rows):
            ordinal = self.global_draw_ordinals[offset]
            digest.update(ordinal.to_bytes(8, "big"))
            digest.update(bytes.fromhex(row.sequence_id))
            digest.update(int(self.levels[offset]).to_bytes(8, "big"))
            digest.update(
                np.packbits(self.corruption.scheduled_mask[offset], bitorder="little").tobytes()
            )
            digest.update(
                np.packbits(
                    self.corruption.context_dropout_mask[offset], bitorder="little"
                ).tobytes()
            )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PilotObjective:
    """Sequence-balanced label-smoothed objective and per-row diagnostics."""

    loss: Tensor
    row_losses: Tensor
    row_accuracies: Tensor
    selected_counts: Tensor
    combined_logits: Tensor


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    step: int
    learning_rate: float
    loss: float
    mean_row_accuracy: float
    gradient_norm_before_clipping: float
    selected_tokens: int
    model_dropout_seed: int
    batch_sha256: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("step must be a positive exact integer")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("loss", self.loss),
            ("mean_row_accuracy", self.mean_row_accuracy),
            ("gradient_norm_before_clipping", self.gradient_norm_before_clipping),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite exact float")
        if type(self.selected_tokens) is not int or self.selected_tokens <= 0:
            raise ValueError("selected_tokens must be a positive exact integer")
        if type(self.model_dropout_seed) is not int or not 0 <= self.model_dropout_seed < 2**64:
            raise ValueError("model_dropout_seed must be uint64")
        if type(self.batch_sha256) is not str or _SHA256_RE.fullmatch(self.batch_sha256) is None:
            raise ValueError("batch_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class _SealedCheckpointReceipt:
    step: int
    checkpoint_file_sha256: str
    checkpoint_logical_state_sha256: str
    checkpoint_metadata_sha256: str
    checkpoint_bytes: bytes
    metadata_bytes: bytes
    checkpoint_path: str
    metadata_path: str
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _CHECKPOINT_SEAL_CAPABILITY:
            raise RuntimeError("checkpoint receipt requires the internal seal capability")
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("sealed checkpoint receipt step must be positive")
        for digest in (
            self.checkpoint_file_sha256,
            self.checkpoint_logical_state_sha256,
            self.checkpoint_metadata_sha256,
        ):
            if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("sealed checkpoint receipt digest is invalid")
        if type(self.checkpoint_bytes) is not bytes or not self.checkpoint_bytes:
            raise ValueError("sealed checkpoint receipt bytes are invalid")
        if type(self.metadata_bytes) is not bytes or not self.metadata_bytes:
            raise ValueError("sealed checkpoint metadata bytes are invalid")
        if hashlib.sha256(self.checkpoint_bytes).hexdigest() != self.checkpoint_file_sha256:
            raise ValueError("sealed checkpoint receipt physical digest differs")
        if hashlib.sha256(self.metadata_bytes).hexdigest() != self.checkpoint_metadata_sha256:
            raise ValueError("sealed checkpoint receipt metadata digest differs")
        for label, value in (
            ("checkpoint_path", self.checkpoint_path),
            ("metadata_path", self.metadata_path),
        ):
            if type(value) is not str or not value or not os.path.isabs(value):
                raise ValueError(f"sealed checkpoint receipt {label} must be absolute")
        reopened_checkpoint = _read_regular_bytes(
            self.checkpoint_path,
            maximum_bytes=1 << 30,
            label="registered sealed checkpoint",
            required_mode=0o444,
        )
        reopened_metadata = _read_regular_bytes(
            self.metadata_path,
            maximum_bytes=1 << 20,
            label="registered sealed checkpoint metadata",
            required_mode=0o444,
        )
        if reopened_checkpoint != self.checkpoint_bytes or reopened_metadata != self.metadata_bytes:
            raise ValueError("registered sealed checkpoint pair changed on disk")


class PilotFitSession:
    """Mutable progress for exactly one authenticated fold-local R128 fit.

    Identity-bearing members are exposed read-only.  Only the successful
    optimizer-step path can advance ``completed_step``, and public boundaries
    independently validate every binding plus model/optimizer continuity.
    """

    __slots__ = (
        "_advancing",
        "_completed_step",
        "_contract",
        "_contract_object_id",
        "_count_log_probability",
        "_count_prior",
        "_count_prior_object_id",
        "_count_tensor_object_id",
        "_count_tensor_sha256",
        "_decay_names",
        "_failed",
        "_fit_identity_sha256",
        "_fold",
        "_initialization_seed",
        "_model",
        "_model_object_id",
        "_model_state_sha256",
        "_optimizer",
        "_optimizer_object_id",
        "_optimizer_state_sha256",
        "_outer_fold",
        "_production_bound",
        "_production_environment",
        "_projection",
        "_projection_object_id",
        "_recipe",
        "_sealed_checkpoints",
        "_sealed_count_prior",
        "_sealed_count_prior_object_id",
        "_zero_decay_names",
    )

    def __init__(
        self,
        *,
        contract: NativeDiffusionV1PilotContract,
        outer_fold: int,
        fold: PilotFoldContract,
        fit_identity_sha256: str,
        projection: PilotTrainingProjection,
        count_prior: AuthenticatedCountPrior,
        sealed_count_prior: SealedCountPrior | None,
        count_log_probability: Tensor,
        model: R128Denoiser,
        optimizer: torch.optim.AdamW,
        recipe: PilotTrainingRecipe,
        decay_names: tuple[str, ...],
        zero_decay_names: tuple[str, ...],
        initialization_seed: int,
        production_bound: bool,
        production_environment: dict[str, object] | None,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _SESSION_CONSTRUCTION_TOKEN:
            raise RuntimeError("PilotFitSession must be created by an authenticated factory")
        self._contract = contract
        self._outer_fold = outer_fold
        self._fold = fold
        self._fit_identity_sha256 = fit_identity_sha256
        self._projection = projection
        self._count_prior = count_prior
        self._sealed_count_prior = sealed_count_prior
        self._count_log_probability = count_log_probability
        self._model = model
        self._optimizer = optimizer
        self._recipe = recipe
        self._decay_names = decay_names
        self._zero_decay_names = zero_decay_names
        self._initialization_seed = initialization_seed
        self._completed_step = 0
        self._production_bound = production_bound
        self._production_environment = production_environment
        self._advancing = False
        self._failed = False
        self._contract_object_id = id(contract)
        self._projection_object_id = id(projection)
        self._count_prior_object_id = id(count_prior)
        self._sealed_count_prior_object_id = (
            None if sealed_count_prior is None else id(sealed_count_prior)
        )
        self._count_tensor_object_id = id(count_log_probability)
        self._model_object_id = id(model)
        self._optimizer_object_id = id(optimizer)
        self._model_state_sha256 = _CANONICAL_R128_MODEL_SHA256_FUNCTION(model.state_dict())
        self._optimizer_state_sha256 = _OPTIMIZER_STATE_SHA256_FUNCTION(
            optimizer,
            decay_names + zero_decay_names,
        )
        self._count_tensor_sha256 = _CANONICAL_COUNT_TENSOR_SHA256_FUNCTION(count_log_probability)
        self._sealed_checkpoints: tuple[_SealedCheckpointReceipt, ...] = ()

    @property
    def contract(self) -> NativeDiffusionV1PilotContract:
        return self._contract

    @property
    def outer_fold(self) -> int:
        return self._outer_fold

    @property
    def fold(self) -> PilotFoldContract:
        return self._fold

    @property
    def fit_identity_sha256(self) -> str:
        return self._fit_identity_sha256

    @property
    def projection(self) -> PilotTrainingProjection:
        return self._projection

    @property
    def count_prior(self) -> AuthenticatedCountPrior:
        return self._count_prior

    @property
    def sealed_count_prior(self) -> SealedCountPrior | None:
        return self._sealed_count_prior

    @property
    def count_log_probability(self) -> Tensor:
        return self._count_log_probability

    @property
    def model(self) -> R128Denoiser:
        return self._model

    @property
    def optimizer(self) -> torch.optim.AdamW:
        return self._optimizer

    @property
    def recipe(self) -> PilotTrainingRecipe:
        return self._recipe

    @property
    def decay_parameter_names(self) -> tuple[str, ...]:
        return self._decay_names

    @property
    def zero_decay_parameter_names(self) -> tuple[str, ...]:
        return self._zero_decay_names

    @property
    def initialization_seed(self) -> int:
        return self._initialization_seed

    @property
    def completed_step(self) -> int:
        return self._completed_step

    @property
    def production_bound(self) -> bool:
        return self._production_bound

    @property
    def production_environment(self) -> dict[str, object] | None:
        if self._production_environment is None:
            return None
        return dict(self._production_environment)

    @property
    def next_checkpoint_step(self) -> int | None:
        return next(
            (step for step in self._recipe.checkpoint_steps if step > self._completed_step),
            None,
        )

    @property
    def sealed_checkpoints(self) -> tuple[tuple[int, str, str, str], ...]:
        """Registered ``(step, physical, logical, metadata)`` receipts."""

        return tuple(
            (
                receipt.step,
                receipt.checkpoint_file_sha256,
                receipt.checkpoint_logical_state_sha256,
                receipt.checkpoint_metadata_sha256,
            )
            for receipt in self._sealed_checkpoints
        )

    @property
    def last_sealed_checkpoint_step(self) -> int:
        return self._sealed_checkpoints[-1].step if self._sealed_checkpoints else 0

    def _register_sealed_checkpoint(
        self,
        *,
        step: int,
        checkpoint_file_sha256: str,
        checkpoint_logical_state_sha256: str,
        checkpoint_metadata_sha256: str,
        checkpoint_bytes: bytes,
        metadata_bytes: bytes,
        checkpoint_path: str,
        metadata_path: str,
        _capability: object,
    ) -> None:
        if _capability is not _CHECKPOINT_SEAL_CAPABILITY:
            raise RuntimeError("checkpoint registration requires a physical-seal capability")
        receipt = _SealedCheckpointReceipt(
            step=step,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint_logical_state_sha256=checkpoint_logical_state_sha256,
            checkpoint_metadata_sha256=checkpoint_metadata_sha256,
            checkpoint_bytes=checkpoint_bytes,
            metadata_bytes=metadata_bytes,
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            _capability=_capability,
        )
        _reauthenticate_registered_checkpoint(self, receipt)
        if self._sealed_checkpoints and self._sealed_checkpoints[-1].step == step:
            if self._sealed_checkpoints[-1] != receipt:
                raise ValueError("checkpoint step was already registered with different bytes")
            return
        if step != self._completed_step or step <= self.last_sealed_checkpoint_step:
            raise ValueError("checkpoint receipt is not for the current unsealed fit state")
        if checkpoint_logical_state_sha256 != self._model_state_sha256:
            raise ValueError("checkpoint receipt logical state differs from the fit session")
        self._sealed_checkpoints = (*self._sealed_checkpoints, receipt)


def _reauthenticate_registered_checkpoint(
    session: PilotFitSession,
    receipt: _SealedCheckpointReceipt,
) -> None:
    """Rebuild production checkpoint semantics from the sealed pair itself."""

    if not session.production_bound:
        return
    _assert_training_call_surface()
    try:
        state = _VALIDATE_R128_STATE_FUNCTION(_LOAD_SAFETENSORS_FUNCTION(receipt.checkpoint_bytes))
    except Exception as error:
        raise ValueError(
            "registered production checkpoint is not valid R128 safetensors"
        ) from error
    logical = _CANONICAL_R128_MODEL_SHA256_FUNCTION(state)
    if logical != receipt.checkpoint_logical_state_sha256:
        raise ValueError("registered production checkpoint logical state differs")
    contract = session.contract
    fold = session.fold
    index = contract.checkpoint_steps.index(receipt.step)
    pilot = contract.table("pilot")
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": contract.document["artifact"],
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "fit_identity_sha256": fold.fit_identity_sha256,
        "outer_fold": session.outer_fold,
        "checkpoint_step": receipt.step,
        "variant": pilot["variant"],
        "output_mode": pilot["output_mode"],
        "model_config": _R128_MODEL_CONFIG_DOCUMENT_FUNCTION(),
        "count_prior_file_sha256": session.count_prior.sha256,
        "optimizer_step_completed": receipt.step,
        "checkpoint_file": contract.checkpoint_paths[index],
        "checkpoint_file_sha256": receipt.checkpoint_file_sha256,
        "checkpoint_logical_state_sha256": logical,
        "tensors": _CHECKPOINT_TENSOR_RECORDS_FUNCTION(state),
    }
    metadata_contract = contract.table("checkpoint_metadata")
    if (
        not isinstance(metadata_contract["fields"], tuple)
        or tuple(document) != metadata_contract["fields"]
        or metadata_contract["tensor_count"] != len(document["tensors"])
    ):
        raise ValueError("registered production checkpoint metadata schema differs")
    expected_metadata = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if expected_metadata != receipt.metadata_bytes:
        raise ValueError("registered production checkpoint metadata bindings differ")


def training_recipe_from_contract(
    contract: NativeDiffusionV1PilotContract,
) -> PilotTrainingRecipe:
    """Resolve and authenticate the exact child recipe; no defaults are silently merged."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    contract.revalidate()
    training = contract.table("training")
    pilot = contract.table("pilot")
    raw_betas = training["betas"]
    if not isinstance(raw_betas, tuple):
        raise TypeError("training.betas must be an exact frozen tuple")
    recipe = PilotTrainingRecipe(
        batch_sequences=_exact(training["batch_sequences"], int, "batch_sequences"),
        max_steps=_exact(pilot["max_steps"], int, "max_steps"),
        learning_rate=_exact(training["learning_rate"], float, "learning_rate"),
        betas=raw_betas,
        epsilon=_exact(training["epsilon"], float, "epsilon"),
        weight_decay=_exact(training["weight_decay"], float, "weight_decay"),
        warmup_steps=_exact(training["warmup_steps"], int, "warmup_steps"),
        final_learning_rate=_exact(training["final_learning_rate"], float, "final_learning_rate"),
        gradient_clip_norm=_exact(training["gradient_clip_norm"], float, "gradient_clip_norm"),
        label_smoothing=_exact(training["label_smoothing"], float, "label_smoothing"),
        context_dropout=_exact(
            training["visible_context_dropout"], float, "visible_context_dropout"
        ),
        checkpoint_steps=contract.checkpoint_steps,
    )
    if recipe != PilotTrainingRecipe():
        raise ValueError("authenticated child training table differs from the R128 pilot recipe")
    exact_requirements = {
        "sample_with_replacement": True,
        "sampling_weight_application": "weighted_draw_only",
        "optimizer": "adamw_unfused",
        "adamw_fused": False,
        "adamw_foreach": False,
        "adamw_amsgrad": False,
        "adamw_capturable": False,
        "adamw_differentiable": False,
        "adamw_maximize": False,
        "weight_decay_includes": ("matrices", "all_embeddings", "tied_residue_embedding"),
        "exclude_from_weight_decay": ("bias", "normalization"),
        "parameter_group_order": ("ascending_name_decay", "ascending_name_zero_decay"),
        "visible_context_dropout_changes_loss_mask": False,
        "gradient_accumulation_steps": 1,
        "zero_grad_set_to_none": True,
        "validation_during_training": False,
        "ema": False,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "resume_from_checkpoint_allowed": False,
    }
    for key, expected in exact_requirements.items():
        candidate = training[key]
        if type(candidate) is not type(expected) or candidate != expected:
            raise ValueError(f"authenticated child training.{key} differs from the pilot")
    return recipe


def learning_rate_for_step(recipe: PilotTrainingRecipe, step: int) -> float:
    """Return the one-indexed 200-step warmup then cosine-decay rate."""

    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    if type(step) is not int or not 1 <= step <= recipe.max_steps:
        raise ValueError("step must be an exact integer in 1..max_steps")
    if step <= recipe.warmup_steps:
        return recipe.learning_rate * step / recipe.warmup_steps
    progress = (step - recipe.warmup_steps) / (recipe.max_steps - recipe.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return recipe.final_learning_rate + (recipe.learning_rate - recipe.final_learning_rate) * cosine


def learning_rate_schedule_sha256(recipe: PilotTrainingRecipe) -> str:
    """Hash every one-indexed Python-float schedule value in the frozen framing."""

    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    digest = hashlib.sha256()
    digest.update(_SCHEDULE_DOMAIN)
    for step in range(1, recipe.max_steps + 1):
        digest.update(step.to_bytes(8, "big"))
        encoded = _LEARNING_RATE_FOR_STEP_FUNCTION(recipe, step).hex().encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_pilot_adamw(
    model: R128Denoiser,
    recipe: PilotTrainingRecipe,
) -> tuple[torch.optim.AdamW, tuple[str, ...], tuple[str, ...]]:
    """Create exact sorted decay/zero-decay groups, including every embedding in decay."""

    if type(model) is not R128Denoiser:
        raise TypeError("model must be an R128Denoiser")
    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    assert_r128_deterministic_runtime()
    decay, zero_decay, decay_names, zero_decay_names = _pilot_parameter_partition(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": recipe.weight_decay},
            {"params": zero_decay, "weight_decay": 0.0},
        ],
        lr=recipe.learning_rate,
        betas=recipe.betas,
        eps=recipe.epsilon,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    _bind_adamw_step_surface()
    return optimizer, decay_names, zero_decay_names


def _pilot_parameter_partition(
    model: R128Denoiser,
) -> tuple[list[Tensor], list[Tensor], tuple[str, ...], tuple[str, ...]]:
    normalization_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            normalization_ids.update(
                id(parameter) for parameter in module.parameters(recurse=False)
            )
    decay: list[Tensor] = []
    zero_decay: list[Tensor] = []
    decay_names: list[str] = []
    zero_decay_names: list[str] = []
    named = tuple(sorted(model.named_parameters(), key=lambda item: item[0]))
    for name, parameter in named:
        if not parameter.requires_grad or parameter.dtype != torch.float32:
            raise TypeError("every optimizer parameter must be trainable float32")
        excluded = name.endswith("bias") or id(parameter) in normalization_ids
        if excluded:
            zero_decay.append(parameter)
            zero_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if not decay or not zero_decay or len(decay) + len(zero_decay) != len(named):
        raise RuntimeError("AdamW parameter partition is empty or incomplete")
    if {id(value) for value in decay} & {id(value) for value in zero_decay}:
        raise RuntimeError("AdamW parameter groups overlap")
    return decay, zero_decay, tuple(decay_names), tuple(zero_decay_names)


def prepare_training_batch(
    projection: PilotTrainingProjection,
    *,
    fit_identity_sha256: str,
    optimizer_step: int,
    recipe: PilotTrainingRecipe,
    root_seed: int = TRAINING_ROOT_SEED,
) -> PreparedTrainingBatch:
    """Materialize the deterministic weighted draw, level, corruption, and context mask."""

    if type(projection) is not PilotTrainingProjection:
        raise TypeError("projection must be a PilotTrainingProjection")
    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    if type(optimizer_step) is not int or not 1 <= optimizer_step <= recipe.max_steps:
        raise ValueError("optimizer_step must be an exact integer in 1..max_steps")
    draw_start = (optimizer_step - 1) * recipe.batch_sequences
    _assert_training_call_surface()
    rows = _WEIGHTED_MINIBATCH_ROWS_FUNCTION(
        projection,
        fit_identity_sha256=fit_identity_sha256,
        global_draw_start=draw_start,
        draw_count=recipe.batch_sequences,
        root_seed=root_seed,
    )
    clean, attention = _ENCODE_ROWS_FUNCTION(rows)
    levels = _TIMESTEP_LEVELS_FUNCTION(
        fit_identity_sha256=fit_identity_sha256,
        global_draw_start=draw_start,
        draw_count=recipe.batch_sequences,
        root_seed=root_seed,
    )
    ordinals = tuple(range(draw_start, draw_start + recipe.batch_sequences))
    corruption = _CORRUPT_TRAINING_BATCH_FUNCTION(
        clean,
        attention,
        levels,
        global_draw_ordinals=ordinals,
        sequence_ids=tuple(row.sequence_id for row in rows),
        fit_identity_sha256=fit_identity_sha256,
        root_seed=root_seed,
        context_dropout_probability=recipe.context_dropout,
    )
    return PreparedTrainingBatch(
        rows=rows,
        global_draw_ordinals=ordinals,
        clean_tokens=clean,
        attention_mask=attention,
        levels=levels,
        corruption=corruption,
    )


def count_prior_logits(
    log_relative_position_probability: Tensor,
    attention_mask: Tensor,
    lengths: Tensor,
) -> Tensor:
    """Gather float32 C0 logits in [length-bin, position-bin, residue] order."""

    _tensor(
        log_relative_position_probability,
        label="log_relative_position_probability",
        dtype=torch.float32,
        rank=3,
    )
    _tensor(attention_mask, label="attention_mask", dtype=torch.bool, rank=2)
    _tensor(lengths, label="lengths", dtype=torch.int64, rank=1)
    if log_relative_position_probability.shape != (5, 10, 20):
        raise ValueError("log_relative_position_probability must have shape [5,10,20]")
    batch, width = attention_mask.shape
    if batch <= 0 or width <= 0 or lengths.shape != (batch,):
        raise ValueError("attention_mask and lengths are not aligned")
    if any(
        value.device != log_relative_position_probability.device
        for value in (attention_mask, lengths)
    ):
        raise ValueError("C0 gather tensors must share one device")
    if not bool(torch.isfinite(log_relative_position_probability).all().item()):
        raise ValueError("C0 log probabilities must be finite")
    positions = torch.arange(width, device=lengths.device, dtype=torch.int64).unsqueeze(0)
    if not torch.equal(attention_mask, positions < lengths.unsqueeze(1)):
        raise ValueError("attention_mask must be a prefix matching lengths")
    edges = torch.tensor(_LENGTH_EDGES, device=lengths.device, dtype=torch.int64)
    length_bins = torch.searchsorted(edges, lengths, right=True) - 1
    if bool(torch.any((length_bins < 0) | (length_bins >= 5)).item()):
        raise ValueError("length has no frozen C0 bin")
    position_bins = torch.div(
        10 * positions.expand(batch, width),
        lengths.unsqueeze(1),
        rounding_mode="floor",
    ).clamp(max=9)
    result = log_relative_position_probability[length_bins.unsqueeze(1), position_bins]
    result = result.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
    if result.dtype != torch.float32 or result.shape != (batch, width, 20):
        raise RuntimeError("C0 gather returned an invalid tensor")
    return result


def masked_sequence_objective(
    residual_logits: Tensor,
    count_logits: Tensor,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    scheduled_mask: Tensor,
    attention_mask: Tensor,
    *,
    label_smoothing: float = 0.05,
) -> PilotObjective:
    """Add C0 in float32 and reduce smoothed CE by sequence, then by batch."""

    _tensor(residual_logits, label="residual_logits", dtype=torch.float32, rank=3)
    _tensor(count_logits, label="count_logits", dtype=torch.float32, rank=3)
    for label, value, dtype in (
        ("clean_tokens", clean_tokens, torch.int64),
        ("corrupted_tokens", corrupted_tokens, torch.int64),
        ("scheduled_mask", scheduled_mask, torch.bool),
        ("attention_mask", attention_mask, torch.bool),
    ):
        _tensor(value, label=label, dtype=dtype, rank=2)
    batch, width = clean_tokens.shape
    if residual_logits.shape != (batch, width, 20) or count_logits.shape != (
        batch,
        width,
        20,
    ):
        raise ValueError("both logit tensors must have shape [batch,width,20]")
    if not (
        corrupted_tokens.shape == scheduled_mask.shape == attention_mask.shape == clean_tokens.shape
    ):
        raise ValueError("token and mask tensors must have identical two-dimensional shapes")
    tensors = (
        count_logits,
        clean_tokens,
        corrupted_tokens,
        scheduled_mask,
        attention_mask,
    )
    if any(value.device != residual_logits.device for value in tensors):
        raise ValueError("objective tensors must share one device")
    if not bool(torch.isfinite(residual_logits).all().item()) or not bool(
        torch.isfinite(count_logits).all().item()
    ):
        raise ValueError("objective logits must be finite")
    if type(label_smoothing) is not float or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be an exact float in [0,1)")
    lengths = attention_mask.sum(dim=1, dtype=torch.int64)
    positions = torch.arange(width, device=attention_mask.device).unsqueeze(0)
    if bool(torch.any(lengths < 1).item()) or not torch.equal(
        attention_mask, positions < lengths.unsqueeze(1)
    ):
        raise ValueError("attention_mask must be a non-empty contiguous prefix")
    if bool(torch.any(attention_mask & ((clean_tokens < 0) | (clean_tokens >= 20))).item()):
        raise ValueError("valid clean tokens must be residues")
    if bool(torch.any(~attention_mask & clean_tokens.ne(PAD_TOKEN_INDEX)).item()):
        raise ValueError("clean padding positions must contain PAD")
    if bool(torch.any(scheduled_mask & ~attention_mask).item()):
        raise ValueError("scheduled_mask cannot select padding")
    selected_counts = scheduled_mask.sum(dim=1, dtype=torch.int64)
    if bool(torch.any(selected_counts < 1).item()):
        raise ValueError("scheduled_mask must select at least one token per row")
    if bool(torch.any(scheduled_mask & corrupted_tokens.ne(MASK_TOKEN_INDEX)).item()):
        raise ValueError("every scheduled position must contain MASK")
    if bool(torch.any(~attention_mask & corrupted_tokens.ne(PAD_TOKEN_INDEX)).item()):
        raise ValueError("corrupted padding positions must contain PAD")
    visible = attention_mask & ~scheduled_mask
    if bool(
        torch.any(
            visible & corrupted_tokens.ne(clean_tokens) & corrupted_tokens.ne(MASK_TOKEN_INDEX)
        ).item()
    ):
        raise ValueError("visible positions may differ from clean only by context MASK")
    combined = count_logits + residual_logits
    safe_targets = torch.where(scheduled_mask, clean_tokens, torch.zeros_like(clean_tokens))
    token_losses = _FUNCTIONAL_CROSS_ENTROPY(
        combined.reshape(batch * width, RESIDUE_VOCABULARY_SIZE),
        safe_targets.reshape(batch * width),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(batch, width)
    selected_float = scheduled_mask.to(dtype=torch.float32)
    row_losses = (token_losses * selected_float).sum(dim=1) / selected_counts.to(
        dtype=torch.float32
    )
    correct = combined.argmax(dim=-1).eq(clean_tokens) & scheduled_mask
    row_accuracies = correct.sum(dim=1).to(torch.float32) / selected_counts.to(torch.float32)
    loss = row_losses.mean()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (combined, row_losses, row_accuracies, loss)
    ):
        raise FloatingPointError("masked sequence objective produced a non-finite result")
    return PilotObjective(
        loss=loss,
        row_losses=row_losses,
        row_accuracies=row_accuracies,
        selected_counts=selected_counts,
        combined_logits=combined,
    )


def initialize_fit_session(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    projection: PilotTrainingProjection,
    trainer_root: str | os.PathLike[str],
    *,
    device: str | torch.device = "cuda:0",
) -> PilotFitSession:
    """Initialize one production fit in the only contract-authorized order."""

    _assert_training_call_surface()
    recipe = _TRAINING_RECIPE_FROM_CONTRACT_FUNCTION(contract)
    return _initialize_bound_fit_session(
        contract,
        outer_fold,
        projection,
        count_prior=None,
        trainer_root=trainer_root,
        recipe=recipe,
        device=device,
        production_bound=True,
    )


def initialize_test_fit_session(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    projection: PilotTrainingProjection,
    count_prior: AuthenticatedCountPrior,
    *,
    recipe: PilotTrainingRecipe,
    device: str | torch.device = "cpu",
) -> PilotFitSession:
    """Initialize an explicitly non-publishable small session for unit tests."""

    _assert_training_call_surface()
    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    return _initialize_bound_fit_session(
        contract,
        outer_fold,
        projection,
        count_prior=count_prior,
        trainer_root=None,
        recipe=recipe,
        device=device,
        production_bound=False,
    )


def _initialize_bound_fit_session(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    projection: PilotTrainingProjection,
    *,
    count_prior: AuthenticatedCountPrior | None,
    trainer_root: str | os.PathLike[str] | None,
    recipe: PilotTrainingRecipe,
    device: str | torch.device,
    production_bound: bool,
) -> PilotFitSession:
    _assert_training_call_surface()
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(outer_fold) is not int:
        raise TypeError("outer_fold must be an exact integer")
    if type(projection) is not PilotTrainingProjection:
        raise TypeError("projection must be a PilotTrainingProjection")
    if type(recipe) is not PilotTrainingRecipe:
        raise TypeError("recipe must be a PilotTrainingRecipe")
    if type(production_bound) is not bool:
        raise TypeError("production_bound must be an exact bool")
    bound_device = torch.device(device)
    if production_bound and torch.cuda.is_initialized():
        raise RuntimeError("production fit must attest runtime settings before CUDA initialization")

    # CUDA-sensitive environment variables and numerical controls are pinned
    # before projection revalidation or any C0 NumPy construction.
    establish_r128_deterministic_runtime(contract)
    if production_bound:
        if bound_device != torch.device("cuda:0"):
            raise RuntimeError("production fit sessions require exact device cuda:0")
        if count_prior is not None:
            raise ValueError(
                "production count prior must be built and sealed inside initialization"
            )
        if trainer_root is None:
            raise ValueError("production fit sessions require an exact trainer root")
        production_environment = verify_r128_production_environment(
            contract,
            bound_device,
        )
    else:
        if bound_device != torch.device("cpu"):
            raise RuntimeError("test-only fit sessions require exact CPU device")
        if type(count_prior) is not AuthenticatedCountPrior:
            raise TypeError("test count_prior must be an AuthenticatedCountPrior")
        if trainer_root is not None:
            raise ValueError("test-only fit sessions do not accept a trainer root")
        production_environment = None
    fold = contract.fold(outer_fold)
    fit_identity = contract.fit_identity_sha256(outer_fold)
    if fit_identity != fold.fit_identity_sha256:
        raise ValueError("derived fit identity differs from the authenticated fold identity")
    if production_bound:
        if projection.sha256 != fold.train_sha256 or len(projection.rows) != fold.train_rows:
            raise ValueError("training projection differs from the authenticated fold pins")
        if recipe != _TRAINING_RECIPE_FROM_CONTRACT_FUNCTION(contract):
            raise ValueError("production fit recipe differs from the authenticated contract")
    projection.revalidate()
    sealed_count_prior: SealedCountPrior | None
    if production_bound:
        sealed_count_prior = _SEAL_COUNT_PRIOR_NPZ_FUNCTION(projection, trainer_root)
        count_prior = sealed_count_prior.revalidate()
    else:
        sealed_count_prior = None
        if count_prior is None:  # pragma: no cover - exact-type branch above
            raise RuntimeError("test count prior unexpectedly disappeared")
        expected_count_prior = _COUNT_PRIOR_FROM_PROJECTION_FUNCTION(
            AuthenticatedCountPrior,
            projection,
        )
        count_prior.revalidate()
        if (
            count_prior.sha256 != expected_count_prior.sha256
            or count_prior.payload != expected_count_prior.payload
        ):
            raise ValueError("count prior was not derived from the bound training projection")

    count_log_probability = _COUNT_PRIOR_TRAINING_BRIDGE_FUNCTION(
        count_prior,
        device=bound_device,
    )
    expected_count_tensor_sha256 = _count_tensor_sha256_from_prior(count_prior)
    if (
        _CANONICAL_COUNT_TENSOR_SHA256_FUNCTION(count_log_probability)
        != expected_count_tensor_sha256
    ):
        raise RuntimeError("C0 training bridge differs from the authenticated float32 conversion")
    model, initialization_seed = _BUILD_R128_MODEL_FROM_CONTRACT_FUNCTION(
        contract,
        outer_fold,
        device=bound_device,
    )
    if (
        _CANONICAL_R128_MODEL_SHA256_FUNCTION(model.state_dict())
        != _R128_INITIAL_MODEL_SHA256[outer_fold]
    ):
        raise RuntimeError("R128 model factory differs from the frozen initial state")
    optimizer, decay_names, zero_decay_names = _BUILD_PILOT_ADAMW_FUNCTION(model, recipe)
    session = PilotFitSession(
        contract=contract,
        outer_fold=outer_fold,
        fold=fold,
        fit_identity_sha256=fit_identity,
        projection=projection,
        count_prior=count_prior,
        sealed_count_prior=sealed_count_prior,
        count_log_probability=count_log_probability,
        model=model,
        optimizer=optimizer,
        recipe=recipe,
        decay_names=decay_names,
        zero_decay_names=zero_decay_names,
        initialization_seed=initialization_seed,
        production_bound=production_bound,
        production_environment=production_environment,
        _construction_token=_SESSION_CONSTRUCTION_TOKEN,
    )
    validate_fit_session(session, require_completed_checkpoint=True)
    return session


def initialize_fit_components(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    projection: PilotTrainingProjection,
    trainer_root: str | os.PathLike[str],
    *,
    device: str | torch.device = "cuda:0",
) -> PilotFitSession:
    """Compatibility spelling for :func:`initialize_fit_session`."""

    return initialize_fit_session(
        contract,
        outer_fold,
        projection,
        trainer_root,
        device=device,
    )


def run_training_steps(session: PilotFitSession) -> tuple[TrainingStepResult, ...]:
    """Advance contiguously to exactly the next scheduled checkpoint."""

    return advance_fit_to_next_checkpoint(session)


def advance_fit_to_next_checkpoint(
    session: PilotFitSession,
) -> tuple[TrainingStepResult, ...]:
    """Run only the next sealed interval; callers cannot choose or skip steps."""

    _assert_training_call_surface()
    validate_fit_session(session, require_completed_checkpoint=True)
    if session.last_sealed_checkpoint_step != session.completed_step:
        raise RuntimeError(
            "the completed checkpoint must be physically sealed before training continues"
        )
    checkpoint_step = session.next_checkpoint_step
    if checkpoint_step is None:
        raise ValueError("fit session has already completed its final checkpoint")
    if session._advancing:  # pragma: no cover - guarded by validation above
        raise RuntimeError("fit session is already advancing")
    session._advancing = True
    device = next(session.model.parameters()).device
    session.model.train()
    results: list[TrainingStepResult] = []
    try:
        for step in range(session.completed_step + 1, checkpoint_step + 1):
            _assert_training_call_surface()
            assert_r128_deterministic_runtime(session.contract)
            assert_r128_model_execution_surface(session.model)
            _validate_optimizer(
                session,
                expected_completed_step=session.completed_step,
                verify_digest=False,
            )
            batch = _PREPARE_TRAINING_BATCH_FUNCTION(
                session.projection,
                fit_identity_sha256=session.fit_identity_sha256,
                optimizer_step=step,
                recipe=session.recipe,
                root_seed=session.contract.seed,
            )
            clean = torch.from_numpy(batch.clean_tokens).to(device=device)
            corrupted = torch.from_numpy(batch.corruption.tokens).to(device=device)
            attention = torch.from_numpy(batch.attention_mask).to(device=device)
            selected = torch.from_numpy(batch.corruption.scheduled_mask).to(device=device)
            levels = torch.from_numpy(batch.levels).to(device=device)
            lengths = attention.sum(dim=1, dtype=torch.int64)
            prior_logits = _COUNT_PRIOR_LOGITS_FUNCTION(
                session.count_log_probability,
                attention,
                lengths,
            )
            learning_rate = _LEARNING_RATE_FOR_STEP_FUNCTION(session.recipe, step)
            for group in session.optimizer.param_groups:
                group["lr"] = learning_rate
            session.optimizer.zero_grad(set_to_none=True)
            dropout_seed = _RESEED_MODEL_DROPOUT_FUNCTION(
                session.fit_identity_sha256,
                step,
                root_seed=session.contract.seed,
            )
            residual = session.model(corrupted, attention, levels, lengths)
            objective = _MASKED_SEQUENCE_OBJECTIVE_FUNCTION(
                residual,
                prior_logits,
                clean,
                corrupted,
                selected,
                attention,
                label_smoothing=session.recipe.label_smoothing,
            )
            assert_r128_model_execution_surface(session.model)
            _assert_optimizer_execution_surface(session.optimizer)
            objective.loss.backward()
            assert_r128_model_execution_surface(session.model)
            _assert_optimizer_execution_surface(session.optimizer)
            gradient_norm_tensor = _CLIP_GRAD_NORM_FUNCTION(
                session.model.parameters(),
                session.recipe.gradient_clip_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            session.optimizer.step()
            assert_r128_model_execution_surface(session.model)
            _validate_optimizer(
                session,
                expected_completed_step=step,
                verify_digest=False,
            )
            if type(step) is not int or step != session._completed_step + 1:
                raise RuntimeError("fit-session completion may advance by exactly one step")
            session._completed_step = step
            loss = float(objective.loss.detach().cpu().item())
            accuracy = float(objective.row_accuracies.detach().mean().cpu().item())
            gradient_norm = float(gradient_norm_tensor.detach().cpu().item())
            selected_tokens = int(objective.selected_counts.detach().sum().cpu().item())
            results.append(
                TrainingStepResult(
                    step=step,
                    learning_rate=learning_rate,
                    loss=loss,
                    mean_row_accuracy=accuracy,
                    gradient_norm_before_clipping=gradient_norm,
                    selected_tokens=selected_tokens,
                    model_dropout_seed=dropout_seed,
                    batch_sha256=batch.sha256,
                )
            )
    except BaseException:
        session._failed = True
        raise
    finally:
        session._advancing = False
    _assert_training_call_surface()
    session._model_state_sha256 = _CANONICAL_R128_MODEL_SHA256_FUNCTION(session.model.state_dict())
    session._optimizer_state_sha256 = _OPTIMIZER_STATE_SHA256_FUNCTION(
        session.optimizer,
        session.decay_parameter_names + session.zero_decay_parameter_names,
    )
    if session.completed_step != checkpoint_step:
        session._failed = True
        raise RuntimeError("fit session did not stop at its next checkpoint")
    validate_fit_session(session, require_completed_checkpoint=True)
    return tuple(results)


def validate_fit_session(
    session: PilotFitSession,
    *,
    require_completed_checkpoint: bool,
    require_production: bool = False,
) -> PilotFitSession:
    """Revalidate all fit identities and live numerical continuity."""

    _assert_training_call_surface()
    if type(session) is not PilotFitSession:
        raise TypeError("session must be an exact PilotFitSession")
    if type(require_completed_checkpoint) is not bool:
        raise TypeError("require_completed_checkpoint must be an exact bool")
    if type(require_production) is not bool:
        raise TypeError("require_production must be an exact bool")
    if session._advancing:
        raise RuntimeError("fit session is already advancing")
    if session._failed:
        raise RuntimeError("fit session is unusable after an interrupted advancement")
    identity_bindings = (
        (session._contract_object_id, session.contract, "contract"),
        (session._projection_object_id, session.projection, "projection"),
        (session._count_prior_object_id, session.count_prior, "count prior"),
        (
            session._count_tensor_object_id,
            session.count_log_probability,
            "count tensor",
        ),
        (session._model_object_id, session.model, "model"),
        (session._optimizer_object_id, session.optimizer, "optimizer"),
    )
    if any(id(value) != expected for expected, value, _ in identity_bindings):
        changed = next(
            label for expected, value, label in identity_bindings if id(value) != expected
        )
        raise ValueError(f"fit-session bound {changed} object was replaced")
    if type(session.contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("fit-session contract has an invalid type")
    session.contract.revalidate()
    if type(session.outer_fold) is not int:
        raise TypeError("fit-session outer fold must be an exact integer")
    fold = session.contract.fold(session.outer_fold)
    if type(session.fold) is not PilotFoldContract or session.fold is not fold:
        raise ValueError("fit-session fold is not the contract's bound fold")
    fit_identity = session.contract.fit_identity_sha256(session.outer_fold)
    if (
        session.fit_identity_sha256 != fit_identity
        or session.fit_identity_sha256 != fold.fit_identity_sha256
    ):
        raise ValueError("fit-session fit identity differs from its contract and fold")
    if type(session.projection) is not PilotTrainingProjection:
        raise TypeError("fit-session projection has an invalid type")
    if type(session.count_prior) is not AuthenticatedCountPrior:
        raise TypeError("fit-session count prior has an invalid type")
    if type(session.recipe) is not PilotTrainingRecipe:
        raise TypeError("fit-session recipe has an invalid type")
    if type(session._production_bound) is not bool:
        raise TypeError("fit-session production binding must be an exact bool")
    if require_production and not session.production_bound:
        raise ValueError("test-only fit sessions cannot produce checkpoint metadata")
    assert_r128_deterministic_runtime(session.contract)
    if session.production_bound:
        if type(
            session.sealed_count_prior
        ) is not SealedCountPrior or session._sealed_count_prior_object_id != id(
            session.sealed_count_prior
        ):
            raise ValueError("production fit-session sealed C0 receipt was replaced")
        sealed_artifact = session.sealed_count_prior.revalidate()
        if sealed_artifact is not session.count_prior:
            raise ValueError("production fit-session C0 is not its physical sealed artifact")
        observed_environment = verify_r128_production_environment(
            session.contract,
            "cuda:0",
        )
        if (
            type(session._production_environment) is not dict
            or session._production_environment != observed_environment
        ):
            raise ValueError("fit-session production environment binding changed")
        if (
            session.projection.sha256 != fold.train_sha256
            or len(session.projection.rows) != fold.train_rows
        ):
            raise ValueError("fit-session projection differs from its fold pins")
        if session.recipe != _TRAINING_RECIPE_FROM_CONTRACT_FUNCTION(session.contract):
            raise ValueError("fit-session recipe differs from the production contract")
    elif (
        session._production_environment is not None
        or session.sealed_count_prior is not None
        or session._sealed_count_prior_object_id is not None
    ):
        raise ValueError("test-only fit session retained a production provenance claim")
    session.projection.revalidate()
    expected_prior = _COUNT_PRIOR_FROM_PROJECTION_FUNCTION(
        AuthenticatedCountPrior,
        session.projection,
    )
    session.count_prior.revalidate()
    if (
        session.count_prior.sha256 != expected_prior.sha256
        or session.count_prior.payload != expected_prior.payload
    ):
        raise ValueError("fit-session C0 does not derive from its bound projection")
    if (
        type(session._completed_step) is not int
        or not 0 <= session.completed_step <= session.recipe.max_steps
    ):
        raise ValueError("fit-session completed step is invalid")
    if require_completed_checkpoint and session.completed_step not in (
        0,
        *session.recipe.checkpoint_steps,
    ):
        raise ValueError("fit-session is not at an actually completed checkpoint")
    if type(session._sealed_checkpoints) is not tuple:
        raise TypeError("fit-session checkpoint receipt history must be a tuple")
    receipt_steps: list[int] = []
    metadata_paths = session.contract.table("checkpoints")["metadata_relative_paths"]
    if not isinstance(metadata_paths, tuple):
        raise ValueError("fit-session checkpoint metadata paths are invalid")
    for receipt_index, receipt in enumerate(session._sealed_checkpoints):
        if type(receipt) is not _SealedCheckpointReceipt:
            raise ValueError("fit-session checkpoint receipt has an invalid type")
        _SealedCheckpointReceipt(
            step=receipt.step,
            checkpoint_file_sha256=receipt.checkpoint_file_sha256,
            checkpoint_logical_state_sha256=receipt.checkpoint_logical_state_sha256,
            checkpoint_metadata_sha256=receipt.checkpoint_metadata_sha256,
            checkpoint_bytes=receipt.checkpoint_bytes,
            metadata_bytes=receipt.metadata_bytes,
            checkpoint_path=receipt.checkpoint_path,
            metadata_path=receipt.metadata_path,
            _capability=receipt._capability,
        )
        _reauthenticate_registered_checkpoint(session, receipt)
        if receipt.step > session.completed_step:
            raise ValueError("fit-session checkpoint receipt has an invalid step")
        if session.production_bound:
            sealed_count_prior = session.sealed_count_prior
            if sealed_count_prior is None:  # pragma: no cover - checked above
                raise RuntimeError("production fit-session lost its sealed C0 receipt")
            expected_checkpoint = (
                sealed_count_prior.trainer_root / session.contract.checkpoint_paths[receipt_index]
            )
            expected_metadata = sealed_count_prior.trainer_root / metadata_paths[receipt_index]
            if (
                Path(receipt.checkpoint_path) != expected_checkpoint
                or Path(receipt.metadata_path) != expected_metadata
            ):
                raise ValueError("fit-session checkpoint receipt path was relabeled")
        receipt_steps.append(receipt.step)
    expected_receipt_prefix = session.recipe.checkpoint_steps[: len(receipt_steps)]
    if tuple(receipt_steps) != expected_receipt_prefix:
        raise ValueError("fit-session checkpoint receipts are skipped or out of order")
    if (
        session._sealed_checkpoints
        and session.last_sealed_checkpoint_step == session.completed_step
        and session._sealed_checkpoints[-1].checkpoint_logical_state_sha256
        != session._model_state_sha256
    ):
        raise ValueError("current sealed receipt no longer matches the live model binding")
    expected_initialization_seed = _INITIALIZATION_SEED_FUNCTION(
        session.fit_identity_sha256,
        root_seed=session.contract.seed,
    )
    if (
        type(session._initialization_seed) is not int
        or session.initialization_seed != expected_initialization_seed
    ):
        raise ValueError("fit-session initialization seed is not contract-derived")

    if type(session.model) is not R128Denoiser:
        raise TypeError("fit-session model has an invalid type")
    assert_r128_model_execution_surface(session.model)
    model_parameters = tuple(session.model.parameters())
    if not model_parameters:
        raise RuntimeError("fit-session model has no parameters")
    model_device = model_parameters[0].device
    required_device = torch.device("cuda:0") if session.production_bound else torch.device("cpu")
    if model_device != required_device:
        raise ValueError("fit-session model moved off its bound production/test device")
    if any(
        parameter.device != model_device
        or parameter.dtype != torch.float32
        or not parameter.requires_grad
        for parameter in model_parameters
    ):
        raise ValueError("fit-session model parameters changed device, dtype, or trainability")
    count_tensor = session.count_log_probability
    if not isinstance(count_tensor, Tensor):
        raise TypeError("fit-session count tensor is not a torch.Tensor")
    if (
        count_tensor.device != model_device
        or count_tensor.dtype != torch.float32
        or count_tensor.shape != (5, 10, 20)
        or not count_tensor.is_contiguous()
        or count_tensor.requires_grad
        or count_tensor.grad_fn is not None
        or not bool(torch.isfinite(count_tensor).all().item())
    ):
        raise ValueError("fit-session count tensor changed from its frozen bridge value")
    if (
        type(session._count_tensor_sha256) is not str
        or _SHA256_RE.fullmatch(session._count_tensor_sha256) is None
        or _CANONICAL_COUNT_TENSOR_SHA256_FUNCTION(count_tensor) != session._count_tensor_sha256
        or session._count_tensor_sha256 != _count_tensor_sha256_from_prior(session.count_prior)
    ):
        raise ValueError("fit-session count tensor digest continuity was broken")
    current_model_sha256 = _CANONICAL_R128_MODEL_SHA256_FUNCTION(session.model.state_dict())
    if current_model_sha256 != session._model_state_sha256:
        raise ValueError("fit-session model state continuity was broken")
    _validate_optimizer(
        session,
        expected_completed_step=session.completed_step,
        verify_digest=True,
    )
    return session


def _validate_optimizer(
    session: PilotFitSession,
    *,
    expected_completed_step: int,
    verify_digest: bool,
) -> None:
    optimizer = session.optimizer
    recipe = session.recipe
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("fit-session optimizer must be exact torch.optim.AdamW")
    _assert_optimizer_execution_surface(optimizer)
    if type(expected_completed_step) is not int or expected_completed_step < 0:
        raise ValueError("expected optimizer step must be a non-negative exact integer")
    if type(verify_digest) is not bool:
        raise TypeError("verify_digest must be an exact bool")
    decay, zero_decay, decay_names, zero_decay_names = _pilot_parameter_partition(session.model)
    if session._decay_names != decay_names or session._zero_decay_names != zero_decay_names:
        raise ValueError("fit-session optimizer parameter names changed")
    if type(optimizer.param_groups) is not list or len(optimizer.param_groups) != 2:
        raise ValueError("AdamW must retain exactly two parameter groups")
    expected_groups = (decay, zero_decay)
    expected_weight_decay = (recipe.weight_decay, 0.0)
    expected_lr = (
        recipe.learning_rate
        if expected_completed_step == 0
        else _LEARNING_RATE_FOR_STEP_FUNCTION(recipe, expected_completed_step)
    )
    for index, (group, expected_parameters) in enumerate(
        zip(optimizer.param_groups, expected_groups, strict=True)
    ):
        parameters = group.get("params")
        if type(parameters) is not list or len(parameters) != len(expected_parameters):
            raise ValueError("AdamW parameter group length or container changed")
        if any(
            observed is not expected
            for observed, expected in zip(parameters, expected_parameters, strict=True)
        ):
            raise ValueError("AdamW parameter identities or order changed")
        exact_values: tuple[tuple[str, object], ...] = (
            ("lr", expected_lr),
            ("betas", recipe.betas),
            ("eps", recipe.epsilon),
            ("weight_decay", expected_weight_decay[index]),
            ("amsgrad", False),
            ("foreach", False),
            ("maximize", False),
            ("capturable", False),
            ("differentiable", False),
            ("fused", False),
            ("decoupled_weight_decay", True),
        )
        for name, expected in exact_values:
            observed = group.get(name)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"AdamW group {index} field {name} changed")
    default_values: tuple[tuple[str, object], ...] = (
        ("lr", recipe.learning_rate),
        ("betas", recipe.betas),
        ("eps", recipe.epsilon),
        ("weight_decay", 0.0),
        ("amsgrad", False),
        ("foreach", False),
        ("maximize", False),
        ("capturable", False),
        ("differentiable", False),
        ("fused", False),
        ("decoupled_weight_decay", True),
    )
    for name, expected in default_values:
        observed = optimizer.defaults.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"AdamW default field {name} changed")

    ordered_parameters = tuple(decay + zero_decay)
    state_parameters = tuple(optimizer.state)
    if expected_completed_step == 0:
        if state_parameters:
            raise ValueError("AdamW state exists before the first completed step")
    else:
        if len(state_parameters) != len(ordered_parameters) or {
            id(parameter) for parameter in state_parameters
        } != {id(parameter) for parameter in ordered_parameters}:
            raise ValueError("AdamW state parameter identities are incomplete or changed")
        for parameter in ordered_parameters:
            state = optimizer.state[parameter]
            if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("AdamW per-parameter state schema changed")
            step_tensor = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if (
                not isinstance(step_tensor, Tensor)
                or step_tensor.dtype != torch.float32
                or step_tensor.device.type != "cpu"
                or step_tensor.numel() != 1
                or float(step_tensor.item()) != float(expected_completed_step)
            ):
                raise ValueError("AdamW state step is not continuous with the fit session")
            for name, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
                if (
                    not isinstance(value, Tensor)
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                    or value.shape != parameter.shape
                    or value.layout != torch.strided
                    or not value.is_contiguous()
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError(f"AdamW state member {name} changed")
            if bool(torch.any(exp_avg_sq < 0).item()):
                raise ValueError("AdamW second-moment state cannot be negative")
    if verify_digest:
        observed_digest = _OPTIMIZER_STATE_SHA256_FUNCTION(
            optimizer,
            decay_names + zero_decay_names,
        )
        if observed_digest != session._optimizer_state_sha256:
            raise ValueError("fit-session optimizer state continuity was broken")


def _assert_optimizer_execution_surface(optimizer: torch.optim.AdamW) -> None:
    if (
        torch_adam_module.adam is not _ADAM_FUNCTIONAL
        or getattr(torch_adam_module.adam, "__code__", None) is not _ADAM_FUNCTIONAL_CODE
        or not _CLOSURE_MATCHES_FUNCTION(
            torch_adam_module.adam,
            _ADAM_FUNCTIONAL_CLOSURE,
            _ADAM_FUNCTIONAL_CLOSURE_CODES,
        )
        or _ADAMW_CORE_STEP_METHOD is None
        or getattr(_ADAMW_RAW_STEP_METHOD, "__code__", None) is not _ADAMW_RAW_STEP_CODE
        or getattr(_ADAMW_RAW_STEP_METHOD, "__wrapped__", None) is not _ADAMW_CORE_STEP_METHOD
        or getattr(_ADAMW_CORE_STEP_METHOD, "__code__", None) is not _ADAMW_CORE_STEP_CODE
        or getattr(_ADAMW_CORE_STEP_METHOD, "__defaults__", None) != _ADAMW_CORE_STEP_DEFAULTS
        or (
            None
            if getattr(_ADAMW_CORE_STEP_METHOD, "__kwdefaults__", None) is None
            else tuple(_ADAMW_CORE_STEP_METHOD.__kwdefaults__.items())
        )
        != _ADAMW_CORE_STEP_KWDEFAULTS
        or not _CLOSURE_MATCHES_FUNCTION(
            _ADAMW_RAW_STEP_METHOD,
            _ADAMW_RAW_STEP_CLOSURE,
            _ADAMW_RAW_STEP_CLOSURE_CODES,
        )
        or _ADAMW_CORE_STEP_METHOD.__globals__.get("adam") is not _ADAM_FUNCTIONAL
    ):
        raise RuntimeError("Adam functional update was overridden")
    if (
        torch_adam_module._single_tensor_adam is not _SINGLE_TENSOR_ADAM_FUNCTION
        or getattr(torch_adam_module._single_tensor_adam, "__code__", None)
        is not _SINGLE_TENSOR_ADAM_CODE
        or _ADAM_FUNCTIONAL.__globals__.get("_single_tensor_adam")
        is not _SINGLE_TENSOR_ADAM_FUNCTION
    ):
        raise RuntimeError("Adam single-tensor update was overridden")
    if torch.optim.Optimizer.profile_hook_step is not _OPTIMIZER_PROFILE_HOOK_STEP_METHOD:
        raise RuntimeError("optimizer profile step wrapper factory was overridden")
    _validate_adamw_step_wrapper(torch.optim.AdamW.step)
    if _ADAMW_STEP_METHOD is None or torch.optim.AdamW.step is not _ADAMW_STEP_METHOD:
        raise RuntimeError("AdamW step method was overridden")
    if torch.optim.AdamW.zero_grad is not _ADAMW_ZERO_GRAD_METHOD:
        raise RuntimeError("AdamW zero_grad method was overridden")
    if torch.optim.AdamW.state_dict is not _ADAMW_STATE_DICT_METHOD:
        raise RuntimeError("AdamW state_dict method was overridden")
    if torch.optim.AdamW.load_state_dict is not _ADAMW_LOAD_STATE_DICT_METHOD:
        raise RuntimeError("AdamW load_state_dict method was overridden")
    if torch.optim.Optimizer.zero_grad is not _OPTIMIZER_ZERO_GRAD_METHOD:
        raise RuntimeError("optimizer zero_grad method was overridden")
    if torch.optim.Optimizer.state_dict is not _OPTIMIZER_STATE_DICT_METHOD:
        raise RuntimeError("optimizer state_dict method was overridden")
    if torch.optim.Optimizer.load_state_dict is not _OPTIMIZER_LOAD_STATE_DICT_METHOD:
        raise RuntimeError("optimizer load_state_dict method was overridden")
    if {"step", "zero_grad", "state_dict", "load_state_dict"} & set(optimizer.__dict__):
        raise RuntimeError("optimizer method was overridden on the instance")
    hook_fields = (
        "_optimizer_step_pre_hooks",
        "_optimizer_step_post_hooks",
        "_optimizer_state_dict_pre_hooks",
        "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks",
        "_optimizer_load_state_dict_post_hooks",
    )
    for field in hook_fields:
        if getattr(optimizer, field, None):
            raise RuntimeError(f"optimizer hook registry {field} must be empty")
    for field in ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks"):
        if getattr(torch_optimizer_module, field, None):
            raise RuntimeError(f"global optimizer hook registry {field} must be empty")


def _closure_matches(
    value: object,
    expected: tuple[object, ...],
    expected_codes: tuple[object, ...],
) -> bool:
    closure = getattr(value, "__closure__", None)
    if closure is None:
        return not expected and not expected_codes
    return (
        type(closure) is tuple
        and len(closure) == len(expected)
        and len(closure) == len(expected_codes)
        and all(
            cell.cell_contents is expected_value
            and getattr(cell.cell_contents, "__code__", None) is expected_code
            for cell, expected_value, expected_code in zip(
                closure,
                expected,
                expected_codes,
                strict=True,
            )
        )
    )


def _bind_adamw_step_surface() -> None:
    """Capture PyTorch's one-time exact profiling wrapper after optimizer init."""

    global _ADAMW_STEP_METHOD
    observed = torch.optim.AdamW.step
    _validate_adamw_step_wrapper(observed)
    if _ADAMW_STEP_METHOD is None:
        _ADAMW_STEP_METHOD = observed
    elif observed is not _ADAMW_STEP_METHOD:
        raise RuntimeError("AdamW step wrapper identity changed")


def _validate_adamw_step_wrapper(value: object) -> None:
    if (
        not callable(value)
        or getattr(value, "hooked", None) is not True
        or getattr(value, "__wrapped__", None) is not _ADAMW_RAW_STEP_METHOD
        or getattr(value, "__code__", None) is not _ADAMW_PROFILE_STEP_CODE
    ):
        raise RuntimeError("AdamW step is not the exact PyTorch profiling wrapper")
    closure = getattr(value, "__closure__", None)
    if (
        type(closure) is not tuple
        or len(closure) != 1
        or closure[0].cell_contents is not _ADAMW_RAW_STEP_METHOD
    ):
        raise RuntimeError("AdamW step wrapper closure changed")


def _optimizer_state_sha256(
    optimizer: torch.optim.AdamW,
    parameter_names: tuple[str, ...],
) -> str:
    parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if len(parameters) != len(parameter_names):
        raise ValueError("optimizer state digest parameter names are not aligned")
    digest = hashlib.sha256()
    digest.update(_OPTIMIZER_STATE_DOMAIN)
    for name, parameter in zip(parameter_names, parameters, strict=True):
        _digest_frame(digest, name.encode("utf-8"))
        state = optimizer.state.get(parameter)
        if state is None:
            digest.update(b"\x00")
            continue
        digest.update(b"\x01")
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            value = state[state_name]
            _digest_frame(digest, state_name.encode("ascii"))
            _digest_frame(digest, str(value.dtype).encode("ascii"))
            digest.update(struct.pack("<Q", value.ndim))
            for dimension in value.shape:
                digest.update(struct.pack("<Q", dimension))
            raw = value.detach().cpu().contiguous().numpy().tobytes(order="C")
            _digest_frame(digest, raw)
    return digest.hexdigest()


def _canonical_count_tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("count tensor digest input must be a torch.Tensor")
    if (
        value.dtype != torch.float32
        or value.shape != (5, 10, 20)
        or value.layout != torch.strided
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError("count tensor digest input must be finite contiguous float32 [5,10,20]")
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(_COUNT_TENSOR_DOMAIN)
    _digest_frame(digest, b"float32")
    digest.update(struct.pack("<QQQ", 5, 10, 20))
    _digest_frame(digest, raw)
    return digest.hexdigest()


def _count_tensor_sha256_from_prior(prior: AuthenticatedCountPrior) -> str:
    """Independently derive the exact float32 bridge digest from authenticated C0."""

    if type(prior) is not AuthenticatedCountPrior:
        raise TypeError("count tensor derivation requires an AuthenticatedCountPrior")
    decoded = prior.revalidate()
    source = decoded.log_relative_position_probability
    if (
        source.dtype != np.dtype("<f8")
        or source.shape != (5, 10, 20)
        or not source.flags.c_contiguous
        or not bool(np.isfinite(source).all())
    ):
        raise ValueError("authenticated C0 source is not finite contiguous float64 [5,10,20]")
    raw = np.asarray(source, dtype="<f4", order="C").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(_COUNT_TENSOR_DOMAIN)
    _digest_frame(digest, b"float32")
    digest.update(struct.pack("<QQQ", 5, 10, 20))
    _digest_frame(digest, raw)
    return digest.hexdigest()


def _digest_frame(digest: object, value: bytes) -> None:
    update = digest.update
    update(struct.pack("<Q", len(value)))
    update(value)


def _encode_rows(
    rows: Sequence[PilotTrainingRow],
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    source = tuple(rows)
    if not source or any(type(row) is not PilotTrainingRow for row in source):
        raise TypeError("rows must contain PilotTrainingRow values")
    index = {residue: position for position, residue in enumerate(ALPHABET)}
    clean = np.full((len(source), 50), PAD_TOKEN_INDEX, dtype="<i8")
    attention = np.zeros((len(source), 50), dtype="|b1")
    for row_number, row in enumerate(source):
        length = len(row.sequence)
        clean[row_number, :length] = [index[residue] for residue in row.sequence]
        attention[row_number, :length] = True
    return clean, attention


def _tensor(value: object, *, label: str, dtype: torch.dtype, rank: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    if value.ndim != rank:
        raise ValueError(f"{label} must have rank {rank}")
    return value


def _exact(value: object, expected_type: type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"training.{label} has a non-exact type")
    return value


_TRAINING_RECIPE_FROM_CONTRACT_FUNCTION = training_recipe_from_contract
_TRAINING_RECIPE_FROM_CONTRACT_CODE = training_recipe_from_contract.__code__
_BUILD_PILOT_ADAMW_FUNCTION = build_pilot_adamw
_BUILD_PILOT_ADAMW_CODE = build_pilot_adamw.__code__
_PREPARE_TRAINING_BATCH_FUNCTION = prepare_training_batch
_PREPARE_TRAINING_BATCH_CODE = prepare_training_batch.__code__
_COUNT_PRIOR_LOGITS_FUNCTION = count_prior_logits
_COUNT_PRIOR_LOGITS_CODE = count_prior_logits.__code__
_LEARNING_RATE_FOR_STEP_FUNCTION = learning_rate_for_step
_LEARNING_RATE_FOR_STEP_CODE = learning_rate_for_step.__code__
_MASKED_SEQUENCE_OBJECTIVE_FUNCTION = masked_sequence_objective
_MASKED_SEQUENCE_OBJECTIVE_CODE = masked_sequence_objective.__code__
_CANONICAL_R128_MODEL_SHA256_FUNCTION = canonical_r128_model_sha256
_CANONICAL_R128_MODEL_SHA256_CODE = canonical_r128_model_sha256.__code__
_OPTIMIZER_STATE_SHA256_FUNCTION = _optimizer_state_sha256
_OPTIMIZER_STATE_SHA256_CODE = _optimizer_state_sha256.__code__
_CANONICAL_COUNT_TENSOR_SHA256_FUNCTION = _canonical_count_tensor_sha256
_CANONICAL_COUNT_TENSOR_SHA256_CODE = _canonical_count_tensor_sha256.__code__
_ENCODE_ROWS_FUNCTION = _encode_rows
_ENCODE_ROWS_CODE = _encode_rows.__code__
_CLOSURE_MATCHES_FUNCTION = _closure_matches
_CLOSURE_MATCHES_CODE = _closure_matches.__code__
_ADVANCE_FIT_FUNCTION = advance_fit_to_next_checkpoint
_ADVANCE_FIT_CODE = advance_fit_to_next_checkpoint.__code__


def _assert_training_call_surface() -> None:
    """Reject replacement or in-place code edits of frozen training callees."""

    callables = (
        (
            "count-prior bridge",
            count_prior_training_bridge,
            _COUNT_PRIOR_TRAINING_BRIDGE_FUNCTION,
            _COUNT_PRIOR_TRAINING_BRIDGE_CODE,
        ),
        (
            "R128 model factory",
            build_r128_model_from_contract,
            _BUILD_R128_MODEL_FROM_CONTRACT_FUNCTION,
            _BUILD_R128_MODEL_FROM_CONTRACT_CODE,
        ),
        (
            "count-prior seal",
            seal_count_prior_npz,
            _SEAL_COUNT_PRIOR_NPZ_FUNCTION,
            _SEAL_COUNT_PRIOR_NPZ_CODE,
        ),
        (
            "count-prior derivation",
            AuthenticatedCountPrior.from_projection.__func__,
            _COUNT_PRIOR_FROM_PROJECTION_FUNCTION,
            _COUNT_PRIOR_FROM_PROJECTION_CODE,
        ),
        (
            "weighted minibatch RNG",
            weighted_minibatch_rows,
            _WEIGHTED_MINIBATCH_ROWS_FUNCTION,
            _WEIGHTED_MINIBATCH_ROWS_CODE,
        ),
        (
            "timestep RNG",
            timestep_levels,
            _TIMESTEP_LEVELS_FUNCTION,
            _TIMESTEP_LEVELS_CODE,
        ),
        (
            "corruption RNG",
            corrupt_training_batch,
            _CORRUPT_TRAINING_BATCH_FUNCTION,
            _CORRUPT_TRAINING_BATCH_CODE,
        ),
        (
            "dropout RNG",
            reseed_torch_for_model_dropout,
            _RESEED_MODEL_DROPOUT_FUNCTION,
            _RESEED_MODEL_DROPOUT_CODE,
        ),
        (
            "initialization seed",
            initialization_seed,
            _INITIALIZATION_SEED_FUNCTION,
            _INITIALIZATION_SEED_CODE,
        ),
        (
            "cross entropy",
            F.cross_entropy,
            _FUNCTIONAL_CROSS_ENTROPY,
            _FUNCTIONAL_CROSS_ENTROPY_CODE,
        ),
        (
            "gradient clipping",
            torch.nn.utils.clip_grad_norm_,
            _CLIP_GRAD_NORM_FUNCTION,
            _CLIP_GRAD_NORM_CODE,
        ),
        (
            "training recipe",
            training_recipe_from_contract,
            _TRAINING_RECIPE_FROM_CONTRACT_FUNCTION,
            _TRAINING_RECIPE_FROM_CONTRACT_CODE,
        ),
        (
            "optimizer factory",
            build_pilot_adamw,
            _BUILD_PILOT_ADAMW_FUNCTION,
            _BUILD_PILOT_ADAMW_CODE,
        ),
        (
            "training batch",
            prepare_training_batch,
            _PREPARE_TRAINING_BATCH_FUNCTION,
            _PREPARE_TRAINING_BATCH_CODE,
        ),
        (
            "count-prior gather",
            count_prior_logits,
            _COUNT_PRIOR_LOGITS_FUNCTION,
            _COUNT_PRIOR_LOGITS_CODE,
        ),
        (
            "learning-rate schedule",
            learning_rate_for_step,
            _LEARNING_RATE_FOR_STEP_FUNCTION,
            _LEARNING_RATE_FOR_STEP_CODE,
        ),
        (
            "masked objective",
            masked_sequence_objective,
            _MASKED_SEQUENCE_OBJECTIVE_FUNCTION,
            _MASKED_SEQUENCE_OBJECTIVE_CODE,
        ),
        (
            "model-state digest",
            canonical_r128_model_sha256,
            _CANONICAL_R128_MODEL_SHA256_FUNCTION,
            _CANONICAL_R128_MODEL_SHA256_CODE,
        ),
        (
            "safetensors loader",
            load_safetensors,
            _LOAD_SAFETENSORS_FUNCTION,
            _LOAD_SAFETENSORS_CODE,
        ),
        (
            "checkpoint tensor schema",
            checkpoint_tensor_records,
            _CHECKPOINT_TENSOR_RECORDS_FUNCTION,
            _CHECKPOINT_TENSOR_RECORDS_CODE,
        ),
        (
            "model config document",
            r128_model_config_document,
            _R128_MODEL_CONFIG_DOCUMENT_FUNCTION,
            _R128_MODEL_CONFIG_DOCUMENT_CODE,
        ),
        (
            "R128 state validator",
            validate_r128_state,
            _VALIDATE_R128_STATE_FUNCTION,
            _VALIDATE_R128_STATE_CODE,
        ),
        (
            "optimizer-state digest",
            _optimizer_state_sha256,
            _OPTIMIZER_STATE_SHA256_FUNCTION,
            _OPTIMIZER_STATE_SHA256_CODE,
        ),
        (
            "count-tensor digest",
            _canonical_count_tensor_sha256,
            _CANONICAL_COUNT_TENSOR_SHA256_FUNCTION,
            _CANONICAL_COUNT_TENSOR_SHA256_CODE,
        ),
        ("row encoding", _encode_rows, _ENCODE_ROWS_FUNCTION, _ENCODE_ROWS_CODE),
        (
            "optimizer closure validator",
            _closure_matches,
            _CLOSURE_MATCHES_FUNCTION,
            _CLOSURE_MATCHES_CODE,
        ),
        (
            "RNG execution surface",
            assert_pilot_rng_execution_surface,
            _ASSERT_PILOT_RNG_SURFACE_FUNCTION,
            _ASSERT_PILOT_RNG_SURFACE_CODE,
        ),
        (
            "authenticated fit advancement",
            advance_fit_to_next_checkpoint,
            _ADVANCE_FIT_FUNCTION,
            _ADVANCE_FIT_CODE,
        ),
    )
    for label, observed, expected, expected_code in callables:
        if observed is not expected or getattr(observed, "__code__", None) is not expected_code:
            raise RuntimeError(f"frozen training callable was overridden: {label}")
    if tuple(R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD) != _R128_INITIAL_MODEL_SHA256:
        raise RuntimeError("R128 initial-state golden hashes were relabeled")
    _ASSERT_PILOT_RNG_SURFACE_FUNCTION()

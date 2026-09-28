"""Byte-exact R128 score-ledger inference for the sealed v1 pilot.

Checkpoint authentication is deliberately outside this numerical module.  A
producer supplies an already-authenticated model (or a loader which returns
one), while this module owns the frozen tensor construction, batching,
selected-token ordering, and exact float32 byte boundary.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128Denoiser,
    assert_r128_deterministic_runtime,
    assert_r128_model_execution_surface,
    canonical_r128_model_sha256,
    validate_r128_state,
    verify_r128_production_environment,
)
from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    CHECKPOINT_STEPS,
    MAX_LENGTH,
    ScoreCorruptionLedger,
)

EVALUATION_BATCH_SEQUENCES = 256
RESIDUE_CLASSES = 20
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

Float32Array = NDArray[np.float32]
LoadedCheckpoint: TypeAlias = R128Denoiser | tuple[int, R128Denoiser]
AuthenticatedCheckpointLoader: TypeAlias = Callable[[int], LoadedCheckpoint]
AuthenticatedCheckpointSequence: TypeAlias = Sequence[LoadedCheckpoint]
AuthenticatedLedgerArrays: TypeAlias = Mapping[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class ReinferenceComparison:
    """Exact receipt values for one archived/re-inferred checkpoint slice."""

    archived_residual_logit_slice_sha256: str
    reinferred_residual_logit_slice_sha256: str
    byte_equal: bool

    def __post_init__(self) -> None:
        for label, value in (
            (
                "archived_residual_logit_slice_sha256",
                self.archived_residual_logit_slice_sha256,
            ),
            (
                "reinferred_residual_logit_slice_sha256",
                self.reinferred_residual_logit_slice_sha256,
            ),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if self.byte_equal is not True:
            raise ValueError("a reinference comparison may only represent exact byte equality")
        if self.archived_residual_logit_slice_sha256 != self.reinferred_residual_logit_slice_sha256:
            raise ValueError("equal reinference slices must have identical SHA-256 digests")

    def canonical_record(self) -> dict[str, object]:
        """Return a fresh record matching the child contract's receipt schema."""

        return {
            "archived_residual_logit_slice_sha256": (self.archived_residual_logit_slice_sha256),
            "reinferred_residual_logit_slice_sha256": (self.reinferred_residual_logit_slice_sha256),
            "byte_equal": True,
        }


def infer_checkpoint_slice(
    model: R128Denoiser,
    ledger: ScoreCorruptionLedger,
    *,
    contract: NativeDiffusionV1PilotContract,
    _test_only_allow_cpu: bool = False,
) -> Float32Array:
    """Infer one checkpoint in frozen case/position order as exact ``<f4``.

    Production has no device override: the authenticated model must already be
    on the sole visible ``cuda:0`` A100.  The underscored CPU escape hatch is
    intentionally explicit and exists only for deterministic unit fixtures.
    """

    _validate_contract(contract)
    if type(_test_only_allow_cpu) is not bool:
        raise TypeError("_test_only_allow_cpu must be an exact bool")
    if type(model) is not R128Denoiser:
        raise TypeError("model must be an exact R128Denoiser")
    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    arrays = _authenticated_ledger_array_snapshot(ledger, contract=contract)
    try:
        return _infer_checkpoint_slice_from_authenticated_arrays(
            model,
            arrays,
            contract=contract,
            _test_only_allow_cpu=_test_only_allow_cpu,
        )
    finally:
        # The snapshot isolates inference from source-array mutation, while this
        # exit authentication ensures that a persistent mid-flight change can
        # never be followed by a successful result.
        ledger.revalidate()


def _infer_checkpoint_slice_from_authenticated_arrays(
    model: R128Denoiser,
    arrays: AuthenticatedLedgerArrays,
    *,
    contract: NativeDiffusionV1PilotContract,
    _test_only_allow_cpu: bool,
) -> Float32Array:
    """Infer one checkpoint from one private, already-authenticated snapshot."""

    _validate_contract(contract)
    if type(_test_only_allow_cpu) is not bool:
        raise TypeError("_test_only_allow_cpu must be an exact bool")
    if type(model) is not R128Denoiser:
        raise TypeError("model must be an exact R128Denoiser")

    assert_r128_model_execution_surface(model)
    model_device = _exact_model_device(model)
    if _test_only_allow_cpu:
        if model_device != torch.device("cpu"):
            raise RuntimeError("the test-only inference path requires an exact CPU model")
        assert_r128_deterministic_runtime(contract)
    else:
        if model_device != torch.device("cuda:0"):
            raise RuntimeError("production inference requires the model on exact cuda:0")
        verify_r128_production_environment(contract, device="cuda:0")

    before_state = canonical_r128_model_sha256(validate_r128_state(model.state_dict()))
    model.eval()
    assert_r128_model_execution_surface(model)
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("R128 inference requires every model module in eval mode")

    case_count = len(arrays["case_id"])
    selected_count = int(np.sum(arrays["mask_count"], dtype=np.uint64))
    result = np.empty((selected_count, RESIDUE_CLASSES), dtype="<f4", order="C")
    cursor = 0
    for first in range(0, case_count, EVALUATION_BATCH_SEQUENCES):
        last = min(first + EVALUATION_BATCH_SEQUENCES, case_count)
        row_index = arrays["row_index"][first:last].astype(np.int64, copy=True)
        tokens = arrays["corrupted_tokens"][first:last].astype(np.int64, copy=True)
        attention = arrays["attention_mask"][row_index].astype(np.bool_, copy=True)
        levels = arrays["level"][first:last].astype(np.int64, copy=True)
        lengths = arrays["length"][row_index].astype(np.int64, copy=True)
        selected = arrays["selected_mask"][first:last]

        token_tensor = _to_exact_tensor(tokens, dtype=torch.int64, device=model_device)
        attention_tensor = _to_exact_tensor(
            attention,
            dtype=torch.bool,
            device=model_device,
        )
        level_tensor = _to_exact_tensor(levels, dtype=torch.int64, device=model_device)
        length_tensor = _to_exact_tensor(lengths, dtype=torch.int64, device=model_device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=model_device.type, enabled=False),
        ):
            residual = model(
                token_tensor,
                attention_tensor,
                level_tensor,
                length_tensor,
            )
        expected_shape = (last - first, MAX_LENGTH, RESIDUE_CLASSES)
        if (
            residual.dtype != torch.float32
            or residual.device != model_device
            or residual.shape != expected_shape
            or residual.requires_grad
        ):
            raise RuntimeError("R128 returned a noncanonical residual-logit tensor")
        if not bool(torch.isfinite(residual).all().item()):
            raise FloatingPointError("R128 returned non-finite residual logits")
        cpu_residual = (
            residual.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
                non_blocking=False,
                copy=True,
            )
            .contiguous()
            .numpy()
        )
        case_offset, position = np.nonzero(selected)
        gathered = np.ascontiguousarray(
            cpu_residual[case_offset, position],
            dtype="<f4",
        )
        next_cursor = cursor + len(gathered)
        result[cursor:next_cursor] = gathered
        cursor = next_cursor

    if cursor != selected_count:
        raise RuntimeError("inferred selected-token census differs from the sealed ledger")
    _validate_residual_logit_slice(result)
    assert_r128_model_execution_surface(model)
    after_state = canonical_r128_model_sha256(validate_r128_state(model.state_dict()))
    if after_state != before_state:
        raise RuntimeError("R128 checkpoint state changed during inference")
    result.flags.writeable = False
    return cast(Float32Array, result)


def infer_all_checkpoint_slices(
    checkpoints: AuthenticatedCheckpointLoader | AuthenticatedCheckpointSequence,
    ledger: ScoreCorruptionLedger,
    *,
    contract: NativeDiffusionV1PilotContract,
    _test_only_allow_cpu: bool = False,
) -> Float32Array:
    """Infer all five authenticated checkpoints in exact contract step order.

    A callback is invoked with each required checkpoint step.  A sequence may
    contain exact models in positional step order or ``(step, model)`` pairs;
    pairs add a runtime check that the sequence was not permuted.  One private
    authenticated ledger snapshot is shared by all five passes, after which
    the source ledger is reauthenticated before the result is exposed.
    """

    _validate_contract(contract)
    if type(_test_only_allow_cpu) is not bool:
        raise TypeError("_test_only_allow_cpu must be an exact bool")
    if callable(checkpoints):
        sequence: tuple[LoadedCheckpoint, ...] | None = None
        loader: AuthenticatedCheckpointLoader | None = checkpoints
    else:
        if not isinstance(checkpoints, Sequence) or isinstance(
            checkpoints, str | bytes | bytearray
        ):
            raise TypeError("checkpoints must be an authenticated loader or sequence")
        sequence = tuple(checkpoints)
        if len(sequence) != len(CHECKPOINT_STEPS):
            raise ValueError("checkpoint sequence must contain exactly five entries")
        loader = None

    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    arrays = _authenticated_ledger_array_snapshot(ledger, contract=contract)
    try:
        slices: list[Float32Array] = []
        model_identities: set[int] = set()
        for index, step in enumerate(CHECKPOINT_STEPS):
            loaded = loader(step) if loader is not None else sequence[index]  # type: ignore[index]
            model = _model_for_step(loaded, expected_step=step)
            # Reusing one mutable model through a loader is expected: the loader
            # authenticates and replaces its state for each requested step.  A
            # materialized sequence, however, must not alias one model five times.
            if sequence is not None:
                identity = id(model)
                if identity in model_identities:
                    raise ValueError("checkpoint model sequence contains an aliased model")
                model_identities.add(identity)
            inferred = _infer_checkpoint_slice_from_authenticated_arrays(
                model,
                arrays,
                contract=contract,
                _test_only_allow_cpu=_test_only_allow_cpu,
            )
            slices.append(_validate_residual_logit_slice(inferred))
        output = np.ascontiguousarray(np.stack(slices, axis=0), dtype="<f4")
        expected_shape = (len(CHECKPOINT_STEPS), slices[0].shape[0], RESIDUE_CLASSES)
        if output.shape != expected_shape or output.dtype.str != "<f4":
            raise RuntimeError("all-checkpoint residual-logit array is noncanonical")
        output.flags.writeable = False
    finally:
        # A loader may close over the source ledger.  Reauthenticate after the
        # complete five-checkpoint flight before exposing any stacked result.
        ledger.revalidate()
    return cast(Float32Array, output)


def _authenticated_ledger_array_snapshot(
    ledger: ScoreCorruptionLedger,
    *,
    contract: NativeDiffusionV1PilotContract,
) -> AuthenticatedLedgerArrays:
    """Authenticate and copy one ledger for an isolated inference flight."""

    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    # arrays() performs the full source-ledger authentication and returns one
    # owning copy of every member.  Freeze those private copies so the five
    # checkpoint passes can safely share them without touching source state.
    copied = ledger.arrays()
    if ledger.parent_contract_sha256 != contract.parent_config_sha256:
        raise ValueError("ledger parent contract differs from the authenticated contract")
    for raw in copied.values():
        raw.flags.writeable = False
    return MappingProxyType(copied)


def residual_logit_slice_sha256(values: Float32Array) -> str:
    """Hash the exact C-order little-endian bytes of one checkpoint slice."""

    _validate_residual_logit_slice(values)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def compare_archived_reinference(
    archived: Float32Array,
    reinferred: Float32Array,
) -> ReinferenceComparison:
    """Reject any shape, dtype, layout, or byte difference between two slices."""

    _validate_residual_logit_slice(archived)
    _validate_residual_logit_slice(reinferred)
    if archived.shape != reinferred.shape:
        raise ValueError("archived and re-inferred residual-logit shapes differ")
    archived_sha256 = residual_logit_slice_sha256(archived)
    reinferred_sha256 = residual_logit_slice_sha256(reinferred)
    # Digest equality is not used as a substitute for the contract's required
    # byte-for-byte comparison.
    byte_equal = (
        memoryview(archived).cast("B").tobytes() == memoryview(reinferred).cast("B").tobytes()
    )
    if not byte_equal:
        raise ValueError("archived and re-inferred residual-logit bytes differ")
    if archived_sha256 != reinferred_sha256:  # pragma: no cover - implied by byte equality
        raise RuntimeError("equal residual-logit bytes produced different SHA-256 digests")
    return ReinferenceComparison(
        archived_residual_logit_slice_sha256=archived_sha256,
        reinferred_residual_logit_slice_sha256=reinferred_sha256,
        byte_equal=True,
    )


def _validate_contract(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    evaluation = contract.table("evaluation")
    expected = {
        "batch_sequences": EVALUATION_BATCH_SEQUENCES,
        "levels": 64,
        "replicates_per_sequence_level": 1,
    }
    for name, value in expected.items():
        if type(evaluation.get(name)) is not int or evaluation[name] != value:
            raise ValueError(f"authenticated evaluation.{name} differs from inference")
    if contract.checkpoint_steps != CHECKPOINT_STEPS:
        raise ValueError("authenticated checkpoint order differs from inference")


def _exact_model_device(model: R128Denoiser) -> torch.device:
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("R128 model has no parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise RuntimeError("R128 parameters do not share one exact device")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise TypeError("R128 parameters must remain torch.float32")
    return next(iter(devices))


def _to_exact_tensor(
    values: NDArray[np.generic],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if type(values) is not np.ndarray or not values.flags.c_contiguous:
        raise TypeError("inference input must be an exact C-contiguous ndarray")
    return torch.from_numpy(values).to(
        device=device,
        dtype=dtype,
        non_blocking=False,
        copy=True,
    )


def _model_for_step(loaded: object, *, expected_step: int) -> R128Denoiser:
    if type(loaded) is R128Denoiser:
        return loaded
    if type(loaded) is tuple and len(loaded) == 2:
        step, model = loaded
        if type(step) is not int or step != expected_step:
            raise ValueError("authenticated checkpoint sequence is outside exact step order")
        if type(model) is not R128Denoiser:
            raise TypeError("authenticated checkpoint loader returned an invalid model")
        return model
    raise TypeError("authenticated checkpoint loader returned an invalid entry")


def _validate_residual_logit_slice(values: object) -> Float32Array:
    if type(values) is not np.ndarray:
        raise TypeError("residual-logit slice must be an exact ndarray")
    if values.dtype.str != "<f4":
        raise TypeError("residual-logit slice must have exact little-endian float32 dtype")
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != RESIDUE_CLASSES:
        raise ValueError("residual-logit slice must have shape [selected_tokens,20]")
    if not values.flags.c_contiguous:
        raise ValueError("residual-logit slice must be C-contiguous")
    if not np.isfinite(values).all():
        raise FloatingPointError("residual-logit slice contains non-finite values")
    return cast(Float32Array, values)

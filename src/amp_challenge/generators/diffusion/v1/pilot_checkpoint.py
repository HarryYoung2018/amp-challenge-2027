"""Deterministic R128 checkpoint identity, sealing, and loading helpers.

These helpers bind complete SafeTensor bytes to the authenticated child
contract, sealed C0 archive, fit identity, exact 31-tensor logical state, and a
caller-supplied trusted checkpoint identity.  Pair publication is durable and
no-overwrite; an interrupted partial pair is deliberately unusable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import AuthenticatedCountPrior
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128Denoiser,
    assert_r128_deterministic_runtime,
    assert_r128_model_execution_surface,
    canonical_r128_model_sha256,
    checkpoint_tensor_records,
    r128_model_config_document,
    validate_r128_state,
)
from amp_challenge.generators.diffusion.v1.pilot_training import (
    _CHECKPOINT_SEAL_CAPABILITY,
    PilotFitSession,
    validate_fit_session,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_CHECKPOINT_BYTES = 1 << 30
_MAX_METADATA_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Physical, logical, and metadata identities for one completed step."""

    checkpoint_file_sha256: str
    checkpoint_logical_state_sha256: str
    checkpoint_metadata_sha256: str
    metadata_bytes: bytes

    def __post_init__(self) -> None:
        for label, value in (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_logical_state_sha256", self.checkpoint_logical_state_sha256),
            ("checkpoint_metadata_sha256", self.checkpoint_metadata_sha256),
        ):
            _sha256(value, label=label)
        if type(self.metadata_bytes) is not bytes or not self.metadata_bytes:
            raise ValueError("metadata_bytes must be non-empty exact bytes")
        if hashlib.sha256(self.metadata_bytes).hexdigest() != self.checkpoint_metadata_sha256:
            raise ValueError("checkpoint_metadata_sha256 does not match metadata_bytes")


def r128_safetensors_bytes(model: R128Denoiser) -> bytes:
    """Serialize the exact sorted finite R128 state without pickle or metadata."""

    if type(model) is not R128Denoiser:
        raise TypeError("model must be an R128Denoiser")
    assert_r128_model_execution_surface(model)
    state = validate_r128_state(model.state_dict())
    try:
        from safetensors.torch import save
    except ImportError as error:  # pragma: no cover - optional-extra failure
        raise RuntimeError("safetensors is required for R128 checkpoints") from error
    payload = save(state, metadata=None)
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_CHECKPOINT_BYTES:
        raise ValueError("serialized R128 checkpoint has an invalid byte size")
    return payload


def authenticate_r128_checkpoint_bytes(
    payload: bytes,
    *,
    metadata_bytes: bytes,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    checkpoint_step: int,
    count_prior: AuthenticatedCountPrior,
    expected_identity: CheckpointIdentity,
) -> CheckpointIdentity:
    """Authenticate a checkpoint pair without loading it into a live model."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    contract.revalidate()
    assert_r128_deterministic_runtime(contract)
    if type(outer_fold) is not int:
        raise TypeError("outer_fold must be an exact integer")
    if type(checkpoint_step) is not int or checkpoint_step not in contract.checkpoint_steps:
        raise ValueError("checkpoint_step is not a contract checkpoint")
    if type(count_prior) is not AuthenticatedCountPrior:
        raise TypeError("count_prior must be an AuthenticatedCountPrior")
    count_prior.revalidate()
    if type(expected_identity) is not CheckpointIdentity:
        raise TypeError("expected_identity must be an exact CheckpointIdentity")
    CheckpointIdentity(
        checkpoint_file_sha256=expected_identity.checkpoint_file_sha256,
        checkpoint_logical_state_sha256=expected_identity.checkpoint_logical_state_sha256,
        checkpoint_metadata_sha256=expected_identity.checkpoint_metadata_sha256,
        metadata_bytes=expected_identity.metadata_bytes,
    )
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_CHECKPOINT_BYTES:
        raise ValueError("R128 checkpoint payload has an invalid byte size")
    if type(metadata_bytes) is not bytes or not 0 < len(metadata_bytes) <= _MAX_METADATA_BYTES:
        raise ValueError("R128 checkpoint metadata has an invalid byte size")
    physical = hashlib.sha256(payload).hexdigest()
    metadata_physical = hashlib.sha256(metadata_bytes).hexdigest()
    if (
        physical != expected_identity.checkpoint_file_sha256
        or metadata_physical != expected_identity.checkpoint_metadata_sha256
        or metadata_bytes != expected_identity.metadata_bytes
    ):
        raise ValueError("checkpoint pair differs from the trusted identity")
    state = _state_from_safetensors_bytes(payload)
    logical = canonical_r128_model_sha256(state)
    if logical != expected_identity.checkpoint_logical_state_sha256:
        raise ValueError("checkpoint logical state differs from the trusted identity")
    metadata = _load_canonical_metadata_bytes(metadata_bytes)
    expected_metadata = _checkpoint_metadata_document(
        contract,
        outer_fold=outer_fold,
        checkpoint_step=checkpoint_step,
        count_prior_sha256=count_prior.sha256,
        checkpoint_file_sha256=physical,
        checkpoint_logical_state_sha256=logical,
        tensor_records=checkpoint_tensor_records(state),
    )
    if metadata_bytes != canonical_json_bytes(expected_metadata) or metadata != expected_metadata:
        raise ValueError("checkpoint companion metadata bindings do not match")
    observed_identity = CheckpointIdentity(
        checkpoint_file_sha256=physical,
        checkpoint_logical_state_sha256=logical,
        checkpoint_metadata_sha256=metadata_physical,
        metadata_bytes=metadata_bytes,
    )
    if observed_identity != expected_identity:  # pragma: no cover - complete checks above
        raise ValueError("authenticated checkpoint differs from the trusted identity")
    return observed_identity


def load_r128_safetensors_bytes(
    model: R128Denoiser,
    payload: bytes,
    *,
    metadata_bytes: bytes,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    checkpoint_step: int,
    count_prior: AuthenticatedCountPrior,
    expected_identity: CheckpointIdentity,
) -> CheckpointIdentity:
    """Authenticate trusted metadata and state before and after exact loading."""

    if type(model) is not R128Denoiser:
        raise TypeError("model must be an R128Denoiser")
    assert_r128_model_execution_surface(model)
    observed_identity = authenticate_r128_checkpoint_bytes(
        payload,
        metadata_bytes=metadata_bytes,
        contract=contract,
        outer_fold=outer_fold,
        checkpoint_step=checkpoint_step,
        count_prior=count_prior,
        expected_identity=expected_identity,
    )
    state = _state_from_safetensors_bytes(payload)
    assert_r128_model_execution_surface(model)
    model.load_state_dict(state, strict=True)
    assert_r128_model_execution_surface(model)
    live_logical = canonical_r128_model_sha256(model.state_dict())
    if live_logical != observed_identity.checkpoint_logical_state_sha256:
        raise ValueError("live model state differs after authenticated checkpoint load")
    return observed_identity


def checkpoint_metadata_identity(
    session: PilotFitSession,
    *,
    checkpoint_file_bytes: bytes,
) -> CheckpointIdentity:
    """Create canonical metadata for one post-optimizer-step R128 checkpoint."""

    validate_fit_session(
        session,
        require_completed_checkpoint=True,
        require_production=True,
    )
    if session.completed_step == 0:
        raise ValueError("cannot create metadata before an optimizer checkpoint completes")
    contract = session.contract
    contract.revalidate()
    model = session.model
    outer_fold = session.outer_fold
    checkpoint_step = session.completed_step
    fold = session.fold
    count_prior_sha256 = session.count_prior.sha256
    session.count_prior.revalidate()
    if (
        type(checkpoint_file_bytes) is not bytes
        or not 0 < len(checkpoint_file_bytes) <= _MAX_CHECKPOINT_BYTES
    ):
        raise ValueError("checkpoint_file_bytes have an invalid byte size")
    step_index = contract.checkpoint_steps.index(checkpoint_step)
    checkpoint_file = contract.checkpoint_paths[step_index]
    physical_sha256 = hashlib.sha256(checkpoint_file_bytes).hexdigest()
    state = _state_from_safetensors_bytes(checkpoint_file_bytes)
    logical_sha256 = canonical_r128_model_sha256(state)
    live_state = validate_r128_state(model.state_dict())
    live_logical_sha256 = canonical_r128_model_sha256(live_state)
    if logical_sha256 != live_logical_sha256:
        raise ValueError("checkpoint bytes do not encode the supplied live model state")
    fit_identity = session.fit_identity_sha256
    if fit_identity != fold.fit_identity_sha256:
        raise ValueError("derived fit identity differs from the frozen fold identity")
    artifact = contract.document["artifact"]
    if type(artifact) is not str:
        raise RuntimeError("authenticated child artifact is not an exact string")
    pilot = contract.table("pilot")
    tensor_records = checkpoint_tensor_records(state)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "artifact": artifact,
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "fit_identity_sha256": fit_identity,
        "outer_fold": outer_fold,
        "checkpoint_step": checkpoint_step,
        "variant": pilot["variant"],
        "output_mode": pilot["output_mode"],
        "model_config": r128_model_config_document(),
        "count_prior_file_sha256": count_prior_sha256,
        "optimizer_step_completed": checkpoint_step,
        "checkpoint_file": checkpoint_file,
        "checkpoint_file_sha256": physical_sha256,
        "checkpoint_logical_state_sha256": logical_sha256,
        "tensors": tensor_records,
    }
    metadata_contract = contract.table("checkpoint_metadata")
    fields = metadata_contract["fields"]
    if not isinstance(fields, tuple) or tuple(metadata) != fields:
        raise ValueError("checkpoint metadata fields differ from the authenticated contract")
    if metadata_contract["tensor_count"] != len(tensor_records):
        raise ValueError("checkpoint tensor count differs from the authenticated contract")
    payload = canonical_json_bytes(metadata)
    identity = CheckpointIdentity(
        checkpoint_file_sha256=physical_sha256,
        checkpoint_logical_state_sha256=logical_sha256,
        checkpoint_metadata_sha256=hashlib.sha256(payload).hexdigest(),
        metadata_bytes=payload,
    )
    return identity


def seal_checkpoint(
    session: PilotFitSession,
    trainer_root: str | os.PathLike[str],
) -> CheckpointIdentity:
    """Durably publish and reauthenticate the current production checkpoint."""

    validate_fit_session(
        session,
        require_completed_checkpoint=True,
        require_production=True,
    )
    if session.completed_step == 0:
        raise ValueError("cannot seal a checkpoint before training")
    if session.last_sealed_checkpoint_step == session.completed_step:
        raise FileExistsError("current checkpoint is already sealed")
    sealed_count_prior = session.sealed_count_prior
    if sealed_count_prior is None:  # pragma: no cover - production validation above
        raise RuntimeError("production checkpoint lost its sealed count-prior receipt")
    expected_root = sealed_count_prior.trainer_root
    observed_root = Path(os.path.abspath(os.fspath(trainer_root)))
    if observed_root != expected_root:
        raise ValueError("checkpoint trainer root differs from the sealed count-prior root")
    index = session.contract.checkpoint_steps.index(session.completed_step)
    checkpoint_relative = session.contract.checkpoint_paths[index]
    metadata_paths = session.contract.table("checkpoints")["metadata_relative_paths"]
    if not isinstance(metadata_paths, tuple) or len(metadata_paths) != len(
        session.contract.checkpoint_steps
    ):
        raise ValueError("authenticated checkpoint metadata paths are invalid")
    metadata_relative = metadata_paths[index]
    if type(metadata_relative) is not str:
        raise ValueError("authenticated checkpoint metadata path is invalid")
    checkpoint_bytes = r128_safetensors_bytes(session.model)
    identity = checkpoint_metadata_identity(
        session,
        checkpoint_file_bytes=checkpoint_bytes,
    )
    checkpoint_path, metadata_path = _publish_checkpoint_pair(
        trainer_root,
        checkpoint_relative=checkpoint_relative,
        metadata_relative=metadata_relative,
        checkpoint_bytes=checkpoint_bytes,
        metadata_bytes=identity.metadata_bytes,
    )
    reopened_checkpoint = _read_sealed_bytes(
        checkpoint_path,
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
        label="sealed checkpoint",
    )
    reopened_metadata = _read_sealed_bytes(
        metadata_path,
        maximum_bytes=_MAX_METADATA_BYTES,
        label="sealed checkpoint metadata",
    )
    expected = checkpoint_metadata_identity(
        session,
        checkpoint_file_bytes=reopened_checkpoint,
    )
    if expected != identity or reopened_metadata != identity.metadata_bytes:
        raise ValueError("reopened checkpoint pair differs from its authenticated identity")
    session._register_sealed_checkpoint(
        step=session.completed_step,
        checkpoint_file_sha256=identity.checkpoint_file_sha256,
        checkpoint_logical_state_sha256=identity.checkpoint_logical_state_sha256,
        checkpoint_metadata_sha256=identity.checkpoint_metadata_sha256,
        checkpoint_bytes=reopened_checkpoint,
        metadata_bytes=reopened_metadata,
        checkpoint_path=os.fspath(checkpoint_path),
        metadata_path=os.fspath(metadata_path),
        _capability=_CHECKPOINT_SEAL_CAPABILITY,
    )
    return identity


def _seal_test_checkpoint(
    session: PilotFitSession,
    trainer_root: str | os.PathLike[str],
) -> tuple[CheckpointIdentity, Path, Path]:
    """Physically seal a non-production checkpoint without production metadata."""

    validate_fit_session(session, require_completed_checkpoint=True)
    if session.production_bound:
        raise ValueError("test checkpoint seal cannot accept a production session")
    if session.completed_step == 0:
        raise ValueError("cannot seal a test checkpoint before training")
    checkpoint_bytes = r128_safetensors_bytes(session.model)
    logical = canonical_r128_model_sha256(session.model.state_dict())
    metadata_bytes = canonical_json_bytes(
        {
            "artifact": "explicitly_test_only_r128_checkpoint",
            "checkpoint_file_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_logical_state_sha256": logical,
            "count_prior_file_sha256": session.count_prior.sha256,
            "fit_identity_sha256": session.fit_identity_sha256,
            "optimizer_step_completed": session.completed_step,
            "outer_fold": session.outer_fold,
        }
    )
    identity = CheckpointIdentity(
        checkpoint_file_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        checkpoint_logical_state_sha256=logical,
        checkpoint_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        metadata_bytes=metadata_bytes,
    )
    checkpoint_relative = f"test_checkpoints/step_{session.completed_step:06d}.safetensors"
    metadata_relative = f"test_checkpoints/step_{session.completed_step:06d}.metadata.json"
    checkpoint_path, metadata_path = _publish_checkpoint_pair(
        trainer_root,
        checkpoint_relative=checkpoint_relative,
        metadata_relative=metadata_relative,
        checkpoint_bytes=checkpoint_bytes,
        metadata_bytes=metadata_bytes,
    )
    reopened_checkpoint = _read_sealed_bytes(
        checkpoint_path,
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
        label="sealed test checkpoint",
    )
    reopened_metadata = _read_sealed_bytes(
        metadata_path,
        maximum_bytes=_MAX_METADATA_BYTES,
        label="sealed test checkpoint metadata",
    )
    state = _state_from_safetensors_bytes(reopened_checkpoint)
    if (
        canonical_r128_model_sha256(state) != logical
        or reopened_metadata != metadata_bytes
        or hashlib.sha256(reopened_checkpoint).hexdigest() != identity.checkpoint_file_sha256
    ):
        raise ValueError("reopened test checkpoint pair failed authentication")
    session._register_sealed_checkpoint(
        step=session.completed_step,
        checkpoint_file_sha256=identity.checkpoint_file_sha256,
        checkpoint_logical_state_sha256=identity.checkpoint_logical_state_sha256,
        checkpoint_metadata_sha256=identity.checkpoint_metadata_sha256,
        checkpoint_bytes=reopened_checkpoint,
        metadata_bytes=reopened_metadata,
        checkpoint_path=os.fspath(checkpoint_path),
        metadata_path=os.fspath(metadata_path),
        _capability=_CHECKPOINT_SEAL_CAPABILITY,
    )
    return identity, checkpoint_path, metadata_path


def canonical_json_bytes(value: object) -> bytes:
    """Serialize finite metadata as sorted compact UTF-8 with one terminal LF."""

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


def _checkpoint_metadata_document(
    contract: NativeDiffusionV1PilotContract,
    *,
    outer_fold: int,
    checkpoint_step: int,
    count_prior_sha256: str,
    checkpoint_file_sha256: str,
    checkpoint_logical_state_sha256: str,
    tensor_records: list[dict[str, object]],
) -> dict[str, object]:
    fold = contract.fold(outer_fold)
    if checkpoint_step not in contract.checkpoint_steps:
        raise ValueError("checkpoint step is not in the authenticated contract")
    index = contract.checkpoint_steps.index(checkpoint_step)
    artifact = contract.document["artifact"]
    if type(artifact) is not str:
        raise ValueError("authenticated child artifact is invalid")
    pilot = contract.table("pilot")
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": artifact,
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "fit_identity_sha256": fold.fit_identity_sha256,
        "outer_fold": outer_fold,
        "checkpoint_step": checkpoint_step,
        "variant": pilot["variant"],
        "output_mode": pilot["output_mode"],
        "model_config": r128_model_config_document(),
        "count_prior_file_sha256": _sha256(
            count_prior_sha256,
            label="count_prior_sha256",
        ),
        "optimizer_step_completed": checkpoint_step,
        "checkpoint_file": contract.checkpoint_paths[index],
        "checkpoint_file_sha256": _sha256(
            checkpoint_file_sha256,
            label="checkpoint_file_sha256",
        ),
        "checkpoint_logical_state_sha256": _sha256(
            checkpoint_logical_state_sha256,
            label="checkpoint_logical_state_sha256",
        ),
        "tensors": tensor_records,
    }
    metadata_contract = contract.table("checkpoint_metadata")
    fields = metadata_contract["fields"]
    if not isinstance(fields, tuple) or tuple(document) != fields:
        raise ValueError("checkpoint metadata fields differ from the authenticated contract")
    if metadata_contract["tensor_count"] != len(tensor_records):
        raise ValueError("checkpoint tensor count differs from the authenticated contract")
    return document


def _load_canonical_metadata_bytes(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_METADATA_BYTES:
        raise ValueError("checkpoint metadata has an invalid byte size")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("checkpoint metadata is not canonical single-LF JSON")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("checkpoint metadata is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("checkpoint metadata must be an exact object")
    if canonical_json_bytes(value) != payload:
        raise ValueError("checkpoint metadata is not canonical JSON")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint metadata key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite checkpoint metadata number: {value}")


def _publish_checkpoint_pair(
    trainer_root: str | os.PathLike[str],
    *,
    checkpoint_relative: str,
    metadata_relative: str,
    checkpoint_bytes: bytes,
    metadata_bytes: bytes,
) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(trainer_root)))
    _reject_symlink_chain(root)
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise ValueError("trainer root must already exist") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("trainer root must be a non-symlink directory")
    checkpoint_path = _derived_artifact_path(root, checkpoint_relative)
    metadata_path = _derived_artifact_path(root, metadata_relative)
    if checkpoint_path.parent != metadata_path.parent:
        raise ValueError("checkpoint and metadata must share one publication directory")
    directory = checkpoint_path.parent
    if not os.path.lexists(directory):
        if directory.parent != root:
            raise ValueError("checkpoint publication permits one derived subdirectory")
        try:
            os.mkdir(directory, 0o755)
        except OSError as error:
            raise ValueError("cannot create checkpoint publication directory") from error
        _fsync_directory(root)
    _reject_symlink_chain(directory)
    try:
        directory_stat = os.lstat(directory)
    except OSError as error:
        raise ValueError("cannot inspect checkpoint publication directory") from error
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("checkpoint publication path is not a directory")
    if os.path.lexists(checkpoint_path) or os.path.lexists(metadata_path):
        raise FileExistsError("checkpoint publication is strictly no-overwrite")

    checkpoint_temporary = _write_temporary_bytes(
        directory,
        checkpoint_bytes,
        label="checkpoint",
    )
    metadata_temporary: Path | None = None
    try:
        metadata_temporary = _write_temporary_bytes(
            directory,
            metadata_bytes,
            label="checkpoint metadata",
        )
        os.link(checkpoint_temporary, checkpoint_path, follow_symlinks=False)
        os.unlink(checkpoint_temporary)
        checkpoint_temporary = None
        os.chmod(checkpoint_path, 0o444, follow_symlinks=False)
        os.link(metadata_temporary, metadata_path, follow_symlinks=False)
        os.unlink(metadata_temporary)
        metadata_temporary = None
        os.chmod(metadata_path, 0o444, follow_symlinks=False)
        _fsync_directory(directory)
    except OSError as error:
        raise ValueError("checkpoint pair could not be durably published") from error
    finally:
        for temporary in (checkpoint_temporary, metadata_temporary):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temporary)
    return checkpoint_path, metadata_path


def _derived_artifact_path(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError("checkpoint relative path must be a non-empty exact string")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or any(
        part in ("", ".", "..") for part in candidate_relative.parts
    ):
        raise ValueError("checkpoint relative path is unsafe")
    candidate = root.joinpath(candidate_relative)
    if candidate.parent.parent != root:
        raise ValueError("checkpoint relative path has an unexpected depth")
    return candidate


def _write_temporary_bytes(directory: Path, payload: bytes, *, label: str) -> Path:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} payload must be non-empty exact bytes")
    descriptor, raw_path = tempfile.mkstemp(prefix=".pilot-seal-", dir=directory)
    path = Path(raw_path)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short checkpoint write")
            written += count
        os.fdatasync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(path)
        raise
    os.close(descriptor)
    return path


def _read_sealed_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(named_before.st_mode)
        or named_before.st_nlink != 1
        or stat.S_IMODE(named_before.st_mode) != 0o444
        or not 0 < named_before.st_size <= maximum_bytes
    ):
        raise ValueError(f"{label} must be bounded single-link regular mode 0444")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(65536, maximum_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte bound")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} changed while reopening") from error
    _reject_symlink_chain(source)
    payload = b"".join(chunks)
    fingerprints = (
        _stat_fingerprint(named_before),
        _stat_fingerprint(opened_before),
        _stat_fingerprint(opened_after),
        _stat_fingerprint(named_after),
    )
    if (
        len(set(fingerprints)) != 1
        or len(payload) != opened_before.st_size
        or not stat.S_ISREG(opened_before.st_mode)
        or opened_before.st_nlink != 1
        or stat.S_IMODE(opened_before.st_mode) != 0o444
    ):
        raise ValueError(f"{label} changed or lost its sealed file properties")
    return payload


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect checkpoint path {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"checkpoint path traverses a symlink: {candidate}")


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _fsync_directory(path: Path) -> None:
    _reject_symlink_chain(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _state_from_safetensors_bytes(payload: bytes) -> dict[str, Tensor]:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_CHECKPOINT_BYTES:
        raise ValueError("R128 checkpoint payload has an invalid byte size")
    try:
        from safetensors.torch import load

        loaded = load(payload)
    except Exception as error:
        raise ValueError("checkpoint is not valid safetensors data") from error
    return validate_r128_state(loaded)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value

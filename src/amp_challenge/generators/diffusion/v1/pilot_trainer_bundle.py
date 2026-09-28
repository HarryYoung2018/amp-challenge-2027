"""Neutral authentication of sealed native-diffusion v1 trainer bundles.

This reader is shared by the score-barrier coordinator and the independent
CPU verifier.  It derives every readiness value from immutable bundle bytes;
it never imports or trusts the trainer, a producer result, or an evaluator.
Checkpoint verification deserializes and hashes the exact SafeTensor state but
does not construct a model or perform a neural forward pass.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    BundleSnapshot,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_sha256_sidecar,
    parse_sha256sums,
    sha256sums_bytes,
    verify_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_checkpoint import (
    CheckpointIdentity,
    authenticate_r128_checkpoint_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    ARTIFACT,
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_control import (
    CheckpointDigest,
    TrainerReadinessObservation,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    AuthenticatedCountPrior,
    load_count_prior_npz_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD,
    establish_r128_deterministic_runtime,
    r128_model_config_document,
)
from amp_challenge.generators.diffusion.v1.pilot_rng import (
    initialization_seed,
    model_dropout_seed,
)
from amp_challenge.generators.diffusion.v1.pilot_training import (
    learning_rate_for_step,
    learning_rate_schedule_sha256,
    training_recipe_from_contract,
)

_AUTHENTICATED_BUNDLE_CAPABILITY = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_TRAINER_MANIFEST_FIELDS = (
    "schema_version",
    "artifact",
    "child_contract_sha256",
    "parent_contract_sha256",
    "git_commit",
    "outer_fold",
    "fit_identity",
    "projection",
    "model",
    "rng",
    "training",
    "checkpoints",
    "artifacts",
)
_CHECKPOINT_DIGEST_FIELDS = (
    "checkpoint_file_sha256",
    "checkpoint_logical_state_sha256",
    "checkpoint_metadata_sha256",
)
_TRAINING_TRACE_FIELDS = (
    "step",
    "learning_rate",
    "loss",
    "mean_row_accuracy",
    "gradient_norm_before_clipping",
    "selected_tokens",
    "model_dropout_seed",
    "batch_sha256",
)
_TRAIN_METRIC_FIELDS = (
    "schema_version",
    "artifact",
    "outer_fold",
    "fit_identity_sha256",
    "completed_steps",
    "batch_sequences",
    "total_sequence_draws",
    "total_selected_tokens",
    "mean_loss",
    "mean_row_accuracy",
    "mean_gradient_norm_before_clipping",
    "learning_rate_schedule_sha256",
    "training_trace_sha256",
    "checkpoint_steps",
)
_RNG_FIELDS = (
    "schema_version",
    "artifact",
    "derivation",
    "training_root_seed",
    "fit_identity_sha256",
    "initialization_seed",
    "training_namespaces",
    "global_draw_ordinal",
    "checkpoint_restores_mutable_rng_state",
)
_ENVIRONMENT_FIELDS = ("schema_version", "artifact", "runtime", "determinism")
_RUNTIME_FIELDS = (
    "python",
    "numpy",
    "torch",
    "torch_cuda",
    "safetensors",
    "packaging",
    "triton",
    "nvidia_cudnn_cu13",
    "gpu_name",
    "compute_capability",
    "visible_cuda_devices",
    "allocator",
    "cudnn_runtime_version",
)
_DETERMINISM_FIELDS = (
    "cublas_workspace_config",
    "pytorch_allocator",
    "deterministic_algorithms",
    "math_sdpa_only",
    "tf32",
    "amp",
)
_MAX_CODE_MANIFEST_BYTES = 64 << 20
_MAX_CONTRACT_BYTES = 262_144
_MAX_COUNT_PRIOR_BYTES = 64 << 20
_MAX_JSON_BYTES = 8 << 20
_MAX_TRACE_BYTES = 128 << 20
_MAX_CHECKPOINT_BYTES = 1 << 30
_MAX_METADATA_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class AuthenticatedTrainerBundle:
    """Detached, path-free evidence derived from one sealed trainer tree."""

    bundle: BundleSnapshot
    count_prior: AuthenticatedCountPrior
    outer_fold: int
    fit_identity_sha256: str
    manifest_git_commit: str
    checkpoint_digest_by_step: Mapping[str, CheckpointDigest]
    checkpoint_metadata_bytes_by_step: Mapping[str, bytes]
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _AUTHENTICATED_BUNDLE_CAPABILITY:
            raise RuntimeError("trainer bundle requires the internal authentication capability")
        if type(self.bundle) is not BundleSnapshot:
            raise TypeError("bundle must be an exact BundleSnapshot")
        if type(self.count_prior) is not AuthenticatedCountPrior:
            raise TypeError("count_prior must be an exact AuthenticatedCountPrior")
        self.count_prior.revalidate()
        if type(self.outer_fold) is not int or self.outer_fold not in (0, 1, 2, 3):
            raise ValueError("outer_fold must be an exact integer in 0..3")
        _sha256(self.fit_identity_sha256, label="fit identity")
        _git_commit(self.manifest_git_commit)
        digests = _freeze_checkpoint_digests(self.checkpoint_digest_by_step)
        metadata = _freeze_metadata_bytes(self.checkpoint_metadata_bytes_by_step)
        for key in _CHECKPOINT_KEYS:
            if hashlib.sha256(metadata[key]).hexdigest() != (
                digests[key].checkpoint_metadata_sha256
            ):
                raise ValueError(f"checkpoint metadata bytes changed at step {key}")
        object.__setattr__(self, "checkpoint_digest_by_step", digests)
        object.__setattr__(self, "checkpoint_metadata_bytes_by_step", metadata)
        if self.bundle.file("count_prior.npz").sha256 != self.count_prior.sha256:
            raise ValueError("authenticated bundle and count-prior identities differ")

    def make_readiness_observation(
        self,
        *,
        trusted_git_commit: str,
        node_name: str,
        device_uuid: str,
    ) -> TrainerReadinessObservation:
        """Bind trusted scheduler identity to values reconstructed from files."""

        git_commit = _git_commit(trusted_git_commit)
        if git_commit != self.manifest_git_commit:
            raise ValueError("trusted Git commit differs from the trainer manifest")
        return TrainerReadinessObservation(
            git_commit=git_commit,
            outer_fold=self.outer_fold,
            fit_identity_sha256=self.fit_identity_sha256,
            trainer_bundle_sha256=self.bundle.tree_sha256,
            count_prior_file_sha256=self.count_prior.sha256,
            checkpoint_digest_by_step=self.checkpoint_digest_by_step,
            node_name=node_name,
            device_uuid=device_uuid,
        )


def authenticate_trainer_bundle(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    expected_outer_fold: int,
    expected_code_sha256sums: bytes,
    expected_tree_sha256: str | None = None,
) -> AuthenticatedTrainerBundle:
    """Authenticate one complete trainer bundle without producer assertions."""

    _validate_contract(contract)
    if type(expected_outer_fold) is not int or expected_outer_fold not in (0, 1, 2, 3):
        raise ValueError("expected_outer_fold must be an exact integer in 0..3")
    if expected_tree_sha256 is not None:
        _sha256(expected_tree_sha256, label="expected trainer tree")
    expected_code = _expected_code_manifest(expected_code_sha256sums)
    bundle = verify_bundle(
        contract,
        bundle_kind="trainer",
        root=root,
        expected_tree_sha256=expected_tree_sha256,
    )

    def payload(name: str, maximum: int) -> bytes:
        return bundle.read_bytes(name, maximum_bytes=maximum)

    manifest = _json_object(
        parse_canonical_json(
            payload("manifest.json", _MAX_JSON_BYTES),
            label="trainer manifest",
        ),
        _TRAINER_MANIFEST_FIELDS,
        label="trainer manifest",
    )
    _validate_manifest_header(
        manifest,
        contract=contract,
        expected_outer_fold=expected_outer_fold,
    )
    _validate_artifact_bindings(manifest, bundle=bundle, contract=contract)

    child_payload = payload("pilot_execution_v1.toml", _MAX_CONTRACT_BYTES)
    parent_payload = payload("unconditional_v1.toml", _MAX_CONTRACT_BYTES)
    if hashlib.sha256(child_payload).hexdigest() != contract.config_sha256:
        raise ValueError("bundled child contract bytes differ from the authenticated contract")
    if hashlib.sha256(parent_payload).hexdigest() != contract.parent_config_sha256:
        raise ValueError("bundled parent contract bytes differ from the authenticated contract")

    code_payload = payload("CODE_SHA256SUMS", _MAX_CODE_MANIFEST_BYTES)
    code = parse_sha256sums(code_payload, label="trainer CODE_SHA256SUMS")
    if code_payload != expected_code_sha256sums or code != expected_code:
        raise ValueError("trainer CODE_SHA256SUMS differs from the trusted repository snapshot")
    _validate_frozen_inputs(
        payload("FROZEN_INPUT_SHA256SUMS", _MAX_JSON_BYTES),
        contract=contract,
        outer_fold=expected_outer_fold,
    )

    fit_payload = payload("fit_identity.json", _MAX_JSON_BYTES)
    if fit_payload != contract.fit_identity_bytes(expected_outer_fold):
        raise ValueError("trainer fit_identity.json differs from the authenticated fold")
    fit_identity = _json_object(
        parse_canonical_json(fit_payload, label="trainer fit identity"),
        tuple(contract.table("rng")["fit_identity_fields"]),
        label="trainer fit identity",
    )
    if not _strict_equal(manifest["fit_identity"], fit_identity):
        raise ValueError("trainer manifest fit identity differs from fit_identity.json")

    _validate_projection_binding(
        manifest["projection"],
        contract=contract,
        outer_fold=expected_outer_fold,
    )
    _validate_model_binding(
        manifest["model"],
        contract=contract,
        outer_fold=expected_outer_fold,
    )
    rng_document = _validate_rng_binding(
        payload("rng.json", _MAX_JSON_BYTES),
        contract=contract,
        outer_fold=expected_outer_fold,
    )
    if not _strict_equal(manifest["rng"], rng_document):
        raise ValueError("trainer manifest RNG differs from rng.json")
    _validate_environment_binding(
        payload("environment.json", _MAX_JSON_BYTES),
        contract=contract,
    )
    training_document = _validate_training_binding(
        bundle=bundle,
        contract=contract,
        outer_fold=expected_outer_fold,
    )
    if not _strict_equal(manifest["training"], training_document):
        raise ValueError("trainer manifest training evidence differs from train_metrics.json")

    count_prior_payload = payload("count_prior.npz", _MAX_COUNT_PRIOR_BYTES)
    count_prior_sha256 = hashlib.sha256(count_prior_payload).hexdigest()
    count_prior = load_count_prior_npz_bytes(
        count_prior_payload,
        expected_sha256=count_prior_sha256,
    )
    checkpoint_digests, metadata_bytes = _authenticate_checkpoints(
        bundle=bundle,
        contract=contract,
        outer_fold=expected_outer_fold,
        count_prior=count_prior,
    )
    manifest_checkpoint_digests = _checkpoint_map_from_json(manifest["checkpoints"])
    if manifest_checkpoint_digests != checkpoint_digests:
        raise ValueError("trainer manifest checkpoint map differs from checkpoint bytes")
    contract.readiness_checkpoint_digest_map_bytes(
        {key: value.document() for key, value in checkpoint_digests.items()}
    )

    return AuthenticatedTrainerBundle(
        bundle=bundle,
        count_prior=count_prior,
        outer_fold=expected_outer_fold,
        fit_identity_sha256=contract.fit_identity_sha256(expected_outer_fold),
        manifest_git_commit=_git_commit(manifest["git_commit"]),
        checkpoint_digest_by_step=checkpoint_digests,
        checkpoint_metadata_bytes_by_step=metadata_bytes,
        _capability=_AUTHENTICATED_BUNDLE_CAPABILITY,
    )


def load_authenticated_trainer_bundle(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    expected_outer_fold: int,
    expected_code_sha256sums: bytes,
    expected_tree_sha256: str | None = None,
) -> AuthenticatedTrainerBundle:
    """Compatibility spelling for :func:`authenticate_trainer_bundle`."""

    return authenticate_trainer_bundle(
        contract,
        root,
        expected_outer_fold=expected_outer_fold,
        expected_code_sha256sums=expected_code_sha256sums,
        expected_tree_sha256=expected_tree_sha256,
    )


def _validate_contract(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    outputs = contract.table("outputs")
    if (
        tuple(outputs["trainer_manifest_fields"]) != _TRAINER_MANIFEST_FIELDS
        or tuple(outputs["trainer_bundle_files"]) != contract.trainer_bundle_files
        or contract.trainer_bundle_files[-1] != "manifest.json"
        or tuple(outputs["trainer_bundle_directories"]) != (".", "checkpoints")
    ):
        raise ValueError("trainer bundle or manifest contract changed")


def _expected_code_manifest(payload: bytes) -> dict[str, str]:
    if type(payload) is not bytes or not payload:
        raise ValueError("expected_code_sha256sums must be non-empty exact bytes")
    document = parse_sha256sums(payload, label="trusted CODE_SHA256SUMS")
    if sha256sums_bytes(document) != payload:
        raise ValueError("trusted CODE_SHA256SUMS is not canonical")
    return document


def _validate_manifest_header(
    manifest: Mapping[str, object],
    *,
    contract: NativeDiffusionV1PilotContract,
    expected_outer_fold: int,
) -> None:
    expected = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "outer_fold": expected_outer_fold,
    }
    for name, value in expected.items():
        _expect(manifest, name, value, label="trainer manifest")
    _git_commit(manifest["git_commit"])


def _validate_artifact_bindings(
    manifest: Mapping[str, object],
    *,
    bundle: BundleSnapshot,
    contract: NativeDiffusionV1PilotContract,
) -> None:
    artifacts = _json_object(
        manifest["artifacts"],
        contract.trainer_bundle_files[:-1],
        label="trainer manifest artifacts",
    )
    expected = {path: bundle.file(path).sha256 for path in contract.trainer_bundle_files[:-1]}
    for path in contract.trainer_bundle_files[:-1]:
        digest = _sha256(artifacts[path], label=f"trainer artifact {path}")
        if digest != expected[path]:
            raise ValueError(f"trainer manifest artifact digest differs for {path}")


def _validate_frozen_inputs(
    payload: bytes,
    *,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
) -> None:
    observed = parse_sha256sums(payload, label="trainer FROZEN_INPUT_SHA256SUMS")
    expected = {
        "pilot_execution_v1.toml": contract.config_sha256,
        "train.jsonl": contract.fold(outer_fold).train_sha256,
        "unconditional_v1.toml": contract.parent_config_sha256,
    }
    if observed != expected or payload != sha256sums_bytes(expected):
        raise ValueError("trainer FROZEN_INPUT_SHA256SUMS differs from exact frozen inputs")


def _validate_projection_binding(
    value: object,
    *,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
) -> None:
    fields = (
        "accepted_bundle_tree_sha256",
        "accepted_top_manifest_sha256",
        "fit_folds",
        "train_sha256",
        "train_rows",
        "train_homology_components",
        "train_union_components",
    )
    projection = _json_object(value, fields, label="trainer manifest projection")
    fold = contract.fold(outer_fold)
    expected: dict[str, object] = {
        "accepted_bundle_tree_sha256": contract.projection_tree_sha256,
        "accepted_top_manifest_sha256": contract.projection_top_sha256,
        "fit_folds": list(fold.fit_folds),
        "train_sha256": fold.train_sha256,
        "train_rows": fold.train_rows,
        "train_homology_components": fold.train_homology_components,
        "train_union_components": fold.train_union_components,
    }
    _require_strict_equal(projection, expected, label="trainer projection binding")


def _validate_model_binding(
    value: object,
    *,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
) -> None:
    model = _json_object(value, ("config", "initial_model_sha256"), label="trainer model")
    config = r128_model_config_document()
    child_model = contract.table("model")
    for name, expected in config.items():
        candidate = child_model[name]
        if isinstance(expected, list):
            if not isinstance(candidate, tuple) or list(candidate) != expected:
                raise ValueError(f"authenticated child model.{name} differs from R128")
        elif type(candidate) is not type(expected) or candidate != expected:
            raise ValueError(f"authenticated child model.{name} differs from R128")
    expected_model = {
        "config": config,
        "initial_model_sha256": R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD[outer_fold],
    }
    _require_strict_equal(model, expected_model, label="trainer model binding")


def _validate_rng_binding(
    payload: bytes,
    *,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
) -> dict[str, object]:
    observed = _json_object(
        parse_canonical_json(payload, label="trainer rng.json"),
        _RNG_FIELDS,
        label="trainer rng.json",
    )
    rng = contract.table("rng")
    fit_identity = contract.fit_identity_sha256(outer_fold)
    expected: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_training_rng",
        "derivation": rng["derivation"],
        "training_root_seed": rng["training_root_seed"],
        "fit_identity_sha256": fit_identity,
        "initialization_seed": initialization_seed(fit_identity, root_seed=contract.seed),
        "training_namespaces": list(rng["training_namespaces"]),
        "global_draw_ordinal": rng["global_draw_ordinal"],
        "checkpoint_restores_mutable_rng_state": rng["checkpoint_restores_mutable_rng_state"],
    }
    _require_strict_equal(observed, expected, label="trainer RNG binding")
    return observed


def _validate_environment_binding(
    payload: bytes,
    *,
    contract: NativeDiffusionV1PilotContract,
) -> None:
    environment = _json_object(
        parse_canonical_json(payload, label="trainer environment.json"),
        _ENVIRONMENT_FIELDS,
        label="trainer environment.json",
    )
    runtime = _json_object(
        environment["runtime"],
        _RUNTIME_FIELDS,
        label="trainer environment runtime",
    )
    determinism = _json_object(
        environment["determinism"],
        _DETERMINISM_FIELDS,
        label="trainer environment determinism",
    )
    parent_environment = contract.parent_table("environment")
    parent_determinism = contract.parent_table("determinism")
    cudnn_parts = str(parent_environment["nvidia_cudnn_cu13"]).split(".")
    if len(cudnn_parts) < 3 or any(not part.isdigit() for part in cudnn_parts[:3]):
        raise ValueError("authenticated parent cuDNN version is invalid")
    expected_runtime: dict[str, object] = {
        "python": parent_environment["python"],
        "numpy": parent_environment["numpy"],
        "torch": parent_environment["torch"],
        "torch_cuda": parent_environment["torch_cuda"],
        "safetensors": parent_environment["safetensors"],
        "packaging": parent_environment["packaging"],
        "triton": parent_environment["triton"],
        "nvidia_cudnn_cu13": parent_environment["nvidia_cudnn_cu13"],
        "gpu_name": parent_environment["gpu_name"],
        "compute_capability": list(parent_environment["compute_capability"]),
        "visible_cuda_devices": 1,
        "allocator": parent_determinism["pytorch_allocator"],
        "cudnn_runtime_version": (
            int(cudnn_parts[0]) * 10_000 + int(cudnn_parts[1]) * 100 + int(cudnn_parts[2])
        ),
    }
    expected_determinism: dict[str, object] = {
        "cublas_workspace_config": parent_determinism["cublas_workspace_config"],
        "pytorch_allocator": parent_determinism["pytorch_allocator"],
        "deterministic_algorithms": True,
        "math_sdpa_only": True,
        "tf32": False,
        "amp": False,
    }
    _expect(environment, "schema_version", 1, label="trainer environment")
    _expect(
        environment,
        "artifact",
        "native_categorical_diffusion_v1_r128_training_environment",
        label="trainer environment",
    )
    _require_strict_equal(runtime, expected_runtime, label="trainer runtime binding")
    _require_strict_equal(
        determinism,
        expected_determinism,
        label="trainer deterministic-runtime binding",
    )


def _validate_training_binding(
    *,
    bundle: BundleSnapshot,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
) -> dict[str, object]:
    recipe = training_recipe_from_contract(contract)
    schedule_sha256 = learning_rate_schedule_sha256(recipe)
    observed_schedule = parse_sha256_sidecar(
        bundle.read_bytes("training_schedule.sha256", maximum_bytes=65),
        label="trainer training_schedule.sha256",
    )
    if observed_schedule != schedule_sha256:
        raise ValueError("trainer learning-rate schedule differs from the contract")
    trace_payload = bundle.read_bytes("training_trace.jsonl", maximum_bytes=_MAX_TRACE_BYTES)
    rows = parse_canonical_jsonl(trace_payload, label="trainer training_trace.jsonl")
    if len(rows) != recipe.max_steps:
        raise ValueError("trainer trace must contain exactly 4000 optimizer steps")
    losses: list[float] = []
    accuracies: list[float] = []
    gradients: list[float] = []
    total_selected_tokens = 0
    fit_identity = contract.fit_identity_sha256(outer_fold)
    for step, raw in enumerate(rows, start=1):
        row = _json_object(
            raw,
            _TRAINING_TRACE_FIELDS,
            label=f"trainer trace step {step}",
        )
        _expect(row, "step", step, label=f"trainer trace step {step}")
        _expect(
            row,
            "learning_rate",
            learning_rate_for_step(recipe, step),
            label=f"trainer trace step {step}",
        )
        loss = _finite_float(row["loss"], label=f"trainer trace step {step} loss")
        accuracy = _finite_float(
            row["mean_row_accuracy"],
            label=f"trainer trace step {step} mean_row_accuracy",
        )
        gradient = _finite_float(
            row["gradient_norm_before_clipping"],
            label=f"trainer trace step {step} gradient norm",
        )
        if loss < 0.0 or not 0.0 <= accuracy <= 1.0 or gradient < 0.0:
            raise ValueError(f"trainer trace step {step} has out-of-range metrics")
        selected = _integer(
            row["selected_tokens"],
            label=f"trainer trace step {step} selected_tokens",
            minimum=1,
        )
        if selected > recipe.batch_sequences * 50:
            raise ValueError(f"trainer trace step {step} selected-token count is impossible")
        expected_dropout_seed = model_dropout_seed(
            fit_identity,
            step,
            root_seed=contract.seed,
        )
        _expect(
            row,
            "model_dropout_seed",
            expected_dropout_seed,
            label=f"trainer trace step {step}",
        )
        _sha256(row["batch_sha256"], label=f"trainer trace step {step} batch")
        losses.append(loss)
        accuracies.append(accuracy)
        gradients.append(gradient)
        total_selected_tokens += selected

    metrics = _json_object(
        parse_canonical_json(
            bundle.read_bytes("train_metrics.json", maximum_bytes=_MAX_JSON_BYTES),
            label="trainer train_metrics.json",
        ),
        _TRAIN_METRIC_FIELDS,
        label="trainer train_metrics.json",
    )
    expected_metrics: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_train_metrics",
        "outer_fold": outer_fold,
        "fit_identity_sha256": fit_identity,
        "completed_steps": recipe.max_steps,
        "batch_sequences": recipe.batch_sequences,
        "total_sequence_draws": recipe.max_steps * recipe.batch_sequences,
        "total_selected_tokens": total_selected_tokens,
        "mean_loss": math.fsum(losses) / len(losses),
        "mean_row_accuracy": math.fsum(accuracies) / len(accuracies),
        "mean_gradient_norm_before_clipping": math.fsum(gradients) / len(gradients),
        "learning_rate_schedule_sha256": schedule_sha256,
        "training_trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "checkpoint_steps": list(contract.checkpoint_steps),
    }
    _require_strict_equal(metrics, expected_metrics, label="trainer training metrics")
    return metrics


def _authenticate_checkpoints(
    *,
    bundle: BundleSnapshot,
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    count_prior: AuthenticatedCountPrior,
) -> tuple[Mapping[str, CheckpointDigest], Mapping[str, bytes]]:
    metadata_paths = contract.table("checkpoints")["metadata_relative_paths"]
    if not isinstance(metadata_paths, tuple) or len(metadata_paths) != len(
        contract.checkpoint_steps
    ):
        raise ValueError("checkpoint metadata path contract changed")
    establish_r128_deterministic_runtime(contract)
    digests: dict[str, CheckpointDigest] = {}
    metadata_by_step: dict[str, bytes] = {}
    for step, checkpoint_path, metadata_path in zip(
        contract.checkpoint_steps,
        contract.checkpoint_paths,
        metadata_paths,
        strict=True,
    ):
        if type(metadata_path) is not str:
            raise ValueError("checkpoint metadata path must be an exact string")
        checkpoint_payload = bundle.read_bytes(
            checkpoint_path,
            maximum_bytes=_MAX_CHECKPOINT_BYTES,
        )
        metadata_payload = bundle.read_bytes(
            metadata_path,
            maximum_bytes=_MAX_METADATA_BYTES,
        )
        metadata = _json_object(
            parse_canonical_json(
                metadata_payload,
                label=f"trainer checkpoint metadata {step}",
            ),
            tuple(contract.table("checkpoint_metadata")["fields"]),
            label=f"trainer checkpoint metadata {step}",
        )
        physical_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
        logical_sha256 = _sha256(
            metadata["checkpoint_logical_state_sha256"],
            label=f"trainer checkpoint {step} logical state",
        )
        expected_identity = CheckpointIdentity(
            checkpoint_file_sha256=physical_sha256,
            checkpoint_logical_state_sha256=logical_sha256,
            checkpoint_metadata_sha256=metadata_sha256,
            metadata_bytes=metadata_payload,
        )
        observed = authenticate_r128_checkpoint_bytes(
            checkpoint_payload,
            metadata_bytes=metadata_payload,
            contract=contract,
            outer_fold=outer_fold,
            checkpoint_step=step,
            count_prior=count_prior,
            expected_identity=expected_identity,
        )
        key = f"{step:06d}"
        digests[key] = CheckpointDigest(
            checkpoint_file_sha256=observed.checkpoint_file_sha256,
            checkpoint_logical_state_sha256=observed.checkpoint_logical_state_sha256,
            checkpoint_metadata_sha256=observed.checkpoint_metadata_sha256,
        )
        metadata_by_step[key] = bytes(metadata_payload)
    return MappingProxyType(digests), MappingProxyType(metadata_by_step)


def _checkpoint_map_from_json(value: object) -> Mapping[str, CheckpointDigest]:
    document = _json_object(value, _CHECKPOINT_KEYS, label="trainer checkpoint map")
    result = {
        key: CheckpointDigest.from_document(
            _json_object(
                document[key],
                _CHECKPOINT_DIGEST_FIELDS,
                label=f"trainer checkpoint map.{key}",
            ),
            label=f"trainer checkpoint map.{key}",
        )
        for key in _CHECKPOINT_KEYS
    }
    return MappingProxyType(result)


def _freeze_checkpoint_digests(
    value: Mapping[str, CheckpointDigest],
) -> Mapping[str, CheckpointDigest]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint_digest_by_step must be a mapping")
    _exact_keys(value, _CHECKPOINT_KEYS, label="checkpoint_digest_by_step")
    result: dict[str, CheckpointDigest] = {}
    for key in _CHECKPOINT_KEYS:
        item = value[key]
        if type(item) is not CheckpointDigest:
            raise TypeError(f"checkpoint digest {key} must be an exact CheckpointDigest")
        result[key] = CheckpointDigest(**item.document())
    return MappingProxyType(result)


def _freeze_metadata_bytes(value: Mapping[str, bytes]) -> Mapping[str, bytes]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint_metadata_bytes_by_step must be a mapping")
    _exact_keys(value, _CHECKPOINT_KEYS, label="checkpoint_metadata_bytes_by_step")
    result: dict[str, bytes] = {}
    for key in _CHECKPOINT_KEYS:
        payload = value[key]
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_METADATA_BYTES:
            raise ValueError(f"checkpoint metadata {key} must be non-empty bounded exact bytes")
        result[key] = bytes(payload)
    return MappingProxyType(result)


def _json_object(
    value: object,
    fields: tuple[str, ...],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    _exact_keys(value, fields, label=label)
    return value


def _exact_keys(value: Mapping[object, object], fields: tuple[str, ...], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} contains a non-string key")
    expected = set(fields)
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _expect(
    document: Mapping[str, object],
    name: str,
    expected: object,
    *,
    label: str,
) -> None:
    observed = document[name]
    if not _strict_equal(observed, expected):
        raise ValueError(f"{label}.{name} differs from its authenticated binding")


def _require_strict_equal(left: object, right: object, *, label: str) -> None:
    if not _strict_equal(left, right):
        raise ValueError(f"{label} differs from its authenticated value")


def _strict_equal(left: object, right: object) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        if type(left) is not dict or type(right) is not dict or set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        return (
            type(left) is list
            and type(right) is list
            and len(left) == len(right)
            and all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer at least {minimum}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        raise ValueError("Git commit must be a lowercase forty-character object ID")
    return value


__all__ = [
    "AuthenticatedTrainerBundle",
    "authenticate_trainer_bundle",
    "load_authenticated_trainer_bundle",
]

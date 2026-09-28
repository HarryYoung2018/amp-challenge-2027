"""Authority-bearing train-only producer for one native-diffusion v1 pilot fold.

The command-line surface intentionally has no score input, projection-root
input, seed, hyperparameter, resume, or checkpoint-choice option.  A Slurm
worker must first place exactly one authenticated ``train.jsonl`` in a private
node-local directory and terminate this process before staging score data.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    BundleSnapshot,
    build_repository_snapshot,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_bundle,
    sha256sums_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_checkpoint import (
    CheckpointIdentity,
    seal_checkpoint,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    load_pilot_execution_v1_contract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    _read_regular_bytes,
    load_pilot_training_projection,
)
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD,
    r128_model_config_document,
)
from amp_challenge.generators.diffusion.v1.pilot_training import (
    PilotFitSession,
    TrainingStepResult,
    advance_fit_to_next_checkpoint,
    initialize_fit_session,
    learning_rate_schedule_sha256,
    validate_fit_session,
)

_GIT_RE = re.compile(r"[0-9a-f]{40}")
_MAX_CONTRACT_BYTES = 262_144
_MAX_TRAIN_BYTES = 64 * 1024 * 1024
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_CHECKPOINT_DIGEST_FIELDS = (
    "checkpoint_file_sha256",
    "checkpoint_logical_state_sha256",
    "checkpoint_metadata_sha256",
)
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "AMP_PILOT_PROJECTION_ROOT",
        "AMP_PILOT_SCORE_PATH",
        "AMP_PILOT_SCORE_ROOT",
        "AMP_PILOT_SCORE_JSONL",
    }
)


@dataclass(frozen=True, slots=True)
class TrainerBundleResult:
    """Sealed trainer identity returned to the operational supervisor."""

    bundle: BundleSnapshot
    outer_fold: int
    fit_identity_sha256: str
    count_prior_file_sha256: str
    checkpoint_digest_by_step: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if type(self.bundle) is not BundleSnapshot:
            raise TypeError("bundle must be an exact BundleSnapshot")
        if type(self.outer_fold) is not int or self.outer_fold not in (0, 1, 2, 3):
            raise ValueError("outer_fold must be an exact integer in 0..3")
        _require_sha256(self.fit_identity_sha256, label="fit identity SHA-256")
        _require_sha256(self.count_prior_file_sha256, label="count-prior SHA-256")
        if not isinstance(self.checkpoint_digest_by_step, Mapping) or set(
            self.checkpoint_digest_by_step
        ) != set(_CHECKPOINT_KEYS):
            raise ValueError("checkpoint digest map must contain the exact five steps")
        frozen: dict[str, Mapping[str, str]] = {}
        for key in _CHECKPOINT_KEYS:
            raw = self.checkpoint_digest_by_step[key]
            if not isinstance(raw, Mapping) or set(raw) != set(_CHECKPOINT_DIGEST_FIELDS):
                raise ValueError(f"checkpoint digest {key} has the wrong exact schema")
            frozen[key] = MappingProxyType(
                {
                    name: _require_sha256(
                        raw[name],
                        label=f"checkpoint {key} {name}",
                    )
                    for name in _CHECKPOINT_DIGEST_FIELDS
                }
            )
        object.__setattr__(
            self,
            "checkpoint_digest_by_step",
            MappingProxyType(frozen),
        )


def run_pilot_trainer(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    node_local_train_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    outer_fold: int,
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> TrainerBundleResult:
    """Run and seal exactly one production R128 fold without score authority."""

    _reject_score_authority()
    if type(outer_fold) is not int or outer_fold not in (0, 1, 2, 3):
        raise ValueError("outer_fold must be an exact integer in 0..3")
    if type(expected_git_commit) is not str or _GIT_RE.fullmatch(expected_git_commit) is None:
        raise ValueError("expected_git_commit must be a lowercase 40-character Git ID")
    train_source = _node_local_train_source(node_local_train_path)
    contract = load_pilot_execution_v1_contract(
        child_contract_path,
        parent_path=parent_contract_path,
    )
    contract.revalidate()
    fold = contract.fold(outer_fold)
    child_payload = _read_regular_bytes(
        child_contract_path,
        maximum_bytes=_MAX_CONTRACT_BYTES,
        label="pilot child contract",
    )
    parent_payload = _read_regular_bytes(
        parent_contract_path,
        maximum_bytes=_MAX_CONTRACT_BYTES,
        label="pilot parent contract",
    )
    if hashlib.sha256(child_payload).hexdigest() != contract.config_sha256:
        raise ValueError("pilot child contract changed after authentication")
    if hashlib.sha256(parent_payload).hexdigest() != contract.parent_config_sha256:
        raise ValueError("pilot parent contract changed after authentication")
    repository = build_repository_snapshot(
        repository_root,
        expected_commit=expected_git_commit,
    )
    train_payload = _read_regular_bytes(
        train_source,
        maximum_bytes=_MAX_TRAIN_BYTES,
        label="node-local train-only projection",
        required_mode=0o400,
    )
    if hashlib.sha256(train_payload).hexdigest() != fold.train_sha256:
        raise ValueError("node-local train-only projection differs from its fold pin")
    projection = load_pilot_training_projection(
        train_source,
        expected_sha256=fold.train_sha256,
        expected_rows=fold.train_rows,
    )

    temporary_parent = _private_temporary_parent()
    with tempfile.TemporaryDirectory(
        prefix=f"amp-r128-trainer-f{outer_fold}-",
        dir=temporary_parent,
    ) as raw_work_root:
        work_root = Path(raw_work_root)
        os.chmod(work_root, 0o700)
        session = initialize_fit_session(
            contract,
            outer_fold,
            projection,
            work_root,
            device="cuda:0",
        )
        results: list[TrainingStepResult] = []
        checkpoint_identities: dict[int, CheckpointIdentity] = {}
        while session.next_checkpoint_step is not None:
            interval = advance_fit_to_next_checkpoint(session)
            results.extend(interval)
            identity = seal_checkpoint(session, work_root)
            checkpoint_identities[session.completed_step] = identity
        validate_fit_session(
            session,
            require_completed_checkpoint=True,
            require_production=True,
        )
        payloads, manifest = _trainer_bundle_payloads(
            contract=contract,
            session=session,
            training_steps=tuple(results),
            checkpoint_identities=checkpoint_identities,
            work_root=work_root,
            child_contract_payload=child_payload,
            parent_contract_payload=parent_payload,
            code_sha256sums=repository.code_sha256sums,
            git_commit=repository.git_commit,
        )

        _revalidate_immutable_inputs(
            contract=contract,
            child_contract_path=child_contract_path,
            parent_contract_path=parent_contract_path,
            child_payload=child_payload,
            parent_payload=parent_payload,
            train_source=train_source,
            train_payload=train_payload,
            repository_root=repository_root,
            expected_git_commit=expected_git_commit,
            expected_code_sha256=repository.code_sha256,
        )
        bundle = publish_bundle(
            contract,
            bundle_kind="trainer",
            output_dir=output_dir,
            payloads=payloads,
            manifest=manifest,
        )

    checkpoint_map = _checkpoint_digest_map(contract, checkpoint_identities)
    count_prior_sha256 = session.count_prior.sha256
    if bundle.file("count_prior.npz").sha256 != count_prior_sha256:
        raise RuntimeError("published trainer bundle changed its count prior")
    return TrainerBundleResult(
        bundle=bundle,
        outer_fold=outer_fold,
        fit_identity_sha256=fold.fit_identity_sha256,
        count_prior_file_sha256=count_prior_sha256,
        checkpoint_digest_by_step=checkpoint_map,
    )


def _trainer_bundle_payloads(
    *,
    contract: NativeDiffusionV1PilotContract,
    session: PilotFitSession,
    training_steps: tuple[TrainingStepResult, ...],
    checkpoint_identities: Mapping[int, CheckpointIdentity],
    work_root: Path,
    child_contract_payload: bytes,
    parent_contract_payload: bytes,
    code_sha256sums: bytes,
    git_commit: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    contract.revalidate()
    validate_fit_session(
        session,
        require_completed_checkpoint=True,
        require_production=True,
    )
    if tuple(result.step for result in training_steps) != tuple(range(1, 4001)):
        raise ValueError("training trace must contain each optimizer step 1..4000 exactly once")
    checkpoints = _checkpoint_digest_map(contract, checkpoint_identities)
    trace_rows = [_training_step_record(result) for result in training_steps]
    trace_payload = canonical_jsonl_bytes(trace_rows)
    schedule_sha256 = learning_rate_schedule_sha256(session.recipe)
    metrics = _training_metrics_document(
        session,
        training_steps=training_steps,
        schedule_sha256=schedule_sha256,
        trace_sha256=hashlib.sha256(trace_payload).hexdigest(),
    )
    environment = _environment_document(session)
    rng = _rng_document(session)
    frozen_inputs = sha256sums_bytes(
        {
            "pilot_execution_v1.toml": contract.config_sha256,
            "train.jsonl": session.fold.train_sha256,
            "unconditional_v1.toml": contract.parent_config_sha256,
        }
    )
    payloads: dict[str, bytes] = {
        "CODE_SHA256SUMS": code_sha256sums,
        "FROZEN_INPUT_SHA256SUMS": frozen_inputs,
        "pilot_execution_v1.toml": child_contract_payload,
        "unconditional_v1.toml": parent_contract_payload,
        "count_prior.npz": session.count_prior.payload,
        "environment.json": canonical_json_bytes(environment),
        "fit_identity.json": contract.fit_identity_bytes(session.outer_fold),
        "rng.json": canonical_json_bytes(rng),
        "training_schedule.sha256": contract.sha256_sidecar_bytes(schedule_sha256),
        "training_trace.jsonl": trace_payload,
        "train_metrics.json": canonical_json_bytes(metrics),
    }
    metadata_paths = contract.table("checkpoints")["metadata_relative_paths"]
    if not isinstance(metadata_paths, tuple):
        raise TypeError("checkpoint metadata paths lost their frozen tuple")
    for step, checkpoint_path, metadata_path in zip(
        contract.checkpoint_steps,
        contract.checkpoint_paths,
        metadata_paths,
        strict=True,
    ):
        identity = checkpoint_identities[step]
        payloads[checkpoint_path] = _read_regular_bytes(
            work_root / checkpoint_path,
            maximum_bytes=1 << 30,
            label=f"sealed checkpoint {step}",
            required_mode=0o444,
        )
        payloads[metadata_path] = _read_regular_bytes(
            work_root / metadata_path,
            maximum_bytes=1 << 20,
            label=f"sealed checkpoint metadata {step}",
            required_mode=0o444,
        )
        if (
            hashlib.sha256(payloads[checkpoint_path]).hexdigest() != identity.checkpoint_file_sha256
            or hashlib.sha256(payloads[metadata_path]).hexdigest()
            != identity.checkpoint_metadata_sha256
        ):
            raise ValueError("sealed checkpoint pair changed before bundle construction")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": contract.document["artifact"],
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": git_commit,
        "outer_fold": session.outer_fold,
        "fit_identity": dict(contract.fit_identity_document(session.outer_fold)),
        "projection": {
            "accepted_bundle_tree_sha256": contract.projection_tree_sha256,
            "accepted_top_manifest_sha256": contract.projection_top_sha256,
            "fit_folds": list(session.fold.fit_folds),
            "train_sha256": session.fold.train_sha256,
            "train_rows": session.fold.train_rows,
            "train_homology_components": session.fold.train_homology_components,
            "train_union_components": session.fold.train_union_components,
        },
        "model": {
            "config": r128_model_config_document(),
            "initial_model_sha256": R128_INITIAL_MODEL_SHA256_BY_OUTER_FOLD[session.outer_fold],
        },
        "rng": rng,
        "training": metrics,
        "checkpoints": checkpoints,
    }
    return payloads, manifest


def _training_step_record(value: TrainingStepResult) -> dict[str, object]:
    if type(value) is not TrainingStepResult:
        raise TypeError("training trace accepts exact TrainingStepResult values")
    value.__post_init__()
    return {
        "step": value.step,
        "learning_rate": value.learning_rate,
        "loss": value.loss,
        "mean_row_accuracy": value.mean_row_accuracy,
        "gradient_norm_before_clipping": value.gradient_norm_before_clipping,
        "selected_tokens": value.selected_tokens,
        "model_dropout_seed": value.model_dropout_seed,
        "batch_sha256": value.batch_sha256,
    }


def _training_metrics_document(
    session: PilotFitSession,
    *,
    training_steps: tuple[TrainingStepResult, ...],
    schedule_sha256: str,
    trace_sha256: str,
) -> dict[str, object]:
    _require_sha256(schedule_sha256, label="schedule SHA-256")
    _require_sha256(trace_sha256, label="training trace SHA-256")
    losses = tuple(value.loss for value in training_steps)
    accuracies = tuple(value.mean_row_accuracy for value in training_steps)
    gradient_norms = tuple(value.gradient_norm_before_clipping for value in training_steps)
    if not losses or any(
        not math.isfinite(value) for value in (*losses, *accuracies, *gradient_norms)
    ):
        raise ValueError("training metrics contain non-finite evidence")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_train_metrics",
        "outer_fold": session.outer_fold,
        "fit_identity_sha256": session.fit_identity_sha256,
        "completed_steps": session.completed_step,
        "batch_sequences": session.recipe.batch_sequences,
        "total_sequence_draws": session.completed_step * session.recipe.batch_sequences,
        "total_selected_tokens": sum(value.selected_tokens for value in training_steps),
        "mean_loss": math.fsum(losses) / len(losses),
        "mean_row_accuracy": math.fsum(accuracies) / len(accuracies),
        "mean_gradient_norm_before_clipping": math.fsum(gradient_norms) / len(gradient_norms),
        "learning_rate_schedule_sha256": schedule_sha256,
        "training_trace_sha256": trace_sha256,
        "checkpoint_steps": list(session.recipe.checkpoint_steps),
    }


def _environment_document(session: PilotFitSession) -> dict[str, object]:
    observed = session.production_environment
    if observed is None:
        raise ValueError("production session did not retain its environment identity")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_training_environment",
        "runtime": observed,
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pytorch_allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
            "deterministic_algorithms": True,
            "math_sdpa_only": True,
            "tf32": False,
            "amp": False,
        },
    }


def _rng_document(session: PilotFitSession) -> dict[str, object]:
    rng = session.contract.table("rng")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_training_rng",
        "derivation": rng["derivation"],
        "training_root_seed": rng["training_root_seed"],
        "fit_identity_sha256": session.fit_identity_sha256,
        "initialization_seed": session.initialization_seed,
        "training_namespaces": list(rng["training_namespaces"]),
        "global_draw_ordinal": rng["global_draw_ordinal"],
        "checkpoint_restores_mutable_rng_state": rng["checkpoint_restores_mutable_rng_state"],
    }


def _checkpoint_digest_map(
    contract: NativeDiffusionV1PilotContract,
    values: Mapping[int, CheckpointIdentity],
) -> dict[str, dict[str, str]]:
    if set(values) != set(contract.checkpoint_steps):
        raise ValueError("checkpoint identities differ from the frozen five-step set")
    result = {
        f"{step:06d}": {
            "checkpoint_file_sha256": values[step].checkpoint_file_sha256,
            "checkpoint_logical_state_sha256": values[step].checkpoint_logical_state_sha256,
            "checkpoint_metadata_sha256": values[step].checkpoint_metadata_sha256,
        }
        for step in contract.checkpoint_steps
    }
    contract.readiness_checkpoint_digest_map_bytes(result)
    return result


def _revalidate_immutable_inputs(
    *,
    contract: NativeDiffusionV1PilotContract,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    child_payload: bytes,
    parent_payload: bytes,
    train_source: Path,
    train_payload: bytes,
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
    expected_code_sha256: str,
) -> None:
    contract.revalidate()
    if _node_local_train_source(train_source) != train_source:
        raise RuntimeError("node-local train-only projection identity changed")
    inputs = (
        (child_contract_path, child_payload, _MAX_CONTRACT_BYTES, "pilot child contract"),
        (parent_contract_path, parent_payload, _MAX_CONTRACT_BYTES, "pilot parent contract"),
        (train_source, train_payload, _MAX_TRAIN_BYTES, "node-local train-only projection"),
    )
    for path, expected, maximum, label in inputs:
        observed = _read_regular_bytes(
            path,
            maximum_bytes=maximum,
            label=label,
            required_mode=0o400 if path == train_source else None,
        )
        if observed != expected:
            raise RuntimeError(f"{label} changed during training")
    repository = build_repository_snapshot(
        repository_root,
        expected_commit=expected_git_commit,
    )
    if repository.code_sha256 != expected_code_sha256:
        raise RuntimeError("repository changed during training")


def _node_local_train_source(path: str | os.PathLike[str]) -> Path:
    source = Path(os.path.abspath(os.fspath(path)))
    temporary_raw = os.environ.get("TMPDIR")
    if not temporary_raw:
        raise ValueError("TMPDIR must identify the private node-local trainer directory")
    temporary = Path(os.path.abspath(temporary_raw))
    temporary_stat = os.lstat(temporary)
    parent_stat = os.lstat(source.parent)
    source_stat = os.lstat(source)
    if stat.S_ISLNK(temporary_stat.st_mode) or not stat.S_ISDIR(temporary_stat.st_mode):
        raise ValueError("TMPDIR must be a real directory")
    try:
        source.relative_to(temporary)
    except ValueError as error:
        raise ValueError("train-only input must be staged beneath TMPDIR") from error
    if source.name != "train.jsonl" or tuple(child.name for child in source.parent.iterdir()) != (
        "train.jsonl",
    ):
        raise ValueError("trainer input directory must contain exactly train.jsonl")
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or stat.S_ISLNK(source_stat.st_mode)
        or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_nlink != 1
        or stat.S_IMODE(source_stat.st_mode) != 0o400
    ):
        raise ValueError("node-local train-only input is not private and sealed")
    return source


def _private_temporary_parent() -> str:
    raw = os.environ.get("TMPDIR")
    if not raw:
        raise ValueError("TMPDIR is required for private trainer working state")
    path = Path(os.path.abspath(raw))
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("TMPDIR is not a real directory")
    return os.fspath(path)


def _reject_score_authority() -> None:
    present = sorted(_FORBIDDEN_ENVIRONMENT_NAMES & set(os.environ))
    if present:
        raise RuntimeError("trainer environment contains forbidden score/projection authority")
    for name, value in os.environ.items():
        if name.startswith("AMP_PILOT_SCORE_") or "/development-projections/" in value:
            raise RuntimeError("trainer environment leaks score/projection authority")


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-contract", required=True)
    parser.add_argument("--parent-contract", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pilot_trainer(
        child_contract_path=args.child_contract,
        parent_contract_path=args.parent_contract,
        node_local_train_path=args.train_jsonl,
        output_dir=args.output_dir,
        outer_fold=args.outer_fold,
        repository_root=args.repository_root,
        expected_git_commit=args.expected_git_commit,
    )
    print(
        canonical_json_bytes(
            {
                "bundle_tree_sha256": result.bundle.tree_sha256,
                "checkpoint_digest_by_step": result.checkpoint_digest_by_step,
                "count_prior_file_sha256": result.count_prior_file_sha256,
                "fit_identity_sha256": result.fit_identity_sha256,
                "outer_fold": result.outer_fold,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

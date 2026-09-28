"""Release-gated evaluator for one native-diffusion v1 R128 pilot fold.

The evaluator is a separate process from training.  It accepts exactly one
private node-local score projection and opens it only after an immutable
four-fold release token has been tied to this fold's reopened trainer bundle.
Checkpoint authentication, scoring, bundle publication, and the mandatory
fresh GPU reinference pass all happen inside this boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    BundleSnapshot,
    RepositorySnapshot,
    build_repository_snapshot,
    canonical_json_bytes,
    publish_bundle,
    sha256sums_bytes,
    verify_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_checkpoint import (
    CheckpointIdentity,
    load_r128_safetensors_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    load_pilot_execution_v1_contract,
)
from amp_challenge.generators.diffusion.v1.pilot_control import (
    CheckpointDigest,
    ScoreReleaseReceipt,
    TrainerReadinessReceipt,
    parse_score_release_receipt,
    parse_trainer_readiness_receipt,
    trusted_checkpoint_identity_from_readiness,
    verify_trainer_readiness_receipt,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    AuthenticatedCountPrior,
    _read_regular_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_inference import (
    AuthenticatedCheckpointLoader,
    LoadedCheckpoint,
    ReinferenceComparison,
    compare_archived_reinference,
    infer_all_checkpoint_slices,
)
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128Denoiser,
    build_r128_model_from_contract,
    verify_r128_production_environment,
)
from amp_challenge.generators.diffusion.v1.pilot_progress import (
    EvaluatorProgressWriter,
)
from amp_challenge.generators.diffusion.v1.pilot_records import (
    fold_metrics_document,
)
from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    CHECKPOINT_STEPS,
    FoldMethodMetrics,
    ScoreCorruptionLedger,
    ScoringArchive,
    build_scoring_archive,
    gather_count_log_probability,
    load_contract_score_corruption_ledger,
    load_deterministic_npz_bytes,
    residual_logits_npz_schema,
    score_all_fold_methods,
)
from amp_challenge.generators.diffusion.v1.pilot_trainer_bundle import (
    AuthenticatedTrainerBundle,
    authenticate_trainer_bundle,
)

_GIT_RE = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_MAX_CONTRACT_BYTES = 262_144
_MAX_RECEIPT_BYTES = 1 << 20
_MAX_SCORE_BYTES = 64 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 1 << 30
_MAX_METADATA_BYTES = 1 << 20
_FORBIDDEN_SCORE_ENVIRONMENT = frozenset(
    {
        "AMP_PILOT_PROJECTION_ROOT",
        "AMP_PILOT_SCORE_ROOT",
        "AMP_PILOT_SCORE_PATH_BY_FOLD",
        "AMP_PILOT_SCORE_JSONL_BY_FOLD",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluatorBundleResult:
    """Sealed evaluator identity and path-free producer reinference evidence."""

    bundle: BundleSnapshot
    outer_fold: int
    trainer_bundle_sha256: str
    count_prior_file_sha256: str
    checkpoint_digest_by_step: Mapping[str, CheckpointDigest]
    reinference_by_step: Mapping[str, ReinferenceComparison]

    def __post_init__(self) -> None:
        if type(self.bundle) is not BundleSnapshot:
            raise TypeError("bundle must be an exact BundleSnapshot")
        if type(self.outer_fold) is not int or self.outer_fold not in (0, 1, 2, 3):
            raise ValueError("outer_fold must be an exact integer in 0..3")
        _sha256(self.trainer_bundle_sha256, label="trainer bundle SHA-256")
        _sha256(self.count_prior_file_sha256, label="count-prior SHA-256")
        if not isinstance(self.checkpoint_digest_by_step, Mapping) or set(
            self.checkpoint_digest_by_step
        ) != set(_CHECKPOINT_KEYS):
            raise ValueError("checkpoint digest map must contain the exact five steps")
        checkpoint_map: dict[str, CheckpointDigest] = {}
        for key in _CHECKPOINT_KEYS:
            digest = self.checkpoint_digest_by_step[key]
            if type(digest) is not CheckpointDigest:
                raise TypeError("checkpoint digest map values must be exact digests")
            checkpoint_map[key] = digest
        object.__setattr__(
            self,
            "checkpoint_digest_by_step",
            MappingProxyType(checkpoint_map),
        )
        if not isinstance(self.reinference_by_step, Mapping) or set(
            self.reinference_by_step
        ) != set(_CHECKPOINT_KEYS):
            raise ValueError("reinference map must follow the exact five checkpoint keys")
        frozen: dict[str, ReinferenceComparison] = {}
        for key in _CHECKPOINT_KEYS:
            value = self.reinference_by_step[key]
            if type(value) is not ReinferenceComparison:
                raise TypeError("reinference map values must be exact comparisons")
            frozen[key] = value
        object.__setattr__(self, "reinference_by_step", MappingProxyType(frozen))

    def reinference_document(self) -> dict[str, dict[str, object]]:
        """Return the exact own-fold value consumed by the GPU supervisor."""

        return {key: self.reinference_by_step[key].canonical_record() for key in _CHECKPOINT_KEYS}


@dataclass(frozen=True, slots=True)
class _ReleaseAuthority:
    readiness: TrainerReadinessReceipt
    release: ScoreReleaseReceipt


@dataclass(frozen=True, slots=True)
class _ReleasedTrainer:
    trainer: AuthenticatedTrainerBundle
    readiness: TrainerReadinessReceipt
    release: ScoreReleaseReceipt


def run_pilot_evaluator(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    node_local_score_path: str | os.PathLike[str],
    trainer_bundle_dir: str | os.PathLike[str],
    readiness_receipt_path: str | os.PathLike[str],
    score_release_path: str | os.PathLike[str],
    progress_dir: str | os.PathLike[str],
    trace_monotonic_origin_ns: int,
    supervisor_process_pid: int,
    output_dir: str | os.PathLike[str],
    outer_fold: int,
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> EvaluatorBundleResult:
    """Authenticate, evaluate, publish, and freshly reinfer one production fold."""

    _validate_fold_and_commit(outer_fold, expected_git_commit)
    _reject_sibling_score_authority()
    contract = load_pilot_execution_v1_contract(
        child_contract_path,
        parent_path=parent_contract_path,
    )
    contract.revalidate()
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
    # No score path is inspected or opened before the four-fold release token
    # is parsed and tied to this fold's exact readiness receipt.  Resume and
    # mark evaluator entry before the expensive repository and trainer-bundle
    # authentication that completes the released-trainer proof.
    release_authority = _open_release_authority(
        contract=contract,
        readiness_receipt_path=readiness_receipt_path,
        score_release_path=score_release_path,
        outer_fold=outer_fold,
        expected_git_commit=expected_git_commit,
    )
    progress = EvaluatorProgressWriter.resume_evaluator(
        _evaluator_progress_directory(
            progress_dir,
            output_dir=output_dir,
            outer_fold=outer_fold,
        ),
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
        git_commit=expected_git_commit,
        outer_fold=outer_fold,
        score_release_sha256=release_authority.release.sha256,
        expected_supervisor_process_pid=supervisor_process_pid,
        trace_monotonic_origin_ns=trace_monotonic_origin_ns,
    )
    progress.advance("release_authenticated")
    repository = build_repository_snapshot(
        repository_root,
        expected_commit=expected_git_commit,
    )
    released = _authenticate_released_trainer(
        contract=contract,
        trainer_bundle_dir=trainer_bundle_dir,
        release_authority=release_authority,
        repository=repository,
        outer_fold=outer_fold,
        expected_git_commit=expected_git_commit,
    )
    progress.advance("score_input_authentication")
    score_source = _node_local_score_source(node_local_score_path)
    score_payload = _read_regular_bytes(
        score_source,
        maximum_bytes=_MAX_SCORE_BYTES,
        label="node-local released score projection",
        required_mode=0o400,
    )
    fold = contract.fold(outer_fold)
    if hashlib.sha256(score_payload).hexdigest() != fold.score_sha256:
        raise ValueError("node-local score projection differs from its fold pin")
    ledger = load_contract_score_corruption_ledger(
        contract,
        outer_fold,
        score_source,
    )
    progress.advance("cuda_runtime_attestation")
    runtime = verify_r128_production_environment(contract, device="cuda:0")
    residual_logits = infer_all_checkpoint_slices(
        _progress_checkpoint_loader(
            _authenticated_checkpoint_loader(
                contract=contract,
                trainer_bundle=released.trainer.bundle,
                readiness=released.readiness,
                count_prior=released.trainer.count_prior,
                outer_fold=outer_fold,
            ),
            progress=progress,
            pass_name="archive",
        ),
        ledger,
        contract=contract,
    )
    progress.advance("scoring_archive_construction")
    archive, methods = _build_scoring_evidence(
        ledger=ledger,
        count_prior=released.trainer.count_prior,
        residual_logits=residual_logits,
        _progress=progress,
    )
    progress.advance("bundle_serialization")
    payloads, manifest = _evaluator_bundle_payloads(
        contract=contract,
        ledger=ledger,
        archive=archive,
        methods=methods,
        runtime=runtime,
        trainer_bundle_sha256=released.trainer.bundle.tree_sha256,
        count_prior_sha256=released.trainer.count_prior.sha256,
        child_contract_payload=child_payload,
        parent_contract_payload=parent_payload,
        code_sha256sums=repository.code_sha256sums,
        git_commit=repository.git_commit,
        outer_fold=outer_fold,
        readiness_receipt_sha256=released.readiness.sha256,
        score_release_sha256=released.release.sha256,
    )
    progress.advance("prepublication_input_revalidation")
    _revalidate_evaluator_inputs(
        contract=contract,
        child_contract_path=child_contract_path,
        parent_contract_path=parent_contract_path,
        child_payload=child_payload,
        parent_payload=parent_payload,
        score_source=score_source,
        score_payload=score_payload,
        trainer_bundle_dir=trainer_bundle_dir,
        trainer_tree_sha256=released.trainer.bundle.tree_sha256,
        readiness_receipt_path=readiness_receipt_path,
        readiness_bytes=released.readiness.canonical_bytes(),
        score_release_path=score_release_path,
        release_bytes=released.release.canonical_bytes(),
        repository_root=repository_root,
        repository_code_sha256=repository.code_sha256,
        expected_git_commit=expected_git_commit,
    )
    progress.advance("bundle_publication")
    published = publish_bundle(
        contract,
        bundle_kind="evaluator",
        output_dir=output_dir,
        payloads=payloads,
        manifest=manifest,
    )

    # Publication is the boundary: reopen both bundles, reconstruct the
    # archived NPZ, and use five new model instances for the second pass.
    progress.advance("published_bundle_verification")
    evaluator_bundle = verify_bundle(
        contract,
        bundle_kind="evaluator",
        root=published.root,
        expected_tree_sha256=published.tree_sha256,
        expected_tree_bytes=published.tree_bytes,
    )
    progress.advance("reinference_setup")
    released_reopened = _open_released_trainer(
        contract=contract,
        trainer_bundle_dir=trainer_bundle_dir,
        readiness_receipt_path=readiness_receipt_path,
        score_release_path=score_release_path,
        repository=repository,
        outer_fold=outer_fold,
        expected_git_commit=expected_git_commit,
    )
    archived_arrays = _load_published_score_archive(
        evaluator_bundle,
        ledger=ledger,
    )
    replay_logits = infer_all_checkpoint_slices(
        _progress_checkpoint_loader(
            _authenticated_checkpoint_loader(
                contract=contract,
                trainer_bundle=released_reopened.trainer.bundle,
                readiness=released_reopened.readiness,
                count_prior=released_reopened.trainer.count_prior,
                outer_fold=outer_fold,
            ),
            progress=progress,
            pass_name="reinference",
        ),
        ledger,
        contract=contract,
    )
    archived_logits = cast(NDArray[np.float32], archived_arrays["residual_logit"])
    progress.advance("reinference_comparison")
    comparisons = {
        f"{step:06d}": compare_archived_reinference(
            archived_logits[index],
            replay_logits[index],
        )
        for index, step in enumerate(CHECKPOINT_STEPS)
    }
    progress.advance("final_input_revalidation")
    _revalidate_evaluator_inputs(
        contract=contract,
        child_contract_path=child_contract_path,
        parent_contract_path=parent_contract_path,
        child_payload=child_payload,
        parent_payload=parent_payload,
        score_source=score_source,
        score_payload=score_payload,
        trainer_bundle_dir=trainer_bundle_dir,
        trainer_tree_sha256=released.trainer.bundle.tree_sha256,
        readiness_receipt_path=readiness_receipt_path,
        readiness_bytes=released.readiness.canonical_bytes(),
        score_release_path=score_release_path,
        release_bytes=released.release.canonical_bytes(),
        repository_root=repository_root,
        repository_code_sha256=repository.code_sha256,
        expected_git_commit=expected_git_commit,
    )
    progress.advance("final_bundle_verification")
    verify_bundle(
        contract,
        bundle_kind="evaluator",
        root=evaluator_bundle.root,
        expected_tree_sha256=evaluator_bundle.tree_sha256,
        expected_tree_bytes=evaluator_bundle.tree_bytes,
    )
    progress.advance("evaluation_result_ready")
    progress.seal()
    return EvaluatorBundleResult(
        bundle=evaluator_bundle,
        outer_fold=outer_fold,
        trainer_bundle_sha256=released.trainer.bundle.tree_sha256,
        count_prior_file_sha256=released.trainer.count_prior.sha256,
        checkpoint_digest_by_step=released.trainer.checkpoint_digest_by_step,
        reinference_by_step=comparisons,
    )


def _open_released_trainer(
    *,
    contract: NativeDiffusionV1PilotContract,
    trainer_bundle_dir: str | os.PathLike[str],
    readiness_receipt_path: str | os.PathLike[str],
    score_release_path: str | os.PathLike[str],
    repository: RepositorySnapshot,
    outer_fold: int,
    expected_git_commit: str,
) -> _ReleasedTrainer:
    release_authority = _open_release_authority(
        contract=contract,
        readiness_receipt_path=readiness_receipt_path,
        score_release_path=score_release_path,
        outer_fold=outer_fold,
        expected_git_commit=expected_git_commit,
    )
    return _authenticate_released_trainer(
        contract=contract,
        trainer_bundle_dir=trainer_bundle_dir,
        release_authority=release_authority,
        repository=repository,
        outer_fold=outer_fold,
        expected_git_commit=expected_git_commit,
    )


def _open_release_authority(
    *,
    contract: NativeDiffusionV1PilotContract,
    readiness_receipt_path: str | os.PathLike[str],
    score_release_path: str | os.PathLike[str],
    outer_fold: int,
    expected_git_commit: str,
) -> _ReleaseAuthority:
    readiness_bytes = _read_regular_bytes(
        readiness_receipt_path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="trainer readiness receipt",
        required_mode=0o444,
    )
    release_bytes = _read_regular_bytes(
        score_release_path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="score-release receipt",
        required_mode=0o444,
    )
    readiness = parse_trainer_readiness_receipt(readiness_bytes, contract=contract)
    release = parse_score_release_receipt(release_bytes, contract=contract)
    if (
        readiness.outer_fold != outer_fold
        or readiness.git_commit != expected_git_commit
        or readiness.fit_identity_sha256 != contract.fit_identity_sha256(outer_fold)
    ):
        raise ValueError("readiness receipt differs from the assigned fold or commit")
    if release.git_commit != expected_git_commit:
        raise ValueError("score-release receipt differs from the expected commit")
    fold_key = str(outer_fold)
    readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    if release.readiness_receipt_sha256_by_fold[fold_key] != readiness_sha256:
        raise ValueError("score release does not bind this fold's exact readiness bytes")
    return _ReleaseAuthority(readiness=readiness, release=release)


def _authenticate_released_trainer(
    *,
    contract: NativeDiffusionV1PilotContract,
    trainer_bundle_dir: str | os.PathLike[str],
    release_authority: _ReleaseAuthority,
    repository: RepositorySnapshot,
    outer_fold: int,
    expected_git_commit: str,
) -> _ReleasedTrainer:
    if type(release_authority) is not _ReleaseAuthority:
        raise TypeError("release authority must be an exact _ReleaseAuthority")
    readiness = release_authority.readiness
    release = release_authority.release
    trainer = authenticate_trainer_bundle(
        contract,
        trainer_bundle_dir,
        expected_outer_fold=outer_fold,
        expected_code_sha256sums=repository.code_sha256sums,
        expected_tree_sha256=readiness.trainer_bundle_sha256,
    )
    observation = trainer.make_readiness_observation(
        trusted_git_commit=expected_git_commit,
        node_name=readiness.node_name,
        device_uuid=readiness.device_uuid,
    )
    verify_trainer_readiness_receipt(
        readiness,
        contract=contract,
        observation=observation,
    )
    return _ReleasedTrainer(
        trainer=trainer,
        readiness=readiness,
        release=release,
    )


def _authenticated_checkpoint_loader(
    *,
    contract: NativeDiffusionV1PilotContract,
    trainer_bundle: BundleSnapshot,
    readiness: TrainerReadinessReceipt,
    count_prior: AuthenticatedCountPrior,
    outer_fold: int,
    _test_only_allow_cpu: bool = False,
):
    """Return a step-addressed loader that constructs one fresh model per call."""

    if type(_test_only_allow_cpu) is not bool:
        raise TypeError("_test_only_allow_cpu must be an exact bool")
    device = "cpu" if _test_only_allow_cpu else "cuda:0"
    metadata_paths = contract.table("checkpoints")["metadata_relative_paths"]
    if not isinstance(metadata_paths, tuple):
        raise TypeError("checkpoint metadata paths lost their frozen tuple")
    checkpoint_by_step = dict(
        zip(contract.checkpoint_steps, contract.checkpoint_paths, strict=True)
    )
    metadata_by_step = dict(zip(contract.checkpoint_steps, metadata_paths, strict=True))

    def load(step: int) -> tuple[int, R128Denoiser]:
        if type(step) is not int or step not in CHECKPOINT_STEPS:
            raise ValueError("checkpoint loader received a non-contract step")
        checkpoint_bytes = trainer_bundle.read_bytes(
            checkpoint_by_step[step],
            maximum_bytes=_MAX_CHECKPOINT_BYTES,
        )
        metadata_bytes = trainer_bundle.read_bytes(
            metadata_by_step[step],
            maximum_bytes=_MAX_METADATA_BYTES,
        )
        identity = trusted_checkpoint_identity_from_readiness(
            metadata_bytes,
            readiness,
            step,
        )
        if type(identity) is not CheckpointIdentity:
            raise TypeError("readiness adapter returned a non-checkpoint identity")
        model, _ = build_r128_model_from_contract(
            contract,
            outer_fold,
            device=device,
        )
        load_r128_safetensors_bytes(
            model,
            checkpoint_bytes,
            metadata_bytes=metadata_bytes,
            contract=contract,
            outer_fold=outer_fold,
            checkpoint_step=step,
            count_prior=count_prior,
            expected_identity=identity,
        )
        return step, model

    return load


def _progress_checkpoint_loader(
    loader: AuthenticatedCheckpointLoader,
    *,
    progress: EvaluatorProgressWriter,
    pass_name: str,
) -> AuthenticatedCheckpointLoader:
    """Wrap one loader with exact load/inference phase transitions.

    The inference marker is published only after the authenticated loader has
    returned and immediately before control returns to ``infer_all_checkpoint_slices``.
    A subsequent load marker therefore proves that the preceding checkpoint's
    inference returned successfully.
    """

    if not callable(loader):
        raise TypeError("checkpoint progress wrapper requires a callable loader")
    if type(progress) is not EvaluatorProgressWriter:
        raise TypeError("checkpoint progress wrapper requires an exact progress writer")
    if pass_name not in {"archive", "reinference"}:
        raise ValueError("checkpoint progress pass must be archive or reinference")

    def load(step: int) -> LoadedCheckpoint:
        if type(step) is not int or step not in CHECKPOINT_STEPS:
            raise ValueError("checkpoint progress wrapper received a non-contract step")
        progress.advance(f"{pass_name}_checkpoint_{step:06d}_load")
        loaded = loader(step)
        progress.advance(f"{pass_name}_checkpoint_{step:06d}_inference")
        return loaded

    return load


def _build_scoring_evidence(
    *,
    ledger: ScoreCorruptionLedger,
    count_prior: AuthenticatedCountPrior,
    residual_logits: NDArray[np.float32],
    _progress: EvaluatorProgressWriter | None = None,
) -> tuple[ScoringArchive, tuple[FoldMethodMetrics, ...]]:
    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    ledger.revalidate()
    if type(count_prior) is not AuthenticatedCountPrior:
        raise TypeError("count_prior must be an exact AuthenticatedCountPrior")
    decoded = count_prior.revalidate()
    offsets, positions, targets, count_log = gather_count_log_probability(
        ledger,
        decoded.log_relative_position_probability,
    )
    archive = build_scoring_archive(
        ledger,
        count_prior=count_prior,
        case_id=ledger.arrays()["case_id"],
        case_offsets=offsets,
        position=positions,
        target_token=targets,
        count_log_probability=count_log,
        checkpoint_step=np.asarray(CHECKPOINT_STEPS, dtype="<u2"),
        residual_logit=residual_logits,
    )
    if _progress is not None:
        if type(_progress) is not EvaluatorProgressWriter:
            raise TypeError("scoring progress must be an exact evaluator progress writer")
        _progress.advance("fold_method_scoring")
    methods = score_all_fold_methods(archive)
    return archive, methods


def _evaluator_bundle_payloads(
    *,
    contract: NativeDiffusionV1PilotContract,
    ledger: ScoreCorruptionLedger,
    archive: ScoringArchive,
    methods: tuple[FoldMethodMetrics, ...],
    runtime: Mapping[str, object],
    trainer_bundle_sha256: str,
    count_prior_sha256: str,
    child_contract_payload: bytes,
    parent_contract_payload: bytes,
    code_sha256sums: bytes,
    git_commit: str,
    outer_fold: int,
    readiness_receipt_sha256: str,
    score_release_sha256: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    contract.revalidate()
    ledger.revalidate()
    archive.revalidate()
    _sha256(readiness_receipt_sha256, label="readiness receipt SHA-256")
    _sha256(score_release_sha256, label="score-release SHA-256")
    fold = contract.fold(outer_fold)
    metrics = fold_metrics_document(
        methods,
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
    )
    metrics_bytes = canonical_json_bytes(metrics)
    corruption_bytes = ledger.npz_bytes()
    archive_bytes = archive.npz_bytes()
    environment = _environment_document(runtime)
    rng = _rng_document(contract)
    payloads = {
        "CODE_SHA256SUMS": code_sha256sums,
        "FROZEN_INPUT_SHA256SUMS": sha256sums_bytes(
            {
                "pilot_execution_v1.toml": contract.config_sha256,
                "checkpoint-ready.json": readiness_receipt_sha256,
                "count_prior.npz": count_prior_sha256,
                "score.jsonl": fold.score_sha256,
                "score-release.json": score_release_sha256,
                "trainer_bundle.tree.json": trainer_bundle_sha256,
                "unconditional_v1.toml": contract.parent_config_sha256,
            }
        ),
        "pilot_execution_v1.toml": child_contract_payload,
        "unconditional_v1.toml": parent_contract_payload,
        "environment.json": canonical_json_bytes(environment),
        "rng.json": canonical_json_bytes(rng),
        "trainer_bundle.sha256": contract.sha256_sidecar_bytes(trainer_bundle_sha256),
        "count_prior.sha256": contract.sha256_sidecar_bytes(count_prior_sha256),
        "score_corruptions.npz": corruption_bytes,
        "score_residual_logits.npz": archive_bytes,
        "fold_metrics.json": metrics_bytes,
    }
    method_order = [
        method.method
        if method.checkpoint_step is None
        else f"{method.method}-{method.checkpoint_step:06d}"
        for method in methods
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": contract.document["artifact"],
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": git_commit,
        "outer_fold": outer_fold,
        "fit_identity_sha256": fold.fit_identity_sha256,
        "trainer_bundle_sha256": trainer_bundle_sha256,
        "projection": {
            "accepted_bundle_tree_sha256": contract.projection_tree_sha256,
            "accepted_top_manifest_sha256": contract.projection_top_sha256,
            "score_sha256": fold.score_sha256,
            "score_rows": fold.score_rows,
            "score_homology_components": fold.score_homology_components,
            "score_union_components": fold.score_union_components,
            "score_cases": fold.score_cases,
            "score_selected_tokens": fold.score_selected_tokens,
            "score_selected_tokens_by_timestep_bin": list(
                fold.score_selected_tokens_by_timestep_bin
            ),
        },
        "rng": rng,
        "count_prior": {
            "file_sha256": count_prior_sha256,
            "source": "exact_reopened_trainer_bundle_bytes",
            "recomputed_from_score": False,
        },
        "score": {
            "method_order": method_order,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "corruption_file_sha256": hashlib.sha256(corruption_bytes).hexdigest(),
            "residual_logits_file_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "fold_metrics_file_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        },
        "status": {
            "score_release_bound_before_score_open": True,
            "score_release_sha256": score_release_sha256,
            "readiness_receipt_sha256": readiness_receipt_sha256,
            "trainer_bundle_reopened": True,
            "all_five_checkpoints_authenticated": True,
            "fold_metrics_complete": True,
            "post_publication_reinference_required": True,
            "reinference_record_location": "path_free_supervisor_result",
        },
    }
    return payloads, manifest


def _load_published_score_archive(
    bundle: BundleSnapshot,
    *,
    ledger: ScoreCorruptionLedger,
) -> dict[str, NDArray[np.generic]]:
    ledger_arrays = ledger.arrays()
    selected_count = int(np.sum(ledger_arrays["mask_count"], dtype=np.uint64))
    snapshot = bundle.file("score_residual_logits.npz")
    payload = bundle.read_bytes("score_residual_logits.npz")
    return load_deterministic_npz_bytes(
        payload,
        expected_sha256=snapshot.sha256,
        schema=residual_logits_npz_schema(
            len(ledger_arrays["case_id"]),
            selected_count,
        ),
    )


def _environment_document(runtime: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(runtime, Mapping) or not runtime:
        raise TypeError("runtime must be a non-empty authenticated mapping")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_evaluator_environment",
        "runtime": dict(runtime),
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pytorch_allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
            "deterministic_algorithms": True,
            "math_sdpa_only": True,
            "tf32": False,
            "amp": False,
        },
    }


def _rng_document(contract: NativeDiffusionV1PilotContract) -> dict[str, object]:
    rng = contract.table("rng")
    evaluation = contract.table("evaluation")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_evaluation_rng",
        "derivation": rng["derivation"],
        "evaluation_root_seed": rng["evaluation_root_seed"],
        "validation_namespace": rng["validation_namespace"],
        "validation_key": list(rng["validation_key"]),
        "validation_rng": rng["validation_rng"],
        "levels": evaluation["levels"],
        "replicates_per_sequence_level": evaluation["replicates_per_sequence_level"],
    }


def _revalidate_evaluator_inputs(
    *,
    contract: NativeDiffusionV1PilotContract,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    child_payload: bytes,
    parent_payload: bytes,
    score_source: Path,
    score_payload: bytes,
    trainer_bundle_dir: str | os.PathLike[str],
    trainer_tree_sha256: str,
    readiness_receipt_path: str | os.PathLike[str],
    readiness_bytes: bytes,
    score_release_path: str | os.PathLike[str],
    release_bytes: bytes,
    repository_root: str | os.PathLike[str],
    repository_code_sha256: str,
    expected_git_commit: str,
) -> None:
    contract.revalidate()
    _reject_sibling_score_authority()
    if _node_local_score_source(score_source) != score_source:
        raise RuntimeError("node-local score projection identity changed")
    values = (
        (
            child_contract_path,
            child_payload,
            _MAX_CONTRACT_BYTES,
            None,
            "pilot child contract",
        ),
        (
            parent_contract_path,
            parent_payload,
            _MAX_CONTRACT_BYTES,
            None,
            "pilot parent contract",
        ),
        (score_source, score_payload, _MAX_SCORE_BYTES, 0o400, "node-local score projection"),
        (
            readiness_receipt_path,
            readiness_bytes,
            _MAX_RECEIPT_BYTES,
            0o444,
            "trainer readiness receipt",
        ),
        (
            score_release_path,
            release_bytes,
            _MAX_RECEIPT_BYTES,
            0o444,
            "score-release receipt",
        ),
    )
    for path, expected, maximum, mode, label in values:
        observed = _read_regular_bytes(
            path,
            maximum_bytes=maximum,
            label=label,
            required_mode=mode,
        )
        if observed != expected:
            raise RuntimeError(f"{label} changed during evaluation")
    verify_bundle(
        contract,
        bundle_kind="trainer",
        root=trainer_bundle_dir,
        expected_tree_sha256=trainer_tree_sha256,
    )
    repository = build_repository_snapshot(
        repository_root,
        expected_commit=expected_git_commit,
    )
    if repository.code_sha256 != repository_code_sha256:
        raise RuntimeError("repository changed during evaluation")


def _node_local_score_source(path: str | os.PathLike[str]) -> Path:
    source = Path(os.path.abspath(os.fspath(path)))
    temporary_raw = os.environ.get("TMPDIR")
    if not temporary_raw:
        raise ValueError("TMPDIR must identify the private node-local evaluator directory")
    temporary = Path(os.path.abspath(temporary_raw))
    temporary_stat = os.lstat(temporary)
    parent_stat = os.lstat(source.parent)
    source_stat = os.lstat(source)
    if stat.S_ISLNK(temporary_stat.st_mode) or not stat.S_ISDIR(temporary_stat.st_mode):
        raise ValueError("TMPDIR must be a real directory")
    try:
        source.relative_to(temporary)
    except ValueError as error:
        raise ValueError("score input must be staged beneath TMPDIR") from error
    if source.name != "score.jsonl" or tuple(child.name for child in source.parent.iterdir()) != (
        "score.jsonl",
    ):
        raise ValueError("evaluator input directory must contain exactly one score.jsonl")
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or stat.S_ISLNK(source_stat.st_mode)
        or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_nlink != 1
        or stat.S_IMODE(source_stat.st_mode) != 0o400
    ):
        raise ValueError("node-local score input is not private and sealed")
    return source


def _evaluator_progress_directory(
    progress_dir: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str],
    outer_fold: int,
) -> Path:
    """Bind operational progress to this evaluator's own run and fold."""

    if type(outer_fold) is not int or outer_fold not in (0, 1, 2, 3):
        raise ValueError("outer_fold must be an exact integer in 0..3")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    progress = Path(os.path.abspath(os.fspath(progress_dir)))
    if output.name != str(outer_fold) or output.parent.name != "evaluators":
        raise ValueError("evaluator output directory does not bind its own outer fold")
    expected = output.parent.parent / "control" / "evaluator-progress" / str(outer_fold)
    if progress != expected:
        raise ValueError("evaluator progress directory does not bind its own output and fold")
    return progress


def _reject_sibling_score_authority() -> None:
    present = sorted(
        name
        for name in os.environ
        if name in _FORBIDDEN_SCORE_ENVIRONMENT or name.startswith("AMP_PILOT_SCORE_")
    )
    if present:
        raise RuntimeError("evaluator environment contains sibling score authority")


def _validate_fold_and_commit(outer_fold: int, expected_git_commit: str) -> None:
    if type(outer_fold) is not int or outer_fold not in (0, 1, 2, 3):
        raise ValueError("outer_fold must be an exact integer in 0..3")
    if type(expected_git_commit) is not str or _GIT_RE.fullmatch(expected_git_commit) is None:
        raise ValueError("expected_git_commit must be a lowercase 40-character Git ID")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-contract", required=True)
    parser.add_argument("--parent-contract", required=True)
    parser.add_argument("--score-jsonl", required=True)
    parser.add_argument("--trainer-bundle", required=True)
    parser.add_argument("--readiness-receipt", required=True)
    parser.add_argument("--score-release", required=True)
    parser.add_argument("--progress-dir", required=True)
    parser.add_argument("--trace-monotonic-origin-ns", required=True, type=int)
    parser.add_argument("--supervisor-process-pid", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pilot_evaluator(
        child_contract_path=args.child_contract,
        parent_contract_path=args.parent_contract,
        node_local_score_path=args.score_jsonl,
        trainer_bundle_dir=args.trainer_bundle,
        readiness_receipt_path=args.readiness_receipt,
        score_release_path=args.score_release,
        progress_dir=args.progress_dir,
        trace_monotonic_origin_ns=args.trace_monotonic_origin_ns,
        supervisor_process_pid=args.supervisor_process_pid,
        output_dir=args.output_dir,
        outer_fold=args.outer_fold,
        repository_root=args.repository_root,
        expected_git_commit=args.expected_git_commit,
    )
    print(
        canonical_json_bytes(
            {
                "count_prior_file_sha256": result.count_prior_file_sha256,
                "checkpoint_digest_by_step": {
                    key: result.checkpoint_digest_by_step[key].document()
                    for key in _CHECKPOINT_KEYS
                },
                "evaluator_bundle_sha256": result.bundle.tree_sha256,
                "outer_fold": result.outer_fold,
                "reinference_by_step": result.reinference_document(),
                "trainer_bundle_sha256": result.trainer_bundle_sha256,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

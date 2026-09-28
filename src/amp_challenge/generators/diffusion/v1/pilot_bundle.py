"""Neutral authentication and publication for native-diffusion pilot evidence.

This module is the sealed semantic boundary between the four single-fold GPU
evaluators and the independent CPU verifier.  It does not import a trainer,
evaluator, producer, or model and never performs neural inference.  Scientific
authorization remains with :class:`VerifiedPilotEvidence`; this layer only
publishes a pilot bundle after reproducing its numerical decision and binding
every upstream digest.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    BundleSnapshot,
    RepositorySnapshot,
    build_bundle_manifest,
    canonical_json_bytes,
    parse_canonical_json,
    parse_sha256_sidecar,
    parse_sha256sums,
    publish_bundle,
    sha256sums_bytes,
    verify_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    CONFIG_SHA256,
    PARENT_CONFIG_SHA256,
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import AuthenticatedCountPrior
from amp_challenge.generators.diffusion.v1.pilot_records import (
    bootstrap_record,
    checkpoint_selection_record,
    decision_document,
    fold_metrics_document,
    gate_decision_record,
    pilot_metrics_document,
)
from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    ALPHABET,
    CHECKPOINT_STEPS,
    LEVELS,
    EqualFoldMetrics,
    FoldMethodMetrics,
    PilotEvaluation,
    ScoreCorruptionLedger,
    ScoreRow,
    ScoringArchive,
    VerifiedPilotEvidence,
    build_scoring_archive,
    corruption_npz_schema,
    evaluate_verified_pilot_gate,
    load_deterministic_npz_bytes,
    residual_logits_npz_schema,
)

_AUTHENTICATED_EVALUATOR_CAPABILITY = object()
_AUTHENTICATED_PILOT_CAPABILITY = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_FOLDS = (0, 1, 2, 3)
_FOLD_KEYS = ("0", "1", "2", "3")
_METHOD_ORDER = (
    "C0",
    "C0T",
    "R128-000250",
    "R128-000500",
    "R128-001000",
    "R128-002000",
    "R128-004000",
)
_EVALUATOR_MANIFEST_FIELDS = (
    "schema_version",
    "artifact",
    "child_contract_sha256",
    "parent_contract_sha256",
    "git_commit",
    "outer_fold",
    "fit_identity_sha256",
    "trainer_bundle_sha256",
    "projection",
    "rng",
    "count_prior",
    "score",
    "status",
    "artifacts",
)
_EVALUATOR_RUNTIME_FIELDS = (
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

# These logical names are a frozen schema, not runtime paths.  Every value is
# the physical or canonical-map SHA-256 named by the suffix.
EVALUATOR_FROZEN_INPUT_PATHS = (
    "checkpoint-ready.json",
    "count_prior.npz",
    "pilot_execution_v1.toml",
    "score-release.json",
    "score.jsonl",
    "trainer_bundle.tree.json",
    "unconditional_v1.toml",
)
PILOT_FROZEN_INPUT_PATHS = (
    "checkpoint_digest_by_fold_and_step.json",
    "count_prior_sha256_by_fold.json",
    "cpu_reconstruction.json",
    "evaluator_bundle_sha256_by_fold.json",
    "fold_bundles.map.json",
    "pilot_execution_v1.toml",
    "producer_gpu_reinference.json",
    "score-release.json",
    "trainer_bundle_sha256_by_fold.json",
    "unconditional_v1.toml",
)
PILOT_FROZEN_EVIDENCE_BINDING_BY_PATH = MappingProxyType(
    {
        "checkpoint_digest_by_fold_and_step.json": "checkpoint_digest_map_sha256",
        "count_prior_sha256_by_fold.json": "count_prior_digest_map_sha256",
        "cpu_reconstruction.json": "cpu_reconstruction_sha256",
        "evaluator_bundle_sha256_by_fold.json": "evaluator_bundle_digest_map_sha256",
        "producer_gpu_reinference.json": "producer_reinference_sha256",
        "score-release.json": "release_receipt_sha256",
        "trainer_bundle_sha256_by_fold.json": "trainer_bundle_digest_map_sha256",
    }
)

_MAX_JSON_BYTES = 64 << 20
_MAX_CODE_MANIFEST_BYTES = 64 << 20
_MAX_NPZ_BYTES = 2 << 30


@dataclass(frozen=True, slots=True)
class AuthenticatedEvaluatorBundle:
    """Path-free identity minted from a sealed, reconstructed evaluator tree."""

    bundle: BundleSnapshot
    outer_fold: int
    git_commit: str
    trainer_bundle_sha256: str
    count_prior_sha256: str
    readiness_receipt_sha256: str
    score_release_sha256: str
    score_corruptions_sha256: str
    score_residual_logits_sha256: str
    fold_metrics_sha256: str
    ledger: ScoreCorruptionLedger = field(repr=False, compare=False)
    scoring_archive: ScoringArchive = field(repr=False, compare=False)
    methods: tuple[FoldMethodMetrics, ...] = field(repr=False, compare=False)
    identity_sha256: str = field(init=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _AUTHENTICATED_EVALUATOR_CAPABILITY:
            raise RuntimeError("evaluator identity requires neutral bundle authentication")
        _validate_authenticated_evaluator_contents(self)
        record = _evaluator_identity_record(self)
        object.__setattr__(
            self,
            "identity_sha256",
            hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )
        object.__setattr__(self, "_capability", None)

    def revalidate(self) -> None:
        """Reject relabeling while retaining no authority to mint a new identity."""

        if self._capability is not None:
            raise ValueError("evaluator identity capability state changed")
        _validate_authenticated_evaluator_contents(self)
        expected = hashlib.sha256(
            canonical_json_bytes(_evaluator_identity_record(self))
        ).hexdigest()
        if _sha256(self.identity_sha256, label="evaluator identity SHA-256") != expected:
            raise ValueError("evaluator identity changed after authentication")


@dataclass(frozen=True, slots=True)
class AuthenticatedPilotBundle:
    """Identity of a scientific pilot bundle reopened against reconstructed inputs."""

    bundle: BundleSnapshot
    git_commit: str
    fold_bundle_sha256_by_fold: Mapping[str, Mapping[str, str]]
    fold_bundle_map_sha256: str
    pilot_bootstrap_sha256: str
    pilot_metrics_sha256: str
    decision_sha256: str
    decision_status: str
    verified_evidence_sha256: str
    identity_sha256: str = field(init=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _AUTHENTICATED_PILOT_CAPABILITY:
            raise RuntimeError("pilot identity requires neutral bundle authentication")
        frozen = _freeze_fold_bundle_map(self.fold_bundle_sha256_by_fold)
        object.__setattr__(self, "fold_bundle_sha256_by_fold", frozen)
        record = _pilot_identity_record(self)
        object.__setattr__(
            self,
            "identity_sha256",
            hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )
        object.__setattr__(self, "_capability", None)

    def revalidate(self) -> None:
        """Reject mutation or semantic relabeling of this authenticated identity."""

        if self._capability is not None:
            raise ValueError("pilot identity capability state changed")
        if type(self.fold_bundle_sha256_by_fold) is not MappingProxyType or any(
            type(item) is not MappingProxyType for item in self.fold_bundle_sha256_by_fold.values()
        ):
            raise TypeError("authenticated pilot fold map lost immutable storage")
        _freeze_fold_bundle_map(self.fold_bundle_sha256_by_fold)
        expected = hashlib.sha256(canonical_json_bytes(_pilot_identity_record(self))).hexdigest()
        if _sha256(self.identity_sha256, label="pilot identity SHA-256") != expected:
            raise ValueError("pilot identity changed after authentication")


@dataclass(frozen=True, slots=True)
class _ExpectedPilotBundle:
    payloads: Mapping[str, bytes]
    manifest: Mapping[str, object]
    complete_manifest: Mapping[str, object]
    fold_bundle_map: Mapping[str, Mapping[str, str]]
    fold_bundle_map_sha256: str


def authenticate_evaluator_bundle(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    expected_outer_fold: int,
    expected_git_commit: str,
    repository: RepositorySnapshot,
    expected_trainer_bundle_sha256: str,
    expected_count_prior_sha256: str,
    count_prior: AuthenticatedCountPrior,
    expected_readiness_receipt_sha256: str,
    expected_score_release_sha256: str,
    reconstructed_methods: Sequence[FoldMethodMetrics],
    expected_tree_sha256: str | None = None,
    expected_score_corruptions_bytes: bytes | None = None,
    expected_score_corruptions_sha256: str | None = None,
    expected_score_residual_logits_bytes: bytes | None = None,
    expected_score_residual_logits_sha256: str | None = None,
) -> AuthenticatedEvaluatorBundle:
    """Authenticate one evaluator without importing or trusting its producer.

    The deterministic archives are always schema-checked.  An independent
    verifier may additionally require byte equality or a separately obtained
    physical digest for either archive.
    """

    _validate_contract(contract)
    fold = contract.fold(_outer_fold(expected_outer_fold))
    commit = _validate_repository(repository, expected_git_commit)
    trainer_sha = _sha256(
        expected_trainer_bundle_sha256,
        label="expected trainer bundle SHA-256",
    )
    count_prior_sha = _sha256(
        expected_count_prior_sha256,
        label="expected count-prior SHA-256",
    )
    if type(count_prior) is not AuthenticatedCountPrior:
        raise TypeError("count_prior must be an exact AuthenticatedCountPrior")
    count_prior.revalidate()
    if count_prior.sha256 != count_prior_sha:
        raise ValueError("authenticated count prior differs from its expected SHA-256")
    readiness_sha = _sha256(
        expected_readiness_receipt_sha256,
        label="expected readiness-receipt SHA-256",
    )
    release_sha = _sha256(
        expected_score_release_sha256,
        label="expected score-release SHA-256",
    )
    snapshot = verify_bundle(
        contract,
        bundle_kind="evaluator",
        root=root,
        expected_tree_sha256=(
            None
            if expected_tree_sha256 is None
            else _sha256(expected_tree_sha256, label="expected evaluator tree SHA-256")
        ),
    )
    if (
        snapshot.read_bytes(
            "CODE_SHA256SUMS",
            maximum_bytes=_MAX_CODE_MANIFEST_BYTES,
        )
        != repository.code_sha256sums
    ):
        raise ValueError("evaluator CODE_SHA256SUMS differs from the repository snapshot")
    frozen = parse_sha256sums(
        snapshot.read_bytes(
            "FROZEN_INPUT_SHA256SUMS",
            maximum_bytes=_MAX_CODE_MANIFEST_BYTES,
        ),
        label="evaluator FROZEN_INPUT_SHA256SUMS",
    )
    expected_frozen = {
        "checkpoint-ready.json": readiness_sha,
        "count_prior.npz": count_prior_sha,
        "pilot_execution_v1.toml": contract.config_sha256,
        "score-release.json": release_sha,
        "score.jsonl": fold.score_sha256,
        "trainer_bundle.tree.json": trainer_sha,
        "unconditional_v1.toml": contract.parent_config_sha256,
    }
    if tuple(frozen) != EVALUATOR_FROZEN_INPUT_PATHS or frozen != expected_frozen:
        raise ValueError("evaluator frozen-input manifest differs from the exact gate chain")
    observed_trainer = parse_sha256_sidecar(
        snapshot.read_bytes("trainer_bundle.sha256", maximum_bytes=65),
        label="evaluator trainer_bundle.sha256",
    )
    observed_count_prior = parse_sha256_sidecar(
        snapshot.read_bytes("count_prior.sha256", maximum_bytes=65),
        label="evaluator count_prior.sha256",
    )
    if observed_trainer != trainer_sha or observed_count_prior != count_prior_sha:
        raise ValueError("evaluator sidecars differ from authenticated upstream identities")

    corruption_payload = snapshot.read_bytes(
        "score_corruptions.npz",
        maximum_bytes=_MAX_NPZ_BYTES,
    )
    residual_payload = snapshot.read_bytes(
        "score_residual_logits.npz",
        maximum_bytes=_MAX_NPZ_BYTES,
    )
    corruption_sha = _validate_expected_payload(
        corruption_payload,
        expected_bytes=expected_score_corruptions_bytes,
        expected_sha256=expected_score_corruptions_sha256,
        label="score-corruption archive",
    )
    residual_sha = _validate_expected_payload(
        residual_payload,
        expected_bytes=expected_score_residual_logits_bytes,
        expected_sha256=expected_score_residual_logits_sha256,
        label="score-residual-logit archive",
    )
    corruption_arrays = load_deterministic_npz_bytes(
        corruption_payload,
        expected_sha256=corruption_sha,
        schema=corruption_npz_schema(fold.score_rows),
    )
    residual_arrays = load_deterministic_npz_bytes(
        residual_payload,
        expected_sha256=residual_sha,
        schema=residual_logits_npz_schema(
            fold.score_cases,
            fold.score_selected_tokens,
        ),
    )
    ledger = _authenticate_corruption_arrays(
        contract,
        outer_fold=expected_outer_fold,
        arrays=corruption_arrays,
        expected_payload=corruption_payload,
    )
    scoring_archive = build_scoring_archive(
        ledger=ledger,
        count_prior=count_prior,
        case_id=cast(NDArray[np.bytes_], residual_arrays["case_id"]),
        case_offsets=cast(NDArray[np.uint64], residual_arrays["case_offsets"]),
        position=cast(NDArray[np.uint8], residual_arrays["position"]),
        target_token=cast(NDArray[np.uint8], residual_arrays["target_token"]),
        count_log_probability=cast(
            NDArray[np.float64],
            residual_arrays["count_log_probability"],
        ),
        checkpoint_step=cast(NDArray[np.uint16], residual_arrays["checkpoint_step"]),
        residual_logit=cast(NDArray[np.float32], residual_arrays["residual_logit"]),
    )
    if scoring_archive.npz_bytes() != residual_payload:
        raise ValueError("residual archive differs after semantic reconstruction")

    methods = _ordered_fold_methods(reconstructed_methods, expected_outer_fold)
    expected_metrics = canonical_json_bytes(
        fold_metrics_document(
            methods,
            child_contract_sha256=contract.config_sha256,
            parent_contract_sha256=contract.parent_config_sha256,
        )
    )
    observed_metrics = snapshot.read_bytes(
        "fold_metrics.json",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if observed_metrics != expected_metrics:
        raise ValueError("fold_metrics.json differs from independent reconstruction")
    _validate_method_rows(methods, ledger)
    metrics_sha = hashlib.sha256(observed_metrics).hexdigest()

    _validate_evaluator_environment(contract, snapshot)
    expected_rng = _evaluator_rng_document(contract)
    observed_rng = parse_canonical_json(
        snapshot.read_bytes("rng.json", maximum_bytes=_MAX_JSON_BYTES),
        label="evaluator rng.json",
    )
    _require_strict_equal(observed_rng, expected_rng, label="evaluator RNG binding")

    manifest = _json_object(
        parse_canonical_json(
            snapshot.read_bytes("manifest.json", maximum_bytes=_MAX_JSON_BYTES),
            label="evaluator manifest.json",
        ),
        _EVALUATOR_MANIFEST_FIELDS,
        label="evaluator manifest.json",
    )
    expected_manifest = _expected_evaluator_manifest(
        contract,
        outer_fold=expected_outer_fold,
        git_commit=commit,
        trainer_bundle_sha256=trainer_sha,
        count_prior_sha256=count_prior_sha,
        corruption_sha256=corruption_sha,
        residual_sha256=residual_sha,
        fold_metrics_sha256=metrics_sha,
        readiness_receipt_sha256=readiness_sha,
        score_release_sha256=release_sha,
        artifacts=cast(Mapping[str, object], manifest["artifacts"]),
    )
    _require_strict_equal(manifest, expected_manifest, label="evaluator manifest binding")
    return AuthenticatedEvaluatorBundle(
        bundle=snapshot,
        outer_fold=expected_outer_fold,
        git_commit=commit,
        trainer_bundle_sha256=trainer_sha,
        count_prior_sha256=count_prior_sha,
        readiness_receipt_sha256=readiness_sha,
        score_release_sha256=release_sha,
        score_corruptions_sha256=corruption_sha,
        score_residual_logits_sha256=residual_sha,
        fold_metrics_sha256=metrics_sha,
        ledger=ledger,
        scoring_archive=scoring_archive,
        methods=methods,
        _capability=_AUTHENTICATED_EVALUATOR_CAPABILITY,
    )


def publish_pilot_bundle(
    contract: NativeDiffusionV1PilotContract,
    *,
    output_dir: str | os.PathLike[str],
    evaluator_bundles: Sequence[AuthenticatedEvaluatorBundle],
    fold_bundle_sha256_by_fold: Mapping[str, Mapping[str, str]],
    equal_fold_metrics: Sequence[EqualFoldMetrics],
    evaluation: PilotEvaluation,
    verified_evidence: VerifiedPilotEvidence,
    child_contract_payload: bytes,
    parent_contract_payload: bytes,
    repository: RepositorySnapshot,
    expected_git_commit: str,
) -> AuthenticatedPilotBundle:
    """Publish a new exact pilot bundle after reproducing its authorized gate."""

    expected = _build_expected_pilot_bundle(
        contract,
        evaluator_bundles=evaluator_bundles,
        fold_bundle_sha256_by_fold=fold_bundle_sha256_by_fold,
        equal_fold_metrics=equal_fold_metrics,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
        child_contract_payload=child_contract_payload,
        parent_contract_payload=parent_contract_payload,
        repository=repository,
        expected_git_commit=expected_git_commit,
    )
    published = publish_bundle(
        contract,
        bundle_kind="pilot",
        output_dir=output_dir,
        payloads=expected.payloads,
        manifest=expected.manifest,
    )
    return _authenticate_expected_pilot_bundle(
        contract,
        published.root,
        expected=expected,
        expected_tree_sha256=published.tree_sha256,
        expected_git_commit=expected_git_commit,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
    )


def authenticate_pilot_bundle(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    evaluator_bundles: Sequence[AuthenticatedEvaluatorBundle],
    fold_bundle_sha256_by_fold: Mapping[str, Mapping[str, str]],
    equal_fold_metrics: Sequence[EqualFoldMetrics],
    evaluation: PilotEvaluation,
    verified_evidence: VerifiedPilotEvidence,
    child_contract_payload: bytes,
    parent_contract_payload: bytes,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    expected_tree_sha256: str | None = None,
) -> AuthenticatedPilotBundle:
    """Reopen a pilot against independently reconstructed typed evidence."""

    expected = _build_expected_pilot_bundle(
        contract,
        evaluator_bundles=evaluator_bundles,
        fold_bundle_sha256_by_fold=fold_bundle_sha256_by_fold,
        equal_fold_metrics=equal_fold_metrics,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
        child_contract_payload=child_contract_payload,
        parent_contract_payload=parent_contract_payload,
        repository=repository,
        expected_git_commit=expected_git_commit,
    )
    return _authenticate_expected_pilot_bundle(
        contract,
        root,
        expected=expected,
        expected_tree_sha256=expected_tree_sha256,
        expected_git_commit=expected_git_commit,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
    )


def _build_expected_pilot_bundle(
    contract: NativeDiffusionV1PilotContract,
    *,
    evaluator_bundles: Sequence[AuthenticatedEvaluatorBundle],
    fold_bundle_sha256_by_fold: Mapping[str, Mapping[str, str]],
    equal_fold_metrics: Sequence[EqualFoldMetrics],
    evaluation: PilotEvaluation,
    verified_evidence: VerifiedPilotEvidence,
    child_contract_payload: bytes,
    parent_contract_payload: bytes,
    repository: RepositorySnapshot,
    expected_git_commit: str,
) -> _ExpectedPilotBundle:
    _validate_contract(contract)
    commit = _validate_repository(repository, expected_git_commit)
    child_payload = _contract_payload(
        child_contract_payload,
        expected_sha256=contract.config_sha256,
        label="child contract",
    )
    parent_payload = _contract_payload(
        parent_contract_payload,
        expected_sha256=contract.parent_config_sha256,
        label="parent contract",
    )
    identities = _ordered_evaluator_bundles(evaluator_bundles, commit)
    metrics = _ordered_equal_fold_metrics(equal_fold_metrics)
    if type(verified_evidence) is not VerifiedPilotEvidence:
        raise TypeError("verified_evidence must be an exact VerifiedPilotEvidence")
    verified_evidence.revalidate()
    if (
        verified_evidence.child_contract_sha256 != contract.config_sha256
        or verified_evidence.parent_contract_sha256 != contract.parent_config_sha256
        or verified_evidence.git_commit != commit
    ):
        raise ValueError("verified evidence differs from the contract or Git identity")

    derived_fold_map = {
        str(identity.outer_fold): {
            "trainer_bundle_sha256": identity.trainer_bundle_sha256,
            "evaluator_bundle_sha256": identity.bundle.tree_sha256,
        }
        for identity in identities
    }
    fold_map = _freeze_fold_bundle_map(fold_bundle_sha256_by_fold)
    if canonical_json_bytes(fold_map) != canonical_json_bytes(derived_fold_map):
        raise ValueError("supplied fold-bundle map differs from authenticated identities")
    fold_map_bytes = contract.fold_bundle_digest_map_bytes(fold_map)
    fold_map_sha = hashlib.sha256(fold_map_bytes).hexdigest()
    trainer_map = {
        str(identity.outer_fold): identity.trainer_bundle_sha256 for identity in identities
    }
    evaluator_map = {
        str(identity.outer_fold): identity.bundle.tree_sha256 for identity in identities
    }
    count_prior_map = {
        str(identity.outer_fold): identity.count_prior_sha256 for identity in identities
    }
    derived_bindings = {
        "trainer_bundle_digest_map_sha256": _flat_fold_map_sha256(contract, trainer_map),
        "evaluator_bundle_digest_map_sha256": _flat_fold_map_sha256(contract, evaluator_map),
        "count_prior_digest_map_sha256": _flat_fold_map_sha256(contract, count_prior_map),
    }
    for name, expected in derived_bindings.items():
        if verified_evidence.bindings[name] != expected:
            raise ValueError(f"verified evidence {name} differs from evaluator identities")
    if len({identity.score_release_sha256 for identity in identities}) != 1:
        raise ValueError("evaluator identities do not share one score-release receipt")
    if (
        next(iter({identity.score_release_sha256 for identity in identities}))
        != verified_evidence.bindings["release_receipt_sha256"]
    ):
        raise ValueError("verified evidence release receipt differs from evaluator identities")

    expected_evaluation = evaluate_verified_pilot_gate(
        metrics[2:],
        metrics[0],
        metrics[1],
        parent_contract_sha256=contract.parent_config_sha256,
        verified_evidence=verified_evidence,
    )
    _require_evaluation_equal(evaluation, expected_evaluation)
    if evaluation.decision.status not in {
        contract.table("status")["pilot_no_go"],
        contract.table("status")["pilot_continue"],
    }:
        raise ValueError("pilot bundle requires a scientific evidence-authorized decision")

    bootstrap_payload = evaluation.npz_bytes()
    metrics_document = pilot_metrics_document(
        metrics,
        checkpoint_selection=evaluation.checkpoint_selection,
        comparator_method=evaluation.comparator_method,
        bootstrap=evaluation.bootstrap,
        gate=evaluation.decision,
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
    )
    metrics_payload = canonical_json_bytes(metrics_document)
    decision = decision_document(
        evaluation.decision,
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
        pilot_metrics_sha256=hashlib.sha256(metrics_payload).hexdigest(),
        pilot_bootstrap_sha256=hashlib.sha256(bootstrap_payload).hexdigest(),
    )
    decision_payload = canonical_json_bytes(decision)
    frozen_entries = {
        "fold_bundles.map.json": fold_map_sha,
        "pilot_execution_v1.toml": contract.config_sha256,
        "unconditional_v1.toml": contract.parent_config_sha256,
        **{
            path: verified_evidence.bindings[binding]
            for path, binding in PILOT_FROZEN_EVIDENCE_BINDING_BY_PATH.items()
        },
    }
    if tuple(sorted(frozen_entries)) != PILOT_FROZEN_INPUT_PATHS:
        raise RuntimeError("internal pilot frozen-input schema changed")
    payloads = {
        "CODE_SHA256SUMS": repository.code_sha256sums,
        "FROZEN_INPUT_SHA256SUMS": sha256sums_bytes(frozen_entries),
        "pilot_execution_v1.toml": child_payload,
        "unconditional_v1.toml": parent_payload,
        "fold_bundles.sha256": contract.sha256_sidecar_bytes(fold_map_sha),
        "pilot_bootstrap.npz": bootstrap_payload,
        "pilot_metrics.json": metrics_payload,
        "decision.json": decision_payload,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": contract.document["artifact"],
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": commit,
        "fold_bundle_sha256": fold_map_sha,
        "checkpoint_selection": checkpoint_selection_record(evaluation.checkpoint_selection),
        "comparator_selection": {"method": evaluation.comparator_method},
        "bootstrap": bootstrap_record(evaluation.bootstrap),
        "gates": gate_decision_record(evaluation.decision),
        "decision_status": evaluation.decision.status,
    }
    complete_manifest = build_bundle_manifest(
        contract,
        bundle_kind="pilot",
        payloads=payloads,
        fields=manifest,
    )
    return _ExpectedPilotBundle(
        payloads=MappingProxyType(payloads),
        manifest=MappingProxyType(manifest),
        complete_manifest=MappingProxyType(complete_manifest),
        fold_bundle_map=_freeze_fold_bundle_map(fold_map),
        fold_bundle_map_sha256=fold_map_sha,
    )


def _authenticate_expected_pilot_bundle(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    expected: _ExpectedPilotBundle,
    expected_tree_sha256: str | None,
    expected_git_commit: str,
    evaluation: PilotEvaluation,
    verified_evidence: VerifiedPilotEvidence,
) -> AuthenticatedPilotBundle:
    snapshot = verify_bundle(
        contract,
        bundle_kind="pilot",
        root=root,
        expected_tree_sha256=(
            None
            if expected_tree_sha256 is None
            else _sha256(expected_tree_sha256, label="expected pilot tree SHA-256")
        ),
    )
    for path, payload in expected.payloads.items():
        observed = snapshot.read_bytes(path, maximum_bytes=max(len(payload), 1))
        if observed != payload:
            raise ValueError(f"pilot bundle payload differs from reconstruction: {path}")
    manifest_payload = snapshot.read_bytes("manifest.json", maximum_bytes=_MAX_JSON_BYTES)
    if manifest_payload != canonical_json_bytes(expected.complete_manifest):
        raise ValueError("pilot manifest differs from reconstructed semantic bindings")
    sidecar = parse_sha256_sidecar(
        snapshot.read_bytes("fold_bundles.sha256", maximum_bytes=65),
        label="pilot fold_bundles.sha256",
    )
    if sidecar != expected.fold_bundle_map_sha256:
        raise ValueError("pilot fold-bundle sidecar differs from its canonical map")
    bootstrap_sha = snapshot.file("pilot_bootstrap.npz").sha256
    metrics_sha = snapshot.file("pilot_metrics.json").sha256
    decision_sha = snapshot.file("decision.json").sha256
    return AuthenticatedPilotBundle(
        bundle=snapshot,
        git_commit=_git_commit(expected_git_commit),
        fold_bundle_sha256_by_fold=expected.fold_bundle_map,
        fold_bundle_map_sha256=expected.fold_bundle_map_sha256,
        pilot_bootstrap_sha256=bootstrap_sha,
        pilot_metrics_sha256=metrics_sha,
        decision_sha256=decision_sha,
        decision_status=evaluation.decision.status,
        verified_evidence_sha256=verified_evidence.evidence_sha256,
        _capability=_AUTHENTICATED_PILOT_CAPABILITY,
    )


def _expected_evaluator_manifest(
    contract: NativeDiffusionV1PilotContract,
    *,
    outer_fold: int,
    git_commit: str,
    trainer_bundle_sha256: str,
    count_prior_sha256: str,
    corruption_sha256: str,
    residual_sha256: str,
    fold_metrics_sha256: str,
    readiness_receipt_sha256: str,
    score_release_sha256: str,
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    fold = contract.fold(outer_fold)
    return {
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
        "rng": _evaluator_rng_document(contract),
        "count_prior": {
            "file_sha256": count_prior_sha256,
            "source": "exact_reopened_trainer_bundle_bytes",
            "recomputed_from_score": False,
        },
        "score": {
            "method_order": list(_METHOD_ORDER),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "corruption_file_sha256": corruption_sha256,
            "residual_logits_file_sha256": residual_sha256,
            "fold_metrics_file_sha256": fold_metrics_sha256,
        },
        "status": {
            "score_release_bound_before_score_open": True,
            "trainer_bundle_reopened": True,
            "all_five_checkpoints_authenticated": True,
            "fold_metrics_complete": True,
            "post_publication_reinference_required": True,
            "reinference_record_location": "path_free_supervisor_result",
            "readiness_receipt_sha256": readiness_receipt_sha256,
            "score_release_sha256": score_release_sha256,
        },
        "artifacts": dict(artifacts),
    }


def _evaluator_rng_document(contract: NativeDiffusionV1PilotContract) -> dict[str, object]:
    rng = contract.table("rng")
    evaluation = contract.table("evaluation")
    return {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_evaluation_rng",
        "derivation": rng["derivation"],
        "evaluation_root_seed": rng["evaluation_root_seed"],
        "validation_namespace": rng["validation_namespace"],
        "validation_key": list(cast(Sequence[object], rng["validation_key"])),
        "validation_rng": rng["validation_rng"],
        "levels": evaluation["levels"],
        "replicates_per_sequence_level": evaluation["replicates_per_sequence_level"],
    }


def _validate_evaluator_environment(
    contract: NativeDiffusionV1PilotContract,
    snapshot: BundleSnapshot,
) -> None:
    environment = _json_object(
        parse_canonical_json(
            snapshot.read_bytes("environment.json", maximum_bytes=_MAX_JSON_BYTES),
            label="evaluator environment.json",
        ),
        ("schema_version", "artifact", "runtime", "determinism"),
        label="evaluator environment.json",
    )
    runtime = _json_object(
        environment["runtime"],
        _EVALUATOR_RUNTIME_FIELDS,
        label="evaluator runtime",
    )
    determinism = _json_object(
        environment["determinism"],
        (
            "cublas_workspace_config",
            "pytorch_allocator",
            "deterministic_algorithms",
            "math_sdpa_only",
            "tf32",
            "amp",
        ),
        label="evaluator determinism",
    )
    parent_environment = contract.parent_table("environment")
    parent_determinism = contract.parent_table("determinism")
    cudnn_parts = str(parent_environment["nvidia_cudnn_cu13"]).split(".")
    if len(cudnn_parts) < 3 or any(not value.isdigit() for value in cudnn_parts[:3]):
        raise ValueError("authenticated parent cuDNN version is invalid")
    expected_runtime = {
        "python": parent_environment["python"],
        "numpy": parent_environment["numpy"],
        "torch": parent_environment["torch"],
        "torch_cuda": parent_environment["torch_cuda"],
        "safetensors": parent_environment["safetensors"],
        "packaging": parent_environment["packaging"],
        "triton": parent_environment["triton"],
        "nvidia_cudnn_cu13": parent_environment["nvidia_cudnn_cu13"],
        "gpu_name": parent_environment["gpu_name"],
        "compute_capability": list(
            cast(Sequence[object], parent_environment["compute_capability"])
        ),
        "visible_cuda_devices": 1,
        "allocator": parent_determinism["pytorch_allocator"],
        "cudnn_runtime_version": (
            int(cudnn_parts[0]) * 10_000 + int(cudnn_parts[1]) * 100 + int(cudnn_parts[2])
        ),
    }
    expected_determinism = {
        "cublas_workspace_config": parent_determinism["cublas_workspace_config"],
        "pytorch_allocator": parent_determinism["pytorch_allocator"],
        "deterministic_algorithms": True,
        "math_sdpa_only": True,
        "tf32": False,
        "amp": False,
    }
    expected_environment = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_evaluator_environment",
        "runtime": expected_runtime,
        "determinism": expected_determinism,
    }
    _require_strict_equal(environment, expected_environment, label="evaluator environment")
    _require_strict_equal(runtime, expected_runtime, label="evaluator runtime")
    _require_strict_equal(
        determinism,
        expected_determinism,
        label="evaluator deterministic runtime",
    )


def _authenticate_corruption_arrays(
    contract: NativeDiffusionV1PilotContract,
    *,
    outer_fold: int,
    arrays: Mapping[str, NDArray[np.generic]],
    expected_payload: bytes,
) -> ScoreCorruptionLedger:
    fold = contract.fold(outer_fold)
    lengths = cast(NDArray[np.uint8], arrays["length"])
    clean = cast(NDArray[np.uint8], arrays["clean_tokens"])
    rows: list[ScoreRow] = []
    for index in range(fold.score_rows):
        length = int(lengths[index])
        if not 8 <= length <= 50 or np.any(clean[index, :length] >= len(ALPHABET)):
            raise ValueError("corruption archive contains an invalid clean sequence")
        sequence = "".join(ALPHABET[int(value)] for value in clean[index, :length])
        rows.append(
            ScoreRow(
                sequence_id=_ascii_sha(arrays["sequence_id"][index], label="sequence ID"),
                sequence=sequence,
                fold=outer_fold,
                homology_component_id=_ascii_sha(
                    arrays["homology_component_id"][index],
                    label="homology component ID",
                ),
                union_component_id=_ascii_sha(
                    arrays["union_component_id"][index],
                    label="union component ID",
                ),
                sampling_weight=float(arrays["sampling_weight"][index]),
            )
        )
    if len({row.homology_component_id for row in rows}) != fold.score_homology_components:
        raise ValueError("corruption homology-component census differs from the fold contract")
    if len({row.union_component_id for row in rows}) != fold.score_union_components:
        raise ValueError("corruption union-component census differs from the fold contract")
    ledger = ScoreCorruptionLedger(
        rows=tuple(rows),
        parent_contract_sha256=contract.parent_config_sha256,
        sequence_id=cast(NDArray[np.bytes_], arrays["sequence_id"]),
        homology_component_id=cast(NDArray[np.bytes_], arrays["homology_component_id"]),
        union_component_id=cast(NDArray[np.bytes_], arrays["union_component_id"]),
        length=lengths,
        sampling_weight=cast(NDArray[np.float64], arrays["sampling_weight"]),
        clean_tokens=clean,
        attention_mask=cast(NDArray[np.bool_], arrays["attention_mask"]),
        case_id=cast(NDArray[np.bytes_], arrays["case_id"]),
        row_index=cast(NDArray[np.uint16], arrays["row_index"]),
        level=cast(NDArray[np.uint8], arrays["level"]),
        replicate=cast(NDArray[np.uint8], arrays["replicate"]),
        row_seed=cast(NDArray[np.uint64], arrays["row_seed"]),
        mask_count=cast(NDArray[np.uint8], arrays["mask_count"]),
        corrupted_tokens=cast(NDArray[np.uint8], arrays["corrupted_tokens"]),
        selected_mask=cast(NDArray[np.bool_], arrays["selected_mask"]),
    )
    if ledger.npz_bytes() != expected_payload:
        raise ValueError("corruption archive differs after semantic reconstruction")
    mask_count = arrays["mask_count"].reshape(fold.score_rows, LEVELS)
    bin_counts = tuple(
        int(np.sum(mask_count[:, start:stop], dtype=np.uint64))
        for start, stop in ((0, 16), (16, 32), (32, 48), (48, 64))
    )
    if (
        int(np.sum(mask_count, dtype=np.uint64)) != fold.score_selected_tokens
        or bin_counts != fold.score_selected_tokens_by_timestep_bin
    ):
        raise ValueError("corruption selected-token census differs from the fold contract")
    return ledger


def _validate_authenticated_evaluator_contents(
    value: AuthenticatedEvaluatorBundle,
) -> None:
    if type(value.ledger) is not ScoreCorruptionLedger:
        raise TypeError("authenticated evaluator ledger has an invalid type")
    if type(value.scoring_archive) is not ScoringArchive:
        raise TypeError("authenticated evaluator scoring archive has an invalid type")
    if type(value.methods) is not tuple:
        raise TypeError("authenticated evaluator methods must be an immutable tuple")
    value.ledger.revalidate()
    value.scoring_archive.revalidate()
    methods = _ordered_fold_methods(value.methods, _outer_fold(value.outer_fold))
    if hashlib.sha256(value.ledger.npz_bytes()).hexdigest() != _sha256(
        value.score_corruptions_sha256,
        label="score-corruptions SHA-256",
    ):
        raise ValueError("authenticated evaluator ledger digest changed")
    if hashlib.sha256(value.scoring_archive.npz_bytes()).hexdigest() != _sha256(
        value.score_residual_logits_sha256,
        label="score-residual-logits SHA-256",
    ):
        raise ValueError("authenticated evaluator scoring archive digest changed")
    if value.scoring_archive.count_prior_sha256 != _sha256(
        value.count_prior_sha256,
        label="count-prior SHA-256",
    ):
        raise ValueError("authenticated evaluator count-prior binding changed")
    if value.scoring_archive.ledger.npz_bytes() != value.ledger.npz_bytes():
        raise ValueError("authenticated evaluator ledger/archive linkage changed")
    metrics_payload = canonical_json_bytes(
        fold_metrics_document(
            methods,
            child_contract_sha256=CONFIG_SHA256,
            parent_contract_sha256=PARENT_CONFIG_SHA256,
        )
    )
    if hashlib.sha256(metrics_payload).hexdigest() != _sha256(
        value.fold_metrics_sha256,
        label="fold-metrics SHA-256",
    ):
        raise ValueError("authenticated evaluator fold metrics changed")


def _ordered_fold_methods(
    methods: Sequence[FoldMethodMetrics],
    outer_fold: int,
) -> tuple[FoldMethodMetrics, ...]:
    values = tuple(methods)
    if len(values) != len(_METHOD_ORDER) or any(
        type(value) is not FoldMethodMetrics for value in values
    ):
        raise TypeError("reconstructed_methods must contain seven exact fold metrics")
    observed = tuple(
        value.method
        if value.checkpoint_step is None
        else f"{value.method}-{value.checkpoint_step:06d}"
        for value in values
    )
    if observed != _METHOD_ORDER or any(value.outer_fold != outer_fold for value in values):
        raise ValueError("reconstructed fold metrics differ from frozen method/fold order")
    for value in values:
        value.revalidate()
    return values


def _validate_method_rows(
    methods: Sequence[FoldMethodMetrics],
    ledger: ScoreCorruptionLedger,
) -> None:
    expected = tuple(
        (
            row.sequence_id,
            row.homology_component_id,
            row.union_component_id,
            row.sampling_weight.hex(),
        )
        for row in ledger.rows
    )
    for method in methods:
        observed = tuple(
            (
                row.sequence_id,
                row.homology_component_id,
                row.union_component_id,
                row.sampling_weight.hex(),
            )
            for row in method.row_nll
        )
        if observed != expected:
            raise ValueError("reconstructed metric rows differ from the corruption ledger")


def _ordered_equal_fold_metrics(
    metrics: Sequence[EqualFoldMetrics],
) -> tuple[EqualFoldMetrics, ...]:
    values = tuple(metrics)
    if len(values) != len(_METHOD_ORDER) or any(
        type(value) is not EqualFoldMetrics for value in values
    ):
        raise TypeError("equal_fold_metrics must contain seven exact aggregates")
    observed = tuple(
        value.method
        if value.checkpoint_step is None
        else f"{value.method}-{value.checkpoint_step:06d}"
        for value in values
    )
    if observed != _METHOD_ORDER:
        raise ValueError("equal-fold metrics differ from the frozen method order")
    for value in values:
        value.revalidate()
    return values


def _ordered_evaluator_bundles(
    values: Sequence[AuthenticatedEvaluatorBundle],
    git_commit: str,
) -> tuple[AuthenticatedEvaluatorBundle, ...]:
    identities = tuple(values)
    if len(identities) != 4 or any(
        type(value) is not AuthenticatedEvaluatorBundle for value in identities
    ):
        raise TypeError("evaluator_bundles must contain four authenticated identities")
    for value in identities:
        value.revalidate()
    if tuple(value.outer_fold for value in identities) != _FOLDS:
        raise ValueError("evaluator identities must be ordered outer folds 0, 1, 2, 3")
    if any(value.git_commit != git_commit for value in identities):
        raise ValueError("evaluator identities differ from the expected Git commit")
    if len({value.bundle.tree_sha256 for value in identities}) != 4:
        raise ValueError("evaluator identities must have four distinct tree digests")
    if len({value.trainer_bundle_sha256 for value in identities}) != 4:
        raise ValueError("evaluator identities must bind four distinct trainer trees")
    return identities


def _require_evaluation_equal(
    observed: PilotEvaluation,
    expected: PilotEvaluation,
) -> None:
    if type(observed) is not PilotEvaluation:
        raise TypeError("evaluation must be an exact PilotEvaluation")
    observed.revalidate()
    expected.revalidate()
    if (
        observed.checkpoint_selection != expected.checkpoint_selection
        or observed.comparator_method != expected.comparator_method
        or bootstrap_record(observed.bootstrap) != bootstrap_record(expected.bootstrap)
        or gate_decision_record(observed.decision) != gate_decision_record(expected.decision)
        or observed.npz_bytes() != expected.npz_bytes()
    ):
        raise ValueError("pilot evaluation differs from reproduced verified gate")


def _flat_fold_map_sha256(
    contract: NativeDiffusionV1PilotContract,
    value: Mapping[str, object],
) -> str:
    return hashlib.sha256(contract.audit_fold_digest_map_bytes(value)).hexdigest()


def _validate_expected_payload(
    payload: bytes,
    *,
    expected_bytes: bytes | None,
    expected_sha256: str | None,
    label: str,
) -> str:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} must be non-empty exact bytes")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_bytes is not None:
        if type(expected_bytes) is not bytes or not expected_bytes:
            raise ValueError(f"expected {label} bytes must be non-empty exact bytes")
        if expected_bytes != payload:
            raise ValueError(f"{label} differs from independently reconstructed bytes")
    if (
        expected_sha256 is not None
        and _sha256(
            expected_sha256,
            label=f"expected {label} SHA-256",
        )
        != observed
    ):
        raise ValueError(f"{label} differs from its independently supplied digest")
    return observed


def _contract_payload(payload: bytes, *, expected_sha256: str, label: str) -> bytes:
    if type(payload) is not bytes or not payload or len(payload) > 262_144:
        raise ValueError(f"{label} payload must be non-empty bounded exact bytes")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} payload differs from its authenticated SHA-256")
    return payload


def _validate_contract(contract: NativeDiffusionV1PilotContract) -> None:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()


def _validate_repository(repository: RepositorySnapshot, expected_git_commit: str) -> str:
    commit = _git_commit(expected_git_commit)
    if type(repository) is not RepositorySnapshot:
        raise TypeError("repository must be an exact RepositorySnapshot")
    # Reconstructing invokes all path-free manifest invariants without touching Git.
    rebuilt = RepositorySnapshot(
        git_commit=repository.git_commit,
        code_sha256sums=repository.code_sha256sums,
        code_sha256=repository.code_sha256,
        entries=repository.entries,
    )
    if rebuilt != repository or repository.git_commit != commit:
        raise ValueError("repository snapshot differs from expected_git_commit")
    return commit


def _freeze_fold_bundle_map(
    value: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping) or tuple(value) != _FOLD_KEYS:
        raise ValueError("fold bundle map must contain ordered keys 0, 1, 2, 3")
    result: dict[str, Mapping[str, str]] = {}
    for fold in _FOLD_KEYS:
        item = value[fold]
        if not isinstance(item, Mapping) or set(item) != {
            "trainer_bundle_sha256",
            "evaluator_bundle_sha256",
        }:
            raise ValueError("fold bundle map values have an invalid schema")
        result[fold] = MappingProxyType(
            {
                "trainer_bundle_sha256": _sha256(
                    item["trainer_bundle_sha256"],
                    label=f"fold {fold} trainer bundle SHA-256",
                ),
                "evaluator_bundle_sha256": _sha256(
                    item["evaluator_bundle_sha256"],
                    label=f"fold {fold} evaluator bundle SHA-256",
                ),
            }
        )
    return MappingProxyType(result)


def _evaluator_identity_record(value: AuthenticatedEvaluatorBundle) -> dict[str, object]:
    if type(value.bundle) is not BundleSnapshot or value.bundle.bundle_kind != "evaluator":
        raise TypeError("authenticated evaluator must contain an evaluator BundleSnapshot")
    fold = _outer_fold(value.outer_fold)
    return {
        "schema_version": 1,
        "outer_fold": fold,
        "git_commit": _git_commit(value.git_commit),
        "evaluator_bundle_sha256": _sha256(
            value.bundle.tree_sha256,
            label="evaluator bundle SHA-256",
        ),
        "trainer_bundle_sha256": _sha256(
            value.trainer_bundle_sha256,
            label="trainer bundle SHA-256",
        ),
        "count_prior_sha256": _sha256(
            value.count_prior_sha256,
            label="count-prior SHA-256",
        ),
        "readiness_receipt_sha256": _sha256(
            value.readiness_receipt_sha256,
            label="readiness-receipt SHA-256",
        ),
        "score_release_sha256": _sha256(
            value.score_release_sha256,
            label="score-release SHA-256",
        ),
        "score_corruptions_sha256": _sha256(
            value.score_corruptions_sha256,
            label="score-corruptions SHA-256",
        ),
        "score_residual_logits_sha256": _sha256(
            value.score_residual_logits_sha256,
            label="score-residual-logits SHA-256",
        ),
        "fold_metrics_sha256": _sha256(
            value.fold_metrics_sha256,
            label="fold-metrics SHA-256",
        ),
    }


def _pilot_identity_record(value: AuthenticatedPilotBundle) -> dict[str, object]:
    if type(value.bundle) is not BundleSnapshot or value.bundle.bundle_kind != "pilot":
        raise TypeError("authenticated pilot must contain a pilot BundleSnapshot")
    return {
        "schema_version": 1,
        "git_commit": _git_commit(value.git_commit),
        "pilot_bundle_sha256": _sha256(
            value.bundle.tree_sha256,
            label="pilot bundle SHA-256",
        ),
        "fold_bundle_sha256_by_fold": {
            fold: dict(value.fold_bundle_sha256_by_fold[fold]) for fold in _FOLD_KEYS
        },
        "fold_bundle_map_sha256": _sha256(
            value.fold_bundle_map_sha256,
            label="fold-bundle map SHA-256",
        ),
        "pilot_bootstrap_sha256": _sha256(
            value.pilot_bootstrap_sha256,
            label="pilot bootstrap SHA-256",
        ),
        "pilot_metrics_sha256": _sha256(
            value.pilot_metrics_sha256,
            label="pilot metrics SHA-256",
        ),
        "decision_sha256": _sha256(value.decision_sha256, label="decision SHA-256"),
        "decision_status": _decision_status(value.decision_status),
        "verified_evidence_sha256": _sha256(
            value.verified_evidence_sha256,
            label="verified evidence SHA-256",
        ),
    }


def _ascii_sha(value: object, *, label: str) -> str:
    if not isinstance(value, np.bytes_):
        raise TypeError(f"{label} must be an exact NumPy byte scalar")
    try:
        decoded = bytes(value).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not ASCII") from error
    return _sha256(decoded, label=label)


def _json_object(
    value: object,
    fields: Sequence[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} has an invalid exact schema")
    return cast(dict[str, object], value)


def _require_strict_equal(left: object, right: object, *, label: str) -> None:
    if not _strict_equal(left, right):
        raise ValueError(f"{label} differs from its frozen value")


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[object, object], left)
        right_map = cast(dict[object, object], right)
        return set(left_map) == set(right_map) and all(
            _strict_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            _strict_equal(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    if type(left) is float:
        return cast(float, left).hex() == cast(float, right).hex()
    return bool(left == right)


def _outer_fold(value: object) -> int:
    if type(value) is not int or value not in _FOLDS:
        raise ValueError("outer fold must be an exact integer in 0..3")
    return value


def _decision_status(value: object) -> str:
    if type(value) is not str or value not in {
        "development_no_go_v1_pilot",
        "development_continue_v1_full_matrix_authorized",
    }:
        raise ValueError("pilot bundle decision status is not scientific")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        raise ValueError("expected_git_commit must be a lowercase 40-character Git ID")
    return value


__all__ = [
    "EVALUATOR_FROZEN_INPUT_PATHS",
    "PILOT_FROZEN_EVIDENCE_BINDING_BY_PATH",
    "PILOT_FROZEN_INPUT_PATHS",
    "AuthenticatedEvaluatorBundle",
    "AuthenticatedPilotBundle",
    "authenticate_evaluator_bundle",
    "authenticate_pilot_bundle",
    "publish_pilot_bundle",
]

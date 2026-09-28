"""Independent excluded-CPU verification for the native-diffusion v1 pilot.

The verifier treats every GPU-side object as untrusted bytes.  It reopens the
sealed trainer and evaluator trees, reconstructs all CPU-computable evidence,
and does not construct a neural model or run inference.  The small functions
in this module are composable so the audit launcher can fail closed after each
immutable boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from amp_challenge.generators.diffusion.v1.pilot_artifacts import (
    RepositorySnapshot,
    build_repository_snapshot,
    canonical_json_bytes,
    parse_canonical_json,
    parse_sha256sums,
    verify_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_bundle import (
    AuthenticatedEvaluatorBundle,
    AuthenticatedPilotBundle,
    authenticate_evaluator_bundle,
    authenticate_pilot_bundle,
    publish_pilot_bundle,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
    load_pilot_execution_v1_contract,
)
from amp_challenge.generators.diffusion.v1.pilot_control import (
    ScoreReleaseReceipt,
    TrainerReadinessObservation,
    TrainerReadinessReceipt,
    parse_score_release_receipt,
    parse_trainer_readiness_receipt,
    verify_score_release_receipt,
    verify_trainer_readiness_receipt,
)
from amp_challenge.generators.diffusion.v1.pilot_data import (
    AuthenticatedCountPrior,
    load_contract_fold_training_projection,
)
from amp_challenge.generators.diffusion.v1.pilot_records import (
    bootstrap_record,
    checkpoint_selection_record,
    equal_fold_metrics_record,
    fold_metrics_document,
    gate_decision_record,
)
from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    EqualFoldMetrics,
    PilotEvaluation,
    ScoringArchive,
    VerifiedPilotEvidence,
    aggregate_equal_folds,
    build_scoring_archive,
    build_verified_pilot_evidence,
    evaluate_pilot_gate,
    evaluate_verified_pilot_gate,
    load_contract_score_corruption_ledger,
    load_deterministic_npz_bytes,
    residual_logits_npz_schema,
    score_all_fold_methods,
)
from amp_challenge.generators.diffusion.v1.pilot_trainer_bundle import (
    AuthenticatedTrainerBundle,
    authenticate_trainer_bundle,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_FOLD_KEYS = ("0", "1", "2", "3")
_CHECKPOINT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_PROJECTION_FILES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "SHA256SUMS",
    "folds/0/score.jsonl",
    "folds/0/train.jsonl",
    "folds/1/score.jsonl",
    "folds/1/train.jsonl",
    "folds/2/score.jsonl",
    "folds/2/train.jsonl",
    "folds/3/score.jsonl",
    "folds/3/train.jsonl",
    "manifest.json",
    "summary.json",
)
_PROJECTION_TOP_MANIFEST_FILES = tuple(path for path in _PROJECTION_FILES if path != "SHA256SUMS")
_PROJECTION_DIRECTORIES = (".", "folds", "folds/0", "folds/1", "folds/2", "folds/3")
_MAX_PROJECTION_FILE_BYTES = 64 << 20
_MAX_RECEIPT_BYTES = 1 << 20
_MAX_ARCHIVE_BYTES = 1 << 30
_PROJECTION_CAPABILITY = object()
_REINFERENCE_CAPABILITY = object()
_RELEASE_CAPABILITY = object()
_COUNT_PRIOR_PANEL_CAPABILITY = object()
_EVALUATOR_PANEL_CAPABILITY = object()
_VERIFICATION_CAPABILITY = object()


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256 hexadecimal")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        raise ValueError("expected Git commit must be 40 lowercase hexadecimal characters")
    return value


def _exact_fold_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_FOLD_KEYS):
        raise ValueError(f"{label} must contain exactly folds 0, 1, 2, and 3")
    return value


@dataclass(frozen=True, slots=True)
class ProjectionEvidencePins:
    """All accepted path-free identities for the frozen development projection."""

    bundle_top_manifest_sha256: str
    bundle_tree_sha256: str
    manifest_sha256: str
    summary_sha256: str
    independent_receipt_sha256: str
    operational_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "bundle_top_manifest_sha256",
            "bundle_tree_sha256",
            "manifest_sha256",
            "summary_sha256",
            "independent_receipt_sha256",
            "operational_receipt_sha256",
        ):
            _sha256(getattr(self, name), label=f"projection {name}")

    @classmethod
    def from_contract(
        cls,
        contract: NativeDiffusionV1PilotContract,
    ) -> ProjectionEvidencePins:
        """Extract the six pins, while reauthenticating the child contract."""

        if type(contract) is not NativeDiffusionV1PilotContract:
            raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
        contract.revalidate()
        projection = contract.table("projection")
        return cls(
            bundle_top_manifest_sha256=projection["bundle_top_manifest_sha256"],
            bundle_tree_sha256=projection["bundle_tree_sha256"],
            manifest_sha256=projection["manifest_sha256"],
            summary_sha256=projection["summary_sha256"],
            independent_receipt_sha256=projection["independent_receipt_sha256"],
            operational_receipt_sha256=projection["operational_receipt_sha256"],
        )

    def document(self) -> dict[str, str]:
        return {
            "bundle_top_manifest_sha256": self.bundle_top_manifest_sha256,
            "bundle_tree_sha256": self.bundle_tree_sha256,
            "manifest_sha256": self.manifest_sha256,
            "summary_sha256": self.summary_sha256,
            "independent_receipt_sha256": self.independent_receipt_sha256,
            "operational_receipt_sha256": self.operational_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ProjectionFile:
    relative_path: str
    payload_sha256: str
    fingerprint: tuple[int, int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _relative_projection_path(self.relative_path)
        _sha256(self.payload_sha256, label="projection file digest")
        if (
            type(self.fingerprint) is not tuple
            or len(self.fingerprint) != 7
            or any(type(value) is not int or value < 0 for value in self.fingerprint)
        ):
            raise ValueError("projection file fingerprint is invalid")


@dataclass(frozen=True, slots=True)
class AuthenticatedProjection:
    """Immutable projection tree checked against every frozen evidence pin."""

    root: Path
    pins: ProjectionEvidencePins
    files: Mapping[str, _ProjectionFile]
    directory_fingerprints: Mapping[str, tuple[int, int, int, int, int]]
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _PROJECTION_CAPABILITY:
            raise RuntimeError("projection snapshot requires internal authentication")
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("projection root must be an absolute Path")
        if type(self.pins) is not ProjectionEvidencePins:
            raise TypeError("projection pins must be exact ProjectionEvidencePins")
        raw_files = dict(self.files)
        if tuple(sorted(raw_files)) != tuple(sorted(_PROJECTION_FILES)) or any(
            type(value) is not _ProjectionFile for value in raw_files.values()
        ):
            raise ValueError("projection file snapshot has the wrong inventory")
        raw_directories = dict(self.directory_fingerprints)
        if tuple(sorted(raw_directories)) != tuple(sorted(_PROJECTION_DIRECTORIES)):
            raise ValueError("projection directory snapshot has the wrong inventory")
        object.__setattr__(self, "files", MappingProxyType(raw_files))
        object.__setattr__(
            self,
            "directory_fingerprints",
            MappingProxyType(raw_directories),
        )
        object.__setattr__(self, "_capability", None)

    def read_bytes(self, relative_path: str, *, maximum_bytes: int) -> bytes:
        """Reopen one file only if its full filesystem and byte identity persists."""

        if self._capability is not None:
            raise ValueError("projection authentication capability state changed")
        relative = _relative_projection_path(relative_path)
        if relative not in self.files:
            raise KeyError(relative)
        payload, fingerprint = _read_sealed_file(
            self.root.joinpath(*PurePosixPath(relative).parts),
            maximum_bytes=maximum_bytes,
            label=f"projection {relative}",
        )
        expected = self.files[relative]
        if (
            fingerprint != expected.fingerprint
            or hashlib.sha256(payload).hexdigest() != expected.payload_sha256
        ):
            raise ValueError(f"frozen projection file changed: {relative}")
        return payload

    def revalidate(self) -> None:
        """Resnapshot the complete tree and reject any replacement or mutation."""

        if self._capability is not None:
            raise ValueError("projection authentication capability state changed")
        observed_files, observed_directories, tree_sha256 = _snapshot_projection_tree(self.root)
        if (
            tree_sha256 != self.pins.bundle_tree_sha256
            or observed_files != dict(self.files)
            or observed_directories != dict(self.directory_fingerprints)
        ):
            raise ValueError("frozen projection tree changed after authentication")


def authenticate_frozen_projection(
    contract: NativeDiffusionV1PilotContract,
    root: str | os.PathLike[str],
    *,
    evidence_pins: ProjectionEvidencePins,
) -> AuthenticatedProjection:
    """Authenticate the accepted immutable projection without producer imports."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(evidence_pins) is not ProjectionEvidencePins:
        raise TypeError("evidence_pins must be exact ProjectionEvidencePins")
    expected_pins = ProjectionEvidencePins.from_contract(contract)
    if evidence_pins != expected_pins:
        raise ValueError("projection evidence pins differ from the authenticated child contract")
    absolute = Path(os.path.abspath(os.fspath(root)))
    files, directories, tree_sha256 = _snapshot_projection_tree(absolute)
    if tree_sha256 != evidence_pins.bundle_tree_sha256:
        raise ValueError("projection tree SHA-256 differs from its accepted pin")
    top = files["SHA256SUMS"]
    if top.payload_sha256 != evidence_pins.bundle_top_manifest_sha256:
        raise ValueError("projection top manifest differs from its accepted pin")
    if files["manifest.json"].payload_sha256 != evidence_pins.manifest_sha256:
        raise ValueError("projection manifest differs from its accepted pin")
    if files["summary.json"].payload_sha256 != evidence_pins.summary_sha256:
        raise ValueError("projection summary differs from its accepted pin")
    top_payload, _ = _read_sealed_file(
        absolute / "SHA256SUMS",
        maximum_bytes=1 << 20,
        label="projection SHA256SUMS",
    )
    top_entries = parse_sha256sums(top_payload, label="projection SHA256SUMS")
    if set(top_entries) != set(_PROJECTION_TOP_MANIFEST_FILES):
        raise ValueError("projection SHA256SUMS has the wrong exact inventory")
    for relative in _PROJECTION_TOP_MANIFEST_FILES:
        if top_entries[relative] != files[relative].payload_sha256:
            raise ValueError(f"projection SHA256SUMS mismatch for {relative}")
    result = AuthenticatedProjection(
        root=absolute,
        pins=evidence_pins,
        files=files,
        directory_fingerprints=directories,
        _capability=_PROJECTION_CAPABILITY,
    )
    result.revalidate()
    return result


def _relative_projection_path(value: object) -> str:
    if type(value) is not str or value not in _PROJECTION_FILES:
        raise ValueError("path is not in the exact frozen projection inventory")
    return value


def _reject_symlink_ancestors(path: Path) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    for candidate in (*reversed(target.parents), target):
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"path traverses a symbolic link: {candidate}")


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _read_sealed_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive exact integer")
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_ancestors(absolute)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} is not a bounded immutable single-link file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    fingerprint = _stat_fingerprint(before)
    if (
        fingerprint != _stat_fingerprint(after)
        or fingerprint != _stat_fingerprint(named)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload, fingerprint


def _snapshot_projection_tree(
    root: Path,
) -> tuple[
    dict[str, _ProjectionFile],
    dict[str, tuple[int, int, int, int, int]],
    str,
]:
    absolute = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_ancestors(absolute)
    expected_children = {
        ".": {
            "CODE_SHA256SUMS",
            "FROZEN_INPUT_SHA256SUMS",
            "SHA256SUMS",
            "folds",
            "manifest.json",
            "summary.json",
        },
        "folds": {"0", "1", "2", "3"},
        "folds/0": {"score.jsonl", "train.jsonl"},
        "folds/1": {"score.jsonl", "train.jsonl"},
        "folds/2": {"score.jsonl", "train.jsonl"},
        "folds/3": {"score.jsonl", "train.jsonl"},
    }
    directories: dict[str, tuple[int, int, int, int, int]] = {}
    for relative in _PROJECTION_DIRECTORIES:
        path = absolute if relative == "." else absolute.joinpath(*relative.split("/"))
        observed = os.lstat(path)
        if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o555:
            raise ValueError(f"projection directory {relative} must be an immutable real directory")
        children = {entry.name for entry in os.scandir(path)}
        if children != expected_children[relative]:
            raise ValueError(f"projection directory {relative} has the wrong inventory")
        child_stats = {name: os.lstat(path / name) for name in children}
        if any(stat.S_ISLNK(value.st_mode) for value in child_stats.values()):
            raise ValueError(f"projection directory {relative} contains a symbolic link")
        expected_links = 2 + sum(stat.S_ISDIR(value.st_mode) for value in child_stats.values())
        if observed.st_nlink != expected_links:
            raise ValueError(f"projection directory {relative} has a wrong link count")
        fingerprint = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
            observed.st_nlink,
        )
        after = os.lstat(path)
        if fingerprint != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise ValueError(f"projection directory {relative} changed during snapshot")
        directories[relative] = fingerprint
    files: dict[str, _ProjectionFile] = {}
    transcript = bytearray()
    for relative in sorted(_PROJECTION_FILES):
        payload, fingerprint = _read_sealed_file(
            absolute.joinpath(*PurePosixPath(relative).parts),
            maximum_bytes=_MAX_PROJECTION_FILE_BYTES,
            label=f"projection {relative}",
        )
        digest = hashlib.sha256(payload).hexdigest()
        files[relative] = _ProjectionFile(relative, digest, fingerprint)
        transcript.extend(f"444 {digest} {relative}\n".encode("ascii"))
    return files, directories, hashlib.sha256(transcript).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthenticatedReleaseChain:
    """Four independently reopened trainers and their exact score release."""

    repository: RepositorySnapshot
    trainers_by_fold: Mapping[str, AuthenticatedTrainerBundle]
    readiness_by_fold: Mapping[str, TrainerReadinessReceipt]
    readiness_bytes_by_fold: Mapping[str, bytes]
    release: ScoreReleaseReceipt
    release_bytes: bytes
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _RELEASE_CAPABILITY:
            raise RuntimeError("release chain requires internal authentication")
        if type(self.repository) is not RepositorySnapshot:
            raise TypeError("repository must be an exact RepositorySnapshot")
        trainers = _exact_fold_mapping(self.trainers_by_fold, label="trainer map")
        readiness = _exact_fold_mapping(self.readiness_by_fold, label="readiness map")
        payloads = _exact_fold_mapping(
            self.readiness_bytes_by_fold,
            label="readiness byte map",
        )
        frozen_trainers: dict[str, AuthenticatedTrainerBundle] = {}
        frozen_readiness: dict[str, TrainerReadinessReceipt] = {}
        frozen_payloads: dict[str, bytes] = {}
        for key in _FOLD_KEYS:
            trainer = trainers[key]
            receipt = readiness[key]
            payload = payloads[key]
            if type(trainer) is not AuthenticatedTrainerBundle:
                raise TypeError(f"trainer map fold {key} has an invalid value")
            if type(receipt) is not TrainerReadinessReceipt:
                raise TypeError(f"readiness map fold {key} has an invalid value")
            if type(payload) is not bytes or receipt.canonical_bytes() != payload:
                raise ValueError(f"readiness bytes changed at fold {key}")
            if trainer.outer_fold != int(key) or receipt.outer_fold != int(key):
                raise ValueError(f"release-chain value is under the wrong fold {key}")
            frozen_trainers[key] = trainer
            frozen_readiness[key] = receipt
            frozen_payloads[key] = payload
        if type(self.release) is not ScoreReleaseReceipt:
            raise TypeError("release must be an exact ScoreReleaseReceipt")
        if type(self.release_bytes) is not bytes or self.release.canonical_bytes() != (
            self.release_bytes
        ):
            raise ValueError("score-release bytes changed")
        object.__setattr__(
            self,
            "trainers_by_fold",
            MappingProxyType(frozen_trainers),
        )
        object.__setattr__(
            self,
            "readiness_by_fold",
            MappingProxyType(frozen_readiness),
        )
        object.__setattr__(
            self,
            "readiness_bytes_by_fold",
            MappingProxyType(frozen_payloads),
        )
        object.__setattr__(self, "_capability", None)

    @property
    def release_sha256(self) -> str:
        if self._capability is not None:
            raise ValueError("release-chain capability state changed")
        return hashlib.sha256(self.release_bytes).hexdigest()

    def trainer_digest_map(self) -> dict[str, str]:
        if self._capability is not None:
            raise ValueError("release-chain capability state changed")
        return {key: self.trainers_by_fold[key].bundle.tree_sha256 for key in _FOLD_KEYS}

    def count_prior_digest_map(self) -> dict[str, str]:
        if self._capability is not None:
            raise ValueError("release-chain capability state changed")
        return {key: self.trainers_by_fold[key].count_prior.sha256 for key in _FOLD_KEYS}

    def checkpoint_digest_map(self) -> dict[str, dict[str, dict[str, str]]]:
        if self._capability is not None:
            raise ValueError("release-chain capability state changed")
        return {
            fold: {
                step: self.trainers_by_fold[fold].checkpoint_digest_by_step[step].document()
                for step in _CHECKPOINT_KEYS
            }
            for fold in _FOLD_KEYS
        }

    def revalidate(self, contract: NativeDiffusionV1PilotContract) -> None:
        """Recheck all sealed trees and rebuild the release from typed evidence."""

        if self._capability is not None:
            raise ValueError("release-chain capability state changed")
        if type(contract) is not NativeDiffusionV1PilotContract:
            raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
        contract.revalidate()
        observations: dict[str, TrainerReadinessObservation] = {}
        for key in _FOLD_KEYS:
            trainer = self.trainers_by_fold[key]
            receipt = self.readiness_by_fold[key]
            verify_bundle(
                contract,
                bundle_kind="trainer",
                root=trainer.bundle.root,
                expected_tree_sha256=trainer.bundle.tree_sha256,
                expected_tree_bytes=trainer.bundle.tree_bytes,
            )
            trainer.count_prior.revalidate()
            observation = trainer.make_readiness_observation(
                trusted_git_commit=self.repository.git_commit,
                node_name=receipt.node_name,
                device_uuid=receipt.device_uuid,
            )
            verify_trainer_readiness_receipt(
                receipt,
                contract=contract,
                observation=observation,
            )
            if receipt.canonical_bytes() != self.readiness_bytes_by_fold[key]:
                raise ValueError(f"readiness bytes changed at fold {key}")
            observations[key] = observation
        verify_score_release_receipt(
            self.release,
            contract=contract,
            git_commit=self.repository.git_commit,
            readiness_receipt_bytes_by_fold=self.readiness_bytes_by_fold,
            observations_by_fold=observations,
        )
        if self.release.canonical_bytes() != self.release_bytes:
            raise ValueError("score-release bytes changed")


def authenticate_release_chain(
    contract: NativeDiffusionV1PilotContract,
    *,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    trainer_roots_by_fold: Mapping[str, str | os.PathLike[str]],
    readiness_receipt_bytes_by_fold: Mapping[str, bytes],
    score_release_bytes: bytes,
) -> AuthenticatedReleaseChain:
    """Rebuild all four readiness receipts and the release from bundle bytes."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(repository) is not RepositorySnapshot:
        raise TypeError("repository must be an exact clean RepositorySnapshot")
    commit = _git_commit(expected_git_commit)
    if repository.git_commit != commit:
        raise ValueError("repository snapshot differs from the expected Git commit")
    roots = _exact_fold_mapping(trainer_roots_by_fold, label="trainer root map")
    receipt_payloads = _exact_fold_mapping(
        readiness_receipt_bytes_by_fold,
        label="readiness receipt bytes by fold",
    )
    if (
        type(score_release_bytes) is not bytes
        or not 0 < len(score_release_bytes) <= _MAX_RECEIPT_BYTES
    ):
        raise ValueError("score-release receipt must be non-empty bounded exact bytes")
    trainers: dict[str, AuthenticatedTrainerBundle] = {}
    receipts: dict[str, TrainerReadinessReceipt] = {}
    observations: dict[str, TrainerReadinessObservation] = {}
    exact_payloads: dict[str, bytes] = {}
    for key in _FOLD_KEYS:
        payload = receipt_payloads[key]
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_RECEIPT_BYTES:
            raise ValueError(f"readiness receipt fold {key} must be bounded exact bytes")
        receipt = parse_trainer_readiness_receipt(payload, contract=contract)
        if receipt.outer_fold != int(key) or receipt.git_commit != commit:
            raise ValueError(f"readiness receipt is under the wrong fold or commit {key}")
        trainer = authenticate_trainer_bundle(
            contract,
            roots[key],
            expected_outer_fold=int(key),
            expected_code_sha256sums=repository.code_sha256sums,
            expected_tree_sha256=receipt.trainer_bundle_sha256,
        )
        observation = trainer.make_readiness_observation(
            trusted_git_commit=commit,
            node_name=receipt.node_name,
            device_uuid=receipt.device_uuid,
        )
        verify_trainer_readiness_receipt(
            receipt,
            contract=contract,
            observation=observation,
        )
        trainers[key] = trainer
        receipts[key] = receipt
        observations[key] = observation
        exact_payloads[key] = payload
    release = parse_score_release_receipt(score_release_bytes, contract=contract)
    if release.git_commit != commit:
        raise ValueError("score release differs from the expected Git commit")
    verify_score_release_receipt(
        release,
        contract=contract,
        git_commit=commit,
        readiness_receipt_bytes_by_fold=exact_payloads,
        observations_by_fold=observations,
    )
    if len({value.node_name for value in receipts.values()}) != 4:
        raise ValueError("release chain does not contain four distinct producer nodes")
    if len({value.device_uuid for value in receipts.values()}) != 4:
        raise ValueError("release chain does not contain four distinct producer devices")
    return AuthenticatedReleaseChain(
        repository=repository,
        trainers_by_fold=trainers,
        readiness_by_fold=receipts,
        readiness_bytes_by_fold=exact_payloads,
        release=release,
        release_bytes=score_release_bytes,
        _capability=_RELEASE_CAPABILITY,
    )


@dataclass(frozen=True, slots=True)
class ReconstructedCountPriorPanel:
    """Four C0 payloads rebuilt byte-for-byte from frozen train JSONL."""

    priors_by_fold: Mapping[str, AuthenticatedCountPrior]
    projection_tree_sha256: str
    trainer_digest_map_sha256: str
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _COUNT_PRIOR_PANEL_CAPABILITY:
            raise RuntimeError("count-prior panel requires reconstruction capability")
        priors = _exact_fold_mapping(self.priors_by_fold, label="count-prior panel")
        frozen: dict[str, AuthenticatedCountPrior] = {}
        for key in _FOLD_KEYS:
            prior = priors[key]
            if type(prior) is not AuthenticatedCountPrior:
                raise TypeError(f"count-prior panel fold {key} has an invalid type")
            prior.revalidate()
            frozen[key] = prior
        _sha256(self.projection_tree_sha256, label="count-prior projection tree")
        _sha256(self.trainer_digest_map_sha256, label="count-prior trainer map")
        object.__setattr__(self, "priors_by_fold", MappingProxyType(frozen))
        object.__setattr__(self, "_capability", None)

    def revalidate(self) -> None:
        """Reject relabeling and revalidate every independently rebuilt prior."""

        if self._capability is not None:
            raise ValueError("count-prior reconstruction capability state changed")
        if type(self.priors_by_fold) is not MappingProxyType:
            raise TypeError("count-prior panel lost immutable storage")
        priors = _exact_fold_mapping(self.priors_by_fold, label="count-prior panel")
        for key in _FOLD_KEYS:
            prior = priors[key]
            if type(prior) is not AuthenticatedCountPrior:
                raise TypeError(f"count-prior panel fold {key} has an invalid type")
            prior.revalidate()
        _sha256(self.projection_tree_sha256, label="count-prior projection tree")
        _sha256(self.trainer_digest_map_sha256, label="count-prior trainer map")

    def digest_map(self) -> dict[str, str]:
        self.revalidate()
        return {key: self.priors_by_fold[key].sha256 for key in _FOLD_KEYS}


def reconstruct_count_priors(
    contract: NativeDiffusionV1PilotContract,
    *,
    projection: AuthenticatedProjection,
    release_chain: AuthenticatedReleaseChain,
) -> ReconstructedCountPriorPanel:
    """Recompute all four C0 archives and byte-compare their sealed payloads."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(projection) is not AuthenticatedProjection:
        raise TypeError("projection must be an AuthenticatedProjection")
    if type(release_chain) is not AuthenticatedReleaseChain:
        raise TypeError("release_chain must be an AuthenticatedReleaseChain")
    reconstructed: dict[str, AuthenticatedCountPrior] = {}
    for key in _FOLD_KEYS:
        fold = int(key)
        relative = f"folds/{fold}/train.jsonl"
        trusted_bytes = projection.read_bytes(
            relative,
            maximum_bytes=_MAX_PROJECTION_FILE_BYTES,
        )
        training = load_contract_fold_training_projection(
            contract,
            fold,
            projection.root.joinpath(*PurePosixPath(relative).parts),
        )
        if training.canonical_bytes() != trusted_bytes:
            raise ValueError(f"training projection reconstruction changed at fold {key}")
        observed = AuthenticatedCountPrior.from_projection(training)
        sealed = release_chain.trainers_by_fold[key].count_prior
        sealed.revalidate()
        if observed.sha256 != sealed.sha256 or observed.payload != sealed.payload:
            raise ValueError(f"count-prior bytes differ from train reconstruction at fold {key}")
        reconstructed[key] = observed
    projection.revalidate()
    return ReconstructedCountPriorPanel(
        priors_by_fold=reconstructed,
        projection_tree_sha256=projection.pins.bundle_tree_sha256,
        trainer_digest_map_sha256=_fold_digest_map_sha256(
            contract,
            release_chain.trainer_digest_map(),
        ),
        _capability=_COUNT_PRIOR_PANEL_CAPABILITY,
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedProducerReinference:
    """Canonical path-free 4x5 archived-versus-reinferred digest pairs."""

    canonical_bytes: bytes
    sha256: str
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _REINFERENCE_CAPABILITY:
            raise RuntimeError("reinference evidence requires internal authentication")
        if type(self.canonical_bytes) is not bytes or not self.canonical_bytes:
            raise ValueError("reinference evidence must be non-empty canonical bytes")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != _sha256(
            self.sha256,
            label="producer reinference digest",
        ):
            raise ValueError("producer reinference bytes differ from their digest")
        object.__setattr__(self, "_capability", None)

    def document(self) -> dict[str, object]:
        if self._capability is not None:
            raise ValueError("reinference authentication capability state changed")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != _sha256(
            self.sha256,
            label="producer reinference digest",
        ):
            raise ValueError("producer reinference bytes changed after authentication")
        value = parse_canonical_json(
            self.canonical_bytes,
            label="producer reinference evidence",
        )
        if type(value) is not dict:
            raise TypeError("producer reinference evidence must be an exact JSON object")
        return value

    def archived_slice_sha256(self, fold: str, step: str) -> str:
        if fold not in _FOLD_KEYS or step not in _CHECKPOINT_KEYS:
            raise KeyError((fold, step))
        document = self.document()
        by_fold = document["by_fold_and_step"]
        if type(by_fold) is not dict:  # pragma: no cover - authenticated invariant
            raise RuntimeError("reinference fold map lost its exact JSON type")
        fold_map = by_fold[fold]
        if type(fold_map) is not dict:  # pragma: no cover - authenticated invariant
            raise RuntimeError("reinference step map lost its exact JSON type")
        item = fold_map[step]
        if type(item) is not dict:  # pragma: no cover - authenticated invariant
            raise RuntimeError("reinference entry lost its exact JSON type")
        return _sha256(
            item["archived_residual_logit_slice_sha256"],
            label="archived residual-logit slice",
        )


def authenticate_producer_reinference(
    contract: NativeDiffusionV1PilotContract,
    evidence: bytes | Mapping[str, object],
) -> AuthenticatedProducerReinference:
    """Validate exact 4x5 digest-pair evidence and cut all mutable aliases."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(evidence) is bytes:
        parsed = parse_canonical_json(evidence, label="producer reinference evidence")
        if type(parsed) is not dict:
            raise TypeError("producer reinference evidence must be an exact JSON object")
        supplied = evidence
        normalized = contract.producer_gpu_reinference_receipt_bytes(parsed)
    elif isinstance(evidence, Mapping):
        normalized = contract.producer_gpu_reinference_receipt_bytes(evidence)
        supplied = canonical_json_bytes(evidence)
    else:
        raise TypeError("producer reinference evidence must be canonical bytes or a mapping")
    if supplied != normalized:
        raise ValueError("producer reinference evidence is not in the exact contract encoding")
    return AuthenticatedProducerReinference(
        canonical_bytes=normalized,
        sha256=hashlib.sha256(normalized).hexdigest(),
        _capability=_REINFERENCE_CAPABILITY,
    )


def verify_producer_reinference_against_archives(
    evidence: AuthenticatedProducerReinference,
    archives_by_fold: Mapping[str, ScoringArchive],
) -> str:
    """Tie every GPU digest pair to the archived residual slice read by the CPU."""

    if type(evidence) is not AuthenticatedProducerReinference:
        raise TypeError("evidence must be AuthenticatedProducerReinference")
    archives = _exact_fold_mapping(archives_by_fold, label="scoring archive map")
    for fold in _FOLD_KEYS:
        archive = archives[fold]
        if type(archive) is not ScoringArchive:
            raise TypeError(f"scoring archive fold {fold} has an invalid type")
        archive.revalidate()
        residual = archive.arrays()["residual_logit"]
        for index, step in enumerate(_CHECKPOINT_KEYS):
            digest = hashlib.sha256(residual[index].tobytes(order="C")).hexdigest()
            if digest != evidence.archived_slice_sha256(fold, step):
                raise ValueError(
                    f"producer reinference is not bound to fold {fold} step {step} archive"
                )
    return evidence.sha256


@dataclass(frozen=True, slots=True)
class AuthenticatedEvaluatorPanel:
    """Four evaluator bundles reconstructed from projection rows and raw archives."""

    evaluators: tuple[AuthenticatedEvaluatorBundle, ...]
    count_prior_reconstruction: ReconstructedCountPriorPanel = field(
        repr=False,
        compare=False,
    )
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _EVALUATOR_PANEL_CAPABILITY:
            raise RuntimeError("evaluator panel requires internal authentication")
        if type(self.count_prior_reconstruction) is not ReconstructedCountPriorPanel:
            raise TypeError("evaluator panel requires reconstructed count-prior evidence")
        if (
            type(self.evaluators) is not tuple
            or len(self.evaluators) != 4
            or any(type(value) is not AuthenticatedEvaluatorBundle for value in self.evaluators)
        ):
            raise TypeError("evaluator panel requires four exact authenticated bundles")
        for value in self.evaluators:
            value.revalidate()
        if tuple(value.outer_fold for value in self.evaluators) != (0, 1, 2, 3):
            raise ValueError("evaluator panel must be ordered by outer fold")
        if len({value.bundle.tree_sha256 for value in self.evaluators}) != 4:
            raise ValueError("evaluator panel must contain four distinct bundle trees")
        for value in self.evaluators:
            expected = self.count_prior_reconstruction.priors_by_fold[str(value.outer_fold)].sha256
            if value.count_prior_sha256 != expected:
                raise ValueError("evaluator panel count-prior binding changed")
        self.count_prior_reconstruction.revalidate()
        object.__setattr__(self, "_capability", None)

    def by_fold(self) -> Mapping[str, AuthenticatedEvaluatorBundle]:
        if self._capability is not None:
            raise ValueError("evaluator-panel capability state changed")
        return MappingProxyType({str(value.outer_fold): value for value in self.evaluators})

    def digest_map(self) -> dict[str, str]:
        if self._capability is not None:
            raise ValueError("evaluator-panel capability state changed")
        return {str(value.outer_fold): value.bundle.tree_sha256 for value in self.evaluators}

    def scoring_archive_map(self) -> Mapping[str, ScoringArchive]:
        if self._capability is not None:
            raise ValueError("evaluator-panel capability state changed")
        return MappingProxyType(
            {str(value.outer_fold): value.scoring_archive for value in self.evaluators}
        )

    def revalidate(self, contract: NativeDiffusionV1PilotContract) -> None:
        if self._capability is not None:
            raise ValueError("evaluator-panel capability state changed")
        if type(contract) is not NativeDiffusionV1PilotContract:
            raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
        contract.revalidate()
        self.count_prior_reconstruction.revalidate()
        if (
            type(self.evaluators) is not tuple
            or len(self.evaluators) != 4
            or any(type(value) is not AuthenticatedEvaluatorBundle for value in self.evaluators)
        ):
            raise TypeError("evaluator panel requires four exact authenticated bundles")
        if tuple(value.outer_fold for value in self.evaluators) != (0, 1, 2, 3):
            raise ValueError("evaluator panel must be ordered by outer fold")
        if len({value.bundle.tree_sha256 for value in self.evaluators}) != 4:
            raise ValueError("evaluator panel must contain four distinct bundle trees")
        for evaluator in self.evaluators:
            evaluator.revalidate()
            verify_bundle(
                contract,
                bundle_kind="evaluator",
                root=evaluator.bundle.root,
                expected_tree_sha256=evaluator.bundle.tree_sha256,
                expected_tree_bytes=evaluator.bundle.tree_bytes,
            )
            expected_prior = self.count_prior_reconstruction.priors_by_fold[
                str(evaluator.outer_fold)
            ]
            expected_prior.revalidate()
            if evaluator.count_prior_sha256 != expected_prior.sha256:
                raise ValueError("evaluator panel count-prior binding changed")


def authenticate_evaluator_panel(
    contract: NativeDiffusionV1PilotContract,
    *,
    projection: AuthenticatedProjection,
    release_chain: AuthenticatedReleaseChain,
    reconstructed_count_priors: ReconstructedCountPriorPanel,
    evaluator_roots_by_fold: Mapping[str, str | os.PathLike[str]],
) -> AuthenticatedEvaluatorPanel:
    """Reconstruct ledgers, count logits, schemas, LOUCO, and all fold metrics."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(projection) is not AuthenticatedProjection:
        raise TypeError("projection must be an exact AuthenticatedProjection")
    if type(release_chain) is not AuthenticatedReleaseChain:
        raise TypeError("release_chain must be an exact AuthenticatedReleaseChain")
    if type(reconstructed_count_priors) is not ReconstructedCountPriorPanel:
        raise TypeError("reconstructed_count_priors must be an exact reconstructed panel")
    if (
        reconstructed_count_priors.projection_tree_sha256 != projection.pins.bundle_tree_sha256
        or reconstructed_count_priors.trainer_digest_map_sha256
        != _fold_digest_map_sha256(contract, release_chain.trainer_digest_map())
    ):
        raise ValueError("count-prior reconstruction belongs to different protected inputs")
    count_priors = reconstructed_count_priors.priors_by_fold
    roots = _exact_fold_mapping(evaluator_roots_by_fold, label="evaluator root map")
    authenticated: list[AuthenticatedEvaluatorBundle] = []
    for key in _FOLD_KEYS:
        outer_fold = int(key)
        count_prior = count_priors[key]
        if type(count_prior) is not AuthenticatedCountPrior:
            raise TypeError(f"count-prior fold {key} has an invalid type")
        count_prior.revalidate()
        relative_score = f"folds/{key}/score.jsonl"
        expected_score_bytes = projection.read_bytes(
            relative_score,
            maximum_bytes=_MAX_PROJECTION_FILE_BYTES,
        )
        expected_ledger = load_contract_score_corruption_ledger(
            contract,
            outer_fold,
            projection.root.joinpath(*PurePosixPath(relative_score).parts),
        )
        expected_corruption_bytes = expected_ledger.npz_bytes()

        # This first pass grants no semantic authority.  It only obtains exact
        # sealed bytes so the CPU can independently rebuild the score objects
        # required by the neutral evaluator authenticator.
        raw = verify_bundle(
            contract,
            bundle_kind="evaluator",
            root=roots[key],
        )
        corruption_bytes = raw.read_bytes(
            "score_corruptions.npz",
            maximum_bytes=_MAX_ARCHIVE_BYTES,
        )
        if corruption_bytes != expected_corruption_bytes:
            raise ValueError(
                f"evaluator corruption archive differs from CPU reconstruction at fold {key}"
            )
        residual_bytes = raw.read_bytes(
            "score_residual_logits.npz",
            maximum_bytes=_MAX_ARCHIVE_BYTES,
        )
        residual_sha256 = hashlib.sha256(residual_bytes).hexdigest()
        fold = contract.fold(outer_fold)
        residual_arrays = load_deterministic_npz_bytes(
            residual_bytes,
            expected_sha256=residual_sha256,
            schema=residual_logits_npz_schema(
                fold.score_cases,
                fold.score_selected_tokens,
            ),
        )
        archive = build_scoring_archive(
            expected_ledger,
            count_prior=count_prior,
            case_id=residual_arrays["case_id"],
            case_offsets=residual_arrays["case_offsets"],
            position=residual_arrays["position"],
            target_token=residual_arrays["target_token"],
            count_log_probability=residual_arrays["count_log_probability"],
            checkpoint_step=residual_arrays["checkpoint_step"],
            residual_logit=residual_arrays["residual_logit"],
        )
        if archive.npz_bytes() != residual_bytes:
            raise ValueError(f"evaluator residual archive is not canonical at fold {key}")
        methods = score_all_fold_methods(archive)
        reconstructed_metrics = canonical_json_bytes(
            fold_metrics_document(
                methods,
                child_contract_sha256=contract.config_sha256,
                parent_contract_sha256=contract.parent_config_sha256,
            )
        )
        if raw.read_bytes("fold_metrics.json", maximum_bytes=64 << 20) != (reconstructed_metrics):
            raise ValueError(f"evaluator fold metrics differ from CPU scoring at fold {key}")
        readiness_sha256 = hashlib.sha256(release_chain.readiness_bytes_by_fold[key]).hexdigest()
        evaluator = authenticate_evaluator_bundle(
            contract,
            roots[key],
            expected_outer_fold=outer_fold,
            expected_git_commit=release_chain.repository.git_commit,
            repository=release_chain.repository,
            expected_trainer_bundle_sha256=release_chain.trainers_by_fold[key].bundle.tree_sha256,
            expected_count_prior_sha256=count_prior.sha256,
            count_prior=count_prior,
            expected_readiness_receipt_sha256=readiness_sha256,
            expected_score_release_sha256=release_chain.release_sha256,
            reconstructed_methods=methods,
            expected_tree_sha256=raw.tree_sha256,
            expected_score_corruptions_bytes=expected_corruption_bytes,
            expected_score_corruptions_sha256=hashlib.sha256(expected_corruption_bytes).hexdigest(),
            expected_score_residual_logits_bytes=residual_bytes,
            expected_score_residual_logits_sha256=residual_sha256,
        )
        if (
            evaluator.ledger.npz_bytes() != expected_corruption_bytes
            or evaluator.scoring_archive.npz_bytes() != residual_bytes
            or canonical_json_bytes(
                fold_metrics_document(
                    evaluator.methods,
                    child_contract_sha256=contract.config_sha256,
                    parent_contract_sha256=contract.parent_config_sha256,
                )
            )
            != reconstructed_metrics
        ):
            raise ValueError(f"neutral evaluator identity changed reconstruction at fold {key}")
        if (
            projection.read_bytes(
                relative_score,
                maximum_bytes=_MAX_PROJECTION_FILE_BYTES,
            )
            != expected_score_bytes
        ):
            raise ValueError(f"projection score bytes changed during fold {key} audit")
        authenticated.append(evaluator)
    projection.revalidate()
    return AuthenticatedEvaluatorPanel(
        evaluators=tuple(authenticated),
        count_prior_reconstruction=reconstructed_count_priors,
        _capability=_EVALUATOR_PANEL_CAPABILITY,
    )


def aggregate_reconstructed_metrics(
    panel: AuthenticatedEvaluatorPanel,
) -> tuple[EqualFoldMetrics, ...]:
    """Equal-weight all four folds in the exact seven-method order."""

    if type(panel) is not AuthenticatedEvaluatorPanel:
        raise TypeError("panel must be an exact AuthenticatedEvaluatorPanel")
    for evaluator in panel.evaluators:
        evaluator.revalidate()
    result = tuple(
        aggregate_equal_folds(
            tuple(evaluator.methods[method_index] for evaluator in panel.evaluators)
        )
        for method_index in range(7)
    )
    expected = (
        "C0",
        "C0T",
        "R128-000250",
        "R128-000500",
        "R128-001000",
        "R128-002000",
        "R128-004000",
    )
    observed = tuple(
        value.method
        if value.checkpoint_step is None
        else f"{value.method}-{value.checkpoint_step:06d}"
        for value in result
    )
    if observed != expected:
        raise ValueError("equal-fold reconstruction changed the frozen method order")
    return result


def _fold_digest_map_sha256(
    contract: NativeDiffusionV1PilotContract,
    values: Mapping[str, str],
) -> str:
    return hashlib.sha256(contract.audit_fold_digest_map_bytes(values)).hexdigest()


def _checkpoint_digest_map_sha256(
    contract: NativeDiffusionV1PilotContract,
    values: Mapping[str, object],
) -> str:
    return hashlib.sha256(contract.audit_checkpoint_digest_map_bytes(values)).hexdigest()


def _cpu_reconstruction_document(
    contract: NativeDiffusionV1PilotContract,
    *,
    projection: AuthenticatedProjection,
    release_chain: AuthenticatedReleaseChain,
    evaluator_panel: AuthenticatedEvaluatorPanel,
    equal_metrics: Sequence[EqualFoldMetrics],
    preliminary_evaluation: PilotEvaluation,
) -> dict[str, object]:
    metrics = tuple(equal_metrics)
    if len(metrics) != 7 or any(type(value) is not EqualFoldMetrics for value in metrics):
        raise TypeError("CPU reconstruction requires seven exact equal-fold metrics")
    preliminary_evaluation.revalidate()
    evaluator_by_fold = evaluator_panel.by_fold()
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_cpu_reconstruction",
        "child_contract_sha256": contract.config_sha256,
        "parent_contract_sha256": contract.parent_config_sha256,
        "git_commit": release_chain.repository.git_commit,
        "projection_evidence": projection.pins.document(),
        "score_release_sha256": release_chain.release_sha256,
        "readiness_receipt_sha256_by_fold": {
            key: hashlib.sha256(release_chain.readiness_bytes_by_fold[key]).hexdigest()
            for key in _FOLD_KEYS
        },
        "trainer_bundle_sha256_by_fold": release_chain.trainer_digest_map(),
        "evaluator_bundle_sha256_by_fold": evaluator_panel.digest_map(),
        "count_prior_sha256_by_fold": release_chain.count_prior_digest_map(),
        "checkpoint_digest_by_fold_and_step": release_chain.checkpoint_digest_map(),
        "score_corruptions_sha256_by_fold": {
            key: evaluator_by_fold[key].score_corruptions_sha256 for key in _FOLD_KEYS
        },
        "score_residual_logits_sha256_by_fold": {
            key: evaluator_by_fold[key].score_residual_logits_sha256 for key in _FOLD_KEYS
        },
        "fold_metrics_sha256_by_fold": {
            key: evaluator_by_fold[key].fold_metrics_sha256 for key in _FOLD_KEYS
        },
        "equal_fold_metrics": [equal_fold_metrics_record(value) for value in metrics],
        "checkpoint_selection": checkpoint_selection_record(
            preliminary_evaluation.checkpoint_selection
        ),
        "comparator_selection": {"method": preliminary_evaluation.comparator_method},
        "bootstrap": bootstrap_record(preliminary_evaluation.bootstrap),
        "pilot_bootstrap_sha256": hashlib.sha256(preliminary_evaluation.npz_bytes()).hexdigest(),
        "checks": {
            "projection_tree_authenticated": True,
            "release_chain_rebuilt": True,
            "four_trainer_bundles_authenticated": True,
            "twenty_checkpoint_identities_authenticated": True,
            "four_count_priors_byte_reconstructed": True,
            "four_corruption_ledgers_byte_reconstructed": True,
            "four_count_logit_archives_reconstructed": True,
            "twenty_eight_fold_method_records_reconstructed": True,
            "louco_sufficient_statistics_reconstructed": True,
            "shared_bootstrap_10000_reconstructed": True,
            "checkpoint_selection_reconstructed": True,
        },
    }
    canonical_json_bytes(document)
    return document


def build_independent_verified_evidence(
    contract: NativeDiffusionV1PilotContract,
    *,
    projection: AuthenticatedProjection,
    release_chain: AuthenticatedReleaseChain,
    evaluator_panel: AuthenticatedEvaluatorPanel,
    equal_metrics: Sequence[EqualFoldMetrics],
    producer_reinference: AuthenticatedProducerReinference,
) -> tuple[VerifiedPilotEvidence, PilotEvaluation, bytes]:
    """Rebuild bootstrap/selection, mint evidence, then issue the scientific gate."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(projection) is not AuthenticatedProjection:
        raise TypeError("projection must be an exact AuthenticatedProjection")
    if type(release_chain) is not AuthenticatedReleaseChain:
        raise TypeError("release_chain must be an exact AuthenticatedReleaseChain")
    if type(evaluator_panel) is not AuthenticatedEvaluatorPanel:
        raise TypeError("evaluator_panel must be an exact AuthenticatedEvaluatorPanel")
    if type(producer_reinference) is not AuthenticatedProducerReinference:
        raise TypeError("producer_reinference must be authenticated")
    projection.revalidate()
    release_chain.revalidate(contract)
    evaluator_panel.revalidate(contract)
    if (
        evaluator_panel.count_prior_reconstruction.digest_map()
        != release_chain.count_prior_digest_map()
    ):
        raise ValueError("reconstructed count-prior panel changed before authorization")
    normalized_reinference = contract.producer_gpu_reinference_receipt_bytes(
        producer_reinference.document()
    )
    if normalized_reinference != producer_reinference.canonical_bytes:
        raise ValueError("producer reinference evidence changed before authorization")
    verify_producer_reinference_against_archives(
        producer_reinference,
        evaluator_panel.scoring_archive_map(),
    )
    metrics = tuple(equal_metrics)
    if len(metrics) != 7 or any(type(value) is not EqualFoldMetrics for value in metrics):
        raise TypeError("equal_metrics must contain seven exact aggregates")
    preliminary = evaluate_pilot_gate(
        metrics[2:],
        metrics[0],
        metrics[1],
        parent_contract_sha256=contract.parent_config_sha256,
    )
    cpu_document = _cpu_reconstruction_document(
        contract,
        projection=projection,
        release_chain=release_chain,
        evaluator_panel=evaluator_panel,
        equal_metrics=metrics,
        preliminary_evaluation=preliminary,
    )
    cpu_bytes = canonical_json_bytes(cpu_document)
    trainer_map = release_chain.trainer_digest_map()
    evaluator_map = evaluator_panel.digest_map()
    count_prior_map = release_chain.count_prior_digest_map()
    checkpoint_map = release_chain.checkpoint_digest_map()
    bindings = {
        "release_receipt_sha256": release_chain.release_sha256,
        "trainer_bundle_digest_map_sha256": _fold_digest_map_sha256(
            contract,
            trainer_map,
        ),
        "evaluator_bundle_digest_map_sha256": _fold_digest_map_sha256(
            contract,
            evaluator_map,
        ),
        "checkpoint_digest_map_sha256": _checkpoint_digest_map_sha256(
            contract,
            checkpoint_map,
        ),
        "count_prior_digest_map_sha256": _fold_digest_map_sha256(
            contract,
            count_prior_map,
        ),
        "producer_reinference_sha256": producer_reinference.sha256,
        "cpu_reconstruction_sha256": hashlib.sha256(cpu_bytes).hexdigest(),
    }
    evidence = build_verified_pilot_evidence(
        child_contract_sha256=contract.config_sha256,
        parent_contract_sha256=contract.parent_config_sha256,
        git_commit=release_chain.repository.git_commit,
        bindings=bindings,
        checks={
            "all_four_outer_fits": True,
            "release_chain_verified": True,
            "trainer_bundles_verified": True,
            "evaluator_bundles_verified": True,
            "count_priors_reconstructed": True,
            "checkpoints_authenticated": True,
            "producer_reinference_exact": True,
            "cpu_reconstruction_exact": True,
        },
    )
    evaluation = evaluate_verified_pilot_gate(
        metrics[2:],
        metrics[0],
        metrics[1],
        parent_contract_sha256=contract.parent_config_sha256,
        verified_evidence=evidence,
    )
    if (
        preliminary.checkpoint_selection != evaluation.checkpoint_selection
        or preliminary.comparator_method != evaluation.comparator_method
        or bootstrap_record(preliminary.bootstrap) != bootstrap_record(evaluation.bootstrap)
        or preliminary.npz_bytes() != evaluation.npz_bytes()
    ):
        raise ValueError("evidence-authorized gate changed the reconstructed numerical panel")
    projection.revalidate()
    release_chain.revalidate(contract)
    evaluator_panel.revalidate(contract)
    return evidence, evaluation, cpu_bytes


@dataclass(frozen=True, slots=True)
class IndependentPilotVerification:
    """Complete path-free audit result used to construct the independent receipt."""

    projection: AuthenticatedProjection = field(repr=False, compare=False)
    release_chain: AuthenticatedReleaseChain = field(repr=False, compare=False)
    evaluator_panel: AuthenticatedEvaluatorPanel = field(repr=False, compare=False)
    producer_reinference: AuthenticatedProducerReinference = field(
        repr=False,
        compare=False,
    )
    equal_metrics: tuple[EqualFoldMetrics, ...] = field(repr=False, compare=False)
    verified_evidence: VerifiedPilotEvidence
    evaluation: PilotEvaluation = field(repr=False, compare=False)
    cpu_reconstruction_bytes: bytes = field(repr=False, compare=False)
    pilot_bundle: AuthenticatedPilotBundle
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _VERIFICATION_CAPABILITY:
            raise RuntimeError("independent result requires internal verifier capability")
        if type(self.projection) is not AuthenticatedProjection:
            raise TypeError("result projection has an invalid type")
        if type(self.release_chain) is not AuthenticatedReleaseChain:
            raise TypeError("result release chain has an invalid type")
        if type(self.evaluator_panel) is not AuthenticatedEvaluatorPanel:
            raise TypeError("result evaluator panel has an invalid type")
        if type(self.producer_reinference) is not AuthenticatedProducerReinference:
            raise TypeError("result reinference evidence has an invalid type")
        if type(self.verified_evidence) is not VerifiedPilotEvidence:
            raise TypeError("result verified evidence has an invalid type")
        if type(self.evaluation) is not PilotEvaluation:
            raise TypeError("result pilot evaluation has an invalid type")
        if type(self.pilot_bundle) is not AuthenticatedPilotBundle:
            raise TypeError("result pilot bundle has an invalid type")
        if (
            type(self.equal_metrics) is not tuple
            or len(self.equal_metrics) != 7
            or any(type(value) is not EqualFoldMetrics for value in self.equal_metrics)
        ):
            raise TypeError("result equal metrics must contain seven exact values")
        if type(self.cpu_reconstruction_bytes) is not bytes:
            raise TypeError("CPU reconstruction must be exact canonical bytes")
        parsed = parse_canonical_json(
            self.cpu_reconstruction_bytes,
            label="CPU reconstruction",
        )
        if type(parsed) is not dict:
            raise TypeError("CPU reconstruction must be a canonical object")
        expected_cpu_sha = self.verified_evidence.bindings["cpu_reconstruction_sha256"]
        if hashlib.sha256(self.cpu_reconstruction_bytes).hexdigest() != expected_cpu_sha:
            raise ValueError("CPU reconstruction changed after evidence was minted")
        self.verified_evidence.revalidate()
        self.evaluation.revalidate()
        self.pilot_bundle.revalidate()
        if (
            self.pilot_bundle.verified_evidence_sha256 != self.verified_evidence.evidence_sha256
            or self.pilot_bundle.decision_status != self.evaluation.decision.status
        ):
            raise ValueError("pilot bundle differs from the independent decision")
        object.__setattr__(self, "_capability", None)

    def document(self) -> dict[str, object]:
        """Return every semantic field needed for the independent receipt."""

        if self._capability is not None:
            raise ValueError("independent-result capability state changed")
        evaluator_by_fold = self.evaluator_panel.by_fold()
        cpu = parse_canonical_json(
            self.cpu_reconstruction_bytes,
            label="CPU reconstruction",
        )
        if type(cpu) is not dict:  # pragma: no cover - construction invariant
            raise RuntimeError("CPU reconstruction lost its object type")
        return {
            "schema_version": 1,
            "artifact": "native_categorical_diffusion_v1_r128_independent_verification",
            "decision_status": self.evaluation.decision.status,
            "child_contract_sha256": self.verified_evidence.child_contract_sha256,
            "parent_contract_sha256": self.verified_evidence.parent_contract_sha256,
            "git_commit": self.verified_evidence.git_commit,
            "projection_evidence": self.projection.pins.document(),
            "trainer_bundle_sha256_by_fold": self.release_chain.trainer_digest_map(),
            "evaluator_bundle_sha256_by_fold": self.evaluator_panel.digest_map(),
            "pilot_bundle_sha256": self.pilot_bundle.bundle.tree_sha256,
            "count_prior_sha256_by_fold": self.release_chain.count_prior_digest_map(),
            "checkpoint_digest_by_fold_and_step": (self.release_chain.checkpoint_digest_map()),
            "producer_gpu_reinference": self.producer_reinference.document(),
            "cpu_reconstruction": cpu,
            "metrics": {
                "fold_metrics_sha256_by_fold": {
                    key: evaluator_by_fold[key].fold_metrics_sha256 for key in _FOLD_KEYS
                },
                "checkpoint_selection": checkpoint_selection_record(
                    self.evaluation.checkpoint_selection
                ),
                "comparator_selection": {"method": self.evaluation.comparator_method},
                "bootstrap": bootstrap_record(self.evaluation.bootstrap),
                "pilot_bootstrap_sha256": self.pilot_bundle.pilot_bootstrap_sha256,
                "pilot_metrics_sha256": self.pilot_bundle.pilot_metrics_sha256,
                "decision_sha256": self.pilot_bundle.decision_sha256,
            },
            "gates": gate_decision_record(self.evaluation.decision),
            "checks": {
                "all_four_outer_fits": True,
                "release_chain_verified": True,
                "distinct_producer_nodes_and_devices": True,
                "trainer_bundles_verified": True,
                "evaluator_bundles_verified": True,
                "count_priors_reconstructed": True,
                "corruption_ledgers_reconstructed": True,
                "count_logits_reconstructed": True,
                "checkpoint_physical_logical_metadata_identities_verified": True,
                "producer_reinference_exact": True,
                "bootstrap_and_selection_reconstructed": True,
                "pilot_gate_reconstructed": True,
                "pilot_bundle_reopened": True,
                "producer_imports_absent": True,
                "neural_forward_omitted": True,
            },
            "limitations": {
                "cpu_neural_forward_pass": "not_performed",
                "cross_backend_numeric_or_argmax_claim": "not_made",
                "checkpoint_tensors_forwarded": False,
            },
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())


@dataclass(frozen=True, slots=True)
class _StableInput:
    path: Path
    payload: bytes
    fingerprint: tuple[int, int, int, int, int, int, int]

    def revalidate(self, *, maximum_bytes: int, label: str) -> None:
        payload, fingerprint = _read_stable_regular_file(
            self.path,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        if payload != self.payload or fingerprint != self.fingerprint:
            raise ValueError(f"{label} changed during independent verification")


def _read_stable_regular_file(
    path: str | os.PathLike[str],
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive exact integer")
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_ancestors(absolute)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} must be a non-empty bounded single-link regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    fingerprint = _stat_fingerprint(before)
    if (
        fingerprint != _stat_fingerprint(after)
        or fingerprint != _stat_fingerprint(named)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload, fingerprint


def run_independent_pilot_verifier(
    *,
    child_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str],
    projection_root: str | os.PathLike[str],
    projection_evidence_pins: ProjectionEvidencePins,
    repository: RepositorySnapshot,
    expected_git_commit: str,
    trainer_roots_by_fold: Mapping[str, str | os.PathLike[str]],
    evaluator_roots_by_fold: Mapping[str, str | os.PathLike[str]],
    readiness_receipt_bytes_by_fold: Mapping[str, bytes],
    score_release_bytes: bytes,
    producer_reinference_evidence: bytes | Mapping[str, object],
    pilot_output_dir: str | os.PathLike[str],
) -> IndependentPilotVerification:
    """Run the complete no-forward audit and seal its evidence-authorized pilot."""

    commit = _git_commit(expected_git_commit)
    if type(repository) is not RepositorySnapshot or repository.git_commit != commit:
        raise ValueError("exact clean repository snapshot does not match the expected commit")
    child_bytes, child_fingerprint = _read_stable_regular_file(
        child_contract_path,
        maximum_bytes=262_144,
        label="pilot child contract",
    )
    parent_bytes, parent_fingerprint = _read_stable_regular_file(
        parent_contract_path,
        maximum_bytes=262_144,
        label="pilot parent contract",
    )
    child_input = _StableInput(
        path=Path(os.path.abspath(os.fspath(child_contract_path))),
        payload=child_bytes,
        fingerprint=child_fingerprint,
    )
    parent_input = _StableInput(
        path=Path(os.path.abspath(os.fspath(parent_contract_path))),
        payload=parent_bytes,
        fingerprint=parent_fingerprint,
    )
    contract = load_pilot_execution_v1_contract(
        child_input.path,
        parent_path=parent_input.path,
    )
    contract.revalidate()
    if (
        hashlib.sha256(child_bytes).hexdigest() != contract.config_sha256
        or hashlib.sha256(parent_bytes).hexdigest() != contract.parent_config_sha256
    ):
        raise ValueError("contract bytes changed across authentication")
    projection = authenticate_frozen_projection(
        contract,
        projection_root,
        evidence_pins=projection_evidence_pins,
    )
    release_chain = authenticate_release_chain(
        contract,
        repository=repository,
        expected_git_commit=commit,
        trainer_roots_by_fold=trainer_roots_by_fold,
        readiness_receipt_bytes_by_fold=readiness_receipt_bytes_by_fold,
        score_release_bytes=score_release_bytes,
    )
    count_priors = reconstruct_count_priors(
        contract,
        projection=projection,
        release_chain=release_chain,
    )
    evaluator_panel = authenticate_evaluator_panel(
        contract,
        projection=projection,
        release_chain=release_chain,
        reconstructed_count_priors=count_priors,
        evaluator_roots_by_fold=evaluator_roots_by_fold,
    )
    producer_reinference = authenticate_producer_reinference(
        contract,
        producer_reinference_evidence,
    )
    verify_producer_reinference_against_archives(
        producer_reinference,
        evaluator_panel.scoring_archive_map(),
    )
    equal_metrics = aggregate_reconstructed_metrics(evaluator_panel)
    verified_evidence, evaluation, cpu_bytes = build_independent_verified_evidence(
        contract,
        projection=projection,
        release_chain=release_chain,
        evaluator_panel=evaluator_panel,
        equal_metrics=equal_metrics,
        producer_reinference=producer_reinference,
    )
    fold_bundle_map = {
        key: {
            "trainer_bundle_sha256": release_chain.trainers_by_fold[key].bundle.tree_sha256,
            "evaluator_bundle_sha256": evaluator_panel.by_fold()[key].bundle.tree_sha256,
        }
        for key in _FOLD_KEYS
    }
    published = publish_pilot_bundle(
        contract,
        output_dir=pilot_output_dir,
        evaluator_bundles=evaluator_panel.evaluators,
        fold_bundle_sha256_by_fold=fold_bundle_map,
        equal_fold_metrics=equal_metrics,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
        child_contract_payload=child_bytes,
        parent_contract_payload=parent_bytes,
        repository=repository,
        expected_git_commit=commit,
    )
    pilot_bundle = authenticate_pilot_bundle(
        contract,
        published.bundle.root,
        evaluator_bundles=evaluator_panel.evaluators,
        fold_bundle_sha256_by_fold=fold_bundle_map,
        equal_fold_metrics=equal_metrics,
        evaluation=evaluation,
        verified_evidence=verified_evidence,
        child_contract_payload=child_bytes,
        parent_contract_payload=parent_bytes,
        repository=repository,
        expected_git_commit=commit,
        expected_tree_sha256=published.bundle.tree_sha256,
    )

    # Final bounded reopen of every protected input and output.  Checkpoint
    # tensors were authenticated inside trainer_bundle and are never returned
    # to, or forwarded by, this verifier.
    projection.revalidate()
    for key in _FOLD_KEYS:
        verify_bundle(
            contract,
            bundle_kind="trainer",
            root=trainer_roots_by_fold[key],
            expected_tree_sha256=release_chain.trainers_by_fold[key].bundle.tree_sha256,
        )
        verify_bundle(
            contract,
            bundle_kind="evaluator",
            root=evaluator_roots_by_fold[key],
            expected_tree_sha256=evaluator_panel.by_fold()[key].bundle.tree_sha256,
        )
    child_input.revalidate(maximum_bytes=262_144, label="pilot child contract")
    parent_input.revalidate(maximum_bytes=262_144, label="pilot parent contract")
    verify_bundle(
        contract,
        bundle_kind="pilot",
        root=pilot_bundle.bundle.root,
        expected_tree_sha256=pilot_bundle.bundle.tree_sha256,
        expected_tree_bytes=pilot_bundle.bundle.tree_bytes,
    )
    pilot_bundle.revalidate()
    return IndependentPilotVerification(
        projection=projection,
        release_chain=release_chain,
        evaluator_panel=evaluator_panel,
        producer_reinference=producer_reinference,
        equal_metrics=equal_metrics,
        verified_evidence=verified_evidence,
        evaluation=evaluation,
        cpu_reconstruction_bytes=cpu_bytes,
        pilot_bundle=pilot_bundle,
        _capability=_VERIFICATION_CAPABILITY,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently reconstruct and seal the native-diffusion v1 pilot",
    )
    parser.add_argument("--child-contract", required=True)
    parser.add_argument("--parent-contract", required=True)
    parser.add_argument("--projection-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    for fold in range(4):
        parser.add_argument(f"--trainer-{fold}", required=True)
        parser.add_argument(f"--evaluator-{fold}", required=True)
        parser.add_argument(f"--readiness-{fold}", required=True)
    parser.add_argument("--score-release", required=True)
    parser.add_argument("--producer-reinference-json", required=True)
    parser.add_argument("--pilot-output-dir", required=True)
    return parser


def _read_sealed_cli_input(path: str, *, label: str) -> bytes:
    payload, _ = _read_sealed_file(
        Path(os.path.abspath(path)),
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label=label,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commit = _git_commit(args.expected_git_commit)
    repository = build_repository_snapshot(
        args.repository_root,
        expected_commit=commit,
    )
    contract = load_pilot_execution_v1_contract(
        args.child_contract,
        parent_path=args.parent_contract,
    )
    result = run_independent_pilot_verifier(
        child_contract_path=args.child_contract,
        parent_contract_path=args.parent_contract,
        projection_root=args.projection_root,
        projection_evidence_pins=ProjectionEvidencePins.from_contract(contract),
        repository=repository,
        expected_git_commit=commit,
        trainer_roots_by_fold={str(fold): getattr(args, f"trainer_{fold}") for fold in range(4)},
        evaluator_roots_by_fold={
            str(fold): getattr(args, f"evaluator_{fold}") for fold in range(4)
        },
        readiness_receipt_bytes_by_fold={
            str(fold): _read_sealed_cli_input(
                getattr(args, f"readiness_{fold}"),
                label=f"readiness receipt fold {fold}",
            )
            for fold in range(4)
        },
        score_release_bytes=_read_sealed_cli_input(
            args.score_release,
            label="score-release receipt",
        ),
        producer_reinference_evidence=_read_sealed_cli_input(
            args.producer_reinference_json,
            label="producer reinference evidence",
        ),
        pilot_output_dir=args.pilot_output_dir,
    )
    sys.stdout.buffer.write(result.canonical_bytes())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Externally pinned exclusion authority for scientific/path peptide-GA calls.

The object-only GA helpers deliberately do not use this module and remain
unverified development fixtures.  A path entry authenticates both a complete
exclusion payload and a separately issued receipt from stable trusted inodes;
caller-supplied ``CollisionExclusions`` are only an expected-value cross-check.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.evaluation.evolutionary_kl_protocol import FROZEN_PROTOCOL_SHA256
from amp_challenge.generators.search.peptide_ga_records import (
    PEPTIDE_GA_MAX_ROUNDS,
    PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256,
    PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_BYTES,
    PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT,
    PEPTIDE_GA_TRAINING_EXCLUSION_MAX_BYTES,
    PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT,
    CollisionExclusions,
    PeptideGAError,
    canonical_json_bytes,
    sha256_bytes,
)

PEPTIDE_GA_EXCLUSION_ASSET_ARTIFACT = "fixed_default_peptide_ga_exclusion_asset_v1"
PEPTIDE_GA_EXCLUSION_RECEIPT_ARTIFACT = "fixed_default_peptide_ga_exclusion_receipt_v1"
PEPTIDE_GA_EXCLUSION_AUTHORITY_STATUS = (
    "externally_pinned_complete_development_only_not_execution_authority"
)
PEPTIDE_GA_EXCLUSION_ASSET_MAX_BYTES = (
    PEPTIDE_GA_TRAINING_EXCLUSION_MAX_BYTES + PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_BYTES + 16_384
)
PEPTIDE_GA_EXCLUSION_RECEIPT_MAX_BYTES = 65_536
PEPTIDE_GA_EXCLUSION_ISSUER_IDENTITY_HASH_DOMAIN = (
    b"amp/fixed-default-peptide-ga/exclusion-issuer-identity/v1\0"
)

_PATH_TEXT_MAX_LENGTH = 4096
_BUILTIN_PATH_TYPE = type(Path("."))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ASSET_KEYS = {
    "artifact",
    "authenticated_pre_wave_head_sha256",
    "authenticated_pre_wave_round_count",
    "campaign_id",
    "config_sha256",
    "execution_authorized",
    "phase",
    "policy_registry_sha256",
    "protocol_sha256",
    "schema_version",
    "scientific_evidence_accepted",
    "selection_policy_implementation_manifest_sha256",
    "submitted_count",
    "submitted_sequence_keys",
    "submitted_set_sha256",
    "training_count",
    "training_sequence_keys",
    "training_set_sha256",
    "truth_contract_sha256",
    "wave_id",
}
_RECEIPT_KEYS = {
    "artifact",
    "authenticated_pre_wave_head_sha256",
    "authenticated_pre_wave_round_count",
    "automatic_production_eligible",
    "campaign_id",
    "commit_state",
    "completeness_verified",
    "config_sha256",
    "execution_authorized",
    "exclusion_asset_sha256",
    "issuer_git_commit",
    "issuer_id",
    "issuer_identity_sha256",
    "issuer_source_inventory_sha256",
    "phase",
    "policy_registry_sha256",
    "protocol_sha256",
    "runtime_environment_sha256",
    "schema_version",
    "scientific_evidence_accepted",
    "selection_policy_implementation_manifest_sha256",
    "status",
    "submitted_count",
    "submitted_inventory_complete_through_pre_wave_head",
    "submitted_set_sha256",
    "submitted_truth_receipt_sha256",
    "training_count",
    "training_inventory_complete",
    "training_set_sha256",
    "training_truth_receipt_sha256",
    "truth_contract_sha256",
    "truth_state",
    "wave_id",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeptideGAError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _identifier(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    assert isinstance(value, str)
    return value


def _bounded_integer(value: object, *, maximum: int, label: str) -> int:
    _require(type(value) is int and 0 <= value <= maximum, f"{label} differs")
    assert isinstance(value, int)
    return value


def _exact_path(value: object, *, label: str) -> Path:
    _require(
        type(value) is str or type(value) is _BUILTIN_PATH_TYPE,
        f"{label} must use an exact built-in path type",
    )
    text = value if type(value) is str else str(value)
    assert isinstance(text, str)
    _require(0 < len(text) <= _PATH_TEXT_MAX_LENGTH and "\0" not in text, f"{label} differs")
    result = Path(text)
    _require(".." not in result.parts, f"{label} must not contain parent traversal")
    return result


def _resolved_path(value: Path, *, label: str) -> Path:
    try:
        return value.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PeptideGAError(f"{label} resolution failed") from error


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _strict_object(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict and set(value) == keys, f"{label} schema differs")
    assert isinstance(value, dict)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PeptideGAError("exclusion authority JSON has duplicate keys")
        result[key] = value
    return result


def exclusion_issuer_identity_document(
    *,
    issuer_id: str,
    issuer_git_commit: str,
    issuer_source_inventory_sha256: str,
    runtime_environment_sha256: str,
) -> dict[str, str]:
    _identifier(issuer_id, label="exclusion issuer ID")
    _require(
        type(issuer_git_commit) is str and _GIT_COMMIT_RE.fullmatch(issuer_git_commit) is not None,
        "exclusion issuer Git commit differs",
    )
    return {
        "issuer_git_commit": issuer_git_commit,
        "issuer_id": issuer_id,
        "issuer_source_inventory_sha256": _sha256(
            issuer_source_inventory_sha256,
            label="exclusion issuer source inventory digest",
        ),
        "runtime_environment_sha256": _sha256(
            runtime_environment_sha256,
            label="exclusion issuer runtime environment digest",
        ),
    }


def derive_exclusion_issuer_identity_sha256(
    *,
    issuer_id: str,
    issuer_git_commit: str,
    issuer_source_inventory_sha256: str,
    runtime_environment_sha256: str,
) -> str:
    return sha256_bytes(
        PEPTIDE_GA_EXCLUSION_ISSUER_IDENTITY_HASH_DOMAIN
        + canonical_json_bytes(
            exclusion_issuer_identity_document(
                issuer_id=issuer_id,
                issuer_git_commit=issuer_git_commit,
                issuer_source_inventory_sha256=issuer_source_inventory_sha256,
                runtime_environment_sha256=runtime_environment_sha256,
            )
        )
    )


def _read_one_from_dirfd(
    directory_fd: int,
    *,
    name: str,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        entry_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise PeptideGAError(f"trusted {label} open failed") from error
    try:
        before = os.fstat(file_fd)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_uid == os.geteuid()
            and before.st_mode & 0o022 == 0
            and 0 < before.st_size <= maximum_bytes,
            f"trusted {label} metadata differs",
        )
        _require(
            _stat_identity(entry_before) == _stat_identity(before),
            f"trusted {label} changed before reading",
        )
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            chunk = os.read(file_fd, min(65_536, maximum_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        _require(0 < len(payload) <= maximum_bytes, f"trusted {label} size differs")
        after = os.fstat(file_fd)
        _require(
            _stat_identity(before) == _stat_identity(after),
            f"trusted {label} changed while reading",
        )
        entry_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(
            _stat_identity(after) == _stat_identity(entry_after),
            f"trusted {label} entry changed while reading",
        )
    except OSError as error:
        raise PeptideGAError(f"trusted {label} recheck failed") from error
    finally:
        os.close(file_fd)
    return bytes(payload), after


def _read_trusted_exclusion_sources(
    asset_path: str | Path,
    receipt_path: str | Path,
    *,
    trusted_parent: str | Path,
    campaign_root: str | Path,
    expected_asset_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[bytes, bytes]:
    _sha256(expected_asset_sha256, label="expected exclusion asset digest")
    _sha256(expected_receipt_sha256, label="expected exclusion receipt digest")
    parent = _exact_path(trusted_parent, label="trusted exclusion authority parent")
    asset = _exact_path(asset_path, label="trusted exclusion asset path")
    receipt = _exact_path(receipt_path, label="trusted exclusion receipt path")
    root = _exact_path(campaign_root, label="campaign root")
    _require(
        parent.is_absolute()
        and asset.is_absolute()
        and receipt.is_absolute()
        and root.is_absolute(),
        "exclusion authority and campaign paths must be absolute",
    )
    _require(
        asset.parent == parent and receipt.parent == parent and asset.name != receipt.name,
        "exclusion authority files must be distinct direct trusted-parent children",
    )
    resolved_parent = _resolved_path(parent, label="trusted exclusion authority parent")
    resolved_asset = _resolved_path(asset, label="trusted exclusion asset path")
    resolved_receipt = _resolved_path(receipt, label="trusted exclusion receipt path")
    resolved_root = _resolved_path(root, label="campaign root")
    _require(
        not parent.is_relative_to(root)
        and not asset.is_relative_to(root)
        and not receipt.is_relative_to(root)
        and not resolved_parent.is_relative_to(resolved_root)
        and not resolved_asset.is_relative_to(resolved_root)
        and not resolved_receipt.is_relative_to(resolved_root),
        "exclusion authority must remain outside the campaign output root",
    )
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as error:
        raise PeptideGAError("trusted exclusion authority parent stat failed") from error
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and parent_stat.st_uid == os.geteuid()
        and parent_stat.st_mode & 0o022 == 0,
        "trusted exclusion authority parent metadata differs",
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as error:
        raise PeptideGAError("trusted exclusion authority parent open failed") from error
    try:
        directory_before = os.fstat(directory_fd)
        _require(
            _stat_identity(parent_stat) == _stat_identity(directory_before),
            "trusted exclusion authority parent changed before reading",
        )
        asset_bytes, asset_stat = _read_one_from_dirfd(
            directory_fd,
            name=asset.name,
            maximum_bytes=PEPTIDE_GA_EXCLUSION_ASSET_MAX_BYTES,
            label="exclusion asset",
        )
        receipt_bytes, receipt_stat = _read_one_from_dirfd(
            directory_fd,
            name=receipt.name,
            maximum_bytes=PEPTIDE_GA_EXCLUSION_RECEIPT_MAX_BYTES,
            label="exclusion receipt",
        )
        _require(
            (asset_stat.st_dev, asset_stat.st_ino) != (receipt_stat.st_dev, receipt_stat.st_ino),
            "exclusion authority files share one inode",
        )
        directory_after = os.fstat(directory_fd)
        _require(
            _stat_identity(directory_before) == _stat_identity(directory_after),
            "trusted exclusion authority parent changed while reading",
        )
    finally:
        os.close(directory_fd)
    _require(
        sha256_bytes(asset_bytes) == expected_asset_sha256,
        "expected exclusion asset digest differs",
    )
    _require(
        sha256_bytes(receipt_bytes) == expected_receipt_sha256,
        "expected exclusion receipt digest differs",
    )
    return asset_bytes, receipt_bytes


@dataclass(frozen=True, slots=True)
class AuthenticatedCollisionExclusionAuthority:
    """Complete exclusions authenticated by external bytes and external digest pins."""

    exclusions: CollisionExclusions
    asset_sha256: str
    receipt_sha256: str
    issuer_identity_sha256: str
    training_truth_receipt_sha256: str
    submitted_truth_receipt_sha256: str
    authentication_status: str
    execution_authorized: bool = False
    scientific_evidence_accepted: bool = False

    def __post_init__(self) -> None:
        _require(type(self.exclusions) is CollisionExclusions, "authenticated exclusions differ")
        self.exclusions.__post_init__()
        for value, label in (
            (self.asset_sha256, "authenticated exclusion asset digest"),
            (self.receipt_sha256, "authenticated exclusion receipt digest"),
            (self.issuer_identity_sha256, "authenticated exclusion issuer digest"),
            (self.training_truth_receipt_sha256, "training truth receipt digest"),
            (self.submitted_truth_receipt_sha256, "submitted truth receipt digest"),
        ):
            _sha256(value, label=label)
        _require(
            type(self.authentication_status) is str
            and self.authentication_status == PEPTIDE_GA_EXCLUSION_AUTHORITY_STATUS,
            "exclusion authority authentication status differs",
        )
        _require(
            self.execution_authorized is False and self.scientific_evidence_accepted is False,
            "exclusion authority cannot grant execution or evidence",
        )


def load_collision_exclusion_authority_from_paths(
    asset_path: str | Path,
    receipt_path: str | Path,
    *,
    trusted_parent: str | Path,
    campaign_root: str | Path,
    expected_asset_sha256: str,
    expected_receipt_sha256: str,
    expected_issuer_identity_sha256: str,
    campaign_id: str,
    phase: str,
    wave_id: str,
    authenticated_pre_wave_head_sha256: str,
    authenticated_pre_wave_round_count: int,
    protocol_sha256: str,
    config_sha256: str,
    policy_registry_sha256: str,
    selection_policy_implementation_manifest_sha256: str,
    truth_contract_sha256: str,
) -> AuthenticatedCollisionExclusionAuthority:
    """Authenticate complete exclusions against external path and identity pins."""

    _sha256(expected_issuer_identity_sha256, label="expected exclusion issuer identity digest")
    _identifier(campaign_id, label="exclusion campaign ID")
    _require(type(phase) is str and phase in {"screen", "confirmation"}, "exclusion phase differs")
    _identifier(wave_id, label="exclusion wave ID")
    _sha256(authenticated_pre_wave_head_sha256, label="exclusion pre-wave head")
    _bounded_integer(
        authenticated_pre_wave_round_count,
        maximum=PEPTIDE_GA_MAX_ROUNDS,
        label="exclusion pre-wave round count",
    )
    for value, label in (
        (protocol_sha256, "exclusion protocol digest"),
        (config_sha256, "exclusion config digest"),
        (policy_registry_sha256, "exclusion policy registry digest"),
        (
            selection_policy_implementation_manifest_sha256,
            "exclusion selection implementation manifest digest",
        ),
        (truth_contract_sha256, "exclusion truth contract digest"),
    ):
        _sha256(value, label=label)
    _require(protocol_sha256 == FROZEN_PROTOCOL_SHA256, "exclusion frozen protocol differs")
    _require(
        selection_policy_implementation_manifest_sha256
        == PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256,
        "exclusion selection implementation manifest differs",
    )
    asset_bytes, receipt_bytes = _read_trusted_exclusion_sources(
        asset_path,
        receipt_path,
        trusted_parent=trusted_parent,
        campaign_root=campaign_root,
        expected_asset_sha256=expected_asset_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    try:
        asset_value = json.loads(asset_bytes, object_pairs_hook=_reject_duplicate_keys)
        receipt_value = json.loads(receipt_bytes, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PeptideGAError("exclusion authority JSON is invalid") from error
    asset = _strict_object(asset_value, _ASSET_KEYS, label="exclusion asset")
    receipt = _strict_object(receipt_value, _RECEIPT_KEYS, label="exclusion receipt")
    _require(
        canonical_json_bytes(asset_value) + b"\n" == asset_bytes
        and canonical_json_bytes(receipt_value) + b"\n" == receipt_bytes,
        "exclusion authority is not canonical JSON",
    )
    common_expected: dict[str, object] = {
        "authenticated_pre_wave_head_sha256": authenticated_pre_wave_head_sha256,
        "authenticated_pre_wave_round_count": authenticated_pre_wave_round_count,
        "campaign_id": campaign_id,
        "config_sha256": config_sha256,
        "phase": phase,
        "policy_registry_sha256": policy_registry_sha256,
        "protocol_sha256": protocol_sha256,
        "selection_policy_implementation_manifest_sha256": (
            selection_policy_implementation_manifest_sha256
        ),
        "truth_contract_sha256": truth_contract_sha256,
        "wave_id": wave_id,
    }
    _require(
        all(
            asset.get(key) == value and type(asset.get(key)) is type(value)
            for key, value in common_expected.items()
        ),
        "exclusion asset campaign binding differs",
    )
    _require(
        all(
            receipt.get(key) == value and type(receipt.get(key)) is type(value)
            for key, value in common_expected.items()
        ),
        "exclusion receipt campaign binding differs",
    )
    _require(
        asset.get("artifact") == PEPTIDE_GA_EXCLUSION_ASSET_ARTIFACT
        and type(asset.get("schema_version")) is int
        and asset.get("schema_version") == 1
        and asset.get("execution_authorized") is False
        and asset.get("scientific_evidence_accepted") is False,
        "exclusion asset authority fields differ",
    )
    training = asset.get("training_sequence_keys")
    submitted = asset.get("submitted_sequence_keys")
    training_count = _bounded_integer(
        asset.get("training_count"),
        maximum=PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT,
        label="exclusion asset training count",
    )
    submitted_count = _bounded_integer(
        asset.get("submitted_count"),
        maximum=PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT,
        label="exclusion asset submitted count",
    )
    _require(
        type(training) is list and len(training) == training_count,
        "exclusion asset training inventory differs",
    )
    _require(
        type(submitted) is list and len(submitted) == submitted_count,
        "exclusion asset submitted inventory differs",
    )
    for label, values in (("training", training), ("submitted", submitted)):
        assert isinstance(values, list)
        previous: str | None = None
        for value in values:
            key = _sha256(value, label=f"exclusion asset {label} sequence key")
            _require(previous is None or previous < key, f"exclusion asset {label} keys differ")
            previous = key
    assert isinstance(training, list) and isinstance(submitted, list)
    exclusions = CollisionExclusions.from_keys(
        training_sequence_keys=tuple(training),
        submitted_sequence_keys=tuple(submitted),
    )
    _require(
        asset.get("training_set_sha256") == exclusions.training_set_sha256
        and asset.get("submitted_set_sha256") == exclusions.submitted_set_sha256,
        "exclusion asset payload digest differs",
    )
    _require(
        receipt.get("artifact") == PEPTIDE_GA_EXCLUSION_RECEIPT_ARTIFACT
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("status") == PEPTIDE_GA_EXCLUSION_AUTHORITY_STATUS
        and receipt.get("exclusion_asset_sha256") == expected_asset_sha256,
        "exclusion receipt identity differs",
    )
    _require(
        receipt.get("training_count") == exclusions.training_count
        and type(receipt.get("training_count")) is int
        and receipt.get("submitted_count") == exclusions.submitted_count
        and type(receipt.get("submitted_count")) is int
        and receipt.get("training_set_sha256") == exclusions.training_set_sha256
        and receipt.get("submitted_set_sha256") == exclusions.submitted_set_sha256,
        "exclusion receipt payload binding differs",
    )
    _require(
        receipt.get("completeness_verified") is True
        and receipt.get("training_inventory_complete") is True
        and receipt.get("submitted_inventory_complete_through_pre_wave_head") is True
        and receipt.get("truth_state") == "externally_attested_complete_exclusion_inventory"
        and receipt.get("commit_state") == "sealed_external_asset"
        and receipt.get("execution_authorized") is False
        and receipt.get("scientific_evidence_accepted") is False
        and receipt.get("automatic_production_eligible") is False,
        "exclusion receipt completeness/truth state differs",
    )
    issuer_identity_sha256 = derive_exclusion_issuer_identity_sha256(
        issuer_id=receipt.get("issuer_id"),  # type: ignore[arg-type]
        issuer_git_commit=receipt.get("issuer_git_commit"),  # type: ignore[arg-type]
        issuer_source_inventory_sha256=receipt.get("issuer_source_inventory_sha256"),  # type: ignore[arg-type]
        runtime_environment_sha256=receipt.get("runtime_environment_sha256"),  # type: ignore[arg-type]
    )
    _require(
        receipt.get("issuer_identity_sha256") == issuer_identity_sha256
        and issuer_identity_sha256 == expected_issuer_identity_sha256,
        "exclusion receipt issuer identity differs",
    )
    training_truth_receipt_sha256 = _sha256(
        receipt.get("training_truth_receipt_sha256"),
        label="training truth receipt digest",
    )
    submitted_truth_receipt_sha256 = _sha256(
        receipt.get("submitted_truth_receipt_sha256"),
        label="submitted truth receipt digest",
    )
    result = AuthenticatedCollisionExclusionAuthority(
        exclusions=exclusions,
        asset_sha256=expected_asset_sha256,
        receipt_sha256=expected_receipt_sha256,
        issuer_identity_sha256=issuer_identity_sha256,
        training_truth_receipt_sha256=training_truth_receipt_sha256,
        submitted_truth_receipt_sha256=submitted_truth_receipt_sha256,
        authentication_status=PEPTIDE_GA_EXCLUSION_AUTHORITY_STATUS,
    )
    result.__post_init__()
    return result


__all__ = [
    "PEPTIDE_GA_EXCLUSION_ASSET_ARTIFACT",
    "PEPTIDE_GA_EXCLUSION_AUTHORITY_STATUS",
    "PEPTIDE_GA_EXCLUSION_RECEIPT_ARTIFACT",
    "AuthenticatedCollisionExclusionAuthority",
    "derive_exclusion_issuer_identity_sha256",
    "exclusion_issuer_identity_document",
    "load_collision_exclusion_authority_from_paths",
]

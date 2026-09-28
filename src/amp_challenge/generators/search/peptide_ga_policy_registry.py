"""Trusted-path policy registry for the fixed-default peptide-GA lifecycle."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from amp_challenge.generators.search import peptide_ga_selection_policy_impl_v1
from amp_challenge.generators.search.peptide_ga_records import (
    PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT,
    PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT,
    PEPTIDE_GA_MAX_ROUNDS,
    PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
    PEPTIDE_GA_POLICY_REGISTRY_MAX_ROWS,
    PEPTIDE_GA_SELECTION_ELIGIBLE_MAX_COUNT,
    PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256,
    PEPTIDE_GA_SELECTION_POLICY_SOURCE_PATH,
    PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256,
    PEPTIDE_GA_SIGNED_63_MAX,
    AuthenticatedSelectionPolicyRegistry,
    PeptideGAError,
    SelectionPolicyRegistryRow,
    canonical_json_bytes,
    derive_wave_selection_set_id,
    ordered_eligible_proposal_ids_sha256,
    sha256_bytes,
)

_PATH_TEXT_MAX_LENGTH = 4096
_BUILTIN_PATH_TYPE = type(Path("."))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = {
    "config_sha256",
    "namespace",
    "phase",
    "proposal_policy_version",
    "round_end",
    "round_start",
    "selection_id_derivation",
    "selection_policy_entry_point",
    "selection_policy_implementation_sha256",
    "selection_policy_version",
}
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CAMPAIGN_GENESIS_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-genesis/v1\0"
_SELECTION_BINDING_PARAMETER_KEYS = {
    "selection_batch_output_sha256",
    "selection_campaign_id",
    "selection_config_sha256",
    "selection_id_derivation",
    "selection_namespace",
    "selection_ordered_eligible_sha256",
    "selection_policy_implementation_sha256",
    "selection_pre_wave_head_sha256",
    "selection_pre_wave_round_count",
    "selection_wave_id",
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


def _read_trusted_registry_source(
    registry_path: str | Path,
    *,
    trusted_registry_parent: str | Path,
    expected_registry_sha256: str,
) -> bytes:
    """Read one stable, owned registry inode under an owned trusted directory."""

    _sha256(expected_registry_sha256, label="expected policy registry digest")
    parent = _exact_path(trusted_registry_parent, label="trusted policy registry parent")
    target = _exact_path(registry_path, label="trusted policy registry path")
    _require(
        parent.is_absolute() and target.is_absolute(), "policy registry paths must be absolute"
    )
    _require(target.parent == parent, "policy registry is not a direct trusted-parent child")
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as error:
        raise PeptideGAError("trusted policy registry parent stat failed") from error
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and parent_stat.st_uid == os.geteuid()
        and parent_stat.st_mode & 0o022 == 0,
        "trusted policy registry parent metadata differs",
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as error:
        raise PeptideGAError("trusted policy registry parent open failed") from error
    try:
        directory_before = os.fstat(directory_fd)
        _require(
            _stat_identity(parent_stat) == _stat_identity(directory_before),
            "trusted policy registry parent changed before opening",
        )
        try:
            entry_before = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
            file_fd = os.open(target.name, file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise PeptideGAError("trusted policy registry open failed") from error
        try:
            before = os.fstat(file_fd)
            _require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and before.st_uid == os.geteuid()
                and before.st_mode & 0o022 == 0
                and 0 < before.st_size <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
                "trusted policy registry metadata differs",
            )
            _require(
                _stat_identity(entry_before) == _stat_identity(before),
                "trusted policy registry entry changed before reading",
            )
            payload_buffer = bytearray()
            while len(payload_buffer) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES:
                chunk = os.read(
                    file_fd,
                    min(
                        65_536,
                        PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES + 1 - len(payload_buffer),
                    ),
                )
                if not chunk:
                    break
                payload_buffer.extend(chunk)
            _require(
                0 < len(payload_buffer) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
                "trusted policy registry size differs",
            )
            after = os.fstat(file_fd)
            _require(
                _stat_identity(before) == _stat_identity(after),
                "trusted policy registry changed while reading",
            )
            try:
                entry_after = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise PeptideGAError("trusted policy registry entry recheck failed") from error
            _require(
                _stat_identity(after) == _stat_identity(entry_after),
                "trusted policy registry entry changed while reading",
            )
        finally:
            os.close(file_fd)
        directory_after = os.fstat(directory_fd)
        _require(
            _stat_identity(directory_before) == _stat_identity(directory_after),
            "trusted policy registry parent changed while reading",
        )
    finally:
        os.close(directory_fd)
    payload = bytes(payload_buffer)
    _require(
        sha256_bytes(payload) == expected_registry_sha256,
        "expected policy registry digest differs",
    )
    return payload


def _verify_selection_policy_implementation_source() -> tuple[str, str]:
    """Authenticate the complete source file used by both executable policies."""

    source_value = peptide_ga_selection_policy_impl_v1.__file__
    _require(type(source_value) is str, "selection implementation source path differs")
    source_path = _exact_path(source_value, label="selection implementation source path")
    _require(source_path.is_absolute(), "selection implementation source path must be absolute")
    _require(
        source_path.as_posix().endswith("/" + PEPTIDE_GA_SELECTION_POLICY_SOURCE_PATH),
        "selection implementation source location differs",
    )
    parent = source_path.parent
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as error:
        raise PeptideGAError("selection implementation parent stat failed") from error
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and parent_stat.st_uid == os.geteuid()
        and parent_stat.st_mode & 0o022 == 0,
        "selection implementation parent metadata differs",
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as error:
        raise PeptideGAError("selection implementation parent open failed") from error
    try:
        directory_before = os.fstat(directory_fd)
        _require(
            _stat_identity(parent_stat) == _stat_identity(directory_before),
            "selection implementation parent changed before opening",
        )
        try:
            entry_before = os.stat(source_path.name, dir_fd=directory_fd, follow_symlinks=False)
            file_fd = os.open(source_path.name, file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise PeptideGAError("selection implementation source open failed") from error
        try:
            before = os.fstat(file_fd)
            _require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and before.st_uid == os.geteuid()
                and before.st_mode & 0o022 == 0
                and 0 < before.st_size <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
                "selection implementation source metadata differs",
            )
            _require(
                _stat_identity(entry_before) == _stat_identity(before),
                "selection implementation source changed before reading",
            )
            payload_buffer = bytearray()
            while len(payload_buffer) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES:
                chunk = os.read(
                    file_fd,
                    min(
                        65_536,
                        PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES + 1 - len(payload_buffer),
                    ),
                )
                if not chunk:
                    break
                payload_buffer.extend(chunk)
            _require(
                0 < len(payload_buffer) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
                "selection implementation source size differs",
            )
            after = os.fstat(file_fd)
            _require(
                _stat_identity(before) == _stat_identity(after),
                "selection implementation source changed while reading",
            )
            entry_after = os.stat(source_path.name, dir_fd=directory_fd, follow_symlinks=False)
            _require(
                _stat_identity(after) == _stat_identity(entry_after),
                "selection implementation source entry changed while reading",
            )
        except OSError as error:
            raise PeptideGAError("selection implementation source recheck failed") from error
        finally:
            os.close(file_fd)
        directory_after = os.fstat(directory_fd)
        _require(
            _stat_identity(directory_before) == _stat_identity(directory_after),
            "selection implementation parent changed while reading",
        )
    finally:
        os.close(directory_fd)
    payload = bytes(payload_buffer)
    source_sha256 = sha256_bytes(payload)
    _require(
        source_sha256 == PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256,
        "selection implementation source digest differs",
    )
    _require(
        peptide_ga_selection_policy_impl_v1.bootstrap_selection_membership.__name__
        == PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT
        and peptide_ga_selection_policy_impl_v1.controller_first_available_prefix_positions.__name__
        == PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT,
        "selection implementation entry points differ",
    )
    return source_sha256, PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256


def load_selection_policy_registry_from_path(
    registry_path: str | Path,
    *,
    trusted_registry_parent: str | Path,
    expected_registry_sha256: str,
) -> AuthenticatedSelectionPolicyRegistry:
    """Authenticate, parse, and canonicalize the bounded policy registry."""

    payload = _read_trusted_registry_source(
        registry_path,
        trusted_registry_parent=trusted_registry_parent,
        expected_registry_sha256=expected_registry_sha256,
    )

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            _require(type(key) is str and key not in result, "policy registry duplicates JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise PeptideGAError("policy registry is not strict canonical JSON") from error
    _require(
        type(value) is dict
        and set(value)
        == {
            "artifact",
            "execution_authorized",
            "rows",
            "schema_version",
            "scientific_evidence_accepted",
            "status",
        },
        "policy registry schema differs",
    )
    assert isinstance(value, dict)
    _require(
        value["artifact"] == "fixed_default_peptide_ga_policy_registry_v1"
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == "development_only_not_execution_authority"
        and value["execution_authorized"] is False
        and value["scientific_evidence_accepted"] is False,
        "policy registry authority fields differ",
    )
    rows = value["rows"]
    _require(
        type(rows) is list and 0 < len(rows) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_ROWS,
        "policy registry row inventory differs",
    )
    parsed_rows: list[SelectionPolicyRegistryRow] = []
    for row in rows:
        _require(type(row) is dict and set(row) == _ROW_KEYS, "policy registry row schema differs")
        assert isinstance(row, dict)
        parsed_rows.append(
            SelectionPolicyRegistryRow(
                namespace=row["namespace"],
                phase=row["phase"],
                round_start=row["round_start"],
                round_end=row["round_end"],
                proposal_policy_version=row["proposal_policy_version"],
                selection_policy_version=row["selection_policy_version"],
                selection_policy_entry_point=row["selection_policy_entry_point"],
                selection_policy_implementation_sha256=(
                    row["selection_policy_implementation_sha256"]
                ),
                config_sha256=row["config_sha256"],
                selection_id_derivation=row["selection_id_derivation"],
            )
        )
    _require(
        canonical_json_bytes(value) + b"\n" == payload,
        "policy registry is not canonical JSON",
    )
    implementation_source_sha256, implementation_manifest_sha256 = (
        _verify_selection_policy_implementation_source()
    )
    return AuthenticatedSelectionPolicyRegistry(
        rows=tuple(parsed_rows),
        registry_sha256=expected_registry_sha256,
        registry_source_bytes=payload,
        implementation_source_sha256=implementation_source_sha256,
        implementation_manifest_sha256=implementation_manifest_sha256,
        authentication_status="verified_trusted_path",
        implementation_authentication_status="verified_complete_source_bytes",
    )


def validate_campaign_proposal_policy_binding(
    *,
    registry: AuthenticatedSelectionPolicyRegistry,
    config_sha256: str,
    campaign_id: str,
    phase: str,
    header_sha256: str,
    round_seals: tuple[str, ...],
    round_index: int,
    proposal_record: dict[str, object],
    edge_record: dict[str, object],
) -> tuple[str, str, str]:
    """Validate one history proposal against exactly one authenticated registry row."""

    _require(
        type(registry) is AuthenticatedSelectionPolicyRegistry,
        "policy registry type differs",
    )
    registry.__post_init__()
    _sha256(config_sha256, label="history config digest")
    _identifier(campaign_id, label="history campaign ID")
    _identifier(phase, label="history phase")
    _sha256(header_sha256, label="history header digest")
    _bounded_integer(round_index, maximum=PEPTIDE_GA_MAX_ROUNDS, label="history round index")
    _require(
        type(round_seals) is tuple and 0 < len(round_seals) <= PEPTIDE_GA_MAX_ROUNDS,
        "history round seal inventory differs",
    )
    for digest in round_seals:
        _sha256(digest, label="history round seal")
    _require(round_index < len(round_seals), "history proposal round is unsealed")
    _require(type(proposal_record) is dict, "history proposal record differs")
    _require(type(edge_record) is dict, "history edge record differs")
    selection = proposal_record.get("selection")
    _require(type(selection) is dict, "history proposal selection differs")
    assert isinstance(selection, dict)
    namespace = _identifier(proposal_record.get("niche_id"), label="history namespace")
    proposal_policy = _identifier(
        proposal_record.get("policy_version"),
        label="history proposal policy",
    )
    proposal_id = _identifier(
        proposal_record.get("proposal_id"),
        label="history proposal ID",
    )
    proposal_round = _bounded_integer(
        proposal_record.get("proposal_round"),
        maximum=PEPTIDE_GA_MAX_ROUNDS,
        label="history proposal round",
    )
    _require(proposal_round == round_index, "history proposal round differs")
    selection_policy = _identifier(
        selection.get("policy_version"),
        label="history selection policy",
    )
    selection_seed = _bounded_integer(
        selection.get("seed"),
        maximum=PEPTIDE_GA_SIGNED_63_MAX,
        label="history selection seed",
    )
    selected = selection.get("selected")
    _require(type(selected) is bool, "history selection flag differs")
    eligible = selection.get("eligible_proposal_ids")
    _require(
        type(eligible) is list and 0 < len(eligible) <= PEPTIDE_GA_SELECTION_ELIGIBLE_MAX_COUNT,
        "history eligible proposal inventory differs",
    )
    assert isinstance(eligible, list)
    eligible_tuple = tuple(
        _identifier(value, label="history eligible proposal ID") for value in eligible
    )
    eligible_sha256 = ordered_eligible_proposal_ids_sha256(eligible_tuple)
    sampling_parameters = edge_record.get("sampling_parameters")
    _require(
        type(sampling_parameters) is list and len(sampling_parameters) <= 32,
        "history sampling parameter inventory differs",
    )
    assert isinstance(sampling_parameters, list)
    parameters: dict[str, object] = {}
    for parameter in sampling_parameters:
        _require(
            type(parameter) is list and len(parameter) == 2,
            "history sampling parameter row differs",
        )
        assert isinstance(parameter, list)
        name = _identifier(parameter[0], label="history sampling parameter name")
        _require(name not in parameters, "history sampling parameter names are not unique")
        _require(
            "private" not in name.lower() and "reserve" not in name.lower(),
            "history selection metadata exposes private reserve state",
        )
        parameters[name] = parameter[1]
    _require(
        {name for name in parameters if name.startswith("selection_")}
        == _SELECTION_BINDING_PARAMETER_KEYS,
        "history selection binding parameters differ",
    )
    row = registry.matching_row(
        namespace=namespace,
        phase=phase,
        proposal_round=proposal_round,
        proposal_policy_version=proposal_policy,
        selection_policy_version=selection_policy,
        config_sha256=config_sha256,
    )
    if row.selection_policy_entry_point == PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT:
        try:
            bootstrap_membership = (
                peptide_ga_selection_policy_impl_v1.bootstrap_selection_membership(
                    proposal_id=proposal_id,
                    eligible_proposal_ids=eligible_tuple,
                    selected=selected,
                )
            )
        except ValueError as error:
            raise PeptideGAError("bootstrap executable selection policy rejected input") from error
        _require(bootstrap_membership, "bootstrap executable selection policy differs")
    else:
        _require(
            row.selection_policy_entry_point == PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT
            and selected
            and proposal_id in eligible_tuple,
            "controller executable selection membership differs",
        )
    expected_pre_wave_head = (
        sha256_bytes(_CAMPAIGN_GENESIS_HASH_DOMAIN + bytes.fromhex(header_sha256))
        if round_index == 0
        else round_seals[round_index - 1]
    )
    wave_id = _identifier(parameters["selection_wave_id"], label="history wave ID")
    batch_output_sha256 = _sha256(
        parameters["selection_batch_output_sha256"],
        label="history batch output digest",
    )
    _require(parameters["selection_campaign_id"] == campaign_id, "history campaign binding differs")
    _require(parameters["selection_namespace"] == namespace, "history namespace binding differs")
    _require(
        parameters["selection_pre_wave_head_sha256"] == expected_pre_wave_head,
        "history pre-wave head binding differs",
    )
    _require(
        _bounded_integer(
            parameters["selection_pre_wave_round_count"],
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="history pre-wave round count",
        )
        == round_index,
        "history pre-wave round binding differs",
    )
    _require(
        parameters["selection_ordered_eligible_sha256"] == eligible_sha256,
        "history eligible binding differs",
    )
    _require(
        parameters["selection_policy_implementation_sha256"]
        == row.selection_policy_implementation_sha256,
        "history policy implementation binding differs",
    )
    _require(
        parameters["selection_config_sha256"] == config_sha256,
        "history config binding differs",
    )
    _require(
        parameters["selection_id_derivation"] == row.selection_id_derivation,
        "history selection-ID derivation binding differs",
    )
    expected_selection_set_id = derive_wave_selection_set_id(
        namespace=namespace,
        campaign_id=campaign_id,
        phase=phase,
        wave_id=wave_id,
        authenticated_pre_wave_head_sha256=expected_pre_wave_head,
        authenticated_pre_wave_round_count=round_index,
        batch_output_sha256=batch_output_sha256,
        ordered_eligible_sha256=eligible_sha256,
        policy_version=selection_policy,
        policy_implementation_sha256=row.selection_policy_implementation_sha256,
        seed=selection_seed,
        config_sha256=config_sha256,
        derivation=row.selection_id_derivation,
    )
    _require(
        selection.get("selection_set_id") == expected_selection_set_id,
        "history selection-set ID derivation differs",
    )
    return namespace, wave_id, expected_selection_set_id


__all__ = [
    "load_selection_policy_registry_from_path",
    "validate_campaign_proposal_policy_binding",
]

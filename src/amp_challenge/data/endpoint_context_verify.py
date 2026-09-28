"""Independently verify frozen DRAMP endpoint-context twin artifacts.

This verifier intentionally does not import :mod:`endpoint_context`.  Stable
identifiers, canonical serialization, provenance links, aggregate counts, and
protocol sentinels are recomputed here so a shared implementation error cannot
make the builder attest itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import tomllib
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_PMID_RE = re.compile(r"[1-9][0-9]*")
_ENDPOINTS = frozenset({"mic", "hc50", "hemolysis_percent"})
_GRAMS = frozenset({"positive", "negative", "unknown"})
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NONHUMAN_BLOOD_SPECIES = frozenset(
    {"rabbit", "mouse", "rat", "pig", "cattle", "sheep", "goat", "horse", "chicken"}
)
_APPARENT_BINOMIAL = re.compile(
    r"(?<![A-Za-z])(?P<genus>[A-Z][a-z]{2,}|[A-Z]\.)\s+"
    r"(?P<species>[a-z][a-z-]{2,})\b"
)
_NONTAXONOMIC_LEADING_WORDS = frozenset({"clinical", "drug", "filamentous", "the", "type"})
_MULTI_TARGET_LANGUAGE = re.compile(r"\b(?:strains|isolates)\b", re.IGNORECASE)
_SUSCEPTIBLE_STATUS = re.compile(r"\b(?:standard|susceptible|sensitive)\b", re.IGNORECASE)
_RESISTANT_STATUS = re.compile(
    r"\b(?:(?:multi[- ]?)?drug[- ]resistant|antibiotic[- ]resistant|resistant)\b",
    re.IGNORECASE,
)
_STRAIN_GROUP_NOUN = re.compile(r"\b(?:strains?|isolates?)\b", re.IGNORECASE)
_SECONDARY_RESISTANCE_GROUP = re.compile(
    r"(?:,|;|\band\b|\bor\b)\s*(?:MRSA|MSSA|VRE|VRSA)\b", re.IGNORECASE
)
_TOP_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "endpoint_context/audit.json",
        "endpoint_context/contexts.jsonl",
        "endpoint_context/endpoint_context_ledger.jsonl",
        "endpoint_context/manifest.json",
        "endpoint_context/study_membership.jsonl",
    }
)
_TOP_MANIFEST_ENTRIES = _TOP_FILES - {"SHA256SUMS"}
_REQUIRED_CODE_PATHS = frozenset(
    {
        "cluster/validate_endpoint_context_output.sh",
        "cluster/slurm/build_endpoint_context_v7_twins.sbatch",
        "pyproject.toml",
        "uv.lock",
    }
)
_LEDGER_FIELDS = frozenset(
    {
        "apparent_taxon_mentions",
        "assay_context_id",
        "assay_row_sha256",
        "blood_organism_mapping_version",
        "blood_organism_resolution",
        "canonical_target",
        "composite_language_marker",
        "composite_reason_codes",
        "configured_target_occurrences",
        "context_id",
        "eligible_tasks",
        "endpoint",
        "exclusion_codes",
        "expected_gram",
        "exposure_concentration",
        "gram_resolution",
        "input_line",
        "mapping_status",
        "mapping_version",
        "measurement",
        "measurement_status",
        "mic16_label",
        "mic16_reason",
        "normalized_organism",
        "normalized_target",
        "observation_id",
        "provenance_id",
        "resistance_status",
        "resolved_blood_organism",
        "schema_version",
        "sequence_id",
        "source_assay",
        "source_conditions",
        "source_field",
        "source_gram",
        "source_organism",
        "source_record_id",
        "source_record_key",
        "source_row_number",
        "source_target",
        "source_text",
        "strain_identifier_resolution",
        "strain_identifier_review_ids",
        "strain_level_eligible",
        "study_keys",
        "study_review_codes",
        "study_status",
        "target_domain",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "apparent_taxon_mentions",
        "blood_organism_mapping_version",
        "blood_organism_resolution",
        "canonical_target",
        "composite_language_marker",
        "composite_reason_codes",
        "configured_target_occurrences",
        "context_id",
        "endpoint",
        "expected_gram",
        "gram_resolutions",
        "mapping_status",
        "mapping_version",
        "normalized_organism",
        "normalized_target",
        "observations",
        "resolved_blood_organism",
        "schema_version",
        "sequences",
        "source_grams",
        "source_organism_variants",
        "source_records",
        "source_target_variants",
        "strain_identifier_resolution",
        "strain_identifier_review_ids",
        "strain_level_eligible",
        "target_domain",
    }
)
_MEMBERSHIP_FIELDS = frozenset(
    {
        "citation_reference",
        "citation_title",
        "ignored_pubmed_tokens",
        "pmids",
        "provenance_id",
        "schema_version",
        "sequence_id",
        "source",
        "source_record_id",
        "source_record_key",
        "source_row_number",
        "source_sha256",
        "source_version",
        "study_keys",
        "study_review_codes",
        "study_status",
    }
)
_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSAY_FIELDS = frozenset(
    {
        "assay",
        "endpoint",
        "exposure_concentration",
        "gram",
        "lower",
        "lower_inclusive",
        "organism",
        "provenance",
        "raw_value",
        "relation",
        "sequence",
        "sequence_id",
        "source_text",
        "source_unit",
        "strain",
        "unit",
        "upper",
        "upper_inclusive",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "lower",
        "lower_inclusive",
        "raw_value",
        "relation",
        "source_unit",
        "unit",
        "upper",
        "upper_inclusive",
    }
)
_SHARED_CONTEXT_FIELDS = (
    "endpoint",
    "normalized_target",
    "normalized_organism",
    "resolved_blood_organism",
    "blood_organism_resolution",
    "strain_identifier_resolution",
    "strain_identifier_review_ids",
    "strain_level_eligible",
    "mapping_version",
    "blood_organism_mapping_version",
    "mapping_status",
    "canonical_target",
    "expected_gram",
    "target_domain",
    "apparent_taxon_mentions",
    "configured_target_occurrences",
    "composite_language_marker",
    "composite_reason_codes",
)

_COMPOSITE_SENTINELS = (
    (
        9,
        "DRAMP18356",
        3159,
        "S. aureus MRSA USA300 strain FPR3757, S. aureus KB/8658, S. aureus ATCC 25923",
        "multiple_configured_target_occurrences",
    ),
    (
        11,
        "DRAMP18356",
        3159,
        "S. intermedius ATCC 29663, S. intermedius R-2725",
        "multiple_apparent_taxa",
    ),
    (
        264,
        "DRAMP02403",
        664,
        "Escherichia coli standard and drug-resistant strains",
        "mixed_resistance_groups",
    ),
    (
        5194,
        "DRAMP02402",
        663,
        "Bacillus dysenteriae standard and drug-resistant strains",
        "mixed_resistance_groups",
    ),
    (
        2457,
        "DRAMP18365",
        1141,
        "S. aureus CMCC 26003, MRSA",
        "secondary_resistance_group",
    ),
    (
        4506,
        "DRAMP18361",
        1143,
        "S. aureus KCTC 1928, MRSA B15",
        "secondary_resistance_group",
    ),
    (
        2572,
        "DRAMP04115",
        1182,
        "Clinical isolates: Acinetobacter baumannii b-01",
        "explicit_multiple_strains_or_isolates",
    ),
    (
        4471,
        "DRAMP18650",
        5011,
        "Clinical isolates: Acinetobacter sp",
        "explicit_multiple_strains_or_isolates",
    ),
)
_NONCOMPOSITE_SENTINELS = (
    (7, "DRAMP18356", "L. lactis subsp. lactis LOCK 0871 strain 239"),
    (4195, "DRAMP02354", "A. salmonicida 97-4 field isolate"),
    (5267, "DRAMP02353", "A. salmonicida 97-4 field isolate"),
)
_ALIASED_BLOOD_LINES = frozenset({812, 813, 2410, 4256})
_UNRESOLVED_BLOOD_SENTINELS = {
    612: ("DRAMP03002", 786, "erythrocytes"),
    2172: ("DRAMP18193", 68, "red blood cells"),
}
_EXPECTED_CONTEXT_CENSUS = {
    "unique_contexts": 1205,
    "mapping_counts": {
        "ambiguous_composite_target": 100,
        "mapped_single_supported_species": 2788,
        "not_applicable_non_mic": 176,
        "unmapped_target": 3015,
    },
    "gram_resolution_counts": {
        "concordant": 2619,
        "conflict": 12,
        "missing_source_gram": 157,
        "not_applicable": 176,
        "unresolved": 3115,
    },
    "configured_target_observations": {
        "acinetobacter_baumannii": 11,
        "enterococcus_faecalis": 113,
        "enterococcus_faecium": 7,
        "escherichia_coli": 1072,
        "klebsiella_pneumoniae": 80,
        "pseudomonas_aeruginosa": 559,
        "staphylococcus_aureus": 946,
    },
    "composite_contexts": 45,
    "composite_observations": 100,
    "condition_bearing_observations": 192,
    "eligible_task_counts": {
        "bacterial_mic16": 2606,
        "bacterial_mic_interval": 2619,
        "gram_negative_mic16": 1610,
        "gram_positive_mic16": 996,
        "human_hc50_interval": 34,
        "human_hemolysis_percent_at_dose": 85,
        "nonhuman_hc50_interval_aux": 3,
        "nonhuman_hemolysis_percent_at_dose_aux": 52,
    },
    "exclusion_counts": {
        "ambiguous_composite_target": 100,
        "gram_conflict": 12,
        "gram_missing_source_gram": 157,
        "unmapped_target": 3015,
        "unresolved_blood_species": 2,
    },
    "blood_organism_resolution_counts": {
        "not_applicable_mic": 5903,
        "parser_explicit_human": 115,
        "parser_explicit_nonhuman": 55,
        "reviewed_exact_alias": 4,
        "unresolved": 2,
    },
    "strain_identifier_review_observation_counts": {
        "ATCC2592": 41,
        "ATCC25923": 173,
        "ATCC29213": 16,
        "ATCC8530": 8,
    },
}
_EXPECTED_STUDY_CENSUS = {
    "provenance_memberships": 1136,
    "membership_status_counts": {
        "explicit_pmid": 1100,
        "explicit_pmid_with_ignored_tokens": 11,
        "reference_title_fallback": 24,
        "source_record_singleton": 1,
    },
    "sequence_status_counts": {
        "explicit_pmid": 1078,
        "explicit_pmid_with_ignored_tokens": 11,
        "reference_title_fallback": 24,
        "source_record_singleton": 1,
    },
    "multi_status_sequences": 1,
    "unique_study_keys": 435,
    "shared_study_keys": 199,
    "source_record_singleton_keys": 1,
    "maximum_sequences_per_study_key": 81,
    "ignored_pubmed_token_counts": {"PubMed ID is not available": 35, "Unknown": 1},
    "review_code_membership_counts": {
        "discordant_citation_metadata": 4,
        "multi_pmid_citation_delimiter_missing": 1,
    },
}
_REQUIRED_LIMITATIONS = frozenset(
    {
        "configured mappings cover only frozen supported species and are not a complete taxonomy",
        "strain identifiers are unresolved source literals; reviewed contradictions are annotations only",
        "publication keys are proxies for experimental batches",
        "reference/title fallback may over-merge and missing metadata may under-merge studies",
        "parser-v7 MIC marker accounting and reversed-range handling require a future parser version",
        "DRAMP outcomes have been inspected repeatedly and are not an untouched test panel",
        "study keys are grouping-only metadata and must never be model features",
    }
)
_TARGET_MAPPING_POLICY = (
    "configured regex support only; multiple apparent or configured taxa are ambiguous; "
    "no fuzzy taxonomy or MDR inference"
)
_BLOOD_MAPPING_POLICY = (
    "parser-explicit human/nonhuman values plus only frozen exact source-target aliases; "
    "unresolved blood species are task-ineligible"
)
_STUDY_KEY_POLICY = (
    "all strict decimal PubMed tokens split on the literal configured delimiter; otherwise "
    "exact whitespace-normalized reference/title hash; otherwise a source-record singleton; "
    "grouping-only, never a model feature"
)
_MIC16_POLICY = "MIC <= threshold only when the complete normalized interval proves the class"


class VerificationError(ValueError):
    """Raised when an artifact violates the frozen verification contract."""


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """Only the independently audited fields needed from the v1 config."""

    raw: Mapping[str, Any]
    sha256: str
    parser_id: str
    parser_schema: int
    source_name: str
    source_sha256: str
    source_version: str
    sequences_sha256: str
    assays_sha256: str
    summary_sha256: str
    unique_sequences: int
    assay_observations: int
    endpoint_counts: Mapping[str, int]
    activity_threshold_um: float
    pubmed_delimiter: str
    citation_missing_values: frozenset[str]
    mapping_version: str
    blood_mapping_version: str
    blood_aliases: Mapping[str, str]
    study_anomalies: tuple[Mapping[str, Any], ...]
    strain_reviews: tuple[Mapping[str, str], ...]
    targets: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class JsonlDocument:
    rows: tuple[dict[str, Any], ...]
    raw_lines: tuple[bytes, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, *, label: str) -> bytes:
    _require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    try:
        before = path.stat()
    except FileNotFoundError as error:
        raise VerificationError(f"{label} is missing: {path}") from error
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
    payload = path.read_bytes()
    after = path.stat()
    fingerprint_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    fingerprint_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    _require(
        fingerprint_before == fingerprint_after and len(payload) == before.st_size,
        f"{label} changed while being read: {path}",
    )
    return payload


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _loads_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise VerificationError(f"{label} is not valid JSON: {error}") from error


def _canonical_compact(value: Any, *, ensure_ascii: bool = False) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    ).encode("utf-8")


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_regular_file(path, label=label)
    _require(payload.endswith(b"\n"), f"{label} must end with LF")
    _require(b"\r" not in payload, f"{label} must not contain CR bytes")
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    canonical = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _require(payload == canonical, f"{label} is not canonical sorted UTF-8 JSON")
    return value


def _read_jsonl(path: Path, *, label: str, require_canonical: bool) -> JsonlDocument:
    payload = _read_regular_file(path, label=label)
    _require(payload, f"{label} must not be empty")
    _require(payload.endswith(b"\n"), f"{label} must end with LF")
    _require(b"\r" not in payload, f"{label} must not contain CR bytes")
    raw_lines = tuple(payload[:-1].split(b"\n"))
    _require(all(raw_lines), f"{label} must not contain blank lines")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        value = _loads_json(raw_line, label=f"{label} line {line_number}")
        _require(isinstance(value, dict), f"{label} line {line_number} is not an object")
        if require_canonical:
            _require(
                raw_line == _canonical_compact(value),
                f"{label} line {line_number} is not canonical compact sorted JSON",
            )
        rows.append(value)
    return JsonlDocument(rows=tuple(rows), raw_lines=raw_lines)


def _safe_manifest_name(raw: str, *, label: str) -> str:
    _require(raw and "\\" not in raw and "\x00" not in raw, f"{label} has an unsafe path")
    stripped = raw[2:] if raw.startswith("./") else raw
    path = PurePosixPath(stripped)
    _require(not path.is_absolute(), f"{label} has an absolute path: {raw!r}")
    _require(stripped == path.as_posix(), f"{label} has a non-canonical path: {raw!r}")
    _require(all(part not in {"", ".", ".."} for part in path.parts), f"{label} has traversal")
    return path.as_posix()


def _parse_checksum_manifest(path: Path, *, label: str) -> dict[str, str]:
    payload = _read_regular_file(path, label=label)
    _require(payload and payload.endswith(b"\n"), f"{label} must end with LF")
    _require(b"\r" not in payload, f"{label} must not contain CR bytes")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"{label} line {line_number} is malformed")
        assert match is not None
        name = _safe_manifest_name(match.group(2), label=f"{label} line {line_number}")
        _require(name not in entries, f"{label} repeats path {name!r}")
        entries[name] = match.group(1)
    return entries


def _verify_manifest_files(root: Path, entries: Mapping[str, str], *, label: str) -> None:
    resolved_root = root.resolve(strict=True)
    for relative, expected in entries.items():
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        resolved = candidate.resolve(strict=True)
        _require(resolved.is_relative_to(resolved_root), f"{label} escapes its root: {relative}")
        observed = _sha256_bytes(_read_regular_file(candidate, label=f"{label}:{relative}"))
        _require(observed == expected, f"{label} checksum mismatch for {relative}")


def _regular_inventory(root: Path) -> frozenset[str]:
    inventory: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"artifact tree contains symbolic link: {path}")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
    return frozenset(inventory)


def _load_config(path: Path) -> FrozenConfig:
    payload = _read_regular_file(path, label="endpoint-context config")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("endpoint-context config is not valid UTF-8 TOML") from error

    def string(name: str) -> str:
        value = raw.get(name)
        _require(isinstance(value, str) and value.strip() == value and value, f"bad config {name}")
        return value

    def integer(name: str) -> int:
        value = raw.get(name)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0, f"bad {name}"
        )
        return value

    allowed = {
        "schema_version",
        "normalized_parser_id",
        "normalized_schema_version",
        "source_name",
        "source_sha256",
        "source_version",
        "normalized_sequences_sha256",
        "normalized_assays_sha256",
        "normalized_summary_sha256",
        "expected_unique_sequences",
        "expected_assay_observations",
        "expected_endpoint_counts",
        "activity_threshold_um",
        "pubmed_delimiter",
        "citation_missing_values",
        "mapping_version",
        "blood_organism_mapping_version",
        "blood_organism_alias",
        "study_anomaly",
        "strain_identifier_review",
        "target",
    }
    _require(set(raw) == allowed, "endpoint-context config key set changed")
    _require(
        isinstance(raw.get("schema_version"), int)
        and not isinstance(raw.get("schema_version"), bool)
        and raw["schema_version"] == 1,
        "unsupported endpoint-context config schema",
    )
    hashes = {
        name: string(name)
        for name in (
            "source_sha256",
            "normalized_sequences_sha256",
            "normalized_assays_sha256",
            "normalized_summary_sha256",
        )
    }
    _require(all(_SHA256_RE.fullmatch(value) for value in hashes.values()), "bad config digest")
    counts = raw.get("expected_endpoint_counts")
    _require(isinstance(counts, dict) and set(counts) == _ENDPOINTS, "bad endpoint histogram")
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        ),
        "bad endpoint count",
    )
    assay_observations = integer("expected_assay_observations")
    _require(sum(counts.values()) == assay_observations, "endpoint counts do not sum to assays")
    threshold = raw.get("activity_threshold_um")
    _require(
        isinstance(threshold, int | float)
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and threshold > 0,
        "bad activity threshold",
    )
    missing = raw.get("citation_missing_values")
    _require(
        isinstance(missing, list) and all(isinstance(item, str) for item in missing),
        "bad missing values",
    )
    normalized_missing = frozenset(_normalize_whitespace(item).casefold() for item in missing)
    _require(
        "" in normalized_missing and len(normalized_missing) == len(missing), "bad missing values"
    )
    anomalies = raw.get("study_anomaly", [])
    _require(
        isinstance(anomalies, list) and all(isinstance(item, dict) for item in anomalies),
        "bad study anomalies",
    )
    anomaly_fields = {
        "scope",
        "selector",
        "code",
        "policy",
        "expected_source_record_ids",
        "expected_study_keys",
        "discordant_source_record_ids",
        "expected_observations",
    }
    _require(all(set(item) == anomaly_fields for item in anomalies), "study anomaly schema changed")
    aliases = raw.get("blood_organism_alias")
    _require(isinstance(aliases, list), "bad blood aliases")
    alias_map: dict[str, str] = {}
    for item in aliases:
        _require(
            isinstance(item, dict) and set(item) == {"source_target", "canonical_organism"},
            "blood alias schema changed",
        )
        source_target = item["source_target"]
        organism = item["canonical_organism"]
        _require(isinstance(source_target, str) and isinstance(organism, str), "bad blood alias")
        _require(source_target not in alias_map, "duplicate blood alias")
        _require(
            organism == "human" or organism in _NONHUMAN_BLOOD_SPECIES,
            "unsupported blood alias organism",
        )
        alias_map[source_target] = organism
    strain_reviews = raw.get("strain_identifier_review")
    _require(isinstance(strain_reviews, list), "bad strain reviews")
    for item in strain_reviews:
        _require(
            isinstance(item, dict)
            and set(item) == {"identifier", "target_regex", "code", "policy"}
            and all(isinstance(value, str) and value for value in item.values()),
            "strain review schema changed",
        )
        try:
            re.compile(item["target_regex"])
        except re.error as error:
            raise VerificationError("invalid strain review regex") from error
    targets = raw.get("target")
    _require(isinstance(targets, list) and targets, "bad target rules")
    for item in targets:
        _require(
            isinstance(item, dict)
            and set(item) == {"name", "strain_regex", "expected_gram"}
            and all(isinstance(value, str) and value for value in item.values()),
            "target rule schema changed",
        )
        _require(item["expected_gram"] in {"positive", "negative"}, "bad target Gram")
        try:
            re.compile(item["strain_regex"])
        except re.error as error:
            raise VerificationError("invalid target regex") from error
    return FrozenConfig(
        raw=raw,
        sha256=_sha256_bytes(payload),
        parser_id=string("normalized_parser_id"),
        parser_schema=integer("normalized_schema_version"),
        source_name=string("source_name"),
        source_sha256=hashes["source_sha256"],
        source_version=string("source_version"),
        sequences_sha256=hashes["normalized_sequences_sha256"],
        assays_sha256=hashes["normalized_assays_sha256"],
        summary_sha256=hashes["normalized_summary_sha256"],
        unique_sequences=integer("expected_unique_sequences"),
        assay_observations=assay_observations,
        endpoint_counts=dict(counts),
        activity_threshold_um=float(threshold),
        pubmed_delimiter=string("pubmed_delimiter"),
        citation_missing_values=normalized_missing,
        mapping_version=string("mapping_version"),
        blood_mapping_version=string("blood_organism_mapping_version"),
        blood_aliases=alias_map,
        study_anomalies=tuple(anomalies),
        strain_reviews=tuple(strain_reviews),
        targets=tuple(targets),
    )


def _normalize_whitespace(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def _typed_id_frame(value: Any) -> Any:
    """Encode one identity value with an explicit recursive type tag."""

    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        _require(
            value == value and value not in {float("inf"), float("-inf")}, "non-finite ID float"
        )
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, tuple):
        return ["tuple", [_typed_id_frame(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_id_frame(item) for item in value]]
    if isinstance(value, Mapping):
        _require(all(isinstance(key, str) for key in value), "ID mapping key is not a string")
        return [
            "mapping",
            [[key, _typed_id_frame(value[key])] for key in sorted(value)],
        ]
    raise VerificationError(f"unsupported stable ID type: {type(value).__name__}")


def _json_framed_digest(namespace: str, *parts: Any) -> str:
    """Independent implementation of the frozen JSON-framed identifier."""

    _require(isinstance(namespace, str) and namespace, "stable ID namespace is empty")
    payload = _canonical_compact(
        {
            "namespace": _typed_id_frame(namespace),
            "parts": [_typed_id_frame(part) for part in parts],
        }
    )
    return _sha256_bytes(payload)


def _exact_fields(row: Mapping[str, Any], fields: frozenset[str], *, label: str) -> None:
    _require(set(row) == fields, f"{label} schema mismatch")


def _extra_mapping(provenance: Mapping[str, Any], *, label: str) -> dict[str, str]:
    _require(
        set(provenance) == {"source", "record_id", "path", "row_number", "extra"},
        f"{label} provenance schema mismatch",
    )
    extra = provenance.get("extra")
    _require(isinstance(extra, list), f"{label} provenance extra is not a list")
    result: dict[str, str] = {}
    for index, item in enumerate(extra):
        _require(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and item[0]
            and isinstance(item[1], str),
            f"{label} provenance extra entry {index} is invalid",
        )
        _require(item[0] not in result, f"{label} provenance repeats extra key")
        result[item[0]] = item[1]
    _require(
        {"DRAMP_ID", "Sequence", "source_sha256", "source_version"}.issubset(result),
        f"{label} provenance lacks core extra fields",
    )
    return result


def _provenance_identity(
    provenance: Mapping[str, Any], config: FrozenConfig, *, label: str
) -> tuple[str, str, int, dict[str, str], bytes]:
    extra = _extra_mapping(provenance, label=label)
    source = provenance.get("source")
    record_id = provenance.get("record_id")
    row_number = provenance.get("row_number")
    _require(source == config.source_name, f"{label} provenance source mismatch")
    _require(
        isinstance(record_id, str) and record_id.strip() == record_id and record_id,
        f"{label} bad record ID",
    )
    _require(
        isinstance(row_number, int) and not isinstance(row_number, bool) and row_number >= 2,
        f"{label} bad source row",
    )
    _require(extra["DRAMP_ID"] == record_id, f"{label} DRAMP_ID mismatch")
    _require(extra["source_sha256"] == config.source_sha256, f"{label} source hash mismatch")
    _require(extra["source_version"] == config.source_version, f"{label} source version mismatch")
    provenance_id = _json_framed_digest(
        "amp-challenge:source-provenance:v1",
        source,
        config.source_version,
        config.source_sha256,
        record_id,
        row_number,
    )
    semantic = {
        "provenance_id": provenance_id,
        "source": source,
        "record_id": record_id,
        "row_number": row_number,
        "extra": extra,
    }
    return provenance_id, record_id, row_number, extra, _canonical_compact(semantic)


def _citation(raw: Any, config: FrozenConfig) -> str | None:
    value = _normalize_whitespace(raw)
    return None if value.casefold() in config.citation_missing_values else value


def _review_codes(record_id: str, study_keys: Sequence[str], config: FrozenConfig) -> list[str]:
    codes = {
        str(anomaly["code"])
        for anomaly in config.study_anomalies
        if (anomaly.get("scope") == "record" and anomaly.get("selector") == record_id)
        or (anomaly.get("scope") == "key" and anomaly.get("selector") in study_keys)
    }
    return sorted(codes)


def _derive_membership(
    *, sequence_id: str, provenance: Mapping[str, Any], config: FrozenConfig, label: str
) -> tuple[dict[str, Any], bytes]:
    provenance_id, record_id, row_number, extra, semantic = _provenance_identity(
        provenance, config, label=label
    )
    pubmed = _normalize_whitespace(extra.get("Pubmed_ID", ""))
    raw_tokens = [token.strip() for token in pubmed.split(config.pubmed_delimiter) if token.strip()]
    pmids = sorted({token for token in raw_tokens if _PMID_RE.fullmatch(token)}, key=int)
    ignored = sorted({token for token in raw_tokens if not _PMID_RE.fullmatch(token)})
    reference = _citation(extra.get("Reference", ""), config)
    title = _citation(extra.get("Title", ""), config)
    source_record_key = f"source-record:{provenance_id}"
    if pmids:
        keys = [f"pmid:{pmid}" for pmid in pmids]
        status = "explicit_pmid_with_ignored_tokens" if ignored else "explicit_pmid"
    elif reference is not None or title is not None:
        digest = _json_framed_digest(
            "amp-challenge:reference-title-study:v1", reference or "", title or ""
        )
        keys = [f"reference-title:{digest}"]
        status = "reference_title_fallback"
    else:
        keys = [source_record_key]
        status = "source_record_singleton"
    return (
        {
            "schema_version": 1,
            "sequence_id": sequence_id,
            "provenance_id": provenance_id,
            "source": config.source_name,
            "source_version": config.source_version,
            "source_sha256": config.source_sha256,
            "source_record_id": record_id,
            "source_row_number": row_number,
            "source_record_key": source_record_key,
            "study_status": status,
            "study_keys": keys,
            "pmids": pmids,
            "ignored_pubmed_tokens": ignored,
            "citation_reference": reference,
            "citation_title": title,
            "study_review_codes": _review_codes(record_id, keys, config),
        },
        semantic,
    )


def _measurement_from_assay(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in _MEASUREMENT_FIELDS}


def _source_conditions(assay: Any) -> list[str]:
    if assay is None:
        return []
    marker = "; source condition: "
    text = str(assay)
    if marker not in text:
        return []
    return list(
        dict.fromkeys(
            item.strip() for item in text.split(marker, 1)[1].split(" | ") if item.strip()
        )
    )


def _exposure_identity(exposure: Any) -> tuple[Any, ...] | None:
    if exposure is None:
        return None
    _require(isinstance(exposure, Mapping), "exposure identity requires an object or null")
    return (
        exposure["relation"],
        exposure["lower"],
        exposure["lower_inclusive"],
        exposure["upper"],
        exposure["upper_inclusive"],
        exposure["unit"],
    )


def _validate_measurement(value: Mapping[str, Any], *, label: str) -> None:
    _exact_fields(value, _MEASUREMENT_FIELDS, label=label)
    relation = value["relation"]
    _require(relation in {"eq", "approx", "lt", "le", "gt", "ge", "range"}, f"{label} bad relation")
    lower = value["lower"]
    upper = value["upper"]
    for name, bound in (("lower", lower), ("upper", upper)):
        _require(
            bound is None
            or (
                isinstance(bound, int | float)
                and not isinstance(bound, bool)
                and math.isfinite(bound)
                and bound >= 0
            ),
            f"{label} bad {name} bound",
        )
    _require(lower is not None or upper is not None, f"{label} has no bound")
    _require(lower is None or upper is None or lower <= upper, f"{label} has reversed bounds")
    lower_inclusive = value["lower_inclusive"]
    upper_inclusive = value["upper_inclusive"]
    _require(isinstance(lower_inclusive, bool), f"{label} bad lower inclusivity")
    _require(isinstance(upper_inclusive, bool), f"{label} bad upper inclusivity")
    expected_lower = relation in {"eq", "approx", "ge", "range"}
    expected_upper = relation in {"eq", "approx", "le", "range"}
    _require(lower_inclusive == expected_lower, f"{label} inconsistent lower inclusivity")
    _require(upper_inclusive == expected_upper, f"{label} inconsistent upper inclusivity")
    if relation in {"eq", "approx"}:
        _require(lower is not None and lower == upper, f"{label} exact relation is not exact")
    elif relation in {"lt", "le"}:
        _require(lower is None and upper is not None, f"{label} upper relation shape mismatch")
    elif relation in {"gt", "ge"}:
        _require(lower is not None and upper is None, f"{label} lower relation shape mismatch")
    else:
        _require(lower is not None and upper is not None, f"{label} range shape mismatch")
    for name in ("unit", "source_unit", "raw_value"):
        item = value[name]
        _require(item is None or (isinstance(item, str) and item), f"{label} bad {name}")


def _positive_bounds(value: Mapping[str, Any]) -> bool:
    bounds = [value[name] for name in ("lower", "upper") if value[name] is not None]
    return bool(bounds) and all(bound > 0 for bound in bounds)


def _classify_target(
    *, endpoint: str, normalized_target: str | None, source_gram: str, config: FrozenConfig
) -> dict[str, Any]:
    if endpoint != "mic":
        return {
            "mapping_status": "not_applicable_non_mic",
            "canonical_target": None,
            "expected_gram": None,
            "gram_resolution": "not_applicable",
            "target_domain": "blood_cells",
            "apparent_taxon_mentions": 0,
            "configured_target_occurrences": 0,
            "composite_language_marker": False,
            "composite_reason_codes": [],
        }
    if not normalized_target:
        return {
            "mapping_status": "missing_target",
            "canonical_target": None,
            "expected_gram": None,
            "gram_resolution": "unresolved",
            "target_domain": "unknown",
            "apparent_taxon_mentions": 0,
            "configured_target_occurrences": 0,
            "composite_language_marker": False,
            "composite_reason_codes": [],
        }
    occurrences = [
        (rule, sum(1 for _ in re.finditer(rule["strain_regex"], normalized_target)))
        for rule in config.targets
    ]
    total_occurrences = sum(count for _, count in occurrences)
    matched_rules = [rule for rule, count in occurrences if count]
    apparent_mentions = sum(
        match.group("genus").casefold() not in _NONTAXONOMIC_LEADING_WORDS
        for match in _APPARENT_BINOMIAL.finditer(normalized_target)
    )
    reasons: list[str] = []
    if total_occurrences > 1:
        reasons.append("multiple_configured_target_occurrences")
    if len(matched_rules) > 1:
        reasons.append("multiple_configured_species")
    if apparent_mentions > 1:
        reasons.append("multiple_apparent_taxa")
    if _MULTI_TARGET_LANGUAGE.search(normalized_target):
        reasons.append("explicit_multiple_strains_or_isolates")
    if (
        _SUSCEPTIBLE_STATUS.search(normalized_target)
        and _RESISTANT_STATUS.search(normalized_target)
        and _STRAIN_GROUP_NOUN.search(normalized_target)
    ):
        reasons.append("mixed_resistance_groups")
    if _SECONDARY_RESISTANCE_GROUP.search(normalized_target):
        reasons.append("secondary_resistance_group")
    if reasons:
        return {
            "mapping_status": "ambiguous_composite_target",
            "canonical_target": None,
            "expected_gram": None,
            "gram_resolution": "unresolved",
            "target_domain": "unknown",
            "apparent_taxon_mentions": apparent_mentions,
            "configured_target_occurrences": total_occurrences,
            "composite_language_marker": True,
            "composite_reason_codes": sorted(reasons),
        }
    if len(matched_rules) != 1:
        return {
            "mapping_status": "unmapped_target",
            "canonical_target": None,
            "expected_gram": None,
            "gram_resolution": "unresolved",
            "target_domain": "unknown",
            "apparent_taxon_mentions": apparent_mentions,
            "configured_target_occurrences": total_occurrences,
            "composite_language_marker": False,
            "composite_reason_codes": [],
        }
    rule = matched_rules[0]
    expected_gram = rule["expected_gram"]
    if source_gram == "unknown":
        gram_resolution = "missing_source_gram"
    elif source_gram == expected_gram:
        gram_resolution = "concordant"
    else:
        gram_resolution = "conflict"
    return {
        "mapping_status": "mapped_single_supported_species",
        "canonical_target": rule["name"],
        "expected_gram": expected_gram,
        "gram_resolution": gram_resolution,
        "target_domain": "bacteria",
        "apparent_taxon_mentions": apparent_mentions,
        "configured_target_occurrences": total_occurrences,
        "composite_language_marker": False,
        "composite_reason_codes": [],
    }


def _resolve_blood(
    *,
    endpoint: str,
    normalized_target: str | None,
    source_organism: str | None,
    config: FrozenConfig,
) -> tuple[str | None, str]:
    if endpoint == "mic":
        return None, "not_applicable_mic"
    if source_organism == "human":
        return "human", "parser_explicit_human"
    if source_organism in _NONHUMAN_BLOOD_SPECIES:
        return source_organism, "parser_explicit_nonhuman"
    if source_organism is None and normalized_target in config.blood_aliases:
        assert normalized_target is not None
        return config.blood_aliases[normalized_target], "reviewed_exact_alias"
    return None, "unresolved"


def _blood_task(endpoint: str, organism: str | None) -> str | None:
    if organism == "human":
        prefix = "human"
    elif organism in _NONHUMAN_BLOOD_SPECIES:
        prefix = "nonhuman"
    else:
        return None
    suffix = "hc50_interval" if endpoint == "hc50" else "hemolysis_percent_at_dose"
    return f"{prefix}_{suffix}" + ("_aux" if prefix == "nonhuman" else "")


def _mic16_label(measurement: Mapping[str, Any], threshold: float) -> tuple[int | None, str]:
    if measurement["unit"] != "uM":
        return None, "unsupported_or_non_um_unit"
    if measurement["relation"] == "approx":
        return None, "approximate_value_has_undeclared_tolerance"
    upper = measurement["upper"]
    lower = measurement["lower"]
    if upper is not None and upper <= threshold:
        return 1, "interval_guarantees_mic_at_or_below_threshold"
    if lower is not None and (
        lower > threshold or (lower == threshold and not measurement["lower_inclusive"])
    ):
        return 0, "interval_guarantees_mic_above_threshold"
    return None, "measurement_interval_crosses_or_touches_decision_boundary"


def _derive_assay_semantics(
    assay: Mapping[str, Any], *, input_line: int, config: FrozenConfig
) -> dict[str, Any]:
    endpoint = assay["endpoint"]
    source_gram = assay["gram"]
    _require(endpoint in _ENDPOINTS, f"assay line {input_line} has bad endpoint")
    _require(source_gram in _GRAMS, f"assay line {input_line} has bad Gram")
    for field in ("assay", "organism", "source_text", "strain"):
        _require(
            assay[field] is None or isinstance(assay[field], str),
            f"assay line {input_line} has bad {field}",
        )
    measurement = _measurement_from_assay(assay)
    _validate_measurement(measurement, label=f"assay line {input_line} measurement")
    exposure = assay["exposure_concentration"]
    if endpoint in {"mic", "hc50"}:
        _require(
            measurement["unit"] == "uM" and _positive_bounds(measurement),
            f"assay line {input_line} concentration contract changed",
        )
        _require(exposure is None, f"assay line {input_line} has unexpected exposure")
    else:
        _require(isinstance(exposure, dict), f"assay line {input_line} lacks exposure")
        _validate_measurement(exposure, label=f"assay line {input_line} exposure")
        effect_bounds = [
            measurement[name] for name in ("lower", "upper") if measurement[name] is not None
        ]
        _require(
            measurement["unit"] == "%"
            and effect_bounds
            and min(effect_bounds) >= 0
            and max(effect_bounds) <= 100
            and exposure["unit"] == "uM"
            and _positive_bounds(exposure),
            f"assay line {input_line} percent-at-dose contract changed",
        )
    normalized_target = None if assay["strain"] is None else _normalize_whitespace(assay["strain"])
    normalized_organism = (
        None if assay["organism"] is None else _normalize_whitespace(assay["organism"])
    )
    strain_ids = (
        []
        if endpoint != "mic" or normalized_target is None
        else sorted(
            review["identifier"]
            for review in config.strain_reviews
            if re.search(review["target_regex"], normalized_target)
        )
    )
    if endpoint != "mic":
        strain_resolution = "not_applicable_non_mic"
    elif normalized_target is None:
        strain_resolution = "missing_source_target"
    elif strain_ids:
        strain_resolution = "known_conflicting_identifier"
    else:
        strain_resolution = "unreviewed_source_literal"
    target = _classify_target(
        endpoint=endpoint,
        normalized_target=normalized_target,
        source_gram=source_gram,
        config=config,
    )
    resolved_blood, blood_resolution = _resolve_blood(
        endpoint=endpoint,
        normalized_target=normalized_target,
        source_organism=assay["organism"],
        config=config,
    )
    mic16_label: int | None = None
    mic16_reason: str | None = None
    if endpoint == "mic":
        mic16_label, mic16_reason = _mic16_label(measurement, config.activity_threshold_um)
    exclusions: list[str] = []
    if endpoint == "mic" and target["mapping_status"] != "mapped_single_supported_species":
        exclusions.append(target["mapping_status"])
    if target["gram_resolution"] in {"conflict", "missing_source_gram"}:
        exclusions.append(f"gram_{target['gram_resolution']}")
    tasks: list[str] = []
    if (
        endpoint == "mic"
        and target["mapping_status"] == "mapped_single_supported_species"
        and target["gram_resolution"] == "concordant"
    ):
        tasks.append("bacterial_mic_interval")
        if mic16_label is not None:
            tasks.extend(("bacterial_mic16", f"gram_{source_gram}_mic16"))
    elif endpoint in {"hc50", "hemolysis_percent"}:
        blood_task = _blood_task(endpoint, resolved_blood)
        if blood_task is None:
            exclusions.append("unresolved_blood_species")
        else:
            tasks.append(blood_task)
    return {
        "normalized_target": normalized_target,
        "normalized_organism": normalized_organism,
        "strain_identifier_resolution": strain_resolution,
        "strain_identifier_review_ids": strain_ids,
        "strain_level_eligible": False,
        **target,
        "resolved_blood_organism": resolved_blood,
        "blood_organism_resolution": blood_resolution,
        "mic16_label": mic16_label,
        "mic16_reason": mic16_reason,
        "eligible_tasks": sorted(tasks),
        "exclusion_codes": sorted(exclusions),
    }


def _verify_normalized_summary(
    summary_path: Path, config: FrozenConfig, *, data_manifest_sha256: str
) -> dict[str, Any]:
    payload = _read_regular_file(summary_path, label="normalized summary")
    _require(_sha256_bytes(payload) == config.summary_sha256, "normalized summary hash mismatch")
    value = _loads_json(payload, label="normalized summary")
    _require(isinstance(value, dict), "normalized summary is not an object")
    _require(value.get("schema_version") == config.parser_schema, "normalized schema mismatch")
    _require(value.get("parser_id") == config.parser_id, "normalized parser mismatch")
    _require(
        value.get("unique_sequences") == config.unique_sequences, "summary sequence count mismatch"
    )
    _require(
        value.get("assay_observations") == config.assay_observations, "summary assay count mismatch"
    )
    _require(
        value.get("endpoint_counts") == config.endpoint_counts, "summary endpoint counts mismatch"
    )
    _require(
        value.get("sequences_sha256") == config.sequences_sha256, "summary sequences hash mismatch"
    )
    _require(value.get("assays_sha256") == config.assays_sha256, "summary assays hash mismatch")
    artifacts = value.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == 1, "summary source artifact mismatch"
    )
    source = artifacts[0]
    _require(isinstance(source, dict), "summary source artifact is invalid")
    _require(source.get("name") == config.source_name, "summary source name mismatch")
    _require(source.get("sha256") == config.source_sha256, "summary source hash mismatch")
    _require(
        source.get("source_commit") == config.source_version, "summary source version mismatch"
    )
    _require(source.get("training_status") == "approved", "summary source is not approved")
    _require(_SHA256_RE.fullmatch(data_manifest_sha256) is not None, "bad data manifest hash")
    return value


def _verify_data_run(run: Path, config: FrozenConfig) -> tuple[JsonlDocument, JsonlDocument, str]:
    manifest_path = run / "SHA256SUMS"
    manifest_payload = _read_regular_file(manifest_path, label="normalized data manifest")
    entries = _parse_checksum_manifest(manifest_path, label="normalized data manifest")
    _verify_manifest_files(run, entries, label="normalized data manifest")
    required = {
        "normalized/sequences.jsonl": config.sequences_sha256,
        "normalized/assays.jsonl": config.assays_sha256,
        "normalized/summary.json": config.summary_sha256,
    }
    for name, digest in required.items():
        _require(entries.get(name) == digest, f"normalized data manifest does not pin {name}")
    sequences = _read_jsonl(
        run / "normalized" / "sequences.jsonl",
        label="normalized sequences",
        require_canonical=False,
    )
    assays = _read_jsonl(
        run / "normalized" / "assays.jsonl",
        label="normalized assays",
        require_canonical=False,
    )
    _require(len(sequences.rows) == config.unique_sequences, "normalized sequence count mismatch")
    _require(len(assays.rows) == config.assay_observations, "normalized assay count mismatch")
    manifest_sha = _sha256_bytes(manifest_payload)
    _verify_normalized_summary(
        run / "normalized" / "summary.json", config, data_manifest_sha256=manifest_sha
    )
    return sequences, assays, manifest_sha


def _verify_frozen_input_manifest(
    run: Path, normalized_run: Path, *, expected_data_manifest_sha256: str
) -> str:
    path = run / "FROZEN_INPUT_SHA256SUMS"
    entries = _parse_checksum_manifest(path, label="frozen input manifest")
    _require(entries, "frozen input manifest is empty")
    _require(all(name.startswith("data/") for name in entries), "frozen input path is unscoped")
    stripped = {name.removeprefix("data/"): digest for name, digest in entries.items()}
    expected_inventory = _regular_inventory(normalized_run)
    _require(set(stripped) == expected_inventory, "frozen input manifest inventory mismatch")
    _verify_manifest_files(normalized_run, stripped, label="frozen input manifest")
    _require(
        stripped.get("SHA256SUMS") == expected_data_manifest_sha256,
        "frozen input manifest does not pin the normalized data manifest",
    )
    return _sha256_bytes(_read_regular_file(path, label="frozen input manifest"))


def _verify_code_manifest(run: Path, repo_root: Path, config_path: Path) -> str:
    path = run / "CODE_SHA256SUMS"
    entries = _parse_checksum_manifest(path, label="code manifest")
    expected_paths = {
        *_REQUIRED_CODE_PATHS,
        config_path.relative_to(repo_root).as_posix(),
        *(
            item.relative_to(repo_root).as_posix()
            for item in (repo_root / "src" / "amp_challenge").rglob("*.py")
        ),
    }
    _require(set(entries) == expected_paths, "code manifest inventory mismatch")
    _verify_manifest_files(repo_root, entries, label="code manifest")
    return _sha256_bytes(_read_regular_file(path, label="code manifest"))


def _verify_top_manifest(run: Path) -> str:
    inventory = _regular_inventory(run)
    _require(
        inventory == _TOP_FILES, f"endpoint-context run inventory mismatch: {sorted(inventory)}"
    )
    path = run / "SHA256SUMS"
    entries = _parse_checksum_manifest(path, label="endpoint-context top manifest")
    _require(
        set(entries) == _TOP_MANIFEST_ENTRIES, "endpoint-context top manifest inventory mismatch"
    )
    _verify_manifest_files(run, entries, label="endpoint-context top manifest")
    return _sha256_bytes(_read_regular_file(path, label="endpoint-context top manifest"))


def _verify_manifest(
    output: Path,
    config: FrozenConfig,
    *,
    data_manifest_sha256: str,
    code_manifest_sha256: str,
    artifact_digests: Mapping[str, str],
    expected_git_commit: str | None,
) -> dict[str, Any]:
    manifest = _read_canonical_json(output / "manifest.json", label="sidecar manifest")
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact",
            "status",
            "config_sha256",
            "input",
            "policies",
            "counts",
            "artifacts",
            "provenance",
        },
        "sidecar manifest schema mismatch",
    )
    _require(
        type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
        "sidecar manifest schema version mismatch",
    )
    _require(manifest["artifact"] == "dramp_endpoint_context_sidecar", "sidecar artifact mismatch")
    _require(
        manifest["status"]
        == "development_only_pending_reviewed_taxonomy_and_study_component_split",
        "sidecar status mismatch",
    )
    _require(manifest["config_sha256"] == config.sha256, "sidecar config hash mismatch")
    expected_inputs = {
        "sequences": {"filename": "sequences.jsonl", "sha256": config.sequences_sha256},
        "assays": {"filename": "assays.jsonl", "sha256": config.assays_sha256},
        "normalized_summary": {
            "filename": "summary.json",
            "sha256": config.summary_sha256,
        },
        "normalized_data_manifest": {
            "filename": "SHA256SUMS",
            "sha256": data_manifest_sha256,
        },
    }
    _require(manifest["input"] == expected_inputs, "sidecar input attestation mismatch")
    policies = manifest["policies"]
    _require(isinstance(policies, dict), "sidecar policies must be an object")
    _require(
        set(policies)
        == {
            "mapping_version",
            "blood_organism_mapping_version",
            "target_mapping",
            "blood_organism_mapping",
            "study_keys",
            "study_anomalies",
            "strain_identifier_reviews",
            "mic16",
        },
        "sidecar policy schema mismatch",
    )
    expected_static_policies = {
        "mapping_version": config.mapping_version,
        "blood_organism_mapping_version": config.blood_mapping_version,
        "target_mapping": _TARGET_MAPPING_POLICY,
        "blood_organism_mapping": _BLOOD_MAPPING_POLICY,
        "study_keys": _STUDY_KEY_POLICY,
        "strain_identifier_reviews": [
            {
                "identifier": item["identifier"],
                "target_regex": item["target_regex"],
                "code": item["code"],
                "policy": item["policy"],
            }
            for item in config.strain_reviews
        ],
        "mic16": _MIC16_POLICY,
    }
    for key, expected in expected_static_policies.items():
        _require(policies[key] == expected, f"sidecar policy projection mismatch: {key}")
    _require(isinstance(policies["study_anomalies"], list), "study anomaly policy is not a list")
    expected_artifacts = {
        "endpoint_context_ledger": {
            "filename": "endpoint_context_ledger.jsonl",
            "sha256": artifact_digests["endpoint_context_ledger.jsonl"],
        },
        "contexts": {
            "filename": "contexts.jsonl",
            "sha256": artifact_digests["contexts.jsonl"],
        },
        "study_membership": {
            "filename": "study_membership.jsonl",
            "sha256": artifact_digests["study_membership.jsonl"],
        },
        "audit": {"filename": "audit.json", "sha256": artifact_digests["audit.json"]},
    }
    _require(manifest["artifacts"] == expected_artifacts, "inner artifact manifest mismatch")
    provenance = manifest["provenance"]
    _require(isinstance(provenance, dict), "sidecar provenance must be an object")
    _exact_fields(
        provenance,
        frozenset({"git_commit", "code_manifest"}),
        label="sidecar provenance",
    )
    git_commit = provenance["git_commit"]
    _require(isinstance(git_commit, str) and _GIT_SHA_RE.fullmatch(git_commit), "bad Git commit")
    if expected_git_commit is not None:
        _require(git_commit == expected_git_commit, "sidecar Git commit is not the expected commit")
    _require(
        provenance["code_manifest"]
        == {"filename": "CODE_SHA256SUMS", "sha256": code_manifest_sha256},
        "sidecar code manifest attestation mismatch",
    )
    return manifest


def _verify_memberships(
    sequence_doc: JsonlDocument,
    membership_rows: Sequence[dict[str, Any]],
    config: FrozenConfig,
) -> tuple[
    dict[tuple[str, str], tuple[dict[str, Any], bytes]],
    dict[str, str],
]:
    expected_rows: list[dict[str, Any]] = []
    provenance: dict[tuple[str, str], tuple[dict[str, Any], bytes]] = {}
    sequences: dict[str, str] = {}
    provenance_owners: dict[str, str] = {}
    for row_number, row in enumerate(sequence_doc.rows, start=1):
        _exact_fields(row, _SEQUENCE_FIELDS, label=f"normalized sequence row {row_number}")
        sequence = row["sequence"]
        sequence_id = row["sequence_id"]
        _require(
            isinstance(sequence, str) and isinstance(sequence_id, str), "bad normalized sequence"
        )
        _require(
            sequence == sequence.upper() and not any(char.isspace() for char in sequence),
            "sequence is not canonical",
        )
        _require(8 <= len(sequence) <= 50, "sequence length is outside challenge bounds")
        _require(
            set(sequence).issubset(_STANDARD_AMINO_ACIDS), "sequence alphabet is not canonical"
        )
        _require(_sha256_bytes(sequence.encode("ascii")) == sequence_id, "sequence ID mismatch")
        _require(sequence_id not in sequences, "normalized sequences repeat a sequence ID")
        sequences[sequence_id] = sequence
        raw_provenance = row["provenance"]
        _require(isinstance(raw_provenance, list) and raw_provenance, "sequence lacks provenance")
        for index, item in enumerate(raw_provenance):
            _require(isinstance(item, dict), "sequence provenance is not an object")
            expected, semantic = _derive_membership(
                sequence_id=sequence_id,
                provenance=item,
                config=config,
                label=f"normalized sequence row {row_number} provenance {index}",
            )
            extra = _extra_mapping(item, label=f"normalized sequence row {row_number}")
            _require(extra["Sequence"] == sequence, "provenance Sequence differs from entity")
            provenance_id = expected["provenance_id"]
            assert isinstance(provenance_id, str)
            owner = provenance_owners.setdefault(provenance_id, sequence_id)
            _require(owner == sequence_id, "one provenance row owns multiple sequences")
            key = (sequence_id, provenance_id)
            _require(key not in provenance, "duplicate sequence provenance")
            provenance[key] = (expected, semantic)
            expected_rows.append(expected)
    expected_rows.sort(key=lambda row: (row["sequence_id"], row["provenance_id"]))
    _require(
        list(membership_rows) == expected_rows,
        "study membership rows differ from independently derived provenance memberships",
    )
    for index, row in enumerate(membership_rows, start=1):
        _exact_fields(row, _MEMBERSHIP_FIELDS, label=f"study membership row {index}")
    return provenance, sequences


def _verify_ledger_links(
    assay_doc: JsonlDocument,
    ledger_rows: Sequence[dict[str, Any]],
    provenance: Mapping[tuple[str, str], tuple[dict[str, Any], bytes]],
    sequences: Mapping[str, str],
    config: FrozenConfig,
) -> None:
    _require(
        [row.get("observation_id") for row in ledger_rows]
        == sorted(row.get("observation_id") for row in ledger_rows),
        "endpoint ledger is not sorted by observation ID",
    )
    by_input_line: dict[int, dict[str, Any]] = {}
    observation_ids: set[str] = set()
    assay_hashes: set[str] = set()
    for index, ledger in enumerate(ledger_rows, start=1):
        _exact_fields(ledger, _LEDGER_FIELDS, label=f"endpoint ledger row {index}")
        input_line = ledger["input_line"]
        _require(isinstance(input_line, int) and not isinstance(input_line, bool), "bad input line")
        _require(input_line not in by_input_line, "endpoint ledger repeats an input line")
        _require(1 <= input_line <= len(assay_doc.rows), "endpoint ledger input line out of range")
        by_input_line[input_line] = ledger
        assay = assay_doc.rows[input_line - 1]
        raw_line = assay_doc.raw_lines[input_line - 1]
        _exact_fields(assay, _ASSAY_FIELDS, label=f"normalized assay line {input_line}")
        observation_id = _sha256_bytes(b"amp-challenge:normalized-assay:v1\0" + raw_line)
        assay_hash = _sha256_bytes(raw_line)
        _require(ledger["observation_id"] == observation_id, "observation ID mismatch")
        _require(ledger["assay_row_sha256"] == assay_hash, "assay row hash mismatch")
        _require(observation_id not in observation_ids, "duplicate observation ID")
        _require(assay_hash not in assay_hashes, "duplicate normalized assay row")
        observation_ids.add(observation_id)
        assay_hashes.add(assay_hash)
        sequence_id = assay["sequence_id"]
        _require(
            isinstance(sequence_id, str)
            and sequence_id in sequences
            and assay["sequence"] == sequences[sequence_id],
            "normalized assay sequence differs from its sequence entity",
        )
        _require(ledger["sequence_id"] == sequence_id, "ledger sequence linkage mismatch")
        provenance_raw = assay["provenance"]
        _require(isinstance(provenance_raw, dict), "assay provenance is not an object")
        provenance_id, _, _, _, semantic = _provenance_identity(
            provenance_raw, config, label=f"normalized assay line {input_line}"
        )
        membership_match = provenance.get((sequence_id, provenance_id))
        _require(membership_match is not None, "assay provenance is absent from sequence entity")
        assert membership_match is not None
        membership, expected_semantic = membership_match
        _require(semantic == expected_semantic, "assay provenance differs from sequence provenance")
        direct = {
            "schema_version": 1,
            "sequence_id": sequence_id,
            "endpoint": assay["endpoint"],
            "source_field": (
                "Target_Organism" if assay["endpoint"] == "mic" else "Hemolytic_activity"
            ),
            "provenance_id": provenance_id,
            "source_record_key": membership["source_record_key"],
            "source_record_id": membership["source_record_id"],
            "source_row_number": membership["source_row_number"],
            "source_target": assay["strain"],
            "source_organism": assay["organism"],
            "source_gram": assay["gram"],
            "source_assay": assay["assay"],
            "source_conditions": _source_conditions(assay["assay"]),
            "measurement_status": "accepted_parser_v7",
            "measurement": _measurement_from_assay(assay),
            "exposure_concentration": assay["exposure_concentration"],
            "source_text": assay["source_text"],
            "mapping_version": config.mapping_version,
            "blood_organism_mapping_version": config.blood_mapping_version,
            "resistance_status": "not_inferred",
            "study_status": membership["study_status"],
            "study_keys": membership["study_keys"],
            "study_review_codes": membership["study_review_codes"],
        }
        for field, expected in direct.items():
            _require(ledger[field] == expected, f"ledger field {field} differs from input line")
        semantics = _derive_assay_semantics(assay, input_line=input_line, config=config)
        for field, expected in semantics.items():
            _require(ledger[field] == expected, f"derived ledger field {field} mismatch")
        normalized_target = semantics["normalized_target"]
        normalized_organism = semantics["normalized_organism"]
        context_id = _json_framed_digest(
            "amp-challenge:endpoint-context:v1",
            assay["endpoint"],
            normalized_target,
            normalized_organism,
        )
        _require(ledger["context_id"] == context_id, "context ID mismatch")
        assay_context_id = _json_framed_digest(
            "amp-challenge:assay-context:v1",
            sequence_id,
            assay["endpoint"],
            context_id,
            tuple(_source_conditions(assay["assay"])),
            _exposure_identity(assay["exposure_concentration"]),
        )
        _require(ledger["assay_context_id"] == assay_context_id, "assay-context ID mismatch")
    _require(
        set(by_input_line) == set(range(1, len(assay_doc.rows) + 1)), "assays not covered once"
    )


def _verify_contexts(
    ledger_rows: Sequence[dict[str, Any]], context_rows: Sequence[dict[str, Any]]
) -> None:
    _require(
        [row.get("context_id") for row in context_rows]
        == sorted(row.get("context_id") for row in context_rows),
        "contexts are not sorted by context ID",
    )
    contexts: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(context_rows, start=1):
        _exact_fields(row, _CONTEXT_FIELDS, label=f"context row {index}")
        _require(
            type(row["schema_version"]) is int and row["schema_version"] == 1,
            f"context row {index} schema version mismatch",
        )
        context_id = row["context_id"]
        _require(isinstance(context_id, str) and context_id not in contexts, "duplicate context ID")
        contexts[context_id] = row
        recomputed = _json_framed_digest(
            "amp-challenge:endpoint-context:v1",
            row["endpoint"],
            row["normalized_target"],
            row["normalized_organism"],
        )
        _require(context_id == recomputed, "context table has an invalid context ID")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ledger in ledger_rows:
        grouped[ledger["context_id"]].append(ledger)
    _require(set(grouped) == set(contexts), "context table and ledger IDs differ")
    for context_id, members in grouped.items():
        context = contexts[context_id]
        for field in _SHARED_CONTEXT_FIELDS:
            values = {_canonical_compact(row[field]) for row in members}
            _require(len(values) == 1, f"context members disagree on {field}")
            _require(context[field] == members[0][field], f"context table disagrees on {field}")
        expected = {
            "observations": len({row["observation_id"] for row in members}),
            "sequences": len({row["sequence_id"] for row in members}),
            "source_records": len({row["provenance_id"] for row in members}),
            "source_grams": sorted({row["source_gram"] for row in members}),
            "gram_resolutions": sorted({row["gram_resolution"] for row in members}),
            "source_target_variants": sorted(
                {row["source_target"] for row in members if row["source_target"] is not None}
            ),
            "source_organism_variants": sorted(
                {row["source_organism"] for row in members if row["source_organism"] is not None}
            ),
        }
        for field, value in expected.items():
            _require(context[field] == value, f"context aggregate {field} mismatch")


def _counter(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _flattened_counter(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item) for row in rows for item in row[field]).items()))


def _reviewed_anomalies(
    membership_rows: Sequence[dict[str, Any]],
    ledger_rows: Sequence[dict[str, Any]],
    config: FrozenConfig,
) -> list[dict[str, Any]]:
    observations_by_record = Counter(row["source_record_id"] for row in ledger_rows)
    result: list[dict[str, Any]] = []
    for anomaly in config.study_anomalies:
        if anomaly["scope"] == "key":
            matched = [row for row in membership_rows if anomaly["selector"] in row["study_keys"]]
        else:
            matched = [
                row for row in membership_rows if row["source_record_id"] == anomaly["selector"]
            ]
        records = sorted({row["source_record_id"] for row in matched})
        keys = sorted({key for row in matched for key in row["study_keys"]})
        observations = sum(observations_by_record[record] for record in records)
        _require(records == anomaly["expected_source_record_ids"], "study anomaly records changed")
        _require(keys == anomaly["expected_study_keys"], "study anomaly keys changed")
        _require(
            observations == anomaly["expected_observations"], "study anomaly observations changed"
        )
        _require(
            all(anomaly["code"] in row["study_review_codes"] for row in matched),
            "study anomaly code missing",
        )
        result.append(
            {
                "scope": anomaly["scope"],
                "selector": anomaly["selector"],
                "code": anomaly["code"],
                "policy": anomaly["policy"],
                "source_record_ids": records,
                "study_keys": keys,
                "discordant_source_record_ids": anomaly["discordant_source_record_ids"],
                "memberships": len(matched),
                "sequences": len({row["sequence_id"] for row in matched}),
                "observations": observations,
            }
        )
    return result


def _verify_audit(
    audit: Mapping[str, Any],
    ledger_rows: Sequence[dict[str, Any]],
    context_rows: Sequence[dict[str, Any]],
    membership_rows: Sequence[dict[str, Any]],
    sequence_ids: set[str],
    config: FrozenConfig,
) -> None:
    _require(
        set(audit)
        == {
            "schema_version",
            "artifact",
            "status",
            "input",
            "contexts",
            "studies",
            "invariants",
            "limitations",
        },
        "audit root schema mismatch",
    )
    endpoint_counts = _counter(ledger_rows, "endpoint")
    expected_input = {
        "normalized_parser_id": config.parser_id,
        "normalized_schema_version": config.parser_schema,
        "unique_sequences": config.unique_sequences,
        "assay_observations": config.assay_observations,
        "endpoint_counts": dict(sorted(config.endpoint_counts.items())),
    }
    _require(
        type(audit.get("schema_version")) is int and audit["schema_version"] == 1,
        "audit schema version mismatch",
    )
    _require(audit.get("artifact") == "dramp_endpoint_context_audit", "audit artifact mismatch")
    _require(
        audit.get("status") == "development_sidecar_not_a_new_parser_or_untouched_evaluation_panel",
        "audit development status mismatch",
    )
    limitations = audit.get("limitations")
    _require(
        isinstance(limitations, list)
        and len(limitations) == len(_REQUIRED_LIMITATIONS)
        and set(limitations) == _REQUIRED_LIMITATIONS,
        "audit limitations changed or are incomplete",
    )
    _require(audit.get("input") == expected_input, "audit input summary mismatch")
    _require(
        endpoint_counts == expected_input["endpoint_counts"], "ledger endpoint histogram mismatch"
    )
    contexts = audit.get("contexts")
    _require(isinstance(contexts, dict), "audit contexts is not an object")
    composite = [
        row for row in context_rows if row["mapping_status"] == "ambiguous_composite_target"
    ]
    expected_contexts = {
        "unique_contexts": len(context_rows),
        "mapping_counts": _counter(ledger_rows, "mapping_status"),
        "gram_resolution_counts": _counter(ledger_rows, "gram_resolution"),
        "configured_target_observations": dict(
            sorted(
                Counter(
                    row["canonical_target"]
                    for row in ledger_rows
                    if row["canonical_target"] is not None
                ).items()
            )
        ),
        "composite_contexts": len(composite),
        "composite_observations": sum(row["observations"] for row in composite),
        "condition_bearing_observations": sum(
            bool(row["source_conditions"]) for row in ledger_rows
        ),
        "eligible_task_counts": _flattened_counter(ledger_rows, "eligible_tasks"),
        "exclusion_counts": _flattened_counter(ledger_rows, "exclusion_codes"),
        "blood_organism_resolution_counts": _counter(ledger_rows, "blood_organism_resolution"),
        "strain_identifier_review_observation_counts": _flattened_counter(
            ledger_rows, "strain_identifier_review_ids"
        ),
    }
    _require(expected_contexts == _EXPECTED_CONTEXT_CENSUS, "frozen context census changed")
    _require(contexts == expected_contexts, "audit context counts differ from artifacts")
    key_to_sequences: dict[str, set[str]] = defaultdict(set)
    statuses_by_sequence: dict[str, set[str]] = defaultdict(set)
    sequences_by_status: dict[str, set[str]] = defaultdict(set)
    for row in membership_rows:
        statuses_by_sequence[row["sequence_id"]].add(row["study_status"])
        sequences_by_status[row["study_status"]].add(row["sequence_id"])
        for key in row["study_keys"]:
            key_to_sequences[key].add(row["sequence_id"])
    singleton_keys = {key for key in key_to_sequences if key.startswith("source-record:")}
    reviewed = _reviewed_anomalies(membership_rows, ledger_rows, config)
    expected_studies = {
        "provenance_memberships": len(membership_rows),
        "membership_status_counts": _counter(membership_rows, "study_status"),
        "sequence_status_counts": {
            key: len(value) for key, value in sorted(sequences_by_status.items())
        },
        "multi_status_sequences": sum(len(value) > 1 for value in statuses_by_sequence.values()),
        "unique_study_keys": len(key_to_sequences),
        "shared_study_keys": sum(len(value) > 1 for value in key_to_sequences.values()),
        "source_record_singleton_keys": len(singleton_keys),
        "maximum_sequences_per_study_key": max(map(len, key_to_sequences.values())),
        "ignored_pubmed_token_counts": _flattened_counter(membership_rows, "ignored_pubmed_tokens"),
        "review_code_membership_counts": _flattened_counter(membership_rows, "study_review_codes"),
        "reviewed_anomalies": reviewed,
    }
    _require(
        {key: value for key, value in expected_studies.items() if key != "reviewed_anomalies"}
        == _EXPECTED_STUDY_CENSUS,
        "frozen study census changed",
    )
    _require(audit.get("studies") == expected_studies, "audit study counts differ from artifacts")
    membership_sequence_ids = {row["sequence_id"] for row in membership_rows}
    expected_invariants = {
        "accepted_assays_covered_once": True,
        "all_assay_provenance_links_resolved": True,
        "all_sequences_have_study_membership": membership_sequence_ids == sequence_ids,
        "missing_study_keys_are_record_singletons": all(
            len(key_to_sequences[key]) == 1 for key in singleton_keys
        ),
        "reviewed_study_anomalies_match_config": True,
        "normalized_inputs_unchanged_during_read": True,
    }
    _require(all(expected_invariants.values()), "independently recomputed invariant failed")
    _require(audit.get("invariants") == expected_invariants, "declared audit invariants mismatch")


def _one_line(ledger_rows: Sequence[dict[str, Any]], input_line: int) -> dict[str, Any]:
    matches = [row for row in ledger_rows if row["input_line"] == input_line]
    _require(len(matches) == 1, f"expected exactly one ledger row for input line {input_line}")
    return matches[0]


def _verify_sentinels(
    ledger_rows: Sequence[dict[str, Any]], membership_rows: Sequence[dict[str, Any]]
) -> None:
    for input_line, record_id, source_row, target, reason in _COMPOSITE_SENTINELS:
        row = _one_line(ledger_rows, input_line)
        _require(
            row["source_record_id"] == record_id
            and row["source_row_number"] == source_row
            and row["source_target"] == target,
            f"composite sentinel source changed at input line {input_line}",
        )
        _require(row["mapping_status"] == "ambiguous_composite_target", "composite admitted")
        _require(reason in row["composite_reason_codes"], "composite reason disappeared")
        _require(
            row["eligible_tasks"] == [] and row["canonical_target"] is None, "composite eligible"
        )
    for input_line, record_id, target in _NONCOMPOSITE_SENTINELS:
        row = _one_line(ledger_rows, input_line)
        _require(
            row["source_record_id"] == record_id and row["source_target"] == target,
            f"non-composite sentinel source changed at line {input_line}",
        )
        _require(row["mapping_status"] == "unmapped_target", "singular target became composite")
        _require(row["composite_reason_codes"] == [], "singular target has composite reasons")
    aliased = [row for row in ledger_rows if row["source_target"] == "huamn red blood cells"]
    _require(
        {row["input_line"] for row in aliased} == _ALIASED_BLOOD_LINES, "blood alias set changed"
    )
    for row in aliased:
        _require(row["endpoint"] == "hemolysis_percent", "blood alias endpoint changed")
        _require(row["source_organism"] is None, "blood alias unexpectedly parser-resolved")
        _require(row["resolved_blood_organism"] == "human", "blood alias resolution changed")
        _require(
            row["blood_organism_resolution"] == "reviewed_exact_alias", "blood alias policy changed"
        )
        _require(
            row["eligible_tasks"] == ["human_hemolysis_percent_at_dose"], "blood alias task changed"
        )
        _require(row["exclusion_codes"] == [], "blood alias was excluded")
    unresolved = [
        row for row in ledger_rows if row.get("blood_organism_resolution") == "unresolved"
    ]
    _require(
        {row["input_line"] for row in unresolved} == set(_UNRESOLVED_BLOOD_SENTINELS),
        "unresolved blood set changed",
    )
    for row in unresolved:
        record_id, source_row, target = _UNRESOLVED_BLOOD_SENTINELS[row["input_line"]]
        _require(
            (row["source_record_id"], row["source_row_number"], row["source_target"])
            == (record_id, source_row, target),
            "unresolved blood sentinel source changed",
        )
        _require(row["resolved_blood_organism"] is None, "unresolved blood was resolved")
        _require(row["eligible_tasks"] == [], "unresolved blood became eligible")
        _require(row["exclusion_codes"] == ["unresolved_blood_species"], "blood exclusion changed")
    multi_pmid = [row for row in membership_rows if row["source_record_id"] == "DRAMP03870"]
    _require(len(multi_pmid) == 1, "DRAMP03870 study membership changed")
    _require(
        multi_pmid[0]["study_keys"] == ["pmid:16041366", "pmid:17052614"]
        and multi_pmid[0]["study_review_codes"] == ["multi_pmid_citation_delimiter_missing"],
        "multi-PMID study sentinel changed",
    )
    discordant = [row for row in membership_rows if "pmid:25558400" in row["study_keys"]]
    _require(
        {row["source_record_id"] for row in discordant}
        == {"DRAMP18377", "DRAMP18378", "DRAMP18379", "DRAMP18380"},
        "discordant-citation study union changed",
    )
    _require(
        all(row["study_review_codes"] == ["discordant_citation_metadata"] for row in discordant),
        "discordant-citation review code changed",
    )


def _verify_no_path_leakage(run: Path, forbidden_prefixes: Sequence[bytes]) -> None:
    for relative in _TOP_FILES:
        payload = _read_regular_file(run / relative, label=f"leakage scan {relative}")
        for prefix in forbidden_prefixes:
            _require(prefix not in payload, f"artifact {relative} leaks forbidden path prefix")


def _verify_one_twin(
    *,
    run: Path,
    normalized_run: Path,
    config: FrozenConfig,
    config_path: Path,
    repo_root: Path,
    expected_git_commit: str | None,
    forbidden_prefixes: Sequence[bytes],
) -> dict[str, Any]:
    top_manifest_sha256 = _verify_top_manifest(run)
    sequence_doc, assay_doc, data_manifest_sha256 = _verify_data_run(normalized_run, config)
    frozen_manifest_sha256 = _verify_frozen_input_manifest(
        run, normalized_run, expected_data_manifest_sha256=data_manifest_sha256
    )
    code_manifest_sha256 = _verify_code_manifest(run, repo_root, config_path)
    output = run / "endpoint_context"
    artifact_names = (
        "endpoint_context_ledger.jsonl",
        "contexts.jsonl",
        "study_membership.jsonl",
        "audit.json",
    )
    artifact_digests = {
        name: _sha256_bytes(_read_regular_file(output / name, label=f"sidecar {name}"))
        for name in artifact_names
    }
    ledger_doc = _read_jsonl(
        output / "endpoint_context_ledger.jsonl",
        label="endpoint-context ledger",
        require_canonical=True,
    )
    context_doc = _read_jsonl(
        output / "contexts.jsonl", label="endpoint contexts", require_canonical=True
    )
    membership_doc = _read_jsonl(
        output / "study_membership.jsonl",
        label="study membership",
        require_canonical=True,
    )
    audit = _read_canonical_json(output / "audit.json", label="sidecar audit")
    manifest = _verify_manifest(
        output,
        config,
        data_manifest_sha256=data_manifest_sha256,
        code_manifest_sha256=code_manifest_sha256,
        artifact_digests=artifact_digests,
        expected_git_commit=expected_git_commit,
    )
    provenance, sequences = _verify_memberships(sequence_doc, membership_doc.rows, config)
    _verify_ledger_links(assay_doc, ledger_doc.rows, provenance, sequences, config)
    _verify_contexts(ledger_doc.rows, context_doc.rows)
    _verify_audit(
        audit,
        ledger_doc.rows,
        context_doc.rows,
        membership_doc.rows,
        set(sequences),
        config,
    )
    _require(
        manifest["policies"]["study_anomalies"]
        == _reviewed_anomalies(membership_doc.rows, ledger_doc.rows, config),
        "manifest study anomaly projection differs from independently derived artifacts",
    )
    _verify_sentinels(ledger_doc.rows, membership_doc.rows)
    counts = manifest["counts"]
    expected_counts = {
        "unique_sequences": len(sequences),
        "assay_observations": len(ledger_doc.rows),
        "contexts": len(context_doc.rows),
        "study_memberships": len(membership_doc.rows),
    }
    _require(counts == expected_counts, "sidecar manifest counts differ from artifacts")
    _verify_no_path_leakage(run, forbidden_prefixes)
    return {
        "top_manifest_sha256": top_manifest_sha256,
        "code_manifest_sha256": code_manifest_sha256,
        "frozen_input_manifest_sha256": frozen_manifest_sha256,
        "normalized_data_manifest_sha256": data_manifest_sha256,
        "git_commit": manifest["provenance"]["git_commit"],
        "counts": expected_counts,
        "artifact_sha256": dict(sorted(artifact_digests.items())),
    }


def _verify_twin_bytes(left: Path, right: Path) -> None:
    left_inventory = _regular_inventory(left)
    right_inventory = _regular_inventory(right)
    _require(left_inventory == right_inventory == _TOP_FILES, "twin inventories differ")
    for relative in sorted(left_inventory):
        left_payload = _read_regular_file(left / relative, label=f"left twin {relative}")
        right_payload = _read_regular_file(right / relative, label=f"right twin {relative}")
        _require(left_payload == right_payload, f"twins are not byte-identical: {relative}")


def verify_endpoint_context_twins(
    *,
    twin_root: str | Path,
    normalized_twin_root: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str | None = None,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify both production twins and return a path-free deterministic receipt."""

    twins = Path(twin_root).resolve(strict=True)
    normalized_twins = Path(normalized_twin_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    repository = Path(repo_root).resolve(strict=True)
    _require(config_file.is_relative_to(repository), "config must be inside the repository")
    if expected_git_commit is not None:
        _require(_GIT_SHA_RE.fullmatch(expected_git_commit) is not None, "bad expected Git commit")
    config = _load_config(config_file)
    encoded_prefixes = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(prefix, "forbidden path prefixes must not be empty")
        encoded_prefixes.append(prefix.encode("utf-8"))
    runs = [twins / "0", twins / "1"]
    data_runs = [normalized_twins / "0", normalized_twins / "1"]
    for index, path in enumerate([*runs, *data_runs]):
        _require(
            path.is_dir() and not path.is_symlink(), f"required twin directory is missing: {index}"
        )
    _verify_twin_bytes(runs[0], runs[1])
    normalized_hashes = []
    receipts = []
    for run, data_run in zip(runs, data_runs, strict=True):
        receipts.append(
            _verify_one_twin(
                run=run,
                normalized_run=data_run,
                config=config,
                config_path=config_file,
                repo_root=repository,
                expected_git_commit=expected_git_commit,
                forbidden_prefixes=encoded_prefixes,
            )
        )
        normalized_hashes.append(
            {
                name: _sha256_bytes(
                    _read_regular_file(data_run / name, label=f"normalized twin {name}")
                )
                for name in (
                    "SHA256SUMS",
                    "normalized/sequences.jsonl",
                    "normalized/assays.jsonl",
                    "normalized/summary.json",
                )
            }
        )
    _require(normalized_hashes[0] == normalized_hashes[1], "normalized input twins differ")
    _require(receipts[0] == receipts[1], "semantic twin verification receipts differ")
    receipt = receipts[0]
    return {
        "schema_version": 1,
        "artifact": "dramp_endpoint_context_independent_verification",
        "status": "passed",
        "checks": {
            "top_manifests_valid": True,
            "twins_byte_identical": True,
            "normalized_inputs_config_pinned": True,
            "jsonl_canonical_lf": True,
            "counts_and_invariants_recomputed": True,
            "protocol_sentinels_exact": True,
            "json_framed_ids_recomputed": True,
            "scratch_paths_absent": True,
        },
        **receipt,
    }


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite verification receipt: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        staging.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.link(staging, path)
        staging.unlink()
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-root", type=Path, required=True)
    parser.add_argument("--normalized-twin-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify_endpoint_context_twins(
        twin_root=args.twin_root,
        normalized_twin_root=args.normalized_twin_root,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    if args.output is not None:
        _write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through Slurm CLI
    raise SystemExit(main())

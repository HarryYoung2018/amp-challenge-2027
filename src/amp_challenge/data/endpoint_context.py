"""Build a fail-closed endpoint-context and study-membership sidecar.

The accepted DRAMP parser-v7 files are immutable.  This module derives a
content-addressed sidecar without changing an endpoint value or admitting a
new assay.  Target mappings are deliberately narrow: configured species rules
may identify support for an existing endpoint model, while every other literal
remains explicit and unresolved.  Study keys are grouping metadata only and
must never be exposed to a predictive model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from amp_challenge.benchmarks.oracle_gate1 import derive_activity_label
from amp_challenge.data.records import CensoredValue, CensorRelation
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_PMID = re.compile(r"[1-9][0-9]*")
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
    r"(?:,|;|\band\b|\bor\b)\s*(?:MRSA|MSSA|VRE|VRSA)\b",
    re.IGNORECASE,
)
_GRAMS = frozenset({"positive", "negative", "unknown"})
_ENDPOINTS = frozenset({"mic", "hc50", "hemolysis_percent"})
_SEQUENCE_ROW_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSAY_ROW_FIELDS = frozenset(
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
_REQUIRED_PROVENANCE_EXTRA_FIELDS = frozenset(
    {
        "DRAMP_ID",
        "Sequence",
        "source_sha256",
        "source_version",
    }
)
_NONHUMAN_BLOOD_SPECIES = frozenset(
    {"rabbit", "mouse", "rat", "pig", "cattle", "sheep", "goat", "horse", "chicken"}
)

StudyStatus = Literal[
    "explicit_pmid",
    "explicit_pmid_with_ignored_tokens",
    "reference_title_fallback",
    "source_record_singleton",
]


@dataclass(frozen=True, slots=True)
class TargetRule:
    name: str
    strain_regex: str
    expected_gram: Literal["positive", "negative"]

    def occurrences(self, text: str) -> int:
        return sum(1 for _ in re.finditer(self.strain_regex, text))


@dataclass(frozen=True, slots=True)
class BloodOrganismAlias:
    source_target: str
    canonical_organism: str


@dataclass(frozen=True, slots=True)
class StudyAnomaly:
    scope: Literal["key", "record"]
    selector: str
    code: str
    policy: str
    expected_source_record_ids: tuple[str, ...]
    expected_study_keys: tuple[str, ...]
    discordant_source_record_ids: tuple[str, ...]
    expected_observations: int


@dataclass(frozen=True, slots=True)
class StrainIdentifierReview:
    identifier: str
    target_regex: str
    code: str
    policy: str

    def matches(self, target: str) -> bool:
        return re.search(self.target_regex, target) is not None


@dataclass(frozen=True, slots=True)
class EndpointContextConfig:
    path: Path
    normalized_parser_id: str
    normalized_schema_version: int
    source_name: str
    source_sha256: str
    source_version: str
    normalized_sequences_sha256: str
    normalized_assays_sha256: str
    normalized_summary_sha256: str
    expected_unique_sequences: int
    expected_assay_observations: int
    expected_endpoint_counts: Mapping[str, int]
    activity_threshold_um: float
    pubmed_delimiter: str
    citation_missing_values: frozenset[str]
    mapping_version: str
    blood_organism_mapping_version: str
    blood_organism_aliases: tuple[BloodOrganismAlias, ...]
    study_anomalies: tuple[StudyAnomaly, ...]
    strain_identifier_reviews: tuple[StrainIdentifierReview, ...]
    targets: tuple[TargetRule, ...]


@dataclass(frozen=True, slots=True)
class StudyMembership:
    sequence_id: str
    provenance_id: str
    source: str
    source_version: str
    source_sha256: str
    source_record_id: str
    source_row_number: int
    source_record_key: str
    study_status: StudyStatus
    study_keys: tuple[str, ...]
    pmids: tuple[str, ...]
    ignored_pubmed_tokens: tuple[str, ...]
    citation_reference: str | None
    citation_title: str | None
    study_review_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointContextExecution:
    output_dir: Path
    assay_ledger_path: Path
    contexts_path: Path
    study_membership_path: Path
    audit_path: Path
    manifest_path: Path
    assay_observations: int
    unique_sequences: int
    summary: Mapping[str, object]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_id_frame(value: object) -> object:
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stable ID parts must not contain non-finite floats")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, tuple):
        return ["tuple", [_typed_id_frame(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_id_frame(item) for item in value]]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("stable ID mapping keys must be strings")
        return [
            "mapping",
            [[key, _typed_id_frame(value[key])] for key in sorted(value)],
        ]
    raise TypeError(f"unsupported stable ID part type: {type(value).__name__}")


def _stable_digest(namespace: str, *parts: object) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("stable ID namespace must be a non-empty string")
    payload = json.dumps(
        {
            "namespace": _typed_id_frame(namespace),
            "parts": [_typed_id_frame(part) for part in parts],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_regular_file(path: str | Path, *, name: str) -> bytes:
    source = Path(path).resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    before_fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_fingerprint != after_fingerprint or len(payload) != before.st_size:
        raise ValueError(f"{name} changed while it was being read: {source}")
    return payload


def _read_json_object_bytes(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _read_jsonl_bytes(payload: bytes, *, name: str) -> list[tuple[dict[str, object], bytes]]:
    if not payload:
        raise ValueError(f"{name} must contain at least one row")
    if not payload.endswith(b"\n"):
        raise ValueError(f"{name} must end with one LF byte")
    if b"\r" in payload:
        raise ValueError(f"{name} must use LF line endings without CR bytes")
    lines = payload[:-1].split(b"\n")
    rows: list[tuple[dict[str, object], bytes]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} line {line_number} must be UTF-8") from error
        if not line:
            raise ValueError(f"{name} line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} is not a JSON object")
        rows.append((value, raw_line))
    return rows


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"endpoint-context config {field!r} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"endpoint-context config {field!r} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"endpoint-context config {field!r} must be a non-negative integer")
    return value


def _string_array(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a string array")
    if not all(isinstance(entry, str) and entry.strip() for entry in value):
        raise ValueError(f"{field} has an invalid entry")
    result = tuple(cast(list[str], value))
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} must be sorted and unique")
    return result


def load_config(path: str | Path) -> EndpointContextConfig:
    config_path = Path(path).resolve(strict=True)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
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
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"unexpected endpoint-context config key(s): {sorted(unexpected)}")
    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("endpoint-context config schema_version must be 1")

    parser_schema = raw.get("normalized_schema_version")
    if isinstance(parser_schema, bool) or not isinstance(parser_schema, int) or parser_schema < 1:
        raise ValueError("normalized_schema_version must be a positive integer")
    source_sha256 = _nonempty_string(raw.get("source_sha256"), field="source_sha256")
    if _SHA256.fullmatch(source_sha256) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    normalized_hashes: dict[str, str] = {}
    for field in (
        "normalized_sequences_sha256",
        "normalized_assays_sha256",
        "normalized_summary_sha256",
    ):
        value = _nonempty_string(raw.get(field), field=field)
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        normalized_hashes[field] = value

    endpoint_counts_raw = raw.get("expected_endpoint_counts")
    if not isinstance(endpoint_counts_raw, dict) or set(endpoint_counts_raw) != _ENDPOINTS:
        raise ValueError(f"expected_endpoint_counts must contain exactly {sorted(_ENDPOINTS)}")
    endpoint_counts = {
        endpoint: _nonnegative_int(endpoint_counts_raw[endpoint], field=f"{endpoint}_count")
        for endpoint in sorted(_ENDPOINTS)
    }
    if sum(endpoint_counts.values()) != raw.get("expected_assay_observations"):
        raise ValueError("expected endpoint counts do not sum to expected_assay_observations")

    threshold_raw = raw.get("activity_threshold_um")
    if (
        isinstance(threshold_raw, bool)
        or not isinstance(threshold_raw, int | float)
        or not math.isfinite(threshold_raw)
        or threshold_raw <= 0
    ):
        raise ValueError("activity_threshold_um must be finite and positive")
    threshold = float(threshold_raw)
    delimiter = _nonempty_string(raw.get("pubmed_delimiter"), field="pubmed_delimiter")
    missing_raw = raw.get("citation_missing_values")
    if (
        not isinstance(missing_raw, list)
        or not missing_raw
        or not all(isinstance(item, str) for item in missing_raw)
    ):
        raise ValueError("citation_missing_values must be a non-empty string array")
    missing = frozenset(_normalize_whitespace(item).casefold() for item in missing_raw)
    if len(missing) != len(missing_raw):
        raise ValueError("citation_missing_values must be unique after normalization")
    if "" not in missing:
        raise ValueError("citation_missing_values must explicitly include the empty string")

    blood_alias_raw = raw.get("blood_organism_alias")
    if not isinstance(blood_alias_raw, list):
        raise ValueError("blood_organism_alias must be an array of exact alias tables")
    blood_aliases: list[BloodOrganismAlias] = []
    for index, item in enumerate(blood_alias_raw):
        if not isinstance(item, dict) or set(item) != {
            "source_target",
            "canonical_organism",
        }:
            raise ValueError(f"invalid blood_organism_alias table at index {index}")
        source_target = _nonempty_string(
            item["source_target"], field=f"blood_organism_alias[{index}].source_target"
        )
        if source_target != _normalize_whitespace(source_target):
            raise ValueError("blood organism aliases must already be whitespace-normalized")
        canonical_organism = _nonempty_string(
            item["canonical_organism"],
            field=f"blood_organism_alias[{index}].canonical_organism",
        )
        if canonical_organism not in {"human", *_NONHUMAN_BLOOD_SPECIES}:
            raise ValueError(
                f"blood organism alias {source_target!r} has unsupported canonical organism"
            )
        blood_aliases.append(
            BloodOrganismAlias(
                source_target=source_target,
                canonical_organism=canonical_organism,
            )
        )
    alias_targets = [item.source_target for item in blood_aliases]
    if len(alias_targets) != len(set(alias_targets)):
        raise ValueError("blood organism alias source targets must be unique")

    anomaly_raw = raw.get("study_anomaly", [])
    if not isinstance(anomaly_raw, list):
        raise ValueError("study_anomaly must be an array of reviewed anomaly tables")
    study_anomalies: list[StudyAnomaly] = []
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
    for index, item in enumerate(anomaly_raw):
        if not isinstance(item, dict) or set(item) != anomaly_fields:
            raise ValueError(f"invalid study_anomaly table at index {index}")
        scope = _nonempty_string(item["scope"], field=f"study_anomaly[{index}].scope")
        if scope not in {"key", "record"}:
            raise ValueError(f"study_anomaly[{index}].scope must be key or record")

        anomaly = StudyAnomaly(
            scope=cast(Literal["key", "record"], scope),
            selector=_nonempty_string(item["selector"], field=f"study_anomaly[{index}].selector"),
            code=_nonempty_string(item["code"], field=f"study_anomaly[{index}].code"),
            policy=_nonempty_string(item["policy"], field=f"study_anomaly[{index}].policy"),
            expected_source_record_ids=_string_array(
                item["expected_source_record_ids"],
                field=f"study_anomaly[{index}].expected_source_record_ids",
                allow_empty=False,
            ),
            expected_study_keys=_string_array(
                item["expected_study_keys"],
                field=f"study_anomaly[{index}].expected_study_keys",
                allow_empty=False,
            ),
            discordant_source_record_ids=_string_array(
                item["discordant_source_record_ids"],
                field=f"study_anomaly[{index}].discordant_source_record_ids",
                allow_empty=True,
            ),
            expected_observations=_positive_int(
                item["expected_observations"],
                field=f"study_anomaly[{index}].expected_observations",
            ),
        )
        if anomaly.scope == "key" and anomaly.selector not in anomaly.expected_study_keys:
            raise ValueError("key-scoped study anomaly selector must be an expected study key")
        if anomaly.scope == "record" and anomaly.expected_source_record_ids != (anomaly.selector,):
            raise ValueError("record-scoped study anomaly must select its sole source record")
        if not set(anomaly.discordant_source_record_ids).issubset(
            anomaly.expected_source_record_ids
        ):
            raise ValueError("discordant study records must be expected anomaly members")
        study_anomalies.append(anomaly)
    selectors = [(item.scope, item.selector) for item in study_anomalies]
    if len(selectors) != len(set(selectors)):
        raise ValueError("study anomaly selectors must be unique")

    strain_review_raw = raw.get("strain_identifier_review", [])
    if not isinstance(strain_review_raw, list):
        raise ValueError("strain_identifier_review must be an array of review tables")
    strain_identifier_reviews: list[StrainIdentifierReview] = []
    for index, item in enumerate(strain_review_raw):
        if not isinstance(item, dict) or set(item) != {
            "identifier",
            "target_regex",
            "code",
            "policy",
        }:
            raise ValueError(f"invalid strain_identifier_review table at index {index}")
        target_regex = _nonempty_string(
            item["target_regex"],
            field=f"strain_identifier_review[{index}].target_regex",
        )
        try:
            re.compile(target_regex)
        except re.error as error:
            raise ValueError(
                f"strain_identifier_review[{index}] has invalid target_regex: {error}"
            ) from error
        strain_identifier_reviews.append(
            StrainIdentifierReview(
                identifier=_nonempty_string(
                    item["identifier"],
                    field=f"strain_identifier_review[{index}].identifier",
                ),
                target_regex=target_regex,
                code=_nonempty_string(
                    item["code"], field=f"strain_identifier_review[{index}].code"
                ),
                policy=_nonempty_string(
                    item["policy"], field=f"strain_identifier_review[{index}].policy"
                ),
            )
        )
    strain_identifiers = [item.identifier for item in strain_identifier_reviews]
    if len(strain_identifiers) != len(set(strain_identifiers)):
        raise ValueError("strain identifier review IDs must be unique")

    target_raw = raw.get("target")
    if not isinstance(target_raw, list) or not target_raw:
        raise ValueError("at least one [[target]] rule is required")
    targets: list[TargetRule] = []
    for index, item in enumerate(target_raw):
        if not isinstance(item, dict) or set(item) != {"name", "strain_regex", "expected_gram"}:
            raise ValueError(f"invalid target table at index {index}")
        name = _nonempty_string(item["name"], field=f"target[{index}].name")
        pattern = _nonempty_string(item["strain_regex"], field=f"target[{index}].strain_regex")
        expected_gram = _nonempty_string(
            item["expected_gram"], field=f"target[{index}].expected_gram"
        )
        if expected_gram not in {"positive", "negative"}:
            raise ValueError(f"target {name!r} expected_gram must be positive or negative")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"target {name!r} has invalid strain_regex: {error}") from error
        targets.append(
            TargetRule(
                name=name,
                strain_regex=pattern,
                expected_gram=cast(Literal["positive", "negative"], expected_gram),
            )
        )
    names = [item.name for item in targets]
    if len(names) != len(set(names)):
        raise ValueError("target names must be unique")

    return EndpointContextConfig(
        path=config_path,
        normalized_parser_id=_nonempty_string(
            raw.get("normalized_parser_id"), field="normalized_parser_id"
        ),
        normalized_schema_version=parser_schema,
        source_name=_nonempty_string(raw.get("source_name"), field="source_name"),
        source_sha256=source_sha256,
        source_version=_nonempty_string(raw.get("source_version"), field="source_version"),
        normalized_sequences_sha256=normalized_hashes["normalized_sequences_sha256"],
        normalized_assays_sha256=normalized_hashes["normalized_assays_sha256"],
        normalized_summary_sha256=normalized_hashes["normalized_summary_sha256"],
        expected_unique_sequences=_positive_int(
            raw.get("expected_unique_sequences"), field="expected_unique_sequences"
        ),
        expected_assay_observations=_positive_int(
            raw.get("expected_assay_observations"), field="expected_assay_observations"
        ),
        expected_endpoint_counts=endpoint_counts,
        activity_threshold_um=threshold,
        pubmed_delimiter=delimiter,
        citation_missing_values=missing,
        mapping_version=_nonempty_string(raw.get("mapping_version"), field="mapping_version"),
        blood_organism_mapping_version=_nonempty_string(
            raw.get("blood_organism_mapping_version"),
            field="blood_organism_mapping_version",
        ),
        blood_organism_aliases=tuple(blood_aliases),
        study_anomalies=tuple(study_anomalies),
        strain_identifier_reviews=tuple(strain_identifier_reviews),
        targets=tuple(targets),
    )


def _normalize_whitespace(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def _extra_mapping(provenance: Mapping[str, object]) -> dict[str, str]:
    raw = provenance.get("extra")
    if not isinstance(raw, list):
        raise ValueError("normalized provenance extra must be a list")
    result: dict[str, str] = {}
    for index, item in enumerate(raw):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[0]
        ):
            raise ValueError(f"normalized provenance extra entry {index} is invalid")
        if item[0] in result:
            raise ValueError(f"normalized provenance repeats extra key {item[0]!r}")
        result[item[0]] = item[1]
    missing = sorted(_REQUIRED_PROVENANCE_EXTRA_FIELDS - set(result))
    if missing:
        raise ValueError(f"normalized provenance extra misses required key(s): {missing}")
    return result


def _provenance_identity(
    provenance: Mapping[str, object],
    *,
    config: EndpointContextConfig,
) -> tuple[str, str, int, dict[str, str]]:
    expected_fields = {"source", "record_id", "path", "row_number", "extra"}
    if set(provenance) != expected_fields:
        raise ValueError("normalized provenance has an unexpected schema")
    source = provenance.get("source")
    record_id = provenance.get("record_id")
    source_path = provenance.get("path")
    row_number = provenance.get("row_number")
    if source != config.source_name:
        raise ValueError(f"unsupported provenance source {source!r}")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("normalized provenance requires a source record_id")
    if record_id != record_id.strip():
        raise ValueError("normalized provenance source record_id must be edge-trimmed")
    if source_path is not None and (not isinstance(source_path, str) or not source_path.strip()):
        raise ValueError("normalized provenance path must be a non-empty string or null")
    if isinstance(row_number, bool) or not isinstance(row_number, int) or row_number < 2:
        raise ValueError("normalized provenance requires a valid source row_number")
    extra = _extra_mapping(provenance)
    if extra.get("source_version") != config.source_version:
        raise ValueError("normalized provenance source_version mismatch")
    if extra.get("source_sha256") != config.source_sha256:
        raise ValueError("normalized provenance source_sha256 mismatch")
    if extra.get("DRAMP_ID") != record_id:
        raise ValueError("normalized provenance DRAMP_ID does not match record_id")
    provenance_id = _stable_digest(
        "amp-challenge:source-provenance:v1",
        source,
        config.source_version,
        config.source_sha256,
        record_id,
        row_number,
    )
    return provenance_id, record_id, row_number, extra


def _provenance_semantic_sha256(
    provenance: Mapping[str, object],
    *,
    config: EndpointContextConfig,
) -> str:
    """Hash every provenance field except the environment-specific storage path."""

    provenance_id, record_id, row_number, extra = _provenance_identity(provenance, config=config)
    semantic = {
        "provenance_id": provenance_id,
        "source": config.source_name,
        "record_id": record_id,
        "row_number": row_number,
        "extra": extra,
    }
    return _sha256_bytes(_canonical_json(semantic).encode("utf-8"))


def _citation_value(
    raw: object,
    *,
    missing_values: frozenset[str],
) -> str | None:
    value = _normalize_whitespace(raw)
    return None if value.casefold() in missing_values else value


def derive_study_membership(
    *,
    sequence_id: str,
    provenance: Mapping[str, object],
    config: EndpointContextConfig,
) -> StudyMembership:
    """Derive conservative grouping-only study keys for one source record."""

    provenance_id, record_id, row_number, extra = _provenance_identity(provenance, config=config)
    pubmed_text = _normalize_whitespace(extra.get("Pubmed_ID", ""))
    raw_tokens = tuple(
        token.strip() for token in pubmed_text.split(config.pubmed_delimiter) if token.strip()
    )
    pmids = tuple(sorted({token for token in raw_tokens if _PMID.fullmatch(token)}, key=int))
    ignored = tuple(sorted({token for token in raw_tokens if _PMID.fullmatch(token) is None}))
    reference = _citation_value(
        extra.get("Reference", ""), missing_values=config.citation_missing_values
    )
    title = _citation_value(extra.get("Title", ""), missing_values=config.citation_missing_values)
    source_record_key = f"source-record:{provenance_id}"
    if pmids:
        study_keys = tuple(f"pmid:{pmid}" for pmid in pmids)
        status: StudyStatus = "explicit_pmid_with_ignored_tokens" if ignored else "explicit_pmid"
    elif reference is not None or title is not None:
        citation_key = _stable_digest(
            "amp-challenge:reference-title-study:v1",
            reference or "",
            title or "",
        )
        study_keys = (f"reference-title:{citation_key}",)
        status = "reference_title_fallback"
    else:
        # Missing study metadata is a unique record-level singleton.  It must
        # not connect unrelated records under one shared "unknown" key.
        study_keys = (source_record_key,)
        status = "source_record_singleton"
    review_codes = tuple(
        sorted(
            anomaly.code
            for anomaly in config.study_anomalies
            if (anomaly.scope == "record" and anomaly.selector == record_id)
            or (anomaly.scope == "key" and anomaly.selector in study_keys)
        )
    )
    return StudyMembership(
        sequence_id=sequence_id,
        provenance_id=provenance_id,
        source=config.source_name,
        source_version=config.source_version,
        source_sha256=config.source_sha256,
        source_record_id=record_id,
        source_row_number=row_number,
        source_record_key=source_record_key,
        study_status=status,
        study_keys=study_keys,
        pmids=pmids,
        ignored_pubmed_tokens=ignored,
        citation_reference=reference,
        citation_title=title,
        study_review_codes=review_codes,
    )


def classify_target_context(
    *,
    endpoint: str,
    target_text: str | None,
    source_gram: str,
    rules: Sequence[TargetRule],
) -> dict[str, object]:
    """Classify one literal without fuzzy matching or taxonomic inference."""

    normalized = None if target_text is None else _normalize_whitespace(target_text)
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
    if not normalized:
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

    occurrences = [(rule, rule.occurrences(normalized)) for rule in rules]
    total_occurrences = sum(count for _, count in occurrences)
    matched_rules = [rule for rule, count in occurrences if count]
    apparent_mentions = sum(
        match.group("genus").casefold() not in _NONTAXONOMIC_LEADING_WORDS
        for match in _APPARENT_BINOMIAL.finditer(normalized)
    )
    composite_reasons: list[str] = []
    if total_occurrences > 1:
        composite_reasons.append("multiple_configured_target_occurrences")
    if len(matched_rules) > 1:
        composite_reasons.append("multiple_configured_species")
    if apparent_mentions > 1:
        composite_reasons.append("multiple_apparent_taxa")
    if _MULTI_TARGET_LANGUAGE.search(normalized) is not None:
        composite_reasons.append("explicit_multiple_strains_or_isolates")
    if (
        _SUSCEPTIBLE_STATUS.search(normalized) is not None
        and _RESISTANT_STATUS.search(normalized) is not None
        and _STRAIN_GROUP_NOUN.search(normalized) is not None
    ):
        composite_reasons.append("mixed_resistance_groups")
    if _SECONDARY_RESISTANCE_GROUP.search(normalized) is not None:
        composite_reasons.append("secondary_resistance_group")
    if composite_reasons:
        return {
            "mapping_status": "ambiguous_composite_target",
            "canonical_target": None,
            "expected_gram": None,
            "gram_resolution": "unresolved",
            "target_domain": "unknown",
            "apparent_taxon_mentions": apparent_mentions,
            "configured_target_occurrences": total_occurrences,
            "composite_language_marker": True,
            "composite_reason_codes": sorted(composite_reasons),
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
    if source_gram == "unknown":
        gram_resolution = "missing_source_gram"
    elif source_gram == rule.expected_gram:
        gram_resolution = "concordant"
    else:
        gram_resolution = "conflict"
    return {
        "mapping_status": "mapped_single_supported_species",
        "canonical_target": rule.name,
        "expected_gram": rule.expected_gram,
        "gram_resolution": gram_resolution,
        "target_domain": "bacteria",
        "apparent_taxon_mentions": apparent_mentions,
        "configured_target_occurrences": total_occurrences,
        "composite_language_marker": False,
        "composite_reason_codes": [],
    }


def _resolve_blood_organism(
    *,
    endpoint: str,
    normalized_target: str | None,
    source_organism: str | None,
    aliases: Sequence[BloodOrganismAlias],
) -> tuple[str | None, str]:
    if endpoint == "mic":
        return None, "not_applicable_mic"
    if source_organism == "human":
        return "human", "parser_explicit_human"
    if source_organism in _NONHUMAN_BLOOD_SPECIES:
        return source_organism, "parser_explicit_nonhuman"
    if source_organism is None and normalized_target is not None:
        alias_map = {item.source_target: item.canonical_organism for item in aliases}
        if normalized_target in alias_map:
            return alias_map[normalized_target], "reviewed_exact_alias"
    return None, "unresolved"


def _blood_endpoint_task(endpoint: str, organism: str | None) -> str | None:
    if endpoint not in {"hc50", "hemolysis_percent"}:
        raise ValueError("blood endpoint task requires hc50 or hemolysis_percent")
    if organism == "human":
        prefix = "human"
    elif organism in _NONHUMAN_BLOOD_SPECIES:
        prefix = "nonhuman"
    else:
        return None
    suffix = "hc50_interval" if endpoint == "hc50" else "hemolysis_percent_at_dose"
    return f"{prefix}_{suffix}" + ("_aux" if prefix == "nonhuman" else "")


def _require_exact_fields(
    row: Mapping[str, object], *, expected: frozenset[str], name: str
) -> None:
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} schema mismatch; missing={missing}, extra={extra}")


def _optional_string(row: Mapping[str, object], field: str, *, name: str) -> str | None:
    value = row.get(field)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} {field} must be a string or null")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} {field} must not be empty or whitespace-only")
    return cast(str | None, value)


def _validate_normalized_assay_schema(
    row: Mapping[str, object], *, input_line: int
) -> tuple[str, str]:
    name = f"normalized assay line {input_line}"
    _require_exact_fields(row, expected=_ASSAY_ROW_FIELDS, name=name)
    endpoint = row["endpoint"]
    if not isinstance(endpoint, str) or endpoint not in _ENDPOINTS:
        raise ValueError(f"{name} has unsupported endpoint")
    source_gram = row["gram"]
    if not isinstance(source_gram, str) or source_gram not in _GRAMS:
        raise ValueError(f"{name} has invalid Gram value")
    for field in ("assay", "organism", "source_text", "strain"):
        _optional_string(row, field, name=name)
    if not isinstance(row["sequence"], str) or not isinstance(row["sequence_id"], str):
        raise ValueError(f"{name} sequence and sequence_id must be strings")
    if not isinstance(row["provenance"], dict):
        raise ValueError(f"{name} provenance must be an object")
    exposure = row["exposure_concentration"]
    if exposure is not None:
        if not isinstance(exposure, dict):
            raise ValueError(f"{name} exposure_concentration must be an object or null")
        _require_exact_fields(
            exposure,
            expected=_MEASUREMENT_FIELDS,
            name=f"{name} exposure_concentration",
        )
    return endpoint, source_gram


def _measurement(row: Mapping[str, object]) -> CensoredValue:
    relation_raw = row.get("relation")
    if not isinstance(relation_raw, str):
        raise ValueError("normalized assay relation must be a string")
    for field in ("lower_inclusive", "upper_inclusive"):
        if not isinstance(row.get(field), bool):
            raise ValueError(f"normalized assay {field} must be a boolean")

    def bound(field: str) -> float | None:
        raw = row.get(field)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"normalized assay {field} must be numeric or null")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"normalized assay {field} must be finite")
        return value

    unit_raw = row.get("unit")
    source_unit_raw = row.get("source_unit")
    raw_value = row.get("raw_value")
    for field, value in (
        ("unit", unit_raw),
        ("source_unit", source_unit_raw),
        ("raw_value", raw_value),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"normalized assay {field} must be a non-empty string or null")
    return CensoredValue(
        relation=cast(CensorRelation, relation_raw),
        lower=bound("lower"),
        upper=bound("upper"),
        lower_inclusive=cast(bool, row["lower_inclusive"]),
        upper_inclusive=cast(bool, row["upper_inclusive"]),
        unit=cast(str | None, unit_raw),
        raw=cast(str | None, raw_value),
        source_unit=cast(str | None, source_unit_raw),
    )


def _measurement_row(value: CensoredValue) -> dict[str, object]:
    return {
        "relation": value.relation,
        "lower": value.lower,
        "upper": value.upper,
        "lower_inclusive": value.lower_inclusive,
        "upper_inclusive": value.upper_inclusive,
        "unit": value.unit,
        "source_unit": value.source_unit,
        "raw_value": value.raw,
    }


def _exposure_identity(value: CensoredValue | None) -> tuple[object, ...] | None:
    """Return only the canonical exposure semantics that define an assay context."""
    if value is None:
        return None
    return (
        value.relation,
        value.lower,
        value.lower_inclusive,
        value.upper,
        value.upper_inclusive,
        value.unit,
    )


def _positive_bounds(value: CensoredValue) -> bool:
    bounds = [bound for bound in (value.lower, value.upper) if bound is not None]
    return bool(bounds) and all(bound > 0 for bound in bounds)


def _source_conditions(assay: object) -> tuple[str, ...]:
    if assay is None:
        return ()
    text = str(assay)
    marker = "; source condition: "
    if marker not in text:
        return ()
    return tuple(
        dict.fromkeys(
            item.strip() for item in text.split(marker, 1)[1].split(" | ") if item.strip()
        )
    )


def _verify_summary(
    summary: Mapping[str, object],
    *,
    config: EndpointContextConfig,
    sequence_sha256: str,
    assay_sha256: str,
) -> None:
    if summary.get("schema_version") != config.normalized_schema_version:
        raise ValueError("normalized summary schema version mismatch")
    if summary.get("parser_id") != config.normalized_parser_id:
        raise ValueError("normalized summary parser_id mismatch")
    if summary.get("unique_sequences") != config.expected_unique_sequences:
        raise ValueError("normalized summary unique_sequences mismatch")
    if summary.get("assay_observations") != config.expected_assay_observations:
        raise ValueError("normalized summary assay_observations mismatch")
    if summary.get("endpoint_counts") != dict(config.expected_endpoint_counts):
        raise ValueError("normalized summary endpoint_counts mismatch")
    if summary.get("sequences_sha256") != sequence_sha256:
        raise ValueError("normalized summary sequences_sha256 mismatch")
    if summary.get("assays_sha256") != assay_sha256:
        raise ValueError("normalized summary assays_sha256 mismatch")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("normalized summary must declare exactly one source artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict):
        raise ValueError("normalized summary source artifact is invalid")
    required = {
        "name": config.source_name,
        "sha256": config.source_sha256,
        "source_commit": config.source_version,
        "training_status": "approved",
    }
    if any(artifact.get(key) != value for key, value in required.items()):
        raise ValueError("normalized summary source artifact is not the approved configured source")


def _parse_sha256_manifest(path: Path, *, name: str) -> dict[str, str]:
    try:
        text = _read_regular_file(path, name=name).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{name} line {line_number} is not a SHA-256 entry")
        digest, _, filename = match.groups()
        if filename in entries:
            raise ValueError(f"{name} repeats path {filename!r}")
        entries[filename] = digest
    if not entries:
        raise ValueError(f"{name} must contain at least one entry")
    return entries


def _require_manifest_suffix(
    entries: Mapping[str, str],
    *,
    suffix: str,
    digest: str,
    manifest_name: str,
) -> None:
    matches = {
        path: value
        for path, value in entries.items()
        if path == suffix or path.endswith(f"/{suffix}")
    }
    if len(matches) != 1 or next(iter(matches.values())) != digest:
        raise ValueError(f"{manifest_name} does not uniquely attest {suffix!r}")


def _verify_code_manifest(path: Path, *, config_path: Path) -> None:
    entries = _parse_sha256_manifest(path, name="code manifest")
    repo_root = Path(__file__).resolve().parents[3]
    source_files = {
        item.relative_to(repo_root).as_posix()
        for item in (repo_root / "src" / "amp_challenge").rglob("*.py")
        if item.is_file()
    }
    manifest_sources = {
        item for item in entries if item.startswith("src/amp_challenge/") and item.endswith(".py")
    }
    if manifest_sources != source_files:
        missing = sorted(source_files - manifest_sources)
        extra = sorted(manifest_sources - source_files)
        raise ValueError(
            f"code manifest source inventory mismatch; missing={missing}, extra={extra}"
        )
    required = {
        config_path.resolve().relative_to(repo_root).as_posix(),
        "cluster/validate_endpoint_context_output.sh",
        "cluster/slurm/build_endpoint_context_v7_twins.sbatch",
        "pyproject.toml",
        "uv.lock",
    }
    missing_required = sorted(required - set(entries))
    if missing_required:
        raise ValueError(f"code manifest misses required path(s): {missing_required}")
    for relative, expected in entries.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"code manifest contains unsafe path {relative!r}")
        actual_path = repo_root / candidate
        if not actual_path.is_file() or _sha256(actual_path) != expected:
            raise ValueError(f"code manifest entry does not match repository file {relative!r}")


def _study_row(item: StudyMembership) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sequence_id": item.sequence_id,
        "provenance_id": item.provenance_id,
        "source": item.source,
        "source_version": item.source_version,
        "source_sha256": item.source_sha256,
        "source_record_id": item.source_record_id,
        "source_row_number": item.source_row_number,
        "source_record_key": item.source_record_key,
        "study_status": item.study_status,
        "study_keys": list(item.study_keys),
        "pmids": list(item.pmids),
        "ignored_pubmed_tokens": list(item.ignored_pubmed_tokens),
        "citation_reference": item.citation_reference,
        "citation_title": item.citation_title,
        "study_review_codes": list(item.study_review_codes),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_endpoint_context(
    *,
    sequences_path: str | Path,
    assays_path: str | Path,
    normalized_summary_path: str | Path,
    config_path: str | Path,
    normalized_data_manifest_path: str | Path,
    code_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> EndpointContextExecution:
    """Build deterministic sidecars and atomically publish a new output directory."""

    config = load_config(config_path)
    if _GIT_SHA1.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character Git SHA")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse endpoint-context output directory: {output}")

    sequences_file = Path(sequences_path).resolve(strict=True)
    assays_file = Path(assays_path).resolve(strict=True)
    summary_file = Path(normalized_summary_path).resolve(strict=True)
    data_manifest_file = Path(normalized_data_manifest_path).resolve(strict=True)
    code_manifest_file = Path(code_manifest_path).resolve(strict=True)
    sequence_payload = _read_regular_file(sequences_file, name="normalized sequences")
    assay_payload = _read_regular_file(assays_file, name="normalized assays")
    summary_payload = _read_regular_file(summary_file, name="normalized summary")
    sequence_sha256 = _sha256_bytes(sequence_payload)
    assay_sha256 = _sha256_bytes(assay_payload)
    summary_sha256 = _sha256_bytes(summary_payload)
    configured_hashes = {
        "normalized sequences": config.normalized_sequences_sha256,
        "normalized assays": config.normalized_assays_sha256,
        "normalized summary": config.normalized_summary_sha256,
    }
    observed_hashes = {
        "normalized sequences": sequence_sha256,
        "normalized assays": assay_sha256,
        "normalized summary": summary_sha256,
    }
    for name, expected in configured_hashes.items():
        if observed_hashes[name] != expected:
            raise ValueError(f"{name} hash differs from the frozen config")
    summary = _read_json_object_bytes(summary_payload, name="normalized summary")
    _verify_summary(
        summary,
        config=config,
        sequence_sha256=sequence_sha256,
        assay_sha256=assay_sha256,
    )
    data_manifest = _parse_sha256_manifest(data_manifest_file, name="normalized data manifest")
    _require_manifest_suffix(
        data_manifest,
        suffix="normalized/sequences.jsonl",
        digest=sequence_sha256,
        manifest_name="normalized data manifest",
    )
    _require_manifest_suffix(
        data_manifest,
        suffix="normalized/assays.jsonl",
        digest=assay_sha256,
        manifest_name="normalized data manifest",
    )
    _require_manifest_suffix(
        data_manifest,
        suffix="normalized/summary.json",
        digest=summary_sha256,
        manifest_name="normalized data manifest",
    )
    _verify_code_manifest(code_manifest_file, config_path=config.path)

    sequence_rows = _read_jsonl_bytes(sequence_payload, name="normalized sequences")
    if len(sequence_rows) != config.expected_unique_sequences:
        raise ValueError("normalized sequence row count differs from the frozen config")
    sequences: dict[str, str] = {}
    memberships: list[StudyMembership] = []
    membership_by_sequence_provenance: dict[tuple[str, str], tuple[StudyMembership, str]] = {}
    provenance_owner: dict[str, str] = {}
    for row_number, (row, _) in enumerate(sequence_rows, start=1):
        name = f"normalized sequence row {row_number}"
        _require_exact_fields(row, expected=_SEQUENCE_ROW_FIELDS, name=name)
        sequence_raw = row["sequence"]
        sequence_id_raw = row["sequence_id"]
        if not isinstance(sequence_raw, str) or not isinstance(sequence_id_raw, str):
            raise ValueError(f"{name} sequence and sequence_id must be strings")
        sequence = canonicalize_sequence(sequence_raw)
        sequence_id = sequence_id_raw
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"normalized sequence row {row_number} has a sequence_id mismatch")
        if sequence_id in sequences:
            raise ValueError(f"normalized sequence row {row_number} repeats a sequence_id")
        sequences[sequence_id] = sequence
        raw_provenance = row.get("provenance")
        if not isinstance(raw_provenance, list) or not raw_provenance:
            raise ValueError(f"normalized sequence row {row_number} lacks provenance")
        for provenance in raw_provenance:
            if not isinstance(provenance, dict):
                raise ValueError(f"normalized sequence row {row_number} has invalid provenance")
            membership = derive_study_membership(
                sequence_id=sequence_id,
                provenance=provenance,
                config=config,
            )
            extra = _extra_mapping(provenance)
            if canonicalize_sequence(extra.get("Sequence", "")) != sequence:
                raise ValueError(
                    "normalized sequence provenance Sequence does not match its entity"
                )
            previous_owner = provenance_owner.setdefault(membership.provenance_id, sequence_id)
            if previous_owner != sequence_id:
                raise ValueError("one source provenance row maps to multiple canonical sequences")
            key = (sequence_id, membership.provenance_id)
            if key in membership_by_sequence_provenance:
                raise ValueError("normalized sequence repeats an identical provenance entry")
            membership_by_sequence_provenance[key] = (
                membership,
                _provenance_semantic_sha256(provenance, config=config),
            )
            memberships.append(membership)

    assay_rows = _read_jsonl_bytes(assay_payload, name="normalized assays")
    if len(assay_rows) != config.expected_assay_observations:
        raise ValueError("normalized assay row count differs from the frozen config")
    endpoint_counts: Counter[str] = Counter()
    mapping_counts: Counter[str] = Counter()
    gram_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    blood_resolution_counts: Counter[str] = Counter()
    strain_review_counts: Counter[str] = Counter()
    condition_observations = 0
    ledger_rows: list[dict[str, object]] = []
    context_accumulator: dict[str, dict[str, object]] = {}
    observation_ids: set[str] = set()
    assay_row_hashes: set[str] = set()

    for input_line, (row, raw_line) in enumerate(assay_rows, start=1):
        endpoint, source_gram = _validate_normalized_assay_schema(row, input_line=input_line)
        sequence_id = cast(str, row["sequence_id"])
        sequence = canonicalize_sequence(cast(str, row["sequence"]))
        if sequence_id not in sequences or sequences[sequence_id] != sequence:
            raise ValueError(f"normalized assay line {input_line} has invalid sequence linkage")
        endpoint_counts[endpoint] += 1
        measurement = _measurement(row)
        exposure_raw = row["exposure_concentration"]
        exposure = None if exposure_raw is None else _measurement(exposure_raw)
        if endpoint in {"mic", "hc50"}:
            if measurement.unit != "uM" or not _positive_bounds(measurement):
                raise ValueError(
                    f"normalized assay line {input_line} has an invalid concentration endpoint"
                )
            if exposure is not None:
                raise ValueError(
                    f"normalized assay line {input_line} unexpectedly has an exposure concentration"
                )
        else:
            effect_bounds = [
                bound for bound in (measurement.lower, measurement.upper) if bound is not None
            ]
            if (
                measurement.unit != "%"
                or not effect_bounds
                or min(effect_bounds) < 0
                or max(effect_bounds) > 100
                or exposure is None
                or exposure.unit != "uM"
                or not _positive_bounds(exposure)
            ):
                raise ValueError(
                    f"normalized assay line {input_line} has an invalid percent-at-dose endpoint"
                )
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"normalized assay line {input_line} lacks provenance")
        provenance_id, _, _, _ = _provenance_identity(provenance, config=config)
        membership_match = membership_by_sequence_provenance.get((sequence_id, provenance_id))
        if membership_match is None:
            raise ValueError(
                f"normalized assay line {input_line} provenance is absent from its sequence entity"
            )
        membership, expected_provenance_sha256 = membership_match
        if _provenance_semantic_sha256(provenance, config=config) != expected_provenance_sha256:
            raise ValueError(
                f"normalized assay line {input_line} provenance differs from its sequence entity"
            )

        target_raw = row.get("strain")
        target_text = cast(str | None, target_raw)
        organism_raw = row.get("organism")
        organism_text = cast(str | None, organism_raw)
        normalized_target = None if target_text is None else _normalize_whitespace(target_text)
        normalized_organism = (
            None if organism_text is None else _normalize_whitespace(organism_text)
        )
        matched_strain_reviews = (
            []
            if endpoint != "mic" or normalized_target is None
            else [
                review
                for review in config.strain_identifier_reviews
                if review.matches(normalized_target)
            ]
        )
        strain_review_ids = sorted(review.identifier for review in matched_strain_reviews)
        strain_review_counts.update(strain_review_ids)
        if endpoint != "mic":
            strain_identifier_resolution = "not_applicable_non_mic"
        elif normalized_target is None:
            strain_identifier_resolution = "missing_source_target"
        elif strain_review_ids:
            strain_identifier_resolution = "known_conflicting_identifier"
        else:
            strain_identifier_resolution = "unreviewed_source_literal"
        target = classify_target_context(
            endpoint=endpoint,
            target_text=target_text,
            source_gram=source_gram,
            rules=config.targets,
        )
        mapping_status = str(target["mapping_status"])
        gram_resolution = str(target["gram_resolution"])
        mapping_counts[mapping_status] += 1
        gram_counts[gram_resolution] += 1
        if target["canonical_target"] is not None:
            target_counts[str(target["canonical_target"])] += 1
        resolved_blood_organism, blood_organism_resolution = _resolve_blood_organism(
            endpoint=endpoint,
            normalized_target=normalized_target,
            source_organism=organism_text,
            aliases=config.blood_organism_aliases,
        )
        blood_resolution_counts[blood_organism_resolution] += 1

        conditions = _source_conditions(row.get("assay"))
        condition_observations += int(bool(conditions))
        mic16_label: int | None = None
        mic16_reason: str | None = None
        if endpoint == "mic":
            mic16_label, mic16_reason = derive_activity_label(
                measurement, threshold_um=config.activity_threshold_um
            )

        exclusion_codes: list[str] = []
        if mapping_status != "mapped_single_supported_species" and endpoint == "mic":
            exclusion_codes.append(mapping_status)
        if gram_resolution in {"conflict", "missing_source_gram"}:
            exclusion_codes.append(f"gram_{gram_resolution}")
        eligible_tasks: list[str] = []
        if (
            endpoint == "mic"
            and mapping_status == "mapped_single_supported_species"
            and gram_resolution == "concordant"
        ):
            eligible_tasks.append("bacterial_mic_interval")
            if mic16_label is not None:
                eligible_tasks.extend(("bacterial_mic16", f"gram_{source_gram}_mic16"))
        elif endpoint in {"hc50", "hemolysis_percent"}:
            blood_task = _blood_endpoint_task(endpoint, resolved_blood_organism)
            if blood_task is not None:
                eligible_tasks.append(blood_task)
            else:
                exclusion_codes.append("unresolved_blood_species")
        task_counts.update(eligible_tasks)
        exclusion_counts.update(exclusion_codes)

        context_id = _stable_digest(
            "amp-challenge:endpoint-context:v1",
            endpoint,
            normalized_target,
            normalized_organism,
        )
        assay_context_id = _stable_digest(
            "amp-challenge:assay-context:v1",
            sequence_id,
            endpoint,
            context_id,
            conditions,
            _exposure_identity(exposure),
        )
        assay_row_sha256 = _sha256_bytes(raw_line)
        observation_id = _sha256_bytes(b"amp-challenge:normalized-assay:v1\0" + raw_line)
        if observation_id in observation_ids or assay_row_sha256 in assay_row_hashes:
            raise ValueError("normalized assays contain an exact duplicate row")
        observation_ids.add(observation_id)
        assay_row_hashes.add(assay_row_sha256)
        ledger = {
            "schema_version": 1,
            "observation_id": observation_id,
            "assay_row_sha256": assay_row_sha256,
            "input_line": input_line,
            "sequence_id": sequence_id,
            "endpoint": endpoint,
            "source_field": "Target_Organism" if endpoint == "mic" else "Hemolytic_activity",
            "provenance_id": provenance_id,
            "source_record_key": membership.source_record_key,
            "source_record_id": membership.source_record_id,
            "source_row_number": membership.source_row_number,
            "context_id": context_id,
            "assay_context_id": assay_context_id,
            "source_target": target_text,
            "normalized_target": normalized_target,
            "source_organism": organism_text,
            "normalized_organism": normalized_organism,
            "resolved_blood_organism": resolved_blood_organism,
            "blood_organism_resolution": blood_organism_resolution,
            "strain_identifier_resolution": strain_identifier_resolution,
            "strain_identifier_review_ids": strain_review_ids,
            "strain_level_eligible": False,
            "source_gram": source_gram,
            "source_assay": None if row.get("assay") is None else str(row["assay"]),
            "source_conditions": list(conditions),
            "measurement_status": "accepted_parser_v7",
            "measurement": _measurement_row(measurement),
            "exposure_concentration": (None if exposure is None else _measurement_row(exposure)),
            "source_text": None if row.get("source_text") is None else str(row["source_text"]),
            "mapping_version": config.mapping_version,
            "blood_organism_mapping_version": config.blood_organism_mapping_version,
            **target,
            "resistance_status": "not_inferred",
            "study_status": membership.study_status,
            "study_keys": list(membership.study_keys),
            "study_review_codes": list(membership.study_review_codes),
            "mic16_label": mic16_label,
            "mic16_reason": mic16_reason,
            "eligible_tasks": sorted(eligible_tasks),
            "exclusion_codes": sorted(exclusion_codes),
        }
        ledger_rows.append(ledger)

        context = context_accumulator.setdefault(
            context_id,
            {
                "schema_version": 1,
                "context_id": context_id,
                "endpoint": endpoint,
                "normalized_target": normalized_target,
                "normalized_organism": normalized_organism,
                "resolved_blood_organism": resolved_blood_organism,
                "blood_organism_resolution": blood_organism_resolution,
                "strain_identifier_resolution": strain_identifier_resolution,
                "strain_identifier_review_ids": strain_review_ids,
                "strain_level_eligible": False,
                "mapping_version": config.mapping_version,
                "blood_organism_mapping_version": config.blood_organism_mapping_version,
                "mapping_status": target["mapping_status"],
                "canonical_target": target["canonical_target"],
                "expected_gram": target["expected_gram"],
                "target_domain": target["target_domain"],
                "apparent_taxon_mentions": target["apparent_taxon_mentions"],
                "configured_target_occurrences": target["configured_target_occurrences"],
                "composite_language_marker": target["composite_language_marker"],
                "composite_reason_codes": target["composite_reason_codes"],
                "observation_ids": set(),
                "sequence_ids": set(),
                "provenance_ids": set(),
                "source_grams": set(),
                "gram_resolutions": set(),
                "source_target_variants": set(),
                "source_organism_variants": set(),
            },
        )
        for field, expected in (
            ("endpoint", endpoint),
            ("normalized_target", normalized_target),
            ("normalized_organism", normalized_organism),
            ("resolved_blood_organism", resolved_blood_organism),
            ("blood_organism_resolution", blood_organism_resolution),
            ("strain_identifier_resolution", strain_identifier_resolution),
            ("strain_identifier_review_ids", strain_review_ids),
            ("mapping_status", target["mapping_status"]),
            ("canonical_target", target["canonical_target"]),
        ):
            if context[field] != expected:
                raise AssertionError(f"context hash collision changes {field}")
        cast(set[str], context["observation_ids"]).add(observation_id)
        cast(set[str], context["sequence_ids"]).add(sequence_id)
        cast(set[str], context["provenance_ids"]).add(provenance_id)
        cast(set[str], context["source_grams"]).add(source_gram)
        cast(set[str], context["gram_resolutions"]).add(gram_resolution)
        if target_text is not None:
            cast(set[str], context["source_target_variants"]).add(target_text)
        if organism_text is not None:
            cast(set[str], context["source_organism_variants"]).add(organism_text)

    observed_endpoint_counts = {
        endpoint: endpoint_counts[endpoint] for endpoint in sorted(_ENDPOINTS)
    }
    if observed_endpoint_counts != dict(config.expected_endpoint_counts):
        raise ValueError(
            "observed endpoint histogram differs from the frozen config: "
            f"observed={observed_endpoint_counts}, "
            f"expected={dict(config.expected_endpoint_counts)}"
        )

    contexts: list[dict[str, object]] = []
    for context_id in sorted(context_accumulator):
        raw = context_accumulator[context_id]
        observation_set = cast(set[str], raw.pop("observation_ids"))
        sequence_set = cast(set[str], raw.pop("sequence_ids"))
        provenance_set = cast(set[str], raw.pop("provenance_ids"))
        source_grams = cast(set[str], raw.pop("source_grams"))
        gram_resolutions = cast(set[str], raw.pop("gram_resolutions"))
        source_target_variants = cast(set[str], raw.pop("source_target_variants"))
        source_organism_variants = cast(set[str], raw.pop("source_organism_variants"))
        contexts.append(
            {
                **raw,
                "observations": len(observation_set),
                "sequences": len(sequence_set),
                "source_records": len(provenance_set),
                "source_grams": sorted(source_grams),
                "gram_resolutions": sorted(gram_resolutions),
                "source_target_variants": sorted(source_target_variants),
                "source_organism_variants": sorted(source_organism_variants),
            }
        )

    membership_rows = [
        _study_row(item)
        for item in sorted(memberships, key=lambda item: (item.sequence_id, item.provenance_id))
    ]
    membership_status_counts = Counter(item.study_status for item in memberships)
    sequences_by_status: dict[str, set[str]] = defaultdict(set)
    key_to_sequences: dict[str, set[str]] = defaultdict(set)
    for item in memberships:
        sequences_by_status[item.study_status].add(item.sequence_id)
        for key in item.study_keys:
            key_to_sequences[key].add(item.sequence_id)
    ignored_token_counts = Counter(
        token for item in memberships for token in item.ignored_pubmed_tokens
    )
    review_code_counts = Counter(code for item in memberships for code in item.study_review_codes)
    statuses_by_sequence: dict[str, set[str]] = defaultdict(set)
    for item in memberships:
        statuses_by_sequence[item.sequence_id].add(item.study_status)
    observations_by_record = Counter(cast(str, row["source_record_id"]) for row in ledger_rows)
    reviewed_study_anomalies: list[dict[str, object]] = []
    for anomaly in config.study_anomalies:
        if anomaly.scope == "key":
            matched = [item for item in memberships if anomaly.selector in item.study_keys]
        else:
            matched = [item for item in memberships if item.source_record_id == anomaly.selector]
        observed_records = tuple(sorted({item.source_record_id for item in matched}))
        observed_keys = tuple(sorted({key for item in matched for key in item.study_keys}))
        observed_observations = sum(
            observations_by_record[record_id] for record_id in observed_records
        )
        if observed_records != anomaly.expected_source_record_ids:
            raise ValueError(
                f"reviewed study anomaly {anomaly.code!r} source-record members changed"
            )
        if observed_keys != anomaly.expected_study_keys:
            raise ValueError(f"reviewed study anomaly {anomaly.code!r} study keys changed")
        if observed_observations != anomaly.expected_observations:
            raise ValueError(f"reviewed study anomaly {anomaly.code!r} observation count changed")
        if any(anomaly.code not in item.study_review_codes for item in matched):
            raise AssertionError("reviewed study anomaly code was not propagated")
        reviewed_study_anomalies.append(
            {
                "scope": anomaly.scope,
                "selector": anomaly.selector,
                "code": anomaly.code,
                "policy": anomaly.policy,
                "source_record_ids": list(observed_records),
                "study_keys": list(observed_keys),
                "discordant_source_record_ids": list(anomaly.discordant_source_record_ids),
                "memberships": len(matched),
                "sequences": len({item.sequence_id for item in matched}),
                "observations": observed_observations,
            }
        )
    singleton_keys = {key for key in key_to_sequences if key.startswith("source-record:")}
    composite_contexts = [
        row for row in contexts if row["mapping_status"] == "ambiguous_composite_target"
    ]
    audit: dict[str, object] = {
        "schema_version": 1,
        "artifact": "dramp_endpoint_context_audit",
        "status": "development_sidecar_not_a_new_parser_or_untouched_evaluation_panel",
        "input": {
            "normalized_parser_id": config.normalized_parser_id,
            "normalized_schema_version": config.normalized_schema_version,
            "unique_sequences": len(sequences),
            "assay_observations": len(ledger_rows),
            "endpoint_counts": observed_endpoint_counts,
        },
        "contexts": {
            "unique_contexts": len(contexts),
            "mapping_counts": dict(sorted(mapping_counts.items())),
            "gram_resolution_counts": dict(sorted(gram_counts.items())),
            "configured_target_observations": dict(sorted(target_counts.items())),
            "composite_contexts": len(composite_contexts),
            "composite_observations": sum(int(row["observations"]) for row in composite_contexts),
            "condition_bearing_observations": condition_observations,
            "eligible_task_counts": dict(sorted(task_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "blood_organism_resolution_counts": dict(sorted(blood_resolution_counts.items())),
            "strain_identifier_review_observation_counts": dict(
                sorted(strain_review_counts.items())
            ),
        },
        "studies": {
            "provenance_memberships": len(memberships),
            "membership_status_counts": dict(sorted(membership_status_counts.items())),
            "sequence_status_counts": {
                key: len(value) for key, value in sorted(sequences_by_status.items())
            },
            "multi_status_sequences": sum(
                len(statuses) > 1 for statuses in statuses_by_sequence.values()
            ),
            "unique_study_keys": len(key_to_sequences),
            "shared_study_keys": sum(len(value) > 1 for value in key_to_sequences.values()),
            "source_record_singleton_keys": len(singleton_keys),
            "maximum_sequences_per_study_key": max(map(len, key_to_sequences.values())),
            "ignored_pubmed_token_counts": dict(sorted(ignored_token_counts.items())),
            "review_code_membership_counts": dict(sorted(review_code_counts.items())),
            "reviewed_anomalies": reviewed_study_anomalies,
        },
        "invariants": {
            "accepted_assays_covered_once": len(observation_ids) == len(ledger_rows),
            "all_assay_provenance_links_resolved": True,
            "all_sequences_have_study_membership": set(sequences)
            == {item.sequence_id for item in memberships},
            "missing_study_keys_are_record_singletons": all(
                len(key_to_sequences[key]) == 1 for key in singleton_keys
            ),
            "reviewed_study_anomalies_match_config": True,
            "normalized_inputs_unchanged_during_read": (
                _sha256(sequences_file) == sequence_sha256
                and _sha256(assays_file) == assay_sha256
                and _sha256(summary_file) == summary_sha256
            ),
        },
        "limitations": [
            "configured mappings cover only frozen supported species and are not a complete taxonomy",
            "strain identifiers are unresolved source literals; reviewed contradictions are annotations only",
            "publication keys are proxies for experimental batches",
            "reference/title fallback may over-merge and missing metadata may under-merge studies",
            "parser-v7 MIC marker accounting and reversed-range handling require a future parser version",
            "DRAMP outcomes have been inspected repeatedly and are not an untouched test panel",
            "study keys are grouping-only metadata and must never be model features",
        ],
    }
    if not all(
        cast(bool, value) for value in cast(dict[str, object], audit["invariants"]).values()
    ):
        raise AssertionError("endpoint-context conservation invariant failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        assay_ledger = staging / "endpoint_context_ledger.jsonl"
        contexts_path = staging / "contexts.jsonl"
        membership_path = staging / "study_membership.jsonl"
        audit_path = staging / "audit.json"
        manifest_path = staging / "manifest.json"
        _write_jsonl(assay_ledger, sorted(ledger_rows, key=lambda row: str(row["observation_id"])))
        _write_jsonl(contexts_path, contexts)
        _write_jsonl(membership_path, membership_rows)
        _write_json(audit_path, audit)
        artifact_paths = {
            "endpoint_context_ledger": assay_ledger,
            "contexts": contexts_path,
            "study_membership": membership_path,
            "audit": audit_path,
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact": "dramp_endpoint_context_sidecar",
            "status": "development_only_pending_reviewed_taxonomy_and_study_component_split",
            "config_sha256": _sha256(config.path),
            "input": {
                "sequences": {"filename": sequences_file.name, "sha256": sequence_sha256},
                "assays": {"filename": assays_file.name, "sha256": assay_sha256},
                "normalized_summary": {
                    "filename": summary_file.name,
                    "sha256": summary_sha256,
                },
                "normalized_data_manifest": {
                    "filename": data_manifest_file.name,
                    "sha256": _sha256(data_manifest_file),
                },
            },
            "policies": {
                "mapping_version": config.mapping_version,
                "blood_organism_mapping_version": config.blood_organism_mapping_version,
                "target_mapping": (
                    "configured regex support only; multiple apparent or configured taxa are "
                    "ambiguous; no fuzzy taxonomy or MDR inference"
                ),
                "blood_organism_mapping": (
                    "parser-explicit human/nonhuman values plus only frozen exact source-target "
                    "aliases; unresolved blood species are task-ineligible"
                ),
                "study_keys": (
                    "all strict decimal PubMed tokens split on the literal configured delimiter; "
                    "otherwise exact whitespace-normalized reference/title hash; otherwise a "
                    "source-record singleton; grouping-only, never a model feature"
                ),
                "study_anomalies": reviewed_study_anomalies,
                "strain_identifier_reviews": [
                    {
                        "identifier": item.identifier,
                        "target_regex": item.target_regex,
                        "code": item.code,
                        "policy": item.policy,
                    }
                    for item in config.strain_identifier_reviews
                ],
                "mic16": (
                    "MIC <= threshold only when the complete normalized interval proves the class"
                ),
            },
            "counts": {
                "unique_sequences": len(sequences),
                "assay_observations": len(ledger_rows),
                "contexts": len(contexts),
                "study_memberships": len(memberships),
            },
            "artifacts": {
                key: {"filename": path.name, "sha256": _sha256(path)}
                for key, path in artifact_paths.items()
            },
            "provenance": {
                "git_commit": git_commit,
                "code_manifest": {
                    "filename": code_manifest_file.name,
                    "sha256": _sha256(code_manifest_file),
                },
            },
        }
        _write_json(manifest_path, manifest)
        if output.exists():
            raise FileExistsError(f"endpoint-context output appeared during build: {output}")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return EndpointContextExecution(
        output_dir=output,
        assay_ledger_path=output / "endpoint_context_ledger.jsonl",
        contexts_path=output / "contexts.jsonl",
        study_membership_path=output / "study_membership.jsonl",
        audit_path=output / "audit.json",
        manifest_path=output / "manifest.json",
        assay_observations=len(ledger_rows),
        unique_sequences=len(sequences),
        summary=audit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable DRAMP endpoint-context and study-membership sidecar"
    )
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--assays", type=Path, required=True)
    parser.add_argument("--normalized-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--normalized-data-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = build_endpoint_context(
        sequences_path=args.sequences,
        assays_path=args.assays,
        normalized_summary_path=args.normalized_summary,
        config_path=args.config,
        normalized_data_manifest_path=args.normalized_data_manifest,
        code_manifest_path=args.code_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(execution.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

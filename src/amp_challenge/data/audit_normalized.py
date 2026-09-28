"""Audit a normalized AMP assay snapshot against the competition reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from amp_challenge.data.prepare import ENDPOINT_REJECTION_REASON_CODES, endpoint_rejection_id
from amp_challenge.data.records import CensoredValue, Provenance
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.workflows.audit import read_organizer_fasta
from amp_challenge.workflows.generate import ReferenceIndex

_DRAMP_WORKBOOK_COLUMNS = frozenset(
    {
        "DRAMP_ID",
        "Sequence",
        "Sequence_Length",
        "Name",
        "Swiss_Prot_Entry",
        "Family",
        "Gene",
        "Source",
        "Activity",
        "Protein_existence",
        "Structure",
        "Structure_Description",
        "PDB_ID",
        "Comments",
        "Target_Organism",
        "Hemolytic_activity",
        "Binding_Target",
        "Pubmed_ID",
        "Reference",
        "Author",
        "Title",
    }
)
_ENDPOINT_PROVENANCE_METADATA = frozenset(
    {
        "source_repository",
        "source_version",
        "source_sha256",
        "source_license",
        "training_status",
    }
)
_SCHEMA_V2_ARTIFACT_FIELDS = frozenset(
    {
        "name",
        "sha256",
        "license",
        "training_status",
        "bytes",
        "relative_path",
        "source_repository",
        "source_commit",
    }
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield line_number, value


def _read_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for line_number, row in _read_jsonl(path):
        sequence = canonicalize_sequence(str(row.get("sequence", "")))
        sequence_id = str(row.get("sequence_id", ""))
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"{path}:{line_number}: sequence_id does not match sequence")
        if sequence_id in sequences:
            raise ValueError(f"{path}:{line_number}: duplicate sequence_id")
        sequences[sequence_id] = sequence
    if not sequences:
        raise ValueError(f"normalized sequence table is empty: {path}")
    return sequences


def _verify_summary(root: Path) -> tuple[dict[str, object], dict[str, Path]]:
    paths = {
        "sequences": root / "sequences.jsonl",
        "assays": root / "assays.jsonl",
        "rejects": root / "rejects.jsonl",
        "summary": root / "summary.json",
    }
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    schema_version = int(summary.get("schema_version", 0))
    if schema_version not in {1, 2}:
        raise ValueError("normalized summary schema_version must be 1 or 2")
    checksum_names = ["sequences", "assays", "rejects"]
    if schema_version == 2:
        paths["endpoint_rejects"] = root / "endpoint_rejects.jsonl"
        checksum_names.append("endpoint_rejects")
    for name in checksum_names:
        expected = str(summary.get(f"{name}_sha256", ""))
        actual = _sha256(paths[name])
        if expected != actual:
            raise ValueError(
                f"normalized {name} checksum mismatch: expected {expected!r}, got {actual!r}"
            )
    return summary, paths


def _activity_label(row: Mapping[str, object], threshold: float) -> str | None:
    if str(row.get("endpoint", "")).strip().lower() != "mic":
        return None
    relation = str(row.get("relation", ""))
    if relation == "approx" or row.get("unit") != "uM":
        return "ambiguous"
    lower = None if row.get("lower") is None else float(row["lower"])
    upper = None if row.get("upper") is None else float(row["upper"])
    if upper is not None and upper <= threshold:
        return "active"
    if lower is not None and (
        lower > threshold or (lower == threshold and not bool(row.get("lower_inclusive", False)))
    ):
        return "inactive"
    return "ambiguous"


def _contract_value(
    row: Mapping[str, object],
    *,
    prefix: str,
) -> tuple[CensoredValue | None, tuple[str, ...]]:
    issues: list[str] = []
    for name in ("lower_inclusive", "upper_inclusive"):
        if not isinstance(row.get(name), bool):
            issues.append(f"{prefix}_{name}_not_boolean")
    if issues:
        return None, tuple(issues)

    def optional_float(name: str) -> float | None:
        raw = row.get(name)
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise ValueError(f"{name} is boolean")
        return float(raw)

    try:
        value = CensoredValue(
            relation=str(row.get("relation", "")),  # type: ignore[arg-type]
            lower=optional_float("lower"),
            upper=optional_float("upper"),
            lower_inclusive=row["lower_inclusive"],  # type: ignore[arg-type]
            upper_inclusive=row["upper_inclusive"],  # type: ignore[arg-type]
            unit=None if row.get("unit") is None else str(row["unit"]),
            raw=None if row.get("raw_value") is None else str(row["raw_value"]),
            source_unit=(None if row.get("source_unit") is None else str(row["source_unit"])),
        )
    except (KeyError, TypeError, ValueError):
        return None, (f"{prefix}_invalid_censoring",)
    return value, ()


_BLOOD_CONTEXT = re.compile(
    r"\b(?:erythrocytes?|red\s+blood\s+cells?|RBCs?|blood)\b",
    re.IGNORECASE,
)


def _endpoint_contract_issues(row: Mapping[str, object]) -> tuple[str, ...]:
    endpoint = str(row.get("endpoint", "")).strip().lower()
    if endpoint not in {"hc50", "hemolysis_percent"}:
        return ()

    issues: list[str] = []
    value, value_issues = _contract_value(row, prefix="value")
    issues.extend(value_issues)
    source_text = str(row.get("source_text") or "").strip()
    if not source_text:
        issues.append("missing_source_text")
    target = str(row.get("strain") or "").strip()
    if not target:
        issues.append("missing_blood_target")
    elif _BLOOD_CONTEXT.search(target) is None:
        issues.append("non_blood_target")
    elif re.search(r"\bblood\s+agar\b", target, re.IGNORECASE):
        issues.append("blood_agar_target")
    if target.count("(") != target.count(")"):
        issues.append("unbalanced_blood_target_parentheses")
    if source_text and target and target not in " ".join(source_text.split()):
        issues.append("blood_target_not_in_source_text")

    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        extra = provenance.get("extra")
        if isinstance(extra, list):
            extra_mapping = {
                str(pair[0]): str(pair[1])
                for pair in extra
                if isinstance(pair, list) and len(pair) == 2
            }
            source_cell = extra_mapping.get("Hemolytic_activity")
            if source_cell is not None and source_text != source_cell:
                issues.append("source_text_provenance_mismatch")

    exposure_raw = row.get("exposure_concentration")
    if endpoint == "hc50":
        if exposure_raw is not None:
            issues.append("hc50_has_exposure_concentration")
        if value is not None:
            if value.unit != "uM":
                issues.append("hc50_unit_not_um")
            if value.source_unit is None:
                issues.append("hc50_missing_source_unit")
            if value.raw is None:
                issues.append("hc50_missing_raw_value")
            bounds = tuple(bound for bound in (value.lower, value.upper) if bound is not None)
            if not bounds or min(bounds) <= 0:
                issues.append("hc50_not_positive")
    else:
        if value is not None:
            if value.unit != "%":
                issues.append("hemolysis_effect_unit_not_percent")
            if value.source_unit != "%":
                issues.append("hemolysis_effect_source_unit_not_percent")
            if value.raw is None:
                issues.append("hemolysis_effect_missing_raw_value")
            bounds = tuple(bound for bound in (value.lower, value.upper) if bound is not None)
            if not bounds or min(bounds) < 0 or max(bounds) > 100:
                issues.append("hemolysis_effect_outside_percent_range")
        if not isinstance(exposure_raw, Mapping):
            issues.append("hemolysis_missing_exposure_concentration")
        else:
            exposure, exposure_issues = _contract_value(
                exposure_raw,
                prefix="exposure",
            )
            issues.extend(exposure_issues)
            if exposure is not None:
                if exposure.unit != "uM":
                    issues.append("hemolysis_exposure_unit_not_um")
                if exposure.source_unit is None:
                    issues.append("hemolysis_exposure_missing_source_unit")
                if exposure.raw is None:
                    issues.append("hemolysis_exposure_missing_raw_value")
    return tuple(dict.fromkeys(issues))


def _provenance_row_key(provenance: Mapping[str, object]) -> tuple[str, str, int]:
    source = str(provenance.get("source") or "").strip()
    path = str(provenance.get("path") or "")
    row_number = provenance.get("row_number")
    if (
        not source
        or isinstance(row_number, bool)
        or not isinstance(row_number, int)
        or row_number < 1
    ):
        raise ValueError(
            "endpoint rejection provenance requires source and positive integer row_number"
        )
    return source, path, row_number


def _provenance_extra(provenance: Mapping[str, object]) -> dict[str, str]:
    raw = provenance.get("extra")
    if not isinstance(raw, list):
        raise ValueError("endpoint rejection provenance.extra must be a list")
    pairs: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("endpoint rejection provenance.extra entries must be pairs")
        key, value = str(item[0]), str(item[1])
        if key in pairs:
            raise ValueError(f"endpoint rejection provenance.extra duplicates {key!r}")
        pairs[key] = value
    return pairs


def _schema_v2_artifacts(raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("normalized schema v2 requires a non-empty artifacts list")
    artifacts: dict[str, dict[str, object]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"normalized artifact {index} must be an object")
        if set(item) != _SCHEMA_V2_ARTIFACT_FIELDS:
            raise ValueError(
                f"normalized artifact {index} keys differ from schema: "
                f"{sorted(set(item) ^ _SCHEMA_V2_ARTIFACT_FIELDS)}"
            )
        artifact = dict(item)
        name = str(artifact["name"] or "").strip()
        if not name or name in artifacts:
            raise ValueError("normalized schema v2 artifact names must be unique and non-empty")
        sha256 = str(artifact["sha256"] or "")
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"normalized artifact {name!r} has invalid SHA-256")
        byte_count = artifact["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError(f"normalized artifact {name!r} has invalid byte count")
        for field in (
            "relative_path",
            "source_repository",
            "source_commit",
            "license",
            "training_status",
        ):
            if not str(artifact[field] or "").strip():
                raise ValueError(f"normalized artifact {name!r} has blank {field}")
        artifacts[name] = artifact
    return artifacts


def _endpoint_reject_summary(
    path: Path,
    *,
    assays_path: Path,
    rejects_path: Path,
    sequences: Mapping[str, str],
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Validate schema-v2 endpoint-field rejections and their assay atomicity."""

    required = {
        "rejection_id",
        "scope",
        "endpoint_family",
        "source_field",
        "sequence_id",
        "sequence",
        "raw_sequence",
        "source_text",
        "reason_code",
        "reason_detail",
        "discarded_candidate_observations",
        "provenance",
    }
    accepted_by_row: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for _, assay in _read_jsonl(assays_path):
        provenance = assay.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("accepted assay provenance must be an object")
        accepted_by_row[_provenance_row_key(provenance)].add(
            str(assay.get("endpoint") or "").strip().lower()
        )

    row_rejects: dict[tuple[str, str, int], str] = {}
    for line_number, rejected in _read_jsonl(rejects_path):
        row_number = rejected.get("row_number")
        if isinstance(row_number, bool) or not isinstance(row_number, int):
            raise ValueError("row rejection row_number must be an integer")
        key = (
            str(rejected.get("source") or "").strip(),
            str(rejected.get("path") or ""),
            row_number,
        )
        if key in row_rejects:
            raise ValueError(f"{rejects_path}:{line_number}: duplicate row-rejection key")
        row_rejects[key] = str(rejected.get("reason") or "")

    ids: set[str] = set()
    fields: set[tuple[str, str, int, str, str]] = set()
    reasons: Counter[str] = Counter()
    rows: set[tuple[str, str, int]] = set()
    accepted_mic_rows: set[tuple[str, str, int]] = set()
    discarded_candidates = 0
    count = 0
    for line_number, rejected in _read_jsonl(path):
        count += 1
        if set(rejected) != required:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection keys differ from schema: "
                f"{sorted(set(rejected) ^ required)}"
            )
        if rejected["scope"] != "endpoint_field":
            raise ValueError(f"{path}:{line_number}: endpoint rejection has invalid scope")
        if rejected["endpoint_family"] != "hemolysis":
            raise ValueError(f"{path}:{line_number}: unsupported endpoint rejection family")
        if rejected["source_field"] != "Hemolytic_activity":
            raise ValueError(f"{path}:{line_number}: unsupported endpoint rejection source field")

        sequence = canonicalize_sequence(str(rejected["sequence"]))
        sequence_id = str(rejected["sequence_id"])
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"{path}:{line_number}: endpoint rejection sequence_id mismatch")
        retained_sequence = sequences.get(sequence_id)
        if retained_sequence is not None and retained_sequence != sequence:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection sequence differs from table"
            )

        source_text = str(rejected["source_text"] or "").strip()
        if not source_text:
            raise ValueError(f"{path}:{line_number}: endpoint rejection source_text is blank")
        reason_code = str(rejected["reason_code"] or "")
        if reason_code not in ENDPOINT_REJECTION_REASON_CODES["hemolysis"]:
            raise ValueError(f"{path}:{line_number}: endpoint rejection reason_code is not allowed")
        if not str(rejected["reason_detail"] or "").strip():
            raise ValueError(f"{path}:{line_number}: endpoint rejection reason_detail is blank")
        discarded = rejected["discarded_candidate_observations"]
        if isinstance(discarded, bool) or not isinstance(discarded, int) or discarded < 0:
            raise ValueError(
                f"{path}:{line_number}: discarded_candidate_observations must be non-negative"
            )

        provenance_raw = rejected["provenance"]
        if not isinstance(provenance_raw, Mapping):
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection provenance is not an object"
            )
        provenance_fields = {"source", "record_id", "path", "row_number", "extra"}
        if set(provenance_raw) != provenance_fields:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection provenance keys differ from schema"
            )
        row_key = _provenance_row_key(provenance_raw)
        extra = _provenance_extra(provenance_raw)
        missing_columns = _DRAMP_WORKBOOK_COLUMNS - set(extra)
        if missing_columns:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection provenance omits workbook "
                f"columns: {sorted(missing_columns)}"
            )
        missing_metadata = _ENDPOINT_PROVENANCE_METADATA - set(extra)
        if missing_metadata:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection provenance omits frozen-source "
                f"metadata: {sorted(missing_metadata)}"
            )
        source_name = str(provenance_raw["source"] or "").strip()
        artifact = artifacts.get(source_name)
        if artifact is None:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection source has no normalized artifact"
            )
        record_id = provenance_raw["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection provenance record_id is blank"
            )
        if record_id != extra["DRAMP_ID"]:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection record_id/DRAMP_ID mismatch"
            )
        if not isinstance(provenance_raw["path"], str) or not provenance_raw["path"].strip():
            raise ValueError(f"{path}:{line_number}: endpoint rejection provenance path is blank")
        frozen_matches = {
            "source_sha256": "sha256",
            "source_version": "source_commit",
            "source_license": "license",
            "training_status": "training_status",
            "source_repository": "source_repository",
        }
        for provenance_field, artifact_field in frozen_matches.items():
            value = extra[provenance_field]
            if not value.strip():
                raise ValueError(
                    f"{path}:{line_number}: endpoint rejection provenance "
                    f"{provenance_field} is blank"
                )
            if value != str(artifact[artifact_field]):
                raise ValueError(
                    f"{path}:{line_number}: endpoint rejection provenance "
                    f"{provenance_field} does not match normalized artifact"
                )
        if extra["training_status"] != "approved":
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection source is not approved for training"
            )
        if extra.get("Hemolytic_activity") != source_text:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection source_text/provenance mismatch"
            )
        raw_sequence = rejected["raw_sequence"]
        if not isinstance(raw_sequence, str) or extra.get("Sequence") != raw_sequence:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection raw_sequence/provenance mismatch"
            )
        if canonicalize_sequence(raw_sequence) != sequence:
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection raw_sequence is not the sequence"
            )
        provenance = Provenance.from_mapping(
            source=source_name,
            record_id=record_id,
            path=str(provenance_raw["path"]),
            row_number=int(provenance_raw["row_number"]),
            extra=extra,
            include_empty=True,
        )
        expected_id = endpoint_rejection_id(
            provenance=provenance,
            endpoint_family="hemolysis",
            source_field="Hemolytic_activity",
            sequence_id=sequence_id,
        )
        rejection_id = str(rejected["rejection_id"])
        if rejection_id != expected_id:
            raise ValueError(f"{path}:{line_number}: endpoint rejection_id mismatch")
        field_key = (*row_key, "hemolysis", "Hemolytic_activity")
        if rejection_id in ids or field_key in fields:
            raise ValueError(f"{path}:{line_number}: duplicate endpoint-field rejection")
        ids.add(rejection_id)
        fields.add(field_key)

        accepted = accepted_by_row.get(row_key, set())
        if accepted & {"hc50", "hemolysis_percent"}:
            raise ValueError(
                f"{path}:{line_number}: rejected hemolysis field also has an accepted "
                "hemolysis observation"
            )
        if "mic" in accepted:
            accepted_mic_rows.add(row_key)
        row_reject_reason = row_rejects.get(row_key)
        if row_reject_reason is not None and not row_reject_reason.startswith(
            ("no explicit MIC, HC50", "no retained supported endpoint")
        ):
            raise ValueError(
                f"{path}:{line_number}: endpoint rejection coexists with a global row rejection"
            )

        reasons[reason_code] += 1
        rows.add(row_key)
        discarded_candidates += discarded

    return {
        "events": count,
        "rows": len(rows),
        "reason_counts": dict(sorted(reasons.items())),
        "accepted_mic_rows": len(accepted_mic_rows),
        "discarded_candidate_observations": discarded_candidates,
        "contract_issues": {},
    }


def _source_row_dispositions(
    *,
    assays_path: Path,
    rejects_path: Path,
    endpoint_rejects_path: Path,
) -> dict[str, object]:
    accepted: set[tuple[str, str, int]] = set()
    for _, assay in _read_jsonl(assays_path):
        provenance = assay.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("accepted assay provenance must be an object")
        accepted.add(_provenance_row_key(provenance))

    rejected: set[tuple[str, str, int]] = set()
    for line_number, row in _read_jsonl(rejects_path):
        row_number = row.get("row_number")
        if isinstance(row_number, bool) or not isinstance(row_number, int):
            raise ValueError("row rejection row_number must be an integer")
        key = (
            str(row.get("source") or "").strip(),
            str(row.get("path") or ""),
            row_number,
        )
        if key in rejected:
            raise ValueError(f"{rejects_path}:{line_number}: duplicate row-rejection key")
        rejected.add(key)
    overlap = accepted & rejected
    if overlap:
        raise ValueError("source rows cannot be both accepted and row-rejected")

    endpoint_rejected: set[tuple[str, str, int]] = set()
    for _, event in _read_jsonl(endpoint_rejects_path):
        provenance = event.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("endpoint rejection provenance must be an object")
        endpoint_rejected.add(_provenance_row_key(provenance))

    return {
        "source_rows": len(accepted | rejected | endpoint_rejected),
        "accepted_source_rows": len(accepted),
        "row_disposition_counts": {
            "accepted": len(accepted),
            "accepted_with_endpoint_reject": len(accepted & endpoint_rejected),
            "rejected": len(rejected),
            "rejected_with_endpoint_reject": len(rejected & endpoint_rejected),
        },
    }


def _target_issue(strain: str) -> str | None:
    lower = strain.lower()
    if not strain or strain == "target context unavailable":
        return "missing_target_context"
    if re.search(r"\bmic\s*[=<>≤≥]", lower):
        return "target_contains_prior_mic"
    if "ref." in lower:
        return "target_contains_reference_prefix"
    if re.match(r"^(?:in|at|under|low|high|salt|medium|ph)\b", lower):
        return "condition_mistaken_for_target"
    if strain.count("(") != strain.count(")"):
        return "unbalanced_target_parentheses"
    return None


def _assay_summary(
    path: Path,
    *,
    sequences: Mapping[str, str],
    threshold: float,
) -> dict[str, object]:
    endpoints: Counter[str] = Counter()
    grams: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    units: Counter[str] = Counter()
    source_units: Counter[str] = Counter()
    exposure_units: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    strains: Counter[str] = Counter()
    target_issues: Counter[str] = Counter()
    endpoint_contract_issues: Counter[str] = Counter()
    observations_by_sequence: Counter[str] = Counter()
    replicate_labels: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    exact_rows: Counter[str] = Counter()
    observations = 0
    for line_number, row in _read_jsonl(path):
        observations += 1
        sequence_id = str(row.get("sequence_id", ""))
        sequence = canonicalize_sequence(str(row.get("sequence", "")))
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"{path}:{line_number}: assay sequence_id does not match sequence")
        if sequences.get(sequence_id) != sequence:
            raise ValueError(f"{path}:{line_number}: assay sequence is absent from sequence table")
        endpoint = str(row.get("endpoint", ""))
        if not endpoint:
            raise ValueError(f"{path}:{line_number}: assay endpoint is empty")
        strain = str(row.get("strain") or "")
        gram = str(row.get("gram", "unknown"))
        relation = str(row.get("relation", ""))
        unit = str(row.get("unit") or "missing")
        label = _activity_label(row, threshold)
        endpoints[endpoint] += 1
        grams[gram] += 1
        relations[relation] += 1
        units[unit] += 1
        source_units[str(row.get("source_unit") or "missing")] += 1
        exposure = row.get("exposure_concentration")
        if isinstance(exposure, Mapping):
            exposure_units[str(exposure.get("unit") or "missing")] += 1
        if label is not None:
            labels[label] += 1
        strains[strain or "<missing>"] += 1
        observations_by_sequence[sequence_id] += 1
        if label is not None:
            replicate_labels[(sequence_id, endpoint, strain)].append(label)
        issue = _target_issue(strain)
        if issue is not None:
            target_issues[issue] += 1
        for contract_issue in _endpoint_contract_issues(row):
            endpoint_contract_issues[contract_issue] += 1
        exact_key = json.dumps(
            {
                key: row.get(key)
                for key in (
                    "sequence_id",
                    "endpoint",
                    "organism",
                    "strain",
                    "gram",
                    "relation",
                    "lower",
                    "upper",
                    "lower_inclusive",
                    "upper_inclusive",
                    "unit",
                    "source_unit",
                    "raw_value",
                    "assay",
                    "exposure_concentration",
                    "source_text",
                    "provenance",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exact_rows[exact_key] += 1

    replicate_groups = Counter()
    for group_labels in replicate_labels.values():
        if len(group_labels) == 1:
            replicate_groups["single_observation"] += 1
        elif "ambiguous" in group_labels:
            replicate_groups["multiple_with_ambiguous"] += 1
        elif len(set(group_labels)) > 1:
            replicate_groups["multiple_conflicting_labels"] += 1
        else:
            replicate_groups["multiple_consistent_labels"] += 1
    counts = np.asarray(list(observations_by_sequence.values()), dtype=np.int64)
    return {
        "observations": observations,
        "endpoints": dict(sorted(endpoints.items())),
        "grams": dict(sorted(grams.items())),
        "relations": dict(sorted(relations.items())),
        "units": dict(sorted(units.items())),
        "source_units": dict(sorted(source_units.items())),
        "exposure_units": dict(sorted(exposure_units.items())),
        "threshold_um": threshold,
        "observation_labels": dict(sorted(labels.items())),
        "distinct_target_contexts": len(strains),
        "top_target_contexts": dict(strains.most_common(25)),
        "target_context_issues": dict(sorted(target_issues.items())),
        "endpoint_contract_issues": dict(sorted(endpoint_contract_issues.items())),
        "observations_per_sequence": {
            "minimum": int(np.min(counts)),
            "median": float(np.median(counts)),
            "maximum": int(np.max(counts)),
        },
        "sequence_target_groups": len(replicate_labels),
        "replicate_groups": dict(sorted(replicate_groups.items())),
        "exact_duplicate_observations": sum(count - 1 for count in exact_rows.values()),
        "exact_duplicate_groups": sum(count > 1 for count in exact_rows.values()),
    }


def _reference_rows(
    sequences: Mapping[str, str],
    *,
    reference_sequences: Sequence[str],
    comparison_sequences: Sequence[str],
    similarity_limit: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    reference_set = set(reference_sequences)
    comparison_set = set(comparison_sequences)
    reference_index = ReferenceIndex(reference_sequences)
    comparison_index = ReferenceIndex(comparison_sequences) if comparison_sequences else None
    rows: list[dict[str, object]] = []
    reference_above = 0
    comparison_above = 0
    for sequence_id, sequence in sorted(sequences.items(), key=lambda item: item[1]):
        exact_reference = sequence in reference_set
        if exact_reference:
            reference_ratio, nearest_reference = 1.0, sequence
        else:
            reference_ratio, nearest_reference = reference_index.max_ratio(
                sequence, threshold=similarity_limit
            )
        reference_above += reference_ratio > similarity_limit

        exact_comparison = sequence in comparison_set
        if exact_comparison:
            comparison_ratio, nearest_comparison = 1.0, sequence
        elif comparison_index is None:
            comparison_ratio, nearest_comparison = 0.0, None
        else:
            comparison_ratio, nearest_comparison = comparison_index.max_ratio(
                sequence, threshold=similarity_limit
            )
        comparison_above += comparison_ratio > similarity_limit
        rows.append(
            {
                "sequence_id": sequence_id,
                "sequence": sequence,
                "exact_reference_overlap": exact_reference,
                "reference_ratio": reference_ratio,
                "nearest_reference": nearest_reference or "",
                "above_reference_limit": reference_ratio > similarity_limit,
                "exact_comparison_overlap": exact_comparison,
                "comparison_ratio": comparison_ratio,
                "nearest_comparison": nearest_comparison or "",
                "above_comparison_limit": comparison_ratio > similarity_limit,
            }
        )
    return rows, {
        "similarity_metric": "Levenshtein.ratio (first value above limit, otherwise exact maximum)",
        "similarity_limit": similarity_limit,
        "reference_records": len(reference_sequences),
        "exact_reference_overlap": sum(row["exact_reference_overlap"] for row in rows),
        "above_reference_limit": reference_above,
        "comparison_records": len(comparison_sequences),
        "exact_comparison_overlap": sum(row["exact_comparison_overlap"] for row in rows),
        "above_comparison_limit": comparison_above,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: format(value, ".12g") if isinstance(value, float) else value
                    for name, value in row.items()
                }
            )


def audit_normalized_dataset(
    normalized_dir: str | Path,
    *,
    reference_path: str | Path,
    output_dir: str | Path,
    comparison_sequences_path: str | Path | None = None,
    activity_threshold_um: float = 16.0,
    similarity_limit: float = 0.8,
    require_clean_endpoint_contract: bool = False,
) -> dict[str, object]:
    """Write a deterministic dataset/reference audit and return its summary."""

    if not math.isfinite(activity_threshold_um) or activity_threshold_um <= 0:
        raise ValueError("activity_threshold_um must be finite and positive")
    if not math.isfinite(similarity_limit) or not 0 <= similarity_limit <= 1:
        raise ValueError("similarity_limit must be in [0, 1]")
    root = Path(normalized_dir).resolve()
    source_summary, paths = _verify_summary(root)
    sequences = _read_sequences(paths["sequences"])
    reference = read_organizer_fasta(reference_path)
    comparison_sequences: tuple[str, ...] = ()
    comparison_sha: str | None = None
    if comparison_sequences_path is not None:
        comparison_path = Path(comparison_sequences_path).resolve()
        comparison_sequences = tuple(_read_sequences(comparison_path).values())
        comparison_sha = _sha256(comparison_path)

    reference_rows, overlap = _reference_rows(
        sequences,
        reference_sequences=reference.sequences,
        comparison_sequences=comparison_sequences,
        similarity_limit=similarity_limit,
    )
    assays = _assay_summary(
        paths["assays"],
        sequences=sequences,
        threshold=activity_threshold_um,
    )
    endpoint_contract_issues = assays["endpoint_contract_issues"]
    if require_clean_endpoint_contract and endpoint_contract_issues:
        raise ValueError(
            "normalized endpoint contract check failed: "
            f"{json.dumps(endpoint_contract_issues, sort_keys=True)}"
        )
    rejected_rows = sum(1 for _ in _read_jsonl(paths["rejects"]))
    normalized_schema_version = int(source_summary["schema_version"])
    endpoint_rejections: dict[str, object] | None = None
    source_row_dispositions: dict[str, object] | None = None
    if normalized_schema_version == 2:
        normalized_artifacts = _schema_v2_artifacts(source_summary.get("artifacts"))
        endpoint_rejections = _endpoint_reject_summary(
            paths["endpoint_rejects"],
            assays_path=paths["assays"],
            rejects_path=paths["rejects"],
            sequences=sequences,
            artifacts=normalized_artifacts,
        )
        source_row_dispositions = _source_row_dispositions(
            assays_path=paths["assays"],
            rejects_path=paths["rejects"],
            endpoint_rejects_path=paths["endpoint_rejects"],
        )
        source_rows_expected = source_summary.get("source_rows_expected")
        if (
            isinstance(source_rows_expected, bool)
            or not isinstance(source_rows_expected, int)
            or source_rows_expected <= 0
        ):
            raise ValueError("normalized schema v2 requires a positive source_rows_expected")
        if source_row_dispositions["source_rows"] != source_rows_expected:
            raise ValueError(
                "normalized source-row conservation mismatch: "
                f"expected {source_rows_expected}, got "
                f"{source_row_dispositions['source_rows']}"
            )
        if "dramp_general_v2" in normalized_artifacts and source_rows_expected != 5_084:
            raise ValueError(
                "normalized DRAMP source-row oracle mismatch: expected 5084, "
                f"got {source_rows_expected}"
            )
    expected_counts = {
        "unique_sequences": len(sequences),
        "assay_observations": assays["observations"],
        "rejected_rows": rejected_rows,
    }
    if endpoint_rejections is not None:
        expected_counts["endpoint_rejects"] = endpoint_rejections["events"]
        expected_counts["rows_with_endpoint_rejects"] = endpoint_rejections["rows"]
        assert source_row_dispositions is not None
        expected_counts["source_rows"] = source_row_dispositions["source_rows"]
        expected_counts["accepted_source_rows"] = source_row_dispositions["accepted_source_rows"]
    count_mismatches = {
        key: (expected, source_summary.get(key))
        for key, expected in expected_counts.items()
        if source_summary.get(key) != expected
    }
    if count_mismatches:
        details = ", ".join(
            f"{key}=expected {expected}, got {actual}"
            for key, (expected, actual) in sorted(count_mismatches.items())
        )
        raise ValueError(f"normalized summary count mismatch: {details}")
    if endpoint_rejections is not None:
        if source_summary.get("endpoint_rejection_counts") != endpoint_rejections["reason_counts"]:
            raise ValueError("normalized summary endpoint rejection reason counts mismatch")
        assert source_row_dispositions is not None
        if (
            source_summary.get("row_disposition_counts")
            != source_row_dispositions["row_disposition_counts"]
        ):
            raise ValueError("normalized summary row disposition counts mismatch")
    declared_endpoint_counts = source_summary.get("endpoint_counts")
    if declared_endpoint_counts is not None:
        if not isinstance(declared_endpoint_counts, Mapping):
            raise ValueError("normalized summary endpoint_counts must be an object")
        normalized_declared = {
            str(endpoint): int(count) for endpoint, count in declared_endpoint_counts.items()
        }
        if normalized_declared != assays["endpoints"]:
            raise ValueError(
                "normalized summary endpoint count mismatch: "
                f"expected {json.dumps(assays['endpoints'], sort_keys=True)}, "
                f"got {json.dumps(normalized_declared, sort_keys=True)}"
            )

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "sequence_reference_audit.csv"
    summary_path = output / "audit_summary.json"
    _write_csv(rows_path, reference_rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "normalized_schema_version": normalized_schema_version,
        "normalized_summary_sha256": _sha256(paths["summary"]),
        "normalized_artifacts": source_summary.get("artifacts"),
        "normalized_parser_id": source_summary.get("parser_id"),
        "normalized_endpoint_counts": source_summary.get("endpoint_counts"),
        "sequences_sha256": _sha256(paths["sequences"]),
        "assays_sha256": _sha256(paths["assays"]),
        "rejects_sha256": _sha256(paths["rejects"]),
        "reference_sha256": _sha256(reference_path),
        "comparison_sequences_sha256": comparison_sha,
        "unique_sequences": len(sequences),
        "rejected_rows": rejected_rows,
        "reference_overlap": overlap,
        "assays": assays,
        "sequence_reference_audit_sha256": _sha256(rows_path),
    }
    if endpoint_rejections is not None:
        summary["endpoint_rejects_sha256"] = _sha256(paths["endpoint_rejects"])
        summary["endpoint_rejections"] = endpoint_rejections
        summary["source_row_dispositions"] = source_row_dispositions
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison-sequences", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activity-threshold-um", type=float, default=16.0)
    parser.add_argument("--similarity-limit", type=float, default=0.8)
    parser.add_argument(
        "--require-clean-endpoint-contract",
        action="store_true",
        help="fail if any normalized HC50 or hemolysis observation violates its schema",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_normalized_dataset(
        args.normalized_dir,
        reference_path=args.reference,
        comparison_sequences_path=args.comparison_sequences,
        output_dir=args.output_dir,
        activity_threshold_um=args.activity_threshold_um,
        similarity_limit=args.similarity_limit,
        require_clean_endpoint_contract=args.require_clean_endpoint_contract,
    )
    print(
        json.dumps(
            {
                "unique_sequences": summary["unique_sequences"],
                "reference_overlap": summary["reference_overlap"],
                "output": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

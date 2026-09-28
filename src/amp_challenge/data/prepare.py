"""Small, auditable FASTA/CSV/TSV ingestion and deterministic deduplication.

Source-specific databases will still need adapters for their assay semantics.
This module deliberately rejects ambiguous numeric prose instead of embedding a
large collection of fragile heuristics in the shared data contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..descriptors import molecular_weight
from ..sequences import SequenceValidationError
from .records import (
    AssayObservation,
    PeptideRecord,
    Provenance,
    convert_concentration_to_um,
    parse_censored_value,
)

ENDPOINT_REJECTION_REASON_CODES: dict[str, frozenset[str]] = {
    "hemolysis": frozenset(
        {
            "ambiguous_or_nonblood_target",
            "blood_agar_target",
            "hemolysis_percent_out_of_range",
            "missing_blood_target",
            "mixed_supported_and_unsupported_numeric_hemolysis",
            "nonpositive_hc50",
            "unsupported_half_max_alias",
            "unsupported_hc50_measurement",
            "unsupported_hemolysis_endpoint_alias",
            "unsupported_hemolysis_effect_measurement",
            "unsupported_hemolysis_exposure_measurement",
            "unsupported_or_ambiguous_concentration_specific_hemolysis",
            "unsupported_or_ambiguous_hc50",
            "unsupported_uncertainty_notation",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """An input entry retained for audit rather than silently discarded."""

    source: str
    path: str
    row_number: int
    raw_sequence: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RejectedEndpoint:
    """One source endpoint field withheld without invalidating the peptide row."""

    rejection_id: str
    scope: Literal["endpoint_field"]
    endpoint_family: str
    source_field: str
    sequence_id: str
    sequence: str
    raw_sequence: str | None
    source_text: str
    reason_code: str
    reason_detail: str
    discarded_candidate_observations: int
    provenance: Provenance

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.rejection_id) is None:
            raise ValueError("endpoint rejection_id must be a lowercase SHA-256")
        if self.scope != "endpoint_field":
            raise ValueError("endpoint rejection scope must be 'endpoint_field'")
        for name in ("endpoint_family", "source_field", "source_text", "reason_code"):
            if not getattr(self, name).strip():
                raise ValueError(f"endpoint rejection {name} cannot be blank")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.reason_code):
            raise ValueError("endpoint rejection reason_code must be snake_case")
        allowed_codes = ENDPOINT_REJECTION_REASON_CODES.get(self.endpoint_family)
        if allowed_codes is None or self.reason_code not in allowed_codes:
            raise ValueError(
                "endpoint rejection reason_code is not allowed for "
                f"{self.endpoint_family!r}: {self.reason_code!r}"
            )
        if not self.reason_detail.strip():
            raise ValueError("endpoint rejection reason_detail cannot be blank")
        if isinstance(self.discarded_candidate_observations, bool) or not isinstance(
            self.discarded_candidate_observations, int
        ):
            raise ValueError("discarded_candidate_observations must be a non-negative integer")
        if self.discarded_candidate_observations < 0:
            raise ValueError("discarded_candidate_observations must be a non-negative integer")


def endpoint_rejection_id(
    *,
    provenance: Provenance,
    endpoint_family: str,
    source_field: str,
    sequence_id: str,
) -> str:
    """Return a storage-path-independent identifier for one rejected source field."""

    extra = dict(provenance.extra)
    identity = {
        "source": provenance.source,
        "source_sha256": extra.get("source_sha256"),
        "source_version": extra.get("source_version") or extra.get("source_commit"),
        "record_id": provenance.record_id,
        "row_number": provenance.row_number,
        "sequence_id": sequence_id,
        "endpoint_family": endpoint_family,
        "source_field": source_field,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IngestionResult:
    records: tuple[PeptideRecord, ...]
    rejected: tuple[RejectedRecord, ...] = ()
    rejected_endpoints: tuple[RejectedEndpoint, ...] = ()

    def __len__(self) -> int:
        return len(self.records)


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sequence": ("sequence", "seq", "peptide", "peptidesequence", "aminoacidsequence"),
    "record_id": ("recordid", "id", "accession", "entry", "name"),
    "mic": ("mic", "micvalue", "minimuminhibitoryconcentration"),
    "mic_unit": ("micunit", "unit", "units", "concentrationunit"),
    "organism": ("organism", "species", "targetorganism", "bacteria"),
    "strain": ("strain", "targetstrain", "isolate"),
    "gram": ("gram", "gramclass", "gramstain"),
    "assay": ("assay", "assaytype", "method"),
}


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _resolve_columns(
    fieldnames: Sequence[str],
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    resolved = dict(explicit or {})
    unknown = set(resolved) - set(_COLUMN_ALIASES)
    if unknown:
        raise ValueError(f"unknown canonical column name(s): {sorted(unknown)}")
    missing_actual = set(resolved.values()) - set(fieldnames)
    if missing_actual:
        raise ValueError(f"mapped input column(s) not found: {sorted(missing_actual)}")

    normalized_to_actual = {_normalized_header(name): name for name in fieldnames}
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        if canonical_name in resolved:
            continue
        for alias in aliases:
            if alias in normalized_to_actual:
                resolved[canonical_name] = normalized_to_actual[alias]
                break
    if "sequence" not in resolved:
        raise ValueError("no sequence column found; provide columns={'sequence': '<input column>'}")
    return resolved


def _normalize_gram(raw: str | None) -> str:
    if raw is None or not raw.strip():
        return "unknown"
    compact = raw.lower().replace(" ", "")
    if compact in {"g+", "gram+", "gram-positive", "grampositive"}:
        return "positive"
    if compact in {"g-", "gram-", "gram-negative", "gramnegative"}:
        return "negative"
    normalized = _normalized_header(raw)
    positive = {"positive", "pos", "grampositive", "gpositive", "gpos"}
    negative = {"negative", "neg", "gramnegative", "gnegative", "gneg"}
    if normalized in positive:
        return "positive"
    if normalized in negative:
        return "negative"
    return "unknown"


def _raise_or_reject(
    *,
    strict: bool,
    rejected: list[RejectedRecord],
    source: str,
    path: Path,
    row_number: int,
    raw_sequence: str | None,
    error: Exception | str,
) -> None:
    reason = str(error)
    if strict:
        raise ValueError(f"{path}:{row_number}: {reason}") from (
            error if isinstance(error, Exception) else None
        )
    rejected.append(
        RejectedRecord(
            source=source,
            path=str(path),
            row_number=row_number,
            raw_sequence=raw_sequence,
            reason=reason,
        )
    )


def ingest_fasta(
    path: str | Path,
    *,
    source: str | None = None,
    strict: bool = False,
) -> IngestionResult:
    """Read FASTA entries, preserving headers and rejected sequences."""

    input_path = Path(path)
    source_name = source or input_path.stem
    records: list[PeptideRecord] = []
    rejected: list[RejectedRecord] = []
    header: str | None = None
    header_line: int | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, header_line, chunks
        if header is None or header_line is None:
            return
        raw_sequence = "".join(chunks)
        parts = header.split(maxsplit=1)
        record_id = parts[0] if parts else None
        description = parts[1] if len(parts) > 1 else ""
        provenance = Provenance.from_mapping(
            source=source_name,
            record_id=record_id,
            path=str(input_path),
            row_number=header_line,
            extra={"description": description},
        )
        try:
            records.append(PeptideRecord.from_sequence(raw_sequence, provenance=provenance))
        except (TypeError, ValueError, SequenceValidationError) as error:
            _raise_or_reject(
                strict=strict,
                rejected=rejected,
                source=source_name,
                path=input_path,
                row_number=header_line,
                raw_sequence=raw_sequence,
                error=error,
            )
        header = None
        header_line = None
        chunks = []

    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.startswith(">"):
                flush()
                header = stripped[1:].strip()
                header_line = line_number
                chunks = []
                if not header:
                    _raise_or_reject(
                        strict=strict,
                        rejected=rejected,
                        source=source_name,
                        path=input_path,
                        row_number=line_number,
                        raw_sequence=None,
                        error="empty FASTA header",
                    )
                    header = None
                    header_line = None
                continue
            if header is None:
                _raise_or_reject(
                    strict=strict,
                    rejected=rejected,
                    source=source_name,
                    path=input_path,
                    row_number=line_number,
                    raw_sequence=stripped,
                    error="sequence data encountered before a FASTA header",
                )
                continue
            chunks.append(stripped)
    flush()
    return IngestionResult(records=tuple(records), rejected=tuple(rejected))


def ingest_delimited(
    path: str | Path,
    *,
    source: str | None = None,
    delimiter: str | None = None,
    columns: Mapping[str, str] | None = None,
    normalize_mic_to_um: bool = True,
    strict: bool = False,
) -> IngestionResult:
    """Read a headered CSV or TSV into normalized peptide records.

    Known MIC units are converted to micromolar while retaining the original
    censor relation and unit.  Unitless MIC values remain unitless; unknown units
    are rejected so they cannot be accidentally pooled with normalized values.
    """

    input_path = Path(path)
    source_name = source or input_path.stem
    actual_delimiter = delimiter
    if actual_delimiter is None:
        actual_delimiter = "\t" if input_path.suffix.lower() in {".tsv", ".tab"} else ","
    if len(actual_delimiter) != 1:
        raise ValueError("delimiter must be one character")

    records: list[PeptideRecord] = []
    rejected: list[RejectedRecord] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=actual_delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path}: missing header row")
        resolved = _resolve_columns(reader.fieldnames, columns)
        for row_number, row in enumerate(reader, start=2):
            raw_sequence = row.get(resolved["sequence"])
            extra = {
                key: value
                for key, value in row.items()
                if key != resolved["sequence"] and value is not None and value.strip()
            }
            record_id = _optional_cell(row, resolved.get("record_id"))
            provenance = Provenance.from_mapping(
                source=source_name,
                record_id=record_id,
                path=str(input_path),
                row_number=row_number,
                extra=extra,
            )
            try:
                base_record = PeptideRecord.from_sequence(
                    raw_sequence or "",
                    provenance=provenance,
                )
                assays: tuple[AssayObservation, ...] = ()
                mic_text = _optional_cell(row, resolved.get("mic"))
                if mic_text is not None:
                    mic_unit = _optional_cell(row, resolved.get("mic_unit"))
                    measurement = parse_censored_value(mic_text, unit=mic_unit)
                    if measurement is not None:
                        if normalize_mic_to_um and measurement.unit is not None:
                            measurement = convert_concentration_to_um(
                                measurement,
                                molecular_weight_da=molecular_weight(base_record.sequence),
                            )
                        assays = (
                            AssayObservation(
                                endpoint="mic",
                                value=measurement,
                                provenance=provenance,
                                organism=_optional_cell(row, resolved.get("organism")),
                                strain=_optional_cell(row, resolved.get("strain")),
                                gram=_normalize_gram(_optional_cell(row, resolved.get("gram"))),
                                assay=_optional_cell(row, resolved.get("assay")),
                            ),
                        )
                records.append(
                    PeptideRecord(
                        sequence_id=base_record.sequence_id,
                        sequence=base_record.sequence,
                        provenance=base_record.provenance,
                        assays=assays,
                    )
                )
            except (TypeError, ValueError, SequenceValidationError) as error:
                _raise_or_reject(
                    strict=strict,
                    rejected=rejected,
                    source=source_name,
                    path=input_path,
                    row_number=row_number,
                    raw_sequence=raw_sequence,
                    error=error,
                )
    return IngestionResult(records=tuple(records), rejected=tuple(rejected))


def _optional_cell(row: Mapping[str, str | None], column: str | None) -> str | None:
    if column is None:
        return None
    value = row.get(column)
    if value is None or not value.strip():
        return None
    return value.strip()


def ingest_path(
    path: str | Path,
    **kwargs: object,
) -> IngestionResult:
    """Dispatch to FASTA or delimited ingestion based on the file extension."""

    suffix = Path(path).suffix.lower()
    if suffix in {".fa", ".faa", ".fas", ".fasta"}:
        allowed = {"source", "strict"}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(f"FASTA ingestion does not accept: {sorted(unexpected)}")
        return ingest_fasta(path, **kwargs)  # type: ignore[arg-type]
    if suffix in {".csv", ".tsv", ".tab"}:
        return ingest_delimited(path, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unsupported input extension: {suffix!r}")


def _provenance_key(item: Provenance) -> tuple[str, str, str, int, tuple[tuple[str, str], ...]]:
    return (
        item.source,
        item.record_id or "",
        item.path or "",
        item.row_number or 0,
        item.extra,
    )


def _assay_key(item: AssayObservation) -> tuple[object, ...]:
    value = item.value
    exposure = item.exposure_concentration
    return (
        item.endpoint,
        item.organism or "",
        item.strain or "",
        item.gram,
        item.assay or "",
        value.unit or "",
        value.source_unit or "",
        value.relation,
        -1.0 if value.lower is None else value.lower,
        -1.0 if value.upper is None else value.upper,
        value.lower_inclusive,
        value.upper_inclusive,
        value.raw or "",
        "" if exposure is None else exposure.unit or "",
        "" if exposure is None else exposure.source_unit or "",
        "" if exposure is None else exposure.relation,
        -1.0 if exposure is None or exposure.lower is None else exposure.lower,
        -1.0 if exposure is None or exposure.upper is None else exposure.upper,
        False if exposure is None else exposure.lower_inclusive,
        False if exposure is None else exposure.upper_inclusive,
        "" if exposure is None else exposure.raw or "",
        item.source_text or "",
        _provenance_key(item.provenance),
    )


def deduplicate_records(records: Iterable[PeptideRecord]) -> tuple[PeptideRecord, ...]:
    """Merge exact canonical sequences deterministically, preserving all evidence."""

    grouped: dict[str, list[PeptideRecord]] = {}
    for record in records:
        grouped.setdefault(record.sequence_id, []).append(record)

    merged: list[PeptideRecord] = []
    for sequence_id in sorted(grouped):
        group = grouped[sequence_id]
        sequences = {record.sequence for record in group}
        if len(sequences) != 1:
            raise RuntimeError(f"SHA-256 collision detected for {sequence_id}")
        provenance = tuple(
            sorted(
                {item for record in group for item in record.provenance},
                key=_provenance_key,
            )
        )
        assays = tuple(
            sorted(
                {item for record in group for item in record.assays},
                key=_assay_key,
            )
        )
        merged.append(
            PeptideRecord(
                sequence_id=sequence_id,
                sequence=next(iter(sequences)),
                provenance=provenance,
                assays=assays,
            )
        )
    return tuple(merged)


def prepare_records(
    paths: Iterable[str | Path],
    *,
    strict: bool = False,
) -> IngestionResult:
    """Ingest supported files and return one deterministic record per sequence."""

    all_records: list[PeptideRecord] = []
    all_rejected: list[RejectedRecord] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        result = ingest_path(path, strict=strict)
        all_records.extend(result.records)
        all_rejected.extend(result.rejected)
    return IngestionResult(
        records=deduplicate_records(all_records),
        rejected=tuple(
            sorted(
                all_rejected,
                key=lambda item: (item.path, item.row_number, item.reason),
            )
        ),
    )


# Conventional read aliases for callers that do not need the rejected-entry log.
def read_fasta(path: str | Path, **kwargs: object) -> tuple[PeptideRecord, ...]:
    return ingest_fasta(path, **kwargs).records  # type: ignore[arg-type]


def read_delimited(path: str | Path, **kwargs: object) -> tuple[PeptideRecord, ...]:
    return ingest_delimited(path, **kwargs).records  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for preparing a small, nested JSONL audit dataset."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="FASTA, CSV, or TSV input files")
    parser.add_argument("--output", type=Path, help="optional normalized JSONL output")
    parser.add_argument("--rejects", type=Path, help="optional rejected-entry JSONL output")
    parser.add_argument("--strict", action="store_true", help="stop at the first invalid entry")
    args = parser.parse_args(argv)

    result = prepare_records(args.inputs, strict=args.strict)
    if args.output is not None:
        _write_json_lines(args.output, (asdict(record) for record in result.records))
    if args.rejects is not None:
        _write_json_lines(args.rejects, (asdict(record) for record in result.rejected))
    print(
        json.dumps(
            {
                "input_files": len(args.inputs),
                "unique_sequences": len(result.records),
                "rejected_entries": len(result.rejected),
            },
            sort_keys=True,
        )
    )
    return 0


def _write_json_lines(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())

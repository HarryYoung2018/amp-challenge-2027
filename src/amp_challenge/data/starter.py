"""Normalize approved labeled artifacts from the organizer starter kits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from amp_challenge.data.fetch import DataArtifact, load_data_manifest, verify_artifact
from amp_challenge.data.prepare import IngestionResult, RejectedRecord, deduplicate_records
from amp_challenge.data.records import (
    AssayObservation,
    CensoredValue,
    PeptideRecord,
    Provenance,
    parse_censored_value,
)
from amp_challenge.sequences import SequenceValidationError

GramClass = Literal["positive", "negative", "unknown"]


def _gram(raw: str) -> GramClass:
    normalized = raw.strip().lower().replace(" ", "")
    if normalized in {"gram+", "g+", "+", "positive", "gram-positive"}:
        return "positive"
    if normalized in {"gram-", "g-", "-", "negative", "gram-negative"}:
        return "negative"
    return "unknown"


def ingest_ampdiffusion_experimental_mic(
    path: str | Path,
    *,
    artifact: DataArtifact,
    strict: bool = False,
) -> IngestionResult:
    """Normalize the CC-BY-4.0 AMP-Diffusion long-form MIC table.

    The source stores the numeric assay ceiling and censor relation in separate
    columns.  They are deliberately recombined before parsing so ``>64`` remains
    a right-censored observation rather than an exact MIC of 64.
    """

    input_path = Path(path)
    required = {
        "peptide_id",
        "sequence",
        "modification",
        "strain",
        "strain_type",
        "mic",
        "mic_unit",
        "mic_relation",
    }
    records: list[PeptideRecord] = []
    rejected: list[RejectedRecord] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"AMP-Diffusion MIC table missing column(s): {missing}")
        for row_number, row in enumerate(reader, start=2):
            sequence = row.get("sequence", "")
            modification = row.get("modification", "").strip()
            provenance = Provenance.from_mapping(
                source=artifact.name,
                record_id=row.get("peptide_id") or None,
                path=str(input_path),
                row_number=row_number,
                extra={
                    **row,
                    "source_repository": artifact.source_repository,
                    "source_commit": artifact.source_commit,
                    "source_sha256": artifact.sha256,
                    "source_license": artifact.license,
                    "training_status": artifact.training_status,
                },
            )
            try:
                if modification:
                    raise ValueError(
                        f"modified peptide is quarantined from free-terminus data: {modification}"
                    )
                relation = row.get("mic_relation", "").strip()
                if relation not in {"=", "<", "<=", ">", ">=", "~"}:
                    raise ValueError(f"unsupported MIC relation: {relation!r}")
                measurement = parse_censored_value(
                    f"{relation}{row.get('mic', '')}",
                    unit=row.get("mic_unit"),
                )
                if measurement is None:  # pragma: no cover - guarded by required source data
                    raise ValueError("missing MIC value")
                base = PeptideRecord.from_sequence(sequence, provenance=provenance)
                assay = AssayObservation(
                    endpoint="mic",
                    value=measurement,
                    provenance=provenance,
                    strain=row.get("strain") or None,
                    gram=_gram(row.get("strain_type", "")),
                    assay="broth microdilution",
                )
                records.append(
                    PeptideRecord(
                        sequence_id=base.sequence_id,
                        sequence=base.sequence,
                        provenance=base.provenance,
                        assays=(assay,),
                    )
                )
            except (TypeError, ValueError, SequenceValidationError) as error:
                if strict:
                    raise ValueError(f"{input_path}:{row_number}: {error}") from error
                rejected.append(
                    RejectedRecord(
                        source=artifact.name,
                        path=str(input_path),
                        row_number=row_number,
                        raw_sequence=sequence or None,
                        reason=str(error),
                    )
                )
    return IngestionResult(
        records=deduplicate_records(records),
        rejected=tuple(rejected),
    )


_APPROVED_ADAPTERS = {
    "ampdiffusion_experimental_mic": ingest_ampdiffusion_experimental_mic,
}


def prepare_approved_starter_data(
    *,
    manifest_path: str | Path,
    snapshot_root: str | Path,
    strict: bool = False,
) -> tuple[IngestionResult, tuple[DataArtifact, ...]]:
    """Verify and normalize every manifest artifact approved for training."""

    manifest = load_data_manifest(manifest_path)
    root = Path(snapshot_root)
    selected = tuple(
        artifact for artifact in manifest.artifacts if artifact.training_status == "approved"
    )
    missing_adapters = {artifact.name for artifact in selected} - set(_APPROVED_ADAPTERS)
    if missing_adapters:
        raise ValueError(
            f"approved artifacts lack normalization adapters: {sorted(missing_adapters)}"
        )
    all_records: list[PeptideRecord] = []
    all_rejected: list[RejectedRecord] = []
    for artifact in selected:
        path = root / artifact.relative_path
        verify_artifact(path, artifact)
        result = _APPROVED_ADAPTERS[artifact.name](path, artifact=artifact, strict=strict)
        all_records.extend(result.records)
        all_rejected.extend(result.rejected)
    return (
        IngestionResult(
            records=deduplicate_records(all_records),
            rejected=tuple(
                sorted(
                    all_rejected,
                    key=lambda item: (item.source, item.path, item.row_number),
                )
            ),
        ),
        selected,
    )


def write_normalized_dataset(
    output_dir: str | Path,
    *,
    result: IngestionResult,
    artifacts: Iterable[DataArtifact],
    schema_version: int = 1,
    expected_source_rows: int | None = None,
) -> dict[str, object]:
    """Write deterministic sequence, assay, rejection, and summary JSONL files."""

    if schema_version not in {1, 2}:
        raise ValueError("normalized schema_version must be 1 or 2")
    if schema_version == 1 and result.rejected_endpoints:
        raise ValueError("schema v1 cannot serialize endpoint-level rejections")
    if schema_version == 2 and (
        isinstance(expected_source_rows, bool)
        or not isinstance(expected_source_rows, int)
        or expected_source_rows <= 0
    ):
        raise ValueError("schema v2 requires a positive expected_source_rows oracle")

    accepted_source_rows = {
        (item.source, item.path or "", item.row_number or 0)
        for record in result.records
        for item in record.provenance
    }
    rejected_source_rows = {(item.source, item.path, item.row_number) for item in result.rejected}
    if len(rejected_source_rows) != len(result.rejected):
        raise ValueError("duplicate row-rejection key")
    if accepted_source_rows & rejected_source_rows:
        raise ValueError("source rows cannot be both accepted and row-rejected")
    endpoint_reject_rows = {
        (
            item.provenance.source,
            item.provenance.path or "",
            item.provenance.row_number or 0,
        )
        for item in result.rejected_endpoints
    }
    endpoint_rejection_ids = {item.rejection_id for item in result.rejected_endpoints}
    endpoint_rejection_fields = {
        (
            item.provenance.source,
            item.provenance.path or "",
            item.provenance.row_number or 0,
            item.endpoint_family,
            item.source_field,
        )
        for item in result.rejected_endpoints
    }
    if len(endpoint_rejection_ids) != len(result.rejected_endpoints) or len(
        endpoint_rejection_fields
    ) != len(result.rejected_endpoints):
        raise ValueError("duplicate endpoint-field rejection")
    source_rows = len(accepted_source_rows | rejected_source_rows | endpoint_reject_rows)
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise ValueError(
            f"source-row conservation failed: expected {expected_source_rows}, got {source_rows}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences_path = output / "sequences.jsonl"
    assays_path = output / "assays.jsonl"
    rejects_path = output / "rejects.jsonl"
    endpoint_rejects_path = output / "endpoint_rejects.jsonl"
    artifacts_tuple = tuple(artifacts)

    def serialized_value(value: CensoredValue | None) -> dict[str, object] | None:
        if value is None:
            return None
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

    _write_jsonl(
        sequences_path,
        (
            {
                "sequence_id": record.sequence_id,
                "sequence": record.sequence,
                "provenance": [asdict(item) for item in record.provenance],
            }
            for record in result.records
        ),
    )
    _write_jsonl(
        assays_path,
        (
            {
                "sequence_id": record.sequence_id,
                "sequence": record.sequence,
                "endpoint": assay.endpoint,
                "relation": assay.value.relation,
                "lower": assay.value.lower,
                "upper": assay.value.upper,
                "lower_inclusive": assay.value.lower_inclusive,
                "upper_inclusive": assay.value.upper_inclusive,
                "unit": assay.value.unit,
                "source_unit": assay.value.source_unit,
                "raw_value": assay.value.raw,
                "organism": assay.organism,
                "strain": assay.strain,
                "gram": assay.gram,
                "assay": assay.assay,
                "exposure_concentration": serialized_value(assay.exposure_concentration),
                "source_text": assay.source_text,
                "provenance": asdict(assay.provenance),
            }
            for record in result.records
            for assay in record.assays
        ),
    )
    _write_jsonl(rejects_path, (asdict(item) for item in result.rejected))
    if schema_version == 2:
        _write_jsonl(
            endpoint_rejects_path,
            (
                asdict(item)
                for item in sorted(
                    result.rejected_endpoints,
                    key=lambda item: item.rejection_id,
                )
            ),
        )

    serialized_artifacts: list[dict[str, object]] = []
    for artifact in artifacts_tuple:
        serialized_artifact: dict[str, object] = {
            "name": artifact.name,
            "sha256": artifact.sha256,
            "license": artifact.license,
            "training_status": artifact.training_status,
        }
        if schema_version == 2:
            serialized_artifact.update(
                {
                    "bytes": artifact.bytes,
                    "relative_path": artifact.relative_path,
                    "source_repository": artifact.source_repository,
                    "source_commit": artifact.source_commit,
                }
            )
        serialized_artifacts.append(serialized_artifact)

    summary: dict[str, object] = {
        "schema_version": schema_version,
        "artifacts": serialized_artifacts,
        "unique_sequences": len(result.records),
        "assay_observations": sum(len(record.assays) for record in result.records),
        "rejected_rows": len(result.rejected),
        "sequences_sha256": _sha256(sequences_path),
        "assays_sha256": _sha256(assays_path),
        "rejects_sha256": _sha256(rejects_path),
    }
    if schema_version == 2:
        summary.update(
            {
                "source_rows": source_rows,
                "source_rows_expected": expected_source_rows,
                "accepted_source_rows": len(accepted_source_rows),
                "endpoint_rejects": len(result.rejected_endpoints),
                "rows_with_endpoint_rejects": len(endpoint_reject_rows),
                "endpoint_rejection_counts": dict(
                    sorted(Counter(item.reason_code for item in result.rejected_endpoints).items())
                ),
                "row_disposition_counts": {
                    "accepted": len(accepted_source_rows),
                    "accepted_with_endpoint_reject": len(
                        accepted_source_rows & endpoint_reject_rows
                    ),
                    "rejected": len(rejected_source_rows),
                    "rejected_with_endpoint_reject": len(
                        rejected_source_rows & endpoint_reject_rows
                    ),
                },
                "endpoint_rejects_sha256": _sha256(endpoint_rejects_path),
            }
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data/starter_snapshots.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, artifacts = prepare_approved_starter_data(
        manifest_path=args.manifest,
        snapshot_root=args.snapshot_root,
        strict=args.strict,
    )
    summary = write_normalized_dataset(args.output_dir, result=result, artifacts=artifacts)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

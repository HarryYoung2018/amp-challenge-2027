"""Export a checksum-verified normalized sequence table as deterministic FASTA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_normalized_fasta(
    normalized_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write one canonical, sequence-ID-keyed FASTA record per normalized entity."""

    root = Path(normalized_dir).resolve()
    sequences_path = root / "sequences.jsonl"
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_hash = str(summary.get("sequences_sha256", ""))
    actual_hash = _sha256(sequences_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"normalized sequences checksum mismatch: expected {expected_hash!r}, "
            f"got {actual_hash!r}"
        )

    records: list[tuple[str, str]] = []
    observed_ids: set[str] = set()
    with sequences_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{sequences_path}:{line_number}: expected a JSON object")
            sequence = canonicalize_sequence(str(row.get("sequence", "")))
            sequence_id = str(row.get("sequence_id", ""))
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(
                    f"{sequences_path}:{line_number}: sequence_id does not match sequence"
                )
            if sequence_id in observed_ids:
                raise ValueError(f"{sequences_path}:{line_number}: duplicate sequence_id")
            observed_ids.add(sequence_id)
            records.append((sequence_id, sequence))
    if not records:
        raise ValueError("normalized sequence table is empty")
    expected_count = summary.get("unique_sequences")
    if expected_count != len(records):
        raise ValueError(
            f"normalized sequence count mismatch: expected {expected_count!r}, got {len(records)}"
        )

    records.sort(key=lambda item: (item[1], item[0]))
    output = Path(output_dir).resolve()
    fasta_path = output / "normalized_sequences.fasta"
    manifest_path = output / "fasta_manifest.json"
    if any(path.exists() for path in (fasta_path, manifest_path)):
        raise FileExistsError(f"refusing to overwrite normalized FASTA outputs in {output}")
    output.mkdir(parents=True, exist_ok=True)
    fasta_text = "".join(
        f">sequence_id={sequence_id}\n{sequence}\n" for sequence_id, sequence in records
    )
    fasta_path.write_text(fasta_text, encoding="ascii", newline="\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "normalized_summary_sha256": _sha256(summary_path),
        "normalized_sequences_sha256": actual_hash,
        "records": len(records),
        "ordering": "canonical sequence, then sequence_id",
        "fasta_sha256": _sha256(fasta_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_normalized_fasta(args.normalized_dir, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

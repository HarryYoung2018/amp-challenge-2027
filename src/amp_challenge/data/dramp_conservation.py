"""Verify parser-v6 endpoint isolation against the frozen DRAMP v4/v5 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

EXPECTED_SOURCE_ROWS = 5_084
EXPECTED_V4_MIC = 5_915
EXPECTED_V5_MIC = 5_516
EXPECTED_V6_MIC = 5_903
EXPECTED_V6_ENDPOINTS = {"hc50": 37, "hemolysis_percent": 152, "mic": 5_903}
EXPECTED_V6_LABELS = {"active": 4_167, "ambiguous": 92, "inactive": 1_644}
EXPECTED_ENDPOINT_REJECT_ROWS = 93
EXPECTED_RECOVERED_ROWS = 71
EXPECTED_RECOVERED_SEQUENCES = 69
EXPECTED_RECOVERED_MIC = 387
EXPECTED_NEW_CHEMISTRY_ROWS = 2
EXPECTED_NEW_CHEMISTRY_MIC = 12

RowKey = tuple[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _extra(provenance: Mapping[str, object]) -> dict[str, str]:
    raw = provenance.get("extra")
    if not isinstance(raw, list):
        raise ValueError("assay provenance.extra must be a list")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("assay provenance.extra entries must be pairs")
        result[str(item[0])] = str(item[1])
    return result


def _row_key(provenance: Mapping[str, object]) -> RowKey:
    source = str(provenance.get("source") or "")
    row_number = provenance.get("row_number")
    if not source or isinstance(row_number, bool) or not isinstance(row_number, int):
        raise ValueError("assay provenance requires source and integer row_number")
    return source, row_number


def _assay_fingerprint(row: Mapping[str, object], *, include_endpoint_context: bool) -> str:
    if include_endpoint_context:
        # HC50 and concentration-specific hemolysis records must remain exactly
        # byte-equivalent at the normalized row level between v5 and v6.
        return json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("assay provenance must be an object")
    extra = _extra(provenance)
    payload: dict[str, object] = {
        "source": provenance.get("source"),
        "source_sha256": extra.get("source_sha256"),
        "record_id": provenance.get("record_id"),
        "row_number": provenance.get("row_number"),
        "sequence_id": row.get("sequence_id"),
        "sequence": row.get("sequence"),
        "endpoint": row.get("endpoint"),
        "organism": row.get("organism"),
        "strain": row.get("strain"),
        "gram": row.get("gram"),
        "relation": row.get("relation"),
        "lower": row.get("lower"),
        "upper": row.get("upper"),
        "lower_inclusive": row.get("lower_inclusive"),
        "upper_inclusive": row.get("upper_inclusive"),
        "unit": row.get("unit"),
        "raw_value": row.get("raw_value"),
        "assay": row.get("assay"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _assays(
    root: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, tuple[RowKey, str]],
    dict[RowKey, set[str]],
]:
    fingerprints: dict[str, set[str]] = defaultdict(set)
    mic_metadata: dict[str, tuple[RowKey, str]] = {}
    endpoints_by_row: dict[RowKey, set[str]] = defaultdict(set)
    rows = 0
    for assay in _read_jsonl(root / "assays.jsonl"):
        rows += 1
        endpoint = str(assay.get("endpoint") or "")
        provenance = assay.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("assay provenance must be an object")
        row_key = _row_key(provenance)
        fingerprint = _assay_fingerprint(
            assay,
            include_endpoint_context=endpoint != "mic",
        )
        if fingerprint in fingerprints[endpoint]:
            raise ValueError(f"duplicate {endpoint} assay fingerprint in {root}")
        fingerprints[endpoint].add(fingerprint)
        endpoints_by_row[row_key].add(endpoint)
        if endpoint == "mic":
            mic_metadata[fingerprint] = (row_key, str(assay.get("sequence_id") or ""))
    if sum(len(values) for values in fingerprints.values()) != rows:
        raise ValueError(f"assay fingerprint accounting failed in {root}")
    return dict(fingerprints), mic_metadata, dict(endpoints_by_row)


def _rejects(root: Path) -> dict[RowKey, str]:
    result: dict[RowKey, str] = {}
    for row in _read_jsonl(root / "rejects.jsonl"):
        row_number = row.get("row_number")
        if isinstance(row_number, bool) or not isinstance(row_number, int):
            raise ValueError("row rejection row_number must be an integer")
        key = str(row.get("source") or ""), row_number
        if key in result:
            raise ValueError(f"duplicate row rejection in {root}: {key}")
        result[key] = str(row.get("reason") or "")
    return result


def _is_hemolysis_reject(reason: str) -> bool:
    return reason.startswith(("hemolysis observation quarantine:", "numeric hemolysis evidence"))


def _is_global_reject(reason: str) -> bool:
    return reason.startswith(("chemistry quarantine:", "non-standard residue", "sequence length"))


def _endpoint_rejects(root: Path) -> tuple[set[RowKey], Counter[str], int]:
    rows: set[RowKey] = set()
    reasons: Counter[str] = Counter()
    discarded = 0
    events = 0
    for event in _read_jsonl(root / "endpoint_rejects.jsonl"):
        events += 1
        provenance = event.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("endpoint rejection provenance must be an object")
        key = _row_key(provenance)
        if key in rows:
            raise ValueError(f"duplicate endpoint-field rejection in {root}: {key}")
        rows.add(key)
        reasons[str(event.get("reason_code") or "")] += 1
        discarded_raw = event.get("discarded_candidate_observations")
        if isinstance(discarded_raw, bool) or not isinstance(discarded_raw, int):
            raise ValueError("discarded_candidate_observations must be an integer")
        discarded += discarded_raw
    if events != len(rows):
        raise ValueError("endpoint rejection row accounting failed")
    return rows, reasons, discarded


def _mic_labels(root: Path, *, threshold: float = 16.0) -> Counter[str]:
    labels: Counter[str] = Counter()
    for row in _read_jsonl(root / "assays.jsonl"):
        if row.get("endpoint") != "mic":
            continue
        relation = str(row.get("relation") or "")
        if relation == "approx" or row.get("unit") != "uM":
            labels["ambiguous"] += 1
            continue
        lower = None if row.get("lower") is None else float(row["lower"])
        upper = None if row.get("upper") is None else float(row["upper"])
        if upper is not None and upper <= threshold:
            labels["active"] += 1
        elif lower is not None and (
            lower > threshold
            or (lower == threshold and not bool(row.get("lower_inclusive", False)))
        ):
            labels["inactive"] += 1
        else:
            labels["ambiguous"] += 1
    return labels


def _summary(root: Path) -> dict[str, object]:
    value = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{root}/summary.json must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_conservation(
    *,
    v4_root: str | Path,
    v5_root: str | Path,
    v6_root: str | Path,
) -> dict[str, object]:
    """Return diagnostics after enforcing the frozen v6 conservation oracle."""

    roots = {
        "v4": Path(v4_root).resolve(),
        "v5": Path(v5_root).resolve(),
        "v6": Path(v6_root).resolve(),
    }
    summaries = {name: _summary(root) for name, root in roots.items()}
    assays: dict[str, dict[str, set[str]]] = {}
    mic_metadata: dict[str, dict[str, tuple[RowKey, str]]] = {}
    endpoints_by_row: dict[str, dict[RowKey, set[str]]] = {}
    rejects: dict[str, dict[RowKey, str]] = {}
    for name, root in roots.items():
        assays[name], mic_metadata[name], endpoints_by_row[name] = _assays(root)
        rejects[name] = _rejects(root)

    v4_mic = assays["v4"].get("mic", set())
    v5_mic = assays["v5"].get("mic", set())
    v6_mic = assays["v6"].get("mic", set())
    _require(len(v4_mic) == EXPECTED_V4_MIC, "frozen v4 MIC count drifted")
    _require(len(v5_mic) == EXPECTED_V5_MIC, "rejected v5 MIC count drifted")
    _require(len(v6_mic) == EXPECTED_V6_MIC, "v6 MIC conservation target failed")
    _require(v5_mic <= v6_mic, "v6 dropped a MIC retained by v5")
    _require(v6_mic <= v4_mic, "v6 introduced a MIC absent from v4")

    v5_hem_reject_rows = {
        key for key, reason in rejects["v5"].items() if _is_hemolysis_reject(reason)
    }
    endpoint_reject_rows, endpoint_reject_reasons, discarded_candidates = _endpoint_rejects(
        roots["v6"]
    )
    _require(
        v5_hem_reject_rows == endpoint_reject_rows,
        "v6 endpoint-reject rows differ from v5 hemolysis-driven rejects",
    )
    _require(
        len(endpoint_reject_rows) == EXPECTED_ENDPOINT_REJECT_ROWS,
        "v6 endpoint-reject row count drifted",
    )

    recovered = v6_mic - v5_mic
    expected_recovered = {
        fingerprint
        for fingerprint, (row_key, _) in mic_metadata["v4"].items()
        if row_key in v5_hem_reject_rows
    }
    _require(
        recovered == expected_recovered, "v6 MIC recovery differs from endpoint-isolated v4 MICs"
    )
    recovered_rows = {mic_metadata["v4"][item][0] for item in recovered}
    recovered_sequences = {mic_metadata["v4"][item][1] for item in recovered}
    _require(len(recovered) == EXPECTED_RECOVERED_MIC, "recovered MIC count drifted")
    _require(len(recovered_rows) == EXPECTED_RECOVERED_ROWS, "recovered MIC row count drifted")
    _require(
        len(recovered_sequences) == EXPECTED_RECOVERED_SEQUENCES,
        "recovered MIC sequence count drifted",
    )

    missing_from_v6 = v4_mic - v6_mic
    new_global_rows = {
        key
        for key, reason in rejects["v6"].items()
        if _is_global_reject(reason) and key not in rejects["v4"]
    }
    missing_rows = {mic_metadata["v4"][item][0] for item in missing_from_v6}
    _require(
        missing_rows == new_global_rows,
        "v4 MICs missing from v6 are not exactly new global rejects",
    )
    _require(len(new_global_rows) == EXPECTED_NEW_CHEMISTRY_ROWS, "new chemistry row count drifted")
    _require(len(missing_from_v6) == EXPECTED_NEW_CHEMISTRY_MIC, "new chemistry MIC count drifted")
    _require(
        all(
            rejects["v6"][key] == "chemistry quarantine: oxidation-state-specific peptide"
            for key in new_global_rows
        ),
        "v6 introduced an unexpected global chemistry exclusion",
    )

    v5_global = {item for item in rejects["v5"].items() if _is_global_reject(item[1])}
    v6_global = {item for item in rejects["v6"].items() if _is_global_reject(item[1])}
    _require(v5_global == v6_global, "v5/v6 global sequence or chemistry rejects differ")
    for endpoint in ("hc50", "hemolysis_percent"):
        _require(
            assays["v5"].get(endpoint, set()) == assays["v6"].get(endpoint, set()),
            f"v5/v6 {endpoint} fingerprints differ",
        )

    v6_summary = summaries["v6"]
    _require(v6_summary.get("schema_version") == 2, "v6 normalized schema must be 2")
    _require(v6_summary.get("source_rows") == EXPECTED_SOURCE_ROWS, "v6 source row count drifted")
    _require(
        v6_summary.get("source_rows_expected") == EXPECTED_SOURCE_ROWS,
        "v6 source row oracle drifted",
    )
    _require(v6_summary.get("unique_sequences") == 1_113, "v6 unique sequence count drifted")
    _require(v6_summary.get("assay_observations") == 6_092, "v6 assay count drifted")
    _require(v6_summary.get("rejected_rows") == 3_948, "v6 rejected row count drifted")
    _require(v6_summary.get("accepted_source_rows") == 1_136, "v6 accepted row count drifted")
    _require(
        v6_summary.get("endpoint_counts") == EXPECTED_V6_ENDPOINTS, "v6 endpoint counts drifted"
    )
    _require(
        v6_summary.get("endpoint_rejects") == EXPECTED_ENDPOINT_REJECT_ROWS,
        "v6 endpoint rejection count drifted",
    )
    labels = dict(sorted(_mic_labels(roots["v6"]).items()))
    _require(labels == EXPECTED_V6_LABELS, "v6 MIC label distribution drifted")

    accepted_endpoint_reject_rows = {
        row for row in endpoint_reject_rows if "mic" in endpoints_by_row["v6"].get(row, set())
    }
    no_endpoint_rows = endpoint_reject_rows - set(endpoints_by_row["v6"])
    _require(
        accepted_endpoint_reject_rows == recovered_rows,
        "endpoint-rejected rows retaining MIC differ from recovered rows",
    )
    _require(len(no_endpoint_rows) == 22, "endpoint-rejected rows without retained assays drifted")

    input_hashes = {
        version: {
            name: _sha256(root / name)
            for name in (
                "assays.jsonl",
                "rejects.jsonl",
                "sequences.jsonl",
                "summary.json",
            )
        }
        for version, root in roots.items()
    }
    input_hashes["v6"]["endpoint_rejects.jsonl"] = _sha256(roots["v6"] / "endpoint_rejects.jsonl")
    return {
        "schema_version": 1,
        "status": "passed",
        "source_rows": EXPECTED_SOURCE_ROWS,
        "versions": {
            version: {
                "parser_id": summaries[version].get("parser_id"),
                "unique_sequences": summaries[version].get("unique_sequences"),
                "assay_observations": summaries[version].get("assay_observations"),
                "rejected_rows": summaries[version].get("rejected_rows"),
                "endpoint_counts": {
                    endpoint: len(values) for endpoint, values in sorted(assays[version].items())
                },
            }
            for version in ("v4", "v5", "v6")
        },
        "mic_conservation": {
            "recovered_observations": len(recovered),
            "recovered_source_rows": len(recovered_rows),
            "recovered_unique_sequences": len(recovered_sequences),
            "intentional_chemistry_excluded_observations": len(missing_from_v6),
            "intentional_chemistry_excluded_rows": len(new_global_rows),
            "labels_at_16_um": labels,
        },
        "endpoint_rejections": {
            "events": len(endpoint_reject_rows),
            "accepted_mic_rows": len(accepted_endpoint_reject_rows),
            "no_retained_assay_rows": len(no_endpoint_rows),
            "discarded_candidate_observations": discarded_candidates,
            "reason_counts": dict(sorted(endpoint_reject_reasons.items())),
        },
        "invariants": {
            "v5_mic_subset_v6": True,
            "v6_mic_subset_v4": True,
            "v4_minus_v6_is_new_global_chemistry": True,
            "v5_v6_hemolysis_fingerprints_equal": True,
            "v5_v6_global_rejects_equal": True,
            "v5_hemolysis_reject_rows_equal_v6_endpoint_reject_rows": True,
        },
        "input_hashes": input_hashes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-normalized", type=Path, required=True)
    parser.add_argument("--v5-normalized", type=Path, required=True)
    parser.add_argument("--v6-normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite conservation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = verify_conservation(
        v4_root=args.v4_normalized,
        v5_root=args.v5_normalized,
        v6_root=args.v6_normalized,
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

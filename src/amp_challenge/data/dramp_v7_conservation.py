"""Verify parser-v7's fail-closed hemolysis corrections against parser v6."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .dramp_conservation import (
    EXPECTED_SOURCE_ROWS,
    EXPECTED_V6_ENDPOINTS,
    EXPECTED_V6_LABELS,
    RowKey,
    _assays,
    _mic_labels,
    _read_jsonl,
    _row_key,
    _sha256,
    _summary,
)

EXPECTED_V7_ENDPOINTS = {"hc50": 37, "hemolysis_percent": 139, "mic": 5_903}
EXPECTED_V7_ASSAYS = 6_079
EXPECTED_V7_ENDPOINT_REJECTS = 180
EXPECTED_REMOVED_UNCERTAINTY_OBSERVATIONS = 13
EXPECTED_NEW_ENDPOINT_REJECTS = 87
EXPECTED_NEW_UNCERTAINTY_REJECTS = 33
EXPECTED_NEW_ALIAS_REJECTS = 54
EXPECTED_ALL_UNCERTAINTY_REJECTS = 43
EXPECTED_V7_REASON_COUNTS = {
    "missing_blood_target": 17,
    "unsupported_half_max_alias": 60,
    "unsupported_hemolysis_endpoint_alias": 54,
    "unsupported_or_ambiguous_concentration_specific_hemolysis": 6,
    "unsupported_uncertainty_notation": 43,
}

_UNCERTAINTY_MARKER = re.compile(
    r"(?:[\u00b1\u2213]|\+\s*/\s*[-\u2212]|\bplus\s+(?:or\s+)?minus\b)",
    re.IGNORECASE,
)
_UNSUPPORTED_HEMOLYSIS_ENDPOINT_ALIAS_MARKER = re.compile(
    r"\b(?:"
    r"MHC|RCH|"
    r"(?:LD|IC|ED|EC)\s*[_-]?\s*\d{1,3}|"
    r"lethal\s+concentration|"
    r"HU\s*/\s*mg"
    r")\b",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _has_uncertainty(value: object) -> bool:
    return isinstance(value, str) and _UNCERTAINTY_MARKER.search(value) is not None


def _has_unsupported_endpoint_alias(value: object) -> bool:
    return (
        isinstance(value, str)
        and _UNSUPPORTED_HEMOLYSIS_ENDPOINT_ALIAS_MARKER.search(value) is not None
    )


def _has_zero_discarded_candidates(value: Mapping[str, object]) -> bool:
    discarded = value.get("discarded_candidate_observations")
    return isinstance(discarded, int) and not isinstance(discarded, bool) and discarded == 0


def _is_dramp_hemolysis_field(value: Mapping[str, object]) -> bool:
    return (
        value.get("scope") == "endpoint_field"
        and value.get("endpoint_family") == "hemolysis"
        and value.get("source_field") == "Hemolytic_activity"
    )


def _endpoint_reject_documents(root: Path) -> dict[RowKey, dict[str, object]]:
    result: dict[RowKey, dict[str, object]] = {}
    for row in _read_jsonl(root / "endpoint_rejects.jsonl"):
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("endpoint rejection provenance must be an object")
        key = _row_key(provenance)
        if key in result:
            raise ValueError(f"duplicate endpoint-field rejection in {root}: {key}")
        result[key] = row
    return result


def _observed_endpoint_counts(assays: Mapping[str, set[str]]) -> dict[str, int]:
    return {endpoint: len(values) for endpoint, values in sorted(assays.items())}


def _validate_summary(
    *,
    root: Path,
    summary: Mapping[str, object],
    assays: Mapping[str, set[str]],
    endpoint_rejects: Mapping[RowKey, Mapping[str, object]],
) -> None:
    observed_endpoints = _observed_endpoint_counts(assays)
    observed_assays = sum(observed_endpoints.values())
    observed_reasons = dict(
        sorted(
            Counter(
                str(item.get("reason_code") or "") for item in endpoint_rejects.values()
            ).items()
        )
    )
    _require(summary.get("schema_version") == 2, f"{root}: normalized schema must be 2")
    _require(
        summary.get("endpoint_counts") == observed_endpoints,
        f"{root}: endpoint summary differs from assay rows",
    )
    _require(
        summary.get("assay_observations") == observed_assays,
        f"{root}: assay summary count differs from assay rows",
    )
    _require(
        summary.get("endpoint_rejects") == len(endpoint_rejects),
        f"{root}: endpoint rejection summary count differs from ledger",
    )
    _require(
        summary.get("endpoint_rejection_counts") == observed_reasons,
        f"{root}: endpoint rejection reasons differ from ledger",
    )
    _require(
        summary.get("endpoint_rejects_sha256") == _sha256(root / "endpoint_rejects.jsonl"),
        f"{root}: endpoint rejection checksum differs from summary",
    )


def compare_v6_v7(
    *,
    v6_root: str | Path,
    v7_root: str | Path,
) -> dict[str, object]:
    """Verify that v7 changes only unsupported uncertainty and alias fields."""

    roots = {"v6": Path(v6_root).resolve(), "v7": Path(v7_root).resolve()}
    summaries = {name: _summary(root) for name, root in roots.items()}
    assay_sets: dict[str, dict[str, set[str]]] = {}
    endpoint_rejects: dict[str, dict[RowKey, dict[str, object]]] = {}
    for name, root in roots.items():
        assay_sets[name], _, _ = _assays(root)
        endpoint_rejects[name] = _endpoint_reject_documents(root)
        _validate_summary(
            root=root,
            summary=summaries[name],
            assays=assay_sets[name],
            endpoint_rejects=endpoint_rejects[name],
        )

    _require(
        summaries["v6"].get("parser_id") == "amp_challenge.data.dramp:v6",
        "comparison source is not parser v6",
    )
    _require(
        summaries["v7"].get("parser_id") == "amp_challenge.data.dramp:v7",
        "candidate source is not parser v7",
    )
    for filename in ("sequences.jsonl", "rejects.jsonl"):
        _require(
            (roots["v6"] / filename).read_bytes() == (roots["v7"] / filename).read_bytes(),
            f"v7 unexpectedly changed {filename}",
        )
    for endpoint in ("mic", "hc50"):
        _require(
            assay_sets["v6"].get(endpoint, set()) == assay_sets["v7"].get(endpoint, set()),
            f"v7 unexpectedly changed {endpoint} observations",
        )

    v6_hemolysis = assay_sets["v6"].get("hemolysis_percent", set())
    v7_hemolysis = assay_sets["v7"].get("hemolysis_percent", set())
    _require(v7_hemolysis <= v6_hemolysis, "v7 introduced a hemolysis observation absent from v6")
    removed_fingerprints = v6_hemolysis - v7_hemolysis
    removed_rows = tuple(json.loads(item) for item in sorted(removed_fingerprints))
    _require(
        all(
            isinstance(item, Mapping) and _has_uncertainty(item.get("source_text"))
            for item in removed_rows
        ),
        "v7 removed a hemolysis row without mean-plus/minus uncertainty",
    )
    removed_rows_by_key: dict[RowKey, Mapping[str, object]] = {}
    for item in removed_rows:
        assert isinstance(item, Mapping)
        provenance = item.get("provenance")
        _require(
            isinstance(provenance, Mapping),
            "removed uncertainty observation lacks provenance",
        )
        assert isinstance(provenance, Mapping)
        key = _row_key(provenance)
        _require(
            key not in removed_rows_by_key,
            "removed uncertainty observations do not map one-to-one to source rows",
        )
        removed_rows_by_key[key] = item
    removed_row_keys = set(removed_rows_by_key)

    v6_rejects = endpoint_rejects["v6"]
    v7_rejects = endpoint_rejects["v7"]
    _require(set(v6_rejects) <= set(v7_rejects), "v7 dropped a parser-v6 endpoint rejection")
    new_reject_keys = set(v7_rejects) - set(v6_rejects)
    _require(
        removed_row_keys <= new_reject_keys,
        "removed v6 uncertainty observations did not become v7 endpoint rejections",
    )
    _require(
        all(
            v7_rejects[key].get("source_text") == removed_rows_by_key[key].get("source_text")
            for key in removed_row_keys
        ),
        "removed v6 uncertainty observation was rebound to different source text",
    )
    new_uncertainty_reject_keys = {
        key for key in new_reject_keys if _has_uncertainty(v7_rejects[key].get("source_text"))
    }
    new_alias_reject_keys = new_reject_keys - new_uncertainty_reject_keys
    _require(
        all(
            v7_rejects[key].get("reason_code") == "unsupported_uncertainty_notation"
            and _has_zero_discarded_candidates(v7_rejects[key])
            and _is_dramp_hemolysis_field(v7_rejects[key])
            for key in new_uncertainty_reject_keys
        ),
        "v7 introduced an invalid uncertainty endpoint rejection",
    )
    _require(
        all(
            _has_unsupported_endpoint_alias(v7_rejects[key].get("source_text"))
            and v7_rejects[key].get("reason_code") == "unsupported_hemolysis_endpoint_alias"
            and _has_zero_discarded_candidates(v7_rejects[key])
            and _is_dramp_hemolysis_field(v7_rejects[key])
            for key in new_alias_reject_keys
        ),
        "v7 introduced a new endpoint rejection outside the allowed uncertainty and alias classes",
    )

    uncertainty_rejects = {
        key: value
        for key, value in v7_rejects.items()
        if _has_uncertainty(value.get("source_text"))
    }
    _require(
        all(
            item.get("reason_code") == "unsupported_uncertainty_notation"
            and _has_zero_discarded_candidates(item)
            and _is_dramp_hemolysis_field(item)
            for item in uncertainty_rejects.values()
        ),
        "v7 uncertainty fields are not uniformly quarantined before candidate parsing",
    )
    _require(
        all(
            not _has_uncertainty(json.loads(item).get("source_text"))
            for endpoint, values in assay_sets["v7"].items()
            if endpoint in {"hc50", "hemolysis_percent"}
            for item in values
        ),
        "v7 retains an assay sourced from mean-plus/minus hemolysis text",
    )
    _require(
        all(
            not _has_unsupported_endpoint_alias(json.loads(item).get("source_text"))
            for endpoint, values in assay_sets["v7"].items()
            if endpoint in {"hc50", "hemolysis_percent"}
            for item in values
        ),
        "v7 retains a hemolysis assay sourced from an unsupported endpoint alias",
    )
    for key in set(v6_rejects) & set(v7_rejects):
        if _has_uncertainty(v6_rejects[key].get("source_text")):
            stable_fields = {
                field: value
                for field, value in v6_rejects[key].items()
                if field
                not in {
                    "discarded_candidate_observations",
                    "reason_code",
                    "reason_detail",
                }
            }
            candidate_stable_fields = {
                field: value
                for field, value in v7_rejects[key].items()
                if field
                not in {
                    "discarded_candidate_observations",
                    "reason_code",
                    "reason_detail",
                }
            }
            _require(
                stable_fields == candidate_stable_fields,
                f"v7 rebound existing uncertainty endpoint rejection {key}",
            )
            continue
        _require(
            v6_rejects[key] == v7_rejects[key],
            f"v7 unexpectedly changed endpoint rejection {key}",
        )

    v6_labels = dict(sorted(_mic_labels(roots["v6"]).items()))
    v7_labels = dict(sorted(_mic_labels(roots["v7"]).items()))
    _require(v6_labels == v7_labels, "v7 changed the MIC label distribution")
    return {
        "schema_version": 1,
        "status": "passed",
        "comparison": "parser_v6_to_v7_hemolysis_quarantine",
        "versions": {
            name: {
                "parser_id": summaries[name].get("parser_id"),
                "unique_sequences": summaries[name].get("unique_sequences"),
                "assay_observations": summaries[name].get("assay_observations"),
                "rejected_rows": summaries[name].get("rejected_rows"),
                "accepted_source_rows": summaries[name].get("accepted_source_rows"),
                "endpoint_counts": _observed_endpoint_counts(assay_sets[name]),
                "endpoint_rejects": len(endpoint_rejects[name]),
                "endpoint_rejection_counts": dict(
                    sorted(
                        Counter(
                            str(item.get("reason_code") or "")
                            for item in endpoint_rejects[name].values()
                        ).items()
                    )
                ),
            }
            for name in ("v6", "v7")
        },
        "correction": {
            "removed_hemolysis_observations": len(removed_rows),
            "removed_source_rows": len(removed_row_keys),
            "new_endpoint_rejections": len(new_reject_keys),
            "new_uncertainty_endpoint_rejections": len(new_uncertainty_reject_keys),
            "new_alias_endpoint_rejections": len(new_alias_reject_keys),
            "newly_audited_uncertainty_fields_without_v6_observation": len(
                new_uncertainty_reject_keys - removed_row_keys
            ),
            "all_uncertainty_endpoint_rejections": len(uncertainty_rejects),
            "removed_record_ids": sorted(
                str(item["provenance"].get("record_id") or "")
                for item in removed_rows
                if isinstance(item.get("provenance"), Mapping)
            ),
            "reason_codes": {
                "endpoint_alias": "unsupported_hemolysis_endpoint_alias",
                "uncertainty": "unsupported_uncertainty_notation",
            },
        },
        "mic_conservation": {
            "observations": len(assay_sets["v7"].get("mic", set())),
            "labels_at_16_um": v7_labels,
        },
        "invariants": {
            "sequences_byte_identical": True,
            "row_rejections_byte_identical": True,
            "mic_observations_identical": True,
            "hc50_observations_identical": True,
            "v7_hemolysis_is_strict_v6_subset": bool(removed_rows),
            "removed_rows_are_new_endpoint_rejections": True,
            "all_new_endpoint_rejections_have_allowed_class": True,
            "new_uncertainty_endpoint_rejections_valid": True,
            "new_alias_endpoint_rejections_valid": True,
            "all_uncertainty_fields_quarantined": True,
            "all_unsupported_alias_fields_quarantined": True,
            "non_uncertainty_endpoint_rejections_identical": True,
            "mic_label_distribution_identical": True,
        },
        "input_hashes": {
            name: {
                filename: _sha256(root / filename)
                for filename in (
                    "assays.jsonl",
                    "endpoint_rejects.jsonl",
                    "rejects.jsonl",
                    "sequences.jsonl",
                    "summary.json",
                )
            }
            for name, root in roots.items()
        },
    }


def verify_promoted_v7_conservation(
    *,
    v6_root: str | Path,
    v7_root: str | Path,
) -> dict[str, object]:
    """Apply the exact promoted-DRAMP expectations after relational checks."""

    report = compare_v6_v7(v6_root=v6_root, v7_root=v7_root)
    versions = report["versions"]
    correction = report["correction"]
    mic = report["mic_conservation"]
    if not isinstance(versions, Mapping) or not isinstance(correction, Mapping):
        raise AssertionError("internal v7 conservation report schema failure")
    v6 = versions["v6"]
    v7 = versions["v7"]
    _require(isinstance(v6, Mapping) and isinstance(v7, Mapping), "invalid version summaries")
    _require(v6.get("endpoint_counts") == EXPECTED_V6_ENDPOINTS, "v6 endpoint counts drifted")
    _require(v7.get("endpoint_counts") == EXPECTED_V7_ENDPOINTS, "v7 endpoint counts drifted")
    _require(v7.get("assay_observations") == EXPECTED_V7_ASSAYS, "v7 assay count drifted")
    _require(v7.get("unique_sequences") == 1_113, "v7 unique sequence count drifted")
    _require(v7.get("rejected_rows") == 3_948, "v7 rejected row count drifted")
    _require(v7.get("accepted_source_rows") == 1_136, "v7 accepted row count drifted")
    _require(
        v7.get("endpoint_rejects") == EXPECTED_V7_ENDPOINT_REJECTS,
        "v7 endpoint rejection count drifted",
    )
    _require(
        v7.get("endpoint_rejection_counts") == EXPECTED_V7_REASON_COUNTS,
        "v7 endpoint rejection reasons drifted",
    )
    _require(
        correction.get("removed_hemolysis_observations")
        == EXPECTED_REMOVED_UNCERTAINTY_OBSERVATIONS,
        "v7 removed uncertainty observation count drifted",
    )
    _require(
        correction.get("new_endpoint_rejections") == EXPECTED_NEW_ENDPOINT_REJECTS,
        "v7 new endpoint rejection count drifted",
    )
    _require(
        correction.get("new_uncertainty_endpoint_rejections") == EXPECTED_NEW_UNCERTAINTY_REJECTS,
        "v7 new uncertainty rejection count drifted",
    )
    _require(
        correction.get("new_alias_endpoint_rejections") == EXPECTED_NEW_ALIAS_REJECTS,
        "v7 new alias rejection count drifted",
    )
    _require(
        correction.get("all_uncertainty_endpoint_rejections") == EXPECTED_ALL_UNCERTAINTY_REJECTS,
        "v7 uncertainty rejection count drifted",
    )
    _require(isinstance(mic, Mapping), "invalid MIC conservation summary")
    _require(mic.get("labels_at_16_um") == EXPECTED_V6_LABELS, "v7 MIC labels drifted")
    v7_summary = _summary(Path(v7_root).resolve())
    _require(v7_summary.get("source_rows") == EXPECTED_SOURCE_ROWS, "v7 source rows drifted")
    _require(
        v7_summary.get("source_rows_expected") == EXPECTED_SOURCE_ROWS,
        "v7 source-row oracle drifted",
    )
    _require(
        "mean-plus/minus uncertainty" in str(v7_summary.get("assay_policy") or ""),
        "v7 assay policy does not declare its uncertainty boundary",
    )
    _require(
        "source-specific MHC, RCH, LD, IC, ED, EC" in str(v7_summary.get("assay_policy") or ""),
        "v7 assay policy does not declare its unsupported endpoint-alias boundary",
    )
    report["expectation_profile"] = "promoted_dramp_general_v2_parser_v7"
    return report


def _file_provenance(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    return {"filename": path.name, "sha256": _sha256(path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-normalized", type=Path, required=True)
    parser.add_argument("--v7-normalized", type=Path, required=True)
    parser.add_argument("--v6-run-checksums", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _GIT_SHA1.fullmatch(args.git_commit) is None:
        raise ValueError("git commit must be a full lowercase SHA-1")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite conservation report: {output}")
    report = verify_promoted_v7_conservation(
        v6_root=args.v6_normalized,
        v7_root=args.v7_normalized,
    )
    report["provenance"] = {
        "git_commit": args.git_commit,
        "v6_run_checksums": _file_provenance(
            args.v6_run_checksums.resolve(), name="v6 run checksums"
        ),
        "code_manifest": _file_provenance(args.code_manifest.resolve(), name="code manifest"),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for item in report["provenance"].values():
        if isinstance(item, Mapping) and "sha256" in item:
            _require(_SHA256.fullmatch(str(item["sha256"])) is not None, "invalid provenance hash")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

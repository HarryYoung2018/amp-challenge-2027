"""Reproducible oracle-only assay projection; no fitting or execution authority.

The accepted namespace receipt anchors both committed input twins. Old folds
are discarded. Every oracle union component receives one new OOF fold using
sequence counts alone. Assay values, censoring and exposure context are copied,
never reinterpreted as point labels or merged across organisms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from amp_challenge.data import generator_oracle_namespace_split as namespace
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

RECEIPT_SHA256 = "c430c8704c853a90fd325765251ede189e0edfdb7226922f5234c1d095c09662"
NAMESPACE_CONFIG_SHA256 = "ce7f5d28b6bbd8930fa78a9ff21a062bd04f3478ed5d8f7b8cab8ec38f07bffc"
RECEIPT_RELATIVE = (
    "data/runs/generator-oracle-namespace-split-v2-audits/233340/"
    "independent-verification-233343.json"
)
NAMESPACE_RELATIVE = "data/runs/generator-oracle-namespace-split-v2/233340"
CONFIG_RELATIVE = "configs/data/oracle_modeling_projection_v1.toml"
FOLD_POLICY = "whole_oracle_union_descending_sequence_count_then_id_to_smallest_fold_then_index"
MARKER_ARTIFACT = "oracle_only_modeling_projection_complete_v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ROWS = 10000
CLAIMS = {
    "source_folds_used": False,
    "model_fitting_performed": False,
    "execution_authorized": False,
    "oracle_calls_authorized": False,
    "scientific_evidence_accepted": False,
    "production_input_eligible": False,
}
DATA_FILES = (
    "oracle_sequences.jsonl",
    "oof_union_groups.jsonl",
    "activity_contexts.jsonl",
    "activity_measurements.jsonl",
    "safety_observations.jsonl",
    "feasibility.json",
)
OUTPUT_FILES = (*DATA_FILES, "manifest.json", "SHA256SUMS")
SOURCE_FILES = (
    CONFIG_RELATIVE,
    "configs/data/generator_oracle_namespace_split_v1.toml",
    "src/amp_challenge/data/oracle_modeling_projection.py",
    "src/amp_challenge/data/generator_oracle_namespace_split.py",
    "src/amp_challenge/sequences.py",
    "src/amp_challenge/constants.py",
    "cluster/slurm/build_oracle_modeling_projection_v1.sbatch",
    "pyproject.toml",
    "uv.lock",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class OracleProjectionError(ValueError):
    """Reject inconsistent inputs without publishing an apparent success."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleProjectionError(message)


def _canonical(document: Any) -> bytes:
    return namespace._canonical(document)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _rows(snapshot: namespace.Snapshot, name: str) -> list[dict[str, Any]]:
    return namespace._jsonl(snapshot, name, MAX_ROWS)


def _read_file(
    path: Path, cap: int = MAX_FILE_BYTES, *, immutable: bool = True
) -> namespace.Snapshot:
    parent = namespace._open_root(path.parent)
    try:
        return namespace._snapshot(parent, path.name, cap, immutable=immutable)
    finally:
        os.close(parent)


def _sequence_map(rows: list[dict[str, Any]], expected_namespace: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("sequence_id")
        sequence = row.get("sequence")
        _require(
            row.get("namespace") == expected_namespace
            and type(sid) is str
            and _SHA.fullmatch(sid) is not None
            and type(sequence) is str
            and sequence == canonicalize_sequence(sequence)
            and sid == canonical_sequence_id(sequence),
            "sequence namespace or canonical identity differs",
        )
        for key in ("union_component_id", "homology_component_id"):
            _require(
                type(row.get(key)) is str and _SHA.fullmatch(row[key]) is not None, f"invalid {key}"
            )
        _require(sid not in result, "duplicate namespace sequence")
        result[sid] = row
    _require(bool(result), "namespace sequence inventory is empty")
    return result


def assign_oracle_folds(
    oracle: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Assign the full oracle union once, independently of endpoint labels."""

    groups: dict[str, list[str]] = defaultdict(list)
    for sid, row in oracle.items():
        groups[row["union_component_id"]].append(sid)
    loads = [0] * 5
    assigned: dict[str, int] = {}
    for uid in sorted(groups, key=lambda key: (-len(groups[key]), key)):
        fold = min(range(5), key=lambda index: (loads[index], index))
        loads[fold] += len(groups[uid])
        assigned[uid] = fold
    group_rows = [
        {
            "union_component_id": uid,
            "oof_fold": assigned[uid],
            "sequence_ids": sorted(groups[uid]),
            "sequence_count": len(groups[uid]),
        }
        for uid in sorted(groups)
    ]
    return {sid: assigned[row["union_component_id"]] for sid, row in oracle.items()}, group_rows


def _safety_role(row: dict[str, Any]) -> str:
    tasks = row["eligible_tasks"]
    if row["endpoint"] == "hc50":
        if "human_hc50_interval" in tasks:
            return "human_hc50_interval"
        if "nonhuman_hc50_interval_aux" in tasks:
            return "nonhuman_hc50_auxiliary"
    elif "human_hemolysis_percent_at_dose" in tasks:
        return "human_hemolysis_at_recorded_dose"
    elif "nonhuman_hemolysis_percent_at_dose_aux" in tasks:
        return "nonhuman_hemolysis_at_recorded_dose_auxiliary"
    return "audit_only_ineligible_context"


def project_oracle_data(
    oracle_rows: list[dict[str, Any]],
    generator_rows: list[dict[str, Any]],
    study_rows: list[dict[str, Any]],
    gate1_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
) -> dict[str, bytes]:
    """Pure projection after byte/receipt authentication by the producer."""

    oracle = _sequence_map(oracle_rows, "oracle")
    generator = _sequence_map(generator_rows, "generator")
    _require(not set(oracle) & set(generator), "generator/oracle sequence overlap")
    for key in ("sequence", "union_component_id", "homology_component_id"):
        _require(
            not {r[key] for r in oracle.values()} & {r[key] for r in generator.values()},
            f"generator/oracle {key} overlap",
        )
    study_by_sequence: dict[str, list[str]] = {}
    study_union: dict[str, str] = {}
    homology_union: dict[str, str] = {}
    for row in oracle.values():
        old = homology_union.setdefault(row["homology_component_id"], row["union_component_id"])
        _require(old == row["union_component_id"], "homology component spans oracle unions")
    for row in study_rows:
        sid = row["sequence_id"]
        _require(
            row.get("namespace") == "oracle" and sid in oracle,
            "study row is outside oracle namespace",
        )
        _require(sid not in study_by_sequence, "duplicate oracle study row")
        keys = row["study_keys"]
        _require(
            type(keys) is list
            and len(keys) <= 64
            and all(type(key) is str and key for key in keys)
            and len(keys) == len(set(keys)),
            "invalid study-key inventory",
        )
        study_by_sequence[sid] = sorted(keys)
        for key in keys:
            old = study_union.setdefault(key, oracle[sid]["union_component_id"])
            _require(old == oracle[sid]["union_component_id"], "study key spans oracle unions")
    _require(set(study_by_sequence) == set(oracle), "oracle study membership is incomplete")
    folds, groups = assign_oracle_folds(oracle)
    sequence_rows = [
        {
            "schema_version": 1,
            "namespace": "oracle",
            "sequence_id": sid,
            "sequence": oracle[sid]["sequence"],
            "union_component_id": oracle[sid]["union_component_id"],
            "homology_component_id": oracle[sid]["homology_component_id"],
            "study_keys": study_by_sequence[sid],
            "oof_fold": folds[sid],
        }
        for sid in sorted(oracle)
    ]
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observations: set[str] = set()
    for row in endpoint_rows:
        oid = row["observation_id"]
        _require(oid not in observations, "duplicate source observation identity")
        observations.add(oid)
        _require(
            row["sequence_id"] in oracle or row["sequence_id"] in generator,
            "endpoint sequence is outside accepted namespaces",
        )
        _require(
            row["endpoint"] not in {"hc50", "hemolysis_percent"} or row["sequence_id"] in oracle,
            "safety-bearing generator row is forbidden",
        )
        by_context[row["assay_context_id"]].append(row)
    activity: list[dict[str, Any]] = []
    activity_measurements: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    for row in sorted(gate1_rows, key=lambda item: item["example_id"]):
        sid = row["sequence_id"]
        _require(sid in oracle or sid in generator, "activity sequence is outside namespaces")
        if sid not in oracle:
            continue
        cid = row["assay_context_id"]
        _require(
            cid == row["example_id"] and cid not in seen_contexts,
            "duplicate or inconsistent activity context",
        )
        seen_contexts.add(cid)
        _require(
            type(row["label"]) is int and row["label"] in {0, 1},
            "accepted activity label is not binary",
        )
        for key in ("sequence", "homology_component_id", "union_component_id"):
            _require(row[key] == oracle[sid][key], "activity identity differs from namespace")
        sources = by_context.get(cid, [])
        _require(
            type(row["source_observations"]) is int
            and len(sources) == row["source_observations"] > 0,
            "accepted activity context source count differs",
        )
        for source in sources:
            _require(
                source["sequence_id"] == sid
                and source["endpoint"] == "mic"
                and "bacterial_mic16" in source["eligible_tasks"]
                and type(source["mic16_label"]) is int
                and source["mic16_label"] == row["label"]
                and source["canonical_target"] == row["canonical_target"]
                and source["source_gram"] == row["gram"],
                "accepted activity context is not unanimous and eligible",
            )
            activity_measurements.append(
                {
                    "schema_version": 1,
                    "namespace": "oracle",
                    "oof_fold": folds[sid],
                    "sequence_id": sid,
                    "union_component_id": oracle[sid]["union_component_id"],
                    "activity_context_id": cid,
                    "source_endpoint": source,
                }
            )
        activity.append(
            {
                "schema_version": 1,
                "namespace": "oracle",
                "oof_fold": folds[sid],
                **{
                    key: row[key]
                    for key in (
                        "example_id",
                        "assay_context_id",
                        "sequence_id",
                        "canonical_target",
                        "gram",
                        "label",
                        "source_observations",
                        "union_component_id",
                        "homology_component_id",
                    )
                },
                "label_definition": "accepted_bacterial_mic_at_or_below_16_uM",
            }
        )
    safety = [
        {
            "schema_version": 1,
            "namespace": "oracle",
            "oof_fold": folds[row["sequence_id"]],
            "sequence_id": row["sequence_id"],
            "union_component_id": oracle[row["sequence_id"]]["union_component_id"],
            "analysis_role": _safety_role(row),
            "source_endpoint": row,
        }
        for row in sorted(endpoint_rows, key=lambda item: item["observation_id"])
        if row["sequence_id"] in oracle and row["endpoint"] in {"hc50", "hemolysis_percent"}
    ]
    activity_measurements.sort(key=lambda item: item["source_endpoint"]["observation_id"])
    report = feasibility_report(sequence_rows, groups, activity, activity_measurements, safety)
    return {
        "oracle_sequences.jsonl": _jsonl(sequence_rows),
        "oof_union_groups.jsonl": _jsonl(groups),
        "activity_contexts.jsonl": _jsonl(activity),
        "activity_measurements.jsonl": _jsonl(activity_measurements),
        "safety_observations.jsonl": _jsonl(safety),
        "feasibility.json": _canonical(report),
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def feasibility_report(sequences, groups, activity, measurements, safety) -> dict[str, Any]:
    """Descriptive support counts; no fitting, imputation or scientific gate."""

    endpoints = {}
    activity_ids = {row["sequence_id"] for row in activity}
    for endpoint in ("hc50", "hemolysis_percent"):
        selected = [row for row in safety if row["source_endpoint"]["endpoint"] == endpoint]
        source = [row["source_endpoint"] for row in selected]
        ids = {row["sequence_id"] for row in selected}
        human = [row for row in selected if row["analysis_role"].startswith("human_")]
        endpoints[endpoint] = {
            "observations": len(selected),
            "sequences": len(ids),
            "union_components": len({row["union_component_id"] for row in selected}),
            "activity_paired_sequences": len(ids & activity_ids),
            "human_observations": len(human),
            "human_sequences": len({row["sequence_id"] for row in human}),
            "human_activity_paired_sequences": len(
                {row["sequence_id"] for row in human} & activity_ids
            ),
            "relations": _counts([row["measurement"] for row in source], "relation"),
            "units": _counts([row["measurement"] for row in source], "unit"),
            "blood_organisms": _counts(source, "resolved_blood_organism"),
            "analysis_roles": _counts(selected, "analysis_role"),
            "missing_exposure_rows": sum(row["exposure_concentration"] is None for row in source),
            "distinct_recorded_exposures": len(
                {
                    _canonical(row["exposure_concentration"])
                    for row in source
                    if row["exposure_concentration"] is not None
                }
            ),
            "source_conditions_present_rows": sum(
                bool(row.get("source_conditions")) for row in source
            ),
            "folds": [
                {
                    "oof_fold": fold,
                    "observations": sum(row["oof_fold"] == fold for row in selected),
                    "sequences": len(
                        {row["sequence_id"] for row in selected if row["oof_fold"] == fold}
                    ),
                    "union_components": len(
                        {row["union_component_id"] for row in selected if row["oof_fold"] == fold}
                    ),
                    "human_observations": sum(row["oof_fold"] == fold for row in human),
                }
                for fold in range(5)
            ],
        }
    target_support = []
    for target in sorted({row["canonical_target"] for row in activity}):
        selected = [row for row in activity if row["canonical_target"] == target]
        selected_ids = {row["sequence_id"] for row in selected}
        target_support.append(
            {
                "canonical_target": target,
                "contexts": len(selected),
                "sequences": len(selected_ids),
                "union_components": len({row["union_component_id"] for row in selected}),
                "positive_contexts": sum(row["label"] == 1 for row in selected),
                "negative_contexts": sum(row["label"] == 0 for row in selected),
                "folds_with_contexts": sorted({row["oof_fold"] for row in selected}),
                "human_hc50_paired_sequences": len(
                    selected_ids
                    & {
                        row["sequence_id"]
                        for row in safety
                        if row["analysis_role"] == "human_hc50_interval"
                    }
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact": "oracle_modeling_data_feasibility_v1",
        "claims": CLAIMS,
        "oracle_sequences": len(sequences),
        "oracle_union_components": len(groups),
        "activity_contexts": len(activity),
        "activity_sequences": len(activity_ids),
        "activity_positive": sum(row["label"] == 1 for row in activity),
        "activity_negative": sum(row["label"] == 0 for row in activity),
        "activity_source_observations": len(measurements),
        "activity_targets": _counts(activity, "canonical_target"),
        "activity_target_support": target_support,
        "activity_measurement_relations": _counts(
            [row["source_endpoint"]["measurement"] for row in measurements], "relation"
        ),
        "safety": endpoints,
        "fold_policy": FOLD_POLICY,
        "folds": [
            {
                "oof_fold": fold,
                "sequences": sum(row["oof_fold"] == fold for row in sequences),
                "union_components": sum(row["oof_fold"] == fold for row in groups),
                "activity_contexts": sum(row["oof_fold"] == fold for row in activity),
                "activity_positive": sum(
                    row["oof_fold"] == fold and row["label"] == 1 for row in activity
                ),
                "activity_negative": sum(
                    row["oof_fold"] == fold and row["label"] == 0 for row in activity
                ),
            }
            for fold in range(5)
        ],
        "measurement_policy": "preserve_original_relations_bounds_inclusivity_units_and_context",
        "activity_safety_pairing_scope": "same_sequence_only_not_matched_assay_conditions",
        "safety_endpoints_interchangeable": False,
        "censored_observations_are_point_measurements": False,
    }


def _read_surface(base: Path, twin: dict[str, Any], surface_name: str):
    surface = twin["surfaces"][surface_name]
    folder, artifact = {
        "audit_namespace": ("split", "generator_oracle_namespace_split_complete_v2"),
        "staging": ("staging", "generator_oracle_namespace_staging_complete_v2"),
    }[surface_name]
    snapshots, marker, binding = namespace._read_committed_tree(
        base / folder / str(twin["twin_id"]),
        expected_marker_artifact=artifact,
        expected_files=set(surface["file_bindings"]),
        maximum_file_bytes=MAX_FILE_BYTES,
        maximum_json_depth=16,
        maximum_json_containers=4096,
        maximum_json_string_bytes=65536,
    )
    _require(
        binding.root_dev == surface["root_dev"]
        and binding.root_ino == surface["root_ino"]
        and binding.marker_sha256 == surface["completion_marker_sha256"]
        and marker["files"] == surface["file_bindings"],
        "committed surface differs from accepted namespace receipt",
    )
    return snapshots


def read_projection(output: Path, *, expected_manifest_sha256: str) -> dict[str, bytes]:
    """Read only a committed projection bound to an externally expected manifest."""

    _require(
        type(expected_manifest_sha256) is str
        and _SHA.fullmatch(expected_manifest_sha256) is not None,
        "invalid manifest pin",
    )
    snapshots, marker, _ = namespace._read_committed_tree(
        output,
        expected_marker_artifact=MARKER_ARTIFACT,
        expected_files=set(OUTPUT_FILES),
        maximum_file_bytes=MAX_FILE_BYTES,
        maximum_json_depth=16,
        maximum_json_containers=4096,
        maximum_json_string_bytes=65536,
    )
    _require(
        snapshots["manifest.json"].sha256 == expected_manifest_sha256
        and marker["identity"].get("manifest_sha256") == expected_manifest_sha256,
        "projection manifest differs from external pin",
    )
    manifest = json.loads(snapshots["manifest.json"].payload)
    _require(
        manifest["artifact"] == "oracle_only_modeling_projection_v1"
        and manifest["claims"] == CLAIMS
        and manifest["namespace_receipt_sha256"] == RECEIPT_SHA256
        and marker["identity"]
        == {
            "producer_commit": manifest["producer_commit"],
            "manifest_sha256": expected_manifest_sha256,
            "namespace_receipt_sha256": RECEIPT_SHA256,
        }
        and set(manifest["files"]) == set(DATA_FILES),
        "projection manifest schema differs",
    )
    for name in DATA_FILES:
        _require(
            manifest["files"][name]
            == {
                "sha256": snapshots[name].sha256,
                "bytes": snapshots[name].size,
            },
            f"projection payload differs from manifest: {name}",
        )
    expected_sums = b"".join(
        f"{snapshots[name].sha256}  {name}\n".encode()
        for name in sorted((*DATA_FILES, "manifest.json"))
    )
    _require(snapshots["SHA256SUMS"].payload == expected_sums, "projection checksums differ")
    return {name: image.payload for name, image in snapshots.items()}


def build_projection(
    *,
    repository: Path,
    scratch: Path,
    expected_commit: str,
    expected_config_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Authenticate accepted twins, project both, and publish equal output bytes."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(repository), *args],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )

    _require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit
        and not git("status", "--porcelain", "--untracked-files=all"),
        "producer repository must be the exact clean commit",
    )
    code = {}
    code_payloads = {}
    for relative in SOURCE_FILES:
        committed = git("show", f"{expected_commit}:{relative}")
        image = _read_file(repository / relative, immutable=False)
        _require(image.payload == committed, "producer source differs from committed blob")
        code[relative] = image.sha256
        code_payloads[relative] = image.payload
    _require(code[CONFIG_RELATIVE] == expected_config_sha256, "projection config digest differs")
    cfg = tomllib.loads(code_payloads[CONFIG_RELATIVE].decode())
    _require(
        type(cfg["schema_version"]) is int
        and cfg["schema_version"] == 1
        and cfg["artifact"] == "oracle_only_modeling_projection_v1"
        and cfg["status"] == "non_authorizing_data_preparation_only"
        and cfg["namespace_receipt_sha256"] == RECEIPT_SHA256
        and cfg["namespace_config_sha256"] == NAMESPACE_CONFIG_SHA256
        and type(cfg["fold_count"]) is int
        and cfg["fold_count"] == 5
        and cfg["fold_policy"] == FOLD_POLICY
        and cfg["activity_policy"]
        == "retain_accepted_mic16_contexts_without_relabeling_and_preserve_source_measurements"
        and cfg["safety_policy"]
        == "retain_hc50_and_percent_hemolysis_with_original_censoring_organism_and_exposure"
        and all(cfg[key] is False for key in CLAIMS),
        "projection contract differs",
    )
    ns_cfg = namespace.load_config(
        repository / "configs/data/generator_oracle_namespace_split_v1.toml",
        NAMESPACE_CONFIG_SHA256,
    )
    receipt_image = _read_file(scratch / RECEIPT_RELATIVE)
    _require(
        receipt_image.sha256 == RECEIPT_SHA256
        and receipt_image.size == cfg["namespace_receipt_bytes"],
        "namespace receipt pin differs",
    )
    receipt = json.loads(receipt_image.payload)
    _require(
        receipt["status"] == "passed_non_authorizing_data_preparation_only"
        and all(value is False for value in receipt["claims"].values()),
        "namespace receipt is not accepted non-authorizing data preparation",
    )
    _require(
        code["src/amp_challenge/data/generator_oracle_namespace_split.py"]
        == receipt["identity"]["producer_source_inventory"][
            "src/amp_challenge/data/generator_oracle_namespace_split.py"
        ],
        "namespace reader/publisher source differs from accepted implementation",
    )
    twins = []
    for twin in receipt["twins"]:
        audit = _read_surface(scratch / NAMESPACE_RELATIVE, twin, "audit_namespace")
        staged = _read_surface(scratch / NAMESPACE_RELATIVE, twin, "staging")
        for name, (relative, size) in ns_cfg.inputs.items():
            _require(
                staged[relative].size == size
                and staged[relative].sha256 == ns_cfg.raw["inputs"][name]["sha256"],
                f"staged source pin differs: {name}",
            )
        twins.append(
            project_oracle_data(
                _rows(audit["oracle_corpus.jsonl"], "oracle corpus"),
                _rows(audit["generator_corpus.jsonl"], "generator corpus"),
                _rows(audit["oracle_study_membership.jsonl"], "oracle studies"),
                _rows(staged["gate1/examples.jsonl"], "accepted gate1 contexts"),
                _rows(staged["endpoint/endpoint_context_ledger.jsonl"], "endpoint ledger"),
            )
        )
    _require(len(twins) == 2 and twins[0] == twins[1], "accepted twins do not project identically")
    payloads = twins[0]
    report = json.loads(payloads["feasibility.json"])
    for key, expected in cfg["expected"].items():
        if key.startswith("hc50_") or key.startswith("hemolysis_percent_"):
            endpoint = "hc50" if key.startswith("hc50_") else "hemolysis_percent"
            actual = report["safety"][endpoint][key[len(endpoint) + 1 :]]
        else:
            actual = report[key]
        _require(actual == expected, f"oracle data census differs: {key}")
    manifest = {
        "schema_version": 1,
        "artifact": "oracle_only_modeling_projection_v1",
        "claims": CLAIMS,
        "producer_commit": expected_commit,
        "source_inventory": code,
        "namespace_receipt_sha256": RECEIPT_SHA256,
        "namespace_producer_job": "233340",
        "namespace_audit_job": "233343",
        "namespace_finalizer_job": "233344",
        "input_twins_project_identically": True,
        "files": {
            name: {"sha256": _digest(payload), "bytes": len(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    payloads["manifest.json"] = _canonical(manifest)
    manifest_sha = _digest(payloads["manifest.json"])
    payloads["SHA256SUMS"] = b"".join(
        f"{_digest(payload)}  {name}\n".encode() for name, payload in sorted(payloads.items())
    )
    _require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit
        and not git("status", "--porcelain", "--untracked-files=all"),
        "producer repository changed before publication",
    )
    binding = namespace._publish_claimed_tree(
        output,
        payloads,
        marker_artifact=MARKER_ARTIFACT,
        identity={
            "producer_commit": expected_commit,
            "manifest_sha256": manifest_sha,
            "namespace_receipt_sha256": RECEIPT_SHA256,
        },
        maximum_file_bytes=MAX_FILE_BYTES,
    )
    return {
        "manifest_sha256": manifest_sha,
        "completion_marker_sha256": binding.marker_sha256,
        "root_dev": binding.root_dev,
        "root_ino": binding.root_ino,
        "claims": CLAIMS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_projection(
        repository=args.repository,
        scratch=args.scratch,
        expected_commit=args.expected_commit,
        expected_config_sha256=args.expected_config_sha256,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

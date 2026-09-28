"""Package the frozen union-v1 activity ledger and replay as one sealed bundle.

The scientific work remains in :mod:`union_oracle_ledger` and :mod:`replay`.
This module is the production boundary: it runs those two stages, binds their
code and accepted inputs, verifies the fixed output schema, and writes one
path-free ten-file artifact.  A successful bundle is development evidence
only; it can never make a policy automatically production eligible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from amp_challenge.evaluation.replay import run_replay
from amp_challenge.evaluation.union_oracle_ledger import build_union_oracle_replay_ledger
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_ARTIFACT = "union_gate1_activity_replay_bundle_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIVERSITY_ID = re.compile(r"div70:[0-9a-f]{64}")
_EXPECTED_CANDIDATES = 650
_EXPECTED_EMBEDDING_DIMENSIONS = 320
_EXPECTED_RUNS = 100
_EXPECTED_SELECTIONS = 1000
_EXPECTED_CANDIDATE_IDS_SHA256 = "990ef4b248f7d95a9864fef7da5bc648a5e6468c1adc74ba87c3af38dedb21c9"
_EXPECTED_FOLD_COUNTS = (202, 126, 112, 97, 113)
_EXPECTED_GROUPS = tuple(f"fold-{index}" for index in range(5))
_EXPECTED_POLICIES = ("mixed", "lcb", "mean", "random")
_EXPECTED_SEEDS = (17, 42, 91, 137, 271)
_SELECTION_CONFIG = "configs/acquisition/replay_union_v1_mixed.toml"
_LEDGER_CONFIG = "configs/evaluation/oracle_activity_ledger_union_v1.toml"
_REPLAY_CONFIG = "configs/evaluation/oracle_activity_replay_union_v1.toml"

# This inventory is deliberately explicit.  It includes every local source,
# configuration, and dependency lock that can change the packaged bytes.
CODE_PATHS = (
    "cluster/slurm/audit_union_activity_replay_v1_twins.sbatch",
    "cluster/slurm/run_union_activity_replay_v1_twins.sbatch",
    "cluster/slurm/union_activity_replay_v1_hardened.sh",
    "cluster/slurm/validate_union_activity_replay_v1.sbatch",
    _SELECTION_CONFIG,
    _LEDGER_CONFIG,
    _REPLAY_CONFIG,
    "pyproject.toml",
    "src/amp_challenge/__init__.py",
    "src/amp_challenge/acquisition/__init__.py",
    "src/amp_challenge/acquisition/mixer.py",
    "src/amp_challenge/constants.py",
    "src/amp_challenge/evaluation/__init__.py",
    "src/amp_challenge/evaluation/replay.py",
    "src/amp_challenge/evaluation/union_oracle_ledger.py",
    "src/amp_challenge/evaluation/union_replay_bundle.py",
    "src/amp_challenge/evaluation/verify_union_replay_bundle.py",
    "src/amp_challenge/sequences.py",
    "src/amp_challenge/similarity.py",
    "src/amp_challenge/workflows/__init__.py",
    "src/amp_challenge/workflows/select.py",
    "uv.lock",
)

BUNDLE_FILES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "SHA256SUMS",
    "ledger/candidate_ledger.csv",
    "ledger/diversity_components.jsonl",
    "ledger/ledger_summary.json",
    "manifest.json",
    "replay/runs.csv",
    "replay/selections.csv",
    "replay/summary.json",
)
_TOP_MANIFEST_FILES = tuple(path for path in BUNDLE_FILES if path != "SHA256SUMS")


@dataclass(frozen=True, slots=True)
class UnionReplayBundleExecution:
    """A completed, immutable union-v1 replay publication."""

    output_dir: Path
    manifest_path: Path
    top_manifest_path: Path
    candidate_count: int
    run_count: int
    selection_count: int
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_mode,
        metadata.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _snapshot_regular(path: Path, *, label: str) -> _Snapshot:
    _reject_symlink_chain(path, label=label)
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    fingerprint = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_mode,
        metadata.st_ctime_ns,
    )
    if _fingerprint(path) != fingerprint:
        raise RuntimeError(f"{label} changed while it was read: {path}")
    return _Snapshot(
        path=path.resolve(strict=True),
        payload=payload,
        sha256=digest,
        fingerprint=fingerprint,
    )


def _assert_unchanged(snapshot: _Snapshot, *, label: str) -> None:
    if (
        _fingerprint(snapshot.path) != snapshot.fingerprint
        or _sha256_file(snapshot.path) != snapshot.sha256
    ):
        raise RuntimeError(f"{label} changed during bundle construction: {snapshot.path}")


def _manifest_text(entries: Mapping[str, str]) -> str:
    previous = ""
    lines: list[str] = []
    for logical_path in sorted(entries):
        digest = entries[logical_path]
        logical = PurePosixPath(logical_path)
        if (
            not logical_path
            or logical.is_absolute()
            or ".." in logical.parts
            or _SHA256.fullmatch(digest) is None
            or (previous and logical_path <= previous)
        ):
            raise ValueError("unsafe checksum-manifest entry")
        lines.append(f"{digest}  {logical_path}\n")
        previous = logical_path
    if not lines:
        raise ValueError("checksum manifest cannot be empty")
    return "".join(lines)


def _write_exclusive(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_exact_tree(root: Path, *, expected_files: Sequence[str]) -> None:
    observed_files = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    observed_dirs = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    )
    expected_dirs = sorted(
        {PurePosixPath(path).parent.as_posix() for path in expected_files} - {"."}
    )
    if observed_files != sorted(expected_files) or observed_dirs != expected_dirs:
        raise ValueError(
            f"bundle inventory changed: files={observed_files!r}, directories={observed_dirs!r}"
        )
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"bundle contains an unsafe entry: {path}")


def _input_paths(
    gate1_twin: Path,
    gate1_independent_receipt: Path,
    embedding_twin: Path,
    embedding_independent_receipt: Path,
) -> Mapping[str, Path]:
    return {
        "embeddings/SHA256SUMS": embedding_twin / "SHA256SUMS",
        "embeddings/embedding_index.csv": embedding_twin / "embeddings" / "embedding_index.csv",
        "embeddings/embedding_manifest.json": embedding_twin
        / "embeddings"
        / "embedding_manifest.json",
        "embeddings/embeddings.npy": embedding_twin / "embeddings" / "embeddings.npy",
        "embeddings/independent-receipt.json": embedding_independent_receipt,
        "embeddings/semantic-SHA256SUMS": embedding_twin / "embeddings" / "SHA256SUMS",
        "gate1/SHA256SUMS": gate1_twin / "SHA256SUMS",
        "gate1/independent-receipt.json": gate1_independent_receipt,
        "gate1/oof_predictions.csv": gate1_twin / "gate1" / "oof_predictions.csv",
        "gate1/semantic-SHA256SUMS": gate1_twin / "gate1" / "SHA256SUMS",
    }


def _validate_config_locations(
    repository_root: Path,
    ledger_config: Path,
    replay_config: Path,
    *,
    snapshots: Mapping[str, _Snapshot],
) -> None:
    expected_ledger = (
        repository_root / "configs/evaluation/oracle_activity_ledger_union_v1.toml"
    ).resolve(strict=True)
    expected_replay = (
        repository_root / "configs/evaluation/oracle_activity_replay_union_v1.toml"
    ).resolve(strict=True)
    if ledger_config.resolve(strict=True) != expected_ledger:
        raise ValueError("ledger config is not the frozen repository union-v1 config")
    if replay_config.resolve(strict=True) != expected_replay:
        raise ValueError("replay config is not the frozen repository union-v1 config")
    try:
        ledger_document = tomllib.loads(snapshots[_LEDGER_CONFIG].payload.decode("utf-8"))
        replay_document = tomllib.loads(snapshots[_REPLAY_CONFIG].payload.decode("utf-8"))
        selection = tomllib.loads(snapshots[_SELECTION_CONFIG].payload.decode("utf-8"))
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("frozen union-v1 configuration snapshot is invalid") from error
    if not all(isinstance(value, dict) for value in (ledger_document, replay_document, selection)):
        raise ValueError("frozen union-v1 configuration snapshot must contain TOML tables")
    selection_relative = replay_document.get("selection_config")
    if not isinstance(selection_relative, str):
        raise ValueError("replay selection_config is missing")
    selection_path = (replay_config.parent / selection_relative).resolve(strict=True)
    expected_selection = (
        repository_root / "configs/acquisition/replay_union_v1_mixed.toml"
    ).resolve(strict=True)
    if selection_path != expected_selection:
        raise ValueError("replay does not select the frozen union-v1 mixed policy")
    if (
        selection.get("batch_size") != 10
        or selection.get("max_per_cluster") != 2
        or selection.get("strict_cluster_cap") is not True
        or "max_per_start" in selection
        or "rollouts_per_start" in selection
    ):
        raise ValueError("union-v1 replay must be non-start-aware with a strict cluster cap of two")


def _validate_ledger(path: Path) -> tuple[Mapping[str, tuple[str, int]], int]:
    fold_counts: Counter[int] = Counter()
    cluster_by_sequence_id: dict[str, tuple[str, int]] = {}
    folds_by_homology: dict[str, int] = {}
    folds_by_union: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if "esm_source_row_index" not in fields or "embedding_source_row_index" in fields:
            raise ValueError("candidate ledger ESM source-row audit field changed")
        embedding_fields = tuple(name for name in fields if name.startswith("embedding_"))
        expected_embeddings = tuple(
            f"embedding_{index:03d}" for index in range(_EXPECTED_EMBEDDING_DIMENSIONS)
        )
        if embedding_fields != expected_embeddings:
            raise ValueError("candidate ledger does not contain exactly the frozen ESM features")
        required = {
            "sequence_id",
            "sequence",
            "cluster_id",
            "diversity_cluster_id_70",
            "homology_component_id",
            "union_component_id",
            "eligible",
            "replay_round",
            "prediction_scope",
            "prediction_fold",
            "outcome_fold",
        }
        if not required.issubset(fields):
            raise ValueError("candidate ledger lacks frozen provenance columns")
        rows = list(reader)
        if len(rows) != _EXPECTED_CANDIDATES:
            raise ValueError("candidate ledger does not contain the frozen 650-sequence cohort")
        for row in rows:
            try:
                fold = int(row["prediction_fold"])
                outcome_fold = int(row["outcome_fold"])
            except (TypeError, ValueError) as error:
                raise ValueError("candidate ledger has invalid fold provenance") from error
            try:
                sequence = canonicalize_sequence(row["sequence"])
            except (TypeError, ValueError) as error:
                raise ValueError("candidate ledger contains a noncanonical sequence") from error
            sequence_id = row["sequence_id"]
            homology_id = row["homology_component_id"]
            union_id = row["union_component_id"]
            if (
                fold not in range(5)
                or outcome_fold != fold
                or row["replay_round"] != f"fold-{fold}"
                or row["prediction_scope"] != "out_of_fold"
                or row["eligible"] != "true"
                or row["cluster_id"] != row["diversity_cluster_id_70"]
                or _DIVERSITY_ID.fullmatch(row["cluster_id"]) is None
                or row["cluster_id"] in {homology_id, union_id}
                or sequence != row["sequence"]
                or sequence_id != canonical_sequence_id(sequence)
                or _SHA256.fullmatch(homology_id) is None
                or _SHA256.fullmatch(union_id) is None
            ):
                raise ValueError("candidate ledger violates frozen replay provenance")
            if sequence_id in cluster_by_sequence_id:
                raise ValueError("candidate ledger duplicates a sequence identity")
            for component, mapping, label in (
                (homology_id, folds_by_homology, "homology"),
                (union_id, folds_by_union, "union"),
            ):
                prior = mapping.setdefault(component, fold)
                if prior != fold:
                    raise ValueError(f"candidate ledger {label} component crosses folds")
            numeric_fields = (
                "novelty",
                "mean_broad_spectrum",
                "mean_gram_positive",
                "mean_gram_negative",
                "std_broad_spectrum",
                "std_gram_positive",
                "std_gram_negative",
                "outcome_broad_spectrum",
                "outcome_gram_positive",
                "outcome_gram_negative",
                *expected_embeddings,
            )
            try:
                numeric = {name: float(row[name]) for name in numeric_fields}
                source_row = int(row["esm_source_row_index"])
                support = [
                    int(row[f"n_{objective}"])
                    for objective in ("broad_spectrum", "gram_positive", "gram_negative")
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("candidate ledger contains invalid numerical fields") from error
            if (
                any(not math.isfinite(value) for value in numeric.values())
                or not 0 <= source_row < 952
                or any(value <= 0 for value in support)
                or not 0 <= numeric["novelty"] <= 1
                or any(
                    not 0 <= numeric[f"mean_{objective}"] <= 1
                    or not 0 <= numeric[f"outcome_{objective}"] <= 1
                    or not 0 <= numeric[f"std_{objective}"] <= 0.5
                    for objective in ("broad_spectrum", "gram_positive", "gram_negative")
                )
            ):
                raise ValueError("candidate ledger numerical contract changed")
            cluster_by_sequence_id[sequence_id] = (row["cluster_id"], fold)
            fold_counts[fold] += 1
        if tuple(fold_counts[index] for index in range(5)) != _EXPECTED_FOLD_COUNTS:
            raise ValueError("candidate ledger per-fold census changed")
        observed_groups = {f"fold-{fold}" for fold in fold_counts if fold_counts[fold]}
        if observed_groups != set(_EXPECTED_GROUPS):
            raise ValueError("candidate ledger replay-round groups changed")
        candidate_digest = hashlib.sha256(
            "".join(f"{sequence_id}\n" for sequence_id in sorted(cluster_by_sequence_id)).encode(
                "ascii"
            )
        ).hexdigest()
        if candidate_digest != _EXPECTED_CANDIDATE_IDS_SHA256:
            raise ValueError("candidate ledger sorted sequence-ID digest changed")
    return cluster_by_sequence_id, len(set(cluster_by_sequence_id.values()))


def _validate_diversity_components(
    path: Path,
    *,
    cluster_by_sequence_id: Mapping[str, tuple[str, int]],
) -> tuple[int, int]:
    rebuilt: dict[str, tuple[str, int]] = {}
    fold_by_cluster: dict[str, int] = {}
    component_count = 0
    largest = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or "\r" in line:
                raise ValueError("diversity ledger is not canonical LF-terminated JSONL")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid diversity JSONL row {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError("diversity JSONL rows must be objects")
            cluster = row.get("diversity_cluster_id_70")
            fold = row.get("prediction_fold")
            sequence_ids = row.get("sequence_ids")
            count = row.get("sequence_count")
            if (
                not isinstance(cluster, str)
                or _DIVERSITY_ID.fullmatch(cluster) is None
                or not isinstance(fold, int)
                or fold not in range(5)
                or row.get("replay_round") != f"fold-{fold}"
                or row.get("identity_algorithm") != "global_alignment_identity_single_link_v1"
                or row.get("identity_threshold") != 0.70
                or not isinstance(sequence_ids, list)
                or not sequence_ids
                or sequence_ids != sorted(sequence_ids)
                or len(sequence_ids) != len(set(sequence_ids))
                or count != len(sequence_ids)
            ):
                raise ValueError("diversity JSONL row violates the frozen clustering contract")
            prior_fold = fold_by_cluster.setdefault(cluster, fold)
            if prior_fold != fold:
                raise ValueError("one diversity component appears in multiple replay folds")
            for sequence_id in sequence_ids:
                if not isinstance(sequence_id, str) or _SHA256.fullmatch(sequence_id) is None:
                    raise ValueError("diversity JSONL contains an invalid sequence identity")
                if sequence_id in rebuilt:
                    raise ValueError("diversity JSONL assigns a sequence more than once")
                rebuilt[sequence_id] = (cluster, fold)
            component_count += 1
            largest = max(largest, len(sequence_ids))
    if rebuilt != dict(cluster_by_sequence_id):
        raise ValueError("diversity JSONL and candidate ledger assignments differ")
    return component_count, largest


def _validate_stage_contracts(
    ledger_dir: Path,
    replay_dir: Path,
    *,
    ledger_config_sha256: str,
    replay_config_sha256: str,
    selection_config_sha256: str,
    frozen_inputs: Mapping[str, _Snapshot],
) -> None:
    cluster_map, _ = _validate_ledger(ledger_dir / "candidate_ledger.csv")
    components, largest = _validate_diversity_components(
        ledger_dir / "diversity_components.jsonl",
        cluster_by_sequence_id=cluster_map,
    )
    ledger_summary = _load_json_object(ledger_dir / "ledger_summary.json", label="ledger summary")
    input_hashes = ledger_summary.get("input_sha256")
    diversity = ledger_summary.get("diversity_clustering")
    artifacts = ledger_summary.get("artifacts")
    expected_input_hashes = {
        "gate1_oof_predictions": frozen_inputs["gate1/oof_predictions.csv"].sha256,
        "gate1_independent_receipt": frozen_inputs["gate1/independent-receipt.json"].sha256,
        "esm2_embedding_index": frozen_inputs["embeddings/embedding_index.csv"].sha256,
        "esm2_embedding_matrix": frozen_inputs["embeddings/embeddings.npy"].sha256,
        "esm2_independent_receipt": frozen_inputs["embeddings/independent-receipt.json"].sha256,
    }
    if (
        ledger_summary.get("artifact") != "union_gate1_activity_replay_ledger_v1"
        or ledger_summary.get("status") != "built_for_union_v1_development_replay"
        or ledger_summary.get("config_sha256") != ledger_config_sha256
        or input_hashes != expected_input_hashes
        or ledger_summary.get("input_examples") != 2492
        or ledger_summary.get("input_sequences") != 952
        or ledger_summary.get("candidate_sequences") != _EXPECTED_CANDIDATES
        or ledger_summary.get("candidate_observed_contexts") != 1967
        or ledger_summary.get("candidate_source_observations") != 2062
        or ledger_summary.get("candidates_by_fold")
        != {str(index): count for index, count in enumerate(_EXPECTED_FOLD_COUNTS)}
        or not isinstance(diversity, dict)
        or diversity.get("algorithm") != "global_alignment_identity_single_link_v1"
        or diversity.get("identity_threshold") != 0.70
        or diversity.get("components") != components
        or diversity.get("largest_component") != largest
        or diversity.get("candidate_cluster_field") != "cluster_id"
        or diversity.get("upstream_split_components_are_not_diversity_clusters") is not True
        or not isinstance(artifacts, dict)
        or artifacts.get("candidate_ledger.csv")
        != _sha256_file(ledger_dir / "candidate_ledger.csv")
        or artifacts.get("diversity_components.jsonl")
        != _sha256_file(ledger_dir / "diversity_components.jsonl")
    ):
        raise ValueError("ledger summary violates the frozen union-v1 contract")

    replay_summary = _load_json_object(replay_dir / "summary.json", label="replay summary")
    if (
        replay_summary.get("schema_version") != 1
        or replay_summary.get("ledger_sha256") != _sha256_file(ledger_dir / "candidate_ledger.csv")
        or replay_summary.get("replay_config_sha256") != replay_config_sha256
        or replay_summary.get("selection_config_sha256") != selection_config_sha256
        or replay_summary.get("objectives") != ["broad_spectrum", "gram_positive", "gram_negative"]
        or tuple(replay_summary.get("policies", ())) != _EXPECTED_POLICIES
        or tuple(replay_summary.get("seeds", ())) != _EXPECTED_SEEDS
        or tuple(replay_summary.get("groups", ())) != _EXPECTED_GROUPS
        or replay_summary.get("run_count") != _EXPECTED_RUNS
        or replay_summary.get("selection_count") != _EXPECTED_SELECTIONS
        or replay_summary.get("runs_sha256") != _sha256_file(replay_dir / "runs.csv")
        or replay_summary.get("selections_sha256") != _sha256_file(replay_dir / "selections.csv")
    ):
        raise ValueError("replay summary violates the frozen union-v1 contract")


def _seal_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444, follow_symlinks=False)
    for path in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        os.chmod(path, 0o555, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)


def build_union_replay_bundle(
    gate1_twin: str | Path,
    gate1_independent_receipt: str | Path,
    embedding_twin: str | Path,
    embedding_independent_receipt: str | Path,
    *,
    ledger_config_path: str | Path,
    replay_config_path: str | Path,
    repository_root: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> UnionReplayBundleExecution:
    """Run and seal the exact frozen union-v1 development replay."""

    if _COMMIT.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase Git commit")
    repository = Path(repository_root).resolve(strict=True)
    if not repository.is_dir():
        raise ValueError("repository_root must be a directory")
    gate_root = Path(gate1_twin)
    gate_receipt = Path(gate1_independent_receipt)
    embedding_root = Path(embedding_twin)
    embedding_receipt = Path(embedding_independent_receipt)
    ledger_config = Path(ledger_config_path)
    replay_config = Path(replay_config_path)
    output = Path(output_dir)
    _reject_symlink_chain(output, label="bundle output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace bundle output: {output}")

    code_snapshots = {
        logical: _snapshot_regular(repository / logical, label=f"code input {logical}")
        for logical in CODE_PATHS
    }
    _validate_config_locations(
        repository,
        ledger_config,
        replay_config,
        snapshots=code_snapshots,
    )
    input_snapshots = {
        logical: _snapshot_regular(path, label=f"frozen input {logical}")
        for logical, path in _input_paths(
            gate_root,
            gate_receipt,
            embedding_root,
            embedding_receipt,
        ).items()
    }

    output.mkdir(parents=True, mode=0o700)
    ledger_dir = output / "ledger"
    replay_dir = output / "replay"
    ledger_execution = build_union_oracle_replay_ledger(
        gate_root,
        gate_receipt,
        embedding_root,
        embedding_receipt,
        config_path=ledger_config,
        output_dir=ledger_dir,
    )
    ledger_snapshot = _snapshot_regular(
        ledger_execution.ledger_path,
        label="built candidate ledger",
    )
    replay_execution = run_replay(
        ledger_execution.ledger_path,
        config_path=replay_config,
        output_dir=replay_dir,
    )
    _assert_unchanged(ledger_snapshot, label="candidate ledger")
    if ledger_execution.candidate_count != _EXPECTED_CANDIDATES:
        raise ValueError("ledger builder did not return the frozen candidate census")
    if (
        replay_execution.run_count != _EXPECTED_RUNS
        or replay_execution.selection_count != _EXPECTED_SELECTIONS
    ):
        raise ValueError("replay did not return the frozen run/selection census")
    if sorted(path.name for path in ledger_dir.iterdir()) != [
        "candidate_ledger.csv",
        "diversity_components.jsonl",
        "ledger_summary.json",
    ]:
        raise ValueError("ledger builder emitted an unexpected artifact inventory")
    if sorted(path.name for path in replay_dir.iterdir()) != [
        "runs.csv",
        "selections.csv",
        "summary.json",
    ]:
        raise ValueError("replay emitted an unexpected artifact inventory")
    _validate_stage_contracts(
        ledger_dir,
        replay_dir,
        ledger_config_sha256=code_snapshots[_LEDGER_CONFIG].sha256,
        replay_config_sha256=code_snapshots[_REPLAY_CONFIG].sha256,
        selection_config_sha256=code_snapshots[_SELECTION_CONFIG].sha256,
        frozen_inputs=input_snapshots,
    )
    _assert_unchanged(ledger_snapshot, label="candidate ledger")

    _write_exclusive(
        output / "CODE_SHA256SUMS",
        _manifest_text({name: snapshot.sha256 for name, snapshot in code_snapshots.items()}),
    )
    _write_exclusive(
        output / "FROZEN_INPUT_SHA256SUMS",
        _manifest_text({name: snapshot.sha256 for name, snapshot in input_snapshots.items()}),
    )
    manifest: dict[str, object] = {
        "artifact": _ARTIFACT,
        "automatic_production_eligible": False,
        "candidate_count": _EXPECTED_CANDIDATES,
        "code_sha256sums_sha256": _sha256_file(output / "CODE_SHA256SUMS"),
        "decision_scope": "development_activity_acquisition_replay_only",
        "embedding_dimensions": _EXPECTED_EMBEDDING_DIMENSIONS,
        "frozen_input_sha256sums_sha256": _sha256_file(output / "FROZEN_INPUT_SHA256SUMS"),
        "git_commit": git_commit,
        "manual_review_required": True,
        "max_per_diversity_cluster": 2,
        "replay": {
            "run_count": _EXPECTED_RUNS,
            "selection_count": _EXPECTED_SELECTIONS,
            "summary_sha256": _sha256_file(replay_execution.summary_path),
        },
        "schema_version": 1,
        "status": "completed_not_automatically_promoted",
    }
    manifest_path = output / "manifest.json"
    _write_exclusive(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    top_manifest_path = output / "SHA256SUMS"
    _write_exclusive(
        top_manifest_path,
        _manifest_text({name: _sha256_file(output / name) for name in _TOP_MANIFEST_FILES}),
    )
    _require_exact_tree(output, expected_files=BUNDLE_FILES)
    for snapshot in (*code_snapshots.values(), *input_snapshots.values()):
        _assert_unchanged(snapshot, label="bundle dependency")
    _assert_unchanged(ledger_snapshot, label="candidate ledger")
    _seal_tree(output)
    _require_exact_tree(output, expected_files=BUNDLE_FILES)
    return UnionReplayBundleExecution(
        output_dir=output.resolve(strict=True),
        manifest_path=manifest_path.resolve(strict=True),
        top_manifest_path=top_manifest_path.resolve(strict=True),
        candidate_count=ledger_execution.candidate_count,
        run_count=replay_execution.run_count,
        selection_count=replay_execution.selection_count,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-twin", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--embedding-twin", type=Path, required=True)
    parser.add_argument("--embedding-independent-receipt", type=Path, required=True)
    parser.add_argument("--ledger-config", type=Path, required=True)
    parser.add_argument("--replay-config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = build_union_replay_bundle(
            args.gate1_twin,
            args.gate1_independent_receipt,
            args.embedding_twin,
            args.embedding_independent_receipt,
            ledger_config_path=args.ledger_config,
            replay_config_path=args.replay_config,
            repository_root=args.repository_root,
            git_commit=args.git_commit,
            output_dir=args.output_dir,
        )
    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"AMP union replay bundle error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "automatic_production_eligible": False,
                "candidates": execution.candidate_count,
                "runs": execution.run_count,
                "selections": execution.selection_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

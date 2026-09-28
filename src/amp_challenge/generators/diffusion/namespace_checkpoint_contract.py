"""Input and numerical contract for generator-namespace initialization assets.

This is an additive research configuration. It does not change acceptance of
the older native v0/v1 corpora, samplers, pilots, or namespace receipts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import stat
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.data.generator_oracle_namespace_split import _read_committed_tree
from amp_challenge.generators.diffusion.data import LengthPrior, TrainingDistribution, TrainingRow
from amp_challenge.generators.diffusion.model import NativeDenoiserConfig
from amp_challenge.sequences import canonical_sequence_id

TRIPLES = tuple("".join(values) for values in itertools.combinations("01234", 3))
GENERATOR_FILES = {
    "component_assignments.jsonl",
    "sequence_ids.jsonl",
    "corpus.jsonl",
    "endpoint_availability.jsonl",
    "study_membership.jsonl",
    "downstream_fold_triples.jsonl",
    "summary.json",
    "manifest.json",
    "SHA256SUMS",
}
CORPUS_FIELDS = {
    "schema_version",
    "namespace",
    "generator_fold",
    "sequence_id",
    "sequence",
    "union_component_id",
    "homology_component_id",
}
ARTIFACT_FILES = {
    "config.toml",
    "CODE_SHA256SUMS",
    "training_projection.jsonl",
    "training_trace.jsonl",
    "checkpoint.safetensors",
    "heldout_losses.jsonl",
    "samples.jsonl",
    "metrics.json",
    "manifest.json",
    "COMPLETE.json",
}


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_regular(path: Path, maximum_bytes: int = 16_777_216) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            "input is not a single-link regular file",
        )
        require(0 < before.st_size <= maximum_bytes, "input exceeds byte bound")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
        entry = path.stat(follow_symlinks=False)

        def fingerprint(value: os.stat_result) -> tuple[int, ...]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

        require(
            fingerprint(before) == fingerprint(after) == fingerprint(entry),
            "input changed while reading",
        )
        require(len(payload) == before.st_size, "input byte count changed")
        return payload
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CheckpointContract:
    document: dict[str, Any]
    payload: bytes
    sha256: str

    @property
    def model_config(self) -> NativeDenoiserConfig:
        values = dict(self.document["model"])
        values.pop("expected_trainable_parameters")
        return NativeDenoiserConfig(**values)


def load_contract(path: Path, expected_sha256: str) -> CheckpointContract:
    payload = read_regular(path, 65_536)
    require(sha(payload) == expected_sha256, "checkpoint config digest differs")
    value = tomllib.loads(payload.decode())
    require(
        set(value)
        == {
            "schema_version",
            "artifact",
            "purpose",
            "oracle_calls",
            "production_generator_change",
            "scientific_superiority_claim",
            "protocol_sha256",
            "input",
            "fits",
            "model",
            "training",
            "diagnostics",
            "resources",
        },
        "checkpoint config top-level schema differs",
    )
    require(
        value["schema_version"] == 1 and type(value["schema_version"]) is int,
        "schema version differs",
    )
    require(value["artifact"] == "generator_namespace_native_checkpoints_v1", "artifact differs")
    require(
        value["oracle_calls"] == 0 and type(value["oracle_calls"]) is int,
        "oracle calls are forbidden",
    )
    require(
        value["production_generator_change"] is False
        and value["scientific_superiority_claim"] is False,
        "asset scope differs",
    )
    require(tuple(value["fits"]["triples"]) == TRIPLES, "fold triples differ")
    require(
        value["fits"]["seeds"] == [530000 + int(triple) for triple in TRIPLES],
        "predeclared fit seeds differ",
    )
    require(value["fits"]["duplicate_ordinal"] == 0, "duplicate fit differs")
    require(
        value["fits"]["checkpoint_selection"] == "final_step_only_no_diagnostic_selection",
        "checkpoint selection differs",
    )
    expected_training = {
        "steps": 1000,
        "batch_sequences": 64,
        "learning_rate": 2e-4,
        "final_learning_rate": 2e-5,
        "warmup_steps": 100,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.05,
        "gradient_clip_norm": 1.0,
        "cosine_offset": 0.008,
        "mixed_precision": False,
        "validation_early_stopping": False,
        "automatic_retries": 0,
    }
    require(
        value["training"] == expected_training
        and all(
            type(value["training"][key]) is type(item) for key, item in expected_training.items()
        ),
        "frozen training recipe differs",
    )
    require(value["diagnostics"]["holdout_levels"] == [8, 24, 40, 56], "diagnostic levels differ")
    require(
        value["diagnostics"]["raw_samples_per_checkpoint"] == 128, "diagnostic sample count differs"
    )
    require(
        value["diagnostics"]["holdout_seed"] == 640000
        and value["diagnostics"]["sample_seed"] == 650000,
        "diagnostic seeds differ",
    )
    require(
        value["diagnostics"]["checkpoint_selection_allowed"] is False
        and value["diagnostics"]["oracle_or_reference_evaluation_allowed"] is False,
        "diagnostic scope differs",
    )
    resources = value["resources"]
    require(
        resources["maximum_total_gpu_seconds"] == 9900
        and resources["maximum_fit_allocations_including_duplicate"] == 11,
        "aggregate resource budget differs",
    )
    require(
        resources["maximum_concurrent_gpus"] == 2
        and resources["maximum_allocation_seconds_per_fit"] == 900
        and resources["maximum_worker_seconds_per_fit"] == 720,
        "per-fit resource budget differs",
    )
    contract = CheckpointContract(value, payload, expected_sha256)
    require(
        contract.model_config
        == NativeDenoiserConfig(layers=2, hidden_dim=128, attention_heads=4, ffn_dim=384),
        "model architecture differs",
    )
    require(value["model"]["expected_trainable_parameters"] == 354068, "parameter count differs")
    return contract


def authenticate_generator_rows(
    contract: CheckpointContract, *, selected_twin: int = 0
) -> tuple[dict[str, Any], ...]:
    """Read both accepted generator views; never open an oracle-namespace file."""

    require(type(selected_twin) is int and selected_twin in (0, 1), "generator twin differs")
    source = contract.document["input"]
    receipt_bytes = read_regular(Path(source["receipt_path"]), 65_536)
    require(sha(receipt_bytes) == source["receipt_sha256"], "namespace receipt digest differs")
    receipt = json.loads(receipt_bytes)
    markers = receipt["validity_markers"]
    require(
        markers["independent_verification_passed"] is True
        and markers["authoritative_scheduler_finalization_passed"] is True,
        "namespace independent verification incomplete",
    )
    corpora = []
    for twin_id in (0, 1):
        snapshots, _, binding = _read_committed_tree(
            Path(source["generator_view_parent"]) / str(twin_id),
            expected_marker_artifact="generator_namespace_view_complete_v2",
            expected_files=GENERATOR_FILES,
            maximum_file_bytes=16_777_216,
            maximum_json_depth=16,
            maximum_json_containers=4096,
            maximum_json_string_bytes=65_536,
        )
        expected = receipt["twins"][twin_id]["surfaces"]["generator_view"]
        require(
            binding.marker_sha256 == expected["completion_marker_sha256"]
            and binding.root_dev == expected["root_dev"]
            and binding.root_ino == expected["root_ino"],
            "generator view path/marker binding differs",
        )
        require(binding.files == expected["file_bindings"], "generator view file bindings differ")
        payload = snapshots["corpus.jsonl"].payload
        require(sha(payload) == source["corpus_sha256"], "generator corpus digest differs")
        corpora.append(payload)
    require(corpora[0] == corpora[1], "generator twin corpora differ")
    rows = tuple(json.loads(line) for line in corpora[selected_twin].splitlines())
    validate_generator_rows(rows)
    require(len(rows) == source["expected_sequences"], "generator sequence count differs")
    require(
        [sum(row["generator_fold"] == fold for row in rows) for fold in range(5)]
        == source["expected_fold_counts"],
        "generator fold counts differ",
    )
    require(
        len({row["union_component_id"] for row in rows}) == source["expected_union_components"],
        "generator component count differs",
    )
    return rows


def validate_generator_rows(rows: tuple[dict[str, Any], ...]) -> None:
    require(0 < len(rows) <= 4096, "generator row count is invalid")
    previous = ""
    component_folds: dict[str, int] = {}
    for row in rows:
        require(set(row) == CORPUS_FIELDS, "generator row contains unexpected fields")
        sequence = row["sequence"]
        require(
            type(sequence) is str
            and 8 <= len(sequence) <= 50
            and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"),
            "generator sequence support differs",
        )
        require(
            canonical_sequence_id(sequence) == row["sequence_id"] and previous < row["sequence_id"],
            "generator sequence identity/order differs",
        )
        previous = row["sequence_id"]
        fold = row["generator_fold"]
        require(
            type(fold) is int and 0 <= fold < 5 and row["namespace"] == "generator",
            "generator namespace/fold differs",
        )
        component = row["union_component_id"]
        require(
            component_folds.setdefault(component, fold) == fold, "union component crosses folds"
        )


def training_distribution(rows: tuple[dict[str, Any], ...], triple: str) -> TrainingDistribution:
    require(triple in TRIPLES, "training fold triple differs")
    chosen = tuple(row for row in rows if str(row["generator_fold"]) in triple)
    require(bool(chosen), "training triple is empty")
    sizes = Counter(row["union_component_id"] for row in chosen)
    raw = [1.0 / sizes[row["union_component_id"]] for row in chosen]
    normalizer = math.fsum(raw)
    weights = tuple(weight / normalizer for weight in raw)
    training = tuple(
        TrainingRow(row["sequence_id"], row["sequence"], weight)
        for row, weight in zip(chosen, weights, strict=True)
    )
    lengths: dict[int, list[float]] = defaultdict(list)
    for row in training:
        lengths[len(row.sequence)].append(row.sampling_weight)
    support = tuple(sorted(lengths))
    mass = tuple(math.fsum(lengths[length]) for length in support)
    total = math.fsum(mass)
    prior = LengthPrior(support, tuple(value / total for value in mass))
    return TrainingDistribution(training, weights, prior)


def projection_bytes(distribution: TrainingDistribution) -> bytes:
    return b"".join(
        canonical(
            {
                "sequence_id": row.sequence_id,
                "sequence": row.sequence,
                "sampling_weight": row.sampling_weight,
            }
        )
        for row in distribution.rows
    )


def learning_rate(step: int) -> float:
    require(type(step) is int and 1 <= step <= 1000, "optimizer step differs")
    if step <= 100:
        return 2e-4 * step / 100
    return 2e-5 + (2e-4 - 2e-5) * 0.5 * (1 + math.cos(math.pi * (step - 100) / 900))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Authenticate the generator-only checkpoint inputs without training"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    arguments = parser.parse_args()
    contract = load_contract(arguments.config, arguments.expected_config_sha256)
    rows = authenticate_generator_rows(contract)
    print(
        json.dumps(
            {
                "config_sha256": contract.sha256,
                "generator_rows": len(rows),
                "training_rows_by_triple": {
                    triple: len(training_distribution(rows, triple).rows) for triple in TRIPLES
                },
                "oracle_calls": 0,
            },
            sort_keys=True,
        )
    )

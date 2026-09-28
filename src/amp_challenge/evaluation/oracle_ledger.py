"""Convert strain-level OOF oracle predictions into a sequence acquisition ledger.

The resulting table keeps measured outcomes beside, but logically separate from,
the model-facing prediction columns expected by :mod:`amp_challenge.evaluation.replay`.
Only independently fitted member models contribute to the mean and disagreement;
a derived ensemble row is deliberately ignored to avoid double counting it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_GRAM_CLASSES = frozenset({"positive", "negative", "unknown"})


@dataclass(frozen=True, slots=True)
class ObjectiveRule:
    """Predicate selecting strain examples for one sequence-level objective."""

    name: str
    grams: frozenset[str]
    strain_regex: str | None = None

    def matches(self, *, gram: str, strain: str) -> bool:
        if gram not in self.grams:
            return False
        return self.strain_regex is None or re.search(self.strain_regex, strain) is not None


@dataclass(frozen=True, slots=True)
class OracleLedgerConfig:
    path: Path
    member_models: tuple[str, ...]
    objectives: tuple[ObjectiveRule, ...]
    minimum_observations: int


@dataclass(frozen=True, slots=True)
class _Example:
    example_id: str
    sequence_id: str
    sequence: str
    strain: str
    gram: str
    label: int
    fold: str
    cluster_id: str
    max_train_identity: float
    predictions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OracleLedgerExecution:
    ledger_path: Path
    summary_path: Path
    candidate_count: int
    excluded_sequence_count: int
    summary: Mapping[str, object]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"oracle-ledger config {field!r} must be a non-empty string")
    return value.strip()


def load_oracle_ledger_config(path: str | Path) -> OracleLedgerConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    allowed = {"schema_version", "member_models", "minimum_observations", "objective"}
    unexpected = set(document) - allowed
    if unexpected:
        raise ValueError(f"unexpected oracle-ledger config key(s): {sorted(unexpected)}")
    if document.get("schema_version") != 1:
        raise ValueError("oracle-ledger config schema_version must be 1")

    members_raw = document.get("member_models")
    if (
        not isinstance(members_raw, list)
        or len(members_raw) < 2
        or not all(isinstance(item, str) and item.strip() for item in members_raw)
    ):
        raise ValueError("member_models must contain at least two non-empty model names")
    members = tuple(item.strip() for item in members_raw)
    if len(members) != len(set(members)):
        raise ValueError("member_models must be unique")

    minimum = document.get("minimum_observations", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_observations must be a positive integer")

    rules_raw = document.get("objective")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ValueError("at least one [[objective]] table is required")
    rules: list[ObjectiveRule] = []
    for index, raw in enumerate(rules_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"objective {index} must be a TOML table")
        extra = set(raw) - {"name", "grams", "strain_regex"}
        if extra:
            raise ValueError(f"objective {index} has unexpected key(s): {sorted(extra)}")
        name = _nonempty_string(raw.get("name"), field=f"objective[{index}].name")
        grams_raw = raw.get("grams", sorted(_GRAM_CLASSES))
        if (
            not isinstance(grams_raw, list)
            or not grams_raw
            or not all(isinstance(item, str) for item in grams_raw)
        ):
            raise ValueError(f"objective {name!r} grams must be a non-empty string array")
        grams = frozenset(item.strip() for item in grams_raw)
        if not grams <= _GRAM_CLASSES:
            raise ValueError(f"objective {name!r} has invalid Gram classes: {sorted(grams)}")
        pattern_raw = raw.get("strain_regex")
        pattern = None
        if pattern_raw is not None:
            pattern = _nonempty_string(pattern_raw, field=f"objective[{index}].strain_regex")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"objective {name!r} has invalid strain_regex: {error}") from error
        rules.append(ObjectiveRule(name=name, grams=grams, strain_regex=pattern))
    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise ValueError("objective names must be unique")
    return OracleLedgerConfig(
        path=config_path,
        member_models=members,
        objectives=tuple(rules),
        minimum_observations=minimum,
    )


def _finite_float(value: str | None, *, field: str, row: int) -> float:
    try:
        parsed = float(value or "")
    except ValueError as error:
        raise ValueError(f"OOF row {row} has invalid {field!r}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"OOF row {row} has non-finite {field!r}")
    return parsed


def _read_examples(path: Path, members: Sequence[str]) -> tuple[_Example, ...]:
    required = {
        "model",
        "example_id",
        "sequence_id",
        "sequence",
        "strain",
        "gram",
        "label",
        "fold",
        "cluster_id",
        "max_train_identity",
        "probability",
    }
    metadata: dict[str, tuple[str, str, str, str, int, str, str, float]] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    member_set = set(members)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OOF prediction CSV is missing a header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"OOF prediction CSV is missing column(s): {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            model = (row.get("model") or "").strip()
            if model not in member_set:
                continue
            example_id = (row.get("example_id") or "").strip()
            if not example_id:
                raise ValueError(f"OOF row {row_number} has an empty example_id")
            sequence = canonicalize_sequence(row.get("sequence") or "")
            sequence_id = (row.get("sequence_id") or "").strip()
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"OOF row {row_number} sequence_id does not match sequence")
            strain = (row.get("strain") or "").strip()
            gram = (row.get("gram") or "").strip()
            if not strain or gram not in _GRAM_CLASSES:
                raise ValueError(f"OOF row {row_number} has invalid strain/Gram metadata")
            label_raw = (row.get("label") or "").strip()
            if label_raw not in {"0", "1"}:
                raise ValueError(f"OOF row {row_number} label must be 0 or 1")
            fold = (row.get("fold") or "").strip()
            cluster_id = (row.get("cluster_id") or "").strip()
            if not fold or not cluster_id:
                raise ValueError(f"OOF row {row_number} has empty fold/cluster metadata")
            max_identity = _finite_float(
                row.get("max_train_identity"), field="max_train_identity", row=row_number
            )
            probability = _finite_float(row.get("probability"), field="probability", row=row_number)
            if not 0 <= max_identity <= 1 or not 0 <= probability <= 1:
                raise ValueError(f"OOF row {row_number} probability/identity must be in [0, 1]")
            current = (
                sequence_id,
                sequence,
                strain,
                gram,
                int(label_raw),
                fold,
                cluster_id,
                max_identity,
            )
            previous = metadata.setdefault(example_id, current)
            if previous != current:
                raise ValueError(f"OOF example {example_id!r} has inconsistent metadata")
            if model in predictions[example_id]:
                raise ValueError(f"OOF example {example_id!r} duplicates model {model!r}")
            predictions[example_id][model] = probability

    if not metadata:
        raise ValueError("OOF prediction CSV has no configured member-model rows")
    examples: list[_Example] = []
    for example_id in sorted(metadata):
        missing_members = member_set - set(predictions[example_id])
        if missing_members:
            raise ValueError(
                f"OOF example {example_id!r} is missing member(s): {sorted(missing_members)}"
            )
        sequence_id, sequence, strain, gram, label, fold, cluster_id, identity = metadata[
            example_id
        ]
        examples.append(
            _Example(
                example_id=example_id,
                sequence_id=sequence_id,
                sequence=sequence,
                strain=strain,
                gram=gram,
                label=label,
                fold=fold,
                cluster_id=cluster_id,
                max_train_identity=identity,
                predictions=dict(predictions[example_id]),
            )
        )
    return tuple(examples)


def _sequence_embedding(sequence: str) -> tuple[float, ...]:
    descriptors = compute_descriptors(sequence)
    length = len(sequence)
    composition = tuple(sequence.count(residue) / length for residue in STANDARD_AMINO_ACIDS)
    return (
        *composition,
        length / 50.0,
        descriptors.charge_density,
        descriptors.mean_hydrophobicity,
        descriptors.hydrophobic_moment,
        descriptors.shannon_entropy / math.log2(20),
    )


def _format(value: object) -> object:
    return format(value, ".12g") if isinstance(value, float) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format(row[name]) for name in fieldnames})


def build_oracle_replay_ledger(
    oof_path: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> OracleLedgerExecution:
    """Aggregate strain examples without exposing their labels to acquisition."""

    input_path = Path(oof_path).resolve()
    config = load_oracle_ledger_config(config_path)
    examples = _read_examples(input_path, config.member_models)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    by_sequence: dict[str, list[_Example]] = defaultdict(list)
    for example in examples:
        by_sequence[example.sequence_id].append(example)
    rows: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    support: dict[str, list[int]] = {rule.name: [] for rule in config.objectives}
    embedding_names = [
        *(f"aa_{residue}" for residue in STANDARD_AMINO_ACIDS),
        "length",
        "charge_density",
        "mean_hydrophobicity",
        "hydrophobic_moment",
        "entropy",
    ]

    for sequence_id in sorted(by_sequence):
        sequence_examples = by_sequence[sequence_id]
        sequence = sequence_examples[0].sequence
        folds = {item.fold for item in sequence_examples}
        clusters = {item.cluster_id for item in sequence_examples}
        identities = {item.max_train_identity for item in sequence_examples}
        if len(folds) != 1 or len(clusters) != 1 or len(identities) != 1:
            raise ValueError(
                f"sequence {sequence_id} has inconsistent fold/cluster/identity metadata"
            )

        objective_values: dict[str, tuple[float, float, float, int]] = {}
        missing: list[str] = []
        for rule in config.objectives:
            selected = [
                item
                for item in sequence_examples
                if rule.matches(gram=item.gram, strain=item.strain)
            ]
            if len(selected) < config.minimum_observations:
                missing.append(rule.name)
                continue
            member_means = np.asarray(
                [
                    np.mean([item.predictions[member] for item in selected])
                    for member in config.member_models
                ],
                dtype=np.float64,
            )
            objective_values[rule.name] = (
                float(np.mean(member_means)),
                float(np.std(member_means)),
                float(np.mean([item.label for item in selected])),
                len(selected),
            )
        if missing:
            exclusions["missing:" + ",".join(missing)] += 1
            continue

        fold = next(iter(folds))
        row: dict[str, object] = {
            "sequence_id": sequence_id,
            "sequence": sequence,
            "novelty": 1.0 - next(iter(identities)),
            "cluster_id": next(iter(clusters)),
            "eligible": "true",
            "replay_round": f"fold-{fold}",
            "prediction_scope": "out_of_fold",
            "prediction_fold": fold,
            "outcome_fold": fold,
        }
        for rule in config.objectives:
            mean, std, outcome, count = objective_values[rule.name]
            row[f"mean_{rule.name}"] = mean
            row[f"std_{rule.name}"] = std
            row[f"outcome_{rule.name}"] = outcome
            row[f"n_{rule.name}"] = count
            support[rule.name].append(count)
        for name, value in zip(embedding_names, _sequence_embedding(sequence), strict=True):
            row[f"embedding_{name}"] = value
        rows.append(row)

    if not rows:
        raise ValueError("no sequence has sufficient observations for every objective")
    rows.sort(key=lambda row: str(row["sequence"]))
    objective_names = [rule.name for rule in config.objectives]
    fieldnames = [
        "sequence_id",
        "sequence",
        *(f"mean_{name}" for name in objective_names),
        *(f"std_{name}" for name in objective_names),
        "novelty",
        *(f"embedding_{name}" for name in embedding_names),
        "cluster_id",
        "eligible",
        "replay_round",
        "prediction_scope",
        "prediction_fold",
        "outcome_fold",
        *(f"outcome_{name}" for name in objective_names),
        *(f"n_{name}" for name in objective_names),
    ]
    ledger_path = output / "candidate_ledger.csv"
    summary_path = output / "ledger_summary.json"
    _write_csv(ledger_path, rows, fieldnames)
    candidates_by_fold = Counter(str(row["prediction_fold"]) for row in rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "oof_sha256": _sha256(input_path),
        "config_sha256": _sha256(config.path),
        "member_models": list(config.member_models),
        "objectives": objective_names,
        "input_examples": len(examples),
        "input_sequences": len(by_sequence),
        "candidate_sequences": len(rows),
        "excluded_sequences": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "candidates_by_fold": dict(sorted(candidates_by_fold.items())),
        "objective_observation_support": {
            name: {
                "minimum": min(values),
                "median": float(np.median(values)),
                "maximum": max(values),
            }
            for name, values in support.items()
        },
        "uncertainty": "population standard deviation across independently fitted member aggregates",
        "ledger_sha256": _sha256(ledger_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return OracleLedgerExecution(
        ledger_path=ledger_path,
        summary_path=summary_path,
        candidate_count=len(rows),
        excluded_sequence_count=sum(exclusions.values()),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oof_predictions", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = build_oracle_replay_ledger(
        args.oof_predictions,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "candidates": execution.candidate_count,
                "excluded_sequences": execution.excluded_sequence_count,
                "ledger": str(execution.ledger_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Select a mixed-acquisition experiment batch from a model-agnostic CSV ledger.

The ledger contains one candidate per row, a required ``sequence`` column, and
 paired ``mean_<objective>``/``std_<objective>`` columns when uncertainty is
 required.  Explicit uncertainty-unavailable policies require only
 ``mean_<objective>``, reject ``std_*`` columns, and require every row to carry
 the literal ``uncertainty_status=unavailable``.  Optional columns are
``sequence_id``, ``novelty``, ``embedding_*``, ``cluster_id``, ``start_id``,
``rollout_id``, ``eligible``, and ``specialist_<name>``.  A supplied
``sequence_id`` must be the canonical sequence's lowercase SHA-256.  A minimal
TOML file looks like::

    objectives = ["potency", "selectivity"]

    [selection]
    batch_size = 100
    seed = 42
    max_per_cluster = 5

    [selection.strategy_mix]
    exploit = 0.5
    pareto = 0.25
    diversity = 0.25

This layer intentionally does not infer endpoint direction or calibrate model
outputs.  Every mean must already be in higher-is-better utility orientation and
comparable enough for the rank-based :class:`MixedAcquisitionSelector`.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from amp_challenge.acquisition import (
    CandidateBatch,
    MixedAcquisitionSelector,
    SelectionConfig,
    SelectionResult,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_DERIVED_AUDIT_COLUMNS = (
    "selection_rank",
    "selection_candidate_index",
    "selection_ledger_row",
    "selection_sequence_id",
    "selection_reason",
    "selection_conservative_score",
    "selection_acquisition_score",
)
_START_DERIVED_AUDIT_COLUMNS = (
    "selection_start_id",
    "selection_rollout_id",
    "selection_start_rank",
    "selection_start_eligible_rollout_count",
    "selection_start_rollout_value_mean",
    "selection_start_rollout_value_dispersion",
    "selection_rollout_ucb_score",
)
_CANONICAL_SEQUENCE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SelectionRunConfig:
    """Selection policy plus the ordered ledger objective names, if declared."""

    selection: SelectionConfig
    objectives: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class LedgerMetadata:
    """Input provenance retained alongside one canonicalized candidate row."""

    ledger_row: int
    sequence_id: str
    values: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class CandidateLedger:
    """Validated selector input with metadata aligned to ``CandidateBatch``."""

    candidates: CandidateBatch
    objectives: tuple[str, ...]
    fieldnames: tuple[str, ...]
    embedding_columns: tuple[str, ...]
    specialist_names: tuple[str, ...]
    metadata: tuple[LedgerMetadata, ...]


@dataclass(frozen=True, slots=True)
class SelectionExecution:
    """In-memory result returned after writing both selection artifacts."""

    config: SelectionRunConfig
    ledger: CandidateLedger
    result: SelectionResult


@dataclass(frozen=True, slots=True)
class _ParsedCandidate:
    sequence: str
    sequence_id: str
    ledger_row: int
    means: tuple[float, ...]
    stds: tuple[float, ...]
    novelty: float | None
    embedding: tuple[float, ...] | None
    cluster_id: str | None
    start_id: str | None
    rollout_id: str | None
    eligible: bool
    specialists: tuple[tuple[str, float], ...]
    values: tuple[tuple[str, str], ...]


def _objective_names(raw: object, *, location: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{location} must be a non-empty TOML string array")
    names = tuple(item.strip() for item in raw)
    if any(not name for name in names):
        raise ValueError(f"{location} cannot contain an empty objective name")
    if len(names) != len(set(names)):
        raise ValueError(f"{location} objective names must be unique")
    return names


def load_selection_run_config(path: str | Path) -> SelectionRunConfig:
    """Load objectives and a validated :class:`SelectionConfig` from TOML.

    Selection fields may live under ``[selection]`` (recommended) or directly
    at the document root.  Objectives may be top-level, under ``[ledger]``, or
    inside ``[selection]``; conflicting declarations are rejected.
    """

    config_path = Path(path)
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):  # pragma: no cover - guaranteed by tomllib
        raise ValueError("selection TOML root must be a table")

    selection_field_names = {field.name for field in fields(SelectionConfig)}
    nested_selection = document.get("selection")
    if nested_selection is not None:
        if not isinstance(nested_selection, dict):
            raise ValueError("[selection] must be a TOML table")
        direct_fields = selection_field_names & set(document)
        if direct_fields:
            raise ValueError(
                "selection fields cannot appear both at the root and under [selection]: "
                f"{sorted(direct_fields)}"
            )
        selection_values: dict[str, Any] = dict(nested_selection)
        allowed_root = {"selection", "ledger", "objectives"}
        unexpected_root = set(document) - allowed_root
        if unexpected_root:
            raise ValueError(f"unexpected top-level TOML key(s): {sorted(unexpected_root)}")
    else:
        selection_values = {
            key: value for key, value in document.items() if key in selection_field_names
        }
        unexpected_root = set(document) - selection_field_names - {"ledger", "objectives"}
        if unexpected_root:
            raise ValueError(f"unexpected top-level TOML key(s): {sorted(unexpected_root)}")

    declarations: list[tuple[str, tuple[str, ...]]] = []
    if "objectives" in document:
        declarations.append(
            ("objectives", _objective_names(document["objectives"], location="objectives"))
        )
    ledger_section = document.get("ledger")
    if ledger_section is not None:
        if not isinstance(ledger_section, dict):
            raise ValueError("[ledger] must be a TOML table")
        unexpected_ledger = set(ledger_section) - {"objectives"}
        if unexpected_ledger:
            raise ValueError(f"unexpected [ledger] key(s): {sorted(unexpected_ledger)}")
        if "objectives" in ledger_section:
            declarations.append(
                (
                    "ledger.objectives",
                    _objective_names(
                        ledger_section["objectives"],
                        location="ledger.objectives",
                    ),
                )
            )
    if "objectives" in selection_values:
        declarations.append(
            (
                "selection.objectives",
                _objective_names(
                    selection_values.pop("objectives"),
                    location="selection.objectives",
                ),
            )
        )
    if declarations and any(value != declarations[0][1] for _, value in declarations[1:]):
        locations = ", ".join(location for location, _ in declarations)
        raise ValueError(f"conflicting objective declarations in {locations}")
    objectives = declarations[0][1] if declarations else None

    unexpected_selection = set(selection_values) - selection_field_names
    if unexpected_selection:
        raise ValueError(f"unexpected [selection] key(s): {sorted(unexpected_selection)}")
    for mapping_name in ("strategy_mix", "specialist_quotas"):
        mapping = selection_values.get(mapping_name)
        if mapping is not None:
            if not isinstance(mapping, dict):
                raise ValueError(f"selection.{mapping_name} must be a TOML table")
            selection_values[mapping_name] = dict(sorted(mapping.items()))
    for integer_name in (
        "batch_size",
        "max_per_cluster",
        "max_per_start",
        "rollouts_per_start",
        "seed",
    ):
        value = selection_values.get(integer_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"selection.{integer_name} must be an integer")
    strict_cluster_cap = selection_values.get("strict_cluster_cap")
    if strict_cluster_cap is not None and not isinstance(strict_cluster_cap, bool):
        raise ValueError("selection.strict_cluster_cap must be a boolean")
    for finite_name in (
        "risk_beta",
        "ucb_beta",
        "diversity_quality_weight",
        "quality_floor_quantile",
    ):
        value = selection_values.get(finite_name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"selection.{finite_name} must be a finite number")
    specialist_quotas = selection_values.get("specialist_quotas")
    if specialist_quotas is not None:
        if any(not str(name).strip() for name in specialist_quotas):
            raise ValueError("selection.specialist_quotas cannot contain an empty name")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in specialist_quotas.values()
        ):
            raise ValueError("selection.specialist_quotas values must be integers")
    try:
        selection = SelectionConfig(**selection_values)
    except TypeError as error:
        raise ValueError(f"invalid [selection] configuration: {error}") from error
    return SelectionRunConfig(selection=selection, objectives=objectives)


def load_selection_config(path: str | Path) -> SelectionConfig:
    """Load only the selector policy from a TOML file."""

    return load_selection_run_config(path).selection


def _resolve_objectives(
    fieldnames: Sequence[str],
    configured: Sequence[str] | None,
    *,
    uncertainty_mode: str,
) -> tuple[str, ...]:
    if uncertainty_mode not in {"required", "unavailable"}:
        raise ValueError("uncertainty_mode must be exactly 'required' or 'unavailable'")
    mean_names = {name.removeprefix("mean_") for name in fieldnames if name.startswith("mean_")}
    std_names = {name.removeprefix("std_") for name in fieldnames if name.startswith("std_")}
    if "" in mean_names or "" in std_names:
        raise ValueError("mean_ and std_ columns require a non-empty objective suffix")
    if uncertainty_mode == "unavailable" and std_names:
        raise ValueError(
            "uncertainty_mode='unavailable' does not allow std_* ledger columns: "
            f"{sorted(f'std_{name}' for name in std_names)}"
        )
    if configured is None and uncertainty_mode == "required":
        unpaired_means = mean_names - std_names
        unpaired_stds = std_names - mean_names
        if unpaired_means or unpaired_stds:
            raise ValueError(
                "auto-detected objective columns are unpaired: "
                f"mean-only={sorted(unpaired_means)}, std-only={sorted(unpaired_stds)}"
            )
        objectives = tuple(sorted(mean_names))
        if not objectives:
            raise ValueError("ledger has no mean_<objective> columns")
    elif configured is None:
        objectives = tuple(sorted(mean_names))
        if not objectives:
            raise ValueError("ledger has no mean_<objective> columns")
    else:
        objectives = tuple(name.strip() for name in configured)
        if not objectives or any(not name.strip() for name in objectives):
            raise ValueError("at least one non-empty objective name is required")
        if len(objectives) != len(set(objectives)):
            raise ValueError("objective names must be unique")

    expected_columns = tuple(
        column
        for objective in objectives
        for column in (
            (f"mean_{objective}", f"std_{objective}")
            if uncertainty_mode == "required"
            else (f"mean_{objective}",)
        )
    )
    missing_columns = [column for column in expected_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"ledger is missing objective column(s): {missing_columns}")
    return objectives


def _finite_float(value: str | None, *, column: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ValueError(f"ledger row {row_number} has an empty {column!r} value")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"ledger row {row_number} has a non-numeric {column!r} value: {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"ledger row {row_number} has a non-finite {column!r} value")
    return parsed


def _eligible_bool(value: str | None, *, row_number: int) -> bool:
    if value is None or not value.strip():
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(
        f"ledger row {row_number} has invalid eligible value {value!r}; "
        "use true/false, yes/no, or 1/0"
    )


def _validated_sequence_id(
    value: str | None,
    *,
    sequence: str,
    row_number: int,
) -> str:
    if value is None or not value.strip():
        raise ValueError(f"ledger row {row_number} has an empty 'sequence_id' value")
    if _CANONICAL_SEQUENCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"ledger row {row_number} has malformed 'sequence_id'; "
            "expected exactly 64 lowercase hexadecimal characters"
        )
    expected = canonical_sequence_id(sequence)
    if value != expected:
        raise ValueError(f"ledger row {row_number} sequence_id does not match canonical sequence")
    return value


def read_candidate_ledger(
    path: str | Path,
    *,
    objectives: Sequence[str] | None = None,
    uncertainty_mode: str = "required",
) -> CandidateLedger:
    """Read, validate, and canonically order a candidate CSV ledger."""

    ledger_path = Path(path)
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("candidate ledger is missing a CSV header")
        fieldnames = tuple(reader.fieldnames)
        if any(name is None or not name for name in fieldnames):
            raise ValueError("candidate ledger contains an empty column name")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("candidate ledger contains duplicate column names")
        if "sequence" not in fieldnames:
            raise ValueError("candidate ledger requires a 'sequence' column")
        has_uncertainty_status = "uncertainty_status" in fieldnames
        if uncertainty_mode == "unavailable" and not has_uncertainty_status:
            raise ValueError("uncertainty_mode='unavailable' requires an uncertainty_status column")
        reserved = set(fieldnames) & set((*_DERIVED_AUDIT_COLUMNS, *_START_DERIVED_AUDIT_COLUMNS))
        if reserved:
            raise ValueError(f"candidate ledger uses reserved audit column(s): {sorted(reserved)}")

        objective_names = _resolve_objectives(
            fieldnames,
            objectives,
            uncertainty_mode=uncertainty_mode,
        )
        embedding_columns = tuple(
            sorted(name for name in fieldnames if name.startswith("embedding_"))
        )
        if "embedding_" in embedding_columns:
            raise ValueError("embedding_ columns require a non-empty feature suffix")
        specialist_columns = tuple(
            sorted(name for name in fieldnames if name.startswith("specialist_"))
        )
        if "specialist_" in specialist_columns:
            raise ValueError("specialist_ columns require a non-empty specialist suffix")
        specialist_names = tuple(name.removeprefix("specialist_") for name in specialist_columns)
        has_novelty = "novelty" in fieldnames
        has_clusters = "cluster_id" in fieldnames
        has_starts = "start_id" in fieldnames
        has_rollouts = "rollout_id" in fieldnames
        has_eligibility = "eligible" in fieldnames
        has_sequence_ids = "sequence_id" in fieldnames
        if has_rollouts and not has_starts:
            raise ValueError("a rollout_id ledger column requires start_id")

        parsed_rows: list[_ParsedCandidate] = []
        sequence_rows: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"ledger row {row_number} has more cells than the header")
            if uncertainty_mode == "unavailable" and row.get("uncertainty_status") != "unavailable":
                raise ValueError(
                    f"ledger row {row_number} must have literal uncertainty_status='unavailable'"
                )
            raw_sequence = row.get("sequence")
            try:
                sequence = canonicalize_sequence(raw_sequence or "")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"ledger row {row_number} has invalid sequence: {error}"
                ) from error
            if sequence in sequence_rows:
                raise ValueError(
                    f"ledger row {row_number} duplicates canonical sequence from row "
                    f"{sequence_rows[sequence]}: {sequence}"
                )
            sequence_rows[sequence] = row_number
            sequence_id = (
                _validated_sequence_id(
                    row.get("sequence_id"),
                    sequence=sequence,
                    row_number=row_number,
                )
                if has_sequence_ids
                else canonical_sequence_id(sequence)
            )

            means = tuple(
                _finite_float(
                    row.get(f"mean_{objective}"),
                    column=f"mean_{objective}",
                    row_number=row_number,
                )
                for objective in objective_names
            )
            stds = (
                tuple(
                    _finite_float(
                        row.get(f"std_{objective}"),
                        column=f"std_{objective}",
                        row_number=row_number,
                    )
                    for objective in objective_names
                )
                if uncertainty_mode == "required"
                else (0.0,) * len(objective_names)
            )
            if any(value < 0 for value in stds):
                raise ValueError(f"ledger row {row_number} has a negative objective std")
            novelty = (
                _finite_float(row.get("novelty"), column="novelty", row_number=row_number)
                if has_novelty
                else None
            )
            embedding = (
                tuple(
                    _finite_float(row.get(name), column=name, row_number=row_number)
                    for name in embedding_columns
                )
                if embedding_columns
                else None
            )
            cluster_id = (row.get("cluster_id") or "").strip() if has_clusters else None
            if has_clusters and not cluster_id:
                raise ValueError(f"ledger row {row_number} has an empty 'cluster_id' value")
            start_id = (row.get("start_id") or "").strip() if has_starts else None
            if has_starts and not start_id:
                raise ValueError(f"ledger row {row_number} has an empty 'start_id' value")
            rollout_id = (row.get("rollout_id") or "").strip() if has_rollouts else None
            if has_rollouts and not rollout_id:
                raise ValueError(f"ledger row {row_number} has an empty 'rollout_id' value")
            eligible = (
                _eligible_bool(row.get("eligible"), row_number=row_number)
                if has_eligibility
                else True
            )
            specialists = tuple(
                (
                    name,
                    _finite_float(
                        row.get(f"specialist_{name}"),
                        column=f"specialist_{name}",
                        row_number=row_number,
                    ),
                )
                for name in specialist_names
            )
            values = tuple(
                (
                    name,
                    start_id
                    if name == "start_id"
                    else rollout_id
                    if name == "rollout_id"
                    else row.get(name) or "",
                )
                for name in fieldnames
            )
            parsed_rows.append(
                _ParsedCandidate(
                    sequence=sequence,
                    sequence_id=sequence_id,
                    ledger_row=row_number,
                    means=means,
                    stds=stds,
                    novelty=novelty,
                    embedding=embedding,
                    cluster_id=cluster_id,
                    start_id=start_id,
                    rollout_id=rollout_id,
                    eligible=eligible,
                    specialists=specialists,
                    values=values,
                )
            )

    if not parsed_rows:
        raise ValueError("candidate ledger contains no data rows")
    parsed_rows.sort(key=lambda item: item.sequence)
    candidates = CandidateBatch(
        sequences=tuple(item.sequence for item in parsed_rows),
        objective_mean=np.asarray([item.means for item in parsed_rows], dtype=np.float64),
        objective_std=np.asarray([item.stds for item in parsed_rows], dtype=np.float64),
        novelty=(
            np.asarray([item.novelty for item in parsed_rows], dtype=np.float64)
            if has_novelty
            else None
        ),
        embeddings=(
            np.asarray([item.embedding for item in parsed_rows], dtype=np.float64)
            if embedding_columns
            else None
        ),
        cluster_ids=(
            tuple(item.cluster_id or "" for item in parsed_rows) if has_clusters else None
        ),
        start_ids=(tuple(item.start_id or "" for item in parsed_rows) if has_starts else None),
        rollout_ids=(
            tuple(item.rollout_id or "" for item in parsed_rows) if has_rollouts else None
        ),
        specialist_scores={
            name: np.asarray(
                [dict(item.specialists)[name] for item in parsed_rows], dtype=np.float64
            )
            for name in specialist_names
        },
        eligible=np.asarray([item.eligible for item in parsed_rows], dtype=bool),
        uncertainty_available=uncertainty_mode == "required",
    )
    metadata = tuple(
        LedgerMetadata(
            ledger_row=item.ledger_row,
            sequence_id=item.sequence_id,
            values=item.values,
        )
        for item in parsed_rows
    )
    return CandidateLedger(
        candidates=candidates,
        objectives=objective_names,
        fieldnames=fieldnames,
        embedding_columns=embedding_columns,
        specialist_names=specialist_names,
        metadata=metadata,
    )


def write_ranked_fasta(
    path: str | Path,
    *,
    ledger: CandidateLedger,
    result: SelectionResult,
) -> None:
    """Write selected sequences in final conservative rank order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(result.indices))))
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, (index, reason, score) in enumerate(
            zip(result.indices, result.reasons, result.conservative_scores, strict=True),
            start=1,
        ):
            metadata = ledger.metadata[index]
            cluster_id = (
                "none"
                if ledger.candidates.cluster_ids is None
                else ledger.candidates.cluster_ids[index]
            )
            header = (
                f"rank={rank:0{width}d} sequence_id={metadata.sequence_id} "
                f"reason={quote(reason, safe=':._-')} score={score:.12g} "
                f"cluster={quote(cluster_id, safe=':._-')}"
            )
            if result.start_evidence:
                evidence = result.start_evidence[rank - 1]
                header += (
                    f" start={quote(evidence.start_id, safe=':._-')}"
                    f" rollout={quote(evidence.rollout_id, safe=':._-')}"
                    f" start_dispersion={evidence.rollout_value_dispersion:.12g}"
                )
            handle.write(f">{header}\n{ledger.candidates.sequences[index]}\n")


def write_selection_audit(
    path: str | Path,
    *,
    ledger: CandidateLedger,
    result: SelectionResult,
) -> None:
    """Write selected input rows with rank, stable ID, reason, and final score."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.start_evidence and len(result.start_evidence) != len(result.indices):
        raise ValueError("start_evidence must align with selected indices")
    derived_columns = (
        (*_DERIVED_AUDIT_COLUMNS, *_START_DERIVED_AUDIT_COLUMNS)
        if result.start_evidence
        else _DERIVED_AUDIT_COLUMNS
    )
    fieldnames = (*derived_columns, *ledger.fieldnames)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank, (index, reason, score, acquisition_score) in enumerate(
            zip(
                result.indices,
                result.reasons,
                result.conservative_scores,
                result.acquisition_scores,
                strict=True,
            ),
            start=1,
        ):
            metadata = ledger.metadata[index]
            input_values = metadata.as_dict()
            input_values["sequence"] = ledger.candidates.sequences[index]
            output_row: dict[str, object] = {
                "selection_rank": rank,
                "selection_candidate_index": index,
                "selection_ledger_row": metadata.ledger_row,
                "selection_sequence_id": metadata.sequence_id,
                "selection_reason": reason,
                "selection_conservative_score": format(score, ".17g"),
                "selection_acquisition_score": format(acquisition_score, ".17g"),
                **input_values,
            }
            if result.start_evidence:
                evidence = result.start_evidence[rank - 1]
                output_row.update(
                    {
                        "selection_start_id": evidence.start_id,
                        "selection_rollout_id": evidence.rollout_id,
                        "selection_start_rank": evidence.start_rank,
                        "selection_start_eligible_rollout_count": (evidence.eligible_rollout_count),
                        "selection_start_rollout_value_mean": format(
                            evidence.rollout_value_mean,
                            ".17g",
                        ),
                        "selection_start_rollout_value_dispersion": format(
                            evidence.rollout_value_dispersion,
                            ".17g",
                        ),
                        "selection_rollout_ucb_score": format(
                            evidence.rollout_ucb_score,
                            ".17g",
                        ),
                    }
                )
            writer.writerow(output_row)


def run_selection(
    ledger_path: str | Path,
    *,
    config_path: str | Path,
    fasta_output: str | Path,
    audit_output: str | Path,
    objectives: Sequence[str] | None = None,
) -> SelectionExecution:
    """Load inputs, select a portfolio, and write both deterministic artifacts."""

    ledger_resolved = Path(ledger_path).resolve()
    config_resolved = Path(config_path).resolve()
    fasta_resolved = Path(fasta_output).resolve()
    audit_resolved = Path(audit_output).resolve()
    if fasta_resolved == audit_resolved:
        raise ValueError("FASTA and audit outputs must be different paths")
    if fasta_resolved in {ledger_resolved, config_resolved} or audit_resolved in {
        ledger_resolved,
        config_resolved,
    }:
        raise ValueError("selection outputs cannot overwrite the ledger or configuration")
    run_config = load_selection_run_config(config_path)
    objective_names = tuple(objectives) if objectives is not None else run_config.objectives
    ledger = read_candidate_ledger(
        ledger_path,
        objectives=objective_names,
        uncertainty_mode=run_config.selection.uncertainty_mode,
    )
    if run_config.selection.max_per_cluster is not None and ledger.candidates.cluster_ids is None:
        raise ValueError("selection.max_per_cluster requires a cluster_id ledger column")
    result = MixedAcquisitionSelector(run_config.selection).select(ledger.candidates)
    write_ranked_fasta(fasta_output, ledger=ledger, result=result)
    write_selection_audit(audit_output, ledger=ledger, result=result)
    return SelectionExecution(config=run_config, ledger=ledger, result=result)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``amp-select`` command-line parser."""

    parser = argparse.ArgumentParser(
        description="Select a deterministic mixed-acquisition AMP experiment batch from CSV."
    )
    parser.add_argument("ledger", type=Path, help="candidate CSV ledger")
    parser.add_argument("--config", type=Path, required=True, help="selection TOML file")
    parser.add_argument("--fasta-out", type=Path, required=True, help="ranked FASTA output")
    parser.add_argument("--audit-out", type=Path, required=True, help="selected-row audit CSV")
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        help="ordered objective name; repeat to override/declare TOML objectives",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selection, returning 0 on success and 2 for invalid input/output."""

    args = build_parser().parse_args(argv)
    try:
        execution = run_selection(
            args.ledger,
            config_path=args.config,
            fasta_output=args.fasta_out,
            audit_output=args.audit_out,
            objectives=args.objectives,
        )
    except (OSError, UnicodeError, csv.Error, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"AMP selection error: {error}", file=sys.stderr)
        return 2
    print(
        "AMP selection complete: "
        f"selected={len(execution.result.indices):,} "
        f"candidates={len(execution.ledger.candidates.sequences):,} "
        f"fasta={args.fasta_out} audit={args.audit_out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script path is preferred
    raise SystemExit(main())

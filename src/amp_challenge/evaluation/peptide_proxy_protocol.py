"""Prospective, paired frozen-oracle protocol; does not alter legacy campaigns.

Distance thresholds are operational, geometry-specific constraints, not claims of
certified whole-model distribution bounds. Clipping is an independent training
setting. The oracle is benchmark authority, never asserted biological truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_UNITS = {
    "kl": "natural_log_nats",
    "wasserstein_1": "bounded_residue_transition_ground_cost",
    "total_variation": "absolute_probability_mass",
    "none": "not_applicable",
}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Arm:
    name: str
    method: str
    constraint: str
    threshold: float
    units: str


@dataclass(frozen=True)
class ProxyProtocol:
    study_id: str
    seeds: tuple[int, ...]
    initial_size: int
    additional_evaluations: int
    batch_size: int
    clip_ratio_width: float
    update_reference: str
    arms: tuple[Arm, ...]
    diagnostics: tuple[str, ...]
    protocol_sha256: str
    constraint_scope: str
    global_sequence_constraint_qualified: bool
    release_status: str
    per_transition_total_variation_limit: float
    ground_cost_id: str
    fixed_budget_selection: bool = False

    @property
    def total_evaluations(self) -> int:
        return self.initial_size + self.additional_evaluations

    @property
    def adaptive_rounds(self) -> int:
        return self.additional_evaluations // self.batch_size

    def accounting(self) -> dict[str, int]:
        runs = len(self.seeds) * len(self.arms)
        initial_physical = len(self.seeds) * self.initial_size
        return {
            "runs": runs,
            "initial_physical_evaluations": initial_physical,
            "initial_logical_imports": runs * self.initial_size,
            "additional_evaluation_allocations": runs * self.additional_evaluations,
            "total_physical_allocations_before_caching": initial_physical
            + runs * self.additional_evaluations,
            "total_logical_allocations": runs * self.total_evaluations,
        }


def load_protocol(path: str | Path) -> ProxyProtocol:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    if (raw["initial_size"], raw["additional_evaluations"]) != (512, 1024):
        raise ValueError(
            "This prospective protocol requires 512 initial + 1024 additional evaluations"
        )
    if raw["clip_ratio_width"] != 0.05 or raw["update_reference"] != "previous_accepted_model":
        raise ValueError(
            "The approved clipping width is 0.05 relative to the previous accepted model"
        )
    prospective = raw["release_status"] in {
        "prospective_native_scalar_proxy_v2",
        "prospective_native_fixed_budget_v3",
    }
    fixed_budget = raw["release_status"] == "prospective_native_fixed_budget_v3"
    if (
        fixed_budget
        and raw.get("query_allocation") != "signed_uncertainty_penalized_joint_kg_fixed_budget"
    ):
        raise ValueError("fixed-budget query allocation must be explicitly declared")
    expected_scope = (
        "raw_denoising_trajectory_conditional_on_fixed_parent_operator"
        if prospective
        else "conditional_residue_transition_distributions"
    )
    if (
        raw["constraint_scope"] != expected_scope
        or raw["global_sequence_constraint_qualified"] is not False
    ):
        raise ValueError("Transition diagnostics must not claim a global sequence guarantee")
    if raw["release_status"] != "planned_requires_method_provider_admission" and not prospective:
        raise ValueError("Native method execution is not yet released for this protocol")
    if prospective and (
        raw.get("method_seats_per_batch") != 14
        or raw.get("common_reserve_seats_per_batch") != 2
        or raw.get("raw_conditional_trajectory_total_variation_limit") != 0.05
        or raw.get("mixture_proposal_weight") != 0.05
        or raw.get("objective_adaptation")
        != "identical_scalar_replicas_not_independent_gram_objectives"
    ):
        raise ValueError("Prospective native scalar protocol rules differ")
    if raw["per_transition_total_variation_limit"] != 0.05:
        raise ValueError("Every treatment retains a 5% per-transition total variation guard")
    if raw["ground_cost_id"] != "canonical_amino_acid_composition_identity_v1":
        raise ValueError("Unrecognized Wasserstein ground-cost geometry")
    if raw["batch_size"] <= 0 or 1024 % raw["batch_size"]:
        raise ValueError("Batch size must divide the additional evaluation budget")
    seeds = tuple(raw["seeds"])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Seeds must be distinct integers")
    arms = tuple(Arm(**entry) for entry in raw["arms"])
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("Arm names must be nonempty and unique")
    for arm in arms:
        if arm.constraint not in _UNITS or arm.units != _UNITS[arm.constraint]:
            raise ValueError("Distance threshold units do not match the declared constraint")
        if not math.isfinite(arm.threshold) or arm.threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        if arm.constraint == "none" and arm.threshold != 0:
            raise ValueError("Unconstrained baseline must not declare an active threshold")
    return ProxyProtocol(
        raw["study_id"],
        seeds,
        512,
        1024,
        raw["batch_size"],
        0.05,
        raw["update_reference"],
        arms,
        tuple(raw["diagnostics"]),
        fingerprint(raw),
        raw["constraint_scope"],
        False,
        raw["release_status"],
        0.05,
        raw["ground_cost_id"],
        fixed_budget,
    )


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    peptide_id, sequence = record["peptide_id"], record["sequence"]
    if not isinstance(peptide_id, str) or not peptide_id:
        raise ValueError("peptide_id must be a nonempty string")
    if (
        not isinstance(sequence, str)
        or not 8 <= len(sequence) <= 50
        or not set(sequence) <= _AMINO_ACIDS
    ):
        raise ValueError("Sequences must be canonical uppercase peptides of length 8 through 50")
    score = float(record["oracle_score"])
    if not math.isfinite(score):
        raise ValueError("Initial oracle scores must be finite; do not silently replace failures")
    return {"peptide_id": peptide_id, "sequence": sequence, "oracle_score": score}


def select_initial_sequences(
    records: Iterable[Mapping[str, Any]], *, seed: int, protocol: ProxyProtocol
) -> list[dict[str, str]]:
    """Select before scoring: no oracle outputs are inspected or returned."""
    if seed not in protocol.seeds:
        raise ValueError("Seed is not declared in protocol")
    rows = []
    for record in records:
        row = _validate_record(
            {
                "peptide_id": record["peptide_id"],
                "sequence": record["sequence"],
                "oracle_score": 0.0,
            }
        )
        rows.append({"peptide_id": row["peptide_id"], "sequence": row["sequence"]})
    if len({r["sequence"] for r in rows}) != len(rows) or len(
        {r["peptide_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Candidate pool contains duplicate sequence or peptide identifier")
    if len(rows) < protocol.initial_size:
        raise ValueError("Fewer than 512 unique admitted candidates")
    rows.sort(
        key=lambda row: (fingerprint([protocol.study_id, seed, row["sequence"]]), row["sequence"])
    )
    return rows[: protocol.initial_size]


def build_initial_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    oracle_id: str,
    oracle_sha256: str,
    dataset_sha256: str,
    protocol: ProxyProtocol,
) -> dict[str, Any]:
    """Freeze exactly the preselected 512 rows after scoring only those rows.

    Call select_initial_sequences first; never score the whole candidate pool.
    Refuse failed labels rather than replacing them after inspecting outcomes.
    """
    if seed not in protocol.seeds:
        raise ValueError("Seed is not declared in protocol")
    if not oracle_id or any(
        len(s) != 64 or any(c not in "0123456789abcdef" for c in s)
        for s in (oracle_sha256, dataset_sha256)
    ):
        raise ValueError("Oracle identity and lowercase SHA-256 fingerprints are required")
    rows = [_validate_record(record) for record in records]
    if len({r["sequence"] for r in rows}) != len(rows) or len(
        {r["peptide_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Candidate pool contains duplicate sequence or peptide identifier")
    if len(rows) != protocol.initial_size:
        raise ValueError("Score and freeze exactly 512 preselected candidates, not the whole pool")
    rows.sort(
        key=lambda row: (fingerprint([protocol.study_id, seed, row["sequence"]]), row["sequence"])
    )
    body = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.protocol_sha256,
        "seed": seed,
        "oracle_id": oracle_id,
        "oracle_sha256": oracle_sha256,
        "dataset_sha256": dataset_sha256,
        "selected_identity_sha256": fingerprint(
            sorted((r["peptide_id"], r["sequence"]) for r in rows)
        ),
        "selection_policy": "seeded_sequence_hash_without_label_access",
        "records": rows[: protocol.initial_size],
    }
    return {**body, "manifest_sha256": fingerprint(body)}


def verify_initial_manifest(manifest: Mapping[str, Any], protocol: ProxyProtocol) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if fingerprint(body) != manifest.get("manifest_sha256"):
        raise ValueError("Initial manifest fingerprint mismatch")
    if (
        manifest["protocol_sha256"] != protocol.protocol_sha256
        or manifest["seed"] not in protocol.seeds
    ):
        raise ValueError("Initial manifest belongs to another protocol or seed")
    rows = [_validate_record(r) for r in manifest["records"]]
    if (
        len(rows) != protocol.initial_size
        or len({r["sequence"] for r in rows}) != len(rows)
        or len({r["peptide_id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Initial manifest must contain exactly 512 unique peptides")


def write_initial_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Freeze once; never overwrite an existing per-seed artifact."""
    serialized = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)


@dataclass
class EvaluationLedger:
    """Per-arm logical charges, with separate physical-call accounting.

    Every attempted adaptive evaluation consumes one slot, including failures
    and repeats. Cached repeats consume a logical slot but no physical call;
    they reveal no new label. Initial imports never issue physical calls here.
    """

    protocol: ProxyProtocol
    initial_manifest: Mapping[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _seen: set[str] = field(default_factory=set, init=False)
    _cached: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        verify_initial_manifest(self.initial_manifest, self.protocol)
        self._seen = {r["sequence"] for r in self.initial_manifest["records"]}
        self._cached = set(self._seen)

    @property
    def remaining(self) -> int:
        return self.protocol.additional_evaluations - len(self.events)

    @property
    def logical_evaluations(self) -> int:
        return self.protocol.initial_size + len(self.events)

    @property
    def physical_oracle_calls(self) -> int:
        return sum(event["physical_oracle_call"] for event in self.events)

    def charge(
        self,
        peptide_id: str,
        sequence: str,
        *,
        status: str = "success",
        physical_oracle_call: bool = True,
    ) -> dict[str, Any]:
        if not self.remaining:
            raise ValueError("Additional evaluation budget exhausted")
        if status not in {"success", "failed", "duplicate"}:
            raise ValueError("Unknown evaluation status")
        if not peptide_id or not sequence:
            raise ValueError("Each attempted evaluation requires an identifier and sequence")
        repeat = sequence in self._seen
        if status == "duplicate" and not repeat:
            raise ValueError("Cannot mark an unseen sequence as duplicate")
        if type(physical_oracle_call) is not bool:
            raise ValueError("physical_oracle_call must be boolean")
        if not physical_oracle_call and sequence not in self._cached:
            raise ValueError("Only a successfully scored peptide can use the cache")
        event = {
            "index": len(self.events),
            "peptide_id": peptide_id,
            "sequence": sequence,
            "status": status,
            "repeat": repeat,
            "physical_oracle_call": physical_oracle_call,
        }
        self.events.append(event)
        self._seen.add(sequence)
        if status == "success":
            self._cached.add(sequence)
        return dict(event)

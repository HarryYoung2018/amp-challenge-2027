"""Bounded native evolutionary records; no oracle or scientific authority."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass

from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_proposals import NativeProposalTrace
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_sequence,
    hash_string,
)

EVOLUTION_CONFIG_SHA256 = "e1d7e314fec9f66580c07e07b6bcd98e4ef4b32f457730b889e4cd6455f0f324"
NO_COUNTERFACTUAL_CONFIG_PATH = "configs/diffusion/native_no_counterfactual_credit_v2.toml"
NO_COUNTERFACTUAL_CONFIG_SHA256 = "a7299a5c73f8f3550715d604066e09cd01125298ad6b78af15afd326ed99488d"
VARIANTS = (
    "full",
    "no_spectral",
    "no_counterfactual",
    "singleton_kg",
    "no_endpoint",
    "no_kl",
)
OPERATORS = ("single_site", "quarter_remask", "half_remask", "full_regeneration")
DIRECTIONS = ((1.0, 0.0), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (0.0, 1.0))


def evolution_configuration_sha256(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown native evolutionary configuration variant")
    return (
        NO_COUNTERFACTUAL_CONFIG_SHA256
        if variant == "no_counterfactual"
        else EVOLUTION_CONFIG_SHA256
    )


@dataclass(frozen=True, slots=True)
class EvolutionVariant:
    name: str = "full"

    def __post_init__(self):
        if self.name not in VARIANTS:
            raise ValueError("unknown native evolutionary variant")

    @property
    def representation(self):
        return (
            "esm320_plus_normalized_length"
            if self.name == "no_spectral"
            else "esm320_plus_normalized_length_plus_spectral32"
        )

    @property
    def kl_enforced(self):
        return self.name != "no_kl"


class EvolutionBudgetExceeded(TimeoutError):
    pass


class EvolutionNoEligibleParent(ValueError):
    pass


class EvolutionDeadline:
    """A non-resetting local cap within the controller-owned absolute clock."""

    def __init__(self, outer_deadline: float):
        self.started = time.monotonic()
        if not math.isfinite(outer_deadline) or outer_deadline > self.started + 7200:
            raise ValueError("invalid outer scientific deadline")
        self.deadline = min(self.started + 180, outer_deadline)
        self.checkpoints: list[tuple[str, float]] = []

    def check(self, phase: str):
        now = time.monotonic()
        self.checkpoints.append((phase, now - self.started))
        if now >= self.deadline:
            raise EvolutionBudgetExceeded("native evolutionary deadline exhausted")


@dataclass(frozen=True, slots=True)
class EvolutionBranch:
    triple: str
    branch: int
    parent: str
    charged_descendants: int = 0
    useful_descendants: int = 0
    credit: float = 1.0

    def __post_init__(self):
        if (
            not canonical_sequence(self.parent)
            or type(self.branch) is not int
            or not 0 <= self.branch < 4
        ):
            raise ValueError("evolutionary branch parent/index differs")
        if (
            any(
                type(n) is not int or n < 0
                for n in (self.charged_descendants, self.useful_descendants)
            )
            or self.useful_descendants > self.charged_descendants
        ):
            raise ValueError("branch charged yield census differs")
        if not math.isfinite(self.credit) or not 0.25 <= self.credit <= 4:
            raise ValueError("branch counterfactual credit differs")


@dataclass(frozen=True, slots=True)
class EvolutionAllocation:
    stage: int
    triple: str
    branches: tuple[EvolutionBranch, ...]
    scores: tuple[float, ...]
    counts: tuple[int, ...]
    parent_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvolutionContrast:
    mean: float
    variance: float
    advantage: float
    absolute_mean: float
    absolute_risk: float
    feasible: bool
    accepted: bool


@dataclass(frozen=True, slots=True)
class EvolutionAttempt:
    ordinal: int
    stage: int
    triple: str
    branch: int
    lineage_parent: str
    operator: str
    operator_log_probability: float
    length_log_probability: float
    trace: NativeProposalTrace
    behavior_version: int
    rejection: str | None
    first_seen_ordinal: int
    contrast: EvolutionContrast | None = None
    posterior_mean: tuple[float, float] | None = None
    thompson_value: float | None = None
    max_thompson_value: float | None = None

    @property
    def sha256(self):
        return _json_hash(asdict(self))

    @property
    def complete_conditional_log_probability(self):
        return (
            self.operator_log_probability
            + self.length_log_probability
            + self.trace.augmented_path_log_probability
        )


@dataclass(frozen=True, slots=True)
class EvolutionWave:
    round_index: int
    variant: str
    history_sha256: str
    semantic_history_sha256: str
    posterior_sha256: str
    feature_binding_sha256: str
    source_sha256: str
    policy_identities: tuple[tuple[str, str], ...]
    frozen_draws: tuple[tuple[float, ...], ...]
    allocations: tuple[EvolutionAllocation, ...]
    attempts: tuple[EvolutionAttempt, ...]
    shortlist_ordinals: tuple[int, ...]
    feature_events: tuple[dict, ...]
    status: str
    timing: tuple[tuple[str, float], ...]
    failure: str | None = None
    configuration_sha256: str = EVOLUTION_CONFIG_SHA256
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False
    max_rounds: int = 28
    prospective_protocol_sha256: str | None = None

    @property
    def sha256(self):
        payload = asdict(self)
        if self.max_rounds == 28 and self.prospective_protocol_sha256 is None:
            payload.pop("max_rounds")
            payload.pop("prospective_protocol_sha256")
        return _json_hash(payload)

    def check(self):
        if (
            self.variant not in VARIANTS
            or type(self.round_index) is not int
            or type(self.max_rounds) is not int
            or self.max_rounds <= 0
            or not 1 <= self.round_index <= self.max_rounds
            or (self.max_rounds != 28 and not hash_string(self.prospective_protocol_sha256))
            or (
                self.prospective_protocol_sha256 is not None
                and not hash_string(self.prospective_protocol_sha256)
            )
            or self.configuration_sha256 != evolution_configuration_sha256(self.variant)
            or self.scientific_evidence_accepted is not False
            or self.production_input_eligible is not False
        ):
            raise ValueError("evolutionary wave source/variant/evidence differs")
        if any(
            not hash_string(value)
            for value in (
                self.history_sha256,
                self.semantic_history_sha256,
                self.posterior_sha256,
                self.feature_binding_sha256,
                self.source_sha256,
            )
        ):
            raise ValueError("evolutionary wave binding differs")
        if (
            len(self.attempts) > 480
            or len(self.shortlist_ordinals) > 256
            or len(set(self.shortlist_ordinals)) != len(self.shortlist_ordinals)
            or any(
                type(index) is not int or not 0 <= index < len(self.attempts)
                for index in self.shortlist_ordinals
            )
            or self.status
            not in (
                "complete",
                "stopped_deadline_partial_wave",
                "stopped_numerical_integrity_partial_wave",
                "stopped_no_eligible_parent",
            )
        ):
            raise ValueError("evolutionary wave attempt/shortlist cap differs")
        if len(str(asdict(self)).encode()) > 128 * 1024**2:
            raise ValueError("evolutionary wave serialization budget exceeded")


def semantic_history(history, eligible_charged_ids=None):
    """Retain charged identity/status, never randomize from excluded values."""
    return _json_hash(
        [
            history.seed,
            history.round_index,
            history.objective_context_sha256,
            [
                (
                    row.charge_index,
                    row.sequence,
                    row.status,
                    row.objectives
                    if eligible_charged_ids is None
                    or hashlib.sha256(row.sequence.encode("ascii")).hexdigest()
                    in eligible_charged_ids
                    else None,
                )
                for row in history.observations
            ],
        ]
    )

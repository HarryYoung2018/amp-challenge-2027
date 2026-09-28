"""Structural/plain-GA handoffs; external receipt and clock truth stay caller-owned."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from amp_challenge.evaluation.sequential_v2_seals import PhaseSeal
from amp_challenge.generators.search import peptide_ga_driver_v2 as legacy
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    NEW_CONTRACT_SHA256,
    EligibleGAPrefix,
    GAContextEligibility,
    eligible_implementation_sha256,
    eligible_source_identities,
    prepare_eligible_ga_input,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    digest,
    hash_string,
    parameters,
    require,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

CONFIG_SHA256 = "5d43e14337bf5be93524a4cef2c7c39d6d3dbf6d10366fd7aad60d1a0517e8ff"
ARTIFACT = "context_eligible_peptide_ga_driver_phase_v3"
PAYLOADS = ("context.json", "history.json", "eligibility.json", "prefix.json", "selection.json")
MAX_PHASES = 128
MAX_PHASE_BYTES = 256 * 1024**2
MAX_RUN_BYTES = 512 * 1024**2
STATUSES = (
    "ready",
    "paused_incomplete_wave",
    "paused_prefix",
    "abstained_no_eligible_parent",
    "abstained_incomplete_prefix",
    "abstained_insufficient_seats",
    "stopped_deadline",
    "budget_complete_pending_controller_terminal",
)
TERMINAL_STATUSES = frozenset(STATUSES[3:])


def finite_clock(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid, "eligible driver clock/deadline must be a finite exact int/float")
    return value


def implementation_sha256():
    """Compose actual accepted dependency inventories with all new runtime bytes."""
    root = Path(__file__).resolve().parents[4]
    paths = [
        "configs/search/eligible_peptide_ga_driver_v3.toml",
        "src/amp_challenge/evaluation/sequential_v2_seals.py",
    ] + [
        "src/amp_challenge/generators/search/" + name + ".py"
        for name in (
            "peptide_ga_eligible_driver_v3_records",
            "peptide_ga_eligible_driver_v3",
            "peptide_ga_eligible_driver_v3_verify",
            "peptide_ga_driver_v2",
            "peptide_ga_selection_policy_impl_v1",
            "peptide_ga_tunable_v2_verify",
        )
    ]
    identities = eligible_source_identities()
    identities.update({name: digest((root / name).read_bytes()) for name in paths})
    require(identities[paths[0]] == CONFIG_SHA256, "eligible driver config bytes differ")
    return digest(canonical_json_bytes(identities))


@dataclass(frozen=True, slots=True)
class EligibleGADriverContext:
    run_id: str
    seed: int
    configuration_id: str
    objective_context_sha256: str
    oracle_bundle_sha256: str
    history_provider_sha256: str
    implementation_sha256: str
    training_sequence_keys: tuple[str, ...]
    kernel_source_sha256: str
    eligibility_provider_sha256: str
    eligibility_source_sha256: str
    deadline_monotonic: float
    clock_epoch_id: str
    kernel_contract_sha256: str = NEW_CONTRACT_SHA256
    driver_contract_sha256: str = CONFIG_SHA256

    def __post_init__(self):
        for value in (self.run_id, self.clock_epoch_id):
            require(
                type(value) is str
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is not None,
                "eligible driver run/clock epoch identity differs",
            )
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "driver seed differs")
        parameters(self.configuration_id)
        require(
            all(
                hash_string(value)
                for value in (
                    self.objective_context_sha256,
                    self.oracle_bundle_sha256,
                    self.history_provider_sha256,
                    self.implementation_sha256,
                    self.kernel_source_sha256,
                    self.eligibility_provider_sha256,
                    self.eligibility_source_sha256,
                )
            ),
            "eligible driver source/context identity differs",
        )
        require(
            self.kernel_contract_sha256 == NEW_CONTRACT_SHA256
            and self.driver_contract_sha256 == CONFIG_SHA256,
            "eligible driver contract differs",
        )
        require(
            type(self.training_sequence_keys) is tuple
            and len(self.training_sequence_keys) <= 65536
            and all(hash_string(value) for value in self.training_sequence_keys)
            and tuple(sorted(set(self.training_sequence_keys))) == self.training_sequence_keys,
            "eligible driver training keys differ",
        )
        finite_clock(self.deadline_monotonic)


class EligibilityReconstructor(Protocol):
    """No default issuer: externally authenticate history and reconstruct applicability.

    The attribute is a caller-pinned identity, not proof of implementation truth.
    This callback must read existing receipts, not fit or make oracle calls.
    """

    provider_sha256: str

    def __call__(self, history: VerifiedHistorySnapshot) -> GAContextEligibility: ...


@dataclass(frozen=True, slots=True)
class EligibleGADriverResult:
    """Historical record; only execute's timely return is a runtime handoff."""

    phase_sha256: str
    status: str
    round_index: int
    charged_count: int
    history_sha256: str
    eligibility_receipt_sha256: str
    prefix_sha256: str | None
    selected_prefix_positions: tuple[int, ...]
    selected_sequences: tuple[str, ...]

    @property
    def method_seats(self):
        return self.selected_sequences if self.status == "ready" else ()


@dataclass(frozen=True, slots=True)
class ReconstructedEligibleGAPhase:
    seal: PhaseSeal
    history: VerifiedHistorySnapshot
    eligibility: GAContextEligibility
    prefix: EligibleGAPrefix | None
    result: EligibleGADriverResult
    physical_bytes: int


class EligibleGADriverFailure(RuntimeError):
    """Terminal failed handoff: retain available bytes, never seats or fresh quota."""

    def __init__(self, message, *, prefix_evidence=None, published_phase_sha256=None):
        super().__init__(message)
        self.prefix_evidence = prefix_evidence
        self.published_phase_sha256 = published_phase_sha256
        self.run_terminal = True

    @property
    def method_seats(self):
        return ()


def check_sources(context):
    require(type(context) is EligibleGADriverContext, "eligible driver context type differs")
    context.__post_init__()
    require(
        context.implementation_sha256 == implementation_sha256()
        and context.kernel_source_sha256 == eligible_implementation_sha256(),
        "eligible driver actual source bytes differ",
    )


def check_resolver(context, resolver):
    require(
        callable(resolver)
        and getattr(resolver, "provider_sha256", None) == context.eligibility_provider_sha256,
        "eligible driver resolver provider differs",
    )


def resolve_eligibility(context, history, resolver):
    require(type(history) is VerifiedHistorySnapshot, "eligible driver history type differs")
    legacy._validate_history(context, history)
    raw = canonical_json_bytes(asdict(history))
    check_resolver(context, resolver)
    expected = resolver(history)
    check_resolver(context, resolver)
    require(canonical_json_bytes(asdict(history)) == raw, "resolver changed raw history")
    require(type(expected) is GAContextEligibility, "eligible driver authority type differs")
    expected.__post_init__()
    # This also enforces exact successful-ID membership, without a default or trim.
    kernel_input(context, history, expected)
    return expected


def expectation_pins(context, history, expected):
    return {
        "expected_history_sha256": history.sha256,
        "expected_objective_context_sha256": context.objective_context_sha256,
        "expected_eligibility_source_sha256": context.eligibility_source_sha256,
        "expected_eligible_query_ids": expected.query_ids,
    }


def kernel_input(context, history, expected):
    return prepare_eligible_ga_input(
        history,
        expected,
        configuration_id=context.configuration_id,
        training_sequence_keys=context.training_sequence_keys,
        **expectation_pins(context, history, expected),
    )


def decode_eligibility(document):
    require(
        type(document) is dict and type(document.get("query_ids")) is list, "authority JSON differs"
    )
    ids = document["query_ids"]
    require(
        all(type(value) is str for value in ids) and ids == sorted(set(ids)), "authority IDs differ"
    )
    return GAContextEligibility(**{**document, "query_ids": frozenset(ids)})


def decode_prefix(document):
    if document is None:
        return None
    require(type(document) is dict, "eligible driver prefix JSON differs")
    return EligibleGAPrefix(**{**document, "prefix": legacy._batch(document["prefix"])})


def validate_growth(prior, history, charged_count):
    if prior:
        require(prior[-1].result.status not in TERMINAL_STATUSES, "eligible driver is terminal")
    require(
        all(
            row.history.sha256 == history.sha256
            for row in prior
            if row.history.round_index == history.round_index and row.prefix is not None
        ),
        "eligible driver round history changed after its retained numerical prefix",
    )
    # Unchanged ordered history/denominator/14+2 seat checks. No numerical helper.
    legacy._validate_growth(prior, history, charged_count)

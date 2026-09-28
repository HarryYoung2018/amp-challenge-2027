"""Strict loader for the frozen evolutionary/KL successor protocol v2.

This module is intentionally independent of the v1 protocol, controller, and
reporter.  It accepts exactly one byte sequence and projects only the fields
needed by the blocked successor runtime scaffold.  It authorizes no execution,
oracle call, scientific claim, hidden confirmation, or production action.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256 = (
    "579d071fac1d70bf514761ff98e6d2019730fbdad484e548f9b672fda1a001af"
)
SUCCESSOR_PROTOCOL_V2_RELATIVE_PATH = Path(
    "configs/search/evolutionary_kl_successor_protocol_v2.toml"
)
SCREEN_SEEDS_V2 = (17, 42, 91, 137, 271)
METHOD_IDS_V2 = (
    "tuned_peptide_ga",
    "categorical_diffusion_posthoc",
    "diffusion_reward_kl_no_search",
    "arcadiamp_style_iterative_d3pm",
    "tr2d2_style_tree_offpolicy",
    "mp2d_style_inference_search",
    "ga_endpoint_distillation_no_kg",
    "counterfactual_softkg_evolutionary_diffusion",
)
ABLATION_IDS_V2 = (
    "ablation_no_spectral_representation",
    "ablation_no_counterfactual_credit",
    "ablation_singleton_kg",
    "ablation_no_endpoint_distillation",
    "ablation_no_kl_controls",
)
CONFIGURATION_IDS_V2 = (*METHOD_IDS_V2, *ABLATION_IDS_V2)
FULL_METHOD_ID_V2 = "counterfactual_softkg_evolutionary_diffusion"


class SuccessorProtocolV2Error(ValueError):
    """Raised when the frozen successor protocol is absent or differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorProtocolV2Error(message)


def _exact_bool(value: object, expected: bool, *, label: str) -> None:
    _require(type(value) is bool and value is expected, f"{label} differs")


def _exact_int(value: object, expected: int, *, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} differs")


def _table(value: object, *, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a TOML table")
    return value  # type: ignore[return-value]


def _array(value: object, *, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a TOML array")
    return value  # type: ignore[return-value]


def _read_regular_file_no_follow(path: Path) -> bytes:
    _require(isinstance(path, Path), "protocol path must be pathlib.Path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SuccessorProtocolV2Error("protocol path is unavailable or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "protocol path is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    _require(bool(payload), "protocol file is empty")
    return payload


@dataclass(frozen=True, slots=True)
class SuccessorProtocolV2:
    """Typed projection of the exact frozen protocol bytes."""

    protocol_sha256: str
    screen_seeds: tuple[int, ...]
    method_ids: tuple[str, ...]
    ablation_ids: tuple[str, ...]
    common_initial_calls: int
    adaptive_batches: int
    calls_per_batch: int
    method_seats_per_batch: int
    reserve_seats_per_batch: int
    total_calls_per_run: int
    scientific_wall_seconds_per_run: int
    bootstrap_samples: int
    bootstrap_seed: int
    ablation_positive_pairs_required: int
    ablation_median_margin: float
    execution_authorized: Literal[False] = False
    oracle_calls_authorized: Literal[False] = False
    scientific_evidence_accepted: Literal[False] = False
    automatic_production_eligible: Literal[False] = False
    biological_superiority_claim_allowed: Literal[False] = False
    hidden_confirmation_available: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.protocol_sha256) is str
            and self.protocol_sha256 == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
            "successor protocol digest differs",
        )
        _require(
            type(self.screen_seeds) is tuple and self.screen_seeds == SCREEN_SEEDS_V2,
            "screen seeds differ",
        )
        _require(
            type(self.method_ids) is tuple and self.method_ids == METHOD_IDS_V2,
            "method inventory differs",
        )
        _require(
            type(self.ablation_ids) is tuple and self.ablation_ids == ABLATION_IDS_V2,
            "ablation inventory differs",
        )
        exact_integers = (
            (self.common_initial_calls, 64, "common initial calls"),
            (self.adaptive_batches, 28, "adaptive batches"),
            (self.calls_per_batch, 16, "calls per batch"),
            (self.method_seats_per_batch, 14, "method seats per batch"),
            (self.reserve_seats_per_batch, 2, "reserve seats per batch"),
            (self.total_calls_per_run, 512, "total calls per run"),
            (self.scientific_wall_seconds_per_run, 7200, "scientific wall seconds"),
            (self.bootstrap_samples, 10_000, "bootstrap samples"),
            (self.bootstrap_seed, 20_260_909, "bootstrap seed"),
            (
                self.ablation_positive_pairs_required,
                4,
                "ablation positive-pair requirement",
            ),
        )
        for value, expected, label in exact_integers:
            _exact_int(value, expected, label=label)
        _require(
            type(self.ablation_median_margin) is float and self.ablation_median_margin == 0.05,
            "ablation median margin differs",
        )
        for label, value in (
            ("execution authorization", self.execution_authorized),
            ("oracle-call authorization", self.oracle_calls_authorized),
            ("scientific-evidence acceptance", self.scientific_evidence_accepted),
            ("automatic-production eligibility", self.automatic_production_eligible),
            ("biological-superiority claim", self.biological_superiority_claim_allowed),
            ("hidden-confirmation availability", self.hidden_confirmation_available),
        ):
            _exact_bool(value, False, label=label)
        _require(
            self.total_calls_per_run
            == self.common_initial_calls + self.adaptive_batches * self.calls_per_batch,
            "logical call arithmetic differs",
        )
        _require(
            self.calls_per_batch == self.method_seats_per_batch + self.reserve_seats_per_batch,
            "adaptive seat arithmetic differs",
        )

    @property
    def configuration_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return (*self.method_ids, *self.ablation_ids)


def load_successor_protocol_v2(path: Path) -> SuccessorProtocolV2:
    """Load only the byte-pinned successor-v2 protocol from a regular file."""

    payload = _read_regular_file_no_follow(path)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    _require(
        observed_sha256 == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
        "successor protocol byte digest differs",
    )
    try:
        parsed = tomllib.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SuccessorProtocolV2Error("successor protocol is not strict UTF-8 TOML") from error
    _require(type(parsed) is dict, "successor protocol root must be a TOML table")

    _exact_int(parsed.get("schema_version"), 2, label="protocol schema version")
    _require(
        type(parsed.get("artifact")) is str
        and parsed["artifact"] == "evolutionary_kl_successor_protocol_v2",
        "protocol artifact differs",
    )
    _require(
        type(parsed.get("status")) is str
        and parsed["status"] == "predeclared_blocked_non_authorizing_corrected_before_execution",
        "protocol status differs",
    )
    for key in (
        "retroactive_reinterpretation_of_v1_allowed",
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "automatic_production_eligible",
        "biological_superiority_claim_allowed",
    ):
        _exact_bool(parsed.get(key), False, label=f"protocol {key}")
    _exact_int(parsed.get("required_screen_seed_count"), 5, label="required screen seed count")
    raw_seeds = _array(parsed.get("screen_seeds"), label="screen seeds")
    _require(tuple(raw_seeds) == SCREEN_SEEDS_V2, "protocol screen seeds differ")
    _require(
        parsed.get("hidden_confirmation_seed_status") == "not_yet_committed_or_revealed",
        "hidden confirmation is not unavailable",
    )

    design = _table(parsed.get("design"), label="design")
    resources = _table(parsed.get("resources"), label="resources")
    statistics = _table(parsed.get("statistics"), label="statistics")
    stopping = _table(parsed.get("stopping"), label="stopping")
    promotion = _table(parsed.get("promotion"), label="promotion")
    evidence = _table(parsed.get("evidence_boundaries"), label="evidence boundaries")
    prerequisites = _table(parsed.get("prerequisites"), label="prerequisites")
    methods = _array(parsed.get("methods"), label="methods")
    ablations = _array(parsed.get("ablations"), label="ablations")

    method_ids = tuple(_table(item, label="method").get("id") for item in methods)
    ablation_ids = tuple(_table(item, label="ablation").get("id") for item in ablations)
    _require(method_ids == METHOD_IDS_V2, "protocol method inventory differs")
    _require(ablation_ids == ABLATION_IDS_V2, "protocol ablation inventory differs")
    _require(
        all(value is False and type(value) is bool for value in prerequisites.values()),
        "a successor prerequisite is not false",
    )
    _require(
        all(value is False and type(value) is bool for value in evidence.values()),
        "an evidence boundary is not false",
    )
    _exact_bool(
        promotion.get("passing_this_protocol_authorizes_production"),
        False,
        label="protocol production authorization",
    )
    _exact_bool(
        resources.get("resumed_clock_may_reset"),
        False,
        label="protocol clock reset permission",
    )
    _exact_bool(
        stopping.get("algorithmic_failures_retained"),
        True,
        label="protocol algorithmic-failure retention",
    )
    _require(
        statistics.get("exact_sign_test_zero_difference_policy")
        == "discard_ties_and_use_n_equal_to_non_tied_pairs",
        "protocol exact-sign tie policy differs",
    )
    _require(
        type(statistics.get("exact_sign_test_all_ties_p_value")) is float
        and statistics["exact_sign_test_all_ties_p_value"] == 1.0,
        "protocol all-ties exact-sign value differs",
    )
    _require(
        type(resources.get("screen_a100_hour_ceiling")) is float
        and resources["screen_a100_hour_ceiling"] == 131.25,
        "protocol total screen A100-hour ceiling differs",
    )

    return SuccessorProtocolV2(
        protocol_sha256=observed_sha256,
        screen_seeds=tuple(raw_seeds),  # type: ignore[arg-type]
        method_ids=method_ids,  # type: ignore[arg-type]
        ablation_ids=ablation_ids,  # type: ignore[arg-type]
        common_initial_calls=design.get("common_initial_design_unique_calls"),  # type: ignore[arg-type]
        adaptive_batches=design.get("adaptive_batches"),  # type: ignore[arg-type]
        calls_per_batch=design.get("unique_calls_per_batch"),  # type: ignore[arg-type]
        method_seats_per_batch=design.get("method_controlled_seats_per_batch"),  # type: ignore[arg-type]
        reserve_seats_per_batch=design.get("prefrozen_common_random_reserve_seats_per_batch"),  # type: ignore[arg-type]
        total_calls_per_run=design.get("total_unique_calls_per_run"),  # type: ignore[arg-type]
        scientific_wall_seconds_per_run=resources.get("scientific_wall_seconds_per_run"),  # type: ignore[arg-type]
        bootstrap_samples=statistics.get("paired_bootstrap_samples"),  # type: ignore[arg-type]
        bootstrap_seed=statistics.get("paired_bootstrap_seed"),  # type: ignore[arg-type]
        ablation_positive_pairs_required=statistics.get(
            "screen_full_vs_each_core_ablation_positive_pairs_required"
        ),  # type: ignore[arg-type]
        ablation_median_margin=statistics.get("screen_full_vs_each_core_ablation_median_margin"),  # type: ignore[arg-type]
    )


def load_repository_successor_protocol_v2() -> SuccessorProtocolV2:
    """Load the repository copy relative to this installed source tree."""

    repository_root = Path(__file__).resolve().parents[3]
    return load_successor_protocol_v2(repository_root / SUCCESSOR_PROTOCOL_V2_RELATIVE_PATH)

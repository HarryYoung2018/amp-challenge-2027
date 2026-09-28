"""Transport-neutral handoff of already verified, charged observations.

The outer controller authenticates receipts and objective meaning. This module
checks handoff identity/shape only; a self-created instance is not oracle truth.
It has no oracle, fake-port, model, reserve schedule, or GA operator imports.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Protocol

from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ChargedObservation,
    canonical_json_bytes,
    digest,
    hash_string,
    require,
)


@dataclass(frozen=True, slots=True)
class VerifiedHistorySnapshot:
    """Externally verified history, possibly short because a wave is incomplete.

    Legacy defaults start with 64 initial charges and end at round 29 with 512.
    Prospective callers declare their initial count, batch size, and round cap;
    512 initial charges plus 64 batches of 16 end at round 65 with 1536.
    Only terminal observations appear, in charged seat order.
    Outstanding submissions remain the controller's responsibility and cannot
    be silently replaced by a driver. Objective dimensions retain the existing
    two finite [0,1] values; their scientific meaning belongs to context_sha256.
    """

    run_id: str
    seed: int
    round_index: int
    objective_context_sha256: str
    oracle_bundle_sha256: str
    previous_wave_head_sha256: str
    observations: tuple[ChargedObservation, ...]
    receipt_sha256: str
    initial_charge_count: int = 64
    max_rounds: int = 28
    charges_per_round: int = 16

    def __post_init__(self) -> None:
        require(
            all(
                type(value) is int and value > 0
                for value in (self.initial_charge_count, self.max_rounds, self.charges_per_round)
            ),
            "history budget must contain positive integer counts",
        )
        require(
            type(self.run_id) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.run_id) is not None,
            "history run identity differs",
        )
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "history seed differs")
        require(
            type(self.round_index) is int and 1 <= self.round_index <= self.max_rounds + 1,
            "history round differs",
        )
        require(
            all(
                hash_string(value)
                for value in (
                    self.objective_context_sha256,
                    self.oracle_bundle_sha256,
                    self.previous_wave_head_sha256,
                    self.receipt_sha256,
                )
            ),
            "history source/context/receipt binding differs",
        )
        require(
            type(self.observations) is tuple
            and len(self.observations) <= self.expected_charge_count,
            "history exceeds the expected charged denominator",
        )
        for index, row in enumerate(self.observations):
            require(type(row) is ChargedObservation, "history observation type differs")
            row.__post_init__()
            require(row.charge_index == index, "history charge order is not contiguous")
        require(
            len({row.query_id for row in self.observations}) == len(self.observations)
            and len({row.sequence for row in self.observations}) == len(self.observations),
            "history contains duplicate charged identities/sequences",
        )

    @property
    def expected_charge_count(self) -> int:
        return self.initial_charge_count + self.charges_per_round * (self.round_index - 1)

    @property
    def total_charge_budget(self) -> int:
        return self.initial_charge_count + self.charges_per_round * self.max_rounds

    @property
    def complete(self) -> bool:
        return len(self.observations) == self.expected_charge_count

    @property
    def sha256(self) -> str:
        payload = asdict(self)
        legacy = (self.initial_charge_count, self.max_rounds, self.charges_per_round) == (
            64,
            28,
            16,
        )
        if legacy:
            for key in ("initial_charge_count", "max_rounds", "charges_per_round"):
                payload.pop(key)
        domain = (
            b"amp/verified-charged-history/v1\0" if legacy else b"amp/verified-charged-history/v2\0"
        )
        return digest(domain + canonical_json_bytes(payload))


class VerifiedHistoryCallback(Protocol):
    """Controller-owned authentication boundary, not an oracle transport.

    provider_sha256 is the outer controller's fixed provider identity. The
    consumer checks it before and after invocation; it does not certify that
    identity's implementation or the truth of its output. Concrete providers
    must bind real verified receipts before this interface is used scientifically.
    """

    provider_sha256: str

    def __call__(
        self, expected_previous_head_sha256: str, expected_charge_count: int
    ) -> VerifiedHistorySnapshot: ...


def read_verified_history(
    callback: VerifiedHistoryCallback,
    *,
    provider_sha256: str,
    run_id: str,
    seed: int,
    round_index: int,
    objective_context_sha256: str,
    oracle_bundle_sha256: str,
    previous_wave_head_sha256: str,
    initial_charge_count: int = 64,
    max_rounds: int = 28,
    charges_per_round: int = 16,
) -> VerifiedHistorySnapshot:
    """Check the fixed provider and exact expected run/source handoff."""
    require(
        hash_string(provider_sha256)
        and callable(callback)
        and getattr(callback, "provider_sha256", None) == provider_sha256,
        "verified-history callback provider differs",
    )
    require(
        all(
            type(value) is int and value > 0
            for value in (initial_charge_count, max_rounds, charges_per_round)
        )
        and type(round_index) is int
        and 1 <= round_index <= max_rounds + 1,
        "requested history round differs",
    )
    expected_count = initial_charge_count + charges_per_round * (round_index - 1)
    snapshot = callback(previous_wave_head_sha256, expected_count)
    require(
        getattr(callback, "provider_sha256", None) == provider_sha256,
        "verified-history callback provider changed during invocation",
    )
    require(type(snapshot) is VerifiedHistorySnapshot, "verified-history callback shape differs")
    snapshot.__post_init__()
    require(
        (
            snapshot.run_id,
            snapshot.seed,
            snapshot.round_index,
            snapshot.objective_context_sha256,
            snapshot.oracle_bundle_sha256,
            snapshot.previous_wave_head_sha256,
            snapshot.initial_charge_count,
            snapshot.max_rounds,
            snapshot.charges_per_round,
        )
        == (
            run_id,
            seed,
            round_index,
            objective_context_sha256,
            oracle_bundle_sha256,
            previous_wave_head_sha256,
            initial_charge_count,
            max_rounds,
            charges_per_round,
        ),
        "verified-history callback run/source/context/head differs",
    )
    return snapshot

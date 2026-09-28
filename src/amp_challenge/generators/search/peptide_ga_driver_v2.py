"""Durable GA method driver; real operators, no oracle/transport/production authority."""

from __future__ import annotations

import json
import math
import re
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import PhaseBuilder, PhaseSeal, verify_phase
from amp_challenge.generators.search.peptide_ga_selection_policy_impl_v1 import (
    controller_first_available_prefix_positions,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2 import generate_prefix
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    CONTRACT_SHA256,
    PREFIX_SIZE,
    ChargedObservation,
    GAAttempt,
    GAEdit,
    GAKernelBatch,
    GAKernelInput,
    canonical_json_bytes,
    digest,
    hash_string,
    parameters,
    require,
    sequence_key,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_verify import verify_prefix
from amp_challenge.generators.search.records import ProbabilityFactor
from amp_challenge.generators.search.verified_charged_history import (
    VerifiedHistoryCallback,
    VerifiedHistorySnapshot,
    read_verified_history,
)

ARTIFACT = "tunable_peptide_ga_driver_phase_v2"
PAYLOADS = ("context.json", "history.json", "prefix.json", "selection.json")
MAX_PHASES = 128
MAX_PHASE_BYTES = 256 * 1024**2
MAX_RUN_BYTES = 512 * 1024**2
STATUSES = (
    "ready",
    "paused_incomplete_wave",
    "paused_prefix",
    "abstained_no_successful_parent",
    "abstained_incomplete_prefix",
    "abstained_insufficient_seats",
    "stopped_deadline",
    "budget_complete_pending_controller_terminal",
)
TERMINAL_STATUSES = frozenset(STATUSES[3:])


def implementation_sha256() -> str:
    """Bind executable bytes, not a caller-provided implementation label."""
    root = Path(__file__).resolve().parents[4]
    search = "src/amp_challenge/generators/search/"
    paths = (
        "src/amp_challenge/evaluation/sequential_v2_seals.py",
        *(
            search + name
            for name in (
                "peptide_ga_driver_v2.py",
                "verified_charged_history.py",
                "peptide_ga.py",
                "peptide_ga_records.py",
                "peptide_ga_verifier.py",
                "records.py",
                "peptide_ga_tunable_v2.py",
                "peptide_ga_tunable_v2_records.py",
                "peptide_ga_tunable_v2_verify.py",
                "peptide_ga_selection_policy_impl_v1.py",
            )
        ),
    )
    return digest(
        canonical_json_bytes({name: digest((root / name).read_bytes()) for name in paths})
    )


@dataclass(frozen=True, slots=True)
class GADriverContext:
    run_id: str
    seed: int
    configuration_id: str
    objective_context_sha256: str
    oracle_bundle_sha256: str
    history_provider_sha256: str
    implementation_sha256: str
    training_sequence_keys: tuple[str, ...]
    kernel_contract_sha256: str = CONTRACT_SHA256

    def __post_init__(self) -> None:
        require(
            type(self.run_id) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.run_id) is not None,
            "driver run identity differs",
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
                )
            ),
            "driver source/context binding differs",
        )
        require(self.kernel_contract_sha256 == CONTRACT_SHA256, "driver kernel contract differs")
        require(
            type(self.training_sequence_keys) is tuple
            and len(self.training_sequence_keys) <= 65536
            and all(hash_string(key) for key in self.training_sequence_keys)
            and tuple(sorted(set(self.training_sequence_keys))) == self.training_sequence_keys,
            "driver training exclusion inventory differs",
        )


@dataclass(frozen=True, slots=True)
class PrivateGACollisions:
    """Controller-only input. Never serialize this into a public driver phase."""

    charged_sequence_keys: tuple[str, ...]
    upcoming_reserve_sequence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        for values, limit in (
            (self.charged_sequence_keys, 512),
            (self.upcoming_reserve_sequence_keys, 56),
        ):
            require(
                type(values) is tuple
                and len(values) <= limit
                and all(hash_string(key) for key in values)
                and tuple(sorted(set(values))) == values,
                "private collision inventory differs",
            )
        require(
            not set(self.charged_sequence_keys).intersection(self.upcoming_reserve_sequence_keys),
            "upcoming reserve was already charged",
        )


@dataclass(frozen=True, slots=True)
class GADriverResult:
    phase_sha256: str
    status: str
    round_index: int
    charged_count: int
    history_sha256: str
    prefix_sha256: str | None
    selected_prefix_positions: tuple[int, ...]
    selected_sequences: tuple[str, ...]

    @property
    def method_seats(self) -> tuple[str, ...]:
        return self.selected_sequences if self.status == "ready" else ()


@dataclass(frozen=True, slots=True)
class ReconstructedGAPhase:
    seal: PhaseSeal
    history: VerifiedHistorySnapshot
    prefix: GAKernelBatch | None
    result: GADriverResult


def _json(payload: bytes):
    value = json.loads(payload)
    require(canonical_json_bytes(value) == payload, "driver payload is not canonical JSON")
    return value


def _history(value: dict) -> VerifiedHistorySnapshot:
    document = dict(value)
    document["observations"] = tuple(
        ChargedObservation(
            **{**row, "objectives": None if row["objectives"] is None else tuple(row["objectives"])}
        )
        for row in document["observations"]
    )
    return VerifiedHistorySnapshot(**document)


def _batch(value: dict | None) -> GAKernelBatch | None:
    if value is None:
        return None
    document = dict(value)
    attempts = []
    for raw in document["attempts"]:
        row, edit = dict(raw), dict(raw["edit"])
        edit["parent_sequence_keys"] = tuple(edit["parent_sequence_keys"])
        edit["description"] = tuple(edit["description"])
        edit["probability_factors"] = tuple(
            ProbabilityFactor(**x) for x in edit["probability_factors"]
        )
        row["edit"] = GAEdit(**edit)
        attempts.append(GAAttempt(**row))
    document["attempts"] = tuple(attempts)
    document["accepted_sequences"] = tuple(document["accepted_sequences"])
    return GAKernelBatch(**document)


def _kernel_input(context: GADriverContext, history: VerifiedHistorySnapshot) -> GAKernelInput:
    return GAKernelInput(
        context.configuration_id,
        context.seed,
        history.round_index,
        history.observations,
        context.training_sequence_keys,
    )


def _validate_history(context: GADriverContext, history: VerifiedHistorySnapshot) -> None:
    history.__post_init__()
    require(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
        )
        == (
            context.run_id,
            context.seed,
            context.objective_context_sha256,
            context.oracle_bundle_sha256,
        ),
        "persisted history source/context differs",
    )
    require(
        not {sequence_key(row.sequence) for row in history.observations}.intersection(
            context.training_sequence_keys
        ),
        "released history contains forbidden training overlap",
    )


def _validate_growth(
    prior: tuple[ReconstructedGAPhase, ...], history: VerifiedHistorySnapshot, charged_count: int
) -> None:
    require(
        type(charged_count) is int and len(history.observations) <= charged_count <= 512,
        "charged count does not cover released terminal observations",
    )
    current_ready = any(
        row.history.round_index == history.round_index and row.result.status == "ready"
        for row in prior
    )
    require(
        charged_count <= min(512, history.expected_charge_count + (16 if current_ready else 0)),
        "charged inventory exceeds the current or already-issued wave",
    )
    if not prior:
        require(history.round_index == 1, "driver must begin with the initial design")
        return
    previous = prior[-1]
    require(previous.result.status not in TERMINAL_STATUSES, "driver is already terminal")
    require(
        previous.history.round_index <= history.round_index <= previous.history.round_index + 1,
        "driver round regressed or skipped",
    )
    require(charged_count >= previous.result.charged_count, "charged denominator regressed")
    require(
        history.observations[: len(previous.history.observations)] == previous.history.observations,
        "released charged history was rewritten or shortened",
    )
    if history.round_index == previous.history.round_index and previous.history.complete:
        require(
            history.previous_wave_head_sha256 == previous.history.previous_wave_head_sha256,
            "same-round previous wave head changed",
        )
    if history.round_index > 1:
        selections = [
            row
            for row in prior
            if row.history.round_index == history.round_index - 1 and row.result.status == "ready"
        ]
        require(bool(selections), "next history lacks its recorded previous method seats")
        seats = selections[-1].result.selected_sequences
        if history.complete:
            require(
                history.previous_wave_head_sha256
                != selections[-1].history.previous_wave_head_sha256,
                "completed next round did not bind a new controller wave head",
            )
        begin = 64 + 16 * (history.round_index - 2)
        released = tuple(row.sequence for row in history.observations[begin : begin + 14])
        require(released == seats[: len(released)], "released method charge seat order differs")


def _validate_selection(
    history: VerifiedHistorySnapshot, prefix: GAKernelBatch | None, document: dict
) -> None:
    require(
        set(document)
        == {
            "status",
            "round_index",
            "charged_count",
            "history_sha256",
            "prefix_sha256",
            "selected_prefix_positions",
            "selected_sequences",
        },
        "driver selection payload shape differs",
    )
    status = document["status"]
    require(
        status in STATUSES
        and document["round_index"] == history.round_index
        and document["history_sha256"] == history.sha256,
        "driver selection history/status binding differs",
    )
    require(
        document["prefix_sha256"] == (None if prefix is None else prefix.output_sha256),
        "driver selection prefix binding differs",
    )
    positions, sequences = document["selected_prefix_positions"], document["selected_sequences"]
    require(type(positions) is list and type(sequences) is list, "driver seat collection differs")
    if status == "ready":
        require(
            history.complete
            and history.round_index <= 28
            and document["charged_count"] == len(history.observations)
            and prefix is not None
            and prefix.status == "complete"
            and len(positions) == len(sequences) == 14
            and all(type(index) is int and 0 <= index < PREFIX_SIZE for index in positions)
            and positions == sorted(set(positions)),
            "ready method seat inventory differs",
        )
        require(
            sequences == [prefix.accepted_sequences[index] for index in positions],
            "method seats do not reconstruct from recorded prefix",
        )
    else:
        require(not positions and not sequences, "non-ready state exposes method seats")
    if status == "paused_incomplete_wave":
        require(
            not history.complete or document["charged_count"] > len(history.observations),
            "incomplete-wave pause lacks outstanding observations",
        )
    elif status == "paused_prefix":
        require(prefix is not None and prefix.status == "in_progress", "prefix pause differs")
    elif status == "abstained_no_successful_parent":
        require(
            history.complete
            and history.round_index <= 28
            and not any(row.status == "successful" for row in history.observations),
            "empty-parent abstention differs",
        )
    elif status == "abstained_incomplete_prefix":
        require(
            prefix is not None and prefix.status == "attempt_cap_exhausted",
            "attempt-cap abstention differs",
        )
    elif status == "abstained_insufficient_seats":
        require(prefix is not None and prefix.status == "complete", "seat abstention differs")
    elif status == "budget_complete_pending_controller_terminal":
        require(
            history.round_index == 29 and history.complete and document["charged_count"] == 512,
            "final logical denominator differs",
        )


def reconstruct_driver(root: Path, context: GADriverContext) -> tuple[ReconstructedGAPhase, ...]:
    """Replay public prefixes/seats/history without loading future reserve identities.

    This does not independently authenticate the external callback's oracle truth
    or prove private vetoes without the controller's separate private schedule.
    """
    context.__post_init__()
    require(
        context.implementation_sha256 == implementation_sha256(),
        "driver implementation bytes differ",
    )
    root = Path(root)
    require(
        root.is_dir() and not root.is_symlink(), "driver root must be an existing real directory"
    )
    entries = sorted(root.iterdir())
    require(len(entries) <= MAX_PHASES, "driver phase count exceeded")
    records, cumulative_bytes = [], 0
    for index, path in enumerate(entries):
        require(
            path.name == f"phase-{index:06d}" and path.is_dir() and not path.is_symlink(),
            "driver phase inventory contains a gap, partial, or unexpected entry",
        )
        phase_bytes = 0
        for child in path.iterdir():
            metadata = child.lstat()
            require(
                child.name in (*PAYLOADS, "receipt.json", "SHA256SUMS")
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1,
                "driver phase has an unexpected nonregular payload",
            )
            phase_bytes += metadata.st_size
        cumulative_bytes += phase_bytes
        require(
            phase_bytes <= MAX_PHASE_BYTES and cumulative_bytes <= MAX_RUN_BYTES,
            "driver persistent byte budget exceeded",
        )
        predecessors = (
            {} if not records else {"previous_driver_phase": records[-1].seal.seal_sha256}
        )
        seal = verify_phase(
            path,
            expected_artifact=ARTIFACT,
            expected_payload_paths=PAYLOADS,
            expected_predecessor_seals=predecessors,
        )
        require(
            _json(seal.read_payload_bytes("context.json"))
            == json.loads(canonical_json_bytes(asdict(context))),
            "persisted driver context differs",
        )
        history = _history(_json(seal.read_payload_bytes("history.json")))
        prefix = _batch(_json(seal.read_payload_bytes("prefix.json")))
        document = _json(seal.read_payload_bytes("selection.json"))
        _validate_history(context, history)
        _validate_growth(tuple(records), history, document["charged_count"])
        if prefix is not None:
            verify_prefix(prefix, _kernel_input(context, history))
            earlier_prefixes = [
                row.prefix
                for row in records
                if row.history.sha256 == history.sha256 and row.prefix is not None
            ]
            if earlier_prefixes:
                earlier = earlier_prefixes[-1]
                require(
                    prefix.attempts[: len(earlier.attempts)] == earlier.attempts,
                    "persisted prefix cursor regressed or changed",
                )
        _validate_selection(history, prefix, document)
        result = GADriverResult(
            seal.seal_sha256,
            document["status"],
            history.round_index,
            document["charged_count"],
            history.sha256,
            document["prefix_sha256"],
            tuple(document["selected_prefix_positions"]),
            tuple(document["selected_sequences"]),
        )
        same_round_ready = [
            row
            for row in records
            if row.history.round_index == history.round_index and row.result.status == "ready"
        ]
        if result.status == "ready" and same_round_ready:
            require(
                result.selected_sequences == same_round_ready[-1].result.selected_sequences,
                "recorded method seats changed within a round",
            )
        records.append(ReconstructedGAPhase(seal, history, prefix, result))
    return tuple(records)


def _private_positions(prefix: GAKernelBatch, private: PrivateGACollisions) -> tuple[int, ...]:
    blocked = set(private.charged_sequence_keys) | set(private.upcoming_reserve_sequence_keys)
    return controller_first_available_prefix_positions(
        tuple(sequence_key(sequence) not in blocked for sequence in prefix.accepted_sequences),
        seat_count=14,
    )


def verify_private_composition(phase: ReconstructedGAPhase, private: PrivateGACollisions) -> None:
    """Optional controller-private audit; no private inventory is returned or persisted."""
    private.__post_init__()
    require(
        len(private.charged_sequence_keys) == phase.result.charged_count
        and {sequence_key(row.sequence) for row in phase.history.observations}.issubset(
            private.charged_sequence_keys
        ),
        "private audit charged history differs",
    )
    require(
        phase.prefix is not None and phase.prefix.status == "complete",
        "private audit lacks complete prefix",
    )
    if phase.result.status == "ready":
        require(
            _private_positions(phase.prefix, private) == phase.result.selected_prefix_positions,
            "private first-available composition differs",
        )
    elif phase.result.status == "abstained_insufficient_seats":
        try:
            _private_positions(phase.prefix, private)
        except ValueError:
            return
        raise ValueError("private inventory does not support insufficient-seat abstention")
    else:
        raise ValueError("phase has no private composition decision")


def execute_ga_step(
    root: Path,
    context: GADriverContext,
    history_callback: VerifiedHistoryCallback,
    *,
    round_index: int,
    previous_wave_head_sha256: str,
    private: PrivateGACollisions,
    deadline_monotonic: float,
    max_new_attempts: int = ATTEMPT_CAP,
    monotonic: Callable[[], float] = time.monotonic,
) -> GADriverResult:
    """Execute/persist a real operator step and return seats only after publication.

    No submission happens here. The caller must keep its original hard deadline,
    authenticate history, own private reserves, and charge every submitted request.
    """
    require(
        type(deadline_monotonic) in (float, int) and math.isfinite(deadline_monotonic),
        "driver deadline differs",
    )
    require(
        type(max_new_attempts) is int and 0 <= max_new_attempts <= ATTEMPT_CAP,
        "driver attempt chunk differs",
    )
    if monotonic() >= deadline_monotonic:
        raise TimeoutError("driver entered after the outer scientific deadline")
    context.__post_init__()
    private.__post_init__()
    prior = reconstruct_driver(root, context)
    if monotonic() >= deadline_monotonic:
        raise TimeoutError("driver reconstruction crossed the outer scientific deadline")
    if prior and prior[-1].result.status in TERMINAL_STATUSES:
        return prior[-1].result
    history = read_verified_history(
        history_callback,
        provider_sha256=context.history_provider_sha256,
        run_id=context.run_id,
        seed=context.seed,
        round_index=round_index,
        objective_context_sha256=context.objective_context_sha256,
        oracle_bundle_sha256=context.oracle_bundle_sha256,
        previous_wave_head_sha256=previous_wave_head_sha256,
    )
    _validate_history(context, history)
    charged_count = len(private.charged_sequence_keys)
    _validate_growth(prior, history, charged_count)
    require(
        {sequence_key(row.sequence) for row in history.observations}.issubset(
            private.charged_sequence_keys
        ),
        "private charged inventory omits released history",
    )
    require(
        not set(context.training_sequence_keys).intersection(private.charged_sequence_keys),
        "charged private inventory contains training overlap",
    )
    prefix, positions = None, ()
    if monotonic() >= deadline_monotonic:
        status = "stopped_deadline"
    elif not history.complete or charged_count > len(history.observations):
        status = "paused_incomplete_wave"
    elif round_index == 29:
        status = "budget_complete_pending_controller_terminal"
    elif not any(row.status == "successful" for row in history.observations):
        status = "abstained_no_successful_parent"
    else:
        matching = [
            row for row in prior if row.history.sha256 == history.sha256 and row.prefix is not None
        ]
        old = matching[-1].prefix if matching else None
        if old is not None and old.status != "in_progress":
            prefix = old
        else:
            prefix = generate_prefix(
                _kernel_input(context, history), resume=old, max_new_attempts=max_new_attempts
            )
        verify_prefix(prefix, _kernel_input(context, history))
        if monotonic() >= deadline_monotonic:
            status = "stopped_deadline"
        elif prefix.status == "in_progress":
            status = "paused_prefix"
        elif prefix.status == "attempt_cap_exhausted":
            status = "abstained_incomplete_prefix"
        else:
            try:
                positions = _private_positions(prefix, private)
                status = "ready"
            except ValueError:
                status = "abstained_insufficient_seats"
    sequences = (
        () if not positions else tuple(prefix.accepted_sequences[index] for index in positions)
    )
    document = {
        "status": status,
        "round_index": history.round_index,
        "charged_count": charged_count,
        "history_sha256": history.sha256,
        "prefix_sha256": None if prefix is None else prefix.output_sha256,
        "selected_prefix_positions": list(positions),
        "selected_sequences": list(sequences),
    }
    _validate_selection(history, prefix, document)
    existing_ready = [
        row
        for row in prior
        if row.history.round_index == history.round_index and row.result.status == "ready"
    ]
    if status == "ready" and existing_ready:
        require(
            sequences == existing_ready[-1].result.selected_sequences,
            "private recomposition would change already recorded method seats",
        )
    payloads = {
        "context.json": asdict(context),
        "history.json": asdict(history),
        "prefix.json": None if prefix is None else asdict(prefix),
        "selection.json": document,
    }
    encoded = {name: canonical_json_bytes(value) for name, value in payloads.items()}
    if prior and all(
        prior[-1].seal.read_payload_bytes(name) == payload for name, payload in encoded.items()
    ):
        if monotonic() >= deadline_monotonic:
            raise TimeoutError("driver idempotent result crossed the outer scientific deadline")
        return prior[-1].result
    require(len(prior) < MAX_PHASES, "driver phase count exceeded")
    prior_bytes = sum(len(payload) for phase in prior for _, payload in phase.seal.payload_bytes)
    current_bytes = sum(len(payload) for payload in encoded.values())
    require(
        current_bytes + 65536 <= MAX_PHASE_BYTES
        and prior_bytes + current_bytes + 65536 <= MAX_RUN_BYTES,
        "driver persistent byte budget exceeded",
    )
    if monotonic() >= deadline_monotonic and status != "stopped_deadline":
        document.update(
            status="stopped_deadline", selected_prefix_positions=[], selected_sequences=[]
        )
        encoded["selection.json"] = canonical_json_bytes(document)
    predecessors = {} if not prior else {"previous_driver_phase": prior[-1].seal.seal_sha256}
    with PhaseBuilder(
        Path(root) / f"phase-{len(prior):06d}", artifact=ARTIFACT, predecessor_seals=predecessors
    ) as builder:
        for name, payload in encoded.items():
            builder.write_bytes(name, payload)
        builder.publish(expected_payload_paths=PAYLOADS)
    result = reconstruct_driver(root, context)[-1].result
    if monotonic() >= deadline_monotonic:
        # Publication cannot mint extra wall time. A late READY phase is retained
        # as evidence but no seats escape; outer controller must stop this run.
        raise TimeoutError(
            "driver publication/reconstruction crossed the outer scientific deadline"
        )
    return result

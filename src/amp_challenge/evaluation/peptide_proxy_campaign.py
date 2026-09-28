"""Fail-stop frozen-oracle controller with explicit provider admission.

No legacy/native provider is silently adapted or registered here. Provider
admission is a caller-supplied evidence assertion, not a substitute for auditing
the implementation. The controller does not claim biological ground truth.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

from .peptide_proxy_protocol import ProxyProtocol, fingerprint, verify_initial_manifest

NATIVE_RELEASED_PROVIDERS: dict[str, type] = {}
_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ProviderCapabilities:
    protocol_sha256: str
    arm_name: str
    initial_size: int
    additional_evaluations: int
    constraint_scope: str
    ground_cost_id: str
    clip_ratio_width: float
    per_transition_total_variation_limit: float
    metric_threshold: float
    metric_units: str
    admission_evidence: str
    implementation_sha256: str


class MethodProvider(Protocol):
    capabilities: ProviderCapabilities

    def propose(self, history: Sequence[Mapping[str, Any]], batch_size: int) -> Sequence[str]:
        """Return exactly batch_size sequences; do not call the oracle."""


def _admit(protocol: ProxyProtocol, provider: MethodProvider) -> None:
    cap = provider.capabilities
    arm = next((arm for arm in protocol.arms if arm.name == cap.arm_name), None)
    if arm is None:
        raise ValueError("Provider arm is not declared")
    expected = (
        protocol.protocol_sha256,
        protocol.initial_size,
        protocol.additional_evaluations,
        protocol.constraint_scope,
        protocol.ground_cost_id,
        protocol.clip_ratio_width,
        protocol.per_transition_total_variation_limit,
        arm.threshold,
        arm.units,
    )
    actual = (
        cap.protocol_sha256,
        cap.initial_size,
        cap.additional_evaluations,
        cap.constraint_scope,
        cap.ground_cost_id,
        cap.clip_ratio_width,
        cap.per_transition_total_variation_limit,
        cap.metric_threshold,
        cap.metric_units,
    )
    if actual != expected:
        raise ValueError("Provider capabilities do not match approved budget or metric scope")
    if (
        not cap.admission_evidence
        or len(cap.implementation_sha256) != 64
        or any(c not in "0123456789abcdef" for c in cap.implementation_sha256)
    ):
        raise ValueError(
            "Provider requires explicit admission evidence and implementation fingerprint"
        )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Journal:
    def __init__(self, path: Path):
        self.stream = path.open("x", encoding="utf-8")
        self.previous = "0" * 64
        self.index = 0

    def append(self, kind: str, **payload: Any) -> None:
        body = {
            "event_index": self.index,
            "previous_sha256": self.previous,
            "kind": kind,
            **payload,
        }
        digest = fingerprint(body)
        self.stream.write(
            json.dumps({**body, "event_sha256": digest}, sort_keys=True, allow_nan=False) + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous = digest
        self.index += 1


def run_campaign(
    *,
    protocol: ProxyProtocol,
    initial_manifest: Mapping[str, Any],
    provider: MethodProvider,
    oracle: Callable[[str], float],
    oracle_id: str,
    oracle_sha256: str,
    output_dir: str | Path,
    max_seconds: float = 7200.0,
    excluded_sequences: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Execute exactly 1024 charged attempts or record a terminal failure.

    Failed oracle calls consume budget and are retained; no automatic retry is
    made. A repeated sequence consumes a logical evaluation even if its frozen
    score is cached. Invalid sequences also consume a slot, without an oracle
    call. Provider failures stop the run. Reusing an output directory is refused;
    interrupted runs cannot silently resume or replay uncertain oracle calls.
    """
    started = monotonic()
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")

    def check_deadline() -> None:
        if monotonic() - started >= max_seconds:
            raise TimeoutError("Original campaign wall-clock budget exhausted")

    verify_initial_manifest(initial_manifest, protocol)
    if type(excluded_sequences) is not frozenset:
        raise TypeError("Common exclusions must be a frozen sequence inventory")
    if any(row["sequence"] in excluded_sequences for row in initial_manifest["records"]):
        raise ValueError("Frozen initialization overlaps common exclusions")
    if (initial_manifest["oracle_id"], initial_manifest["oracle_sha256"]) != (
        oracle_id,
        oracle_sha256,
    ):
        raise ValueError("Oracle does not match the frozen initialization")
    _admit(protocol, provider)
    if not callable(oracle):
        raise TypeError("An explicit frozen scoring callable is required")
    destination = Path(output_dir)
    destination.mkdir(parents=False, exist_ok=False)
    _sync_directory(destination.parent)
    journal = _Journal(destination / "events.jsonl")
    _sync_directory(destination)
    initial = copy.deepcopy(initial_manifest)
    history = [{**row, "phase": "initial", "status": "success"} for row in initial["records"]]
    cache = {row["sequence"]: row["oracle_score"] for row in initial["records"]}
    charges = physical_calls = failures = cached_repeats = 0
    failure: dict[str, str] | None = None
    final_selection: Any = None
    try:
        journal.append(
            "initial_import",
            manifest=initial,
            provider_capabilities=vars(provider.capabilities),
            max_seconds=max_seconds,
            excluded_sequences=sorted(excluded_sequences),
        )
        while charges < protocol.additional_evaluations:
            requested = min(protocol.batch_size, protocol.additional_evaluations - charges)
            check_deadline()
            proposals = list(provider.propose(copy.deepcopy(history), requested))
            check_deadline()
            if len(proposals) != requested or any(not isinstance(s, str) for s in proposals):
                raise ValueError(
                    "Provider must return exactly the requested number of sequence strings"
                )
            for sequence in proposals:
                check_deadline()
                cached = sequence in cache
                valid = (
                    8 <= len(sequence) <= 50
                    and set(sequence) <= _ALPHABET
                    and sequence not in excluded_sequences
                )
                physical = bool(valid and not cached)
                # Receipt is durable before any oracle side effect.
                journal.append(
                    "charge",
                    evaluation_index=charges,
                    sequence=sequence,
                    physical_oracle_call=physical,
                    cached=cached,
                    elapsed_seconds=monotonic() - started,
                )
                index = charges
                charges += 1
                score: float | None = None
                error: str | None = None
                if not valid:
                    status, error = (
                        "failed",
                        (
                            "common_excluded_sequence"
                            if sequence in excluded_sequences
                            else "invalid_canonical_peptide"
                        ),
                    )
                elif cached:
                    status, score = "duplicate", cache[sequence]
                    cached_repeats += 1
                else:
                    check_deadline()
                    physical_calls += 1
                    try:
                        score = float(oracle(sequence))
                        if not math.isfinite(score):
                            raise ValueError("Oracle returned a nonfinite score")
                        cache[sequence] = score
                        status = "success"
                    except Exception as exc:
                        score = None
                        status, error = "failed", f"{type(exc).__name__}: {exc}"
                failures += status == "failed"
                row = {
                    "phase": "adaptive",
                    "evaluation_index": index,
                    "sequence": sequence,
                    "status": status,
                    "oracle_score": score,
                    "error": error,
                    "elapsed_seconds": monotonic() - started,
                    "late": monotonic() - started >= max_seconds,
                }
                journal.append("outcome", **row)
                history.append(row)
                check_deadline()
            observe = getattr(provider, "observe", None)
            if observe is not None:
                check_deadline()
                observe(copy.deepcopy(history))
                check_deadline()
        finalize = getattr(provider, "finalize", None)
        if finalize is not None:
            check_deadline()
            final_selection = finalize(copy.deepcopy(history))
            check_deadline()
            # Reject nonserializable/nonfinite output before terminal commit.
            fingerprint(final_selection)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    result = {
        "schema_version": 1,
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.protocol_sha256,
        "seed": initial["seed"],
        "arm_name": provider.capabilities.arm_name,
        "initial_manifest_sha256": initial["manifest_sha256"],
        "oracle_id": oracle_id,
        "oracle_sha256": oracle_sha256,
        "status": "failed" if failure else "complete",
        "initial_logical_evaluations": protocol.initial_size,
        "additional_charged_evaluations": charges,
        "total_logical_evaluations": protocol.initial_size + charges,
        "physical_adaptive_oracle_calls": physical_calls,
        "failed_evaluations": failures,
        "cached_repeats": cached_repeats,
        "failure": failure,
        "final_selection": final_selection,
        "claim_scope": "frozen_computational_proxy_only",
        "completed_proxy_execution": failure is None and charges == protocol.additional_evaluations,
        "provider_admission_is_caller_attested": True,
        "native_full_provider_registered": bool(NATIVE_RELEASED_PROVIDERS),
        "global_sequence_constraint_qualified": False,
    }
    result["max_seconds"] = max_seconds
    result["elapsed_seconds"] = monotonic() - started
    try:
        journal.append("terminal", result=result)
        result["terminal_event_sha256"] = journal.previous
        with (destination / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(destination)
    finally:
        journal.stream.close()
    return result


def verify_campaign(output_dir: str | Path, protocol: ProxyProtocol) -> dict[str, Any]:
    """Independently check the durable chain and budget, without oracle calls."""
    path = Path(output_dir)
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    if (
        not events
        or events[0].get("kind") != "initial_import"
        or events[-1].get("kind") != "terminal"
    ):
        raise ValueError("Missing initial or terminal journal event")
    previous = "0" * 64
    for index, event in enumerate(events):
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if (
            event.get("event_index") != index
            or event.get("previous_sha256") != previous
            or fingerprint(body) != event.get("event_sha256")
        ):
            raise ValueError("Journal chain mismatch")
        previous = event["event_sha256"]
    initial = events[0]["manifest"]
    verify_initial_manifest(initial, protocol)
    saved = json.loads((path / "result.json").read_text())
    result = events[-1]["result"]
    if saved != {**result, "terminal_event_sha256": previous}:
        raise ValueError("Terminal result mismatch")
    if (
        result["protocol_sha256"],
        result["seed"],
        result["initial_manifest_sha256"],
        result["oracle_id"],
        result["oracle_sha256"],
    ) != (
        protocol.protocol_sha256,
        initial["seed"],
        initial["manifest_sha256"],
        initial["oracle_id"],
        initial["oracle_sha256"],
    ):
        raise ValueError("Source identity mismatch")
    charges, outcomes = [], []
    pending = None
    for event in events[1:-1]:
        if event["kind"] == "charge":
            if pending is not None or event["evaluation_index"] != len(charges):
                raise ValueError("Charge ordering mismatch")
            charges.append(event)
            pending = event
        elif event["kind"] == "outcome":
            if pending is None or (event["evaluation_index"], event["sequence"]) != (
                pending["evaluation_index"],
                pending["sequence"],
            ):
                raise ValueError("Outcome does not match its charged evaluation")
            if event["status"] not in {"success", "duplicate", "failed"}:
                raise ValueError("Unknown outcome status")
            if event["status"] != "failed" and not math.isfinite(event["oracle_score"]):
                raise ValueError("Nonfinite successful outcome")
            outcomes.append(event)
            pending = None
        else:
            raise ValueError("Unexpected interior journal event")
    if len(charges) > protocol.additional_evaluations or result[
        "additional_charged_evaluations"
    ] != len(charges):
        raise ValueError("Charged evaluation accounting mismatch")
    if result["initial_logical_evaluations"] != 512 or result[
        "total_logical_evaluations"
    ] != 512 + len(charges):
        raise ValueError("Initial or total evaluation accounting mismatch")
    if result["failed_evaluations"] != sum(e["status"] == "failed" for e in outcomes) or result[
        "cached_repeats"
    ] != sum(e["status"] == "duplicate" for e in outcomes):
        raise ValueError("Outcome accounting mismatch")
    physical_completed = sum(
        charges[e["evaluation_index"]]["physical_oracle_call"] for e in outcomes
    )
    if (
        not physical_completed
        <= result["physical_adaptive_oracle_calls"]
        <= physical_completed + int(pending is not None and pending["physical_oracle_call"])
    ):
        raise ValueError("Physical call accounting mismatch")
    if result["status"] == "complete":
        if (
            pending is not None
            or len(charges) != 1024
            or result["failure"] is not None
            or any(e["late"] for e in outcomes)
            or result["elapsed_seconds"] >= result["max_seconds"]
        ):
            raise ValueError("Completion requirements not met")
    elif result["status"] != "failed" or result["failure"] is None:
        raise ValueError("Invalid terminal failure status")
    return {
        "verified": True,
        "status": result["status"],
        "charged_evaluations": len(charges),
        "outcomes": len(outcomes),
        "pending_charge_retained": pending is not None,
        "terminal_event_sha256": previous,
    }

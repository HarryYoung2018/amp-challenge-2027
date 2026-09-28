"""Acyclic source-plan and receipt records; no evaluator or release authority."""

from __future__ import annotations

import math
from dataclasses import dataclass

from amp_challenge.evaluation.evolutionary_kl_successor_protocol_v2 import (
    CONFIGURATION_IDS_V2,
    SCREEN_SEEDS_V2,
)
from amp_challenge.generators.search import durable_dispatch_journal_records as j

ARTIFACT = "common_initial_source_phase_v1"
CONTRACT = "docs/benchmarks/common_initial_producer_v1_20260914.md"
SOURCE_FILES = tuple(
    sorted(
        j.SOURCE_FILES
        | {
            CONTRACT,
            "src/amp_challenge/evaluation/evolutionary_kl_successor_protocol_v2.py",
            "src/amp_challenge/generators/search/common_initial_source_records.py",
            "src/amp_challenge/generators/search/common_initial_source.py",
            "src/amp_challenge/generators/search/common_initial_source_verify.py",
            "src/amp_challenge/generators/search/common_initial_source_cli.py",
        }
    )
)
MAX_PHASES = 256
MAX_EVENT_BYTES = 262144
FLAGS = {
    "scientific_evidence_accepted": False,
    "external_issuer_truth_established": False,
    "release_authority": False,
    "observed_gpu_cost_verified": False,
}


@dataclass(frozen=True, slots=True)
class SourcePlan:
    source_run_id: str
    seed: int
    objective_context_sha256: str
    oracle_bundle_sha256: str
    initial_requests: tuple[j.JournalRequest, ...]
    reserve_requests: tuple[j.JournalRequest, ...]
    destinations: tuple[tuple[str, str], ...]

    def __post_init__(self):
        j.require(j.identifier(self.source_run_id), "source run ID differs")
        j.require(type(self.seed) is int and self.seed in SCREEN_SEEDS_V2, "source seed differs")
        j.require(
            j.pin(self.objective_context_sha256) and j.pin(self.oracle_bundle_sha256),
            "source objective/oracle pins differ",
        )
        for rows, count in ((self.initial_requests, 64), (self.reserve_requests, 56)):
            j.require(type(rows) is tuple and len(rows) == count, "source request count differs")
            for row in rows:
                j.require(type(row) is j.JournalRequest, "source request type differs")
                row.__post_init__()
                j.require(
                    row.identity.endpoint_context_sha256 == self.objective_context_sha256,
                    "source request objective context differs",
                )
        rows = self.initial_requests + self.reserve_requests
        expected_identity = self.initial_requests[0].identity
        j.require(
            all(
                getattr(row.identity, field) == getattr(expected_identity, field)
                for row in rows
                for field in j.QUERY_FIELDS[1:-1]
            ),
            "source request oracle identity differs",
        )
        for keys in (
            [row.query_id for row in rows],
            [row.sequence for row in rows],
            [row.identity.key for row in rows],
        ):
            j.require(len(set(keys)) == 120, "source and private reserve identities overlap")
        expected = tuple((name, f"screen.{name}.seed-{self.seed}") for name in CONFIGURATION_IDS_V2)
        j.require(
            type(self.destinations) is tuple
            and all(
                type(row) is tuple and len(row) == 2 and all(type(value) is str for value in row)
                for row in self.destinations
            )
            and self.destinations == expected,
            "source destinations differ",
        )

    def document(self):
        self.__post_init__()
        return {
            "source_run_id": self.source_run_id,
            "seed": self.seed,
            "objective_context_sha256": self.objective_context_sha256,
            "oracle_bundle_sha256": self.oracle_bundle_sha256,
            "initial_requests": [row.document() for row in self.initial_requests],
            "reserve_requests": [row.document() for row in self.reserve_requests],
            "destinations": [list(row) for row in self.destinations],
        }

    @property
    def sha256(self):
        return j.digest(b"amp/common-initial/source-plan/v1\0" + j.canonical(self.document()))


def timing(epoch, deadline, epoch_id):
    j.require(
        all(type(value) is float and math.isfinite(value) for value in (epoch, deadline))
        and 0 <= epoch < deadline <= epoch + 900.0
        and j.identifier(epoch_id),
        "original source clock differs",
    )


@dataclass(frozen=True, slots=True)
class SourceDispatch:
    plan: SourcePlan
    index: int
    original_epoch: float
    original_deadline: float
    clock_epoch_id: str

    def document(self):
        j.require(type(self.plan) is SourcePlan, "dispatch plan type differs")
        self.plan.__post_init__()
        j.require(type(self.index) is int and 0 <= self.index < 64, "source index differs")
        timing(self.original_epoch, self.original_deadline, self.clock_epoch_id)
        identity = {
            "plan_sha256": self.plan.sha256,
            "source_run_id": self.plan.source_run_id,
            "seed": self.plan.seed,
            "index": self.index,
            "request": self.plan.initial_requests[self.index].document(),
        }
        token = j.digest(b"amp/common-initial/dispatch/v1\0" + j.canonical(identity))
        return {
            **identity,
            "token": token,
            "original_epoch": self.original_epoch,
            "original_deadline": self.original_deadline,
            "clock_epoch_id": self.clock_epoch_id,
        }


def dispatch_expected(dispatch, kind, external_submission_id=None):
    j.require(
        type(dispatch) is SourceDispatch
        and kind in ("common_initial_ack", "common_initial_terminal"),
        "source receipt kind/type differs",
    )
    value = {
        "kind": kind,
        "dispatch": dispatch.document(),
        "objective_context_sha256": dispatch.plan.objective_context_sha256,
        "oracle_bundle_sha256": dispatch.plan.oracle_bundle_sha256,
    }
    if kind == "common_initial_terminal":
        j.require(j.identifier(external_submission_id), "source terminal external ID differs")
        value["external_submission_id"] = external_submission_id
    return j.canonical(value)


def initial_expected(plan, kind, *, run_id=None, rows_sha256=None, source_receipt_sha256=None):
    """Match journal expectations before its binding/genesis can exist."""
    j.require(
        type(plan) is SourcePlan and kind in ("initial_source", "initial_copy"),
        "source aggregate kind/type differs",
    )
    plan.__post_init__()
    value = {
        "kind": kind,
        "source_run_id": plan.source_run_id,
        "seed": plan.seed,
        "objective_context_sha256": plan.objective_context_sha256,
        "oracle_bundle_sha256": plan.oracle_bundle_sha256,
        "requests": [row.document() for row in plan.initial_requests],
    }
    if kind == "initial_copy":
        j.require(
            type(run_id) is str
            and run_id in dict(plan.destinations).values()
            and j.pin(rows_sha256)
            and j.pin(source_receipt_sha256),
            "source copy destination/pins differ",
        )
        value.update(
            run_id=run_id, rows_sha256=rows_sha256, source_receipt_sha256=source_receipt_sha256
        )
    return j.canonical(value)

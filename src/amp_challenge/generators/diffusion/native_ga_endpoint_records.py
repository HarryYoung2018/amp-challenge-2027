"""GA endpoint arm handoffs: disclosed charged values, never hidden teachers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from amp_challenge.generators.diffusion.native_baseline_operators import NormalizedObjectiveContext
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    EndpointOrigin,
    EndpointTarget,
    EndpointTeacher,
)
from amp_challenge.generators.search.peptide_ga_driver_v2 import GADriverContext
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    digest,
    hash_string,
    require,
    sequence_key,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

ARM_CONFIG_SHA256 = "0759c2df36a0692a4556da3c7323a5bc01991cb0a0e058b986fa7b37ef0abb18"
ARTIFACT = "native_ga_endpoint_no_kg_wave_v1"


@dataclass(frozen=True, slots=True)
class GAEndpointContext:
    driver: GADriverContext
    objective: NormalizedObjectiveContext
    feature_source_sha256: str
    evaluator_source_sha256: str
    eligibility_source_sha256: str
    selected_tuning_result_available: bool = False

    def __post_init__(self):
        require(
            type(self.driver) is GADriverContext
            and type(self.objective) is NormalizedObjectiveContext,
            "GA endpoint context type differs",
        )
        self.driver.__post_init__()
        self.objective.__post_init__()
        require(
            self.driver.objective_context_sha256 == self.objective.context_sha256,
            "GA endpoint objective identities differ",
        )
        require(
            all(
                hash_string(value)
                for value in (
                    self.feature_source_sha256,
                    self.evaluator_source_sha256,
                    self.eligibility_source_sha256,
                )
            ),
            "GA endpoint provider source pins differ",
        )
        require(
            self.selected_tuning_result_available is False,
            "GA endpoint tuned-winner authentication is not implemented; no default is tuned",
        )


@dataclass(frozen=True, slots=True)
class ChargedEndpointEligibility:
    """Controller-authenticated eligibility and original source versions.

    Matching identities alone do not authenticate scientific eligibility. The
    provider owns its fixed cheap rules, actual receipts and first-seen origins.
    No unsuccessful observation may acquire a value through this interface.
    """

    history_sha256: str
    objective_context_sha256: str
    source_sha256: str
    receipt_sha256: str
    query_ids: tuple[str, ...]
    origins: tuple[EndpointOrigin, ...]

    def validate(self, history: VerifiedHistorySnapshot, context: GAEndpointContext):
        require(
            (self.history_sha256, self.objective_context_sha256, self.source_sha256)
            == (history.sha256, context.objective.context_sha256, context.eligibility_source_sha256)
            and hash_string(self.receipt_sha256),
            "charged target eligibility source/history differs",
        )
        require(
            type(self.query_ids) is tuple
            and self.query_ids == tuple(sorted(set(self.query_ids)))
            and len(self.query_ids) <= 512
            and type(self.origins) is tuple
            and len(self.origins) == len(self.query_ids),
            "charged target eligibility inventory differs",
        )
        require(
            set(self.query_ids) <= {row.query_id for row in history.observations},
            "eligibility refers to an undisclosed query",
        )
        for origin in self.origins:
            require(type(origin) is EndpointOrigin, "charged endpoint origin type differs")
            origin.__post_init__()
            require(
                origin.generation is None or origin.generation < history.round_index,
                "charged endpoint claims a future original generation",
            )


def build_charged_teacher(history, eligibility, context):
    eligibility.validate(history, context)
    origins = dict(zip(eligibility.query_ids, eligibility.origins, strict=True))
    rows = sorted(
        (
            row
            for row in history.observations
            if row.query_id in origins and row.status == "successful"
        ),
        key=lambda row: (-context.objective.scalarize(row.objectives), sequence_key(row.sequence)),
    )[:64]
    maximum = context.objective.scalarize(rows[0].objectives) if rows else 0.0
    targets = tuple(
        EndpointTarget(
            row.sequence,
            "charged_endpoint",
            float(
                max(
                    -math.log(2),
                    min(0.0, (context.objective.scalarize(row.objectives) - maximum) / 0.1),
                )
            ),
            origins[row.query_id],
        )
        for row in rows
    )
    return EndpointTeacher(
        targets,
        "charged_endpoint",
        context.objective.context_sha256,
        digest(canonical_json_bytes(asdict(eligibility))),
        min(28, history.round_index),
    )


@dataclass(frozen=True, slots=True)
class GAEndpointWave:
    record_json: str
    sha256: str
    status: str
    ranked_sequences: tuple[str, ...]
    next_native_ordinal: int
    next_behavior_version: int
    checkpoint_payloads: tuple[tuple[str, bytes], ...]
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False

    @property
    def method_pool(self):
        return self.ranked_sequences if self.status == "ready_private_composition_required" else ()

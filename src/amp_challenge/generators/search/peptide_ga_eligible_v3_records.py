"""Explicit context eligibility without altering the actual charged ledger."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.generators.search.peptide_ga_records import ArchiveIndividual
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    OBJECTIVES,
    GAKernelBatch,
    canonical_json_bytes,
    digest,
    hash_string,
    parameters,
    require,
    sequence_key,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    CONTRACT_SHA256 as LEGACY_CONTRACT_SHA256,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

NEW_CONTRACT_SHA256 = "39b0d86024be30c6277a18ab9fbfe482953bca1bd88d52f2417f6b964ae83123"
SPECIAL_STATUSES = (
    "paused_incomplete_history",
    "budget_complete_pending_controller_terminal",
    "abstained_no_eligible_parent",
    "stopped_deadline",
)


@dataclass(frozen=True, slots=True)
class GAContextEligibility:
    history_sha256: str
    objective_context_sha256: str
    source_sha256: str
    receipt_sha256: str
    query_ids: frozenset[str]

    def __post_init__(self):
        require(
            all(
                hash_string(value)
                for value in (
                    self.history_sha256,
                    self.objective_context_sha256,
                    self.source_sha256,
                    self.receipt_sha256,
                )
            ),
            "eligible GA history/context/source/receipt identity differs",
        )
        require(
            type(self.query_ids) is frozenset
            and len(self.query_ids) <= 512
            and all(type(value) is str for value in self.query_ids),
            "eligible GA subset must be an exact frozenset of query IDs",
        )

    def document(self):
        return {**asdict(self), "query_ids": sorted(self.query_ids)}


@dataclass(frozen=True, slots=True)
class EligibleGAKernelInput:
    configuration_id: str
    history: VerifiedHistorySnapshot
    training_sequence_keys: tuple[str, ...]
    eligibility: GAContextEligibility
    contract_sha256: str = NEW_CONTRACT_SHA256

    def __post_init__(self):
        parameters(self.configuration_id)
        require(type(self.history) is VerifiedHistorySnapshot, "eligible GA raw history differs")
        self.history.__post_init__()
        require(type(self.eligibility) is GAContextEligibility, "eligible GA record type differs")
        self.eligibility.__post_init__()
        require(self.contract_sha256 == NEW_CONTRACT_SHA256, "eligible GA contract differs")
        require(
            type(self.training_sequence_keys) is tuple
            and len(self.training_sequence_keys) <= 65536
            and all(hash_string(value) for value in self.training_sequence_keys)
            and tuple(sorted(set(self.training_sequence_keys))) == self.training_sequence_keys,
            "eligible GA training exclusion inventory differs",
        )
        require(
            not set(self.training_sequence_keys).intersection(
                sequence_key(row.sequence) for row in self.history.observations
            ),
            "eligible GA history overlaps the excluded training inventory",
        )
        require(
            self.eligibility.history_sha256 == self.history.sha256
            and self.eligibility.objective_context_sha256 == self.history.objective_context_sha256,
            "eligible GA raw history/context binding differs",
        )
        require(
            self.eligibility.query_ids
            <= {row.query_id for row in self.history.observations if row.status == "successful"},
            "eligible GA subset must contain only current successful query IDs",
        )

    def document(self):
        return {
            "configuration_id": self.configuration_id,
            "history": asdict(self.history),
            "training_sequence_keys": self.training_sequence_keys,
            "eligibility": self.eligibility.document(),
            "contract_sha256": self.contract_sha256,
        }

    @property
    def sha256(self):
        return digest(
            b"amp/context-eligible-peptide-ga/input/v3\0" + canonical_json_bytes(self.document())
        )

    @property
    def stream_sha256(self):
        # This is an RNG document, NOT a projected ChargedObservation/history.
        document = {
            "configuration_id": self.configuration_id,
            "root_seed": self.history.seed,
            "round_index": self.history.round_index,
            "observations": [
                {
                    "charge_index": row.charge_index,
                    "sequence": row.sequence,
                    "status": row.status,
                    "objectives": row.objectives
                    if row.query_id in self.eligibility.query_ids
                    else None,
                }
                for row in self.history.observations
            ],
            "training_sequence_keys": self.training_sequence_keys,
            "contract_sha256": LEGACY_CONTRACT_SHA256,
        }
        return digest(
            b"amp/tunable-peptide-ga/semantic-stream/v2\0" + canonical_json_bytes(document)
        )

    @property
    def semantic_sha256(self):
        document = {
            "objective_context_sha256": self.history.objective_context_sha256,
            "configuration_id": self.configuration_id,
            "training_sequence_keys": self.training_sequence_keys,
            "seed": self.history.seed,
            "round_index": self.history.round_index,
            "observations": [
                {
                    "charge_index": row.charge_index,
                    "sequence": row.sequence,
                    "status": row.status,
                    "eligible": row.query_id in self.eligibility.query_ids,
                    "objectives": row.objectives
                    if row.query_id in self.eligibility.query_ids
                    else None,
                }
                for row in self.history.observations
            ],
        }
        return digest(
            b"amp/context-eligible-peptide-ga/semantic/v3\0" + canonical_json_bytes(document)
        )

    @property
    def round_seed(self):
        raw = canonical_json_bytes(
            {"root_seed": self.history.seed, "round_index": self.history.round_index}
        )
        return int.from_bytes(
            bytes.fromhex(digest(b"amp/tunable-peptide-ga/round-seed/v2\0" + raw))[:8], "big"
        ) & (2**63 - 1)

    def eligible_population(self):
        population = []
        for row in self.history.observations:
            if row.query_id not in self.eligibility.query_ids:
                continue
            means = tuple((OBJECTIVES[index], row.objectives[index]) for index in range(2))
            population.append(
                ArchiveIndividual(
                    row.sequence,
                    sequence_key(row.sequence),
                    row.query_id,
                    (row.query_id,),
                    means,
                    math.fsum(value * 0.5 for _, value in means),
                )
            )
        return tuple(sorted(population, key=lambda row: (-row.fitness, row.sequence_key)))


def check_expected_input(
    input_record,
    *,
    expected_history_sha256,
    expected_objective_context_sha256,
    expected_eligibility_source_sha256,
    expected_eligible_query_ids,
):
    require(type(input_record) is EligibleGAKernelInput, "eligible GA input type differs")
    input_record.__post_init__()
    require(
        all(
            hash_string(value)
            for value in (
                expected_history_sha256,
                expected_objective_context_sha256,
                expected_eligibility_source_sha256,
            )
        )
        and type(expected_eligible_query_ids) is frozenset
        and all(type(value) is str for value in expected_eligible_query_ids),
        "eligible GA external expectations differ",
    )
    require(
        input_record.history.sha256 == expected_history_sha256
        and input_record.history.objective_context_sha256 == expected_objective_context_sha256
        and input_record.eligibility.source_sha256 == expected_eligibility_source_sha256
        and input_record.eligibility.query_ids == expected_eligible_query_ids,
        "eligible GA exact external history/context/source/subset binding differs",
    )


def prepare_eligible_ga_input(
    history,
    eligibility,
    *,
    configuration_id,
    training_sequence_keys,
    expected_history_sha256,
    expected_objective_context_sha256,
    expected_eligibility_source_sha256,
    expected_eligible_query_ids,
):
    result = EligibleGAKernelInput(configuration_id, history, training_sequence_keys, eligibility)
    check_expected_input(
        result,
        expected_history_sha256=expected_history_sha256,
        expected_objective_context_sha256=expected_objective_context_sha256,
        expected_eligibility_source_sha256=expected_eligibility_source_sha256,
        expected_eligible_query_ids=expected_eligible_query_ids,
    )
    return result


def eligible_source_identities():
    root = Path(__file__).resolve().parents[4]
    paths = [
        "configs/search/eligible_peptide_ga_kernel_v3.toml",
        "configs/search/tunable_peptide_ga_kernel_v2.toml",
    ] + [
        "src/amp_challenge/generators/search/" + name + ".py"
        for name in (
            "peptide_ga_eligible_v3_records",
            "peptide_ga_eligible_v3",
            "peptide_ga_eligible_v3_verify",
            "peptide_ga_tunable_v2",
            "peptide_ga_tunable_v2_records",
            "peptide_ga_records",
            "peptide_ga",
            "peptide_ga_verifier",
            "records",
            "verified_charged_history",
        )
    ]
    result = {name: digest((root / name).read_bytes()) for name in paths}
    require(result[paths[0]] == NEW_CONTRACT_SHA256, "eligible GA config bytes differ")
    require(result[paths[1]] == LEGACY_CONTRACT_SHA256, "eligible GA legacy config differs")
    return result


def eligible_implementation_sha256():
    return digest(canonical_json_bytes(eligible_source_identities()))


@dataclass(frozen=True, slots=True)
class EligibleGAPrefix:
    input_sha256: str
    semantic_sha256: str
    source_sha256: str
    contract_sha256: str
    deadline_monotonic: float
    status: str
    prefix: GAKernelBatch | None
    prior_resume_sha256: str | None
    prior_resume_attempt_count: int
    resume_reconstruction_completed: bool
    output_sha256: str

    def __post_init__(self):
        require(
            all(
                hash_string(value)
                for value in (
                    self.input_sha256,
                    self.semantic_sha256,
                    self.source_sha256,
                    self.contract_sha256,
                    self.output_sha256,
                )
            )
            and self.contract_sha256 == NEW_CONTRACT_SHA256,
            "eligible GA output source/contract/hash differs",
        )
        require(
            type(self.deadline_monotonic) in (float, int)
            and math.isfinite(self.deadline_monotonic),
            "eligible GA original deadline differs",
        )
        require(
            self.status in (*SPECIAL_STATUSES, "complete", "in_progress", "attempt_cap_exhausted"),
            "eligible GA wrapper status differs",
        )
        if self.prefix is not None:
            require(type(self.prefix) is GAKernelBatch, "eligible GA inner prefix type differs")
            self.prefix.__post_init__()
            require(
                self.status in (self.prefix.status, "stopped_deadline"),
                "eligible GA wrapper/inner status differs",
            )
        else:
            require(self.status in SPECIAL_STATUSES, "eligible GA active wrapper lacks prefix")
        require(
            type(self.prior_resume_attempt_count) is int
            and 0 <= self.prior_resume_attempt_count <= ATTEMPT_CAP
            and type(self.resume_reconstruction_completed) is bool,
            "eligible GA prior resume claim count/completion type differs",
        )
        if self.prior_resume_sha256 is None:
            require(
                self.prior_resume_attempt_count == 0 and not self.resume_reconstruction_completed,
                "eligible GA absent prior resume has nonempty claims",
            )
        else:
            require(
                hash_string(self.prior_resume_sha256), "eligible GA prior resume claim hash differs"
            )
            if self.resume_reconstruction_completed:
                require(
                    self.prefix is not None
                    and self.prior_resume_attempt_count <= len(self.prefix.attempts),
                    "eligible GA reconstructed prior count exceeds current prefix",
                )
            else:
                require(
                    self.status == "stopped_deadline" and self.prefix is None,
                    "eligible GA unauthenticated prior replay cannot expose attempts",
                )

    @property
    def accepted_sequences(self):
        return self.prefix.accepted_sequences if self.status == "complete" else ()

    @property
    def next_attempt_index(self):
        if self.prior_resume_sha256 is not None and not self.resume_reconstruction_completed:
            return None  # Unknown authenticated cumulative count, never fresh quota.
        return 0 if self.prefix is None else len(self.prefix.attempts)

    def canonical_bytes(self):
        return canonical_json_bytes(asdict(self))


def eligible_batch_digest(result):
    document = asdict(result)
    document.pop("output_sha256")
    return digest(b"amp/context-eligible-peptide-ga/output/v3\0" + canonical_json_bytes(document))

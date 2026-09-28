"""Structural records for a tunable GA kernel, not oracle/execution authority."""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.generators.search.peptide_ga_records import (
    ArchiveIndividual,
    canonical_json_bytes,
    sequence_key,
)
from amp_challenge.generators.search.records import ProbabilityFactor

CONTRACT_SHA256 = "04cd4fd7d010baeb07bdf2965cc01f4b1816609df44e6b37f0b118b18d4e3f1c"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
OPERATORS = ("substitution", "insertion", "deletion", "two_parent_crossover")
OBJECTIVES = ("gram_positive_activity", "gram_negative_activity")
TUNING_SEEDS = (71017, 71042, 71091, 71137, 71271)
CONFIGURATIONS = (
    ("ga_t3_default", 3, (0.45, 0.15, 0.15, 0.25)),
    ("ga_t3_mutation_heavy", 3, (0.7, 0.1, 0.1, 0.1)),
    ("ga_t5_default", 5, (0.45, 0.15, 0.15, 0.25)),
    ("ga_t5_mutation_heavy", 5, (0.7, 0.1, 0.1, 0.1)),
)
ATTEMPT_CAP = 65536
PREFIX_SIZE = 256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_string(value: str) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical_sequence(value: str) -> bool:
    return type(value) is str and 8 <= len(value) <= 50 and set(value) <= set(ALPHABET)


@dataclass(frozen=True, slots=True)
class TunableGAParameters:
    configuration_id: str
    tournament_size: int
    operator_rates: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        expected = {
            name: (size, tuple(zip(OPERATORS, rates, strict=True)))
            for name, size, rates in CONFIGURATIONS
        }
        require(
            type(self.configuration_id) is str and self.configuration_id in expected,
            "unknown GA configuration",
        )
        require(
            type(self.tournament_size) is int
            and type(self.operator_rates) is tuple
            and all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is float
                for row in self.operator_rates
            ),
            "parameter types differ",
        )
        require(
            (self.tournament_size, self.operator_rates) == expected[self.configuration_id],
            "parameters differ from frozen four-configuration grid",
        )

    # These views deliberately satisfy the existing v1 numerical parent/edit
    # helpers without constructing or relaxing its fixed-default config class.
    alphabet = ALPHABET
    min_length = 8
    max_length = 50
    elite_fraction = 0.25
    elite_parent_probability = 0.5


def parameters(configuration_id: str) -> TunableGAParameters:
    for name, size, rates in CONFIGURATIONS:
        if name == configuration_id:
            return TunableGAParameters(name, size, tuple(zip(OPERATORS, rates, strict=True)))
    raise ValueError("unknown GA configuration")


def load_contract(path: Path, expected_sha256: str = CONTRACT_SHA256) -> dict:
    require(
        path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 16384,
        "contract path/size differs",
    )
    payload = path.read_bytes()
    require(
        expected_sha256 == CONTRACT_SHA256 and digest(payload) == expected_sha256,
        "frozen contract digest differs",
    )
    value = tomllib.loads(payload.decode())
    require(
        tuple(
            (row["id"], row["tournament_size"], tuple(row["operator_rates"]))
            for row in value["configurations"]
        )
        == CONFIGURATIONS,
        "contract grid differs",
    )
    require(tuple(value["prospective_study"]["seeds"]) == TUNING_SEEDS, "contract seeds differ")
    return value


@dataclass(frozen=True, slots=True)
class ChargedObservation:
    """Values already disclosed by a charged response; external origin is unchecked.

    The future controller authenticates the query/response and its charged
    identity. A record's digest-shaped fields do not themselves establish that.
    """

    charge_index: int
    query_id: str
    sequence: str
    response_receipt_sha256: str
    status: str
    objectives: tuple[float, float] | None

    def __post_init__(self) -> None:
        require(
            # Structural record ceiling accommodates the prospective 512+1024
            # study. Controllers still enforce their own exact contiguous
            # inventory, including the unchanged legacy 512-charge contract.
            type(self.charge_index) is int and 0 <= self.charge_index < 1536,
            "charge index differs",
        )
        require(
            type(self.query_id) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.query_id) is not None,
            "query identity differs",
        )
        require(
            canonical_sequence(self.sequence) and hash_string(self.response_receipt_sha256),
            "charged sequence/receipt differs",
        )
        require(
            self.status in ("successful", "failed", "missing", "censored", "partial", "timed_out"),
            "charged outcome status differs",
        )
        if self.status == "successful":
            require(
                type(self.objectives) is tuple
                and len(self.objectives) == 2
                and all(
                    type(value) is float and math.isfinite(value) and 0 <= value <= 1
                    for value in self.objectives
                ),
                "successful objectives must be finite probabilities",
            )
        else:
            require(self.objectives is None, "unsuccessful charges may not expose fitness values")


@dataclass(frozen=True, slots=True)
class GAKernelInput:
    configuration_id: str
    root_seed: int
    round_index: int
    observations: tuple[ChargedObservation, ...]
    training_sequence_keys: tuple[str, ...]
    contract_sha256: str = CONTRACT_SHA256

    def __post_init__(self) -> None:
        parameters(self.configuration_id)
        require(self.contract_sha256 == CONTRACT_SHA256, "kernel contract differs")
        require(type(self.root_seed) is int and 0 <= self.root_seed < 2**63, "root seed differs")
        require(
            type(self.round_index) is int and 1 <= self.round_index <= 28, "adaptive round differs"
        )
        require(
            type(self.observations) is tuple
            and len(self.observations) == 64 + 16 * (self.round_index - 1),
            "charged pre-round inventory differs",
        )
        for index, row in enumerate(self.observations):
            require(type(row) is ChargedObservation, "charged record type differs")
            row.__post_init__()
            require(row.charge_index == index, "charged records must be contiguous and ordered")
        require(
            len({row.query_id for row in self.observations}) == len(self.observations),
            "charged query identity repeats",
        )
        require(
            len({row.sequence for row in self.observations}) == len(self.observations),
            "charged sequence repeats",
        )
        require(
            any(row.status == "successful" for row in self.observations),
            "successful parent population is empty",
        )
        require(
            type(self.training_sequence_keys) is tuple
            and len(self.training_sequence_keys) <= 65536
            and all(hash_string(key) for key in self.training_sequence_keys),
            "training key inventory differs",
        )
        require(
            tuple(sorted(set(self.training_sequence_keys))) == self.training_sequence_keys,
            "training keys must be sorted and unique",
        )
        require(
            not {sequence_key(row.sequence) for row in self.observations}.intersection(
                self.training_sequence_keys
            ),
            "charged history contains forbidden training overlap",
        )

    @property
    def sha256(self) -> str:
        return digest(b"amp/tunable-peptide-ga/input/v2\0" + canonical_json_bytes(asdict(self)))

    @property
    def stream_sha256(self) -> str:
        """Bind numerical state without transport/query receipt identities."""
        document = asdict(self)
        document["observations"] = [
            {
                "charge_index": row.charge_index,
                "sequence": row.sequence,
                "status": row.status,
                "objectives": row.objectives,
            }
            for row in self.observations
        ]
        return digest(
            b"amp/tunable-peptide-ga/semantic-stream/v2\0" + canonical_json_bytes(document)
        )

    @property
    def round_seed(self) -> int:
        document = {"root_seed": self.root_seed, "round_index": self.round_index}
        payload = hashlib.sha256(
            b"amp/tunable-peptide-ga/round-seed/v2\0" + canonical_json_bytes(document)
        ).digest()
        return int.from_bytes(payload[:8], "big") & (2**63 - 1)


def population_from_charged(input_record: GAKernelInput) -> tuple[ArchiveIndividual, ...]:
    population = []
    for row in input_record.observations:
        if row.status != "successful":
            continue
        assert row.objectives is not None
        population.append(
            ArchiveIndividual(
                row.sequence,
                sequence_key(row.sequence),
                row.query_id,
                (row.query_id,),
                tuple(zip(OBJECTIVES, row.objectives, strict=True)),
                math.fsum(0.5 * value for value in row.objectives),
            )
        )
    return tuple(sorted(population, key=lambda row: (-row.fitness, row.sequence_key)))


@dataclass(frozen=True, slots=True)
class GAEdit:
    sequence: str
    operator: str
    parent_sequence_keys: tuple[str, ...]
    description: tuple[str | int, ...]
    probability_factors: tuple[ProbabilityFactor, ...]

    def __post_init__(self) -> None:
        require(
            type(self.sequence) is str
            and 1 <= len(self.sequence) <= 98
            and set(self.sequence) <= set(ALPHABET),
            "edited sequence differs",
        )
        require(
            self.operator in OPERATORS
            and type(self.parent_sequence_keys) is tuple
            and len(self.parent_sequence_keys)
            == (2 if self.operator == "two_parent_crossover" else 1)
            and all(hash_string(key) for key in self.parent_sequence_keys),
            "edit parent/operator differs",
        )
        require(
            len(set(self.parent_sequence_keys)) == len(self.parent_sequence_keys),
            "crossover parents repeat",
        )
        require(
            type(self.description) is tuple
            and 2 <= len(self.description) <= 5
            and all(type(item) in (str, int) for item in self.description),
            "edit description differs",
        )
        require(
            type(self.probability_factors) is tuple
            and 1 <= len(self.probability_factors) <= 16
            and all(type(factor) is ProbabilityFactor for factor in self.probability_factors),
            "probability trace differs",
        )
        for factor in self.probability_factors:
            factor.__post_init__()


@dataclass(frozen=True, slots=True)
class GAAttempt:
    attempt_index: int
    edit: GAEdit
    rejection_reason: str | None
    accepted_position: int | None
    first_attempt_index: int

    def __post_init__(self) -> None:
        require(
            type(self.attempt_index) is int
            and 0 <= self.attempt_index < ATTEMPT_CAP
            and type(self.first_attempt_index) is int
            and 0 <= self.first_attempt_index <= self.attempt_index,
            "attempt index differs",
        )
        require(type(self.edit) is GAEdit, "edit type differs")
        self.edit.__post_init__()
        require(
            self.rejection_reason
            in (
                None,
                "length_out_of_support",
                "exact_training_overlap",
                "previously_submitted_collision",
                "generated_duplicate",
            ),
            "rejection reason differs",
        )
        require(
            (
                self.rejection_reason is None
                and type(self.accepted_position) is int
                and 0 <= self.accepted_position < PREFIX_SIZE
            )
            or (self.rejection_reason is not None and self.accepted_position is None),
            "accepted position differs",
        )


@dataclass(frozen=True, slots=True)
class GAKernelBatch:
    input_sha256: str
    stream_sha256: str
    configuration_id: str
    round_seed: int
    attempts: tuple[GAAttempt, ...]
    accepted_sequences: tuple[str, ...]
    status: str
    output_sha256: str

    def __post_init__(self) -> None:
        require(
            hash_string(self.input_sha256)
            and hash_string(self.stream_sha256)
            and hash_string(self.output_sha256),
            "batch digest differs",
        )
        parameters(self.configuration_id)
        require(type(self.round_seed) is int and 0 <= self.round_seed < 2**63, "batch seed differs")
        require(
            type(self.attempts) is tuple
            and len(self.attempts) <= ATTEMPT_CAP
            and all(type(row) is GAAttempt for row in self.attempts),
            "attempt inventory differs",
        )
        require(
            type(self.accepted_sequences) is tuple
            and len(self.accepted_sequences) <= PREFIX_SIZE
            and all(canonical_sequence(row) for row in self.accepted_sequences),
            "accepted sequence inventory differs",
        )
        require(
            self.status in ("complete", "in_progress", "attempt_cap_exhausted"),
            "batch status differs",
        )

    @property
    def next_attempt_index(self) -> int:
        return len(self.attempts)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(asdict(self))


def batch_digest(batch: GAKernelBatch) -> str:
    document = asdict(batch)
    document.pop("output_sha256")
    return digest(b"amp/tunable-peptide-ga/output/v2\0" + canonical_json_bytes(document))

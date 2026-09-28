"""Prospective partial-operator records; no scientific or publication authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import _NativeUnit
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_initialization import AuditedNativeInitialization
from amp_challenge.generators.diffusion.native_shared_endpoint import endpoint_source_identities
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string, sequence_key

CONFIG_SHA256 = "445525aa693ece3dad07359a5864043c6d55235034b5ca59ea5f517e9b1ecd29"
GA_MATCHED_CONFIG_PATH = "configs/search/native_ga_matched_feasibility_v1.toml"
GA_MATCHED_CONFIG_SHA256 = "d3889ba709849169d14087333adc8bc4edc7b893ef018926ae53b6c262e76c15"
ELIGIBLE_CONTRACT_SHA256 = "39b0d86024be30c6277a18ab9fbfe482953bca1bd88d52f2417f6b964ae83123"
MAX_RECEIPT_BYTES = 128 * 1024**2
MEASURE = "CONDITIONAL_LEGAL_GA_PARTIAL_REMASK_ATTEMPT_LAW"


def read_clock(clock):
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("partial guard clock must return a finite numeric value, not bool")
    return float(value)


def source_identities():
    root = Path(__file__).resolve().parents[4]
    paths = ["configs/search/native_ga_partial_guard_v2.toml", GA_MATCHED_CONFIG_PATH] + [
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            "native_ga_partial_records",
            "native_ga_matched_feasibility",
            "native_ga_matched_feasibility_verify",
            "native_ga_partial_work",
            "native_ga_partial_guard",
            "native_ga_partial_update",
            "native_ga_partial_arm",
            "native_ga_partial_verify",
            "native_ga_endpoint_records",
            "native_ga_endpoint_arm",
            "native_shared_endpoint_verify",
            "native_search_posterior",
        )
    ]
    paths.append("src/amp_challenge/generators/search/peptide_ga_driver_v2.py")
    from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
        eligible_source_identities,
    )

    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    if result[paths[0]] != CONFIG_SHA256:
        raise ValueError("partial guard configuration bytes differ")
    if result[GA_MATCHED_CONFIG_PATH] != GA_MATCHED_CONFIG_SHA256:
        raise ValueError("GA matched feasibility amendment bytes differ")
    return {**endpoint_source_identities(), **eligible_source_identities(), **result}


def plain(value):
    if type(value) is frozenset:
        return sorted(value)
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def encode_record(document):
    raw = json.dumps(plain(document), sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = raw.encode()
    if len(data) > MAX_RECEIPT_BYTES:
        raise ValueError("partial guard aggregate diagnostic byte cap")
    return raw, hashlib.sha256(data).hexdigest()


def unit_binding(unit, version):
    if (
        type(unit) is not _NativeUnit
        or type(unit.initialization) is not AuditedNativeInitialization
    ):
        raise TypeError("partial guard exact native unit/initializer required")
    if type(version) is not int or not 0 <= version <= 27:
        raise ValueError("partial guard actual behavior version differs")
    unit.check()
    init = unit.initialization
    if (
        type(unit.sequences) is not tuple
        or unit.sequences != tuple(sorted(set(unit.sequences)))
        or tuple(sorted(map(sequence_key, unit.sequences))) != init.training_sequence_ids
        or canonical_model_logical_hash(init.model) != init.checkpoint_logical_sha256
        or init.triple != unit.triple
        or unit.reference_sha256 != init.checkpoint_logical_sha256
        or unit.model.config != unit.reference.config
        or unit.model.config.levels != 64
        or unit.model.config.min_length != 8
        or unit.model.config.max_length != 50
    ):
        raise ValueError("partial guard initializer/corpus/native64 binding differs")
    metadata = {
        field.name: getattr(init, field.name) for field in fields(init) if field.name != "model"
    }
    return (
        unit.triple,
        unit.policy_sha256,
        unit.reference_sha256,
        version,
        _json_hash(metadata),
        _json_hash(unit.sequences),
    )


@dataclass(frozen=True, slots=True)
class PartialGuardPlan:
    seed: int
    round_index: int
    native_ordinal: int
    remaining_attempts: int
    checkpoint_order: tuple[int, ...]
    parents: tuple[str, ...]
    kernel_semantic_sha256: str
    kernel_input_sha256: str
    prefix_sha256: str
    history_sha256: str
    eligibility_sha256: str
    prefix_deadline: float
    units: tuple[tuple, ...]
    source_sha256: str
    configuration_sha256: str = CONFIG_SHA256

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.round_index) is not int
            or not 1 <= self.round_index <= 28
            or type(self.native_ordinal) is not int
            or not 0 <= self.native_ordinal <= 28 * 65536
            or type(self.remaining_attempts) is not int
            or not 128 <= self.remaining_attempts <= 65280
            or type(self.checkpoint_order) is not tuple
            or any(type(index) is not int for index in self.checkpoint_order)
            or tuple(sorted(self.checkpoint_order)) != tuple(range(10))
            or type(self.parents) is not tuple
            or len(self.parents) != 128
            or len(set(self.parents)) != 128
            or any(
                type(seq) is not str
                or not 8 <= len(seq) <= 50
                or set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
                for seq in self.parents
            )
            or type(self.units) is not tuple
            or tuple(row[0] for row in self.units) != TRIPLES
            or any(
                type(row) is not tuple
                or len(row) != 6
                or type(row[3]) is not int
                or not 0 <= row[3] < self.round_index
                for row in self.units
            )
            or type(self.prefix_deadline) is not float
            or not math.isfinite(self.prefix_deadline)
            or self.configuration_sha256 != CONFIG_SHA256
            or any(
                not hash_string(value)
                for value in (
                    self.kernel_semantic_sha256,
                    self.kernel_input_sha256,
                    self.prefix_sha256,
                    self.history_sha256,
                    self.eligibility_sha256,
                    self.source_sha256,
                )
            )
        ):
            raise ValueError("partial guard plan inventory/source/version differs")

    @property
    def sha256(self):
        return _json_hash(asdict(self))

    @property
    def numerical_sha256(self):
        return _json_hash(
            [
                "ga-partial-guard-v2",
                self.seed,
                self.round_index,
                self.kernel_semantic_sha256,
                self.native_ordinal,
                self.remaining_attempts,
                self.checkpoint_order,
                self.parents,
            ]
        )


@dataclass(frozen=True, slots=True)
class PartialGuardedUpdate:
    record_json: str
    sha256: str
    status: str
    accepted: bool
    backtracks: int | None
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


@dataclass(frozen=True, slots=True)
class PartialGAWave:
    record_json: str
    sha256: str
    status: str
    ranked_sequences: tuple[str, ...]
    next_native_ordinal: int
    next_behavior_versions: tuple[tuple[str, int], ...]
    checkpoint_payloads: tuple[tuple[str, bytes], ...]
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False

    @property
    def method_pool(self):
        return self.ranked_sequences if self.status == "ready_private_composition_required" else ()

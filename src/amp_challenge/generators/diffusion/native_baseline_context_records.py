"""Portable applicability/envelope records; no native update producer imports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_baseline_operators import (
    BASELINE_CONFIG_SHA256,
    MODES,
    NativeBaselineAdvance,
    NativeBaselineEnsemble,
    NativeBaselineStep,
    NativeCandidatePool,
    NormalizedObjectiveContext,
    _NativeUnit,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeEndpointConfig,
)
from amp_challenge.generators.diffusion.native_initialization import (
    TRIPLES,
    AuditedNativeInitialization,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ChargedObservation,
    canonical_json_bytes,
    hash_string,
    require,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

CONFIG_SHA256 = "9f755a79446fcb195e8e76e979440457ea1ddc7b1cda3db7564a9b3aa4d596bf"
ARTIFACT = "native_baseline_context_eligibility_envelope_v2"
_ROOT = Path(__file__).resolve().parents[4]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def plain(value):
    """Canonical JSON representation, including unchanged legacy step subrecords."""
    # Exact builtins cannot carry dataclass fields or NumPy payloads. Most pool
    # leaves are these scalars; avoid generic reflection for each one.
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value
    if kind is dict:
        return {key: plain(item) for key, item in value.items()}
    if kind in (tuple, list):
        return [plain(item) for item in value]
    if kind is frozenset:
        return sorted(value)
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def document_sha(value):
    return digest(canonical_json_bytes(plain(value)))


def finite_clock(value):
    valid = type(value) in (int, float)
    try:
        valid = valid and math.isfinite(value) and math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        valid = False
    require(valid, "context baseline clock must be exact finite int/float without overflow")
    return float(value)


def read_clock(monotonic):
    require(callable(monotonic), "context baseline clock must be callable")
    return finite_clock(monotonic())


def source_identities():
    names = [
        "configs/diffusion/native_baseline_context_eligibility_v2.toml",
        "configs/diffusion/native_baseline_operators_v1.toml",
        "src/amp_challenge/generators/search/verified_charged_history.py",
        "src/amp_challenge/generators/search/peptide_ga_tunable_v2_records.py",
    ]
    names.extend(
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            "native_baseline_context_records",
            "native_baseline_context",
            "native_baseline_context_verify",
            "native_baseline_operators",
            "native_weighted_training",
            "native_endpoint",
            "native_initialization",
            "native_proposals",
            "model",
            "categorical",
            "subset_kernel",
            "replay",
        )
    )
    result = {name: digest((_ROOT / name).read_bytes()) for name in names}
    require(result[names[0]] == CONFIG_SHA256, "context baseline declaration changed")
    require(result[names[1]] == BASELINE_CONFIG_SHA256, "legacy baseline recipe changed")
    return result


def implementation_sha256():
    return document_sha(source_identities())


@dataclass(frozen=True, slots=True)
class NativeBaselineEligibility:
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
            "context baseline applicability identities differ",
        )
        require(
            type(self.query_ids) is frozenset
            and len(self.query_ids) <= 512
            and all(type(value) is str and 0 < len(value) <= 256 for value in self.query_ids),
            "context baseline applicability query set differs",
        )


@dataclass(frozen=True, slots=True)
class NativeBaselineContext:
    mode: str
    run_id: str
    seed: int
    objective: NormalizedObjectiveContext
    oracle_bundle_sha256: str
    implementation_sha256: str
    deadline_monotonic: float
    clock_epoch_id: str

    def __post_init__(self):
        require(self.mode in MODES[1:], "context wrapper supports only updating baselines")
        require(
            type(self.objective) is NormalizedObjectiveContext, "context objective type differs"
        )
        self.objective.__post_init__()
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "context seed differs")
        require(
            all(
                type(value) is str and 0 < len(value) <= 128
                for value in (
                    self.run_id,
                    self.clock_epoch_id,
                )
            ),
            "context run/clock epoch differs",
        )
        require(
            hash_string(self.oracle_bundle_sha256) and hash_string(self.implementation_sha256),
            "context source/bundle differs",
        )
        finite_clock(self.deadline_monotonic)


@dataclass(frozen=True, slots=True)
class NativeBaselineExpectations:
    history_sha256: str
    objective_context_sha256: str
    eligibility_source_sha256: str
    eligibility_receipt_sha256: str
    eligible_query_ids: frozenset[str]
    previous_wave_head_sha256: str
    previous_update_envelope_sha256: str | None

    def __post_init__(self):
        NativeBaselineEligibility(
            self.history_sha256,
            self.objective_context_sha256,
            self.eligibility_source_sha256,
            self.eligibility_receipt_sha256,
            self.eligible_query_ids,
        )
        require(hash_string(self.previous_wave_head_sha256), "expected wave head differs")
        require(
            self.previous_update_envelope_sha256 is None
            or hash_string(self.previous_update_envelope_sha256),
            "expected update head differs",
        )


def history_from_document(raw):
    values = dict(raw)
    values["observations"] = tuple(
        ChargedObservation(
            **{**row, "objectives": None if row["objectives"] is None else tuple(row["objectives"])}
        )
        for row in values["observations"]
    )
    result = VerifiedHistorySnapshot(**values)
    require(plain(result) == raw, "noncanonical context history document")
    return result


def eligibility_from_document(raw):
    result = NativeBaselineEligibility(**{**raw, "query_ids": frozenset(raw["query_ids"])})
    require(plain(result) == raw, "noncanonical applicability document")
    return result


def numerical_input_sha256(context, history, eligibility):
    return document_sha(
        {
            "mode": context.mode,
            "config": BASELINE_CONFIG_SHA256,
            "context": context.objective.context_sha256,
            "seed": history.seed,
            "round": history.round_index,
            "observations": [
                {
                    "sequence": row.sequence,
                    "status": row.status,
                    "eligible": row.query_id in eligibility.query_ids,
                    "objectives": row.objectives if row.query_id in eligibility.query_ids else None,
                }
                for row in history.observations
            ],
        }
    )


def unit_binding(unit):
    require(
        type(unit) is _NativeUnit and type(unit.initialization) is AuditedNativeInitialization,
        "context baseline exact unit/initializer types differ",
    )
    init = unit.initialization
    for field in fields(init):
        value = getattr(init, field.name)
        if field.name in (
            "training_folds",
            "training_sequence_ids",
            "training_union_component_ids",
        ):
            subtype = int if field.name == "training_folds" else str
            require(
                type(value) is tuple and all(type(item) is subtype for item in value),
                "context initializer exact tuple metadata differs",
            )
        elif field.name in ("production_input_eligible", "scientific_evidence_accepted"):
            require(type(value) is bool, "context initializer exact qualification type differs")
        elif field.name != "model":
            require(type(value) is str, "context initializer exact string metadata differs")
    require(
        all(type(model) is NativeDenoiser for model in (unit.model, unit.reference, init.model)),
        "context baseline native model type differs",
    )
    require(
        all(
            type(model.config) is NativeDenoiserConfig
            for model in (unit.model, unit.reference, init.model)
        ),
        "context baseline exact native configuration type differs",
    )
    require(
        type(unit.triple) is str
        and unit.triple in TRIPLES
        and type(unit.sequences) is tuple
        and 1 <= len(unit.sequences) <= 1113
        and all(
            type(seq) is str and 8 <= len(seq) <= 50 and not set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            for seq in unit.sequences
        )
        and tuple(sorted(set(unit.sequences))) == unit.sequences
        and tuple(sorted(sequence_id(seq) for seq in unit.sequences)) == init.training_sequence_ids
        and unit.triple == init.triple,
        "context baseline corpus/training lineage differs",
    )
    unit.check()
    require(
        canonical_model_logical_hash(init.model) == init.checkpoint_logical_sha256
        and unit.reference_sha256 == init.checkpoint_logical_sha256,
        "context baseline initializer/reference weights changed",
    )
    require(
        unit.model.config == unit.reference.config == init.model.config,
        "context baseline model configuration lineage differs",
    )
    return {
        "triple": unit.triple,
        "sequences": list(unit.sequences),
        "initialization": {
            field.name: plain(getattr(init, field.name))
            for field in fields(init)
            if field.name != "model"
        },
        "model_config": plain(unit.model.config),
        "policy_sha256": unit.policy_sha256,
        "reference_sha256": unit.reference_sha256,
    }


def inner_binding(inner):
    require(type(inner) is NativeBaselineEnsemble, "context baseline requires exact inner ensemble")
    require(
        type(inner.context) is NormalizedObjectiveContext
        and type(inner.config) is NativeEndpointConfig
        and inner.config == NATIVE_ENDPOINT_DEFAULTS,
        "context baseline exact objective/configuration types differ",
    )
    inner.context.__post_init__()
    require(
        type(inner._units) is tuple and 1 <= len(inner._units) <= 10,
        "context baseline exact unit tuple differs",
    )
    triples = tuple(unit.triple for unit in inner._units)
    require(
        triples == tuple(sorted(set(triples)))
        and type(inner.protocol_ten_checkpoint_mixture) is bool
        and inner.protocol_ten_checkpoint_mixture == (triples == TRIPLES),
        "context baseline exact mixture metadata differs",
    )
    require(
        type(inner._training_ids) is frozenset
        and inner._training_ids
        == frozenset(sequence_id(sequence) for unit in inner._units for sequence in unit.sequences),
        "context baseline full training union differs",
    )
    require(
        type(inner.seed) is int
        and all(
            type(value) is str for value in (inner.mode, inner.run_id, inner.oracle_bundle_sha256)
        ),
        "context baseline scalar identity types differ",
    )
    require(
        type(inner._receipts) is tuple
        and all(
            type(row) is NativeBaselineAdvance
            and type(row.steps) is tuple
            and all(type(step) is NativeBaselineStep for step in row.steps)
            for row in inner._receipts
        ),
        "context baseline retained numerical receipt types differ",
    )
    if inner._history is not None:
        require(
            type(inner._history) is VerifiedHistorySnapshot,
            "context baseline retained history type differs",
        )
        inner._history.__post_init__()
        require(inner._history.complete, "context baseline retained history must be complete")
    if inner._pool is not None:
        require(
            type(inner._pool) is NativeCandidatePool and inner._history is not None,
            "context baseline retained pool type/history differs",
        )
        require(
            (
                inner._pool.mode,
                inner._pool.round_index,
                inner._pool.history_sha256,
                inner._pool.policy_sha256,
            )
            == (
                inner.mode,
                inner._history.round_index,
                inner._history.sha256,
                inner.policy_identities,
            ),
            "context baseline retained pool policy binding differs",
        )
    return {
        "mode": inner.mode,
        "run_id": inner.run_id,
        "seed": inner.seed,
        "objective": plain(inner.context),
        "bundle": inner.oracle_bundle_sha256,
        "config": plain(inner.config),
        "units": [unit_binding(unit) for unit in inner._units],
        "history_sha256": None if inner._history is None else inner._history.sha256,
        "pool_sha256": None if inner._pool is None else document_sha(inner._pool),
        "receipts_sha256": document_sha(inner._receipts),
        "training_ids": sorted(inner._training_ids),
        "ten_checkpoint_mixture": inner.protocol_ten_checkpoint_mixture,
    }


def caller_state(units):
    """Separate runtime fingerprint: logical weight hashes do not cover gradients/modes."""
    result = []
    for unit in units:
        for model in (unit.model, unit.reference, unit.initialization.model):
            parameters = []
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                value = (
                    None
                    if gradient is None
                    else (
                        str(gradient.dtype),
                        str(gradient.device),
                        tuple(gradient.shape),
                        digest(gradient.detach().cpu().contiguous().numpy().tobytes()),
                    )
                )
                parameters.append(
                    (name, id(parameter), parameter.requires_grad, id(gradient), value)
                )
            result.append(
                (
                    id(model),
                    tuple(
                        (name, id(module), type(module.training), module.training)
                        for name, module in model.named_modules()
                    ),
                    tuple(parameters),
                )
            )
    return tuple(result)


def pool_identity(pool):
    """Value and structural-type seal; canonical JSON alone erases tuple/list types."""
    if pool is None:
        return None
    require(type(pool) is NativeCandidatePool, "context exact native pool required")

    def structure(value):
        kind = type(value)
        name = kind.__module__ + "." + kind.__qualname__
        if kind in (tuple, list):
            return (name, tuple(structure(item) for item in value))
        if kind in (str, int, float, bool, type(None)):
            return name
        if is_dataclass(value):
            return (
                name,
                tuple(
                    (field.name, structure(getattr(value, field.name))) for field in fields(value)
                ),
            )
        require(False, "unsupported native pool structural type")

    # structure() already contains only JSON-compatible tuples and strings.
    # The canonical encoder emits tuples exactly as the lists plain() creates.
    return document_sha(pool), digest(canonical_json_bytes(structure(pool)))


@dataclass(frozen=True, slots=True)
class ContextEnvelope:
    payload: bytes
    sha256: str

    def __post_init__(self):
        self.document()

    def document(self):
        require(
            type(self.payload) is bytes
            and hash_string(self.sha256)
            and digest(self.payload) == self.sha256,
            "context envelope payload seal differs",
        )
        value = json.loads(self.payload)
        require(
            type(value) is dict and canonical_json_bytes(value) == self.payload,
            "context envelope is not canonical",
        )
        return value


def seal_envelope(document):
    payload = canonical_json_bytes(plain(document))
    return ContextEnvelope(payload, digest(payload))

"""Pure immutable records for the prospective run-owned feature bridge.

Standard library only: the historical artifact audit must not import native
models, Torch, a live service, an oracle or the producer through these records.
Caller-supplied pins bind expectations, not external authentication or timing truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CONTRACT_SHA256 = "4324f90dada88e80afc4fd8b1c9439e58bd3e0aa678c809fc74c0edec0ddcb41"
CONTRACT_PATH = "configs/features/run_feature_cache_bridge_v1.toml"
LAYOUT_SHA256 = "6a2d727734ad58dad01ac842d497dd10602479c984c0512b6d334cd0ccdf7871"
SESSION_BYTES = 2 * 1024**3
RUN_BYTES = 5 * 1024**3
SESSION_ENTRIES = 10000
TR2_CAPACITY_PATH = "configs/features/tr2_grouped_capacity_v1.toml"
TR2_CAPACITY_SHA256 = "d602865a401c240c818daabfff54970609e80849468f692421b19e5e1f9414b0"
TR2_BATCHING_SHA256 = "45be987de4a6e5805e10328bcad2d25f95fca47d024504c0c52a8d0b2f30993e"
SESSION_ADMISSIONS = 128
ACQUIRED_ORIGINS = 16384
METADATA_BYTES = 16 * 1024**2
BINDING_BYTES = 1024**2
ALIASES = {
    "esm_length": ("esm320_plus_normalized_length", 321),
    "esm_length_spectral": ("esm320_plus_normalized_length_plus_spectral32", 353),
}
FULL_ARMS = (
    "counterfactual_softkg_evolutionary_diffusion",
    "ablation_no_spectral_representation",
    "ablation_no_counterfactual_credit",
    "ablation_singleton_kg",
    "ablation_no_endpoint_distillation",
    "ablation_no_kl_controls",
)
CANDIDATE_LIMITS = {
    "tuned_peptide_ga": (0, 0),
    "categorical_diffusion_posthoc": (16, 16),
    "diffusion_reward_kl_no_search": (16, 16),
    "arcadiamp_style_iterative_d3pm": (56, 2),
    "ga_endpoint_distillation_no_kg": (56, 2),
    "tr2d2_style_tree_offpolicy": (2800, 100),
    "mp2d_style_inference_search": (112, 4),
    **{arm: (112, 4) for arm in FULL_ARMS},
}
PURPOSES = ("candidate", "private_preload", "charged", "initial", "terminal", "release", "close")
KINDS = ("open", "raw", "preload", "release", "assemble", "close")
STATUSES = ("completed", "failed")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
WARM_SOURCE_FILES = frozenset(
    {
        "src/amp_challenge/representations/candidate_features.py",
        "src/amp_challenge/representations/peptide_esm.py",
        "src/amp_challenge/representations/laplacian.py",
        "src/amp_challenge/representations/warm_candidate_features.py",
        "src/amp_challenge/representations/fixed_shape_esm.py",
        "integrations/ampdiffusion/esm2_candidate_features_worker.py",
        "integrations/ampdiffusion/esm2_peptide_features_worker.py",
        "integrations/ampdiffusion/esm2_peptide_source_pins.json",
        "integrations/ampdiffusion/esm2_warm_candidate_worker.py",
    }
)
TR2_CAPACITY_SOURCE_FILES = frozenset(
    {
        TR2_CAPACITY_PATH,
        "configs/search/native_tr2_feature_batching_v1.toml",
        "src/amp_challenge/representations/run_feature_cache_records.py",
    }
)
IMPLEMENTATION_SOURCE_FILES = frozenset(
    {
        CONTRACT_PATH,
        "src/amp_challenge/representations/run_feature_cache_records.py",
        "src/amp_challenge/representations/run_feature_cache_bridge.py",
        "src/amp_challenge/representations/run_feature_cache_views.py",
        "src/amp_challenge/representations/run_feature_cache_verify.py",
        "cluster/slurm/audit_run_feature_cache_bridge_v1.py",
        "src/amp_challenge/generators/diffusion/native_search_posterior.py",
        "src/amp_challenge/generators/diffusion/native_evolution_posterior.py",
        "src/amp_challenge/generators/diffusion/native_baseline_operators.py",
        "src/amp_challenge/generators/search/verified_charged_history.py",
        "src/amp_challenge/generators/search/peptide_ga_tunable_v2_records.py",
        "src/amp_challenge/models/charged_probability_learner.py",
        "src/amp_challenge/models/feature_posterior.py",
    }
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    require(type(payload) is bytes, "feature digest requires exact bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def json_object(payload, *, maximum=BINDING_BYTES):
    require(type(payload) is bytes and 0 < len(payload) <= maximum, "feature record bytes differ")
    result = json.loads(payload)
    require(
        type(result) is dict and canonical(result) == payload, "feature record is not canonical"
    )
    return result


def pin(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def identifier(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def finite_clock(value):
    require(type(value) in (int, float), "feature clock must be an exact numeric non-bool")
    try:
        valid = math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    require(valid, "feature clock must be finite and representable")
    return value


def sequences_checked(sequences, *, empty=False, unique=False, maximum=128):
    require(
        type(sequences) is tuple and (0 if empty else 1) <= len(sequences) <= maximum,
        "feature sequence inventory bound differs",
    )
    require(
        all(
            type(seq) is str and 8 <= len(seq) <= 50 and not set(seq) - _ALPHABET
            for seq in sequences
        ),
        "feature sequence support differs",
    )
    require(
        not unique or len(set(sequences)) == len(sequences), "feature sequence duplicates differ"
    )
    return tuple(sha256(seq.encode("ascii")) for seq in sequences)


def representation_checked(raw_name, binding_name=None):
    require(type(raw_name) is str and raw_name in ALIASES, "feature saved-array name differs")
    alias, width = ALIASES[raw_name]
    require(
        binding_name is None or (type(binding_name) is str and binding_name == alias),
        "feature representation alias differs",
    )
    return alias, width


def check_tr2_capacity(profile, repository=None):
    """Opt-in bound assumes one ordered 120-row preload and 28 x 10 grouped requests."""
    if profile is None:
        return None
    require(
        type(profile) is dict
        and set(profile)
        == {"configuration_sha256", "feature_batching_configuration_sha256", "shared_sequence_ids"},
        "TR2 capacity profile schema differs",
    )
    root = Path(__file__).resolve().parents[3]
    for source_root in {root, Path(repository) if repository is not None else root}:
        require(
            sha256((source_root / TR2_CAPACITY_PATH).read_bytes()) == TR2_CAPACITY_SHA256,
            "TR2 capacity configuration bytes differ",
        )
        require(
            sha256(
                (source_root / "configs/search/native_tr2_feature_batching_v1.toml").read_bytes()
            )
            == TR2_BATCHING_SHA256,
            "TR2 grouped configuration bytes differ",
        )
    ids = profile["shared_sequence_ids"]
    require(
        profile["configuration_sha256"] == TR2_CAPACITY_SHA256
        and profile["feature_batching_configuration_sha256"] == TR2_BATCHING_SHA256
        and type(ids) is list
        and len(ids) == 120
        and len(set(ids)) == 120
        and all(pin(value) for value in ids),
        "TR2 capacity pins/shared preload differ",
    )
    return profile


def check_tr2_capacity_request(profile, ordinal, sequences):
    if profile is None:
        return
    check_tr2_capacity(profile)
    ids = [sha256(sequence.encode("ascii")) for sequence in sequences]
    require(
        (ordinal == 0 and ids == profile["shared_sequence_ids"])
        or (0 < ordinal < 281 and 1 <= len(ids) <= 80),
        "TR2 capacity requires exact first preload then at most80 candidate rows",
    )


@dataclass(frozen=True, slots=True)
class FeatureRunBinding:
    run_id: str
    arm_id: str
    seed: int
    objective_context_sha256: str
    original_epoch: float
    original_deadline: float
    repository: str
    bundle: str
    run_root: str
    session_id: str
    expected_commit: str
    warm_source_payload: bytes
    implementation_source_payload: bytes
    runtime_payload: bytes
    model_sha256: str
    job_id: str
    allocation_sha256: str
    physical_run_limit: int
    arm_budget_receipt_sha256: str
    tr2_grouped_capacity_payload: bytes | None = None

    def __post_init__(self):
        require(
            identifier(self.run_id) and identifier(self.session_id),
            "feature run/session ID differs",
        )
        require(type(self.arm_id) is str and self.arm_id in CANDIDATE_LIMITS, "feature arm differs")
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "feature seed differs")
        for value in (
            self.objective_context_sha256,
            self.model_sha256,
            self.allocation_sha256,
            self.arm_budget_receipt_sha256,
        ):
            require(pin(value), "feature external binding pin differs")
        epoch, deadline = finite_clock(self.original_epoch), finite_clock(self.original_deadline)
        require(
            0 <= epoch < deadline and deadline - epoch <= 7200,
            "feature original clock allowance differs",
        )
        for value in (self.repository, self.bundle, self.run_root):
            require(
                type(value) is str
                and Path(value).is_absolute()
                and str(Path(value)) == value
                and ".." not in Path(value).parts,
                "feature root must be an explicit canonical absolute path",
            )
        require(
            type(self.expected_commit) is str
            and re.fullmatch(r"[0-9a-f]{40}", self.expected_commit) is not None,
            "feature source commit differs",
        )
        source = json_object(self.warm_source_payload)
        require(
            set(source) == {"git_commit", "files"}
            and source["git_commit"] == self.expected_commit
            and type(source["files"]) is dict
            and set(source["files"])
            == WARM_SOURCE_FILES
            | (
                TR2_CAPACITY_SOURCE_FILES
                if self.tr2_grouped_capacity_payload is not None
                else frozenset()
            ),
            "feature warm-source inventory differs",
        )
        implementation = json_object(self.implementation_source_payload)
        require(
            set(implementation) == IMPLEMENTATION_SOURCE_FILES,
            "feature implementation-source inventory differs",
        )
        for mapping in (source["files"], implementation):
            require(
                mapping
                and all(
                    type(key) is str
                    and not Path(key).is_absolute()
                    and str(Path(key)) == key
                    and ".." not in Path(key).parts
                    and pin(value)
                    for key, value in mapping.items()
                ),
                "feature source file pins differ",
            )
        _ = self.tr2_grouped_capacity
        require(json_object(self.runtime_payload), "feature runtime binding is empty")
        require(
            type(self.job_id) is str and self.job_id.isascii() and self.job_id.isdigit(),
            "feature allocation job differs",
        )
        require(
            type(self.physical_run_limit) is int and self.physical_run_limit > 0,
            "feature arm physical admission bound differs",
        )
        require(
            self.arm_id not in FULL_ARMS or self.physical_run_limit == 128,
            "full-method aggregate request ceiling differs",
        )

    @property
    def tr2_grouped_capacity(self):
        if self.tr2_grouped_capacity_payload is None:
            return None
        profile = check_tr2_capacity(
            json_object(self.tr2_grouped_capacity_payload), self.repository
        )
        require(
            self.arm_id == "tr2d2_style_tree_offpolicy" and self.physical_run_limit == 281,
            "TR2 capacity requires grouped TR2 arm and281 requests",
        )
        return profile

    @property
    def admission_limit(self):
        return min(
            281 if self.tr2_grouped_capacity else SESSION_ADMISSIONS, self.physical_run_limit
        )

    @property
    def origin_limit(self):
        return 22520 if self.tr2_grouped_capacity else ACQUIRED_ORIGINS

    @property
    def candidate_limits(self):
        return (280, 10) if self.tr2_grouped_capacity else CANDIDATE_LIMITS[self.arm_id]

    @property
    def session_root(self):
        return Path(self.run_root) / f"feature-session-{self.session_id}"

    @property
    def audit_root(self):
        return Path(self.run_root) / f"feature-audit-{self.session_id}"

    def document(self):
        self.__post_init__()
        result = asdict(self)
        result.pop("tr2_grouped_capacity_payload")
        if self.tr2_grouped_capacity is not None:
            result["tr2_grouped_capacity"] = self.tr2_grouped_capacity
        for name in ("warm_source_payload", "implementation_source_payload", "runtime_payload"):
            result[name.removesuffix("_payload")] = json_object(result.pop(name))
        result["contract_sha256"] = CONTRACT_SHA256
        result["layout_sha256"] = LAYOUT_SHA256
        require(
            len(canonical(result)) <= BINDING_BYTES, "feature run binding metadata exceeds bound"
        )
        return result

    @property
    def sha256(self):
        return sha256(canonical(self.document()))


@dataclass(frozen=True, slots=True)
class FeatureIntent:
    purpose: str
    history_sha256: str
    objective_context_sha256: str
    round_index: int
    logical_ordinal: int
    expected_previous_head: str
    original_effective_deadline: float

    def __post_init__(self):
        require(type(self.purpose) is str and self.purpose in PURPOSES, "feature purpose differs")
        require(
            all(
                pin(value)
                for value in (
                    self.history_sha256,
                    self.objective_context_sha256,
                    self.expected_previous_head,
                )
            ),
            "feature intent pin differs",
        )
        require(
            type(self.round_index) is int and 1 <= self.round_index <= 29,
            "feature intent round differs",
        )
        require(
            type(self.logical_ordinal) is int and self.logical_ordinal >= 0,
            "feature logical opportunity ordinal differs",
        )
        finite_clock(self.original_effective_deadline)

    def document(self):
        self.__post_init__()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrivateFeatureRelease:
    run_id: str
    history_sha256: str
    objective_context_sha256: str
    source_sha256: str
    receipt_sha256: str
    expected_previous_release_head: str
    revealed_sequence_ids: tuple[str, ...]

    def __post_init__(self):
        require(identifier(self.run_id), "feature release run differs")
        require(
            all(
                pin(value)
                for value in (
                    self.history_sha256,
                    self.objective_context_sha256,
                    self.source_sha256,
                    self.receipt_sha256,
                    self.expected_previous_release_head,
                )
            ),
            "feature release authority pins differ",
        )
        require(
            type(self.revealed_sequence_ids) is tuple
            and len(self.revealed_sequence_ids) <= 512
            and len(set(self.revealed_sequence_ids)) == len(self.revealed_sequence_ids)
            and all(pin(value) for value in self.revealed_sequence_ids),
            "feature release exact subset differs",
        )

    def document(self):
        self.__post_init__()
        return asdict(self)

    @property
    def sha256(self):
        return sha256(canonical(self.document()))


@dataclass(frozen=True, slots=True)
class FeatureCounters:
    logical_opportunities: int = 0
    candidate_opportunities: int = 0
    auxiliary_opportunities: int = 0
    assembly_opportunities: int = 0
    cache_only_opportunities: int = 0
    physical_attempts: int = 0
    physical_completed: int = 0
    physical_failed: int = 0
    real_rows_dispatched: int = 0
    padded_rows_dispatched: int = 0
    full_accounted_requests: int = 0
    acquired_origins: int = 0
    visible_rows: int = 0
    private_rows: int = 0
    released_rows: int = 0
    candidate_wave_counts: tuple[int, ...] = (0,) * 28

    def __post_init__(self):
        require(
            all(
                type(value) is int and value >= 0
                for key, value in asdict(self).items()
                if key != "candidate_wave_counts"
            ),
            "feature counter type/value differs",
        )
        require(
            type(self.candidate_wave_counts) is tuple
            and len(self.candidate_wave_counts) == 28
            and all(type(value) is int and value >= 0 for value in self.candidate_wave_counts),
            "feature wave counters differ",
        )

    def document(self):
        self.__post_init__()
        return asdict(self)


def legacy_history_document(history):
    """Normalize explicit legacy defaults without admitting a new bridge budget.

    VerifiedHistorySnapshot now serializes optional budget fields, while its
    historical v1 digest deliberately omits their default values. Retained
    pre-amendment JSON has no such fields. Both representations name the same
    legacy history; prospective budgets must use their separate controller.
    """
    require(type(history) is dict, "legacy assembly history must be a document")
    result = dict(history)
    defaults = {"initial_charge_count": 64, "max_rounds": 28, "charges_per_round": 16}
    present = set(defaults).intersection(result)
    require(not present or present == set(defaults), "legacy assembly budget metadata is partial")
    for name in present:
        require(
            type(result[name]) is int and result[name] == defaults[name],
            "legacy feature bridge cannot admit a prospective budget",
        )
        result.pop(name)
    return result


@dataclass(frozen=True, slots=True)
class FeatureAssemblyBinding:
    """External exact charged-history/eligibility pins, never hidden oracle access."""

    raw_history_payload: bytes
    history_sha256: str
    eligibility_source_sha256: str
    eligibility_receipt_sha256: str
    eligible_query_ids: tuple[str, ...]
    transform_sha256: str

    def __post_init__(self):
        history = json_object(self.raw_history_payload)
        require(
            all(
                pin(value)
                for value in (
                    self.history_sha256,
                    self.eligibility_source_sha256,
                    self.eligibility_receipt_sha256,
                    self.transform_sha256,
                )
            ),
            "feature assembly external pins differ",
        )
        expected = sha256(
            b"amp/verified-charged-history/v1\0"
            + canonical(legacy_history_document(history)).rstrip(b"\n")
        )
        require(expected == self.history_sha256, "feature assembly raw history digest differs")
        require(
            type(self.eligible_query_ids) is tuple
            and len(self.eligible_query_ids) <= 512
            and len(set(self.eligible_query_ids)) == len(self.eligible_query_ids)
            and all(type(value) is str and value for value in self.eligible_query_ids),
            "feature assembly eligibility inventory differs",
        )

    def document(self):
        self.__post_init__()
        result = asdict(self)
        result["raw_history"] = json_object(result.pop("raw_history_payload"))
        return result

    @property
    def sha256(self):
        return sha256(canonical(self.document()))


# Nested private schema (JSON lists are canonicalized tuples on disk):
# staging={started: {path,sha256}|None, dispatch: {path,sha256}|None}.
# session={config,config_sha256,ready_sha256,complete_sha256,failed_sha256}.
# batch=None or {root,ordinal,expected_previous_head,request_sha256,
#   command_sha256,response_sha256,invocation_sha256,manifest_sha256,dispatched,returned}.
# A failed batch may have null not-yet-published artifact hashes. Its request is
# reconstructable from operation sequences and the staging dispatch document.
# returned reports a successful API return, not independently attested GPU work.
# Later wrapper failure preserves that completed call but commits no cache rows.
# real/padded rows dispatched describe charged/planned work, not measured forwards.
# row_origins=[{sequence,sequence_id,acquisition_operation,batch_ordinal,row,
#   manifest_sha256,raw_sha256s:{esm_length,esm_length_spectral},origin_sha256}].
# scatter indexes row_origins in exact output order, including repeats.
# inventory_{before,after}={entries:[{path,type,bytes,sha256}],
#   regular_bytes,session_bytes,session_entries,pending_final_path,pending_final_bytes}.
# Directory entries contain only path/type; before has pending path=None/bytes=0.
# The after inventory excludes its own pending final receipt; its exact sealed
# bytes/hash are supplied by FeatureCacheReceipt, avoiding a self-hash cycle.
# All other staging/session/failure files are inventoried. No filesystem-hard-quota
# or independently authenticated elapsed/resource-use claim follows from this list.
# assembly=None or {binding:FeatureAssemblyBinding.document(),selected_query_ids,
#   feature_sequence_ids,child_receipt_sha256s}. Parent assembly on child raw
# operations={binding_sha256,intent:FeatureIntent.document()}; it freezes the
# original parent epoch/history/deadline while children individually commit only
# features. Failed parent returns no learner input and terminalizes the bridge.
# timings=[{phase,monotonic}], with original caller bounds retained throughout.


# All private final operations have these keys. STARTED/dispatch staging files
# are pinned evidence, never accepted cache heads. A failed final operation may
# advance evidence_head but must leave accepted_head/cache state at its predecessor.
RECEIPT_KEYS = frozenset(
    {
        "artifact",
        "contract_sha256",
        "binding",
        "operation_ordinal",
        "kind",
        "status",
        "previous_evidence_head",
        "previous_accepted_head",
        "intent",
        "parent_assembly",
        "sequences",
        "representation",
        "staging",
        "batch",
        "session",
        "release",
        "assembly",
        "row_origins",
        "scatter",
        "public_payload",
        "public_sha256",
        "before_counters",
        "after_counters",
        "inventory_before",
        "inventory_after",
        "timings",
        "failure",
        "oracle_calls",
        "scientific_evidence_accepted",
        "production_eligible",
    }
)


@dataclass(frozen=True, slots=True)
class FeatureCacheReceipt:
    payload: bytes
    sha256: str

    def __post_init__(self):
        require(
            type(self.payload) is bytes
            and pin(self.sha256)
            and sha256(self.payload) == self.sha256,
            "feature receipt exact bytes/seal differ",
        )

    def document(self):
        self.__post_init__()
        result = json_object(self.payload, maximum=METADATA_BYTES)
        require(set(result) == RECEIPT_KEYS, "feature private receipt schema differs")
        require(
            result["artifact"] == "run_feature_cache_operation_v1"
            and result["contract_sha256"] == CONTRACT_SHA256,
            "feature private receipt artifact/config differs",
        )
        require(
            type(result["kind"]) is str
            and result["kind"] in KINDS
            and type(result["status"]) is str
            and result["status"] in STATUSES,
            "feature private receipt kind/status differs",
        )
        require(
            type(result["operation_ordinal"]) is int and result["operation_ordinal"] >= 0,
            "feature private receipt ordinal differs",
        )
        require(
            type(result["oracle_calls"]) is int
            and result["oracle_calls"] == 0
            and result["scientific_evidence_accepted"] is False
            and result["production_eligible"] is False,
            "feature receipt authority differs",
        )
        return result

    @classmethod
    def seal(cls, document):
        payload = canonical(document)
        result = cls(payload, sha256(payload))
        result.document()
        return result


def public_document(sequences, representation, row_sha256s, *, context_sha256, history_sha256):
    ids = sequences_checked(sequences, empty=True, maximum=512)
    alias, width = representation_checked(representation)
    require(
        type(row_sha256s) is tuple
        and len(row_sha256s) == len(ids)
        and all(pin(value) for value in row_sha256s),
        "feature public row hashes differ",
    )
    require(pin(context_sha256) and pin(history_sha256), "feature public context/history differs")
    return {
        "artifact": "run_feature_requested_rows_v1",
        "sequence_ids": list(ids),
        "representation": alias,
        "width": width,
        "row_sha256s": list(row_sha256s),
        "objective_context_sha256": context_sha256,
        "history_sha256": history_sha256,
        "oracle_calls": 0,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }

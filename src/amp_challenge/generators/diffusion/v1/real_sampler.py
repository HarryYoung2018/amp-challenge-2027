"""Fail-closed native-v1 engineering sampler with no production authority.

The schema-v1 contract has an empty accepted-checkpoint inventory, and every
constructible identity, provider, candidate, and result is fixture-only.  The
public loader fails before touching any caller-supplied checkpoint path.  A
separately labelled surface lets tests exercise the real R128 tensor adapter
without creating production authority or scientific evidence.  A positive
checkpoint loader requires a separately reviewed successor contract and code
path; it cannot be enabled by populating runtime hashes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import math
import os
import platform
import re
import stat
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from amp_challenge.generators.diffusion import sampling as native_sampling
from amp_challenge.generators.diffusion.categorical import (
    CosineMaskSchedule,
    PeptideVocabulary,
)
from amp_challenge.generators.diffusion.v1.pilot_bridge import (
    count_prior_training_bridge,
)
from amp_challenge.generators.diffusion.v1.pilot_checkpoint import canonical_json_bytes
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    CONFIG_SHA256 as PILOT_CONTRACT_SHA256,
)
from amp_challenge.generators.diffusion.v1.pilot_contract import (
    PARENT_CONFIG_SHA256,
    NativeDiffusionV1PilotContract,
    _read_contract_bytes,
)
from amp_challenge.generators.diffusion.v1.pilot_data import AuthenticatedCountPrior
from amp_challenge.generators.diffusion.v1.pilot_model import (
    R128Denoiser,
    assert_r128_deterministic_runtime,
    assert_r128_model_execution_surface,
    build_r128_model_from_contract,
    canonical_r128_model_sha256,
)
from amp_challenge.generators.diffusion.v1.pilot_training import count_prior_logits
from amp_challenge.sequences import canonical_sequence_id

ADAPTER_CONFIG_SHA256 = "6a9e2853185678a00f87654dedae318bf67e48ba69580f1309fa926142c3b5da"
ADAPTER_ARTIFACT = "native_categorical_diffusion_v1_real_sampler_adapter"
ADAPTER_STATUS = "blocked_no_independently_accepted_checkpoint"
ENGINEERING_FIXTURE_EVIDENCE = "engineering_fixture_only_not_scientific_evidence"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
DIFFUSION_LEVELS = 64
BATCH_SEQUENCE_CAP = 128
PROPOSAL_COUNT_CAP = 65_536
REQUEST_WALL_SECONDS = 7_200
CALIBRATION_BINS = ((1, 16), (17, 32), (33, 48), (49, 64))
ALLOWED_RESIDUAL_LAMBDAS = (0.125, 0.25, 0.5, 0.75, 1.0)
ALLOWED_TEMPERATURES = (1.0, 1.25, 1.5, 2.0, 3.0)
ALLOWED_BACKOFF_EPSILONS = (0.0, 0.02, 0.05)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_DRAW_DOMAIN = b"amp-challenge/native-categorical-diffusion-v1/real-sampler-draw/v1\0"
_IDENTITY_DOMAIN = b"amp-challenge/native-categorical-diffusion-v1/sampler-identity/v1\0"
_REQUEST_DOMAIN = b"amp-challenge/native-categorical-diffusion-v1/sampling-request/v1\0"
_PROVENANCE_DOMAIN = b"amp-challenge/native-categorical-diffusion-v1/engineering-provenance/v1\0"
_COUNT_TENSOR_DOMAIN = b"amp-challenge/native-categorical-diffusion-v1/count-tensor/v1\0"
_FIXTURE_CONSTRUCTION_TOKEN = object()
_R128_FORWARD = R128Denoiser.forward
_MAX_ADAPTER_CONTRACT_BYTES = 64 << 10
_MAX_ENGINEERING_PROVENANCE_FILE_BYTES = 8 << 20
_ENGINEERING_IMPLEMENTATION_PATHS = (
    "configs/diffusion/native_v1_real_sampler_adapter_v1.toml",
    "pyproject.toml",
    "src/amp_challenge/generators/diffusion/categorical.py",
    "src/amp_challenge/generators/diffusion/sampling.py",
    "src/amp_challenge/generators/diffusion/v1/contract.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_bridge.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_checkpoint.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_contract.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_data.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_model.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_rng.py",
    "src/amp_challenge/generators/diffusion/v1/pilot_training.py",
    "src/amp_challenge/generators/diffusion/v1/real_sampler.py",
    "src/amp_challenge/sequences.py",
    "uv.lock",
)

LogitProvider = Callable[
    [NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]],
    NDArray[np.floating],
]


class NativeV1CheckpointNotAcceptedError(RuntimeError):
    """The content-pinned contract supplies no independent acceptance authority."""


@dataclass(frozen=True, slots=True)
class NativeV1RealSamplerContract:
    """Typed, byte-bound adapter contract with no accepted checkpoint today."""

    config_sha256: str
    payload: bytes
    accepted_checkpoint_sets: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.config_sha256 != ADAPTER_CONFIG_SHA256:
            raise ValueError("schema-v1 contract identity differs from the frozen adapter")
        if (
            type(self.payload) is not bytes
            or not 0 < len(self.payload) <= _MAX_ADAPTER_CONTRACT_BYTES
            or hashlib.sha256(self.payload).hexdigest() != self.config_sha256
        ):
            raise ValueError("schema-v1 contract payload is not its bounded frozen content")
        if type(self.accepted_checkpoint_sets) is not tuple or self.accepted_checkpoint_sets != ():
            raise ValueError("schema-v1 contract cannot construct an accepted checkpoint pin")

    def revalidate(self) -> NativeV1RealSamplerContract:
        if type(self) is not NativeV1RealSamplerContract:
            raise TypeError("sampler contract must be exact NativeV1RealSamplerContract")
        rebuilt = _parse_adapter_contract(self.payload)
        if self != rebuilt:
            raise ValueError("native-v1 real-sampler contract changed after parsing")
        return self

    def checkpoint_pin(self, checkpoint_set_id: str) -> None:
        _identifier(checkpoint_set_id, label="checkpoint_set_id")
        raise NativeV1CheckpointNotAcceptedError(
            "no independently accepted native-v1 checkpoint is pinned by this adapter contract"
        )


@dataclass(frozen=True, slots=True)
class NativeV1SamplerPreflight:
    """Path-free report of the current fail-closed acceptance state."""

    adapter_config_sha256: str
    status: str
    accepted_checkpoint_set_count: int
    blockers: tuple[str, ...]
    execution_authorized: bool
    scientific_evidence_accepted: bool
    production_input_eligible: bool
    automatic_generator_mixture_eligible: bool

    def __post_init__(self) -> None:
        if self.adapter_config_sha256 != ADAPTER_CONFIG_SHA256:
            raise ValueError("preflight adapter identity differs from the frozen contract")
        if self.status != ADAPTER_STATUS:
            raise ValueError("preflight status differs from the frozen blocked state")
        if type(self.accepted_checkpoint_set_count) is not int or (
            self.accepted_checkpoint_set_count != 0
        ):
            raise ValueError("schema-v1 preflight cannot report an accepted checkpoint")
        if self.blockers != ("no_independently_accepted_checkpoint",):
            raise ValueError("schema-v1 preflight must retain its exact blocker")
        for field in (
            "execution_authorized",
            "scientific_evidence_accepted",
            "production_input_eligible",
            "automatic_generator_mixture_eligible",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"schema-v1 preflight {field} must be exact false")

    def as_document(self) -> Mapping[str, object]:
        self.revalidate()
        return {
            "accepted_checkpoint_set_count": self.accepted_checkpoint_set_count,
            "adapter_config_sha256": self.adapter_config_sha256,
            "artifact": "native_categorical_diffusion_v1_real_sampler_preflight",
            "automatic_generator_mixture_eligible": self.automatic_generator_mixture_eligible,
            "blockers": list(self.blockers),
            "execution_authorized": self.execution_authorized,
            "production_input_eligible": self.production_input_eligible,
            "runtime_scope": "engineering_fixture_only",
            "schema_version": 1,
            "scientific_evidence_accepted": self.scientific_evidence_accepted,
            "status": self.status,
        }

    def revalidate(self) -> NativeV1SamplerPreflight:
        if type(self) is not NativeV1SamplerPreflight:
            raise TypeError("preflight must have its exact runtime type")
        self.__post_init__()
        return self


def load_native_v1_real_sampler_contract(
    path: str | os.PathLike[str],
) -> NativeV1RealSamplerContract:
    """Read and authenticate the frozen adapter contract without following links."""

    payload = _read_contract_bytes(Path(os.path.abspath(os.fspath(path))))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ADAPTER_CONFIG_SHA256:
        raise ValueError(f"native-v1 real-sampler adapter contract SHA-256 mismatch: {digest}")
    return _parse_adapter_contract(payload)


def preflight_native_v1_real_sampler(
    path: str | os.PathLike[str],
) -> NativeV1SamplerPreflight:
    """Return the blocked state; this function never opens a checkpoint path."""

    contract = load_native_v1_real_sampler_contract(path)
    contract.revalidate()
    accepted_count = len(contract.accepted_checkpoint_sets)
    blockers = () if accepted_count else ("no_independently_accepted_checkpoint",)
    # A future contract version may pin inputs, but this frozen v1 contract
    # deliberately cannot become execution authority in place.
    return NativeV1SamplerPreflight(
        adapter_config_sha256=contract.config_sha256,
        status=ADAPTER_STATUS,
        accepted_checkpoint_set_count=accepted_count,
        blockers=blockers,
        execution_authorized=False,
        scientific_evidence_accepted=False,
        production_input_eligible=False,
        automatic_generator_mixture_eligible=False,
    )


def _parse_adapter_contract(payload: bytes) -> NativeV1RealSamplerContract:
    if type(payload) is not bytes or not payload:
        raise TypeError("adapter contract payload must be non-empty exact bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ADAPTER_CONFIG_SHA256:
        raise ValueError("native-v1 real-sampler adapter contract is not the frozen payload")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            "native-v1 real-sampler adapter contract is not valid UTF-8 TOML"
        ) from error
    # The byte hash authenticates every table, ceiling, and isolation flag.  We
    # only decode the fields needed by runtime control flow; duplicating the
    # whole TOML schema here would create a second, drift-prone contract.
    required = {
        "schema_version": 1,
        "artifact": ADAPTER_ARTIFACT,
        "status": ADAPTER_STATUS,
        "runtime_scope": "engineering_fixture_only",
        "execution_authorized": False,
        "scientific_evidence_accepted": False,
        "production_input_eligible": False,
        "automatic_generator_mixture_eligible": False,
        "production_identity_construction_allowed": False,
        "production_provider_construction_allowed": False,
    }
    for field, expected in required.items():
        if type(raw.get(field)) is not type(expected) or raw[field] != expected:
            raise ValueError(f"adapter contract {field} differs from its frozen value")
    native = raw.get("native_v1")
    sampling = raw.get("sampling")
    isolation = raw.get("isolation")
    provenance = raw.get("provenance")
    resources = raw.get("resources")
    gate = raw.get("current_gate")
    if not all(
        type(value) is dict for value in (native, sampling, isolation, provenance, resources, gate)
    ):
        raise TypeError("adapter contract runtime tables must be exact TOML tables")
    assert isinstance(native, dict) and isinstance(sampling, dict)
    assert isinstance(isolation, dict) and isinstance(provenance, dict)
    assert isinstance(resources, dict) and isinstance(gate, dict)
    runtime_values = (
        (native, "pilot_contract_sha256", PILOT_CONTRACT_SHA256),
        (native, "parent_contract_sha256", PARENT_CONFIG_SHA256),
        (native, "model_variant", "R128"),
        (sampling, "alphabet", ALPHABET),
        (sampling, "minimum_length", MIN_LENGTH),
        (sampling, "maximum_length", MAX_LENGTH),
        (sampling, "diffusion_levels", DIFFUSION_LEVELS),
        (sampling, "reverse_rule", "fixed_count_monotone_confidence_maskgit"),
        (sampling, "batch_sequence_cap", BATCH_SEQUENCE_CAP),
        (sampling, "proposal_count_cap", PROPOSAL_COUNT_CAP),
        (resources, "execution_scope", "focused_engineering_fixture_only"),
        (resources, "gpus", 0),
        (resources, "request_wall_seconds", REQUEST_WALL_SECONDS),
        (resources, "device", "cpu"),
        (isolation, "oracle_inputs_allowed", False),
        (isolation, "oracle_imports_allowed", False),
        (provenance, "observation_scope", "self_observed_engineering_only"),
        (provenance, "independent_authority", False),
        (provenance, "scientific_evidence_accepted", False),
        (gate, "public_loader_must_fail_before_opening_checkpoint_bundle_paths", True),
        (gate, "schema_v1_fixture_only", True),
        (gate, "production_serialization_allowed", False),
    )
    for table, field, expected in runtime_values:
        if type(table.get(field)) is not type(expected) or table[field] != expected:
            raise ValueError(f"adapter contract {field} differs from its frozen value")

    raw_pins = raw.get("accepted_checkpoint_sets")
    if type(raw_pins) is not list:
        raise TypeError("accepted_checkpoint_sets must be an exact TOML array")
    if raw_pins:
        raise ValueError(
            "frozen adapter v1 cannot gain a checkpoint pin; issue a new content hash/version"
        )
    return NativeV1RealSamplerContract(
        config_sha256=digest,
        payload=payload,
        accepted_checkpoint_sets=(),
    )


@dataclass(frozen=True, slots=True)
class NativeV1EngineeringRuntimeProvenance:
    """Observed, non-authoritative runtime and source identity for fixture replay."""

    implementation_sha256: str
    implementation_files: tuple[tuple[str, str], ...]
    dependency_lock_sha256: str
    deterministic_runtime_sha256: str
    python_version: str
    python_implementation: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str | None
    safetensors_version: str
    device: Literal["cpu"]
    observation_scope: Literal["self_observed_engineering_only"]
    independent_authority: bool
    scientific_evidence_accepted: bool

    def __post_init__(self) -> None:
        _sha256(self.implementation_sha256, label="implementation_sha256")
        _sha256(self.dependency_lock_sha256, label="dependency_lock_sha256")
        _sha256(
            self.deterministic_runtime_sha256,
            label="deterministic_runtime_sha256",
        )
        if (
            type(self.implementation_files) is not tuple
            or tuple(path for path, _ in self.implementation_files)
            != _ENGINEERING_IMPLEMENTATION_PATHS
        ):
            raise ValueError("engineering implementation file inventory is not exact")
        for path, digest in self.implementation_files:
            if (
                type(path) is not str
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
            ):
                raise ValueError("engineering implementation path is invalid")
            _sha256(digest, label=f"engineering implementation digest for {path}")
        implementation_document = {
            "artifact": "native_categorical_diffusion_v1_engineering_implementation",
            "files": [
                {"path": path, "sha256": digest} for path, digest in self.implementation_files
            ],
            "schema_version": 1,
        }
        digest = hashlib.sha256()
        digest.update(_PROVENANCE_DOMAIN)
        digest.update(canonical_json_bytes(implementation_document))
        if digest.hexdigest() != self.implementation_sha256:
            raise ValueError("engineering implementation digest differs from its inventory")
        lock_digest = dict(self.implementation_files)["uv.lock"]
        if self.dependency_lock_sha256 != lock_digest:
            raise ValueError("dependency lock digest differs from the implementation inventory")
        for field in (
            "python_version",
            "python_implementation",
            "numpy_version",
            "torch_version",
            "safetensors_version",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"engineering runtime {field} must be a non-empty exact string")
        if self.torch_cuda_version is not None and (
            type(self.torch_cuda_version) is not str or not self.torch_cuda_version.strip()
        ):
            raise ValueError("engineering runtime torch_cuda_version is invalid")
        if self.device != "cpu":
            raise ValueError("schema-v1 engineering provenance requires exact CPU")
        if self.observation_scope != "self_observed_engineering_only":
            raise ValueError("engineering provenance must remain explicitly self-observed")
        if self.independent_authority is not False:
            raise ValueError("engineering provenance cannot claim independent authority")
        if self.scientific_evidence_accepted is not False:
            raise ValueError("engineering provenance cannot claim scientific evidence")

    def as_document(self) -> Mapping[str, object]:
        self.revalidate()
        return {
            "artifact": "native_categorical_diffusion_v1_engineering_runtime_provenance",
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "deterministic_runtime_sha256": self.deterministic_runtime_sha256,
            "device": self.device,
            "implementation_files": [
                {"path": path, "sha256": digest} for path, digest in self.implementation_files
            ],
            "implementation_sha256": self.implementation_sha256,
            "independent_authority": self.independent_authority,
            "numpy_version": self.numpy_version,
            "observation_scope": self.observation_scope,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "safetensors_version": self.safetensors_version,
            "schema_version": 1,
            "scientific_evidence_accepted": self.scientific_evidence_accepted,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_version": self.torch_version,
        }

    def revalidate(self) -> NativeV1EngineeringRuntimeProvenance:
        if type(self) is not NativeV1EngineeringRuntimeProvenance:
            raise TypeError("engineering provenance must have its exact runtime type")
        self.__post_init__()
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(dict(self.as_document()))).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeV1SamplerIdentity:
    """Path-free identity of one explicitly non-authoritative fixture sampler."""

    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"]
    checkpoint_set_id: str
    adapter_config_sha256: str
    native_pilot_contract_sha256: str
    native_parent_contract_sha256: str
    model_state_sha256: str
    checkpoint_file_sha256: str | None
    checkpoint_metadata_sha256: str | None
    count_prior_sha256: str
    bundle_manifest_sha256: str | None
    independent_receipt_sha256: str | None
    residual_lambda_by_bin: tuple[float, float, float, float]
    temperature_by_bin: tuple[float, float, float, float]
    sampling_backoff_epsilon: float
    engineering_provenance: NativeV1EngineeringRuntimeProvenance
    independently_audited: bool
    production_input_eligible: bool

    def __post_init__(self) -> None:
        if self.evidence_class != ENGINEERING_FIXTURE_EVIDENCE:
            raise ValueError("schema-v1 sampler identity must remain an engineering fixture")
        if self.checkpoint_set_id != "engineering_fixture_no_checkpoint":
            raise ValueError("schema-v1 identity must retain its no-checkpoint fixture identifier")
        for label, digest in (
            ("adapter_config_sha256", self.adapter_config_sha256),
            ("native_pilot_contract_sha256", self.native_pilot_contract_sha256),
            ("native_parent_contract_sha256", self.native_parent_contract_sha256),
            ("model_state_sha256", self.model_state_sha256),
            ("count_prior_sha256", self.count_prior_sha256),
        ):
            _sha256(digest, label=label)
        optional_hashes = (
            ("checkpoint_file_sha256", self.checkpoint_file_sha256),
            ("checkpoint_metadata_sha256", self.checkpoint_metadata_sha256),
            ("bundle_manifest_sha256", self.bundle_manifest_sha256),
            ("independent_receipt_sha256", self.independent_receipt_sha256),
        )
        if any(value is not None for _, value in optional_hashes):
            raise ValueError("schema-v1 engineering fixture must not carry checkpoint authority")
        if self.adapter_config_sha256 != ADAPTER_CONFIG_SHA256:
            raise ValueError("fixture sampler identity changed the adapter contract")
        if self.native_pilot_contract_sha256 != PILOT_CONTRACT_SHA256:
            raise ValueError("fixture sampler identity changed the pilot contract")
        if self.native_parent_contract_sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("fixture sampler identity changed the parent contract")
        if self.independently_audited is not False or self.production_input_eligible is not False:
            raise ValueError("schema-v1 engineering fixture cannot claim production authority")
        if (
            type(self.residual_lambda_by_bin) is not tuple
            or len(self.residual_lambda_by_bin) != 4
            or any(
                type(value) is not float or value not in ALLOWED_RESIDUAL_LAMBDAS
                for value in self.residual_lambda_by_bin
            )
        ):
            raise ValueError("residual calibration must contain four allowed exact floats")
        if (
            type(self.temperature_by_bin) is not tuple
            or len(self.temperature_by_bin) != 4
            or any(
                type(value) is not float or value not in ALLOWED_TEMPERATURES
                for value in self.temperature_by_bin
            )
        ):
            raise ValueError("temperature calibration must contain four allowed exact floats")
        if (
            type(self.sampling_backoff_epsilon) is not float
            or self.sampling_backoff_epsilon not in ALLOWED_BACKOFF_EPSILONS
        ):
            raise ValueError("sampling backoff epsilon is not in the frozen grid")
        if type(self.engineering_provenance) is not NativeV1EngineeringRuntimeProvenance:
            raise TypeError("sampler identity requires exact engineering runtime provenance")
        self.engineering_provenance.revalidate()

    def as_document(self) -> Mapping[str, object]:
        self.revalidate()
        return {
            "adapter_config_sha256": self.adapter_config_sha256,
            "artifact": "native_categorical_diffusion_v1_engineering_fixture_sampler_identity",
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "checkpoint_set_id": self.checkpoint_set_id,
            "count_prior_sha256": self.count_prior_sha256,
            "evidence_class": self.evidence_class,
            "engineering_provenance": dict(self.engineering_provenance.as_document()),
            "engineering_provenance_sha256": self.engineering_provenance.sha256,
            "independent_receipt_sha256": self.independent_receipt_sha256,
            "independently_audited": self.independently_audited,
            "model_state_sha256": self.model_state_sha256,
            "native_parent_contract_sha256": self.native_parent_contract_sha256,
            "native_pilot_contract_sha256": self.native_pilot_contract_sha256,
            "production_input_eligible": self.production_input_eligible,
            "residual_lambda_by_bin": list(self.residual_lambda_by_bin),
            "sampling_backoff_epsilon": self.sampling_backoff_epsilon,
            "runtime_scope": "engineering_fixture_only",
            "schema_version": 1,
            "temperature_by_bin": list(self.temperature_by_bin),
        }

    def revalidate(self) -> NativeV1SamplerIdentity:
        if type(self) is not NativeV1SamplerIdentity:
            raise TypeError("sampler identity must have its exact runtime type")
        self.__post_init__()
        return self

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_IDENTITY_DOMAIN)
        digest.update(canonical_json_bytes(dict(self.as_document())))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeV1SampledCandidate:
    """One raw unfiltered peptide plus immutable sampler provenance."""

    ordinal: int
    seed: int
    sequence_id: str
    sequence: str
    length: int
    sampler_identity_sha256: str
    engineering_provenance_sha256: str
    request_sha256: str
    evidence_class: str
    scientific_evidence_accepted: bool
    production_input_eligible: bool

    def __post_init__(self) -> None:
        _uint64(self.ordinal, label="ordinal")
        _uint64(self.seed, label="seed")
        _sha256(self.sampler_identity_sha256, label="sampler_identity_sha256")
        _sha256(
            self.engineering_provenance_sha256,
            label="engineering_provenance_sha256",
        )
        _sha256(self.request_sha256, label="request_sha256")
        if type(self.length) is not int or not MIN_LENGTH <= self.length <= MAX_LENGTH:
            raise ValueError("candidate length must lie in 8..50")
        if type(self.sequence) is not str or len(self.sequence) != self.length:
            raise ValueError("candidate sequence and length disagree")
        if set(self.sequence) - set(ALPHABET):
            raise ValueError("candidate contains a noncanonical residue")
        if canonical_sequence_id(self.sequence) != self.sequence_id:
            raise ValueError("candidate sequence identity is invalid")
        if self.evidence_class != ENGINEERING_FIXTURE_EVIDENCE:
            raise ValueError("schema-v1 candidate must remain an engineering fixture")
        if self.scientific_evidence_accepted is not False:
            raise ValueError("raw sampler candidates are never scientific evidence")
        if self.production_input_eligible is not False:
            raise ValueError("schema-v1 candidate cannot claim production eligibility")

    def as_document(self) -> Mapping[str, object]:
        self.revalidate()
        return {
            "artifact": "native_categorical_diffusion_v1_engineering_fixture_candidate",
            "evidence_class": self.evidence_class,
            "engineering_provenance_sha256": self.engineering_provenance_sha256,
            "length": self.length,
            "ordinal": self.ordinal,
            "production_input_eligible": self.production_input_eligible,
            "request_sha256": self.request_sha256,
            "runtime_scope": "engineering_fixture_only",
            "sampler_identity_sha256": self.sampler_identity_sha256,
            "schema_version": 1,
            "scientific_evidence_accepted": self.scientific_evidence_accepted,
            "seed": self.seed,
            "sequence": self.sequence,
            "sequence_id": self.sequence_id,
        }

    def revalidate(self) -> NativeV1SampledCandidate:
        if type(self) is not NativeV1SampledCandidate:
            raise TypeError("candidate must have its exact runtime type")
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class NativeV1SamplingResult:
    """Deterministic candidate ledger with no scoring or filtering semantics."""

    identity: NativeV1SamplerIdentity
    candidates: tuple[NativeV1SampledCandidate, ...]
    length_plan_sha256: str
    request_sha256: str
    batch_size: int
    oracle_calls: int
    scientific_evidence_accepted: bool
    automatic_generator_mixture_eligible: bool

    def __post_init__(self) -> None:
        if type(self.identity) is not NativeV1SamplerIdentity:
            raise TypeError("sampling result identity must be exact NativeV1SamplerIdentity")
        self.identity.revalidate()
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or any(type(item) is not NativeV1SampledCandidate for item in self.candidates)
        ):
            raise TypeError("sampling result must contain an exact tuple of candidate records")
        if len(self.candidates) > PROPOSAL_COUNT_CAP:
            raise ValueError("sampling result exceeds the frozen proposal cap")
        ordinals = tuple(item.ordinal for item in self.candidates)
        if ordinals != tuple(sorted(ordinals)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("candidate ordinals must be unique and ascending")
        _sha256(self.length_plan_sha256, label="length_plan_sha256")
        _sha256(self.request_sha256, label="request_sha256")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= BATCH_SEQUENCE_CAP:
            raise ValueError("result batch size exceeds the frozen ceiling")
        if type(self.oracle_calls) is not int or self.oracle_calls != 0:
            raise ValueError("native-v1 sampler cannot record an oracle call")
        if self.scientific_evidence_accepted is not False:
            raise ValueError("raw sampling output cannot be scientific evidence")
        if self.automatic_generator_mixture_eligible is not False:
            raise ValueError("sampler output cannot alter the generator mixture automatically")
        for item in self.candidates:
            item.revalidate()
            if (
                item.seed != self.candidates[0].seed
                or item.sampler_identity_sha256 != self.identity.sha256
                or item.engineering_provenance_sha256 != self.identity.engineering_provenance.sha256
                or item.request_sha256 != self.request_sha256
                or item.production_input_eligible is not False
            ):
                raise ValueError("candidate provenance differs within one sampling result")
        expected_plan, observed_plan_sha256 = native_sampling.canonical_length_plan(
            tuple(item.length for item in self.candidates),
            ordinals=ordinals,
            require_locked_count=False,
        )
        if len(expected_plan) != len(self.candidates) or (
            observed_plan_sha256 != self.length_plan_sha256
        ):
            raise ValueError("sampling result length-plan identity is invalid")
        request_document = {
            "batch_size": self.batch_size,
            "engineering_provenance_sha256": self.identity.engineering_provenance.sha256,
            "length_plan_sha256": self.length_plan_sha256,
            "proposal_count": len(self.candidates),
            "runtime_scope": "engineering_fixture_only",
            "sampler_identity_sha256": self.identity.sha256,
            "schema_version": 1,
            "seed": self.candidates[0].seed,
        }
        request_digest = hashlib.sha256()
        request_digest.update(_REQUEST_DOMAIN)
        request_digest.update(canonical_json_bytes(request_document))
        if request_digest.hexdigest() != self.request_sha256:
            raise ValueError("sampling result request identity is invalid")

    def canonical_jsonl_bytes(self) -> bytes:
        self.revalidate()
        return b"".join(canonical_json_bytes(dict(item.as_document())) for item in self.candidates)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_jsonl_bytes()).hexdigest()

    def manifest_document(self) -> Mapping[str, object]:
        self.revalidate()
        return {
            "artifact": "native_categorical_diffusion_v1_engineering_fixture_raw_proposal_ledger",
            "automatic_generator_mixture_eligible": self.automatic_generator_mixture_eligible,
            "batch_size": self.batch_size,
            "candidate_count": len(self.candidates),
            "candidate_ledger_sha256": self.sha256,
            "engineering_runtime_provenance": dict(
                self.identity.engineering_provenance.as_document()
            ),
            "engineering_runtime_provenance_sha256": (self.identity.engineering_provenance.sha256),
            "evidence_class": ENGINEERING_FIXTURE_EVIDENCE,
            "length_plan_sha256": self.length_plan_sha256,
            "oracle_calls": self.oracle_calls,
            "production_input_eligible": False,
            "request_sha256": self.request_sha256,
            "runtime_scope": "engineering_fixture_only",
            "sampler_identity": dict(self.identity.as_document()),
            "sampler_identity_sha256": self.identity.sha256,
            "schema_version": 1,
            "scientific_evidence_accepted": self.scientific_evidence_accepted,
        }

    def canonical_manifest_bytes(self) -> bytes:
        return canonical_json_bytes(dict(self.manifest_document()))

    def revalidate(self) -> NativeV1SamplingResult:
        if type(self) is not NativeV1SamplingResult:
            raise TypeError("sampling result must have its exact runtime type")
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class _ProviderState:
    model: R128Denoiser
    contract: NativeDiffusionV1PilotContract
    count_prior: AuthenticatedCountPrior
    count_tensor: Tensor
    identity: NativeV1SamplerIdentity


def _calibrated_log_probability(
    residual: Tensor,
    count: Tensor,
    levels: Tensor,
    *,
    residual_lambda_by_bin: tuple[float, float, float, float],
    temperature_by_bin: tuple[float, float, float, float],
    sampling_backoff_epsilon: float,
) -> Tensor:
    """Return stable float32 log probabilities for the frozen calibration rule."""

    if type(residual) is not Tensor or type(count) is not Tensor or type(levels) is not Tensor:
        raise TypeError("calibrated probability inputs must be exact tensors")
    if (
        residual.dtype != torch.float32
        or count.dtype != torch.float32
        or levels.dtype != torch.int64
        or residual.ndim != 3
        or count.shape != residual.shape
        or residual.shape[-1] != 20
        or levels.shape != (residual.shape[0],)
        or count.device != residual.device
        or levels.device != residual.device
    ):
        raise ValueError("calibrated probability tensor schema is invalid")
    if not bool(torch.isfinite(residual).all().item()) or not bool(
        torch.isfinite(count).all().item()
    ):
        raise FloatingPointError("calibrated probability inputs must be finite")
    if (
        type(residual_lambda_by_bin) is not tuple
        or len(residual_lambda_by_bin) != 4
        or any(
            type(value) is not float or value not in ALLOWED_RESIDUAL_LAMBDAS
            for value in residual_lambda_by_bin
        )
    ):
        raise ValueError("residual calibration is outside the frozen grid")
    if (
        type(temperature_by_bin) is not tuple
        or len(temperature_by_bin) != 4
        or any(
            type(value) is not float or value not in ALLOWED_TEMPERATURES
            for value in temperature_by_bin
        )
    ):
        raise ValueError("temperature calibration is outside the frozen grid")
    if (
        type(sampling_backoff_epsilon) is not float
        or sampling_backoff_epsilon not in ALLOWED_BACKOFF_EPSILONS
    ):
        raise ValueError("sampling backoff is outside the frozen grid")
    bins = _calibration_bin_tensor(levels)
    lambdas = torch.tensor(
        residual_lambda_by_bin,
        dtype=torch.float32,
        device=residual.device,
    )[bins]
    temperatures = torch.tensor(
        temperature_by_bin,
        dtype=torch.float32,
        device=residual.device,
    )[bins]
    combined = (count + lambdas[:, None, None] * residual) / temperatures[:, None, None]
    if not bool(torch.isfinite(combined).all().item()):
        raise FloatingPointError("calibrated logits overflowed float32")
    result = torch.log_softmax(combined, dim=-1)
    if sampling_backoff_epsilon:
        epsilon = torch.tensor(
            sampling_backoff_epsilon,
            dtype=torch.float32,
            device=residual.device,
        )
        result = torch.logaddexp(
            torch.log1p(-epsilon) + result,
            torch.log(epsilon) + torch.log_softmax(count, dim=-1),
        )
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("calibrated log probabilities are invalid")
    return result


class _NativeV1R128LogitProvider:
    """Exact R128 plus C0/calibration probability boundary used by the sampler."""

    __slots__ = (
        "__contract",
        "__count_prior",
        "__count_tensor",
        "__count_tensor_guard",
        "__count_tensor_sha256",
        "__device",
        "__identity",
        "__model",
        "__parameter_guard",
    )

    def __init__(self) -> None:
        raise TypeError("native-v1 R128 provider requires an authenticated internal factory")

    def __initialize(self, token: object, state: _ProviderState) -> None:
        if token is not _FIXTURE_CONSTRUCTION_TOKEN or type(state) is not _ProviderState:
            raise TypeError("native-v1 R128 provider factory token is invalid")
        if type(state.model) is not R128Denoiser or R128Denoiser.forward is not _R128_FORWARD:
            raise TypeError("native-v1 provider requires the exact R128 implementation")
        if type(state.contract) is not NativeDiffusionV1PilotContract:
            raise TypeError("native-v1 provider requires the exact pilot contract")
        state.contract.revalidate()
        if type(state.count_prior) is not AuthenticatedCountPrior:
            raise TypeError("native-v1 provider requires an authenticated count prior")
        if type(state.identity) is not NativeV1SamplerIdentity:
            raise TypeError("native-v1 provider requires an exact sampler identity")
        state.identity.revalidate()
        model = state.model
        model.eval()
        model.requires_grad_(False)
        assert_r128_model_execution_surface(model)
        parameters = tuple(model.parameters())
        if not parameters:
            raise RuntimeError("R128 provider has no parameters")
        device = parameters[0].device
        if any(
            parameter.device != device
            or parameter.dtype != torch.float32
            or parameter.requires_grad
            for parameter in parameters
        ):
            raise RuntimeError("R128 provider parameters are not frozen float32 on one device")
        if device != torch.device("cpu"):
            raise RuntimeError("engineering fixture provider requires exact CPU")
        count_tensor = state.count_tensor
        if (
            type(count_tensor) is not Tensor
            or count_tensor.dtype != torch.float32
            or count_tensor.device != device
            or count_tensor.shape != (5, 10, 20)
            or not count_tensor.is_contiguous()
            or count_tensor.requires_grad
            or not bool(torch.isfinite(count_tensor).all().item())
        ):
            raise ValueError("native-v1 provider count tensor is invalid")
        if canonical_r128_model_sha256(model.state_dict()) != state.identity.model_state_sha256:
            raise ValueError("live R128 state differs from sampler identity")
        if state.count_prior.sha256 != state.identity.count_prior_sha256:
            raise ValueError("authenticated count prior differs from sampler identity")
        expected_count_tensor = count_prior_training_bridge(state.count_prior, device="cpu")
        if not torch.equal(count_tensor, expected_count_tensor):
            raise ValueError("live count tensor differs from the authenticated count prior")
        if _engineering_runtime_provenance() != state.identity.engineering_provenance:
            raise ValueError("live engineering runtime differs from sampler identity")
        self.__model = model
        self.__contract = state.contract
        self.__count_prior = state.count_prior
        self.__count_tensor = count_tensor
        self.__count_tensor_sha256 = _count_tensor_sha256(count_tensor)
        self.__count_tensor_guard = (
            id(count_tensor),
            count_tensor.data_ptr(),
            count_tensor._version,
        )
        self.__identity = state.identity
        self.__device = device
        self.__parameter_guard = _parameter_guard(model)
        self._revalidate()

    @classmethod
    def _from_fixture_state(
        cls,
        token: object,
        state: _ProviderState,
    ) -> _NativeV1R128LogitProvider:
        value = object.__new__(cls)
        value.__initialize(token, state)
        return value

    @property
    def identity(self) -> NativeV1SamplerIdentity:
        return self.__identity

    def _revalidate(self) -> None:
        assert_r128_model_execution_surface(self.__model)
        assert_r128_deterministic_runtime()
        if self.__model.training or any(module.training for module in self.__model.modules()):
            raise RuntimeError("native-v1 provider model left evaluation mode")
        if any(parameter.requires_grad for parameter in self.__model.parameters()):
            raise RuntimeError("native-v1 provider model became trainable")
        if _parameter_guard(self.__model) != self.__parameter_guard:
            raise RuntimeError("native-v1 provider model storage or version changed")
        if (
            id(self.__count_tensor),
            self.__count_tensor.data_ptr(),
            self.__count_tensor._version,
        ) != self.__count_tensor_guard:
            raise RuntimeError("native-v1 provider count tensor changed")

    def _revalidate_content(self) -> None:
        self._revalidate()
        self.__contract.revalidate()
        self.__count_prior.revalidate()
        self.__identity.revalidate()
        if canonical_r128_model_sha256(self.__model.state_dict()) != (
            self.__identity.model_state_sha256
        ):
            raise RuntimeError("native-v1 provider model content changed")
        if _count_tensor_sha256(self.__count_tensor) != self.__count_tensor_sha256:
            raise RuntimeError("native-v1 provider count tensor content changed")
        expected_count_tensor = count_prior_training_bridge(self.__count_prior, device="cpu")
        if not torch.equal(self.__count_tensor, expected_count_tensor):
            raise RuntimeError("native-v1 provider count tensor no longer matches its prior")
        if _engineering_runtime_provenance() != self.__identity.engineering_provenance:
            raise RuntimeError("native-v1 engineering implementation or runtime changed")

    def __call__(
        self,
        tokens: NDArray[np.int64],
        attention_mask: NDArray[np.bool_],
        levels: NDArray[np.int64],
        lengths: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        self._revalidate()
        token_values, mask_values, level_values, length_values = _validate_provider_inputs(
            tokens,
            attention_mask,
            levels,
            lengths,
        )
        token_tensor = torch.from_numpy(token_values).to(self.__device)
        mask_tensor = torch.from_numpy(mask_values).to(self.__device)
        level_tensor = torch.from_numpy(level_values).to(self.__device)
        length_tensor = torch.from_numpy(length_values).to(self.__device)
        with torch.inference_mode(), torch.autocast(device_type=self.__device.type, enabled=False):
            residual = self.__model(
                token_tensor,
                mask_tensor,
                level_tensor,
                length_tensor,
            )
            count = count_prior_logits(self.__count_tensor, mask_tensor, length_tensor)
            logits = _calibrated_log_probability(
                residual,
                count,
                level_tensor,
                residual_lambda_by_bin=self.__identity.residual_lambda_by_bin,
                temperature_by_bin=self.__identity.temperature_by_bin,
                sampling_backoff_epsilon=self.__identity.sampling_backoff_epsilon,
            )
            logits = logits.masked_fill(~mask_tensor.unsqueeze(-1), 0.0)
        if (
            logits.dtype != torch.float32
            or logits.device != self.__device
            or logits.shape != (*token_values.shape, 20)
            or logits.requires_grad
            or not bool(torch.isfinite(logits).all().item())
        ):
            raise FloatingPointError("native-v1 provider produced invalid calibrated logits")
        output = logits.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
        output.flags.writeable = False
        self._revalidate()
        return cast(NDArray[np.float32], output)


class NativeV1RealSampler:
    """Unconstructable placeholder for a separately reviewed successor loader."""

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError("schema-v1 has no production sampler construction authority")


def load_native_v1_real_sampler(
    *,
    adapter_contract_path: str | os.PathLike[str],
    checkpoint_set_id: str,
    checkpoint_bundle_root: str | os.PathLike[str],
    pilot_contract_path: str | os.PathLike[str],
    parent_contract_path: str | os.PathLike[str] | None = None,
) -> NativeV1RealSampler:
    """Load a repository-pinned accepted checkpoint or fail before bundle access.

    The current adapter contract has no accepted pin.  Consequently the call to
    :meth:`checkpoint_pin` below is the terminal operation today: bundle and
    pilot-contract paths are not resolved, stated, or opened.
    """

    contract = load_native_v1_real_sampler_contract(adapter_contract_path)
    contract.revalidate()
    contract.checkpoint_pin(checkpoint_set_id)
    # The current content-pinned contract makes this unreachable.  A successor
    # must reuse the native checkpoint loader once an independent receipt exists;
    # v1 deliberately does not speculate about or accept such a receipt schema.
    raise RuntimeError("frozen native-v1 sampler contract unexpectedly supplied a checkpoint pin")


def _build_untrained_r128_fixture_provider(
    *,
    contract: NativeDiffusionV1PilotContract,
    count_prior: AuthenticatedCountPrior,
    outer_fold: int = 0,
    residual_lambda_by_bin: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    temperature_by_bin: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    sampling_backoff_epsilon: float = 0.0,
) -> _NativeV1R128LogitProvider:
    """Build an untrained CPU fixture that cannot cross the production API."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("fixture contract must be exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    if type(count_prior) is not AuthenticatedCountPrior:
        raise TypeError("fixture count_prior must be exact AuthenticatedCountPrior")
    count_prior.revalidate()
    model, _ = build_r128_model_from_contract(contract, outer_fold, device="cpu")
    model.eval()
    model.requires_grad_(False)
    count_tensor = count_prior_training_bridge(count_prior, device="cpu")
    count_tensor.requires_grad_(False)
    identity = NativeV1SamplerIdentity(
        evidence_class=ENGINEERING_FIXTURE_EVIDENCE,
        checkpoint_set_id="engineering_fixture_no_checkpoint",
        adapter_config_sha256=ADAPTER_CONFIG_SHA256,
        native_pilot_contract_sha256=contract.config_sha256,
        native_parent_contract_sha256=contract.parent_config_sha256,
        model_state_sha256=canonical_r128_model_sha256(model.state_dict()),
        checkpoint_file_sha256=None,
        checkpoint_metadata_sha256=None,
        count_prior_sha256=count_prior.sha256,
        bundle_manifest_sha256=None,
        independent_receipt_sha256=None,
        residual_lambda_by_bin=residual_lambda_by_bin,
        temperature_by_bin=temperature_by_bin,
        sampling_backoff_epsilon=sampling_backoff_epsilon,
        engineering_provenance=_engineering_runtime_provenance(),
        independently_audited=False,
        production_input_eligible=False,
    )
    return _NativeV1R128LogitProvider._from_fixture_state(
        _FIXTURE_CONSTRUCTION_TOKEN,
        _ProviderState(
            model=model,
            contract=contract,
            count_prior=count_prior,
            count_tensor=count_tensor,
            identity=identity,
        ),
    )


def _sample_native_v1_engineering_fixture(
    provider: _NativeV1R128LogitProvider,
    lengths: Iterable[int],
    *,
    seed: int,
    ordinals: Iterable[int] | None = None,
    batch_size: int = BATCH_SEQUENCE_CAP,
) -> NativeV1SamplingResult:
    """Exercise the real tensor/sampler path while remaining explicitly non-authoritative."""

    return _sample_with_provider(
        provider,
        lengths,
        seed=seed,
        ordinals=ordinals,
        batch_size=batch_size,
    )


def _sample_with_provider(
    provider: _NativeV1R128LogitProvider,
    lengths: Iterable[int],
    *,
    seed: int,
    ordinals: Iterable[int] | None,
    batch_size: int,
) -> NativeV1SamplingResult:
    if type(provider) is not _NativeV1R128LogitProvider:
        raise TypeError("sampling requires an exact native-v1 R128 provider")
    if (
        type(batch_size) is not int
        or isinstance(batch_size, bool)
        or not (1 <= batch_size <= BATCH_SEQUENCE_CAP)
    ):
        raise ValueError(f"batch_size must be an exact integer in 1..{BATCH_SEQUENCE_CAP}")
    start_time = time.monotonic()
    root_seed = _uint64(seed, label="seed")
    bounded_lengths = _bounded_materialize(
        lengths,
        maximum_items=PROPOSAL_COUNT_CAP,
        label="lengths",
    )
    bounded_ordinals = (
        None
        if ordinals is None
        else _bounded_materialize(
            ordinals,
            maximum_items=PROPOSAL_COUNT_CAP,
            label="ordinals",
        )
    )
    plan, length_plan_sha256 = native_sampling.canonical_length_plan(
        bounded_lengths,
        ordinals=bounded_ordinals,
        require_locked_count=False,
    )
    if len(plan) > PROPOSAL_COUNT_CAP:
        raise ValueError(f"proposal count exceeds the frozen cap of {PROPOSAL_COUNT_CAP}")
    identity = provider.identity
    identity.revalidate()
    identity_sha256 = identity.sha256
    if identity.evidence_class != ENGINEERING_FIXTURE_EVIDENCE:
        raise RuntimeError("schema-v1 sampling requires an engineering fixture identity")
    request_document = {
        "batch_size": batch_size,
        "engineering_provenance_sha256": identity.engineering_provenance.sha256,
        "length_plan_sha256": length_plan_sha256,
        "proposal_count": len(plan),
        "runtime_scope": "engineering_fixture_only",
        "sampler_identity_sha256": identity_sha256,
        "schema_version": 1,
        "seed": root_seed,
    }
    request_digest = hashlib.sha256()
    request_digest.update(_REQUEST_DOMAIN)
    request_digest.update(canonical_json_bytes(request_document))
    request_sha256 = request_digest.hexdigest()
    vocabulary = PeptideVocabulary(ALPHABET)
    schedule = CosineMaskSchedule(offset=0.008)
    candidates: list[NativeV1SampledCandidate] = []
    _check_wall_clock(start_time)
    provider._revalidate_content()

    for start in range(0, len(plan), batch_size):
        _check_wall_clock(start_time)
        batch_plan = plan[start : start + batch_size]
        batch_ordinals = np.asarray([item[0] for item in batch_plan], dtype=np.uint64)
        batch_lengths = np.asarray([item[1] for item in batch_plan], dtype=np.int64)
        attention_mask = np.arange(MAX_LENGTH)[None, :] < batch_lengths[:, None]
        tokens = np.full((len(batch_plan), MAX_LENGTH), vocabulary.pad_index, dtype=np.int64)
        tokens[attention_mask] = vocabulary.mask_index

        for level in range(DIFFUSION_LEVELS, 0, -1):
            current_counts = schedule.mask_counts(
                batch_lengths,
                level,
                total_levels=DIFFUSION_LEVELS,
            )
            target_counts = schedule.mask_counts(
                batch_lengths,
                level - 1,
                total_levels=DIFFUSION_LEVELS,
            )
            observed_counts = np.sum(tokens == vocabulary.mask_index, axis=1, dtype=np.int64)
            if not np.array_equal(observed_counts, current_counts):
                raise RuntimeError("native-v1 reverse trajectory mask count drifted")
            commit_counts = current_counts - target_counts
            if bool(np.any(commit_counts < 0)):
                raise RuntimeError("native-v1 reverse trajectory increased the mask count")
            if not bool(np.any(commit_counts)):
                continue
            _check_wall_clock(start_time)
            provider_inputs = (
                np.ascontiguousarray(tokens.copy(), dtype="<i8"),
                np.ascontiguousarray(attention_mask.copy(), dtype="|b1"),
                np.full(len(batch_plan), level, dtype="<i8"),
                np.ascontiguousarray(batch_lengths.copy(), dtype="<i8"),
            )
            for value in provider_inputs:
                value.flags.writeable = False
            logits = provider(*provider_inputs)
            probabilities = native_sampling._probabilities(
                logits,
                expected_shape=(len(batch_plan), MAX_LENGTH, len(ALPHABET)),
            )
            for row_index, raw_commit_count in enumerate(commit_counts):
                commit_count = int(raw_commit_count)
                if not commit_count:
                    continue
                masked_positions = np.flatnonzero(
                    attention_mask[row_index] & (tokens[row_index] == vocabulary.mask_index)
                )
                confidence = np.max(probabilities[row_index, masked_positions], axis=1)
                ranked = np.lexsort((masked_positions, -confidence))
                selected = masked_positions[ranked[:commit_count]]
                ordinal = int(batch_ordinals[row_index])
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        _sampler_draw_seed(identity_sha256, root_seed, ordinal, level)
                    )
                )
                for position, uniform in zip(selected, rng.random(commit_count), strict=True):
                    tokens[row_index, position] = native_sampling._categorical_index(
                        probabilities[row_index, position],
                        float(uniform),
                    )
            remaining = np.sum(tokens == vocabulary.mask_index, axis=1, dtype=np.int64)
            if not np.array_equal(remaining, target_counts):
                raise RuntimeError("native-v1 sampler did not commit the scheduled residue count")

        if bool(np.any(tokens[attention_mask] >= len(ALPHABET))):
            raise RuntimeError("native-v1 sampler left or emitted a special token")
        if bool(np.any(tokens[~attention_mask] != vocabulary.pad_index)):
            raise RuntimeError("native-v1 sampler modified padded positions")
        for (ordinal, expected_length), sequence in zip(
            batch_plan,
            vocabulary.decode(tokens),
            strict=True,
        ):
            candidates.append(
                NativeV1SampledCandidate(
                    ordinal=ordinal,
                    seed=root_seed,
                    sequence_id=canonical_sequence_id(sequence),
                    sequence=sequence,
                    length=expected_length,
                    sampler_identity_sha256=identity_sha256,
                    engineering_provenance_sha256=identity.engineering_provenance.sha256,
                    request_sha256=request_sha256,
                    evidence_class=identity.evidence_class,
                    scientific_evidence_accepted=False,
                    production_input_eligible=False,
                )
            )

    _check_wall_clock(start_time)
    provider._revalidate_content()
    ordered = tuple(sorted(candidates, key=lambda item: item.ordinal))
    if len(ordered) != len(plan):
        raise RuntimeError("native-v1 sampler did not preserve the proposal census")
    return NativeV1SamplingResult(
        identity=identity,
        candidates=ordered,
        length_plan_sha256=length_plan_sha256,
        request_sha256=request_sha256,
        batch_size=batch_size,
        oracle_calls=0,
        scientific_evidence_accepted=False,
        automatic_generator_mixture_eligible=False,
    )


def _bounded_materialize(
    values: Iterable[int],
    *,
    maximum_items: int,
    label: str,
) -> tuple[int, ...]:
    """Materialize at most one item beyond a hard census ceiling."""

    if type(maximum_items) is not int or maximum_items <= 0:
        raise ValueError("maximum_items must be a positive exact integer")
    if isinstance(values, str | bytes | bytearray):
        raise TypeError(f"{label} must be a bounded iterable of scalar values")
    if isinstance(values, Sized) and len(values) > maximum_items:
        raise ValueError(f"{label} exceeds the frozen cap of {maximum_items}")
    try:
        iterator = iter(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be iterable") from error
    result = tuple(itertools.islice(iterator, maximum_items + 1))
    if len(result) > maximum_items:
        raise ValueError(f"{label} exceeds the frozen cap of {maximum_items}")
    return result


def _count_tensor_sha256(value: Tensor) -> str:
    if (
        type(value) is not Tensor
        or value.dtype != torch.float32
        or value.shape != (5, 10, 20)
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError("count tensor cannot be content-addressed")
    payload = (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    digest = hashlib.sha256()
    digest.update(_COUNT_TENSOR_DOMAIN)
    digest.update(payload)
    return digest.hexdigest()


def _engineering_runtime_provenance() -> NativeV1EngineeringRuntimeProvenance:
    root = Path(__file__).resolve().parents[5]
    records = tuple(
        (relative_path, hashlib.sha256(_read_engineering_file(root / relative_path)).hexdigest())
        for relative_path in _ENGINEERING_IMPLEMENTATION_PATHS
    )
    implementation_document = {
        "artifact": "native_categorical_diffusion_v1_engineering_implementation",
        "files": [{"path": path, "sha256": digest} for path, digest in records],
        "schema_version": 1,
    }
    implementation_digest = hashlib.sha256()
    implementation_digest.update(_PROVENANCE_DOMAIN)
    implementation_digest.update(canonical_json_bytes(implementation_document))
    deterministic_runtime = assert_r128_deterministic_runtime()
    return NativeV1EngineeringRuntimeProvenance(
        implementation_sha256=implementation_digest.hexdigest(),
        implementation_files=records,
        dependency_lock_sha256=dict(records)["uv.lock"],
        deterministic_runtime_sha256=hashlib.sha256(
            canonical_json_bytes(deterministic_runtime)
        ).hexdigest(),
        python_version=str(platform.python_version()),
        python_implementation=str(platform.python_implementation()),
        numpy_version=str(np.__version__),
        torch_version=str(torch.__version__),
        torch_cuda_version=(None if torch.version.cuda is None else str(torch.version.cuda)),
        safetensors_version=str(importlib.metadata.version("safetensors")),
        device="cpu",
        observation_scope="self_observed_engineering_only",
        independent_authority=False,
        scientific_evidence_accepted=False,
    )


def _read_engineering_file(path: Path) -> bytes:
    """Read a small regular file for non-authoritative engineering provenance."""

    try:
        named_before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"engineering provenance file is unavailable: {path}") from error
    if (
        not stat.S_ISREG(named_before.st_mode)
        or named_before.st_nlink != 1
        or not 0 < named_before.st_size <= _MAX_ENGINEERING_PROVENANCE_FILE_BYTES
    ):
        raise ValueError(f"engineering provenance file is not a bounded regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"engineering provenance file cannot be opened: {path}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(
            descriptor,
            min(65536, _MAX_ENGINEERING_PROVENANCE_FILE_BYTES + 1 - total),
        ):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ENGINEERING_PROVENANCE_FILE_BYTES:
                raise ValueError(f"engineering provenance file exceeds its bound: {path}")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(path)
    except OSError as error:
        raise ValueError(f"engineering provenance file changed while read: {path}") from error
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    fingerprints = tuple(
        tuple(getattr(snapshot, field) for field in fields)
        for snapshot in (named_before, opened_before, opened_after, named_after)
    )
    payload = b"".join(chunks)
    if len(set(fingerprints)) != 1 or len(payload) != opened_before.st_size:
        raise ValueError(f"engineering provenance file changed while read: {path}")
    return payload


def _validate_provider_inputs(
    tokens: object,
    attention_mask: object,
    levels: object,
    lengths: object,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    token_values = _readonly_array(tokens, dtype=np.dtype("<i8"), rank=2, label="tokens")
    mask_values = _readonly_array(
        attention_mask,
        dtype=np.dtype("|b1"),
        rank=2,
        label="attention_mask",
    )
    level_values = _readonly_array(levels, dtype=np.dtype("<i8"), rank=1, label="levels")
    length_values = _readonly_array(lengths, dtype=np.dtype("<i8"), rank=1, label="lengths")
    if token_values.shape != mask_values.shape:
        raise ValueError("provider token and attention arrays must align")
    batch, width = token_values.shape
    if not 1 <= batch <= BATCH_SEQUENCE_CAP or width != MAX_LENGTH:
        raise ValueError("provider inputs exceed the exact batch/width boundary")
    if level_values.shape != (batch,) or length_values.shape != (batch,):
        raise ValueError("provider levels and lengths must contain one value per row")
    if bool(np.any((level_values < 1) | (level_values > DIFFUSION_LEVELS))):
        raise ValueError("provider levels must lie in 1..64")
    if bool(np.any((length_values < MIN_LENGTH) | (length_values > MAX_LENGTH))):
        raise ValueError("provider lengths must lie in 8..50")
    expected_mask = np.arange(MAX_LENGTH)[None, :] < length_values[:, None]
    if not np.array_equal(mask_values, expected_mask):
        raise ValueError("provider attention mask must be a length-matched prefix")
    if bool(np.any((token_values < 0) | (token_values > 21))):
        raise ValueError("provider tokens fall outside the 22-token vocabulary")
    if bool(np.any(mask_values & (token_values == 20))):
        raise ValueError("valid provider positions cannot contain PAD")
    if bool(np.any(~mask_values & (token_values != 20))):
        raise ValueError("padded provider positions must contain PAD")
    return (
        cast(NDArray[np.int64], token_values.copy(order="C")),
        cast(NDArray[np.bool_], mask_values.copy(order="C")),
        cast(NDArray[np.int64], level_values.copy(order="C")),
        cast(NDArray[np.int64], length_values.copy(order="C")),
    )


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    rank: int,
    label: str,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact NumPy ndarray")
    array = cast(np.ndarray, value)
    if array.dtype != dtype or array.ndim != rank:
        raise TypeError(f"{label} must be rank-{rank} exact {dtype.str}")
    if not array.flags.c_contiguous or array.flags.writeable:
        raise ValueError(f"{label} must be C-contiguous and read-only")
    return array


def _calibration_bin_tensor(levels: Tensor) -> Tensor:
    if type(levels) is not Tensor or levels.dtype != torch.int64 or levels.ndim != 1:
        raise TypeError("calibration levels must be an exact int64 vector tensor")
    result = torch.div(levels - 1, 16, rounding_mode="floor")
    if bool(torch.any((result < 0) | (result > 3)).item()):
        raise ValueError("calibration levels have no frozen timestep bin")
    return result


def _parameter_guard(model: R128Denoiser) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            parameter.data_ptr(),
            parameter._version,
            tuple(parameter.shape),
            parameter.requires_grad,
        )
        for name, parameter in model.named_parameters()
    )


def _sampler_draw_seed(
    sampler_identity_sha256: str,
    seed: int,
    ordinal: int,
    level: int,
) -> int:
    identity = bytes.fromhex(_sha256(sampler_identity_sha256, label="sampler identity"))
    root_seed = _uint64(seed, label="seed")
    candidate_ordinal = _uint64(ordinal, label="ordinal")
    if type(level) is not int or not 1 <= level <= DIFFUSION_LEVELS:
        raise ValueError("level must be an exact integer in 1..64")
    digest = hashlib.sha256()
    digest.update(_DRAW_DOMAIN)
    for payload in (
        identity,
        root_seed.to_bytes(8, "big"),
        candidate_ordinal.to_bytes(8, "big"),
        level.to_bytes(2, "big"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big")


def _check_wall_clock(start_time: float) -> None:
    if type(start_time) is not float or not math.isfinite(start_time):
        raise TypeError("sampling start time must be a finite exact float")
    elapsed = time.monotonic() - start_time
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("sampling monotonic clock is invalid")
    if elapsed > REQUEST_WALL_SECONDS:
        raise TimeoutError("native-v1 sampler exceeded the frozen fixture-request wall time")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _uint64(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value

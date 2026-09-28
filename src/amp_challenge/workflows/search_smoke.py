"""Deterministic one-round smoke workflow for counterfactual soft-KG search.

The workflow is intentionally tiny: it exercises the mathematical and
provenance seams of the research proposal without fitting a neural network or
running a production search.  Its CSV/JSON outputs are stable audit fixtures,
not biological evidence or benchmark results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from amp_challenge.acquisition.counterfactual import (
    AdvantageGateResult,
    gate_paired_advantages,
    historical_posterior_influence_credit,
    sampled_paired_contrast,
)
from amp_challenge.acquisition.recommendation import (
    PosteriorMeanRecommendation,
    recommend_posterior_mean,
)
from amp_challenge.acquisition.soft_kg import (
    EvaluationBatch,
    GaussianSoftKG,
    JointSoftKGResult,
    PreferenceMeasure,
    SoftKGProblem,
    UpperChanceConstraint,
    maximum_exhaustive_pool_size,
)
from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.replay import (
    EndpointReplayBuilder,
    EndpointReplayPlan,
    effective_sample_size,
)
from amp_challenge.generators.diffusion.tabular import (
    DenoisingPreference,
    TabularConditionalDenoiser,
)
from amp_challenge.generators.search import (
    BatchExecutionPlan,
    BranchRecord,
    EdgeRecord,
    EvaluationRecord,
    FrozenRolloutBatch,
    ProbabilityFactor,
    ProbabilityTrace,
    ProposalRecord,
    RolloutRecord,
    SearchLedger,
    SelectionDecision,
    SubstitutionEmitter,
    TranspositionEvent,
    batch_execution_plan_from_mapping,
    branch_score,
    plan_count_batches,
    plan_length_bucketed_batches,
    progressive_widening_limit,
    sample_frozen_rollouts,
    thompson_substitution_weights_batch,
)
from amp_challenge.generators.search.emitters import (
    validate_seeded_emitter_edge_replay_contract,
)
from amp_challenge.models.posterior import JointGaussianPosterior
from amp_challenge.representations.laplacian import (
    LaplacianLogSpectralDensityConfig,
    laplacian_log_spectral_densities,
)

FloatArray = NDArray[np.float64]
SCHEMA_VERSION = 1
ARTIFACT_NAMES = (
    "proposals.csv",
    "edges.csv",
    "evaluations.csv",
    "rounds.csv",
    "rollouts.json",
)
SEED_SEQUENCES = (
    "AAAAAAAA",
    "DDDDDDDD",
    "KKKKKKKK",
    "RRRRRRRR",
    "ADKRADKR",
    "RADKRADK",
    "KADRKADR",
    "DRKADKRA",
)
SPECTRAL_CONFIG = LaplacianLogSpectralDensityConfig()
SPECTRAL_CONTACT_ADAPTER = "toy-ordinal-contact-v1-nonbiological"
SPECTRAL_PRIOR_VARIANCE_MULTIPLIER = 32.0


@dataclass(frozen=True, slots=True)
class SearchSmokeConfig:
    """Strict, versioned configuration for the bounded local workflow."""

    schema_version: int
    seed: int
    alphabet: str
    length: int
    rounds: int
    branches: int
    proposals: int
    evaluations: int
    context_id: str
    fidelity: str
    evaluator_version: str
    batching: BatchExecutionPlan
    kg_temperature: float
    kg_fantasies: int
    kg_standard_error_multiplier: float
    toxicity_upper: float
    max_violation_probability: float
    exploration_scale: float
    branch_yield_reference: float
    credit_mix: float
    credit_floor: float
    credit_ceiling: float
    credit_decay: float
    widening_coefficient: float
    widening_exponent: float
    thompson_scale: float
    risk_kappa: float
    random_evaluation_reserve: int
    replay_timestep: float
    replay_temperature: float
    replay_clip: float
    learning_rate: float
    preference_beta: float
    preference_clip: float
    local_kl_limit: float
    reference_path_kl_limit: float
    local_kl_p99_limit: float
    reference_transition_kl_p99_limit: float


_TOP_LEVEL_KEYS = {
    "schema_version",
    "seed",
    "alphabet",
    "length",
    "rounds",
    "branches",
    "proposals",
    "evaluations",
    "context_id",
    "fidelity",
    "evaluator_version",
    "batching",
    "kg",
    "search",
    "replay",
}
_KG_KEYS = {
    "temperature",
    "fantasies",
    "standard_error_multiplier",
    "toxicity_upper",
    "max_violation_probability",
}
_SEARCH_KEYS = {
    "exploration_scale",
    "branch_yield_reference",
    "credit_mix",
    "credit_floor",
    "credit_ceiling",
    "credit_decay",
    "widening_coefficient",
    "widening_exponent",
    "thompson_scale",
    "risk_kappa",
    "random_evaluation_reserve",
}
_REPLAY_KEYS = {
    "timestep",
    "temperature",
    "clip",
    "learning_rate",
    "preference_beta",
    "preference_clip",
    "local_kl_limit",
    "reference_path_kl_limit",
    "local_kl_p99_limit",
    "reference_transition_kl_p99_limit",
}


def _require_exact_keys(table: dict[str, Any], expected: set[str], *, name: str) -> None:
    unknown = sorted(set(table) - expected)
    missing = sorted(expected - set(table))
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {unknown}")
    if missing:
        raise ValueError(f"missing {name} configuration keys: {missing}")


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, not a boolean or float")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _table(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    value = raw[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    _require_exact_keys(value, expected, name=name)
    return value


def load_config(path: Path | str) -> SearchSmokeConfig:
    """Load a strict v1 TOML file and enforce the local smoke envelope."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, name="top-level")
    batching_raw = raw["batching"]
    if not isinstance(batching_raw, dict):
        raise ValueError("batching must be a TOML table")
    batching = batch_execution_plan_from_mapping(batching_raw)
    kg = _table(raw, "kg", _KG_KEYS)
    search = _table(raw, "search", _SEARCH_KEYS)
    replay = _table(raw, "replay", _REPLAY_KEYS)
    config = SearchSmokeConfig(
        schema_version=_integer(raw["schema_version"], name="schema_version"),
        seed=_integer(raw["seed"], name="seed"),
        alphabet=_string(raw["alphabet"], name="alphabet"),
        length=_integer(raw["length"], name="length"),
        rounds=_integer(raw["rounds"], name="rounds"),
        branches=_integer(raw["branches"], name="branches"),
        proposals=_integer(raw["proposals"], name="proposals"),
        evaluations=_integer(raw["evaluations"], name="evaluations"),
        context_id=_string(raw["context_id"], name="context_id"),
        fidelity=_string(raw["fidelity"], name="fidelity"),
        evaluator_version=_string(raw["evaluator_version"], name="evaluator_version"),
        batching=batching,
        kg_temperature=_number(kg["temperature"], name="kg.temperature"),
        kg_fantasies=_integer(kg["fantasies"], name="kg.fantasies"),
        kg_standard_error_multiplier=_number(
            kg["standard_error_multiplier"],
            name="kg.standard_error_multiplier",
        ),
        toxicity_upper=_number(kg["toxicity_upper"], name="kg.toxicity_upper"),
        max_violation_probability=_number(
            kg["max_violation_probability"], name="kg.max_violation_probability"
        ),
        exploration_scale=_number(search["exploration_scale"], name="search.exploration_scale"),
        branch_yield_reference=_number(
            search["branch_yield_reference"],
            name="search.branch_yield_reference",
        ),
        credit_mix=_number(search["credit_mix"], name="search.credit_mix"),
        credit_floor=_number(search["credit_floor"], name="search.credit_floor"),
        credit_ceiling=_number(search["credit_ceiling"], name="search.credit_ceiling"),
        credit_decay=_number(search["credit_decay"], name="search.credit_decay"),
        widening_coefficient=_number(
            search["widening_coefficient"], name="search.widening_coefficient"
        ),
        widening_exponent=_number(search["widening_exponent"], name="search.widening_exponent"),
        thompson_scale=_number(search["thompson_scale"], name="search.thompson_scale"),
        risk_kappa=_number(search["risk_kappa"], name="search.risk_kappa"),
        random_evaluation_reserve=_integer(
            search["random_evaluation_reserve"],
            name="search.random_evaluation_reserve",
        ),
        replay_timestep=_number(replay["timestep"], name="replay.timestep"),
        replay_temperature=_number(replay["temperature"], name="replay.temperature"),
        replay_clip=_number(replay["clip"], name="replay.clip"),
        learning_rate=_number(replay["learning_rate"], name="replay.learning_rate"),
        preference_beta=_number(replay["preference_beta"], name="replay.preference_beta"),
        preference_clip=_number(replay["preference_clip"], name="replay.preference_clip"),
        local_kl_limit=_number(replay["local_kl_limit"], name="replay.local_kl_limit"),
        reference_path_kl_limit=_number(
            replay["reference_path_kl_limit"], name="replay.reference_path_kl_limit"
        ),
        local_kl_p99_limit=_number(replay["local_kl_p99_limit"], name="replay.local_kl_p99_limit"),
        reference_transition_kl_p99_limit=_number(
            replay["reference_transition_kl_p99_limit"],
            name="replay.reference_transition_kl_p99_limit",
        ),
    )
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
    if config.alphabet != "ADKR" or config.length != 8:
        raise ValueError("the local smoke scope is exactly alphabet ADKR and length 8")
    if config.rounds != 1 or config.branches != 8:
        raise ValueError("the local smoke scope is exactly one round and eight branches")
    if config.batching.profile != "local_smoke_v1":
        raise ValueError("the local smoke workflow requires batching profile local_smoke_v1")
    if config.batching.rollout_batch_size != 1:
        raise ValueError("the local smoke workflow uses exactly one batched rollout")
    if not 16 <= config.proposals <= 64:
        raise ValueError("proposals must lie in [16, 64]")
    if not 1 <= config.evaluations <= min(8, config.proposals):
        raise ValueError("evaluations must lie in [1, 8] and not exceed proposals")
    if config.kg_fantasies < 2:
        raise ValueError("kg.fantasies must be at least two")
    positive = {
        "kg.temperature": config.kg_temperature,
        "search.credit_floor": config.credit_floor,
        "search.credit_ceiling": config.credit_ceiling,
        "search.widening_coefficient": config.widening_coefficient,
        "replay.temperature": config.replay_temperature,
        "replay.clip": config.replay_clip,
        "replay.learning_rate": config.learning_rate,
        "replay.preference_beta": config.preference_beta,
        "replay.preference_clip": config.preference_clip,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"strictly positive configuration required: {positive}")
    if not 0 < config.credit_floor <= 1 <= config.credit_ceiling:
        raise ValueError("credit bounds must satisfy 0 < floor <= 1 <= ceiling")
    if not 0 <= config.credit_mix <= 1 or not 0 <= config.credit_decay <= 1:
        raise ValueError("credit_mix and credit_decay must lie in [0, 1]")
    if not 0 < config.widening_exponent < 1:
        raise ValueError("widening_exponent must lie in (0, 1)")
    if config.exploration_scale < 0 or config.thompson_scale < 0 or config.risk_kappa < 0:
        raise ValueError("search scales must be non-negative")
    if not 0 <= config.random_evaluation_reserve < config.evaluations:
        raise ValueError(
            "random_evaluation_reserve must be non-negative and smaller than evaluations"
        )
    if not 0 < config.max_violation_probability < 1:
        raise ValueError("max_violation_probability must lie in (0, 1)")
    if not 0 < config.replay_timestep <= 1:
        raise ValueError("replay timestep must lie in (0, 1]")
    if config.kg_standard_error_multiplier < 0:
        raise ValueError("kg.standard_error_multiplier must be non-negative")
    if config.local_kl_limit <= 0 or config.reference_path_kl_limit <= 0:
        raise ValueError("both KL limits must be positive")
    if config.local_kl_p99_limit <= 0 or config.reference_transition_kl_p99_limit <= 0:
        raise ValueError("both KL p99 limits must be positive")
    kg_slots = config.evaluations - config.random_evaluation_reserve
    if kg_slots > config.batching.kg_max_joint_size:
        raise ValueError("KG evaluation slots exceed batching.kg_max_joint_size")
    if config.batching.proposal_batch_size > 64:
        raise ValueError("local smoke proposal batches cannot exceed 64")
    if config.batching.replay_limits.length_bucket_boundaries != (config.length,):
        raise ValueError("local smoke replay batching must use the fixed peptide length")
    return config


class _PlantedOracle:
    """Deterministic additive-plus-adjacent-pair two-output oracle."""

    def __init__(self, config: SearchSmokeConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, str, str]] = []
        self.batch_calls: list[tuple[str, ...]] = []
        self._seen: set[tuple[str, str, str]] = set()

    def evaluate(self, sequence: str) -> tuple[float, float]:
        key = (sequence, self.config.fidelity, self.config.evaluator_version)
        if key in self._seen:
            raise RuntimeError(f"duplicate compatible oracle call: {key}")
        self._seen.add(key)
        self.calls.append(key)
        return _planted_outcomes(sequence)

    def evaluate_batch(
        self,
        sequences: tuple[str, ...],
        *,
        batch_size: int,
    ) -> FloatArray:
        """Evaluate deterministic toy-oracle requests in bounded batches."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("oracle batch_size must be a positive integer")
        outcomes: list[tuple[float, float]] = []
        for planned in plan_count_batches(len(sequences), batch_size):
            batch = sequences[planned.start : planned.stop]
            self.batch_calls.append(batch)
            outcomes.extend(self.evaluate(sequence) for sequence in batch)
        return np.asarray(outcomes, dtype=np.float64).reshape(len(sequences), 2)


def _planted_outcomes(sequence: str) -> tuple[float, float]:
    additive_activity = {"A": 0.04, "D": -0.12, "K": 0.42, "R": 0.34}
    additive_toxicity = {"A": 0.02, "D": -0.04, "K": 0.09, "R": 0.24}
    pair_activity = {("D", "K"): 0.12, ("K", "R"): 0.24, ("R", "K"): 0.16}
    pair_toxicity = {("K", "R"): 0.06, ("R", "R"): 0.38}
    activity = -0.35 + sum(additive_activity[token] for token in sequence)
    toxicity = 0.18 + sum(additive_toxicity[token] for token in sequence)
    activity += sum(pair_activity.get(pair, 0.0) for pair in itertools.pairwise(sequence))
    toxicity += sum(pair_toxicity.get(pair, 0.0) for pair in itertools.pairwise(sequence))
    return float(activity), float(toxicity)


def _toy_contact_probabilities(sequence: str, alphabet: str) -> FloatArray:
    """Return a sequence-dependent contact matrix for smoke wiring only.

    This deterministic adapter uses frozen token ordinals and sequence distance.
    It does not read planted outcomes or oracle parameters and carries no claim
    of biological contact accuracy or useful discrimination.  A cluster
    experiment must replace it with a separately frozen and audited contact
    producer.
    """

    if len(sequence) < 2:
        raise ValueError("spectral smoke features require sequences of length at least two")
    token_index = {token: index for index, token in enumerate(alphabet)}
    try:
        strengths = np.asarray(
            [(token_index[token] + 1) / len(alphabet) for token in sequence],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(
            f"sequence contains token outside the declared alphabet: {error.args[0]}"
        ) from error
    positions = np.arange(len(sequence))
    separation = np.abs(positions[:, None] - positions[None, :])
    distance_decay = 1.0 / (1.0 + 0.2 * np.maximum(separation - 3, 0))
    contacts = np.sqrt(np.outer(strengths, strengths)) * distance_decay
    contacts[separation < 3] = 0.0
    return contacts


def _features(
    sequences: tuple[str, ...],
    alphabet: str,
    *,
    batch_size: int | None = None,
) -> FloatArray:
    """Build feature rows in bounded chunks while retaining source order."""

    if batch_size is None:
        batch_size = max(len(sequences), 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("feature batch_size must be a positive integer")
    token_index = {token: index for index, token in enumerate(alphabet)}
    sequence_width = 1 + len(alphabet) + len(alphabet) ** 2
    spectral_offset = sequence_width
    length_offset = spectral_offset + SPECTRAL_CONFIG.bands
    width = length_offset + 1
    result = np.zeros((len(sequences), width), dtype=np.float64)
    result[:, 0] = 1.0
    for planned in plan_count_batches(len(sequences), batch_size):
        batch = sequences[planned.start : planned.stop]
        signatures = laplacian_log_spectral_densities(
            [_toy_contact_probabilities(sequence, alphabet) for sequence in batch],
            config=SPECTRAL_CONFIG,
            batch_size=max(len(batch), 1),
        )
        result[planned.start : planned.stop, spectral_offset:length_offset] = signatures
        for row, sequence in enumerate(
            batch,
            start=planned.start,
        ):
            for token in sequence:
                try:
                    token_position = token_index[token]
                except KeyError as error:
                    raise ValueError(
                        f"sequence contains token outside the declared alphabet: {error.args[0]}"
                    ) from error
                result[row, 1 + token_position] += 1.0
            pair_offset = 1 + len(alphabet)
            for left, right in itertools.pairwise(sequence):
                pair = token_index[left] * len(alphabet) + token_index[right]
                result[row, pair_offset + pair] += 1.0
            result[row, length_offset] = math.log(len(sequence) / 50.0)
    return result


def _posterior(
    sequences: tuple[str, ...],
    seed_outcomes: FloatArray,
    *,
    batch_size: int | None = None,
) -> JointGaussianPosterior:
    embeddings = _features(sequences, "ADKR", batch_size=batch_size)
    # The acquisition model must learn from the seed observations.  Encoding
    # the planted oracle weights here would leak the smoke fixture's answer.
    # Independent priors over the sequence, spectral, and length columns make
    # this finite Bayesian-linear adapter an additive-kernel smoke analogue.
    weight_mean = np.zeros((embeddings.shape[1], 2), dtype=np.float64)
    n_features, n_outputs = weight_mean.shape
    covariance = np.zeros((n_features, n_outputs, n_features, n_outputs))
    spectral_start = 1 + len("ADKR") + len("ADKR") ** 2
    spectral_stop = spectral_start + SPECTRAL_CONFIG.bands
    for feature in range(n_features):
        scale = (
            SPECTRAL_PRIOR_VARIANCE_MULTIPLIER if spectral_start <= feature < spectral_stop else 1.0
        )
        covariance[feature, 0, feature, 0] = scale * 0.012
        covariance[feature, 1, feature, 1] = scale * 0.006
        covariance[feature, 0, feature, 1] = scale * 0.002
        covariance[feature, 1, feature, 0] = scale * 0.002
    noise = np.zeros((len(sequences), 2, 2), dtype=np.float64)
    noise[:, 0, 0] = 0.0025
    noise[:, 1, 1] = 0.0016
    prior = JointGaussianPosterior.from_bayesian_linear(
        embeddings,
        weight_mean,
        covariance,
        observation_noise=noise,
    )
    return prior.condition(tuple(range(len(SEED_SEQUENCES))), seed_outcomes)


def _with_no_action_decision(
    posterior: JointGaussianPosterior,
) -> JointGaussianPosterior:
    """Append a deterministic status-quo action that is not an evaluable peptide."""

    n_points, n_outputs = posterior.mean.shape
    mean = np.zeros((n_points + 1, n_outputs), dtype=np.float64)
    mean[:n_points] = posterior.mean
    covariance = np.zeros(
        (n_points + 1, n_outputs, n_points + 1, n_outputs),
        dtype=np.float64,
    )
    covariance[:n_points, :, :n_points, :] = posterior.covariance
    observation_noise = np.zeros((n_points + 1, n_outputs, n_outputs), dtype=np.float64)
    observation_noise[:n_points] = posterior.observation_noise
    return JointGaussianPosterior(mean, covariance, observation_noise)


def _utility(outcomes: FloatArray, toxicity_upper: float) -> FloatArray:
    values = np.asarray(outcomes, dtype=np.float64)
    return values[..., 0] - 0.35 * np.maximum(values[..., 1] - 0.5 * toxicity_upper, 0.0)


def _hamming_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> FloatArray:
    distances = np.asarray(
        [[sum(a != b for a, b in zip(x, y, strict=True)) for y in right] for x in left],
        dtype=np.float64,
    )
    return np.exp(-distances / 2.0)


def _single_substitutions(parent: str, alphabet: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for position, current in enumerate(parent):
        for residue in alphabet:
            if residue != current:
                candidates.append(parent[:position] + residue + parent[position + 1 :])
    return tuple(candidates)


def _thompson_substitution_weights(
    parent: str,
    alphabet: str,
    base_residue_probabilities: FloatArray,
    gains: FloatArray,
    *,
    scale: float,
) -> tuple[FloatArray, FloatArray]:
    """Backward-compatible scalar view of the batched Thompson tilt."""

    positions, residues = thompson_substitution_weights_batch(
        (parent,),
        alphabet,
        base_residue_probabilities,
        np.asarray(gains, dtype=np.float64)[None, :, :],
        scale=scale,
    )
    return positions[0], residues[0]


def _hard_valid(sequence: str, config: SearchSmokeConfig) -> tuple[bool, str | None]:
    if len(sequence) != config.length:
        return False, "length"
    if set(sequence) - set(config.alphabet):
        return False, "alphabet"
    return True, None


def _realized_feasible_improvement(
    parent_utility: float,
    parent_feasible: bool,
    child_utilities: FloatArray,
    child_feasible: tuple[bool, ...] | list[bool],
    *,
    reference: float,
) -> float:
    """Return one-dimensional hypervolume gain from declared feasible children."""

    utilities = np.asarray(child_utilities, dtype=np.float64)
    if not isinstance(parent_feasible, bool):
        raise ValueError("parent_feasible must be a bool")
    raw_feasible = np.asarray(child_feasible)
    if raw_feasible.size and raw_feasible.dtype.kind != "b":
        raise ValueError("child feasibility must contain booleans")
    feasible = np.asarray(raw_feasible, dtype=bool)
    if utilities.ndim != 1 or feasible.shape != utilities.shape:
        raise ValueError("child utilities and feasibility must be aligned vectors")
    if (
        not np.isfinite(parent_utility)
        or not np.isfinite(reference)
        or np.any(~np.isfinite(utilities))
    ):
        raise ValueError("realized utilities must be finite")
    hypervolume_before = max(float(parent_utility) - reference, 0.0) if parent_feasible else 0.0
    feasible_values = [float(parent_utility)] if parent_feasible else []
    if np.any(feasible):
        feasible_values.append(float(np.max(utilities[feasible])))
    hypervolume_after = max(max(feasible_values) - reference, 0.0) if feasible_values else 0.0
    return hypervolume_after - hypervolume_before


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _float(value: float) -> str:
    return format(float(value), ".17g")


def _optional_float(value: float | None) -> str:
    return "" if value is None else _float(value)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _root_trace() -> ProbabilityTrace:
    return ProbabilityTrace(
        (
            ProbabilityFactor("branch", 1.0),
            ProbabilityFactor("operator", 1.0),
            ProbabilityFactor("edit", 1.0),
        )
    )


def _selection_decision(
    *,
    selected: bool,
    selection_probability: float,
    selection_set: str,
    eligible_ids: tuple[str, ...],
    policy_version: str,
    seed: int,
) -> SelectionDecision:
    return SelectionDecision(
        selected=selected,
        propensity=ProbabilityTrace(
            (ProbabilityFactor("expensive_selection", selection_probability),)
        ),
        selection_set_id=selection_set,
        eligible_proposal_ids=eligible_ids,
        policy_version=policy_version,
        seed=seed,
    )


def _recommendation_snapshot(
    recommendation: PosteriorMeanRecommendation,
    decision_labels: tuple[str, ...],
) -> dict[str, object]:
    """Serialize a hard terminal ranking without acquisition information."""

    ranking = recommendation.ranked_decision_indices
    rank_by_decision = {decision: rank for rank, decision in enumerate(ranking, start=1)}
    selected = recommendation.selected_decision_index
    return {
        "selected_decision_index": selected,
        "selected_sequence": None if selected is None else decision_labels[selected],
        "selected_posterior_mean_utility": (recommendation.selected_posterior_mean_utility),
        "terminal_posterior_mean_utility": recommendation.terminal_posterior_mean_utility,
        "abstained": selected is None,
        "abstention_reason": recommendation.abstention_reason,
        "outside_option_decision_indices": list(recommendation.outside_option_decision_indices),
        "outside_option_posterior_mean_utility": (
            recommendation.outside_option_posterior_mean_utility
        ),
        "ranked_decision_indices": list(ranking),
        "ranked_sequences": [decision_labels[index] for index in ranking],
        "recommendable_decisions": len(recommendation.decision_indices),
        "chance_feasible_decisions": int(np.count_nonzero(recommendation.chance_feasible)),
        "candidates": [
            {
                "decision_index": decision,
                "sequence": decision_labels[decision],
                "posterior_mean_utility": float(
                    recommendation.expected_posterior_mean_utility[position]
                ),
                "chance_feasible": bool(recommendation.chance_feasible[position]),
                "constraint_satisfaction_probabilities": (
                    recommendation.constraint_satisfaction_probability[position].tolist()
                ),
                "feasible_rank": rank_by_decision.get(decision),
            }
            for position, decision in enumerate(recommendation.decision_indices)
        ],
    }


def _post_hoc_regret_diagnostics(
    *,
    universe: tuple[str, ...],
    problem: SoftKGProblem,
    query_indices: tuple[int, ...],
    query_sources: tuple[str, ...],
    before_query: PosteriorMeanRecommendation,
    after_update: PosteriorMeanRecommendation,
    decision_set_sha256: str,
    initial_design_count: int,
) -> dict[str, object]:
    """Evaluate regret on the analytic toy truth after all search updates.

    This diagnostic deliberately bypasses :class:`_PlantedOracle`: enumerating
    the closed-form smoke landscape is not an experimental assay and none of
    these values may enter selection, posterior conditioning, or policy replay.
    """

    if len(query_indices) != len(query_sources):
        raise ValueError("query indices and sources must be aligned")
    if initial_design_count < 0:
        raise ValueError("initial_design_count cannot be negative")
    outcomes = np.asarray([_planted_outcomes(sequence) for sequence in universe])
    objective_mean = outcomes[:, problem.objective_outputs]
    preference_utility = objective_mean @ problem.preferences.weights.T
    true_utility = preference_utility @ problem.preferences.probabilities
    true_feasible = np.ones(len(universe), dtype=bool)
    for constraint in problem.constraints:
        true_feasible &= outcomes[:, constraint.output_index] <= constraint.upper_bound

    # The KG problem declares an always-safe no-action decision. Its value must
    # remain frozen across the measurement before regrets share a reference.
    # Assigning that value to a truly unsafe candidate keeps constrained regret
    # finite while preserving the safety failure as separate evidence.
    outside_option_value = before_query.outside_option_posterior_mean_utility
    if outside_option_value is None:
        raise ValueError("constrained regret diagnostics require a declared outside option")
    after_update_outside_value = after_update.outside_option_posterior_mean_utility
    if after_update_outside_value is None or not math.isclose(
        outside_option_value,
        after_update_outside_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("outside-option utility must remain fixed across the measurement")
    feasible_values = true_utility[true_feasible]
    reference_value = max(
        outside_option_value,
        float(np.max(feasible_values)) if feasible_values.size else outside_option_value,
    )
    optimal_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            true_feasible & np.isclose(true_utility, reference_value, rtol=0.0, atol=1e-12)
        )
    )
    no_action_is_optimal = math.isclose(
        outside_option_value,
        reference_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    def decision_evidence(index: int | None) -> dict[str, object]:
        if index is None:
            return {
                "decision_index": None,
                "sequence": None,
                "true_objective_utility": None,
                "true_feasible": True,
                "feasibility_adjusted_value": outside_option_value,
                "regret": reference_value - outside_option_value,
                "constraint_evidence": [],
            }
        feasible = bool(true_feasible[index])
        decision_value = float(true_utility[index]) if feasible else outside_option_value
        return {
            "decision_index": index,
            "sequence": universe[index],
            "true_objective_utility": float(true_utility[index]),
            "true_feasible": feasible,
            "feasibility_adjusted_value": decision_value,
            "regret": max(reference_value - decision_value, 0.0),
            "constraint_evidence": [
                {
                    "output_index": constraint.output_index,
                    "true_value": float(outcomes[index, constraint.output_index]),
                    "upper_bound": constraint.upper_bound,
                    "satisfied": bool(
                        outcomes[index, constraint.output_index] <= constraint.upper_bound
                    ),
                }
                for constraint in problem.constraints
            ],
        }

    query_evidence = []
    for index, source in zip(query_indices, query_sources, strict=True):
        record = decision_evidence(index)
        record["source"] = source
        query_evidence.append(record)
    query_regrets = [float(record["regret"]) for record in query_evidence]

    return {
        "scope": "post_hoc_enumerated_analytic_toy_frozen_decision_set",
        "search_visible": False,
        "search_state_updated": False,
        "computed_after_policy_update": True,
        "production_note": "requires genuinely held-out truth; unavailable during live search",
        "frozen_real_decision_set_sha256": decision_set_sha256,
        "post_hoc_synthetic_truth_lookups": len(universe),
        "expensive_oracle_calls_added": 0,
        "objective": {
            "outputs": list(problem.objective_outputs),
            "preference_weights": problem.preferences.weights.tolist(),
            "preference_probabilities": problem.preferences.probabilities.tolist(),
            "unsafe_decision_value": "declared_no_action_posterior_mean_utility",
            "outside_option_value": outside_option_value,
        },
        "reference": {
            "value": reference_value,
            "optimal_real_decision_indices": list(optimal_indices),
            "optimal_real_sequences": [universe[index] for index in optimal_indices],
            "no_action_is_optimal": no_action_is_optimal,
        },
        "query_regret": {
            "scope": "current_adaptive_measurement_batch_only",
            "unique_oracle_call_offset_before_batch": initial_design_count,
            "unique_oracle_calls_after_batch": initial_design_count + len(query_evidence),
            "points": query_evidence,
            "cumulative": float(sum(query_regrets)),
            "mean": None if not query_regrets else float(np.mean(query_regrets)),
            "best_current_batch": None if not query_regrets else float(min(query_regrets)),
        },
        "inference_regret": {
            "before_query": decision_evidence(before_query.selected_decision_index),
            "after_posterior_update": decision_evidence(after_update.selected_decision_index),
        },
    }


def run_smoke(config_path: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Run the deterministic vertical slice and write five audit artifacts."""

    config_path = Path(config_path)
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"output path must be a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    config_hash = _hash_file(config_path)
    oracle = _PlantedOracle(config)

    seed_outcomes = oracle.evaluate_batch(
        SEED_SEQUENCES,
        batch_size=config.batching.oracle_batch_size,
    )
    seed_utility = _utility(seed_outcomes, config.toxicity_upper)
    credit_domain = tuple(
        dict.fromkeys(
            (
                *SEED_SEQUENCES,
                *(
                    neighbor
                    for seed_sequence in SEED_SEQUENCES
                    for neighbor in _single_substitutions(seed_sequence, config.alphabet)
                ),
            )
        )
    )
    credit_domain_hash = _hash_bytes("\n".join(credit_domain).encode())
    credit_posterior = _posterior(
        credit_domain,
        seed_outcomes,
        batch_size=config.batching.surrogate_batch_size,
    )
    toxicity_constraint = UpperChanceConstraint(
        1,
        config.toxicity_upper,
        config.max_violation_probability,
    )
    seed_indices = np.arange(config.branches)
    seed_chance_feasible = toxicity_constraint.is_satisfied(
        credit_posterior.mean[: config.branches, 1],
        credit_posterior.covariance[seed_indices, 1, seed_indices, 1],
    )
    credit_draws = credit_posterior.sample_functions(config.kg_fantasies, seed=config.seed + 1)
    credit_utility_draws = _utility(credit_draws, config.toxicity_upper)
    optimum_proxy = np.max(credit_utility_draws, axis=1)
    influence = historical_posterior_influence_credit(
        optimum_proxy,
        np.mean(credit_utility_draws[:, : config.branches], axis=0),
        np.var(credit_utility_draws[:, : config.branches], axis=0, ddof=1),
        _hamming_similarity(SEED_SEQUENCES, SEED_SEQUENCES),
        credit_floor=config.credit_floor,
        credit_ceiling=config.credit_ceiling,
        sensitivity=1.0,
        iteration=1.0,
        half_life=10.0,
        weight_floor=config.credit_floor,
        weight_ceiling=config.credit_ceiling,
    )
    seed_branch_yields = np.where(
        seed_chance_feasible,
        np.maximum(seed_utility - config.branch_yield_reference, 0.0),
        0.0,
    )
    pooled_branch_yield_std = float(np.std(seed_branch_yields, ddof=1))
    branches = tuple(
        BranchRecord(
            branch_id=f"branch-{index:02d}",
            niche=(f"seed={sequence}", config.context_id),
            elite_sequence_keys=(_hash_bytes(sequence.encode("ascii")),),
            parent_branch_id=None,
            policy_version="policy-v0",
            kl_radius=config.local_kl_limit,
            visits=1,
            unique_evaluated_descendants=0,
            yield_mean=float(seed_branch_yields[index]),
            yield_std=pooled_branch_yield_std,
            credit=float(influence.decayed_weight[index]),
            credit_floor=config.credit_floor,
            credit_ceiling=config.credit_ceiling,
            credit_decay=config.credit_decay,
        )
        for index, sequence in enumerate(SEED_SEQUENCES)
    )
    branch_scores = tuple(
        branch_score(
            branch,
            exploration_scale=config.exploration_scale,
            credit_mix=config.credit_mix,
        )
        for branch in branches
    )
    selected_branch_position = max(
        range(len(branches)), key=lambda index: (branch_scores[index], -index)
    )
    selected_branch = branches[selected_branch_position]
    parent = SEED_SEQUENCES[selected_branch_position]

    widening_limit = progressive_widening_limit(
        visits=selected_branch.visits,
        coefficient=config.widening_coefficient,
        exponent=config.widening_exponent,
    )
    if config.proposals > widening_limit:
        raise ValueError("proposal count exceeds the progressive-widening child budget")

    universe = tuple(
        dict.fromkeys((*SEED_SEQUENCES, *_single_substitutions(parent, config.alphabet)))
    )
    point_index = {sequence: index for index, sequence in enumerate(universe)}
    posterior = _posterior(
        universe,
        seed_outcomes,
        batch_size=config.batching.surrogate_batch_size,
    )
    rollout_preference = np.zeros(posterior.n_outputs, dtype=np.float64)
    rollout_preference[0] = 1.0
    rollout_batch = sample_frozen_rollouts(
        posterior,
        (selected_branch.branch_id,),
        candidate_ids=universe,
        output_ids=("activity", "toxicity"),
        rollout_namespace="toy-smoke-round-00000",
        rollout_indices=(0,),
        contexts=(
            (
                ("context_id", config.context_id),
                ("preference", tuple(float(value) for value in rollout_preference)),
                ("toxicity_upper", config.toxicity_upper),
                ("posterior_conditioned_observations", len(SEED_SEQUENCES)),
            ),
        ),
        preference_weights=np.repeat(
            rollout_preference[None, :],
            config.batching.rollout_batch_size,
            axis=0,
        ),
        policy_version="policy-v0",
        root_seed=config.seed + 2,
    )
    rollout = rollout_batch.records[0]
    fixed_draw = rollout_batch.draw_for_rollout(rollout.rollout_id)
    draw_id = rollout.posterior_draw_id
    sampled_utility = _utility(fixed_draw, config.toxicity_upper)

    vocabulary = PeptideVocabulary(config.alphabet)
    policy = TabularConditionalDenoiser.uniform(
        vocabulary=vocabulary, contexts=(config.context_id,), length=config.length
    )
    current_probabilities = policy.probabilities(config.context_id)
    substitution_gains = np.zeros(
        (config.length, len(config.alphabet)),
        dtype=np.float64,
    )
    parent_sampled_utility = float(sampled_utility[point_index[parent]])
    for position, current in enumerate(parent):
        for residue_index, residue in enumerate(config.alphabet):
            if residue == current:
                continue
            candidate = parent[:position] + residue + parent[position + 1 :]
            candidate_utility = float(sampled_utility[point_index[candidate]])
            substitution_gains[position, residue_index] = candidate_utility - parent_sampled_utility
    batched_position_weights, batched_residue_weights = thompson_substitution_weights_batch(
        (parent,),
        config.alphabet,
        current_probabilities,
        substitution_gains[None, :, :],
        scale=config.thompson_scale,
    )
    position_weights = batched_position_weights[0]
    residue_weights = batched_residue_weights[0]
    emitter = SubstitutionEmitter(alphabet=tuple(config.alphabet))
    emitted = []
    proposal_batch_sizes: list[int] = []
    # Proposal coverage is a search decision, not an assay-budget side effect.
    # Force one draw per position before filling the remaining proposal budget.
    for position in range(config.length):
        forced_positions = np.zeros(config.length, dtype=np.float64)
        forced_positions[position] = position_weights[position]
        emitted.extend(
            emitter.emit(
                parent=parent,
                rollout=rollout,
                count=1,
                random_stream=position,
                position_weights=forced_positions,
                residue_weights=residue_weights,
            )
        )
        proposal_batch_sizes.append(1)
    remaining = config.proposals - len(emitted)
    for planned in plan_count_batches(remaining, config.batching.proposal_batch_size):
        batch_count = planned.size
        emitted.extend(
            emitter.emit(
                parent=parent,
                rollout=rollout,
                count=batch_count,
                sample_start=planned.start,
                random_stream=100,
                position_weights=position_weights,
                residue_weights=residue_weights,
            )
        )
        proposal_batch_sizes.append(batch_count)
    emissions = tuple(emitted)

    ledger = SearchLedger()
    for branch in branches:
        ledger.add_branch(branch)
    proposal_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for index, (branch, sequence, outcomes) in enumerate(
        zip(branches, SEED_SEQUENCES, seed_outcomes, strict=True)
    ):
        seed_rollout = RolloutRecord(
            rollout_id=f"seed-rollout-{index:02d}",
            branch_id=branch.branch_id,
            posterior_draw_id="observed-seed",
            context=(("context_id", config.context_id),),
            policy_version="policy-v0",
            seed=config.seed,
        )
        ledger.add_rollout(seed_rollout)
        proposal_id = f"seed-{index:03d}"
        edge_id = f"seed-edge-{index:03d}"
        seed_proposal = ProposalRecord(
            proposal_id=proposal_id,
            rollout_id=seed_rollout.rollout_id,
            sequence=sequence,
            hard_valid=True,
            rejection_reason=None,
            selection=_selection_decision(
                selected=True,
                selection_probability=1.0,
                selection_set=f"seed-selection-{index:03d}",
                eligible_ids=(proposal_id,),
                policy_version="observed-seed-v1",
                seed=config.seed,
            ),
            policy_version="policy-v0",
            proposal_round=0,
            niche_id=branch.branch_id,
        )
        seed_edge = EdgeRecord(
            edge_id=edge_id,
            proposal_id=proposal_id,
            rollout_id=seed_rollout.rollout_id,
            parent_sequence_keys=(),
            operator="seed",
            edit_description=(("observed_seed", index),),
            proposal_trace=_root_trace(),
        )
        seed_event = ledger.record_proposal(seed_proposal, seed_edge)
        evaluation = EvaluationRecord(
            evaluation_id=f"seed-eval-{index:03d}",
            proposal_id=proposal_id,
            fidelity=config.fidelity,
            evaluator_version=config.evaluator_version,
            cost=1.0,
            outcomes=(("activity", float(outcomes[0])), ("toxicity", float(outcomes[1]))),
            batch_id="seed-batch",
            replicate=0,
        )
        ledger.record_evaluation(evaluation)
        proposal_rows.append(_proposal_row(seed_proposal, seed_event, "seed", "", "", seed_rollout))
        stored_seed_edge = ledger.edge(seed_edge.edge_id)
        edge_rows.append(
            _edge_row(
                stored_seed_edge,
                rollout=ledger.rollout(stored_seed_edge.rollout_id),
            )
        )
        evaluation_rows.append(_evaluation_row(evaluation, sequence))
    ledger.add_rollout(rollout)

    unique_emitted = tuple(dict.fromkeys(emission.sequence for emission in emissions))
    unique_indices = tuple(point_index[sequence] for sequence in unique_emitted)
    hard_eligible = tuple(
        _hard_valid(sequence, config)[0]
        and not ledger.cached_evaluations(
            sequence, fidelity=config.fidelity, evaluator_version=config.evaluator_version
        )
        for sequence in unique_emitted
    )
    kg_posterior = _with_no_action_decision(posterior)
    no_action_index = len(universe)
    kg_problem = SoftKGProblem(
        decision_indices=tuple(range(kg_posterior.n_points)),
        objective_outputs=(0,),
        preferences=PreferenceMeasure(np.ones((1, 1))),
        base_measure=np.ones(kg_posterior.n_points),
        constraints=(toxicity_constraint,),
        always_safe_decisions=(no_action_index,),
    )
    real_decision_indices = tuple(range(len(universe)))
    before_query_recommendation = recommend_posterior_mean(
        kg_posterior,
        kg_problem,
        recommendable_indices=real_decision_indices,
    )
    kg_acquisition = GaussianSoftKG(
        kg_problem,
        temperature=config.kg_temperature,
        observed_outputs=(0, 1),
        n_fantasies=config.kg_fantasies,
        standard_error_multiplier=config.kg_standard_error_multiplier,
        seed=config.seed + 4,
        candidate_chunk_size=config.batching.kg_candidate_chunk_size,
        fantasy_chunk_size=config.batching.kg_fantasy_chunk_size,
    )
    evaluation_batch = EvaluationBatch(
        indices=unique_indices,
        costs=np.ones(len(unique_indices)),
        eligible=np.asarray(hard_eligible, dtype=bool),
    )
    kg_result = kg_acquisition.score(
        kg_posterior,
        evaluation_batch,
    )
    result_position = {point: position for position, point in enumerate(unique_indices)}
    eligible_points = [
        point for point, eligible in zip(unique_indices, hard_eligible, strict=True) if eligible
    ]
    kg_slots = config.evaluations - config.random_evaluation_reserve
    joint_size = min(kg_slots, len(eligible_points))
    if joint_size:
        joint_pool_limit = maximum_exhaustive_pool_size(
            batch_size=joint_size,
            max_combinations=config.batching.kg_max_combinations,
            candidate_count=len(eligible_points),
        )
        eligible_result_positions = np.flatnonzero(kg_result.eligible)
        screened_positions = eligible_result_positions[
            np.argsort(-kg_result.score[eligible_result_positions], kind="stable")[
                :joint_pool_limit
            ]
        ]
        joint_evaluation_batch = EvaluationBatch(
            indices=tuple(unique_indices[int(position)] for position in screened_positions),
            costs=evaluation_batch.costs[screened_positions],
        )
    else:
        joint_pool_limit = 0
        joint_evaluation_batch = None
    joint_kg_result = (
        kg_acquisition.select_joint(
            kg_posterior,
            joint_evaluation_batch,
            batch_size=joint_size,
            max_combinations=config.batching.kg_max_combinations,
        )
        if joint_evaluation_batch is not None
        else None
    )
    kg_selected_points = (
        () if joint_kg_result is None else joint_kg_result.selected_evaluation_indices
    )
    reserve_pool = tuple(point for point in eligible_points if point not in set(kg_selected_points))
    reserve_count = min(config.random_evaluation_reserve, len(reserve_pool))
    if reserve_count:
        reserve_rng = np.random.default_rng(config.seed + 5)
        reserve_positions = reserve_rng.choice(len(reserve_pool), size=reserve_count, replace=False)
        reserve_selected_points = tuple(
            reserve_pool[int(position)] for position in np.atleast_1d(reserve_positions)
        )
    else:
        reserve_selected_points = ()
    selected_points = (*kg_selected_points, *reserve_selected_points)
    selected_sequences = tuple(universe[index] for index in selected_points)
    first_id_by_sequence: dict[str, str] = {}
    for index, emission in enumerate(emissions):
        first_id_by_sequence.setdefault(emission.sequence, f"proposal-{index:03d}")
    eligible_ids = tuple(first_id_by_sequence[universe[index]] for index in eligible_points)
    selected_ids = {first_id_by_sequence[sequence] for sequence in selected_sequences}
    kg_selected_ids = {first_id_by_sequence[universe[index]] for index in kg_selected_points}
    reserve_selected_ids = {
        first_id_by_sequence[universe[index]] for index in reserve_selected_points
    }
    reserve_ids = {first_id_by_sequence[universe[index]] for index in reserve_pool}
    reserve_probability = reserve_count / len(reserve_pool) if reserve_pool else 0.0

    prefix = ProbabilityTrace(
        (ProbabilityFactor("branch", 1.0), ProbabilityFactor("operator", 1.0))
    )
    for index, emission in enumerate(emissions):
        proposal_id = f"proposal-{index:03d}"
        hard_valid, reason = _hard_valid(emission.sequence, config)
        selected = proposal_id in selected_ids
        proposal = ProposalRecord(
            proposal_id=proposal_id,
            rollout_id=rollout.rollout_id,
            sequence=emission.sequence,
            hard_valid=hard_valid,
            rejection_reason=reason,
            selection=_selection_decision(
                selected=selected,
                selection_probability=(
                    1.0
                    if proposal_id in kg_selected_ids
                    else reserve_probability
                    if proposal_id in reserve_ids
                    else 0.0
                ),
                selection_set="selection-round-00",
                eligible_ids=eligible_ids,
                policy_version="joint-soft-kg-uniform-reserve-v2",
                seed=config.seed + 5,
            ),
            policy_version="policy-v0",
            proposal_round=0,
            niche_id=selected_branch.branch_id,
            cheap_predictions=(
                ("activity_mean", float(posterior.mean[point_index[emission.sequence], 0])),
                ("toxicity_mean", float(posterior.mean[point_index[emission.sequence], 1])),
            ),
        )
        edge = emission.to_edge_record(
            edge_id=f"edge-{index:03d}",
            proposal_id=proposal_id,
            proposal_prefix=prefix,
        )
        event = ledger.record_proposal(proposal, edge)
        kg_position = result_position[point_index[emission.sequence]]
        proposal_rows.append(
            _proposal_row(
                proposal,
                event,
                "generated",
                parent,
                _float(kg_result.score[kg_position]),
                rollout,
                selection_source=(
                    "joint_soft_kg"
                    if proposal_id in kg_selected_ids
                    else "uniform_reserve"
                    if proposal_id in reserve_selected_ids
                    else ""
                ),
                singleton_kg_estimate=_float(kg_result.estimate[kg_position]),
                singleton_kg_standard_error=_float(kg_result.standard_error[kg_position]),
                singleton_kg_standard_error_penalized_estimate=_float(
                    kg_result.standard_error_penalized_estimate[kg_position]
                ),
            )
        )
        stored_edge = ledger.edge(edge.edge_id)
        edge_rows.append(
            _edge_row(
                stored_edge,
                rollout=ledger.rollout(stored_edge.rollout_id),
            )
        )
    ledger.validate_selection_sets()

    revealed = oracle.evaluate_batch(
        selected_sequences,
        batch_size=config.batching.oracle_batch_size,
    )
    for index, (sequence, outcomes) in enumerate(zip(selected_sequences, revealed, strict=True)):
        proposal_id = first_id_by_sequence[sequence]
        evaluation = EvaluationRecord(
            evaluation_id=f"evaluation-{index:03d}",
            proposal_id=proposal_id,
            fidelity=config.fidelity,
            evaluator_version=config.evaluator_version,
            cost=1.0,
            outcomes=(("activity", float(outcomes[0])), ("toxicity", float(outcomes[1]))),
            batch_id="round-00",
            replicate=0,
        )
        ledger.record_evaluation(evaluation)
        evaluation_rows.append(_evaluation_row(evaluation, sequence))
    final_posterior = (
        posterior.condition(selected_points, revealed) if selected_points else posterior
    )
    after_update_recommendation = recommend_posterior_mean(
        _with_no_action_decision(final_posterior),
        kg_problem,
        recommendable_indices=real_decision_indices,
    )

    parent_index = point_index[parent]
    contrast_means: list[float] = []
    contrast_variances: list[float] = []
    absolute_risk_score: list[float] = []
    chance_feasible: list[bool] = []
    if selected_points:
        paired_draws = final_posterior.sample_functions(96, seed=config.seed + 6)
        for child_index in selected_points:
            joint = np.stack(
                [paired_draws[:, child_index, :], paired_draws[:, parent_index, :]],
                axis=1,
            )
            contrast = sampled_paired_contrast(
                joint,
                lambda values, threshold: _utility(values, float(threshold)),
                context=config.toxicity_upper,
            )
            contrast_means.append(contrast.mean)
            contrast_variances.append(contrast.variance)
            child_utility_draws = _utility(paired_draws[:, child_index, :], config.toxicity_upper)
            absolute_risk_score.append(
                float(
                    np.mean(child_utility_draws)
                    - config.risk_kappa * np.std(child_utility_draws, ddof=1)
                )
            )
            toxicity_mean = float(final_posterior.mean[child_index, 1])
            toxicity_variance = float(final_posterior.covariance[child_index, 1, child_index, 1])
            chance_feasible.append(
                bool(
                    kg_problem.constraints[0].is_satisfied(
                        toxicity_mean,
                        toxicity_variance,
                    )
                )
            )
    parent_utility = float(_utility(seed_outcomes[selected_branch_position], config.toxicity_upper))
    if selected_points:
        gates = gate_paired_advantages(
            contrast_means,
            contrast_variances,
            absolute_risk_score,
            chance_feasible,
            risk_kappa=config.risk_kappa,
            minimum_utility=np.full(len(selected_points), parent_utility),
        )
        replay_plan = EndpointReplayBuilder(vocabulary=vocabulary).build(
            parent,
            selected_sequences,
            context_id=config.context_id,
            advantages=gates.advantages,
            accepted=gates.accepted,
            timestep=config.replay_timestep,
            temperature=config.replay_temperature,
            clip=config.replay_clip,
            rng=np.random.default_rng(config.seed + 7),
        )
    else:
        gates = AdvantageGateResult(advantages=(), accepted=())
        replay_plan = EndpointReplayPlan(
            update_enabled=False,
            batch=None,
            rejected_child_indices=(),
        )

    observed_utilities = _utility(revealed, config.toxicity_upper)
    endpoints = (parent, *selected_sequences)
    replay_sequences = () if replay_plan.batch is None else replay_plan.batch.sequences
    replay_batches = plan_length_bucketed_batches(
        tuple(len(sequence) for sequence in replay_sequences),
        config.batching.replay_limits,
    )
    replay_microbatch_size = config.batching.replay_sequence_microbatch(padded_width=config.length)
    replay_compute_batches = plan_count_batches(len(replay_sequences), replay_microbatch_size)
    endpoint_utility = (parent_utility, *(float(value) for value in observed_utilities))
    winner: str | None = None
    loser: str | None = None
    preference_reason: str | None = None
    preference: DenoisingPreference | None = None
    parent_feasible = bool(
        kg_problem.constraints[0].is_satisfied(
            final_posterior.mean[parent_index, 1],
            final_posterior.covariance[parent_index, 1, parent_index, 1],
        )
    )
    if selected_points:
        comparable_positions = (
            *((0,) if parent_feasible else ()),
            *(1 + index for index, feasible in enumerate(chance_feasible) if feasible),
        )
        if len(comparable_positions) >= 2:
            winner_position = max(
                comparable_positions,
                key=lambda index: endpoint_utility[index],
            )
            loser_position = min(
                comparable_positions,
                key=lambda index: endpoint_utility[index],
            )
        else:
            winner_position = loser_position = 0
        if endpoint_utility[winner_position] > endpoint_utility[loser_position]:
            winner = endpoints[winner_position]
            loser = endpoints[loser_position]
            preference_reason = "measured_ordering_within_chance_feasible_endpoints"
            preference_mask = (
                tuple(bool(value) for value in replay_plan.batch.corruption_mask[0])
                if replay_plan.batch is not None
                else (True,) * config.length
            )
            preference = DenoisingPreference(
                winner=winner,
                loser=loser,
                context_id=config.context_id,
                corruption_mask=preference_mask,
                timestep=config.replay_timestep,
                weight=float(
                    np.clip(
                        abs(endpoint_utility[winner_position] - endpoint_utility[loser_position]),
                        0.1,
                        2.0,
                    )
                ),
            )

    updated_policy = policy
    local_kl = 0.0
    local_kl_p50 = 0.0
    local_kl_p95 = 0.0
    local_kl_p99 = 0.0
    local_kl_max = 0.0
    reference_path_kl = 0.0
    reference_transition_kl_p50 = 0.0
    reference_transition_kl_p95 = 0.0
    reference_transition_kl_p99 = 0.0
    reference_transition_kl_max = 0.0
    preference_margin_before: float | None = None
    preference_margin_after: float | None = None
    winner_log_probability_before: float | None = None
    winner_log_probability_after: float | None = None
    preferences = () if preference is None else (preference,)
    if winner is not None:
        winner_log_probability_before = policy.log_probability(winner, context_id=config.context_id)
    if replay_plan.batch is not None or preferences:
        policy_update = policy.update(
            context_id=config.context_id,
            replay_batch=replay_plan.batch,
            replay_microbatch_size=replay_microbatch_size,
            preferences=preferences,
            learning_rate=config.learning_rate,
            preference_beta=config.preference_beta,
            local_kl_limit=config.local_kl_limit,
            reference_path_kl_limit=config.reference_path_kl_limit,
            local_kl_p99_limit=config.local_kl_p99_limit,
            reference_transition_kl_p99_limit=(config.reference_transition_kl_p99_limit),
            preference_clip=config.preference_clip,
        )
        updated_policy = policy_update.policy
        local_kl = policy_update.local_kl_old_new
        local_kl_p50 = policy_update.local_transition_kl_summary.p50
        local_kl_p95 = policy_update.local_transition_kl_summary.p95
        local_kl_p99 = policy_update.local_transition_kl_summary.p99
        local_kl_max = policy_update.local_transition_kl_summary.maximum
        reference_path_kl = policy_update.reference_path_kl_new_reference
        reference_transition_kl_p50 = policy_update.reference_transition_kl_summary.p50
        reference_transition_kl_p95 = policy_update.reference_transition_kl_summary.p95
        reference_transition_kl_p99 = policy_update.reference_transition_kl_summary.p99
        reference_transition_kl_max = policy_update.reference_transition_kl_summary.maximum
        if policy_update.preference_margin_before:
            preference_margin_before = policy_update.preference_margin_before[0]
            preference_margin_after = policy_update.preference_margin_after[0]
    if winner is not None:
        winner_log_probability_after = updated_policy.log_probability(
            winner, context_id=config.context_id
        )

    real_decision_set_hash = _hash_bytes("\n".join(universe).encode())
    query_sources = (
        *("joint_soft_kg" for _ in kg_selected_points),
        *("uniform_reserve" for _ in reserve_selected_points),
    )
    regret_diagnostics = _post_hoc_regret_diagnostics(
        universe=universe,
        problem=kg_problem,
        query_indices=selected_points,
        query_sources=query_sources,
        before_query=before_query_recommendation,
        after_update=after_update_recommendation,
        decision_set_sha256=real_decision_set_hash,
        initial_design_count=len(SEED_SEQUENCES),
    )

    round_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "round": 0,
            "selected_branch_id": selected_branch.branch_id,
            "parent_sequence": parent,
            "raw_proposals": len(emissions),
            "unique_proposals": len(unique_emitted),
            "eligible_proposals": len(eligible_ids),
            "evaluated_children": len(selected_sequences),
            "kg_selected_children": len(kg_selected_points),
            "reserve_selected_children": len(reserve_selected_points),
            "accepted_children": len(gates.accepted_indices),
            "posterior_draw_id": draw_id,
            "context_id": config.context_id,
            "policy_version_before": policy.version,
            "policy_version_after": updated_policy.version,
            "local_kl": _float(local_kl),
            "local_kl_limit": _float(config.local_kl_limit),
            "local_kl_p99": _float(local_kl_p99),
            "local_kl_p99_limit": _float(config.local_kl_p99_limit),
            "reference_path_kl": _float(reference_path_kl),
            "reference_path_kl_limit": _float(config.reference_path_kl_limit),
            "reference_transition_kl_p99": _float(reference_transition_kl_p99),
            "reference_transition_kl_p99_limit": _float(config.reference_transition_kl_p99_limit),
            "winner": winner or "",
            "loser": loser or "",
            "winner_log_probability_before": _optional_float(winner_log_probability_before),
            "winner_log_probability_after": _optional_float(winner_log_probability_after),
            "preference_margin_before": _optional_float(preference_margin_before),
            "preference_margin_after": _optional_float(preference_margin_after),
        }
    ]
    _write_csv(
        destination / "proposals.csv",
        (
            "schema_version",
            "proposal_id",
            "kind",
            "sequence_id",
            "sequence",
            "parent_sequence",
            "branch_id",
            "rollout_id",
            "posterior_draw_id",
            "posterior_snapshot_id",
            "candidate_ordering_sha256",
            "output_ordering_sha256",
            "output_ids",
            "normalized_preference",
            "rollout_root_seed",
            "rollout_namespace",
            "rollout_global_index",
            "rollout_draw_seed",
            "rollout_rng_algorithm",
            "rollout_replay_implementation",
            "rollout_child_seed",
            "context_id",
            "policy_version",
            "hard_valid",
            "rejection_reason",
            "duplicate",
            "selected",
            "selection_source",
            "selection_set_id",
            "selection_probability",
            "selection_log_probability",
            "eligible_proposal_ids",
            "activity_mean",
            "toxicity_mean",
            "singleton_kg_estimate",
            "singleton_kg_standard_error",
            "singleton_kg_standard_error_penalized_estimate",
            "singleton_kg_score",
        ),
        proposal_rows,
    )
    _write_csv(
        destination / "edges.csv",
        (
            "schema_version",
            "edge_id",
            "proposal_id",
            "rollout_id",
            "parent_sequence_ids",
            "operator",
            "edit_description",
            "proposal_trace_factors",
            "proposal_probability",
            "proposal_log_probability",
            "behavior_log_probabilities",
            "random_stream",
            "sample_index",
            "sampling_parameters",
        ),
        edge_rows,
    )
    _write_csv(
        destination / "evaluations.csv",
        (
            "schema_version",
            "evaluation_id",
            "proposal_id",
            "sequence_id",
            "sequence",
            "fidelity",
            "evaluator_version",
            "activity",
            "toxicity",
            "cost",
            "batch_id",
            "replicate",
        ),
        evaluation_rows,
    )
    _write_csv(
        destination / "rounds.csv",
        tuple(round_rows[0]),
        round_rows,
    )
    rollout_batch.verify_against_posterior(posterior)
    rollout_manifest = _rollout_ledger_manifest(ledger, rollout_batch)
    (destination / "rollouts.json").write_text(
        json.dumps(rollout_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_hashes = {name: _hash_file(destination / name) for name in ARTIFACT_NAMES}
    artifact_set_hash = _hash_bytes(
        "".join(f"{name}:{artifact_hashes[name]}\n" for name in ARTIFACT_NAMES).encode()
    )
    replay_weights = (
        {
            sequence: float(weight)
            for sequence, weight in zip(
                replay_plan.batch.sequences,
                replay_plan.batch.weights,
                strict=True,
            )
        }
        if replay_plan.batch is not None
        else {}
    )
    decision_labels = (*universe, "__NO_ACTION__")
    decision_set_hash = _hash_bytes("\n".join(decision_labels).encode())
    realized_branch_improvement = _realized_feasible_improvement(
        parent_utility,
        parent_feasible,
        observed_utilities,
        chance_feasible,
        reference=config.branch_yield_reference,
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "counterfactual-soft-kg-search-smoke",
        "config_sha256": config_hash,
        "artifact_sha256": artifact_hashes,
        "artifact_set_sha256": artifact_set_hash,
        "counts": {
            "branches": len(branches),
            "seed_observations": len(SEED_SEQUENCES),
            "raw_proposals": len(emissions),
            "unique_proposals": len(unique_emitted),
            "eligible_proposals": len(eligible_ids),
            "evaluated_children": len(selected_sequences),
            "kg_selected_children": len(kg_selected_points),
            "reserve_selected_children": len(reserve_selected_points),
            "accepted_children": len(gates.accepted_indices),
            "ledger_proposals": len(ledger.proposals),
            "ledger_edges": len(ledger.edges),
            "ledger_evaluations": len(ledger.evaluations),
            "oracle_unique_calls": len(oracle.calls),
        },
        "representation": {
            "role": "additive_bayesian_linear_smoke_analogue",
            "sequence_block": "intercept_residue_counts_ordered_pair_counts",
            "sequence_block_dimensions": 1 + len(config.alphabet) + len(config.alphabet) ** 2,
            "spectral_block": "fixed_band_normalized_laplacian_log_spectral_density",
            "spectral_block_dimensions": SPECTRAL_CONFIG.bands,
            "spectral_prior_variance_multiplier": SPECTRAL_PRIOR_VARIANCE_MULTIPLIER,
            "length_covariates": 1,
            "combined_dimensions": (
                2 + len(config.alphabet) + len(config.alphabet) ** 2 + SPECTRAL_CONFIG.bands
            ),
            "contact_adapter": SPECTRAL_CONTACT_ADAPTER,
            "contact_adapter_is_biological_evidence": False,
            "production_contact_producer_required": True,
            "normalized_laplacian": True,
            "minimum_grid_eigenvalue": SPECTRAL_CONFIG.minimum_grid_eigenvalue,
            "maximum_grid_eigenvalue": SPECTRAL_CONFIG.maximum_grid_eigenvalue,
            "bandwidth_grid_steps": SPECTRAL_CONFIG.bandwidth_grid_steps,
            "contact_scale": SPECTRAL_CONFIG.contact_scale,
            "eigenvalue_floor": SPECTRAL_CONFIG.eigenvalue_floor,
            "zero_tolerance": SPECTRAL_CONFIG.zero_tolerance,
            "probability_tolerance": SPECTRAL_CONFIG.probability_tolerance,
            "output_dtype": "float64",
            "energy_grid_sha256": _hash_bytes(
                SPECTRAL_CONFIG.energy_grid.astype("<f8", copy=False).tobytes()
            ),
            "spectral_config_sha256": _hash_bytes(
                json.dumps(
                    asdict(SPECTRAL_CONFIG),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ),
            "global_grid": True,
            "l1_normalized": True,
        },
        "batching": {
            "profile": config.batching.profile,
            "configured": {
                "rollout_batch_size": config.batching.rollout_batch_size,
                "proposal_batch_size": config.batching.proposal_batch_size,
                "surrogate_batch_size": config.batching.surrogate_batch_size,
                "kg_candidate_chunk_size": config.batching.kg_candidate_chunk_size,
                "kg_fantasy_chunk_size": config.batching.kg_fantasy_chunk_size,
                "kg_max_joint_size": config.batching.kg_max_joint_size,
                "kg_max_combinations": config.batching.kg_max_combinations,
                "oracle_batch_size": config.batching.oracle_batch_size,
                "replay_max_sequences": config.batching.replay_limits.max_sequences,
                "replay_max_tokens": config.batching.replay_limits.max_tokens,
                "replay_length_bucket_boundaries": list(
                    config.batching.replay_limits.length_bucket_boundaries
                ),
                "gradient_accumulation_steps": (config.batching.gradient_accumulation_steps),
            },
            "realized": {
                "rollouts": len(rollout_batch.records),
                "proposal_candidates": len(emissions),
                "proposal_batch_sizes": proposal_batch_sizes,
                "surrogate_candidates": len(unique_indices),
                "kg_singleton_candidate_chunks": math.ceil(
                    len(unique_indices) / config.batching.kg_candidate_chunk_size
                ),
                "kg_joint_candidate_chunks": (
                    0
                    if joint_kg_result is None
                    else math.ceil(
                        len(joint_kg_result.evaluation_batches)
                        / config.batching.kg_candidate_chunk_size
                    )
                ),
                "kg_candidate_chunks_total": (
                    math.ceil(len(unique_indices) / config.batching.kg_candidate_chunk_size)
                    + (
                        0
                        if joint_kg_result is None
                        else math.ceil(
                            len(joint_kg_result.evaluation_batches)
                            / config.batching.kg_candidate_chunk_size
                        )
                    )
                ),
                "kg_fantasy_chunks_per_candidate_chunk": math.ceil(
                    config.kg_fantasies / config.batching.kg_fantasy_chunk_size
                ),
                "kg_fantasy_blocks_total": (
                    math.ceil(config.kg_fantasies / config.batching.kg_fantasy_chunk_size)
                    * (
                        math.ceil(len(unique_indices) / config.batching.kg_candidate_chunk_size)
                        + (
                            0
                            if joint_kg_result is None
                            else math.ceil(
                                len(joint_kg_result.evaluation_batches)
                                / config.batching.kg_candidate_chunk_size
                            )
                        )
                    )
                ),
                "joint_kg_size": joint_size,
                "joint_kg_screened_candidates": joint_pool_limit,
                "joint_kg_combinations": (
                    0 if joint_kg_result is None else len(joint_kg_result.evaluation_batches)
                ),
                "oracle_batch_sizes": [len(batch) for batch in oracle.batch_calls],
                "replay_batches": [
                    {
                        "indices": list(batch.indices),
                        "padded_width": batch.padded_width,
                        "token_count": batch.token_count,
                    }
                    for batch in replay_batches
                ],
                "replay_microbatch_size_ceiling": replay_microbatch_size,
                "replay_microbatch_sizes": [batch.size for batch in replay_compute_batches],
                "effective_replay_sequence_batch": (
                    config.batching.effective_replay_sequence_batch()
                ),
            },
        },
        "provenance": {
            "selected_branch_id": selected_branch.branch_id,
            "parent_sequence": parent,
            "parent_sequence_id": _hash_bytes(parent.encode("ascii")),
            "rollout_id": rollout.rollout_id,
            "posterior_draw_id": draw_id,
            "posterior_snapshot_id": rollout_batch.posterior_snapshot_id,
            "candidate_ids": list(rollout_batch.candidate_ids),
            "candidate_ordering_sha256": rollout_batch.candidate_ordering_sha256,
            "output_ids": list(rollout_batch.output_ids),
            "output_ordering_sha256": rollout_batch.output_ordering_sha256,
            "normalized_preference": rollout_batch.preference_weights[0].tolist(),
            "root_seed": str(rollout_batch.root_seed),
            "rollout_namespace": rollout_batch.rollout_namespace,
            "rollout_global_index": rollout.context_value("rollout_index"),
            "rollout_draw_seed": rollout.context_value("draw_seed"),
            "rollout_rng_algorithm": rollout.context_value("rng_algorithm"),
            "rollout_replay_implementation": rollout.context_value("replay_implementation"),
            "rollout_child_seed": str(rollout.seed),
            "rollout_manifest_sha256": artifact_hashes["rollouts.json"],
            "context_id": config.context_id,
            "policy_version": "policy-v0",
            "posterior_conditioned_observations": len(SEED_SEQUENCES),
            "fidelity": config.fidelity,
            "evaluator_version": config.evaluator_version,
            "progressive_widening_limit": widening_limit,
            "credit_domain_size": len(credit_domain),
            "credit_domain_sha256": credit_domain_hash,
        },
        "branch_allocation": {
            "yield_model": "toy-realized-1d-hv-with-pooled-between-branch-uncertainty",
            "yield_reference": config.branch_yield_reference,
            "visits": [branch.visits for branch in branches],
            "unique_evaluated_descendants": [
                branch.unique_evaluated_descendants for branch in branches
            ],
            "yield_means": [branch.yield_mean for branch in branches],
            "yield_standard_deviations": [branch.yield_std for branch in branches],
            "seed_chance_feasible": [bool(value) for value in seed_chance_feasible],
            "historical_influence_weights": [branch.credit for branch in branches],
            "ucb_scores": [float(score) for score in branch_scores],
            "posterior_optimum_samples": len(optimum_proxy),
            "branch_credit_draw_seed": str(config.seed + 1),
            "selected_branch_update": {
                "visits_before": selected_branch.visits,
                "visits_after": selected_branch.visits + int(bool(selected_points)),
                "unique_evaluated_descendants_before": (
                    selected_branch.unique_evaluated_descendants
                ),
                "unique_evaluated_descendants_after": (
                    selected_branch.unique_evaluated_descendants + len(selected_points)
                ),
                "realized_feasible_hypervolume_improvement": realized_branch_improvement,
            },
        },
        "soft_kg": {
            "decision_labels": list(decision_labels),
            "decision_set_sha256": decision_set_hash,
            "base_measure": [float(value) for value in kg_problem.base_measure],
            "log_base_measure": [float(value) for value in kg_problem.log_base_measure],
            "preference_weights": kg_problem.preferences.weights.tolist(),
            "preference_probabilities": kg_problem.preferences.probabilities.tolist(),
            "preference_log_probabilities": (kg_problem.preferences.log_probabilities.tolist()),
            "constraints": [
                {
                    "output_index": constraint.output_index,
                    "upper_bound": constraint.upper_bound,
                    "max_violation_probability": constraint.max_violation_probability,
                }
                for constraint in kg_problem.constraints
            ],
            "no_action_index": no_action_index,
            "always_safe_decision_indices": list(kg_problem.always_safe_decisions),
            "always_safe_decision_labels": [
                decision_labels[index] for index in kg_problem.always_safe_decisions
            ],
            "temperature": config.kg_temperature,
            "fantasies": config.kg_fantasies,
            "fantasy_seed": str(config.seed + 4),
            "standard_error_multiplier": config.kg_standard_error_multiplier,
            "selection_kind": "exact_joint_q_soft_kg",
            "joint_batch_size": joint_size,
            "selected_joint_batch": list(kg_selected_points),
            "joint_batches_scored": (
                0 if joint_kg_result is None else len(joint_kg_result.evaluation_batches)
            ),
            "screened_joint_candidate_indices": (
                [] if joint_evaluation_batch is None else list(joint_evaluation_batch.indices)
            ),
            "selected_joint_metrics": (
                None
                if joint_kg_result is None or not kg_selected_points
                else _joint_kg_metrics(
                    joint_kg_result,
                    joint_kg_result.evaluation_batches.index(kg_selected_points),
                    universe,
                )
            ),
            "joint_batch_evidence": (
                []
                if joint_kg_result is None
                else [
                    _joint_kg_metrics(joint_kg_result, position, universe)
                    for position in range(len(joint_kg_result.evaluation_batches))
                ]
            ),
            "candidates": [
                {
                    "sequence": universe[point],
                    "eligible": bool(kg_result.eligible[position]),
                    "singleton_estimate": float(kg_result.estimate[position]),
                    "singleton_standard_error": float(kg_result.standard_error[position]),
                    "singleton_standard_error_penalized_estimate": float(
                        kg_result.standard_error_penalized_estimate[position]
                    ),
                    "singleton_score": float(kg_result.score[position]),
                }
                for position, point in enumerate(unique_indices)
            ],
        },
        "terminal_recommendation": {
            "decision_rule": "hard_argmax_posterior_mean_with_no_action_abstention",
            "measurement_rule": "exact_joint_soft_kg_plus_uniform_reserve",
            "frozen_real_decision_set_sha256": real_decision_set_hash,
            "recommendable_decision_indices": list(real_decision_indices),
            "synthetic_no_action_excluded": True,
            "acquisition_score_used": False,
            "uncertainty_bonus_used": False,
            "decision_softmax_used": False,
            "variance_use": "chance_constraints_only",
            "preference_timing": "single_preference_timing_equivalent",
            "before_query": _recommendation_snapshot(
                before_query_recommendation,
                universe,
            ),
            "after_posterior_update": _recommendation_snapshot(
                after_update_recommendation,
                universe,
            ),
            "after_update_selected_was_queried": (
                after_update_recommendation.selected_decision_index in set(selected_points)
            ),
        },
        "regret_diagnostics": regret_diagnostics,
        "evaluation_accounting": {
            "expensive_oracle_unique_calls": len(oracle.calls),
            "ledger_evaluations": len(ledger.evaluations),
            "post_hoc_synthetic_truth_lookups": len(universe),
            "post_hoc_truth_entered_search": False,
        },
        "selection": {
            "policy_version": "joint-soft-kg-uniform-reserve-v2",
            "seed": str(config.seed + 5),
            "kg_slots": kg_slots,
            "uniform_reserve_slots": config.random_evaluation_reserve,
            "reserve_pool_proposal_ids": sorted(reserve_ids),
            "reserve_marginal_probability": reserve_probability,
        },
        "counterfactual_update": {
            "paired_draws": 96 if selected_points else 0,
            "paired_draw_seed": str(config.seed + 6) if selected_points else None,
            "endpoint_corruption_seed": str(config.seed + 7),
            "endpoint_corruption_seed_used": replay_plan.batch is not None,
            "risk_kappa": config.risk_kappa,
            "children": [
                {
                    "sequence": sequence,
                    "contrast_mean": contrast_means[index],
                    "contrast_variance": contrast_variances[index],
                    "absolute_utility_risk_score": absolute_risk_score[index],
                    "chance_feasible": chance_feasible[index],
                    "conservative_advantage": gates.advantages[index],
                    "accepted": gates.accepted[index],
                    "replay_weight": replay_weights.get(sequence, 0.0),
                }
                for index, sequence in enumerate(selected_sequences)
            ],
            "replay_corruption_mask": (
                replay_plan.batch.corruption_mask[0].tolist()
                if replay_plan.batch is not None
                else None
            ),
            "replay_effective_sample_size": (
                effective_sample_size(replay_plan.batch.weights)
                if replay_plan.batch is not None
                else 0.0
            ),
        },
        "policy": {
            "version_before": policy.version,
            "version_after": updated_policy.version,
            "local_kl": local_kl,
            "local_kl_limit": config.local_kl_limit,
            "local_transition_kl_p50": local_kl_p50,
            "local_transition_kl_p95": local_kl_p95,
            "local_transition_kl_p99": local_kl_p99,
            "local_transition_kl_max": local_kl_max,
            "local_transition_kl_p99_limit": config.local_kl_p99_limit,
            "reference_path_kl": reference_path_kl,
            "reference_path_kl_limit": config.reference_path_kl_limit,
            "reference_transition_kl_p50": reference_transition_kl_p50,
            "reference_transition_kl_p95": reference_transition_kl_p95,
            "reference_transition_kl_p99": reference_transition_kl_p99,
            "reference_transition_kl_max": reference_transition_kl_max,
            "reference_transition_kl_p99_limit": (config.reference_transition_kl_p99_limit),
            "winner": winner,
            "loser": loser,
            "winner_log_probability_before": winner_log_probability_before,
            "winner_log_probability_after": winner_log_probability_after,
            "preference_margin_before": preference_margin_before,
            "preference_margin_after": preference_margin_after,
            "preference_reason": preference_reason,
            "positive_replay_enabled": replay_plan.update_enabled,
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _rollout_context_manifest_value(value: object) -> object:
    """Serialize nested immutable provenance without unsafe JSON numbers."""

    if value is None or isinstance(value, str) or type(value) is bool:
        return value
    if isinstance(value, int):
        if not -(2**53 - 1) <= value <= 2**53 - 1:
            raise ValueError(
                "provenance integers must lie in the exactly representable JSON integer range"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance floats must be finite")
        return value
    if isinstance(value, tuple):
        return [_rollout_context_manifest_value(item) for item in value]
    raise TypeError("provenance contains a non-manifest value")


def _immutable_values_are_exactly_equal(left: object, right: object) -> bool:
    """Compare immutable provenance without Python's bool/int coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _immutable_values_are_exactly_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is float:
        return np.asarray([left], dtype="<f8").tobytes(order="C") == np.asarray(
            [right], dtype="<f8"
        ).tobytes(order="C")
    return bool(left == right)


def _rollout_records_are_exactly_equal(
    left: RolloutRecord,
    right: RolloutRecord,
) -> bool:
    return (
        left.rollout_id == right.rollout_id
        and left.branch_id == right.branch_id
        and left.posterior_draw_id == right.posterior_draw_id
        and _immutable_values_are_exactly_equal(left.context, right.context)
        and left.policy_version == right.policy_version
        and left.seed == right.seed
    )


def _rollout_ledger_manifest(
    ledger: SearchLedger,
    frozen_batch: FrozenRolloutBatch,
) -> dict[str, object]:
    """Serialize every rollout foreign-key target plus the replayable frozen batch."""

    frozen_by_id = {record.rollout_id: record for record in frozen_batch.records}
    ledger_by_id = {record.rollout_id: record for record in ledger.rollouts}
    missing = tuple(sorted(set(frozen_by_id).difference(ledger_by_id)))
    if missing:
        raise ValueError(f"frozen rollout records are absent from the ledger: {missing}")
    mismatched = tuple(
        sorted(
            rollout_id
            for rollout_id, frozen_record in frozen_by_id.items()
            if not _rollout_records_are_exactly_equal(
                ledger_by_id[rollout_id],
                frozen_record,
            )
        )
    )
    if mismatched:
        raise ValueError(
            f"frozen rollout records differ from their ledger foreign-key targets: {mismatched}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_thompson_batch": frozen_batch.to_manifest(),
        "ledger_rollouts": [
            {
                "schema_version": SCHEMA_VERSION,
                "kind": (
                    "frozen_thompson" if record.rollout_id in frozen_by_id else "observed_seed"
                ),
                "rollout_id": record.rollout_id,
                "branch_id": record.branch_id,
                "posterior_draw_id": record.posterior_draw_id,
                "context": [
                    [key, _rollout_context_manifest_value(value)] for key, value in record.context
                ],
                "policy_version": record.policy_version,
                "seed": str(record.seed),
            }
            for record in ledger.rollouts
        ],
    }


def _proposal_row(
    proposal: ProposalRecord,
    event: TranspositionEvent,
    kind: str,
    parent: str,
    singleton_kg_score: str,
    rollout: RolloutRecord,
    *,
    selection_source: str = "",
    singleton_kg_estimate: str = "",
    singleton_kg_standard_error: str = "",
    singleton_kg_standard_error_penalized_estimate: str = "",
) -> dict[str, object]:
    cheap_predictions = dict(proposal.cheap_predictions)

    def context_value(name: str) -> object:
        try:
            return rollout.context_value(name)
        except KeyError:
            return ""

    output_ids = context_value("output_ids")
    preference = context_value("preference")
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": proposal.proposal_id,
        "kind": kind,
        "sequence_id": proposal.sequence_key,
        "sequence": proposal.sequence,
        "parent_sequence": parent,
        "branch_id": rollout.branch_id,
        "rollout_id": rollout.rollout_id,
        "posterior_draw_id": rollout.posterior_draw_id,
        "posterior_snapshot_id": context_value("posterior_snapshot_id"),
        "candidate_ordering_sha256": context_value("candidate_ordering_sha256"),
        "output_ordering_sha256": context_value("output_ordering_sha256"),
        "output_ids": (
            ""
            if output_ids == ""
            else json.dumps(output_ids, separators=(",", ":"), allow_nan=False)
        ),
        "normalized_preference": (
            ""
            if preference == ""
            else json.dumps(preference, separators=(",", ":"), allow_nan=False)
        ),
        "rollout_root_seed": context_value("root_seed"),
        "rollout_namespace": context_value("rollout_namespace"),
        "rollout_global_index": context_value("rollout_index"),
        "rollout_draw_seed": context_value("draw_seed"),
        "rollout_rng_algorithm": context_value("rng_algorithm"),
        "rollout_replay_implementation": context_value("replay_implementation"),
        "rollout_child_seed": str(rollout.seed),
        "context_id": rollout.context_value("context_id"),
        "policy_version": proposal.policy_version,
        "hard_valid": str(proposal.hard_valid).lower(),
        "rejection_reason": proposal.rejection_reason or "",
        "duplicate": str(event.duplicate).lower(),
        "selected": str(proposal.selection.selected).lower(),
        "selection_source": selection_source,
        "selection_set_id": proposal.selection.selection_set_id,
        "selection_probability": _float(proposal.selection.propensity.probability),
        "selection_log_probability": _float(proposal.selection.propensity.log_probability),
        "eligible_proposal_ids": "|".join(proposal.selection.eligible_proposal_ids),
        "activity_mean": (
            _float(cheap_predictions["activity_mean"])
            if "activity_mean" in cheap_predictions
            else ""
        ),
        "toxicity_mean": (
            _float(cheap_predictions["toxicity_mean"])
            if "toxicity_mean" in cheap_predictions
            else ""
        ),
        "singleton_kg_estimate": singleton_kg_estimate,
        "singleton_kg_standard_error": singleton_kg_standard_error,
        "singleton_kg_standard_error_penalized_estimate": (
            singleton_kg_standard_error_penalized_estimate
        ),
        "singleton_kg_score": singleton_kg_score,
    }


def _joint_kg_metrics(
    result: JointSoftKGResult,
    position: int,
    universe: tuple[str, ...],
) -> dict[str, object]:
    indices = result.evaluation_batches[position]
    return {
        "evaluation_indices": list(indices),
        "sequences": [universe[index] for index in indices],
        "estimate": float(result.estimate[position]),
        "standard_error": float(result.standard_error[position]),
        "standard_error_penalized_estimate": float(
            result.standard_error_penalized_estimate[position]
        ),
        "score": float(result.score[position]),
        "total_cost": float(result.total_cost[position]),
    }


def _edge_sampling_parameters_manifest(
    edge: EdgeRecord,
    *,
    rollout: RolloutRecord,
) -> tuple[tuple[str, object], ...]:
    """Join replay provenance to its ledger rollout and make it JSON-safe."""

    if not isinstance(rollout, RolloutRecord):
        raise TypeError("rollout must be a RolloutRecord")
    if edge.rollout_id != rollout.rollout_id:
        raise ValueError("edge and supplied ledger rollout IDs must match")

    encoded: list[tuple[str, object]] = []
    has_rollout_identity = False
    for name, value in edge.sampling_parameters:
        if name == "rollout_identity":
            has_rollout_identity = True
            if type(value) is not tuple or len(value) != 6:
                raise ValueError("rollout_identity must contain the exact rollout record fields")
            if any(type(value[index]) is not str for index in (0, 1, 2, 4, 5)):
                raise TypeError(
                    "rollout_identity identifiers, policy, and child seed must be strings"
                )
            if any(
                not value[index] or value[index].strip() != value[index] for index in (0, 1, 2, 4)
            ):
                raise ValueError(
                    "rollout_identity identifiers and policy must be canonical strings"
                )
            if type(value[3]) is not tuple:
                raise TypeError("rollout_identity context must be an immutable tuple")
            context = value[3]
            if any(
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or not entry[0]
                or entry[0].strip() != entry[0]
                for entry in context
            ):
                raise TypeError(
                    "rollout_identity context must contain canonical string-keyed pairs"
                )
            if len({entry[0] for entry in context}) != len(context):
                raise ValueError("rollout_identity context keys must be unique")
            if value[0] != edge.rollout_id:
                raise ValueError("rollout_identity rollout ID must match the edge")
            child_seed = value[-1]
            if child_seed != "0" and (
                not child_seed
                or child_seed[0] == "0"
                or any(character not in "0123456789" for character in child_seed)
            ):
                raise ValueError("rollout_identity child seed must be a canonical decimal string")
        encoded.append((name, _rollout_context_manifest_value(value)))
    if edge.random_stream is not None and not has_rollout_identity:
        raise ValueError("every seeded edge requires rollout_identity")
    if edge.random_stream is not None:
        validate_seeded_emitter_edge_replay_contract(
            edge=edge,
            rollout=rollout,
        )
    return tuple(encoded)


def _edge_row(edge: EdgeRecord, *, rollout: RolloutRecord) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "edge_id": edge.edge_id,
        "proposal_id": edge.proposal_id,
        "rollout_id": edge.rollout_id,
        "parent_sequence_ids": "|".join(edge.parent_sequence_keys),
        "operator": edge.operator,
        "edit_description": json.dumps(
            _rollout_context_manifest_value(edge.edit_description),
            separators=(",", ":"),
            allow_nan=False,
        ),
        "proposal_trace_factors": json.dumps(
            tuple((factor.name, factor.probability) for factor in edge.proposal_trace.factors),
            separators=(",", ":"),
            allow_nan=False,
        ),
        "proposal_probability": _float(edge.proposal_trace.probability),
        "proposal_log_probability": _float(edge.proposal_trace.log_probability),
        "behavior_log_probabilities": "|".join(
            _float(value) for value in edge.behavior_log_probabilities
        ),
        "random_stream": "" if edge.random_stream is None else edge.random_stream,
        "sample_index": "" if edge.sample_index is None else edge.sample_index,
        "sampling_parameters": json.dumps(
            _edge_sampling_parameters_manifest(edge, rollout=rollout),
            separators=(",", ":"),
            allow_nan=False,
        ),
    }


def _evaluation_row(evaluation: EvaluationRecord, sequence: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": evaluation.evaluation_id,
        "proposal_id": evaluation.proposal_id,
        "sequence_id": _hash_bytes(sequence.encode("ascii")),
        "sequence": sequence,
        "fidelity": evaluation.fidelity,
        "evaluator_version": evaluation.evaluator_version,
        "activity": _float(evaluation.outcome("activity")),
        "toxicity": _float(evaluation.outcome("toxicity")),
        "cost": _float(evaluation.cost),
        "batch_id": evaluation.batch_id,
        "replicate": evaluation.replicate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(args.config, args.output_dir)
    concise = {
        "artifact_set_sha256": summary["artifact_set_sha256"],
        "counts": summary["counts"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(concise, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Executable native baseline operators behind the shared charged-history seam.

No oracle client, hidden-value scoring, reserve inventory or production switch.
The outer controller supplies already verified normalized observations and owns
the scientific interpretation, private reserves, full identity plan and clock.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    AnchorKLLimits,
    NativeEndpointConfig,
    _json_hash,
    _seed,
    _validate_model,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_initialization import (
    TRIPLES,
    AuditedNativeInitialization,
)
from amp_challenge.generators.diffusion.native_proposals import (
    NativeProposalTrace,
    replay_native_trace,
    sample_native_proposals,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    NativeWeightedReplay,
    WeightedAnchorDiagnostics,
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

MODES = (
    "categorical_diffusion_posthoc",
    "diffusion_reward_kl_no_search",
    "arcadiamp_style_iterative_d3pm",
)
BASELINE_CONFIG_SHA256 = "7c11d94602d7fef0186df0c0ae733f2a16263266db7b537c567045e1d53fce16"
REWARD_LIMITS = AnchorKLLimits(
    local_mean=0.01, local_p99=0.02, reference_mean=0.05, reference_p99=0.02
)
ATTEMPT_CAP = 65_536


def sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class NormalizedObjectiveContext:
    context_sha256: str
    objective_names: tuple[str, str] = ("gram_positive_activity", "gram_negative_activity")
    weights: tuple[float, float] = (0.5, 0.5)
    value_semantics: str = "controller_supplied_normalized_maximizing_scores_not_assay_units"

    def __post_init__(self) -> None:
        legacy = (
            self.objective_names == ("gram_positive_activity", "gram_negative_activity")
            and self.value_semantics
            == "controller_supplied_normalized_maximizing_scores_not_assay_units"
        )
        scalar_proxy = (
            self.objective_names == ("activity_proxy_replica_1", "activity_proxy_replica_2")
            and self.value_semantics
            == "identical_frozen_activity_proxy_scalar_replicas_not_independent_objectives"
        )
        if (
            not hash_string(self.context_sha256)
            or self.weights != (0.5, 0.5)
            or not (legacy or scalar_proxy)
        ):
            raise ValueError("native baseline normalized objective context differs")

    def scalarize(self, values: tuple[float, float]) -> float:
        if (
            type(values) is not tuple
            or len(values) != 2
            or any(
                type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
                for value in values
            )
        ):
            raise ValueError(
                "native baseline needs explicit finite normalized two-objective values"
            )
        return math.fsum(weight * value for weight, value in zip(self.weights, values, strict=True))


@dataclass(frozen=True, slots=True)
class NativeBaselineStep:
    replay: NativeWeightedReplay
    probes: NativeWeightedReplay
    model_before_sha256: str
    model_after_sha256: str
    gradient_sha256: str
    objective_before: float
    gradient_norm: float
    accepted: bool
    backtracks: int
    kl_enforced: bool
    diagnostics: WeightedAnchorDiagnostics


@dataclass(frozen=True, slots=True)
class NativeBaselineAdvance:
    triple: str
    round_index: int
    history_sha256: str
    history_receipt_sha256: str
    status: str
    steps: tuple[NativeBaselineStep, ...]
    old_policy_sha256: str
    new_policy_sha256: str
    frozen_reference_sha256: str
    configuration_sha256: str = BASELINE_CONFIG_SHA256
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


class _NativeUnit:
    def __init__(self, initialization: AuditedNativeInitialization, sequences: tuple[str, ...]):
        from amp_challenge.workflows.competition_train import TrainedNativeInitialization

        if (
            type(initialization) not in (AuditedNativeInitialization, TrainedNativeInitialization)
            or initialization.triple not in TRIPLES
        ):
            raise ValueError("native baseline requires an explicit audited initializer identity")
        sequences = tuple(sorted(sequences))
        if not 1 <= len(sequences) <= 1113 or any(
            type(seq) is not str
            or not 8 <= len(seq) <= 50
            or set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            for seq in sequences
        ):
            raise ValueError("native anchor corpus must contain bounded canonical peptides")
        if (
            len(set(sequences)) != len(sequences)
            or tuple(sorted(sequence_id(seq) for seq in sequences))
            != initialization.training_sequence_ids
        ):
            raise ValueError(
                "native baseline anchor sequences do not reconstruct initializer training IDs"
            )
        _validate_model(initialization.model, NATIVE_ENDPOINT_DEFAULTS)
        if (
            canonical_model_logical_hash(initialization.model)
            != initialization.checkpoint_logical_sha256
        ):
            raise ValueError("native initializer logical identity differs")
        self.triple = initialization.triple
        self.sequences = sequences
        self.model = copy.deepcopy(initialization.model).eval()
        self.reference = copy.deepcopy(initialization.model).eval()
        self.policy_sha256 = initialization.checkpoint_logical_sha256
        self.reference_sha256 = initialization.checkpoint_logical_sha256
        self.initialization = initialization

    def check(self) -> None:
        if (
            canonical_model_logical_hash(self.model) != self.policy_sha256
            or canonical_model_logical_hash(self.reference) != self.reference_sha256
        ):
            raise ValueError("native baseline policy/reference mutated outside operator")


def _weighted_rows(unit, history, context, mode, step):
    successful = [
        (row.sequence, context.scalarize(row.objectives))
        for row in history.observations
        if row.status == "successful"
    ]
    if mode == "arcadiamp_style_iterative_d3pm":
        successful = [(seq, value) for seq, value in successful if value >= 0.5]
    if not successful:
        return None

    def order(sequence, role):
        return _json_hash(
            [
                "native-baseline-replay-v1",
                history.seed,
                unit.triple,
                history.round_index,
                step,
                role,
                sequence,
            ]
        )

    anchors = tuple(sorted(unit.sequences, key=lambda seq: order(seq, "anchor"))[:64])
    selected = tuple(sorted(successful, key=lambda row: order(row[0], "revealed"))[:64])
    sequences = anchors + tuple(row[0] for row in selected)
    values = np.asarray([row[1] for row in selected], dtype=np.float64)
    if mode == "diffusion_reward_kl_no_search":
        weights = np.exp(np.clip((values - values.max()) / 0.1, -5.0, 0.0))
    else:
        weights = np.clip(values, 0.05, 1.0)
    weights = np.concatenate(
        (np.full(len(anchors), 0.5 / len(anchors)), 0.5 * weights / weights.sum())
    )
    return sequences, weights


def _advance_unit(unit, history, context, mode, config):
    unit.check()
    old_sha = unit.policy_sha256
    steps = []
    eligible = (mode == "diffusion_reward_kl_no_search" and history.round_index == 1) or (
        mode == "arcadiamp_style_iterative_d3pm" and 2 <= history.round_index <= 28
    )
    status = "unchanged_by_declared_schedule"
    working = unit.model
    if eligible:
        for step in range(1 if mode == "diffusion_reward_kl_no_search" else 4):
            rows = _weighted_rows(unit, history, context, mode, step)
            if rows is None:
                status = "no_supported_revealed_rows_no_update"
                break
            sequences, weights = rows
            ordinal = history.round_index * 4 + step
            # Triple is part of the semantic context/seed, never transport receipt identity.
            replay_seed = int(_json_hash([history.seed, unit.triple, "native-training"])[:16], 16)
            replay = build_weighted_replay(
                working,
                sequences,
                weights,
                context_id=context.context_sha256,
                seed=replay_seed,
                ordinal=ordinal,
            )
            probes = build_weighted_replay(
                unit.reference,
                sequences,
                weights,
                context_id=context.context_sha256,
                seed=replay_seed,
                ordinal=ordinal,
                active_probes=True,
            )
            direction = propose_weighted_direction(working, replay, config)
            gradient_hash = hashlib.sha256()
            for name, tensor in direction.gradients:
                gradient_hash.update(name.encode() + b"\0" + tensor.numpy().tobytes())
            enforce = mode == "diffusion_reward_kl_no_search"
            accepted = False
            for backtracks in range(config.maximum_backtracks + 1 if enforce else 1):
                candidate = endpoint_candidate(working, direction, backtracks=backtracks)
                diagnostics = weighted_anchor_diagnostics(
                    working, candidate, unit.reference, probes, config
                )
                local, frozen = (
                    diagnostics.local_old_candidate,
                    diagnostics.candidate_frozen_reference,
                )
                accepted = not enforce or (
                    local.mean <= REWARD_LIMITS.local_mean
                    and local.p99 <= REWARD_LIMITS.local_p99
                    and frozen.mean <= REWARD_LIMITS.reference_mean
                    and frozen.p99 <= REWARD_LIMITS.reference_p99
                )
                if accepted:
                    break
            after = (
                canonical_model_logical_hash(candidate) if accepted else direction.base_model_sha256
            )
            steps.append(
                NativeBaselineStep(
                    replay,
                    probes,
                    direction.base_model_sha256,
                    after,
                    gradient_hash.hexdigest(),
                    direction.objective_before,
                    direction.gradient_norm_before_clip,
                    accepted,
                    backtracks,
                    enforce,
                    diagnostics,
                )
            )
            if not accepted:
                status = "constrained_update_rejected_policy_frozen"
                break
            working = candidate
            status = "updated"
    new = copy.copy(unit)
    new.model = working
    new.policy_sha256 = canonical_model_logical_hash(working)
    unit.check()
    return new, NativeBaselineAdvance(
        unit.triple,
        history.round_index,
        history.sha256,
        history.receipt_sha256,
        status,
        tuple(steps),
        old_sha,
        new.policy_sha256,
        unit.reference_sha256,
    )


@dataclass(frozen=True, slots=True)
class NativeCandidateAttempt:
    attempt_index: int
    triple: str
    trace: NativeProposalTrace
    rejection_reason: str | None
    first_attempt_index: int


@dataclass(frozen=True, slots=True)
class NativeCandidatePool:
    mode: str
    round_index: int
    history_sha256: str
    policy_sha256: tuple[tuple[str, str], ...]
    attempts: tuple[NativeCandidateAttempt, ...]
    accepted_sequences: tuple[str, ...]
    target_size: int
    status: str
    sha256: str
    scope: str = "ordered_worker_pool_not_controller_448_identity_plan"
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


def _pool_digest(mode, round_index, history_sha256, policies, attempts, accepted, target, status):
    return _json_hash(
        {
            "mode": mode,
            "round": round_index,
            "history": history_sha256,
            "policies": policies,
            "config": BASELINE_CONFIG_SHA256,
            "attempts": [
                (
                    row.attempt_index,
                    row.triple,
                    _json_hash(asdict(row.trace)),
                    row.rejection_reason,
                    row.first_attempt_index,
                )
                for row in attempts
            ],
            "accepted": accepted,
            "target": target,
            "status": status,
        }
    )


def verify_native_pool(ensemble: NativeBaselineEnsemble, pool: NativeCandidatePool) -> None:
    """Reconstruct full-mask events/decisions using the independent trace scorer.

    This verifies mechanics against this exact history/policy snapshot; it does
    not authenticate an external oracle or confer production eligibility.
    """
    if type(ensemble) is not NativeBaselineEnsemble or type(pool) is not NativeCandidatePool:
        raise TypeError("native pool verification types differ")
    history = ensemble._history
    if (
        history is None
        or history.round_index == 29
        or (pool.mode, pool.round_index, pool.history_sha256, pool.policy_sha256)
        != (ensemble.mode, history.round_index, history.sha256, ensemble.policy_identities)
    ):
        raise ValueError("native pool history/policy binding differs")
    if (
        pool.scientific_evidence_accepted is not False
        or pool.production_eligible is not False
        or pool.scope != "ordered_worker_pool_not_controller_448_identity_plan"
    ):
        raise ValueError("native pool qualification differs")
    target = 256 if ensemble.mode == "arcadiamp_style_iterative_d3pm" else 2048
    if (
        type(pool.attempts) is not tuple
        or len(pool.attempts) > ATTEMPT_CAP
        or pool.target_size != target
    ):
        raise ValueError("native pool inventory/budget differs")
    for unit in ensemble._units:
        unit.check()
    offset = int(_json_hash([ensemble.seed, "native-equal-mixture-v1"])[:16], 16) % len(
        ensemble._units
    )
    charged = {sequence_id(row.sequence) for row in history.observations}
    accepted, seen = [], {}
    for index, attempt in enumerate(pool.attempts):
        if (
            type(attempt) is not NativeCandidateAttempt
            or attempt.attempt_index != index
            or len(accepted) == target
        ):
            raise ValueError("native attempt order/termination differs")
        unit = ensemble._units[(index + offset) % len(ensemble._units)]
        ordinal = (history.round_index - 1) * ATTEMPT_CAP + index
        parent = unit.sequences[
            int(
                _seed(ensemble.seed, ordinal, "native-length-" + unit.triple, 0).integers(
                    len(unit.sequences)
                )
            )
        ]
        trace = attempt.trace
        if attempt.triple != unit.triple or (
            trace.seed,
            trace.ordinal,
            trace.parent,
            trace.start_level,
        ) != (ensemble.seed, ordinal, parent, unit.model.config.levels):
            raise ValueError("native mixture/length/full-mask stream differs")
        replay_native_trace(unit.model, trace, config=ensemble.config, authenticate_sampling=True)
        key = sequence_id(trace.endpoint)
        reason = (
            "generator_training_overlap"
            if key in ensemble._training_ids
            else "previously_charged"
            if key in charged
            else "generated_duplicate"
            if key in seen
            else None
        )
        first = seen.setdefault(key, index)
        if (attempt.rejection_reason, attempt.first_attempt_index) != (reason, first):
            raise ValueError("native independently reconstructed exclusion differs")
        if reason is None:
            accepted.append(trace.endpoint)
    status = (
        "complete"
        if len(accepted) == target
        else "attempt_cap_exhausted"
        if len(pool.attempts) == ATTEMPT_CAP
        else "in_progress"
    )
    if (
        pool.accepted_sequences != tuple(accepted)
        or pool.status != status
        or pool.sha256
        != _pool_digest(
            pool.mode,
            pool.round_index,
            pool.history_sha256,
            pool.policy_sha256,
            pool.attempts,
            accepted,
            target,
            status,
        )
    ):
        raise ValueError("native pool retained order/status/seal differs")


class NativeBaselineEnsemble:
    """Ten-model campaign coordinator; smaller cardinalities are mechanics-only.

    Advance is transactional across models. It accepts the shared history type,
    not an oracle callback. The outer history reader already binds its provider;
    this operator also checks fixed identity, prefix, expected head and counts.
    """

    def __init__(
        self,
        mode: str,
        initializations: tuple[AuditedNativeInitialization, ...],
        generator_sequences: tuple[tuple[str, ...], ...],
        context: NormalizedObjectiveContext,
        *,
        run_id: str,
        seed: int,
        oracle_bundle_sha256: str,
        config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
    ):
        if mode not in MODES or type(context) is not NormalizedObjectiveContext:
            raise ValueError("native baseline mode/context differs")
        context.__post_init__()
        if type(config) is not NativeEndpointConfig or config != NATIVE_ENDPOINT_DEFAULTS:
            raise ValueError("native baseline configuration differs from predeclared defaults")
        if not 1 <= len(initializations) <= 10 or len(initializations) != len(generator_sequences):
            raise ValueError("native initializer/anchor inventory differs")
        units = tuple(
            _NativeUnit(init, seqs)
            for init, seqs in zip(initializations, generator_sequences, strict=True)
        )
        if (
            len({unit.triple for unit in units}) != len(units)
            or len({unit.model.config for unit in units}) != 1
        ):
            raise ValueError("native mixture requires unique compatible initializer triples")
        if type(seed) is not int or not 0 <= seed < 2**63 or not hash_string(oracle_bundle_sha256):
            raise ValueError("native baseline seed/oracle source differs")
        self.mode, self.context, self.run_id, self.seed = mode, context, run_id, seed
        self.oracle_bundle_sha256, self.config = oracle_bundle_sha256, config
        self._units = tuple(sorted(units, key=lambda unit: unit.triple))
        self.protocol_ten_checkpoint_mixture = tuple(unit.triple for unit in self._units) == TRIPLES
        self._history = None
        self._pool = None
        self._receipts = ()
        self._training_ids = frozenset(sequence_id(seq) for unit in units for seq in unit.sequences)

    @property
    def policy_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple((unit.triple, unit.policy_sha256) for unit in self._units)

    def advance(
        self, history: VerifiedHistorySnapshot, *, expected_previous_head_sha256: str
    ) -> tuple[NativeBaselineAdvance, ...]:
        if type(history) is not VerifiedHistorySnapshot:
            raise TypeError("native baseline requires the shared verified-history handoff")
        history.__post_init__()
        if (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
            history.previous_wave_head_sha256,
        ) != (
            self.run_id,
            self.seed,
            self.context.context_sha256,
            self.oracle_bundle_sha256,
            expected_previous_head_sha256,
        ):
            raise ValueError("native baseline history run/context/source/head differs")
        if any(sequence_id(row.sequence) in self._training_ids for row in history.observations):
            raise ValueError("native charged history overlaps generator training")
        if self._history is not None and history.sha256 == self._history.sha256:
            return self._receipts
        expected_round = 1 if self._history is None else self._history.round_index + 1
        if history.round_index != expected_round:
            raise ValueError("native baseline history skipped/repeated a wave")
        if self._history is not None:
            if self.mode != "arcadiamp_style_iterative_d3pm":
                raise ValueError("frozen native arm may not consume adaptive responses")
            if (
                history.observations[: len(self._history.observations)]
                != self._history.observations
            ):
                raise ValueError("native baseline history rewrites previously charged observations")
            if self._pool is None or self._pool.status != "complete":
                raise ValueError("native next-wave history arrived before completed candidate pool")
        if not history.complete:
            return ()  # paused handoff; no policy, history or pool mutation
        replacements = tuple(
            _advance_unit(unit, history, self.context, self.mode, self.config)
            for unit in self._units
        )
        # No model/state is committed until every checkpoint succeeds.
        self._units = tuple(pair[0] for pair in replacements)
        self._receipts = tuple(pair[1] for pair in replacements)
        self._history, self._pool = history, None
        return self._receipts

    def propose(self, *, max_new_attempts: int = 128) -> NativeCandidatePool:
        if self._history is None or self._history.round_index == 29:
            raise ValueError("native baseline needs a completed nonterminal history")
        if type(max_new_attempts) is not int or not 0 <= max_new_attempts <= ATTEMPT_CAP:
            raise ValueError("native proposal attempt budget differs")
        for unit in self._units:
            unit.check()
        if self._pool is not None and self._pool.status != "in_progress":
            return self._pool
        target = 256 if self.mode == "arcadiamp_style_iterative_d3pm" else 2048
        attempts = list(self._pool.attempts) if self._pool else []
        accepted = list(self._pool.accepted_sequences) if self._pool else []
        seen = {sequence_id(row.trace.endpoint): row.first_attempt_index for row in attempts}
        charged = {sequence_id(row.sequence) for row in self._history.observations}
        offset = int(_json_hash([self.seed, "native-equal-mixture-v1"])[:16], 16) % len(self._units)
        stop = min(ATTEMPT_CAP, len(attempts) + max_new_attempts)
        while len(attempts) < stop and len(accepted) < target:
            begin = len(attempts)
            # Never execute then drop unledgered tail proposals at pool completion.
            indices = tuple(range(begin, min(begin + 128, stop, begin + target - len(accepted))))
            traces = {}
            for unit_index, unit in enumerate(self._units):
                group = tuple(
                    index for index in indices if (index + offset) % len(self._units) == unit_index
                )
                if not group:
                    continue
                ordinals = tuple(
                    (self._history.round_index - 1) * ATTEMPT_CAP + index for index in group
                )
                parents = tuple(
                    unit.sequences[
                        int(
                            _seed(self.seed, ordinal, "native-length-" + unit.triple, 0).integers(
                                len(unit.sequences)
                            )
                        )
                    ]
                    for ordinal in ordinals
                )
                generated = sample_native_proposals(
                    unit.model,
                    parents,
                    start_levels=(unit.model.config.levels,) * len(group),
                    seed=self.seed,
                    ordinals=ordinals,
                    config=self.config,
                )
                traces.update(
                    (index, (unit.triple, trace))
                    for index, trace in zip(group, generated, strict=True)
                )
            for index in indices:
                triple, trace = traces[index]
                key = sequence_id(trace.endpoint)
                reason = (
                    "generator_training_overlap"
                    if key in self._training_ids
                    else "previously_charged"
                    if key in charged
                    else "generated_duplicate"
                    if key in seen
                    else None
                )
                first = seen.setdefault(key, index)
                attempts.append(NativeCandidateAttempt(index, triple, trace, reason, first))
                if reason is None:
                    accepted.append(trace.endpoint)
        status = (
            "complete"
            if len(accepted) == target
            else "attempt_cap_exhausted"
            if len(attempts) == ATTEMPT_CAP
            else "in_progress"
        )
        pool = NativeCandidatePool(
            self.mode,
            self._history.round_index,
            self._history.sha256,
            self.policy_identities,
            tuple(attempts),
            tuple(accepted),
            target,
            status,
            _pool_digest(
                self.mode,
                self._history.round_index,
                self._history.sha256,
                self.policy_identities,
                attempts,
                accepted,
                target,
                status,
            ),
        )
        for unit in self._units:
            unit.check()
        self._pool = pool
        return pool

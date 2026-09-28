"""Deterministic mixed acquisition for AMP internal experiment batches.

The selector is deliberately model-agnostic.  Inputs are calibrated oracle
means/uncertainties in higher-is-better orientation plus optional novelty,
embedding, cluster, start-lineage, and award-specialist information.  It records
why every candidate entered the batch, making acquisition-policy ablations
auditable.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SUPPORTED_STRATEGIES = (
    "exploit",
    "pareto",
    "diversity",
    "novelty",
    "uncertainty",
    "start_ucb",
    "ucb",
    "random",
)
SUPPORTED_UNCERTAINTY_MODES = ("required", "unavailable")


def _default_strategy_mix() -> dict[str, float]:
    return {
        "exploit": 0.35,
        "pareto": 0.20,
        "diversity": 0.15,
        "novelty": 0.10,
        "uncertainty": 0.10,
        "ucb": 0.05,
        "random": 0.05,
    }


@dataclass(frozen=True)
class CandidateBatch:
    """Candidate values passed from an oracle ensemble to acquisition.

    ``objective_std`` remains shape-stable for downstream array operations.  If
    uncertainty is unavailable it is an all-zero sentinel, explicitly marked
    by ``uncertainty_available=False``, rather than uncertainty evidence.
    """

    sequences: Sequence[str]
    objective_mean: FloatArray
    objective_std: FloatArray
    novelty: FloatArray | None = None
    embeddings: FloatArray | None = None
    cluster_ids: Sequence[str] | None = None
    specialist_scores: Mapping[str, FloatArray] = field(default_factory=dict)
    eligible: BoolArray | None = None
    start_ids: Sequence[str] | None = None
    rollout_ids: Sequence[str] | None = None
    uncertainty_available: bool = True

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        mean = np.asarray(self.objective_mean, dtype=np.float64)
        std = np.asarray(self.objective_std, dtype=np.float64)
        if not sequences:
            raise ValueError("candidate batch cannot be empty")
        if len(set(sequences)) != len(sequences):
            raise ValueError("candidate sequences must be unique")
        if mean.ndim != 2 or mean.shape[0] != len(sequences):
            raise ValueError("objective_mean must have shape (n_candidates, n_objectives)")
        if mean.shape[1] == 0 or std.shape != mean.shape:
            raise ValueError("objective_std must match a non-empty objective_mean matrix")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std < 0):
            raise ValueError("objective means/stds must be finite and stds non-negative")
        if not isinstance(self.uncertainty_available, bool):
            raise ValueError("uncertainty_available must be a boolean")
        if not self.uncertainty_available and np.any(std != 0):
            raise ValueError("objective_std must be all zero when uncertainty_available is false")

        novelty = None if self.novelty is None else np.asarray(self.novelty, dtype=np.float64)
        if novelty is not None and (
            novelty.shape != (len(sequences),) or np.any(~np.isfinite(novelty))
        ):
            raise ValueError("novelty must be a finite vector with one value per candidate")

        embeddings = (
            None if self.embeddings is None else np.asarray(self.embeddings, dtype=np.float64)
        )
        if embeddings is not None and (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(sequences)
            or embeddings.shape[1] == 0
            or np.any(~np.isfinite(embeddings))
        ):
            raise ValueError("embeddings must be a finite (n_candidates, n_features) matrix")

        clusters = None if self.cluster_ids is None else tuple(map(str, self.cluster_ids))
        if clusters is not None and len(clusters) != len(sequences):
            raise ValueError("cluster_ids must have one value per candidate")

        starts: tuple[str, ...] | None = None
        if self.start_ids is not None:
            raw_starts = tuple(self.start_ids)
            if any(not isinstance(value, str) for value in raw_starts):
                raise ValueError("start_ids must be strings")
            starts = tuple(value.strip() for value in raw_starts)
            if len(starts) != len(sequences):
                raise ValueError("start_ids must have one value per candidate")
            if any(not value for value in starts):
                raise ValueError("start_ids must be non-empty")

        rollouts: tuple[str, ...] | None = None
        if self.rollout_ids is not None:
            if starts is None:
                raise ValueError("rollout_ids require start_ids")
            raw_rollouts = tuple(self.rollout_ids)
            if any(not isinstance(value, str) for value in raw_rollouts):
                raise ValueError("rollout_ids must be strings")
            rollouts = tuple(value.strip() for value in raw_rollouts)
            if len(rollouts) != len(sequences):
                raise ValueError("rollout_ids must have one value per candidate")
            if any(not value for value in rollouts):
                raise ValueError("rollout_ids must be non-empty")
            pairs = tuple(zip(starts, rollouts, strict=True))
            if len(set(pairs)) != len(pairs):
                raise ValueError("(start_id, rollout_id) pairs must be unique")

        specialists: dict[str, FloatArray] = {}
        for name, values in self.specialist_scores.items():
            vector = np.asarray(values, dtype=np.float64)
            if vector.shape != (len(sequences),) or np.any(~np.isfinite(vector)):
                raise ValueError(f"specialist score {name!r} must be a finite candidate vector")
            specialists[name] = vector

        eligible = (
            np.ones(len(sequences), dtype=bool)
            if self.eligible is None
            else np.asarray(self.eligible, dtype=bool)
        )
        if eligible.shape != (len(sequences),):
            raise ValueError("eligible must have one value per candidate")

        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "objective_mean", mean)
        object.__setattr__(self, "objective_std", std)
        object.__setattr__(self, "novelty", novelty)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "cluster_ids", clusters)
        object.__setattr__(self, "start_ids", starts)
        object.__setattr__(self, "rollout_ids", rollouts)
        object.__setattr__(self, "specialist_scores", specialists)
        object.__setattr__(self, "eligible", eligible)


@dataclass(frozen=True)
class SelectionConfig:
    """Policy for an expensive-label acquisition batch.

    The default ``required`` mode preserves the legacy calibrated-uncertainty
    policy.  ``unavailable`` is an explicit mean-only contract.
    """

    batch_size: int
    strategy_mix: Mapping[str, float] = field(default_factory=_default_strategy_mix)
    objective_weights: Sequence[float] | None = None
    specialist_quotas: Mapping[str, int] = field(default_factory=dict)
    risk_beta: float = 1.0
    ucb_beta: float = 1.0
    diversity_quality_weight: float = 0.35
    quality_floor_quantile: float = 0.20
    max_per_cluster: int | None = None
    strict_cluster_cap: bool = True
    seed: int = 42
    max_per_start: int | None = None
    rollouts_per_start: int | None = None
    uncertainty_mode: str = "required"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        unknown = set(self.strategy_mix) - set(SUPPORTED_STRATEGIES)
        if unknown:
            raise ValueError(f"unsupported acquisition strategies: {sorted(unknown)}")
        if any((not np.isfinite(value) or value < 0) for value in self.strategy_mix.values()):
            raise ValueError("strategy weights must be finite and non-negative")
        if sum(self.strategy_mix.values()) <= 0:
            raise ValueError("at least one strategy weight must be positive")
        if any(quota < 0 for quota in self.specialist_quotas.values()):
            raise ValueError("specialist quotas must be non-negative")
        if sum(self.specialist_quotas.values()) > self.batch_size:
            raise ValueError("specialist quotas cannot exceed batch_size")
        if self.risk_beta < 0 or self.ucb_beta < 0:
            raise ValueError("risk and UCB coefficients must be non-negative")
        if not 0 <= self.diversity_quality_weight <= 1:
            raise ValueError("diversity_quality_weight must be in [0, 1]")
        if not 0 <= self.quality_floor_quantile < 1:
            raise ValueError("quality_floor_quantile must be in [0, 1)")
        if self.max_per_cluster is not None and self.max_per_cluster <= 0:
            raise ValueError("max_per_cluster must be positive when provided")
        if self.max_per_start is not None and (
            isinstance(self.max_per_start, bool)
            or not isinstance(self.max_per_start, int)
            or self.max_per_start <= 0
        ):
            raise ValueError("max_per_start must be a positive integer when provided")
        if self.rollouts_per_start is not None and (
            isinstance(self.rollouts_per_start, bool)
            or not isinstance(self.rollouts_per_start, int)
            or self.rollouts_per_start < 2
        ):
            raise ValueError("rollouts_per_start must be an integer at least 2 when provided")
        if self.strategy_mix.get("start_ucb", 0) > 0 and self.rollouts_per_start is None:
            raise ValueError("start_ucb strategy requires rollouts_per_start >= 2")
        if self.rollouts_per_start is not None and self.max_per_start != 1:
            raise ValueError("start-aware selection requires max_per_start = 1")
        if self.uncertainty_mode not in SUPPORTED_UNCERTAINTY_MODES:
            raise ValueError(
                f"uncertainty_mode must be exactly one of {list(SUPPORTED_UNCERTAINTY_MODES)}"
            )
        if self.uncertainty_mode == "unavailable":
            if self.risk_beta != 0 or self.ucb_beta != 0:
                raise ValueError(
                    "uncertainty_mode='unavailable' requires risk_beta=0 and ucb_beta=0"
                )
            forbidden = {
                strategy
                for strategy in ("uncertainty", "ucb", "start_ucb")
                if self.strategy_mix.get(strategy, 0) > 0
            }
            if forbidden:
                raise ValueError(
                    "uncertainty_mode='unavailable' forbids positive uncertainty-dependent "
                    f"strategy weights: {sorted(forbidden)}"
                )
            if self.rollouts_per_start is not None:
                raise ValueError("uncertainty_mode='unavailable' forbids start-aware rollouts")


@dataclass(frozen=True)
class StartSelectionEvidence:
    """Two-stage provenance aligned to one selected start-aware candidate."""

    start_id: str
    rollout_id: str
    start_rank: int
    eligible_rollout_count: int
    rollout_value_mean: float
    rollout_value_dispersion: float
    rollout_ucb_score: float


@dataclass(frozen=True)
class SelectionResult:
    """Selected candidate indices, final ranks, and acquisition provenance."""

    indices: tuple[int, ...]
    reasons: tuple[str, ...]
    conservative_scores: tuple[float, ...]
    acquisition_scores: tuple[float, ...]
    cluster_counts: Mapping[str, int]
    strategy_counts: Mapping[str, int]
    start_evidence: tuple[StartSelectionEvidence, ...] = ()

    def rows(self, candidates: CandidateBatch) -> list[dict[str, object]]:
        """Return JSON/CSV-friendly audit rows in final rank order."""

        if self.start_evidence and len(self.start_evidence) != len(self.indices):
            raise ValueError("start_evidence must align with selected indices")
        rows: list[dict[str, object]] = []
        for position, (index, reason, score, acquisition_score) in enumerate(
            zip(
                self.indices,
                self.reasons,
                self.conservative_scores,
                self.acquisition_scores,
                strict=True,
            )
        ):
            row: dict[str, object] = {
                "rank": position + 1,
                "candidate_index": index,
                "sequence": candidates.sequences[index],
                "acquisition_reason": reason,
                "conservative_score": score,
                "acquisition_score": acquisition_score,
                "cluster_id": (
                    None if candidates.cluster_ids is None else candidates.cluster_ids[index]
                ),
            }
            if self.start_evidence:
                evidence = self.start_evidence[position]
                row.update(
                    {
                        "start_id": evidence.start_id,
                        "rollout_id": evidence.rollout_id,
                        "start_rank": evidence.start_rank,
                        "start_eligible_rollout_count": evidence.eligible_rollout_count,
                        "start_rollout_value_mean": evidence.rollout_value_mean,
                        "start_rollout_value_dispersion": evidence.rollout_value_dispersion,
                        "rollout_ucb_score": evidence.rollout_ucb_score,
                    }
                )
            rows.append(row)
        return rows


class MixedAcquisitionSelector:
    """Allocate a batch across specialists and complementary strategies."""

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config

    def select(self, candidates: CandidateBatch) -> SelectionResult:
        n_candidates, n_objectives = candidates.objective_mean.shape
        expected_uncertainty = self.config.uncertainty_mode == "required"
        if candidates.uncertainty_available != expected_uncertainty:
            raise ValueError(
                "candidate uncertainty availability does not match selection uncertainty_mode"
            )
        if self.config.uncertainty_mode == "unavailable":
            if np.any(candidates.objective_std != 0):
                raise ValueError(
                    "uncertainty-unavailable candidates cannot carry objective_std evidence"
                )
            if self.config.risk_beta != 0 or self.config.ucb_beta != 0:
                raise ValueError(
                    "uncertainty_mode='unavailable' requires risk_beta=0 and ucb_beta=0"
                )
            forbidden = {
                strategy
                for strategy in ("uncertainty", "ucb", "start_ucb")
                if self.config.strategy_mix.get(strategy, 0) > 0
            }
            if forbidden:
                raise ValueError(
                    "uncertainty_mode='unavailable' forbids positive uncertainty-dependent "
                    f"strategy weights: {sorted(forbidden)}"
                )
            if self.config.rollouts_per_start is not None:
                raise ValueError("uncertainty_mode='unavailable' forbids start-aware rollouts")
        if np.count_nonzero(candidates.eligible) < self.config.batch_size:
            raise ValueError("not enough eligible candidates to fill the requested batch")

        objective_weights = self._objective_weights(n_objectives)
        start_aware = self.config.rollouts_per_start is not None
        mean_rank = (
            _column_rank_scale_eligible(candidates.objective_mean, candidates.eligible)
            if start_aware
            else _column_rank_scale(candidates.objective_mean)
        )
        if not candidates.uncertainty_available:
            std_rank = np.zeros_like(mean_rank)
        elif start_aware:
            std_rank = _column_rank_scale_eligible(
                candidates.objective_std,
                candidates.eligible,
            )
        else:
            # Preserve the legacy no-start policy byte-for-byte: historically,
            # every ledger row participated in percentile scaling.
            std_rank = _column_rank_scale(candidates.objective_std)
        conservative = (mean_rank - self.config.risk_beta * std_rank) @ objective_weights
        ucb = (mean_rank + self.config.ucb_beta * std_rank) @ objective_weights
        uncertainty = (
            std_rank @ objective_weights + 0.15 * conservative
            if candidates.uncertainty_available
            else np.zeros(n_candidates, dtype=np.float64)
        )
        novelty = (
            None
            if candidates.novelty is None
            else (
                _rank_scale_eligible(candidates.novelty, candidates.eligible)
                if start_aware
                else _rank_scale(candidates.novelty)
            )
            + 0.25 * conservative
        )

        requested = {name for name, value in self.config.strategy_mix.items() if value > 0}
        if "novelty" in requested and novelty is None:
            raise ValueError("novelty strategy requested but no novelty values were supplied")
        if "diversity" in requested and candidates.embeddings is None:
            raise ValueError("diversity strategy requested but no embeddings were supplied")
        if start_aware and candidates.start_ids is None:
            raise ValueError("start-aware selection requested but no start_ids were supplied")
        if start_aware and candidates.rollout_ids is None:
            raise ValueError("start-aware selection requested but no rollout_ids were supplied")
        if self.config.max_per_start is not None and candidates.start_ids is None:
            raise ValueError("max_per_start requested but no start_ids were supplied")
        if start_aware:
            assert candidates.start_ids is not None
            assert self.config.rollouts_per_start is not None
            eligible_per_start = Counter(
                start_id
                for start_id, is_eligible in zip(
                    candidates.start_ids,
                    candidates.eligible,
                    strict=True,
                )
                if is_eligible
            )
            observed_starts = set(candidates.start_ids)
            invalid_counts = {
                start_id: eligible_per_start[start_id]
                for start_id in observed_starts
                if eligible_per_start[start_id] != self.config.rollouts_per_start
            }
            if invalid_counts:
                details = ", ".join(
                    f"{start_id}={invalid_counts[start_id]}" for start_id in sorted(invalid_counts)
                )
                raise ValueError(
                    "start-aware selection requires exactly "
                    f"{self.config.rollouts_per_start} eligible rollouts per start; "
                    f"observed {details}"
                )
            rollout_value = mean_rank @ objective_weights
            start_value_summary = _start_value_summary(
                rollout_value,
                candidates.start_ids,
                candidates.eligible,
            )
            start_order = sorted(
                start_value_summary,
                key=lambda value: (-start_value_summary[value][1], value),
            )
            start_ranks = {start_id: rank for rank, start_id in enumerate(start_order, start=1)}
        else:
            start_value_summary = {}
            start_order = []
            start_ranks = {}
        missing_specialists = set(self.config.specialist_quotas) - set(candidates.specialist_scores)
        if missing_specialists:
            raise ValueError(f"missing specialist scores: {sorted(missing_specialists)}")

        rng = np.random.default_rng(self.config.seed)
        random_scores = (
            _keyed_random_scores(candidates.sequences, self.config.seed)
            if start_aware
            else rng.random(n_candidates)
        )
        selected: list[int] = []
        reasons: list[str] = []
        acquisition_scores: list[float] = []
        cluster_counts: Counter[str] = Counter()
        start_counts: Counter[str] = Counter()

        if self.config.max_per_start is not None:
            assert candidates.start_ids is not None
            eligible_start_counts = Counter(
                start_id
                for start_id, is_eligible in zip(
                    candidates.start_ids,
                    candidates.eligible,
                    strict=True,
                )
                if is_eligible
            )
            capacity = sum(
                min(count, self.config.max_per_start) for count in eligible_start_counts.values()
            )
            if capacity < self.config.batch_size:
                raise ValueError(
                    "not enough start-group capacity to fill the requested batch under "
                    "max_per_start"
                )

        quality_threshold = float(
            np.quantile(conservative[candidates.eligible], self.config.quality_floor_quantile)
        )

        remaining = self.config.batch_size - sum(self.config.specialist_quotas.values())
        quotas = _allocate_quotas(self.config.strategy_mix, remaining)
        pending = dict(quotas)

        # Specialist seats guarantee coverage of the five award profiles (or
        # any future assay-specific strata) before general-purpose acquisition.
        for name, quota in self.config.specialist_quotas.items():
            specialist_rank = (
                _rank_scale_eligible(candidates.specialist_scores[name], candidates.eligible)
                if start_aware
                else _rank_scale(candidates.specialist_scores[name])
            )
            scores = specialist_rank + 0.25 * conservative
            for _ in range(quota):
                index = self._choose(
                    scores,
                    candidates,
                    selected,
                    cluster_counts,
                    start_counts,
                    quality_floor=None,
                )
                self._append(
                    index,
                    f"specialist:{name}",
                    scores[index],
                    candidates,
                    selected,
                    reasons,
                    acquisition_scores,
                    cluster_counts,
                    start_counts,
                )

        # This grouped arm mirrors a start-conditioned generator loop: rank
        # distinct starts by the population spread of their eligible rollout
        # values, then take the highest reward-UCB feasible rollout from each
        # chosen start. It is completed as one block so another general arm
        # cannot consume a rollout between the two stages. Specialist seats are
        # reserved first and therefore consume their start under max_per_start.
        start_quota = pending.pop("start_ucb", 0)
        if start_quota:
            assert candidates.start_ids is not None
            filled = 0
            for start_id in start_order:
                member_mask = np.asarray(
                    [value == start_id for value in candidates.start_ids],
                    dtype=bool,
                )
                try:
                    index = self._choose(
                        ucb,
                        candidates,
                        selected,
                        cluster_counts,
                        start_counts,
                        quality_floor=None,
                        candidate_mask=member_mask,
                    )
                except ValueError:
                    continue
                self._append(
                    index,
                    "start_ucb",
                    ucb[index],
                    candidates,
                    selected,
                    reasons,
                    acquisition_scores,
                    cluster_counts,
                    start_counts,
                )
                filled += 1
                if filled == start_quota:
                    break
            if filled != start_quota:
                raise ValueError(
                    "unable to fill start_ucb seats with distinct feasible starts; "
                    "supply more eligible start groups or relax group constraints"
                )

        # Round-robin prevents the first strategy from consuming every member
        # of a useful cluster before the other acquisition modes get a turn.
        while sum(pending.values()) > 0:
            made_progress = False
            for strategy in SUPPORTED_STRATEGIES:
                if pending.get(strategy, 0) <= 0:
                    continue
                scores = self._strategy_scores(
                    strategy=strategy,
                    conservative=conservative,
                    ucb=ucb,
                    uncertainty=uncertainty,
                    novelty=novelty,
                    random_scores=random_scores,
                    mean_rank=mean_rank,
                    std_rank=std_rank,
                    objective_weights=objective_weights,
                    candidates=candidates,
                    selected=selected,
                    rng=rng,
                )
                floor = None if strategy == "exploit" else quality_threshold
                index = self._choose(
                    scores,
                    candidates,
                    selected,
                    cluster_counts,
                    start_counts,
                    quality_floor=floor,
                    conservative=conservative,
                )
                self._append(
                    index,
                    strategy,
                    scores[index],
                    candidates,
                    selected,
                    reasons,
                    acquisition_scores,
                    cluster_counts,
                    start_counts,
                )
                pending[strategy] -= 1
                made_progress = True
            if not made_progress:  # pragma: no cover - defensive invariant
                raise RuntimeError("acquisition allocation made no progress")

        # The batch is a portfolio, but top.fasta is explicitly ranked.  Rank
        # chosen members by conservative ensemble utility while preserving the
        # recorded reason that earned each seat.
        ranked_positions = sorted(
            range(len(selected)), key=lambda pos: (-conservative[selected[pos]], selected[pos])
        )
        ranked_indices = tuple(selected[pos] for pos in ranked_positions)
        ranked_reasons = tuple(reasons[pos] for pos in ranked_positions)
        ranked_scores = tuple(float(conservative[index]) for index in ranked_indices)
        ranked_acquisition_scores = tuple(
            acquisition_scores[position] for position in ranked_positions
        )
        if start_aware:
            assert candidates.start_ids is not None
            assert candidates.rollout_ids is not None
            ranked_start_evidence = tuple(
                StartSelectionEvidence(
                    start_id=candidates.start_ids[index],
                    rollout_id=candidates.rollout_ids[index],
                    start_rank=start_ranks[candidates.start_ids[index]],
                    eligible_rollout_count=eligible_per_start[candidates.start_ids[index]],
                    rollout_value_mean=start_value_summary[candidates.start_ids[index]][0],
                    rollout_value_dispersion=start_value_summary[candidates.start_ids[index]][1],
                    rollout_ucb_score=float(ucb[index]),
                )
                for index in ranked_indices
            )
        else:
            ranked_start_evidence = ()
        strategy_counts = Counter(ranked_reasons)

        return SelectionResult(
            indices=ranked_indices,
            reasons=ranked_reasons,
            conservative_scores=ranked_scores,
            acquisition_scores=ranked_acquisition_scores,
            cluster_counts=dict(sorted(cluster_counts.items())),
            strategy_counts=dict(sorted(strategy_counts.items())),
            start_evidence=ranked_start_evidence,
        )

    def _strategy_scores(
        self,
        *,
        strategy: str,
        conservative: FloatArray,
        ucb: FloatArray,
        uncertainty: FloatArray,
        novelty: FloatArray | None,
        random_scores: FloatArray,
        mean_rank: FloatArray,
        std_rank: FloatArray,
        objective_weights: FloatArray,
        candidates: CandidateBatch,
        selected: Sequence[int],
        rng: np.random.Generator,
    ) -> FloatArray:
        if strategy == "exploit":
            return conservative
        if strategy == "ucb":
            return ucb
        if strategy == "uncertainty":
            return uncertainty
        if strategy == "novelty":
            assert novelty is not None
            return novelty
        if strategy == "random":
            return random_scores + 0.10 * conservative
        if strategy == "pareto":
            direction = rng.dirichlet(np.maximum(objective_weights, 1e-6))
            return (mean_rank - 0.25 * self.config.risk_beta * std_rank) @ direction
        if strategy == "diversity":
            assert candidates.embeddings is not None
            distance = _minimum_cosine_distance(
                candidates.embeddings,
                selected,
                eligible=candidates.eligible
                if self.config.rollouts_per_start is not None
                else None,
            )
            weight = self.config.diversity_quality_weight
            quality_rank = (
                _rank_scale_eligible(conservative, candidates.eligible)
                if self.config.rollouts_per_start is not None
                else _rank_scale(conservative)
            )
            return (1.0 - weight) * distance + weight * quality_rank
        raise ValueError(f"unsupported strategy: {strategy}")

    def _choose(
        self,
        scores: FloatArray,
        candidates: CandidateBatch,
        selected: Sequence[int],
        cluster_counts: Counter[str],
        start_counts: Counter[str],
        *,
        quality_floor: float | None,
        conservative: FloatArray | None = None,
        candidate_mask: BoolArray | None = None,
    ) -> int:
        selected_set = set(selected)

        def allowed(index: int, *, enforce_floor: bool, enforce_cap: bool) -> bool:
            if (
                index in selected_set
                or not candidates.eligible[index]
                or (candidate_mask is not None and not candidate_mask[index])
            ):
                return False
            if (
                enforce_floor
                and quality_floor is not None
                and conservative is not None
                and conservative[index] < quality_floor
            ):
                return False
            return not (
                enforce_cap
                and self.config.max_per_cluster is not None
                and candidates.cluster_ids is not None
                and cluster_counts[candidates.cluster_ids[index]] >= self.config.max_per_cluster
            ) and not (
                self.config.max_per_start is not None
                and candidates.start_ids is not None
                and start_counts[candidates.start_ids[index]] >= self.config.max_per_start
            )

        if self.config.rollouts_per_start is not None:
            rollout_ids = candidates.rollout_ids or ("",) * len(scores)
            order = np.asarray(
                sorted(
                    range(len(scores)),
                    key=lambda index: (
                        -float(scores[index]),
                        candidates.sequences[index],
                        rollout_ids[index],
                    ),
                ),
                dtype=np.int64,
            )
        else:
            order = np.lexsort((np.arange(len(scores)), -scores))
        # Relax only the exploratory quality floor.  Cluster caps remain hard by
        # default because correlated top-100 motifs are a material wet-lab risk.
        for enforce_floor, enforce_cap in (
            (True, True),
            (False, True),
            (False, self.config.strict_cluster_cap),
        ):
            for raw_index in order:
                index = int(raw_index)
                if allowed(index, enforce_floor=enforce_floor, enforce_cap=enforce_cap):
                    return index
        raise ValueError(
            "unable to fill batch under eligibility/cluster constraints; "
            "increase candidate diversity or relax group caps"
        )

    @staticmethod
    def _append(
        index: int,
        reason: str,
        score: float,
        candidates: CandidateBatch,
        selected: list[int],
        reasons: list[str],
        acquisition_scores: list[float],
        cluster_counts: Counter[str],
        start_counts: Counter[str],
    ) -> None:
        selected.append(index)
        reasons.append(reason)
        acquisition_scores.append(float(score))
        if candidates.cluster_ids is not None:
            cluster_counts[candidates.cluster_ids[index]] += 1
        if candidates.start_ids is not None:
            start_counts[candidates.start_ids[index]] += 1

    def _objective_weights(self, n_objectives: int) -> FloatArray:
        if self.config.objective_weights is None:
            return np.full(n_objectives, 1.0 / n_objectives)
        weights = np.asarray(self.config.objective_weights, dtype=np.float64)
        if weights.shape != (n_objectives,):
            raise ValueError("objective_weights must contain one value per objective")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError("objective_weights must be finite, non-negative, and non-zero")
        return weights / np.sum(weights)


def _allocate_quotas(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Use the largest-remainder method for deterministic integer quotas."""

    if total == 0:
        return {name: 0 for name in weights}
    positive = {name: value for name, value in weights.items() if value > 0}
    scale = total / sum(positive.values())
    exact = {name: value * scale for name, value in positive.items()}
    quotas = {name: int(np.floor(value)) for name, value in exact.items()}
    missing = total - sum(quotas.values())
    remainder_order = sorted(
        positive, key=lambda name: (-(exact[name] - quotas[name]), SUPPORTED_STRATEGIES.index(name))
    )
    for name in remainder_order[:missing]:
        quotas[name] += 1
    return quotas


def _rank_scale(values: FloatArray) -> FloatArray:
    """Map values to stable average percentile ranks in [0, 1]."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("rank scaling expects a finite vector")
    if len(vector) == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(vector, kind="stable")
    result = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and vector[order[end]] == vector[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) / (len(vector) - 1)
        result[order[start:end]] = average_rank
        start = end
    return result


def _column_rank_scale(values: FloatArray) -> FloatArray:
    return np.column_stack([_rank_scale(values[:, column]) for column in range(values.shape[1])])


def _rank_scale_eligible(values: FloatArray, eligible: BoolArray) -> FloatArray:
    """Rank only selectable rows so rejected decoys cannot perturb scores."""

    vector = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if vector.ndim != 1 or mask.shape != vector.shape:
        raise ValueError("eligible rank scaling expects aligned vectors")
    if not np.any(mask):
        raise ValueError("eligible rank scaling requires at least one selectable row")
    result = np.zeros(len(vector), dtype=np.float64)
    result[mask] = _rank_scale(vector[mask])
    return result


def _column_rank_scale_eligible(values: FloatArray, eligible: BoolArray) -> FloatArray:
    return np.column_stack(
        [_rank_scale_eligible(values[:, column], eligible) for column in range(values.shape[1])]
    )


def _start_value_summary(
    rollout_value: FloatArray,
    start_ids: Sequence[str],
    eligible: BoolArray,
) -> dict[str, tuple[float, float]]:
    """Return eligible rollout-value mean and population SD for every start."""

    groups: dict[str, list[float]] = {}
    for index, start_id in enumerate(start_ids):
        if eligible[index]:
            groups.setdefault(start_id, []).append(float(rollout_value[index]))
    return {
        start_id: (
            float(np.mean(values, dtype=np.float64)),
            float(np.std(values, ddof=0, dtype=np.float64)),
        )
        for start_id, values in groups.items()
    }


def _keyed_random_scores(sequences: Sequence[str], seed: int) -> FloatArray:
    """Generate row-order-independent random priorities for start-aware replay."""

    denominator = float(1 << 64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}\0{sequence}".encode()).digest()[:8],
                "big",
            )
            / denominator
            for sequence in sequences
        ],
        dtype=np.float64,
    )


def _minimum_cosine_distance(
    embeddings: FloatArray,
    selected: Sequence[int],
    *,
    eligible: BoolArray | None = None,
) -> FloatArray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    if selected:
        similarities = normalized @ normalized[np.asarray(selected)].T
        return np.clip(1.0 - np.max(similarities, axis=1), 0.0, 2.0) / 2.0
    centroid_rows = normalized if eligible is None else normalized[np.asarray(eligible, dtype=bool)]
    centroid = np.mean(centroid_rows, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return np.ones(len(values), dtype=np.float64)
    similarities = normalized @ (centroid / centroid_norm)
    return np.clip(1.0 - similarities, 0.0, 2.0) / 2.0

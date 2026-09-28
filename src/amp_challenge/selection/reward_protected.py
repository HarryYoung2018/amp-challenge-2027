"""Reward-protected construction of the final wet-lab portfolio.

This module is intentionally separate from active-learning acquisition.  The
organizer samples uniformly from the submitted top portfolio, so the baseline
maximizes posterior *expected* reward under the locked quality gates.  A
deterministic local search may exchange candidates to improve diversity and
reference novelty, but only while preserving locked reward, lower-tail reward,
and category-specific reward relative to that baseline.

No uncertainty, UCB, or novelty term is used to build the baseline ranking.
Posterior reward samples are used only to evaluate the downside (CVaR) guard.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RewardProtectedConfig:
    """Locked policy for one final portfolio.

    ``reward_samples`` supplied to the selector must already contain normalized
    higher-is-better category rewards.  The selector does not invent endpoint
    transforms or a competition scoring formula.
    """

    portfolio_size: int = 100
    uniform_sample_size: int = 25
    cvar_alpha: float = 0.10
    panel_draws: int = 10_000
    panel_seed: int = 20260903
    confirmation_panel_draws: int = 10_000
    confirmation_panel_seed: int = 20260904
    objective_weights: Sequence[float] | None = None
    mean_tolerance: float = 0.01
    cvar_tolerance: float = 0.01
    category_tolerances: float | Sequence[float] = 0.02
    novelty_weight: float = 0.25
    novelty_cap: float = 1.0
    minimum_modifier_gain: float = 1e-10
    maximum_swaps: int = 25
    diversity_saturation_fraction: float = 0.95
    minimum_candidate_quality: float = 0.90
    minimum_portfolio_quality: float = 0.95
    addition_shortlist_size: int = 256
    removal_shortlist_size: int = 100
    maximum_proposals_per_step: int = 4_096

    def __post_init__(self) -> None:
        integer_fields = {
            "portfolio_size": self.portfolio_size,
            "uniform_sample_size": self.uniform_sample_size,
            "panel_draws": self.panel_draws,
            "confirmation_panel_draws": self.confirmation_panel_draws,
            "maximum_swaps": self.maximum_swaps,
            "addition_shortlist_size": self.addition_shortlist_size,
            "removal_shortlist_size": self.removal_shortlist_size,
            "maximum_proposals_per_step": self.maximum_proposals_per_step,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.panel_seed, bool) or not isinstance(self.panel_seed, int):
            raise ValueError("panel_seed must be an integer")
        if self.panel_seed < 0:
            raise ValueError("panel_seed must be non-negative")
        if isinstance(self.confirmation_panel_seed, bool) or not isinstance(
            self.confirmation_panel_seed, int
        ):
            raise ValueError("confirmation_panel_seed must be an integer")
        if self.confirmation_panel_seed < 0:
            raise ValueError("confirmation_panel_seed must be non-negative")
        if self.confirmation_panel_seed == self.panel_seed:
            raise ValueError("confirmation_panel_seed must differ from panel_seed")
        if self.uniform_sample_size > self.portfolio_size:
            raise ValueError("uniform_sample_size cannot exceed portfolio_size")
        unit_interval_fields = {
            "cvar_alpha": self.cvar_alpha,
            "diversity_saturation_fraction": self.diversity_saturation_fraction,
            "minimum_candidate_quality": self.minimum_candidate_quality,
            "minimum_portfolio_quality": self.minimum_portfolio_quality,
        }
        for name, value in unit_interval_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1]")
        nonnegative_fields = {
            "mean_tolerance": self.mean_tolerance,
            "cvar_tolerance": self.cvar_tolerance,
            "novelty_weight": self.novelty_weight,
            "minimum_modifier_gain": self.minimum_modifier_gain,
        }
        for name, value in nonnegative_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.novelty_cap, bool)
            or not isinstance(self.novelty_cap, int | float)
            or not math.isfinite(self.novelty_cap)
            or self.novelty_cap <= 0
        ):
            raise ValueError("novelty_cap must be finite and positive")
        if self.minimum_portfolio_quality < self.minimum_candidate_quality:
            raise ValueError("minimum_portfolio_quality cannot be below minimum_candidate_quality")
        _validate_nonnegative_sequence(
            self.objective_weights,
            name="objective_weights",
            require_positive_sum=True,
        )
        if isinstance(self.category_tolerances, int | float):
            if (
                isinstance(self.category_tolerances, bool)
                or not math.isfinite(self.category_tolerances)
                or self.category_tolerances < 0
            ):
                raise ValueError("category_tolerances must be finite and non-negative")
        else:
            _validate_nonnegative_sequence(
                self.category_tolerances,
                name="category_tolerances",
                require_positive_sum=False,
            )


@dataclass(frozen=True, slots=True)
class PortfolioCandidates:
    """Generic normalized reward draws and optional portfolio modifiers.

    Reward samples have shape ``(candidate, posterior_draw, category)``.  Draws
    must be aligned across candidates so their portfolio mean retains posterior
    dependence.  A two-dimensional ``(candidate, draw)`` array is accepted for
    a single category.
    """

    sequences: Sequence[str]
    reward_samples: FloatArray
    novelty: FloatArray | None = None
    embeddings: FloatArray | None = None
    physicochemical_features: FloatArray | None = None
    cluster_ids: Sequence[str] | None = None
    quality_probability: FloatArray | None = None
    out_of_distribution: BoolArray | None = None
    calibrated_lcb: FloatArray | None = None
    eligible: BoolArray | None = None

    def __post_init__(self) -> None:
        sequences = tuple(str(sequence) for sequence in self.sequences)
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("candidate sequences must be non-empty")
        if len(set(sequences)) != len(sequences):
            raise ValueError("candidate sequences must be unique")

        rewards = np.asarray(self.reward_samples)
        if rewards.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("reward_samples must use float32 or float64 storage")
        if rewards.ndim == 2:
            rewards = rewards[:, :, None]
        if (
            rewards.ndim != 3
            or rewards.shape[0] != len(sequences)
            or rewards.shape[1] == 0
            or rewards.shape[2] == 0
        ):
            raise ValueError("reward_samples must have shape (candidate, posterior_draw, category)")
        if np.any(~np.isfinite(rewards)) or np.any((rewards < 0) | (rewards > 1)):
            raise ValueError("reward_samples must be finite normalized values in [0, 1]")

        novelty = _optional_vector(self.novelty, len(sequences), name="novelty")
        if novelty is not None and np.any(novelty < 0):
            raise ValueError("novelty must be non-negative")
        embeddings = _optional_matrix(self.embeddings, len(sequences), name="embeddings")
        physicochemical = _optional_matrix(
            self.physicochemical_features,
            len(sequences),
            name="physicochemical_features",
        )

        clusters = None if self.cluster_ids is None else tuple(map(str, self.cluster_ids))
        if clusters is not None and (
            len(clusters) != len(sequences) or any(not cluster for cluster in clusters)
        ):
            raise ValueError("cluster_ids must contain one non-empty value per candidate")

        quality = (
            np.ones(len(sequences), dtype=np.float64)
            if self.quality_probability is None
            else _optional_vector(
                self.quality_probability,
                len(sequences),
                name="quality_probability",
            )
        )
        assert quality is not None
        if np.any((quality < 0) | (quality > 1)):
            raise ValueError("quality_probability must be in [0, 1]")

        ood = (
            np.zeros(len(sequences), dtype=bool)
            if self.out_of_distribution is None
            else _strict_bool_vector(
                self.out_of_distribution,
                len(sequences),
                name="out_of_distribution",
            )
        )
        lcb = _optional_vector(self.calibrated_lcb, len(sequences), name="calibrated_lcb")
        if lcb is not None and np.any((lcb < 0) | (lcb > 1)):
            raise ValueError("calibrated_lcb must be in [0, 1]")
        if np.any(ood) and lcb is None:
            raise ValueError("calibrated_lcb is required for out-of-distribution candidates")

        eligible = (
            np.ones(len(sequences), dtype=bool)
            if self.eligible is None
            else _strict_bool_vector(self.eligible, len(sequences), name="eligible")
        )

        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "reward_samples", rewards)
        object.__setattr__(self, "novelty", novelty)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "physicochemical_features", physicochemical)
        object.__setattr__(self, "cluster_ids", clusters)
        object.__setattr__(self, "quality_probability", quality)
        object.__setattr__(self, "out_of_distribution", ood)
        object.__setattr__(self, "calibrated_lcb", lcb)
        object.__setattr__(self, "eligible", eligible)


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Reward and modifier metrics for one fixed-size portfolio."""

    expected_reward: float
    cvar: float
    robust_reward: float
    category_expected: tuple[float, ...]
    uniform_sample_expected_total: float
    diversity: float
    novelty: float
    modifier: float
    effective_clusters: float | None
    mean_quality_probability: float


@dataclass(frozen=True, slots=True)
class PortfolioSwap:
    """One accepted, baseline-protected exchange."""

    step: int
    removed_index: int
    removed_sequence: str
    added_index: int
    added_sequence: str
    modifier_before: float
    modifier_after: float


@dataclass(frozen=True, slots=True)
class BaselineSolverEvidence:
    """Certificate for the constrained expected-reward control."""

    method: str
    status: str
    objective_value: float
    scipy_version: str | None = None
    highs_version: str | None = None
    solver_threads: int | None = None
    mip_gap: float | None = None
    mip_node_count: int | None = None


@dataclass(frozen=True, slots=True)
class ConfirmationEvidence:
    """Independent downside check consulted only after local search ends."""

    panel_draws: int
    panel_seed: int
    baseline_cvar: float
    searched_final_cvar: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class RewardProtectedResult:
    """Final ranked portfolio plus its immutable constrained-mean control."""

    indices: tuple[int, ...]
    reasons: tuple[str, ...]
    expected_scores: tuple[float, ...]
    baseline_indices: tuple[int, ...]
    baseline_metrics: PortfolioMetrics
    final_metrics: PortfolioMetrics
    search_swaps: tuple[PortfolioSwap, ...]
    swaps: tuple[PortfolioSwap, ...]
    fell_back_to_mean: bool
    fallback_reason: str | None
    mean_only_cutoff: float
    baseline_solver: BaselineSolverEvidence
    confirmation: ConfirmationEvidence

    @property
    def changed(self) -> bool:
        return bool(self.swaps)


@dataclass(frozen=True, slots=True)
class _PanelDesign:
    """Candidate-keyed common random numbers for uniform panels."""

    candidate_seeds: NDArray[np.uint64]
    candidate_tie_ranks: NDArray[np.int64]
    posterior_indices: NDArray[np.int64]
    sample_size: int


@dataclass(frozen=True, slots=True)
class _PanelState:
    """Exact uniform-panel state for one canonically identified portfolio."""

    candidates: tuple[int, ...]
    design: _PanelDesign
    scalar_samples: FloatArray
    priorities: NDArray[np.uint64]
    inclusion: BoolArray
    worst_selected_slots: NDArray[np.int64]
    next_slots: NDArray[np.int64] | None
    values: FloatArray


def uniform_sample_expectation(
    candidate_expected_rewards: Sequence[float] | FloatArray,
    sample_size: int,
) -> tuple[float, float]:
    """Return exact expected mean and total under uniform sampling without replacement.

    Every member of a portfolio of size ``n`` has inclusion probability
    ``sample_size / n``.  Consequently, the expected sampled mean equals the
    full-portfolio mean and the expected sampled total is ``sample_size`` times
    that mean.  No Monte Carlo approximation is needed.
    """

    values = np.asarray(candidate_expected_rewards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("candidate_expected_rewards must be a non-empty finite vector")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer")
    if sample_size <= 0 or sample_size > values.size:
        raise ValueError("sample_size must be in [1, number of candidates]")
    expected_mean = float(math.fsum(float(value) for value in values) / values.size)
    return expected_mean, float(sample_size * expected_mean)


def uniform_panel_cvar(
    scalar_reward_samples: FloatArray,
    *,
    candidate_keys: Sequence[str],
    sample_size: int,
    alpha: float = 0.10,
    panel_draws: int = 10_000,
    panel_seed: int = 20260903,
) -> float:
    """Estimate joint posterior/uniform-panel lower-tail reward deterministically.

    The input has shape ``(portfolio_member, posterior_draw)`` and
    ``candidate_keys`` supplies the stable unique identity aligned to its rows.
    Each joint draw selects exactly ``sample_size`` members without replacement
    and pairs that panel with a cycled posterior draw. Candidate-keyed priorities
    make the result invariant to a joint row/key permutation.
    """

    rewards = np.asarray(scalar_reward_samples, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("scalar_reward_samples must have shape (member, posterior_draw)")
    if np.any(~np.isfinite(rewards)):
        raise ValueError("scalar_reward_samples must be finite")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if isinstance(panel_draws, bool) or not isinstance(panel_draws, int) or panel_draws <= 0:
        raise ValueError("panel_draws must be a positive integer")
    if isinstance(panel_seed, bool) or not isinstance(panel_seed, int) or panel_seed < 0:
        raise ValueError("panel_seed must be a non-negative integer")
    keys = tuple(candidate_keys)
    if (
        len(keys) != rewards.shape[0]
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("candidate_keys must contain one unique non-empty string per row")
    design = _make_panel_design(
        candidate_keys=keys,
        sample_size=sample_size,
        posterior_draws=rewards.shape[1],
        panel_draws=panel_draws,
        panel_seed=panel_seed,
    )
    state = _make_panel_state(
        tuple(range(rewards.shape[0])),
        rewards[:, :, None],
        np.ones(1, dtype=np.float64),
        keys,
        design,
    )
    return _lower_tail_mean(state.values, alpha)


class RewardProtectedPortfolioSelector:
    """Improve a quality-constrained mean portfolio without spending reward."""

    def __init__(self, config: RewardProtectedConfig) -> None:
        self.config = config

    def select(self, candidates: PortfolioCandidates) -> RewardProtectedResult:
        n_candidates, _, n_categories = candidates.reward_samples.shape
        if n_candidates < self.config.portfolio_size:
            raise ValueError("candidate pool is smaller than portfolio_size")

        weights = self._objective_weights(n_categories)
        category_expected = np.mean(candidates.reward_samples, axis=1, dtype=np.float64)
        candidate_expected = category_expected @ weights
        hard_eligible, mean_only_cutoff = self._hard_eligibility(
            candidates,
            candidate_expected,
        )
        if np.count_nonzero(hard_eligible) < self.config.portfolio_size:
            raise ValueError("hard quality/OOD gates leave too few eligible candidates")

        baseline, baseline_solver = self._quality_constrained_mean_baseline(
            candidates,
            candidate_expected,
            hard_eligible,
        )
        panel_design = _make_panel_design(
            candidate_keys=candidates.sequences,
            sample_size=self.config.uniform_sample_size,
            posterior_draws=candidates.reward_samples.shape[1],
            panel_draws=self.config.panel_draws,
            panel_seed=self.config.panel_seed,
        )
        confirmation_design = _make_panel_design(
            candidate_keys=candidates.sequences,
            sample_size=self.config.uniform_sample_size,
            posterior_draws=candidates.reward_samples.shape[1],
            panel_draws=self.config.confirmation_panel_draws,
            panel_seed=self.config.confirmation_panel_seed,
        )
        prepared_embedding = _prepare_features(
            candidates.embeddings,
            candidates.sequences,
            fit_mask=hard_eligible,
        )
        prepared_physicochemical = _prepare_features(
            candidates.physicochemical_features,
            candidates.sequences,
            fit_mask=hard_eligible,
        )
        baseline_panel_state = _make_panel_state(
            baseline,
            candidates.reward_samples,
            weights,
            candidates.sequences,
            panel_design,
        )
        baseline_metrics = self._metrics(
            baseline,
            candidates,
            candidate_expected,
            category_expected,
            prepared_embedding,
            prepared_physicochemical,
            panel_values=baseline_panel_state.values,
        )

        fallback_reason: str | None = None
        if (
            baseline_metrics.effective_clusters is not None
            and baseline_metrics.effective_clusters
            >= self.config.diversity_saturation_fraction * self.config.portfolio_size - _EPSILON
        ):
            fallback_reason = "diversity_saturated"
        elif (
            candidates.novelty is None
            and candidates.cluster_ids is None
            and prepared_embedding is None
            and prepared_physicochemical is None
        ):
            fallback_reason = "no_diversity_or_novelty_inputs"

        slots = baseline
        swaps: list[PortfolioSwap] = []
        added_at_step: dict[int, int] = {}
        if fallback_reason is None:
            for step in range(1, self.config.maximum_swaps + 1):
                current = slots
                current_panel_state = _make_panel_state(
                    current,
                    candidates.reward_samples,
                    weights,
                    candidates.sequences,
                    panel_design,
                )
                current_metrics = self._metrics(
                    current,
                    candidates,
                    candidate_expected,
                    category_expected,
                    prepared_embedding,
                    prepared_physicochemical,
                    panel_values=current_panel_state.values,
                )
                proposal = self._best_swap(
                    current=current,
                    current_metrics=current_metrics,
                    baseline_metrics=baseline_metrics,
                    candidates=candidates,
                    weights=weights,
                    category_expected=category_expected,
                    candidate_expected=candidate_expected,
                    hard_eligible=hard_eligible,
                    prepared_embedding=prepared_embedding,
                    prepared_physicochemical=prepared_physicochemical,
                    panel_state=current_panel_state,
                )
                if proposal is None:
                    break
                removed, added, proposal_metrics = proposal
                slots = tuple(
                    sorted(
                        (set(current) - {removed}) | {added},
                        key=lambda index: candidates.sequences[index],
                    )
                )
                added_at_step.pop(removed, None)
                added_at_step[added] = step
                swaps.append(
                    PortfolioSwap(
                        step=step,
                        removed_index=removed,
                        removed_sequence=candidates.sequences[removed],
                        added_index=added,
                        added_sequence=candidates.sequences[added],
                        modifier_before=current_metrics.modifier,
                        modifier_after=proposal_metrics.modifier,
                    )
                )
            if not swaps:
                fallback_reason = "no_reward_protected_improvement"

        search_swaps = tuple(swaps)
        searched_final = slots
        searched_panel_state = _make_panel_state(
            searched_final,
            candidates.reward_samples,
            weights,
            candidates.sequences,
            panel_design,
        )
        searched_metrics = self._metrics(
            searched_final,
            candidates,
            candidate_expected,
            category_expected,
            prepared_embedding,
            prepared_physicochemical,
            panel_values=searched_panel_state.values,
        )
        if not self._reward_is_protected(searched_metrics, baseline_metrics):
            raise RuntimeError("internal error: final portfolio violates its reward guard")

        confirmation_baseline_state = _make_panel_state(
            baseline,
            candidates.reward_samples,
            weights,
            candidates.sequences,
            confirmation_design,
        )
        confirmation_final_state = _make_panel_state(
            searched_final,
            candidates.reward_samples,
            weights,
            candidates.sequences,
            confirmation_design,
        )
        confirmation_baseline_cvar = _lower_tail_mean(
            confirmation_baseline_state.values,
            self.config.cvar_alpha,
        )
        confirmation_final_cvar = _lower_tail_mean(
            confirmation_final_state.values,
            self.config.cvar_alpha,
        )
        confirmation_accepted = bool(
            confirmation_final_cvar
            >= confirmation_baseline_cvar - self.config.cvar_tolerance - _EPSILON
        )
        confirmation = ConfirmationEvidence(
            panel_draws=self.config.confirmation_panel_draws,
            panel_seed=self.config.confirmation_panel_seed,
            baseline_cvar=confirmation_baseline_cvar,
            searched_final_cvar=confirmation_final_cvar,
            accepted=confirmation_accepted,
        )
        if swaps and not confirmation_accepted:
            final_unranked = baseline
            final_metrics = baseline_metrics
            swaps = []
            added_at_step.clear()
            fallback_reason = "confirmation_cvar_guard"
        else:
            final_unranked = searched_final
            final_metrics = searched_metrics

        selected = set(final_unranked)
        ranked = tuple(
            sorted(
                selected,
                key=lambda index: (-candidate_expected[index], candidates.sequences[index]),
            )
        )
        baseline_set = set(baseline)
        reasons = tuple(
            (
                "quality_constrained_mean_baseline"
                if index in baseline_set
                else f"reward_protected_swap:{added_at_step[index]}"
            )
            for index in ranked
        )
        return RewardProtectedResult(
            indices=ranked,
            reasons=reasons,
            expected_scores=tuple(float(candidate_expected[index]) for index in ranked),
            baseline_indices=tuple(
                sorted(
                    baseline,
                    key=lambda index: (-candidate_expected[index], candidates.sequences[index]),
                )
            ),
            baseline_metrics=baseline_metrics,
            final_metrics=final_metrics,
            search_swaps=search_swaps,
            swaps=tuple(swaps),
            fell_back_to_mean=not swaps,
            fallback_reason=fallback_reason,
            mean_only_cutoff=float(mean_only_cutoff),
            baseline_solver=baseline_solver,
            confirmation=confirmation,
        )

    def _objective_weights(self, n_categories: int) -> FloatArray:
        if self.config.objective_weights is None:
            return np.full(n_categories, 1.0 / n_categories, dtype=np.float64)
        weights = np.asarray(self.config.objective_weights, dtype=np.float64)
        if weights.shape != (n_categories,):
            raise ValueError("objective_weights must have one value per reward category")
        return weights / np.sum(weights)

    def _category_tolerances(self, n_categories: int) -> FloatArray:
        tolerances = self.config.category_tolerances
        if isinstance(tolerances, int | float):
            return np.full(n_categories, float(tolerances), dtype=np.float64)
        vector = np.asarray(tolerances, dtype=np.float64)
        if vector.shape != (n_categories,):
            raise ValueError("category_tolerances must have one value per reward category")
        return vector

    def _hard_eligibility(
        self,
        candidates: PortfolioCandidates,
        candidate_expected: FloatArray,
    ) -> tuple[BoolArray, float]:
        assert candidates.quality_probability is not None
        assert candidates.out_of_distribution is not None
        assert candidates.eligible is not None
        raw = candidates.eligible & (
            candidates.quality_probability >= self.config.minimum_candidate_quality - _EPSILON
        )
        if np.count_nonzero(raw) < self.config.portfolio_size:
            raise ValueError("individual quality gate leaves too few eligible candidates")
        raw_order = _ranked_indices(raw, candidate_expected, candidates.sequences)
        cutoff = float(candidate_expected[raw_order[self.config.portfolio_size - 1]])
        if candidates.calibrated_lcb is None:
            ood_allowed = ~candidates.out_of_distribution
        else:
            ood_allowed = (~candidates.out_of_distribution) | (
                candidates.calibrated_lcb >= cutoff - _EPSILON
            )
        return raw & ood_allowed, cutoff

    def _quality_constrained_mean_baseline(
        self,
        candidates: PortfolioCandidates,
        candidate_expected: FloatArray,
        hard_eligible: BoolArray,
    ) -> tuple[tuple[int, ...], BaselineSolverEvidence]:
        ranked = _ranked_indices(hard_eligible, candidate_expected, candidates.sequences)
        unconstrained = tuple(ranked[: self.config.portfolio_size])
        assert candidates.quality_probability is not None
        target_quality = self.config.minimum_portfolio_quality * self.config.portfolio_size
        unconstrained_quality = math.fsum(
            float(candidates.quality_probability[index]) for index in unconstrained
        )
        if unconstrained_quality >= target_quality - _EPSILON:
            baseline = tuple(sorted(unconstrained, key=lambda index: candidates.sequences[index]))
            objective = math.fsum(float(candidate_expected[index]) for index in baseline)
            return baseline, BaselineSolverEvidence(
                method="top_k_sort",
                status="optimal_by_sorting",
                objective_value=objective,
            )

        maximum_quality = math.fsum(
            float(candidates.quality_probability[index])
            for index in sorted(
                np.flatnonzero(hard_eligible),
                key=lambda index: (
                    -candidates.quality_probability[index],
                    candidates.sequences[index],
                ),
            )[: self.config.portfolio_size]
        )
        if maximum_quality < target_quality - _EPSILON:
            raise ValueError("portfolio mean-quality gate is infeasible")

        # With arbitrary real-valued quality probabilities this is a cardinality-
        # constrained 0/1 knapsack, so a greedy repair has no optimality guarantee.
        # Import the pinned solver only on the uncommon constrained path.
        try:
            import scipy
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.optimize._highspy._core import (
                HIGHS_VERSION_MAJOR,
                HIGHS_VERSION_MINOR,
                HIGHS_VERSION_PATCH,
            )
        except ImportError as error:  # pragma: no cover - required project dependency
            raise RuntimeError(
                "SciPy/HiGHS is required when the top-k reward baseline violates "
                "the portfolio mean-quality gate"
            ) from error

        eligible = np.asarray(
            sorted(
                (int(index) for index in np.flatnonzero(hard_eligible)),
                key=lambda index: candidates.sequences[index],
            ),
            dtype=np.int64,
        )
        objective = -candidate_expected[eligible]
        quality = candidates.quality_probability[eligible]
        constraints = LinearConstraint(
            np.vstack((np.ones(len(eligible), dtype=np.float64), quality)),
            lb=np.asarray([self.config.portfolio_size, target_quality], dtype=np.float64),
            ub=np.asarray([self.config.portfolio_size, np.inf], dtype=np.float64),
        )
        options: dict[str, object] = {
            "presolve": True,
            "mip_rel_gap": 0.0,
            # SciPy forwards these deterministic HiGHS options verbatim.
            "threads": 1,
            "parallel": False,
            "random_seed": 0,
            "mip_feasibility_tolerance": 1e-9,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected.*",
                category=RuntimeWarning,
            )
            solution = milp(
                c=objective,
                integrality=np.ones(len(eligible), dtype=np.uint8),
                bounds=Bounds(0.0, 1.0),
                constraints=constraints,
                options=options,
            )
        mip_gap = getattr(solution, "mip_gap", None)
        if (
            not solution.success
            or solution.status != 0
            or solution.x is None
            or mip_gap is None
            or not math.isfinite(float(mip_gap))
            or float(mip_gap) > _EPSILON
        ):
            raise RuntimeError(
                "quality-constrained baseline solver did not certify an optimal zero-gap solution: "
                f"status={solution.status} message={solution.message!s} mip_gap={mip_gap}"
            )
        rounded = np.rint(solution.x)
        if np.any(np.abs(solution.x - rounded) > 1e-7):
            raise RuntimeError("quality-constrained baseline solver returned a fractional solution")
        selected = eligible[np.flatnonzero(rounded.astype(bool))]
        if len(selected) != self.config.portfolio_size:
            raise RuntimeError("quality-constrained baseline solver returned the wrong set size")
        selected_quality = math.fsum(
            float(candidates.quality_probability[index]) for index in selected
        )
        if selected_quality < target_quality - _EPSILON:
            raise RuntimeError("quality-constrained baseline solver violated the quality gate")
        baseline = tuple(
            sorted(
                (int(index) for index in selected), key=lambda index: candidates.sequences[index]
            )
        )
        selected_objective = math.fsum(float(candidate_expected[index]) for index in baseline)
        if solution.fun is None or not math.isclose(
            selected_objective,
            -float(solution.fun),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise RuntimeError("quality-constrained baseline solver objective is inconsistent")
        node_count = getattr(solution, "mip_node_count", None)
        return baseline, BaselineSolverEvidence(
            method="scipy_milp_highs",
            status="optimal_zero_gap",
            objective_value=selected_objective,
            scipy_version=scipy.__version__,
            highs_version=(f"{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}"),
            solver_threads=1,
            mip_gap=float(mip_gap),
            mip_node_count=None if node_count is None else int(node_count),
        )

    def _best_swap(
        self,
        *,
        current: tuple[int, ...],
        current_metrics: PortfolioMetrics,
        baseline_metrics: PortfolioMetrics,
        candidates: PortfolioCandidates,
        weights: FloatArray,
        category_expected: FloatArray,
        candidate_expected: FloatArray,
        hard_eligible: BoolArray,
        prepared_embedding: FloatArray | None,
        prepared_physicochemical: FloatArray | None,
        panel_state: _PanelState,
    ) -> tuple[int, int, PortfolioMetrics] | None:
        selected = set(current)
        outside = [int(index) for index in np.flatnonzero(hard_eligible) if index not in selected]
        if not outside:
            return None

        contribution = self._candidate_modifier_proxy(
            tuple(range(len(candidates.sequences))),
            current,
            candidates,
            prepared_embedding,
            prepared_physicochemical,
        )
        additions = sorted(
            outside,
            key=lambda index: (-contribution[index], candidates.sequences[index]),
        )[: self.config.addition_shortlist_size]
        removals = sorted(
            current,
            key=lambda index: (contribution[index], candidates.sequences[index]),
        )[: self.config.removal_shortlist_size]
        pairs = sorted(
            ((removed, added) for removed in removals for added in additions),
            key=lambda pair: (
                -(contribution[pair[1]] - contribution[pair[0]]),
                candidates.sequences[pair[1]],
                candidates.sequences[pair[0]],
            ),
        )[: self.config.maximum_proposals_per_step]

        best: tuple[float, str, str, int, int, PortfolioMetrics] | None = None
        category_tolerances = self._category_tolerances(category_expected.shape[1])
        baseline_categories = np.asarray(baseline_metrics.category_expected)
        current_categories = np.asarray(current_metrics.category_expected)
        assert candidates.quality_probability is not None
        added_panel_cache: dict[int, tuple[FloatArray, NDArray[np.uint64]]] = {}
        for removed, added in pairs:
            proposed_expected = (
                current_metrics.expected_reward
                + float(candidate_expected[added] - candidate_expected[removed])
                / self.config.portfolio_size
            )
            if (
                proposed_expected
                < baseline_metrics.expected_reward - self.config.mean_tolerance - _EPSILON
            ):
                continue
            proposed_categories = (
                current_categories
                + (category_expected[added] - category_expected[removed])
                / self.config.portfolio_size
            )
            if np.any(proposed_categories < baseline_categories - category_tolerances - _EPSILON):
                continue
            proposed_quality = (
                current_metrics.mean_quality_probability
                + float(
                    candidates.quality_probability[added] - candidates.quality_probability[removed]
                )
                / self.config.portfolio_size
            )
            if proposed_quality < self.config.minimum_portfolio_quality - _EPSILON:
                continue
            proposal = tuple(
                sorted(
                    (selected - {removed}) | {added},
                    key=lambda index: candidates.sequences[index],
                )
            )
            added_panel = added_panel_cache.get(added)
            if added_panel is None:
                added_scalar = np.asarray(
                    candidates.reward_samples[added] @ weights,
                    dtype=np.float64,
                )
                added_panel = (
                    added_scalar,
                    _candidate_priorities(added, panel_state.design),
                )
                added_panel_cache[added] = added_panel
            added_scalar, added_priorities = added_panel
            proposal_panel_values = _proposal_panel_values(
                panel_state,
                removed=removed,
                added=added,
                added_scalar_samples=added_scalar,
                added_priorities=added_priorities,
            )
            if _lower_tail_mean(proposal_panel_values, self.config.cvar_alpha) < (
                baseline_metrics.cvar - self.config.cvar_tolerance - _EPSILON
            ):
                continue
            metrics = self._metrics(
                proposal,
                candidates,
                candidate_expected,
                category_expected,
                prepared_embedding,
                prepared_physicochemical,
                panel_values=proposal_panel_values,
            )
            if not self._reward_is_protected(metrics, baseline_metrics):
                continue
            gain = metrics.modifier - current_metrics.modifier
            if gain <= max(self.config.minimum_modifier_gain, _EPSILON):
                continue
            choice = (
                -metrics.modifier,
                candidates.sequences[added],
                candidates.sequences[removed],
                removed,
                added,
                metrics,
            )
            if best is None or choice[:5] < best[:5]:
                best = choice
        if best is None:
            return None
        return best[3], best[4], best[5]

    def _candidate_modifier_proxy(
        self,
        indices: tuple[int, ...],
        selected: tuple[int, ...],
        candidates: PortfolioCandidates,
        prepared_embedding: FloatArray | None,
        prepared_physicochemical: FloatArray | None,
    ) -> FloatArray:
        diversity_components: list[FloatArray] = []
        if candidates.cluster_ids is not None:
            counts = Counter(candidates.cluster_ids[index] for index in selected)
            diversity_components.append(
                np.asarray(
                    [1.0 / max(1, counts[candidates.cluster_ids[index]]) for index in indices],
                    dtype=np.float64,
                )
            )
        for features in (prepared_embedding, prepared_physicochemical):
            if features is not None:
                diversity_components.append(_feature_distance_proxy(features, selected))
        score = (
            np.mean(np.vstack(diversity_components), axis=0)
            if diversity_components
            else np.zeros(len(candidates.sequences), dtype=np.float64)
        )
        if candidates.novelty is not None:
            score = score + self.config.novelty_weight * (
                np.minimum(candidates.novelty, self.config.novelty_cap) / self.config.novelty_cap
            )
        return score

    def _metrics(
        self,
        indices: tuple[int, ...],
        candidates: PortfolioCandidates,
        candidate_expected: FloatArray,
        category_expected: FloatArray,
        prepared_embedding: FloatArray | None,
        prepared_physicochemical: FloatArray | None,
        *,
        panel_values: FloatArray,
    ) -> PortfolioMetrics:
        ordered = tuple(sorted(indices, key=lambda index: candidates.sequences[index]))
        selected = np.asarray(ordered, dtype=np.int64)
        selected_expected = candidate_expected[selected]
        expected = float(
            math.fsum(float(value) for value in selected_expected) / len(selected_expected)
        )
        cvar = _lower_tail_mean(panel_values, self.config.cvar_alpha)
        categories = tuple(
            float(
                math.fsum(float(value) for value in category_expected[selected, column])
                / len(selected)
            )
            for column in range(category_expected.shape[1])
        )
        _, expected_total = uniform_sample_expectation(
            selected_expected,
            self.config.uniform_sample_size,
        )

        diversity_components: list[float] = []
        for features in (prepared_embedding, prepared_physicochemical):
            if features is not None:
                diversity_components.append(_normalized_logdet(features[selected]))
        effective_clusters: float | None = None
        if candidates.cluster_ids is not None:
            counts = Counter(candidates.cluster_ids[index] for index in ordered)
            proportions = np.asarray(list(counts.values()), dtype=np.float64) / len(ordered)
            effective_clusters = float(1.0 / np.sum(proportions**2))
            diversity_components.append(effective_clusters / len(ordered))
        diversity = float(np.mean(diversity_components)) if diversity_components else 0.0
        novelty = (
            0.0
            if candidates.novelty is None
            else float(
                np.mean(
                    np.minimum(candidates.novelty[selected], self.config.novelty_cap)
                    / self.config.novelty_cap
                )
            )
        )
        assert candidates.quality_probability is not None
        mean_quality = float(
            math.fsum(float(candidates.quality_probability[index]) for index in ordered)
            / len(ordered)
        )
        return PortfolioMetrics(
            expected_reward=expected,
            cvar=cvar,
            robust_reward=0.80 * expected + 0.20 * cvar,
            category_expected=categories,
            uniform_sample_expected_total=expected_total,
            diversity=diversity,
            novelty=novelty,
            modifier=diversity + self.config.novelty_weight * novelty,
            effective_clusters=effective_clusters,
            mean_quality_probability=mean_quality,
        )

    def _reward_is_protected(
        self,
        metrics: PortfolioMetrics,
        baseline: PortfolioMetrics,
    ) -> bool:
        tolerances = self._category_tolerances(len(metrics.category_expected))
        category_values = np.asarray(metrics.category_expected)
        category_baseline = np.asarray(baseline.category_expected)
        return bool(
            metrics.expected_reward
            >= baseline.expected_reward - self.config.mean_tolerance - _EPSILON
            and metrics.cvar >= baseline.cvar - self.config.cvar_tolerance - _EPSILON
            and np.all(category_values >= category_baseline - tolerances - _EPSILON)
            and metrics.mean_quality_probability >= self.config.minimum_portfolio_quality - _EPSILON
        )


def _validate_nonnegative_sequence(
    values: Sequence[float] | None,
    *,
    name: str,
    require_positive_sum: bool,
) -> None:
    if values is None:
        return
    if any(isinstance(value, bool | np.bool_) for value in values):
        raise ValueError(f"{name} must not contain boolean values")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    if require_positive_sum and np.sum(array) <= 0:
        raise ValueError(f"{name} must have positive total weight")


def _optional_vector(
    values: Sequence[float] | FloatArray | None,
    length: int,
    *,
    name: str,
) -> FloatArray | None:
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with one value per candidate")
    return vector


def _strict_bool_vector(
    values: Sequence[bool] | BoolArray,
    length: int,
    *,
    name: str,
) -> BoolArray:
    vector = np.asarray(values)
    if vector.dtype.kind != "b" or vector.shape != (length,):
        raise ValueError(f"{name} must be a boolean vector with one value per candidate")
    return vector.astype(bool, copy=False)


def _optional_matrix(
    values: FloatArray | None,
    length: int,
    *,
    name: str,
) -> FloatArray | None:
    if values is None:
        return None
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != length
        or matrix.shape[1] == 0
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError(f"{name} must be a finite (candidate, feature) matrix")
    return matrix


def _ranked_indices(
    eligible: BoolArray,
    values: FloatArray,
    sequences: Sequence[str],
) -> list[int]:
    return sorted(
        (int(index) for index in np.flatnonzero(eligible)),
        key=lambda index: (-values[index], sequences[index]),
    )


def _make_panel_design(
    *,
    candidate_keys: Sequence[str],
    sample_size: int,
    posterior_draws: int,
    panel_draws: int,
    panel_seed: int,
) -> _PanelDesign:
    keys = tuple(candidate_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("candidate_keys must be non-empty and unique")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if posterior_draws <= 0 or panel_draws <= 0:
        raise ValueError("posterior_draws and panel_draws must be positive")
    candidate_seeds = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    b"amp-uniform-panel-v1\0"
                    + str(panel_seed).encode("ascii")
                    + b"\0"
                    + key.encode("utf-8")
                ).digest()[:8],
                "little",
            )
            for key in keys
        ],
        dtype=np.uint64,
    )
    tie_ranks = np.empty(len(keys), dtype=np.int64)
    for rank, index in enumerate(sorted(range(len(keys)), key=lambda index: keys[index])):
        tie_ranks[index] = rank
    posterior_indices = np.arange(panel_draws, dtype=np.int64) % posterior_draws
    return _PanelDesign(
        candidate_seeds=candidate_seeds,
        candidate_tie_ranks=tie_ranks,
        posterior_indices=posterior_indices,
        sample_size=sample_size,
    )


def _candidate_priorities(candidate: int, design: _PanelDesign) -> NDArray[np.uint64]:
    values = np.arange(len(design.posterior_indices), dtype=np.uint64)
    values ^= design.candidate_seeds[candidate]
    with np.errstate(over="ignore"):
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _make_panel_state(
    candidates: tuple[int, ...],
    reward_samples: FloatArray,
    weights: FloatArray,
    sequences: Sequence[str],
    design: _PanelDesign,
) -> _PanelState:
    ordered = tuple(sorted(candidates, key=lambda index: sequences[index]))
    if len(ordered) != len(set(ordered)):
        raise ValueError("portfolio candidates must be unique")
    if design.sample_size > len(ordered):
        raise ValueError("sample_size cannot exceed portfolio size")
    selected = np.asarray(ordered, dtype=np.int64)
    scalar_samples = np.asarray(reward_samples[selected] @ weights, dtype=np.float64)
    panel_count = len(design.posterior_indices)
    priorities = np.empty((panel_count, len(ordered)), dtype=np.uint64)
    for slot, candidate in enumerate(ordered):
        priorities[:, slot] = _candidate_priorities(candidate, design)
    priority_order = np.argsort(priorities, axis=1, kind="stable")
    chosen = priority_order[:, : design.sample_size]
    inclusion = np.zeros_like(priorities, dtype=bool)
    np.put_along_axis(inclusion, chosen, True, axis=1)
    worst_selected_slots = priority_order[:, design.sample_size - 1].astype(np.int64, copy=False)
    next_slots = (
        None
        if design.sample_size == len(ordered)
        else priority_order[:, design.sample_size].astype(np.int64, copy=False)
    )
    values = np.mean(
        scalar_samples[chosen, design.posterior_indices[:, None]],
        axis=1,
        dtype=np.float64,
    )
    return _PanelState(
        candidates=ordered,
        design=design,
        scalar_samples=scalar_samples,
        priorities=priorities,
        inclusion=inclusion,
        worst_selected_slots=worst_selected_slots,
        next_slots=next_slots,
        values=values,
    )


def _priority_precedes(
    challenger_priorities: NDArray[np.uint64],
    incumbent_priorities: NDArray[np.uint64],
    *,
    challenger: int,
    incumbents: NDArray[np.int64],
    design: _PanelDesign,
) -> BoolArray:
    return (challenger_priorities < incumbent_priorities) | (
        (challenger_priorities == incumbent_priorities)
        & (design.candidate_tie_ranks[challenger] < design.candidate_tie_ranks[incumbents])
    )


def _proposal_panel_values(
    state: _PanelState,
    *,
    removed: int,
    added: int,
    added_scalar_samples: FloatArray,
    added_priorities: NDArray[np.uint64],
) -> FloatArray:
    try:
        removed_slot = state.candidates.index(removed)
    except ValueError as error:  # pragma: no cover - internal selector contract
        raise RuntimeError("removed candidate is not in the panel state") from error
    design = state.design
    posterior = design.posterior_indices
    result = state.values.copy()
    if design.sample_size == len(state.candidates):
        result += (
            added_scalar_samples[posterior] - state.scalar_samples[removed_slot, posterior]
        ) / design.sample_size
        return result

    removed_included = state.inclusion[:, removed_slot]
    included_rows = np.flatnonzero(removed_included)
    assert state.next_slots is not None
    if included_rows.size:
        next_slots = state.next_slots[included_rows]
        incumbent_priorities = state.priorities[included_rows, next_slots]
        incumbent_candidates = np.asarray(state.candidates, dtype=np.int64)[next_slots]
        use_added = _priority_precedes(
            added_priorities[included_rows],
            incumbent_priorities,
            challenger=added,
            incumbents=incumbent_candidates,
            design=design,
        )
        draw_indices = posterior[included_rows]
        fill_rewards = state.scalar_samples[next_slots, draw_indices]
        fill_rewards = np.where(use_added, added_scalar_samples[draw_indices], fill_rewards)
        result[included_rows] += (
            fill_rewards - state.scalar_samples[removed_slot, draw_indices]
        ) / design.sample_size

    excluded_rows = np.flatnonzero(~removed_included)
    if excluded_rows.size:
        worst_slots = state.worst_selected_slots[excluded_rows]
        incumbent_priorities = state.priorities[excluded_rows, worst_slots]
        incumbent_candidates = np.asarray(state.candidates, dtype=np.int64)[worst_slots]
        use_added = _priority_precedes(
            added_priorities[excluded_rows],
            incumbent_priorities,
            challenger=added,
            incumbents=incumbent_candidates,
            design=design,
        )
        changed_rows = excluded_rows[use_added]
        changed_slots = worst_slots[use_added]
        draw_indices = posterior[changed_rows]
        result[changed_rows] += (
            added_scalar_samples[draw_indices] - state.scalar_samples[changed_slots, draw_indices]
        ) / design.sample_size
    return result


def _lower_tail_mean(values: FloatArray, alpha: float) -> float:
    tail_count = max(1, math.ceil(alpha * len(values)))
    tail = np.partition(values, tail_count - 1)[:tail_count]
    return float(np.mean(tail))


def _prepare_features(
    values: FloatArray | None,
    sequences: Sequence[str],
    *,
    fit_mask: BoolArray,
) -> FloatArray | None:
    if values is None:
        return None
    canonical_order = np.asarray(
        sorted(
            (int(index) for index in np.flatnonzero(fit_mask)),
            key=lambda index: sequences[index],
        ),
        dtype=np.int64,
    )
    fit_values = values[canonical_order]
    center = np.mean(fit_values, axis=0)
    scale = np.std(fit_values, axis=0)
    standardized = np.empty_like(values, dtype=np.float64)
    np.subtract(values, center, out=standardized)
    np.divide(
        standardized,
        scale,
        out=standardized,
        where=scale > _EPSILON,
    )
    standardized[:, scale <= _EPSILON] = 0.0
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    standardized = np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized),
        where=norms > _EPSILON,
    )
    return standardized


def _feature_distance_proxy(
    features: FloatArray,
    selected: tuple[int, ...],
    *,
    block_size: int = 4_096,
) -> FloatArray:
    """Return a bounded logdet-aligned redundancy proxy without an N x K allocation."""

    selected_array = np.asarray(selected, dtype=np.int64)
    selected_features = features[selected_array]
    selected_slot = {candidate: slot for slot, candidate in enumerate(selected)}
    result = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), block_size):
        stop = min(start + block_size, len(features))
        similarity = np.abs(features[start:stop] @ selected_features.T)
        for candidate, slot in selected_slot.items():
            if start <= candidate < stop:
                similarity[candidate - start, slot] = -np.inf
        maximum = np.max(similarity, axis=1)
        distance = np.where(np.isfinite(maximum), 1.0 - maximum, 1.0)
        nonzero = np.linalg.norm(features[start:stop], axis=1) > _EPSILON
        result[start:stop] = np.where(nonzero, np.clip(distance, 0.0, 1.0), 0.0)
    return result


def _normalized_logdet(features: FloatArray) -> float:
    gram = features @ features.T
    sign, logdet = np.linalg.slogdet(np.eye(len(features), dtype=np.float64) + gram)
    if sign <= 0:  # pragma: no cover - I + X X^T is positive definite
        raise RuntimeError("diversity Gram matrix is not positive definite")
    denominator = len(features) * math.log(2.0)
    return float(np.clip(logdet / denominator, 0.0, 1.0))

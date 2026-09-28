"""Immutable provenance records for evolutionary diffusion search."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

_HEX_DIGITS = frozenset("0123456789abcdef")


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_immutable(value: object, *, field: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite floats")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_immutable(item, field=f"{field}[{index}]")
        return
    raise TypeError(f"{field} values must be immutable scalars or tuples")


def _finite_number(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _canonical_sequence_key(sequence: str) -> str:
    normalized = _normalized_sequence(sequence)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _validated_sequence_key(value: str, *, field: str = "sequence key") -> str:
    key = _identifier(value, field=field).lower()
    if len(key) != 64 or not set(key) <= _HEX_DIGITS:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return key


def _normalized_sequence(sequence: str) -> str:
    normalized = "".join(_identifier(sequence, field="sequence").split()).upper()
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("sequence must contain ASCII residue symbols") from error
    return normalized


def _unique_strings(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(_validated_sequence_key(value, field=field) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must be unique")
    return normalized


def _named_finite_values(
    values: tuple[tuple[str, float], ...], *, field: str
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple of name-value pairs")
    normalized: list[tuple[str, float]] = []
    for index, entry in enumerate(values):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{field} must be a tuple of name-value pairs")
        name, value = entry
        normalized.append(
            (
                _identifier(name, field=f"{field} name {index}"),
                _finite_number(value, field=f"{field} value {index}"),
            )
        )
    names = tuple(name for name, _ in normalized)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} names must be unique")
    return tuple(normalized)


def _named_immutable_values(
    values: tuple[tuple[str, object], ...], *, field: str
) -> tuple[tuple[str, object], ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple of name-value pairs")
    normalized: list[tuple[str, object]] = []
    for index, entry in enumerate(values):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{field} must be a tuple of name-value pairs")
        name, value = entry
        name = _identifier(name, field=f"{field} name {index}")
        _validate_immutable(value, field=f"{field} {name!r}")
        normalized.append((name, value))
    names = tuple(name for name, _ in normalized)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} names must be unique")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ProbabilityFactor:
    """One named conditional probability in an auditable sampling trace."""

    name: str
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="factor name"))
        if isinstance(self.probability, bool):
            raise ValueError("factor probability must be a finite number in [0, 1]")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("factor probability must be finite and in [0, 1]")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True)
class ProbabilityTrace:
    """Ordered conditional factors defining one exact logged probability."""

    factors: tuple[ProbabilityFactor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.factors, tuple) or not self.factors:
            raise ValueError("probability trace factors must be a non-empty tuple")
        if not all(isinstance(factor, ProbabilityFactor) for factor in self.factors):
            raise TypeError("probability trace entries must be ProbabilityFactor records")
        names = tuple(factor.name for factor in self.factors)
        if len(names) != len(set(names)):
            raise ValueError("probability factor names must be unique")

    @property
    def probability(self) -> float:
        """Return the product of the logged conditional factors."""

        return math.prod(factor.probability for factor in self.factors)

    @property
    def log_probability(self) -> float:
        """Return the summed log probability without multiplying tiny factors."""

        if any(factor.probability == 0.0 for factor in self.factors):
            return -math.inf
        return math.fsum(math.log(factor.probability) for factor in self.factors)

    def factor(self, name: str) -> float:
        """Return one factor by its stable ledger name."""

        for factor in self.factors:
            if factor.name == name:
                return factor.probability
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """One rollout's fixed Thompson draw, conditioning context, and seed."""

    rollout_id: str
    branch_id: str
    posterior_draw_id: str
    context: tuple[tuple[str, object], ...]
    policy_version: str
    seed: int

    def __post_init__(self) -> None:
        for field in ("rollout_id", "branch_id", "posterior_draw_id", "policy_version"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field=field))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.context, tuple):
            raise TypeError("context must be an immutable tuple of key-value pairs")
        normalized_context: list[tuple[str, object]] = []
        for index, entry in enumerate(self.context):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("context must be an immutable tuple of key-value pairs")
            key, value = entry
            key = _identifier(key, field=f"context key {index}")
            _validate_immutable(value, field=f"context {key!r}")
            normalized_context.append((key, value))
        keys = tuple(key for key, _ in normalized_context)
        if len(keys) != len(set(keys)):
            raise ValueError("context keys must be unique")
        object.__setattr__(self, "context", tuple(normalized_context))

    def context_value(self, name: str) -> object:
        """Read one immutable context value by name."""

        for key, value in self.context:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Logged inclusion decision conditioned on one ordered proposal set."""

    selected: bool
    propensity: ProbabilityTrace
    selection_set_id: str
    eligible_proposal_ids: tuple[str, ...]
    policy_version: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be a bool")
        if not isinstance(self.propensity, ProbabilityTrace):
            raise TypeError("propensity must be a ProbabilityTrace")
        object.__setattr__(
            self,
            "selection_set_id",
            _identifier(self.selection_set_id, field="selection_set_id"),
        )
        if not isinstance(self.eligible_proposal_ids, tuple):
            raise TypeError("eligible_proposal_ids must be a tuple")
        eligible_ids = tuple(
            _identifier(proposal_id, field="eligible proposal ID")
            for proposal_id in self.eligible_proposal_ids
        )
        if len(eligible_ids) != len(set(eligible_ids)):
            raise ValueError("eligible proposal IDs must be unique")
        object.__setattr__(self, "eligible_proposal_ids", eligible_ids)
        object.__setattr__(
            self,
            "policy_version",
            _identifier(self.policy_version, field="selection policy_version"),
        )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("selection seed must be a non-negative integer")
        log_probability = self.propensity.log_probability
        if self.selected and log_probability == -math.inf:
            raise ValueError("a selected proposal cannot have zero propensity")
        if not self.selected and log_probability == 0.0:
            raise ValueError("an unselected proposal cannot have unit propensity")
        if self.selected and not eligible_ids:
            raise ValueError("a selected proposal requires a non-empty eligible set")
        if not eligible_ids and log_probability != -math.inf:
            raise ValueError("an empty eligible set requires zero propensity")


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """One proposed sequence, retained regardless of validity or evaluation."""

    proposal_id: str
    rollout_id: str
    sequence: str
    hard_valid: bool
    rejection_reason: str | None
    selection: SelectionDecision
    policy_version: str
    proposal_round: int = 0
    niche_id: str | None = None
    cheap_predictions: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for field in ("proposal_id", "rollout_id", "policy_version"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field=field))
        object.__setattr__(self, "sequence", _normalized_sequence(self.sequence))
        if not isinstance(self.hard_valid, bool):
            raise TypeError("hard_valid must be a bool")
        if not isinstance(self.selection, SelectionDecision):
            raise TypeError("selection must be a SelectionDecision")
        if not isinstance(self.proposal_round, int) or isinstance(self.proposal_round, bool):
            raise TypeError("proposal_round must be an integer")
        if self.proposal_round < 0:
            raise ValueError("proposal_round must be non-negative")
        if self.niche_id is not None:
            object.__setattr__(self, "niche_id", _identifier(self.niche_id, field="niche_id"))
        if self.hard_valid:
            if self.rejection_reason is not None:
                raise ValueError("a valid proposal cannot have a rejection reason")
        else:
            if self.rejection_reason is None or not self.rejection_reason.strip():
                raise ValueError("an invalid proposal requires a rejection reason")
            object.__setattr__(
                self,
                "rejection_reason",
                _identifier(self.rejection_reason, field="rejection reason"),
            )
            if self.selection.selected:
                raise ValueError("an invalid proposal cannot be selected")
            if self.selection.propensity.log_probability != -math.inf:
                raise ValueError("an invalid proposal requires zero selection propensity")
            if self.proposal_id in self.selection.eligible_proposal_ids:
                raise ValueError("an invalid proposal cannot be selection-eligible")
        is_selection_eligible = self.proposal_id in self.selection.eligible_proposal_ids
        if self.selection.selected and not is_selection_eligible:
            raise ValueError("a selected proposal must belong to its eligible set")
        if not is_selection_eligible and self.selection.propensity.log_probability != -math.inf:
            raise ValueError("a non-eligible proposal requires zero selection propensity")
        object.__setattr__(
            self,
            "cheap_predictions",
            _named_finite_values(self.cheap_predictions, field="cheap predictions"),
        )

    @property
    def sequence_key(self) -> str:
        """Canonical key used for transposition lookup."""

        return _canonical_sequence_key(self.sequence)


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """One fully logged proposal edge, including every probability factor."""

    edge_id: str
    proposal_id: str
    rollout_id: str
    parent_sequence_keys: tuple[str, ...]
    operator: str
    edit_description: tuple[object, ...]
    proposal_trace: ProbabilityTrace
    behavior_log_probabilities: tuple[float, ...] = ()
    random_stream: int | None = None
    sample_index: int | None = None
    sampling_parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for field in ("edge_id", "proposal_id", "rollout_id", "operator"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field=field))
        parent_keys = _unique_strings(self.parent_sequence_keys, field="parent sequence keys")
        object.__setattr__(
            self,
            "parent_sequence_keys",
            parent_keys,
        )
        if not isinstance(self.edit_description, tuple):
            raise TypeError("edit_description must be an immutable tuple")
        _validate_immutable(self.edit_description, field="edit_description")
        if not isinstance(self.proposal_trace, ProbabilityTrace):
            raise TypeError("proposal_trace must be a ProbabilityTrace")
        if any(factor.probability == 0.0 for factor in self.proposal_trace.factors):
            raise ValueError("a realized edge requires positive probability factors")
        factor_names = {factor.name for factor in self.proposal_trace.factors}
        has_edit_probability = "edit" in factor_names or any(
            name.startswith("edit.") for name in factor_names
        )
        if not {"branch", "operator"}.issubset(factor_names) or not has_edit_probability:
            raise ValueError("proposal_trace must log branch, operator, and edit probabilities")
        if not isinstance(self.behavior_log_probabilities, tuple):
            raise TypeError("behavior_log_probabilities must be a tuple")
        log_probabilities = tuple(
            _finite_number(value, field="behavior log probability")
            for value in self.behavior_log_probabilities
        )
        if any(value > 0.0 for value in log_probabilities):
            raise ValueError("behavior log probabilities must be at most zero")
        if log_probabilities:
            raise ValueError(
                "behavior log probabilities require a typed diffusion trajectory record"
            )
        object.__setattr__(self, "behavior_log_probabilities", log_probabilities)
        if (self.random_stream is None) != (self.sample_index is None):
            raise ValueError("random_stream and sample_index must be logged together")
        for field in ("random_stream", "sample_index"):
            value = getattr(self, field)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer")
        parameters = _named_immutable_values(self.sampling_parameters, field="sampling_parameters")
        if self.random_stream is not None and not parameters:
            raise ValueError("seeded edges require sampling_parameters for replay")
        object.__setattr__(self, "sampling_parameters", parameters)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One immutable oracle or assay result for a logged proposal."""

    evaluation_id: str
    proposal_id: str
    fidelity: str
    evaluator_version: str
    cost: float
    outcomes: tuple[tuple[str, float], ...]
    batch_id: str
    replicate: int

    def __post_init__(self) -> None:
        for field in (
            "evaluation_id",
            "proposal_id",
            "fidelity",
            "evaluator_version",
            "batch_id",
        ):
            object.__setattr__(self, field, _identifier(getattr(self, field), field=field))
        cost = _finite_number(self.cost, field="cost")
        if cost < 0.0:
            raise ValueError("cost must be non-negative")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "outcomes", _named_finite_values(self.outcomes, field="outcomes"))
        if not isinstance(self.replicate, int) or isinstance(self.replicate, bool):
            raise TypeError("replicate must be an integer")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")

    def outcome(self, name: str) -> float:
        """Read one measured outcome by name."""

        for outcome_name, value in self.outcomes:
            if outcome_name == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class BranchRecord:
    """Persistent frontier state used by the auditable branch allocator."""

    branch_id: str
    niche: tuple[object, ...]
    elite_sequence_keys: tuple[str, ...]
    parent_branch_id: str | None
    policy_version: str
    kl_radius: float
    visits: int
    unique_evaluated_descendants: int
    yield_mean: float
    yield_std: float
    credit: float
    credit_floor: float
    credit_ceiling: float
    credit_decay: float

    def __post_init__(self) -> None:
        for field in ("branch_id", "policy_version"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field=field))
        if self.parent_branch_id is not None:
            object.__setattr__(
                self,
                "parent_branch_id",
                _identifier(self.parent_branch_id, field="parent_branch_id"),
            )
        if not isinstance(self.niche, tuple):
            raise TypeError("niche must be an immutable tuple")
        _validate_immutable(self.niche, field="niche")
        elite_keys = _unique_strings(self.elite_sequence_keys, field="elite sequence keys")
        object.__setattr__(
            self,
            "elite_sequence_keys",
            elite_keys,
        )
        kl_radius = _finite_number(self.kl_radius, field="kl_radius")
        if kl_radius < 0.0:
            raise ValueError("kl_radius must be non-negative")
        object.__setattr__(self, "kl_radius", kl_radius)
        for field in ("visits", "unique_evaluated_descendants"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in (
            "yield_mean",
            "yield_std",
            "credit",
            "credit_floor",
            "credit_ceiling",
            "credit_decay",
        ):
            object.__setattr__(self, field, _finite_number(getattr(self, field), field=field))
        if self.yield_std < 0.0:
            raise ValueError("yield_std must be non-negative")
        if not 0.0 < self.credit_floor <= 1.0 <= self.credit_ceiling:
            raise ValueError("credit bounds must satisfy 0 < floor <= 1 <= ceiling")
        if not self.credit_floor <= self.credit <= self.credit_ceiling:
            raise ValueError("credit must lie within its bounds")
        if not 0.0 <= self.credit_decay <= 1.0:
            raise ValueError("credit_decay must lie in [0, 1]")

"""Frozen component mixtures with a whole-output total-variation bound.

For any fixed complete-peptide distributions P and Q, accepting
P_new = (1-alpha) P + alpha Q implies TV(P_new, P) <= alpha.
This holds independently of unknown peptide probabilities or sampled probes.
It requires choosing ONE component for the ENTIRE generation trajectory.
Choosing again at each diffusion step does not implement this contract.

The adapter specifies and samples immutable component identities, not models.
Providers MUST bind each identity to frozen weights, preprocessing, decoding,
conditioning, and all generation settings. Mutating an old component invalidates
the bound. No pruning, compression, or native ensemble integration occurs here.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import numpy as np

MAXIMUM_UPDATE_MASS = Fraction(1, 20)


def _fraction(value: Fraction | str | int | float) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("mixture mass cannot be boolean")
    if not isinstance(value, Fraction | str | int | float):
        raise ValueError("mixture mass must be an exact fraction or numeric value")
    try:
        # Decimal text retains intended .05 boundary instead of binary float noise.
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("mixture mass must be finite and rational") from error


@dataclass(frozen=True)
class FrozenPolicyMixture:
    """Exact declared weights over frozen complete-trajectory providers."""

    component_ids: tuple[str, ...]
    weights: tuple[Fraction, ...]

    def __post_init__(self):
        ids = tuple(self.component_ids)
        weights = tuple(_fraction(value) for value in self.weights)
        if not ids or len(ids) != len(weights):
            raise ValueError("nonempty component IDs and weights must have equal lengths")
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("each frozen component requires a nonempty identity")
        if len(set(ids)) != len(ids):
            raise ValueError("component IDs must be unique")
        if any(value <= 0 for value in weights) or sum(weights) != 1:
            raise ValueError("weights must be positive and sum exactly to one")
        object.__setattr__(self, "component_ids", ids)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def singleton(cls, component_id: str):
        return cls((component_id,), (Fraction(1),))

    def record(self) -> dict:
        return {
            "contract": "frozen_complete_trajectory_policy_mixture_v1",
            "components": [
                {
                    "component_id": name,
                    "weight_numerator": weight.numerator,
                    "weight_denominator": weight.denominator,
                }
                for name, weight in zip(self.component_ids, self.weights, strict=True)
            ],
            "sampling_scope": "one_frozen_component_per_complete_peptide_trajectory",
            "provider_freezing_required": True,
        }

    def sample_component(self, rng: np.random.Generator) -> str:
        """Choose ONCE, then pass this identity through the entire trajectory.

        Caller owns the deterministic random generator. Rejection sampling on
        raw random bits implements the exact rational weights without rounding
        tiny component masses away or silently renormalizing them.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a caller-owned numpy Generator")
        denominator = lcm(*(weight.denominator for weight in self.weights))
        if denominator == 1:
            return self.component_ids[0]
        bits = (denominator - 1).bit_length()
        while True:
            draw = 0
            for _ in range((bits + 63) // 64):
                draw = (draw << 64) | int(
                    rng.integers(0, np.iinfo(np.uint64).max, endpoint=True, dtype=np.uint64)
                )
            draw &= (1 << bits) - 1
            if draw < denominator:
                break
        cumulative = 0
        for name, weight in zip(self.component_ids, self.weights, strict=True):
            cumulative += weight.numerator * (denominator // weight.denominator)
            if draw < cumulative:
                return name
        raise RuntimeError("exact mixture sampling failed to cover [0,1)")


@dataclass(frozen=True)
class MixtureUpdate:
    previous: FrozenPolicyMixture
    accepted: FrozenPolicyMixture
    proposal_component_id: str
    alpha: Fraction

    def __post_init__(self):
        mass = _fraction(self.alpha)
        if not 0 <= mass <= MAXIMUM_UPDATE_MASS:
            raise ValueError("update mass must lie in [0,.05]")
        expected = {
            name: (1 - mass) * weight
            for name, weight in zip(self.previous.component_ids, self.previous.weights, strict=True)
        }
        if mass:
            expected[self.proposal_component_id] = (
                expected.get(self.proposal_component_id, 0) + mass
            )
        actual = dict(zip(self.accepted.component_ids, self.accepted.weights, strict=True))
        if actual != expected:
            raise ValueError("accepted policy does not match the claimed bounded mixture")
        object.__setattr__(self, "alpha", mass)

    @property
    def global_total_variation_upper_bound(self) -> Fraction:
        # Conservative even when the proposal is an existing or equivalent policy.
        return self.alpha

    def record(self) -> dict:
        return {
            "previous_policy": self.previous.record(),
            "accepted_policy": self.accepted.record(),
            "proposal_component_id": self.proposal_component_id,
            "alpha_numerator": self.alpha.numerator,
            "alpha_denominator": self.alpha.denominator,
            "global_total_variation_upper_bound": str(self.alpha),
            "bound_scope": "each_accepted_complete_peptide_distribution_update",
            "proof": "TV((1-alpha)*P+alpha*Q,P)=alpha*TV(Q,P)<=alpha",
            "conditional_on": "unchanged_component_generation_distributions",
            "metric_preference_selected": False,
            "native_sampler_connected": False,
        }


def accept_mixture_update(
    previous: FrozenPolicyMixture,
    proposal_component_id: str,
    alpha: Fraction | str | int | float = MAXIMUM_UPDATE_MASS,
) -> MixtureUpdate:
    """Accept a frozen whole-trajectory proposal with at most 5% mixture mass.

    A separate distance treatment may choose a smaller alpha. No metric threshold
    is inferred from this bound, and clipping of training remains independent.
    Repeated updates can drift further than 5% from the ORIGINAL policy.
    """
    if not isinstance(previous, FrozenPolicyMixture):
        raise TypeError("previous must be a FrozenPolicyMixture")
    if not isinstance(proposal_component_id, str) or not proposal_component_id.strip():
        raise ValueError("proposal requires a nonempty immutable component identity")
    mass = _fraction(alpha)
    if not 0 <= mass <= MAXIMUM_UPDATE_MASS:
        raise ValueError("each accepted update alpha must be in [0, .05]")
    if mass == 0:
        return MixtureUpdate(previous, previous, proposal_component_id, mass)
    components = {
        name: (1 - mass) * weight
        for name, weight in zip(previous.component_ids, previous.weights, strict=True)
    }
    components[proposal_component_id] = components.get(proposal_component_id, Fraction(0)) + mass
    accepted = FrozenPolicyMixture(tuple(components), tuple(components.values()))
    return MixtureUpdate(previous, accepted, proposal_component_id, mass)

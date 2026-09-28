"""Small, seeded genetic emitters with exact atomic probability traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.generators.search.records import (
    EdgeRecord,
    ProbabilityFactor,
    ProbabilityTrace,
    RolloutRecord,
    _canonical_sequence_key,
    _normalized_sequence,
    _validate_immutable,
    _validated_sequence_key,
)

WeightInput = Sequence[float] | np.ndarray

_SUBSTITUTION_STREAM_KEY = 1
_PARTIAL_REMASK_STREAM_KEY = 2
_PCG64_PERIOD = 2**128
EMITTER_RNG_ALGORITHM = "PCG64"
EMITTER_SEED_SEQUENCE_IMPLEMENTATION = (
    "numpy-seedsequence-entropy-tuple-rollout-seed-operator-key-random-stream-v1"
)
SUBSTITUTION_DRAW_LAYOUT_IMPLEMENTATION = (
    "pcg64-nonwrapping-advance-two-times-sample-start-candidate-major-"
    "position-then-residue-two-float64-draws-v2"
)
PARTIAL_REMASK_DRAW_LAYOUT_IMPLEMENTATION = (
    "pcg64-nonwrapping-advance-two-times-remask-count-times-sample-start-"
    "candidate-major-step-major-position-then-token-two-float64-draws-v2"
)


def _validate_alphabet(alphabet: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(alphabet, tuple) or not alphabet:
        raise ValueError("alphabet must be a non-empty tuple")
    normalized: list[str] = []
    for token in alphabet:
        if not isinstance(token, str) or len(token.strip()) != 1:
            raise ValueError("alphabet tokens must be single non-whitespace characters")
        normalized.append(token.strip().upper())
    if len(normalized) != len(set(normalized)):
        raise ValueError("alphabet tokens must be unique")
    return tuple(normalized)


def _validate_parent(parent: str, alphabet: tuple[str, ...]) -> str:
    parent = _normalized_sequence(parent)
    unknown = set(parent).difference(alphabet)
    if unknown:
        raise ValueError(f"parent contains tokens outside the alphabet: {sorted(unknown)}")
    return parent


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")


def _validate_sample_start(sample_start: int) -> None:
    if not isinstance(sample_start, int) or isinstance(sample_start, bool) or sample_start < 0:
        raise ValueError("sample_start must be a non-negative integer")


def _validate_non_aliasing_draw_window(
    *,
    sample_start: int,
    count: int,
    draws_per_sample: int,
) -> None:
    """Reject logical sample windows that wrap around PCG64's finite period."""

    if draws_per_sample <= 0:
        raise ValueError("draws_per_sample must be positive")
    draw_start = sample_start * draws_per_sample
    draw_count = count * draws_per_sample
    if draw_start >= _PCG64_PERIOD or draw_count > _PCG64_PERIOD - draw_start:
        raise ValueError("sample window exceeds the non-aliasing PCG64 period")


def _rng_for_rollout(
    rollout: RolloutRecord, *, random_stream: int, operator_key: int
) -> np.random.Generator:
    if not isinstance(random_stream, int) or isinstance(random_stream, bool) or random_stream < 0:
        raise ValueError("random_stream must be a non-negative integer")
    seed = np.random.SeedSequence((rollout.seed, operator_key, random_stream))
    # Offset-stable batching below relies on PCG64's documented ``advance``
    # operation and one 64-bit raw draw per generated float64. Pinning the bit
    # generator keeps replay stable if NumPy changes ``default_rng`` later.
    return np.random.Generator(np.random.PCG64(seed))


def _array_tuple(values: np.ndarray) -> tuple[object, ...]:
    if values.ndim == 1:
        return tuple(float(value) for value in values)
    return tuple(tuple(float(value) for value in row) for row in values)


def _sampling_value(edge: EdgeRecord, name: str) -> object:
    for parameter_name, value in edge.sampling_parameters:
        if parameter_name == name:
            return value
    raise ValueError(f"edge is missing sampling parameter {name!r}")


def _immutable_values_are_exactly_equal(left: object, right: object) -> bool:
    """Compare nested immutable provenance without Python numeric coercion."""

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


def _rollout_identity(rollout: RolloutRecord) -> tuple[object, ...]:
    """Return replay identity with its potentially 128-bit seed JSON-safe."""

    return (
        rollout.rollout_id,
        rollout.branch_id,
        rollout.posterior_draw_id,
        rollout.context,
        rollout.policy_version,
        str(rollout.seed),
    )


def _emitter_replay_contract(
    *,
    operator_key: int,
    draw_layout_implementation: str,
) -> tuple[tuple[str, object], ...]:
    return (
        ("emitter_rng_algorithm", EMITTER_RNG_ALGORITHM),
        ("emitter_seed_sequence_implementation", EMITTER_SEED_SEQUENCE_IMPLEMENTATION),
        ("emitter_draw_layout_implementation", draw_layout_implementation),
        ("emitter_operator_key", operator_key),
    )


def _validate_emitter_replay_contract(
    edge: EdgeRecord,
    *,
    operator_key: int,
    draw_layout_implementation: str,
) -> None:
    for name, expected in _emitter_replay_contract(
        operator_key=operator_key,
        draw_layout_implementation=draw_layout_implementation,
    ):
        if not _immutable_values_are_exactly_equal(
            _sampling_value(edge, name),
            expected,
        ):
            raise ValueError(f"edge {name} does not match this emitter implementation")


def validate_seeded_emitter_edge_replay_contract(
    *,
    edge: EdgeRecord,
    rollout: RolloutRecord,
) -> None:
    """Join one seeded edge to its exact rollout and pinned emitter contract."""

    if not isinstance(edge, EdgeRecord):
        raise TypeError("edge must be an EdgeRecord")
    if not isinstance(rollout, RolloutRecord):
        raise TypeError("rollout must be a RolloutRecord")
    if edge.random_stream is None or edge.sample_index is None:
        raise ValueError("edge does not contain seeded replay provenance")
    if edge.rollout_id != rollout.rollout_id:
        raise ValueError("edge and replay rollout IDs must match")
    if not _immutable_values_are_exactly_equal(
        _sampling_value(edge, "rollout_identity"),
        _rollout_identity(rollout),
    ):
        raise ValueError("edge and replay rollout identity must match")
    if edge.operator == "substitution":
        operator_key = _SUBSTITUTION_STREAM_KEY
        draw_layout = SUBSTITUTION_DRAW_LAYOUT_IMPLEMENTATION
    elif edge.operator == "partial_remask":
        operator_key = _PARTIAL_REMASK_STREAM_KEY
        draw_layout = PARTIAL_REMASK_DRAW_LAYOUT_IMPLEMENTATION
    else:
        raise ValueError(f"seeded edge operator {edge.operator!r} has no declared replay contract")
    _validate_emitter_replay_contract(
        edge,
        operator_key=operator_key,
        draw_layout_implementation=draw_layout,
    )


def _validate_replay_provenance(*, parent: str, rollout: RolloutRecord, edge: EdgeRecord) -> None:
    if edge.rollout_id != rollout.rollout_id:
        raise ValueError("edge and replay rollout IDs must match")
    if not _immutable_values_are_exactly_equal(
        _sampling_value(edge, "rollout_identity"),
        _rollout_identity(rollout),
    ):
        raise ValueError("edge and replay rollout identity must match")
    if len(edge.parent_sequence_keys) != 1:
        raise ValueError("single-parent emitter edge must log exactly one parent")
    if _canonical_sequence_key(parent) != edge.parent_sequence_keys[0]:
        raise ValueError("edge and replay parent sequences must match")


def _validate_replayed_emission(emission: EmittedProposal, edge: EdgeRecord) -> EmittedProposal:
    logged_edit_factors = tuple(
        ProbabilityFactor(factor.name.removeprefix("edit."), factor.probability)
        for factor in edge.proposal_trace.factors
        if factor.name.startswith("edit.")
    )
    if (
        not _immutable_values_are_exactly_equal(
            emission.edit_description,
            edge.edit_description,
        )
        or not logged_edit_factors
        or emission.edit_trace.factors != logged_edit_factors
        or not _immutable_values_are_exactly_equal(
            emission.sampling_parameters,
            edge.sampling_parameters,
        )
    ):
        raise ValueError("replayed emission does not match the logged edge")
    return emission


def _weight_vector(weights: WeightInput | None, *, size: int, field: str) -> np.ndarray:
    values = np.ones(size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{field} must have shape ({size},)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{field} must contain finite non-negative weights")
    if float(np.max(values)) <= 0.0:
        raise ValueError(f"{field} must have positive total weight")
    return values.copy()


def _normalized_vector(weights: WeightInput | None, *, size: int, field: str) -> np.ndarray:
    values = _weight_vector(weights, size=size, field=field)
    scale = float(np.max(values))
    scaled = values / scale
    return scaled / float(scaled.sum())


def _token_weight_matrix(
    weights: WeightInput | None,
    *,
    length: int,
    alphabet_size: int,
    field: str,
) -> np.ndarray:
    if weights is None:
        values = np.ones((length, alphabet_size), dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape == (alphabet_size,):
            values = np.broadcast_to(values, (length, alphabet_size)).copy()
        elif values.shape != (length, alphabet_size):
            raise ValueError(
                f"{field} must have shape ({alphabet_size},) or ({length}, {alphabet_size})"
            )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{field} must contain finite non-negative weights")
    if np.any(np.max(values, axis=1) <= 0.0):
        raise ValueError(f"each row of {field} must have positive total weight")
    return values


@dataclass(frozen=True, slots=True)
class EmittedProposal:
    """One emitter result tied to a rollout's fixed posterior draw and context."""

    sequence: str
    parent_sequence_key: str
    operator: str
    edit_description: tuple[tuple[object, ...], ...]
    edit_trace: ProbabilityTrace
    rollout: RolloutRecord
    random_stream: int
    sample_index: int
    sampling_parameters: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _normalized_sequence(self.sequence))
        object.__setattr__(
            self,
            "parent_sequence_key",
            _validated_sequence_key(self.parent_sequence_key, field="parent_sequence_key"),
        )
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise ValueError("operator must be a non-empty string")
        object.__setattr__(self, "operator", self.operator.strip())
        if not isinstance(self.edit_description, tuple):
            raise TypeError("edit_description must be an immutable tuple")
        _validate_immutable(self.edit_description, field="edit_description")
        if not isinstance(self.edit_trace, ProbabilityTrace):
            raise TypeError("edit_trace must be a ProbabilityTrace")
        if not isinstance(self.rollout, RolloutRecord):
            raise TypeError("rollout must be a RolloutRecord")
        for field in ("random_stream", "sample_index"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(self.sampling_parameters, tuple):
            raise TypeError("sampling_parameters must be an immutable tuple")
        _validate_immutable(self.sampling_parameters, field="sampling_parameters")

    @property
    def rollout_id(self) -> str:
        return self.rollout.rollout_id

    @property
    def branch_id(self) -> str:
        return self.rollout.branch_id

    @property
    def posterior_draw_id(self) -> str:
        return self.rollout.posterior_draw_id

    @property
    def context(self) -> tuple[tuple[str, object], ...]:
        return self.rollout.context

    @property
    def policy_version(self) -> str:
        return self.rollout.policy_version

    def to_edge_record(
        self,
        *,
        edge_id: str,
        proposal_id: str,
        proposal_prefix: ProbabilityTrace,
        behavior_log_probabilities: tuple[float, ...] = (),
    ) -> EdgeRecord:
        """Create a replayable edge by prefixing branch/operator probabilities."""

        if not isinstance(proposal_prefix, ProbabilityTrace):
            raise TypeError("proposal_prefix must be a ProbabilityTrace")
        edit_factors = tuple(
            ProbabilityFactor(f"edit.{factor.name}", factor.probability)
            for factor in self.edit_trace.factors
        )
        return EdgeRecord(
            edge_id=edge_id,
            proposal_id=proposal_id,
            rollout_id=self.rollout_id,
            parent_sequence_keys=(self.parent_sequence_key,),
            operator=self.operator,
            edit_description=self.edit_description,
            proposal_trace=ProbabilityTrace(proposal_prefix.factors + edit_factors),
            behavior_log_probabilities=behavior_log_probabilities,
            random_stream=self.random_stream,
            sample_index=self.sample_index,
            sampling_parameters=self.sampling_parameters,
        )


@dataclass(frozen=True, slots=True)
class SubstitutionEmitter:
    """Sample single substitutions, excluding the parent's current residue."""

    alphabet: tuple[str, ...] = STANDARD_AMINO_ACIDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphabet", _validate_alphabet(self.alphabet))

    def emit(
        self,
        *,
        parent: str,
        rollout: RolloutRecord,
        count: int,
        sample_start: int = 0,
        random_stream: int = 0,
        position_weights: WeightInput | None = None,
        residue_weights: WeightInput | None = None,
    ) -> tuple[EmittedProposal, ...]:
        """Draw single-edit candidates with their conditional edit probability."""

        _validate_count(count)
        _validate_sample_start(sample_start)
        _validate_non_aliasing_draw_window(
            sample_start=sample_start,
            count=count,
            draws_per_sample=2,
        )
        rng = _rng_for_rollout(
            rollout,
            random_stream=random_stream,
            operator_key=_SUBSTITUTION_STREAM_KEY,
        )
        parent = _validate_parent(parent, self.alphabet)
        raw_position_weights = _weight_vector(
            position_weights, size=len(parent), field="position_weights"
        )
        position_probabilities = _normalized_vector(
            raw_position_weights,
            size=len(parent),
            field="position_weights",
        )
        residue_matrix = _token_weight_matrix(
            residue_weights,
            length=len(parent),
            alphabet_size=len(self.alphabet),
            field="residue_weights",
        )
        token_indices = {token: index for index, token in enumerate(self.alphabet)}
        conditional_tokens: list[np.ndarray] = []
        for position, current_token in enumerate(parent):
            weights = residue_matrix[position].copy()
            weights[token_indices[current_token]] = 0.0
            if float(np.max(weights)) <= 0.0:
                raise ValueError(
                    "residue_weights must give every position a positive-weight alternative"
                )
            conditional_tokens.append(
                _normalized_vector(
                    weights,
                    size=len(self.alphabet),
                    field="residue_weights",
                )
            )
        sampling_parameters = (
            *_emitter_replay_contract(
                operator_key=_SUBSTITUTION_STREAM_KEY,
                draw_layout_implementation=SUBSTITUTION_DRAW_LAYOUT_IMPLEMENTATION,
            ),
            ("alphabet", self.alphabet),
            ("rollout_identity", _rollout_identity(rollout)),
            ("position_weights", _array_tuple(raw_position_weights)),
            ("residue_weights", _array_tuple(residue_matrix)),
        )

        # Draw the independent stochastic decisions as dense arrays.  Object
        # construction remains a small provenance loop, while the sampling
        # path scales to cluster-sized proposal batches without Python RNG
        # calls for every peptide.
        # A fixed stream plus an absolute sample offset makes logical results
        # invariant to compute-batch boundaries. PCG64 consumes one raw draw
        # for each float64 generated here.
        rng.bit_generator.advance(sample_start * 2)
        uniforms = rng.random((count, 2))
        position_cdf = np.cumsum(position_probabilities)
        position_cdf[-1] = 1.0
        sampled_positions = np.sum(
            uniforms[:, :1] >= position_cdf[None, :],
            axis=1,
            dtype=np.int64,
        )
        token_probability_matrix = np.asarray(conditional_tokens, dtype=np.float64)
        token_cdf = np.cumsum(token_probability_matrix, axis=1)
        token_cdf[:, -1] = 1.0
        sampled_tokens = np.sum(
            uniforms[:, 1:] >= token_cdf[sampled_positions],
            axis=1,
            dtype=np.int64,
        )

        candidates: list[EmittedProposal] = []
        for local_index, (raw_position, raw_token_index) in enumerate(
            zip(sampled_positions, sampled_tokens, strict=True)
        ):
            sample_index = sample_start + local_index
            position = int(raw_position)
            token_index = int(raw_token_index)
            token_probability = conditional_tokens[position]
            replacement = self.alphabet[token_index]
            sequence = f"{parent[:position]}{replacement}{parent[position + 1 :]}"
            candidates.append(
                EmittedProposal(
                    sequence=sequence,
                    parent_sequence_key=_canonical_sequence_key(parent),
                    operator="substitution",
                    edit_description=((position, parent[position], replacement),),
                    edit_trace=ProbabilityTrace(
                        (
                            ProbabilityFactor("position", position_probabilities[position]),
                            ProbabilityFactor("residue", token_probability[token_index]),
                        )
                    ),
                    rollout=rollout,
                    random_stream=random_stream,
                    sample_index=sample_index,
                    sampling_parameters=sampling_parameters,
                )
            )
        return tuple(candidates)

    def replay(self, *, parent: str, rollout: RolloutRecord, edge: EdgeRecord) -> EmittedProposal:
        """Replay a stored edge from its rollout seed and sampling configuration."""

        if edge.operator != "substitution":
            raise ValueError("edge operator is not substitution")
        _validate_replay_provenance(parent=parent, rollout=rollout, edge=edge)
        if edge.random_stream is None or edge.sample_index is None:
            raise ValueError("edge does not contain seeded replay provenance")
        validate_seeded_emitter_edge_replay_contract(
            edge=edge,
            rollout=rollout,
        )
        if _sampling_value(edge, "alphabet") != self.alphabet:
            raise ValueError("edge alphabet does not match this emitter")
        stored_position_weights = _sampling_value(edge, "position_weights")
        stored_residue_weights = _sampling_value(edge, "residue_weights")
        emission = self.emit(
            parent=parent,
            rollout=rollout,
            count=1,
            sample_start=edge.sample_index,
            random_stream=edge.random_stream,
            position_weights=stored_position_weights,  # type: ignore[arg-type]
            residue_weights=stored_residue_weights,  # type: ignore[arg-type]
        )[0]
        return _validate_replayed_emission(emission, edge)


@dataclass(frozen=True, slots=True)
class PartialRemaskEmitter:
    """Sample an ordered remask/denoise path without replacement."""

    alphabet: tuple[str, ...] = STANDARD_AMINO_ACIDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphabet", _validate_alphabet(self.alphabet))

    def emit(
        self,
        *,
        parent: str,
        rollout: RolloutRecord,
        count: int,
        remask_count: int,
        sample_start: int = 0,
        random_stream: int = 0,
        position_weights: WeightInput | None = None,
        token_weights: WeightInput | None = None,
    ) -> tuple[EmittedProposal, ...]:
        """Draw candidates and log each ordered position/token choice."""

        _validate_count(count)
        _validate_sample_start(sample_start)
        parent = _validate_parent(parent, self.alphabet)
        if (
            not isinstance(remask_count, int)
            or isinstance(remask_count, bool)
            or not 1 <= remask_count <= len(parent)
        ):
            raise ValueError("remask_count must lie between one and parent length")
        _validate_non_aliasing_draw_window(
            sample_start=sample_start,
            count=count,
            draws_per_sample=2 * remask_count,
        )
        rng = _rng_for_rollout(
            rollout,
            random_stream=random_stream,
            operator_key=_PARTIAL_REMASK_STREAM_KEY,
        )
        raw_position_weights = _weight_vector(
            position_weights, size=len(parent), field="position_weights"
        )
        position_probabilities = _normalized_vector(
            raw_position_weights,
            size=len(parent),
            field="position_weights",
        )
        if int(np.count_nonzero(position_probabilities)) < remask_count:
            raise ValueError("remask_count exceeds the positive-weight positions")
        token_matrix = _token_weight_matrix(
            token_weights,
            length=len(parent),
            alphabet_size=len(self.alphabet),
            field="token_weights",
        )
        token_scale = np.max(token_matrix, axis=1, keepdims=True)
        scaled_tokens = token_matrix / token_scale
        token_probabilities = scaled_tokens / scaled_tokens.sum(axis=1, keepdims=True)
        sampling_parameters = (
            *_emitter_replay_contract(
                operator_key=_PARTIAL_REMASK_STREAM_KEY,
                draw_layout_implementation=PARTIAL_REMASK_DRAW_LAYOUT_IMPLEMENTATION,
            ),
            ("alphabet", self.alphabet),
            ("rollout_identity", _rollout_identity(rollout)),
            ("remask_count", remask_count),
            ("position_weights", _array_tuple(raw_position_weights)),
            ("token_weights", _array_tuple(token_matrix)),
        )

        # Preserve the candidate-major replay stream while drawing the full
        # stochastic batch in one call.  The remask steps remain sequential
        # because sampling is without replacement, but every step is
        # vectorized over candidates.
        rng.bit_generator.advance(sample_start * remask_count * 2)
        uniforms = rng.random((count, remask_count, 2))
        available_positions = np.broadcast_to(position_probabilities, (count, len(parent))).copy()
        sampled_positions = np.empty((count, remask_count), dtype=np.int64)
        sampled_position_probabilities = np.empty((count, remask_count), dtype=np.float64)
        sampled_tokens = np.empty((count, remask_count), dtype=np.int64)
        sampled_token_probabilities = np.empty((count, remask_count), dtype=np.float64)
        rows = np.arange(count)

        token_cdf = np.cumsum(token_probabilities, axis=1)
        token_cdf[:, -1] = 1.0
        for step in range(remask_count):
            conditional_positions = available_positions / available_positions.sum(
                axis=1,
                keepdims=True,
            )
            position_cdf = np.cumsum(conditional_positions, axis=1)
            position_cdf[:, -1] = 1.0
            positions = np.sum(
                uniforms[:, step, :1] >= position_cdf,
                axis=1,
                dtype=np.int64,
            )
            selected_token_cdf = token_cdf[positions]
            tokens = np.sum(
                uniforms[:, step, 1:] >= selected_token_cdf,
                axis=1,
                dtype=np.int64,
            )
            sampled_positions[:, step] = positions
            sampled_position_probabilities[:, step] = conditional_positions[rows, positions]
            sampled_tokens[:, step] = tokens
            sampled_token_probabilities[:, step] = token_probabilities[positions, tokens]
            available_positions[rows, positions] = 0.0

        candidates: list[EmittedProposal] = []
        for local_index in range(count):
            sample_index = sample_start + local_index
            sequence = list(parent)
            edits: list[tuple[object, ...]] = []
            factors = [ProbabilityFactor("remask_count", 1.0)]
            for step in range(remask_count):
                position = int(sampled_positions[local_index, step])
                token_index = int(sampled_tokens[local_index, step])
                replacement = self.alphabet[token_index]
                edits.append((position, parent[position], replacement))
                sequence[position] = replacement
                factors.extend(
                    (
                        ProbabilityFactor(
                            f"position_{step}",
                            sampled_position_probabilities[local_index, step],
                        ),
                        ProbabilityFactor(
                            f"token_{step}",
                            sampled_token_probabilities[local_index, step],
                        ),
                    )
                )
            candidates.append(
                EmittedProposal(
                    sequence="".join(sequence),
                    parent_sequence_key=_canonical_sequence_key(parent),
                    operator="partial_remask",
                    edit_description=tuple(edits),
                    edit_trace=ProbabilityTrace(tuple(factors)),
                    rollout=rollout,
                    random_stream=random_stream,
                    sample_index=sample_index,
                    sampling_parameters=sampling_parameters,
                )
            )
        return tuple(candidates)

    def replay(self, *, parent: str, rollout: RolloutRecord, edge: EdgeRecord) -> EmittedProposal:
        """Replay a stored remask path from its rollout seed and configuration."""

        if edge.operator != "partial_remask":
            raise ValueError("edge operator is not partial_remask")
        _validate_replay_provenance(parent=parent, rollout=rollout, edge=edge)
        if edge.random_stream is None or edge.sample_index is None:
            raise ValueError("edge does not contain seeded replay provenance")
        validate_seeded_emitter_edge_replay_contract(
            edge=edge,
            rollout=rollout,
        )
        if _sampling_value(edge, "alphabet") != self.alphabet:
            raise ValueError("edge alphabet does not match this emitter")
        remask_count = _sampling_value(edge, "remask_count")
        if not isinstance(remask_count, int) or isinstance(remask_count, bool):
            raise ValueError("stored remask_count must be an integer")
        stored_position_weights = _sampling_value(edge, "position_weights")
        stored_token_weights = _sampling_value(edge, "token_weights")
        emission = self.emit(
            parent=parent,
            rollout=rollout,
            count=1,
            remask_count=remask_count,
            sample_start=edge.sample_index,
            random_stream=edge.random_stream,
            position_weights=stored_position_weights,  # type: ignore[arg-type]
            token_weights=stored_token_weights,  # type: ignore[arg-type]
        )[0]
        return _validate_replayed_emission(emission, edge)

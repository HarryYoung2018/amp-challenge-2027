"""Batched, root-sampled Thompson rollouts for coherent search directions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from amp_challenge.generators.search.records import RolloutRecord
from amp_challenge.models.posterior import JointGaussianPosterior

FloatArray = NDArray[np.float64]
FROZEN_ROLLOUT_MANIFEST_SCHEMA_VERSION = 1
FROZEN_ROLLOUT_RNG_ALGORITHM = "PCG64"
FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION = (
    "amp-joint-gaussian-per-seed-pcg64-standard-normal-correlation-eigh-factor-float64-matvec-v1"
)
JSON_SAFE_INTEGER_MAX = 2**53 - 1


def _canonical_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _json_safe_integer(value: int, *, name: str) -> int:
    if not -JSON_SAFE_INTEGER_MAX <= value <= JSON_SAFE_INTEGER_MAX:
        raise ValueError(f"{name} must lie in the exactly representable JSON integer range")
    return value


def _manifest_value(value: object) -> object:
    """Convert one immutable context value to an exact JSON-compatible value."""

    if value is None or isinstance(value, str) or type(value) is bool:
        return value
    if isinstance(value, int):
        return _json_safe_integer(value, name="rollout context integer")
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("rollout context floats must be finite")
        return value
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    raise TypeError("rollout context contains a non-manifest value")


def _immutable_manifest_value(value: object) -> object:
    """Recover immutable tuple structure from a parsed JSON context value."""

    if value is None or isinstance(value, str) or type(value) is bool:
        return value
    if isinstance(value, int):
        return _json_safe_integer(value, name="manifest rollout context integer")
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("manifest rollout context floats must be finite")
        return value
    if isinstance(value, list):
        return tuple(_immutable_manifest_value(item) for item in value)
    raise TypeError("rollout manifest context contains an unsupported value")


def _validate_manifest_float_tree(
    value: object,
    *,
    name: str,
    depth: int,
    canonical_zero: bool = False,
) -> None:
    """Require JSON arrays with exact finite float leaves before NumPy coercion."""

    if depth > 0:
        if type(value) is not list or not value:
            raise TypeError(f"{name} must be a non-empty JSON list at every array axis")
        for item in value:
            _validate_manifest_float_tree(
                item,
                name=name,
                depth=depth - 1,
                canonical_zero=canonical_zero,
            )
        return
    if type(value) is not float:
        raise TypeError(f"{name} must contain exact JSON float leaves")
    if not np.isfinite(value):
        raise ValueError(f"{name} must contain only finite values")
    if canonical_zero and value == 0.0 and np.signbit(value):
        raise ValueError(f"{name} must encode zero canonically")


def _float_tuple_is_exactly_equal(
    value: object,
    expected: tuple[float, ...],
) -> bool:
    """Compare a provenance float tuple without bool/int or signed-zero coercion."""

    if type(value) is not tuple or len(value) != len(expected):
        return False
    if any(type(item) is not float for item in value):
        return False
    return np.asarray(value, dtype="<f8").tobytes(order="C") == np.asarray(
        expected, dtype="<f8"
    ).tobytes(order="C")


def _readonly_floats(value: object, *, name: str, ndim: int) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _update_framed(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _update_canonical_value(digest: Any, value: object) -> None:
    """Hash an immutable value with explicit, language-neutral type framing."""

    if value is None:
        _update_framed(digest, b"null")
    elif type(value) is bool:
        _update_framed(digest, b"bool:1" if value else b"bool:0")
    elif isinstance(value, str):
        _update_framed(digest, b"str")
        _update_framed(digest, value.encode("utf-8"))
    elif isinstance(value, int) and not isinstance(value, bool):
        _update_framed(digest, b"int")
        _update_framed(digest, str(value).encode("ascii"))
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("rollout context floats must be finite")
        canonical = 0.0 if value == 0.0 else value
        _update_framed(digest, b"float64-le")
        _update_framed(
            digest,
            np.asarray([canonical], dtype="<f8").tobytes(order="C"),
        )
    elif isinstance(value, tuple):
        _update_framed(digest, b"tuple")
        _update_framed(digest, str(len(value)).encode("ascii"))
        for item in value:
            _update_canonical_value(digest, item)
    else:
        raise TypeError("rollout context contains an unsupported immutable value")


def _update_canonical_context(
    digest: Any,
    context: tuple[tuple[str, object], ...],
) -> None:
    _update_framed(digest, b"amp-rollout-context-v1")
    _update_framed(digest, str(len(context)).encode("ascii"))
    for key, value in context:
        _update_framed(digest, key.encode("utf-8"))
        _update_canonical_value(digest, value)


def _ordered_ids_sha256(
    values: Sequence[str],
    *,
    domain: bytes,
    name: str,
) -> tuple[tuple[str, ...], str]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be an ordered sequence of identifiers, not a string")
    identifiers = tuple(values)
    if not identifiers or any(
        not isinstance(identifier, str) or not identifier or identifier.strip() != identifier
        for identifier in identifiers
    ):
        raise ValueError(f"{name} must contain non-empty canonical strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique and ordered")
    digest = hashlib.sha256(domain)
    for identifier in identifiers:
        _update_framed(digest, identifier.encode("utf-8"))
    return identifiers, digest.hexdigest()


def _posterior_snapshot_id(
    posterior: JointGaussianPosterior,
    *,
    candidate_ordering_sha256: str,
    output_ordering_sha256: str,
) -> str:
    digest = hashlib.sha256(b"amp-joint-posterior-v1\0")
    _update_framed(digest, candidate_ordering_sha256.encode("ascii"))
    _update_framed(digest, output_ordering_sha256.encode("ascii"))
    for name, value in (
        ("mean", posterior.mean),
        ("covariance", posterior.covariance),
        ("observation_noise", posterior.observation_noise),
    ):
        array = np.asarray(value, dtype="<f8")
        _update_framed(digest, name.encode("ascii"))
        _update_framed(digest, repr(array.shape).encode("ascii"))
        _update_framed(digest, array.tobytes(order="C"))
    return f"posterior-{digest.hexdigest()}"


def _stable_rollout_seed(
    *,
    domain: bytes,
    root_seed: int,
    rollout_namespace: str,
    rollout_index: int,
    branch_id: str,
    policy_version: str,
    posterior_snapshot_id: str,
    candidate_ordering_sha256: str,
    output_ordering_sha256: str,
    draw_id: str | None = None,
) -> int:
    """Derive one platform-independent 128-bit seed from a global rollout key."""

    digest = hashlib.sha256(domain)
    for value in (
        str(root_seed),
        rollout_namespace,
        str(rollout_index),
        branch_id,
        policy_version,
        posterior_snapshot_id,
        candidate_ordering_sha256,
        output_ordering_sha256,
        draw_id or "",
    ):
        _update_framed(digest, value.encode("utf-8"))
    return int.from_bytes(digest.digest()[:16], byteorder="big", signed=False)


def _draw_id(
    draw: FloatArray,
    preference: FloatArray,
    *,
    posterior_snapshot_id: str,
    candidate_ordering_sha256: str,
    output_ordering_sha256: str,
    root_seed: int,
    rollout_namespace: str,
    rollout_index: int,
    draw_seed: int,
) -> str:
    digest = hashlib.sha256(b"amp-frozen-thompson-direction-v1\0")
    _update_framed(digest, FROZEN_ROLLOUT_RNG_ALGORITHM.encode("ascii"))
    _update_framed(digest, FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION.encode("ascii"))
    _update_framed(digest, np.asarray(draw, dtype="<f8").tobytes(order="C"))
    _update_framed(digest, np.asarray(preference, dtype="<f8").tobytes(order="C"))
    _update_framed(digest, posterior_snapshot_id.encode("ascii"))
    _update_framed(digest, candidate_ordering_sha256.encode("ascii"))
    _update_framed(digest, output_ordering_sha256.encode("ascii"))
    _update_framed(digest, str(root_seed).encode("ascii"))
    _update_framed(digest, rollout_namespace.encode("utf-8"))
    _update_framed(digest, str(rollout_index).encode("ascii"))
    _update_framed(digest, str(draw_seed).encode("ascii"))
    return f"draw-{digest.hexdigest()}"


def _rollout_id(
    *,
    namespace: str,
    branch_id: str,
    policy_version: str,
    rollout_index: int,
    draw_id: str,
    child_seed: int,
    context: tuple[tuple[str, object], ...],
) -> str:
    digest = hashlib.sha256(b"amp-frozen-rollout-v1\0")
    for value in (
        namespace,
        branch_id,
        policy_version,
        str(rollout_index),
        draw_id,
        str(child_seed),
    ):
        _update_framed(digest, value.encode("utf-8"))
    _update_canonical_context(digest, context)
    return f"rollout-{digest.hexdigest()}"


def _normalized_preferences(value: object, *, rollout_count: int) -> FloatArray:
    preferences = _readonly_floats(value, name="preference_weights", ndim=2)
    if preferences.shape[0] != rollout_count:
        raise ValueError("preference_weights must have one row per branch rollout")
    if np.any(preferences < 0.0) or np.any(np.max(preferences, axis=1) <= 0.0):
        raise ValueError("each rollout preference must have positive non-negative mass")
    row_scale = np.max(preferences, axis=1, keepdims=True)
    scaled = preferences / row_scale
    if np.any((preferences > 0.0) & (scaled == 0.0)):
        raise ValueError("rollout preference normalization would erase positive mass")
    row_sum = np.sum(scaled, axis=1, keepdims=True)
    normalized = np.array(scaled / row_sum, dtype=np.float64, copy=True)
    if np.any((preferences > 0.0) & (normalized == 0.0)):
        raise ValueError("rollout preference normalization would erase positive mass")
    normalized[normalized == 0.0] = 0.0
    if np.any(~np.isfinite(normalized)) or not np.allclose(
        np.sum(normalized, axis=1),
        1.0,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    ):
        raise ValueError("each rollout preference must normalize to finite unit mass")
    normalized.setflags(write=False)
    return normalized


def _validated_normalized_preferences(value: object, *, rollout_count: int) -> FloatArray:
    preferences = _readonly_floats(value, name="preference_weights", ndim=2)
    if preferences.shape[0] != rollout_count:
        raise ValueError("preference_weights must have one row per branch rollout")
    if np.any(preferences < 0.0) or np.any(np.max(preferences, axis=1) <= 0.0):
        raise ValueError("each rollout preference must have positive non-negative mass")
    canonical = np.array(preferences, dtype=np.float64, copy=True)
    canonical[canonical == 0.0] = 0.0
    if not np.allclose(
        np.sum(canonical, axis=1),
        1.0,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    ):
        raise ValueError("preference_weights must already be normalized")
    canonical.setflags(write=False)
    return canonical


def _rollout_indices(value: Sequence[int], *, rollout_count: int) -> tuple[int, ...]:
    if isinstance(value, str | bytes):
        raise TypeError("rollout_indices must be an ordered sequence of integers")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | np.integer):
            raise ValueError("rollout_indices must contain non-negative integers")
        index = int(item)
        if index < 0:
            raise ValueError("rollout_indices must contain non-negative integers")
        parsed.append(index)
    if len(parsed) != rollout_count:
        raise ValueError("rollout_indices must have one entry per branch rollout")
    if len(set(parsed)) != len(parsed):
        raise ValueError("rollout_indices must be unique within a batch")
    return tuple(parsed)


def _decimal_integer(value: object, *, name: str) -> int:
    if not isinstance(value, str) or (
        value != "0"
        and (
            not value
            or value[0] == "0"
            or any(character not in "0123456789" for character in value)
        )
    ):
        raise ValueError(f"{name} must be a canonical non-negative decimal string")
    return int(value)


@dataclass(frozen=True, slots=True)
class FrozenRolloutBatch:
    """A vectorized posterior draw with one immutable hypothesis per rollout."""

    records: tuple[RolloutRecord, ...]
    outcome_draws: FloatArray
    preference_weights: FloatArray
    candidate_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    posterior_snapshot_id: str
    root_seed: int
    rollout_namespace: str
    rollout_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        if not all(isinstance(record, RolloutRecord) for record in self.records):
            raise TypeError("records must contain RolloutRecord values")
        draws = _readonly_floats(self.outcome_draws, name="outcome_draws", ndim=3)
        preferences = _validated_normalized_preferences(
            self.preference_weights,
            rollout_count=len(self.records),
        )
        if draws.shape[0] != len(self.records):
            raise ValueError("records and draws must share a rollout axis")
        if preferences.shape[1] != draws.shape[2]:
            raise ValueError("preference output dimension must match posterior draws")
        candidate_ids, candidate_ordering_sha256 = _ordered_ids_sha256(
            self.candidate_ids,
            domain=b"amp-frozen-candidate-order-v1\0",
            name="candidate_ids",
        )
        if len(candidate_ids) != draws.shape[1]:
            raise ValueError("candidate_ids must identify the posterior draw point axis")
        output_ids, output_ordering_sha256 = _ordered_ids_sha256(
            self.output_ids,
            domain=b"amp-frozen-output-order-v1\0",
            name="output_ids",
        )
        if len(output_ids) != draws.shape[2]:
            raise ValueError("output_ids must identify the posterior draw output axis")
        if (
            not isinstance(self.posterior_snapshot_id, str)
            or not self.posterior_snapshot_id.startswith("posterior-")
            or len(self.posterior_snapshot_id) != len("posterior-") + 64
            or not set(self.posterior_snapshot_id.removeprefix("posterior-"))
            <= set("0123456789abcdef")
        ):
            raise ValueError("posterior_snapshot_id must be a full SHA-256 identity")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise ValueError("root_seed must be a non-negative integer")
        if self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")
        if (
            not isinstance(self.rollout_namespace, str)
            or not self.rollout_namespace
            or self.rollout_namespace.strip() != self.rollout_namespace
        ):
            raise ValueError("rollout_namespace must be a non-empty canonical string")
        rollout_indices = _rollout_indices(
            self.rollout_indices,
            rollout_count=len(self.records),
        )
        rollout_ids = tuple(record.rollout_id for record in self.records)
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError("rollout IDs must be unique")
        for record, draw, preference, rollout_index in zip(
            self.records,
            draws,
            preferences,
            rollout_indices,
            strict=True,
        ):
            for key, value in record.context:
                _canonical_identifier(key, name="rollout context key")
                _manifest_value(value)
            expected_preference = tuple(float(value) for value in preference)
            try:
                recorded_preference = record.context_value("preference")
                recorded_snapshot = record.context_value("posterior_snapshot_id")
                recorded_ordering = record.context_value("candidate_ordering_sha256")
                recorded_outputs = record.context_value("output_ids")
                recorded_output_ordering = record.context_value("output_ordering_sha256")
                recorded_root_seed = record.context_value("root_seed")
                recorded_namespace = record.context_value("rollout_namespace")
                recorded_index = record.context_value("rollout_index")
                recorded_draw_seed = record.context_value("draw_seed")
                recorded_rng = record.context_value("rng_algorithm")
                recorded_replay_implementation = record.context_value("replay_implementation")
            except KeyError as error:
                raise ValueError("rollout context lacks frozen-direction provenance") from error
            if not _float_tuple_is_exactly_equal(
                recorded_preference,
                expected_preference,
            ):
                raise ValueError("rollout context preference differs from normalized weights")
            if recorded_snapshot != self.posterior_snapshot_id:
                raise ValueError("rollout context posterior snapshot identity differs")
            if recorded_ordering != candidate_ordering_sha256:
                raise ValueError("rollout context candidate ordering identity differs")
            if recorded_outputs != output_ids:
                raise ValueError("rollout context output ordering differs")
            if recorded_output_ordering != output_ordering_sha256:
                raise ValueError("rollout context output ordering identity differs")
            if recorded_root_seed != str(self.root_seed):
                raise ValueError("rollout context root seed differs")
            if recorded_namespace != self.rollout_namespace:
                raise ValueError("rollout context namespace differs")
            if recorded_index != str(rollout_index):
                raise ValueError("rollout context global index differs")
            if recorded_rng != FROZEN_ROLLOUT_RNG_ALGORITHM:
                raise ValueError("rollout context RNG algorithm differs")
            if recorded_replay_implementation != FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION:
                raise ValueError("rollout context replay implementation differs")
            expected_draw_seed = _stable_rollout_seed(
                domain=b"amp-frozen-thompson-draw-seed-v1\0",
                root_seed=self.root_seed,
                rollout_namespace=self.rollout_namespace,
                rollout_index=rollout_index,
                branch_id=record.branch_id,
                policy_version=record.policy_version,
                posterior_snapshot_id=self.posterior_snapshot_id,
                candidate_ordering_sha256=candidate_ordering_sha256,
                output_ordering_sha256=output_ordering_sha256,
            )
            if recorded_draw_seed != str(expected_draw_seed):
                raise ValueError("rollout context draw seed differs")
            expected_draw_id = _draw_id(
                draw,
                preference,
                posterior_snapshot_id=self.posterior_snapshot_id,
                candidate_ordering_sha256=candidate_ordering_sha256,
                output_ordering_sha256=output_ordering_sha256,
                root_seed=self.root_seed,
                rollout_namespace=self.rollout_namespace,
                rollout_index=rollout_index,
                draw_seed=expected_draw_seed,
            )
            if record.posterior_draw_id != expected_draw_id:
                raise ValueError("rollout posterior_draw_id differs from its frozen direction")
            expected_child_seed = _stable_rollout_seed(
                domain=b"amp-frozen-thompson-child-seed-v1\0",
                root_seed=self.root_seed,
                rollout_namespace=self.rollout_namespace,
                rollout_index=rollout_index,
                branch_id=record.branch_id,
                policy_version=record.policy_version,
                posterior_snapshot_id=self.posterior_snapshot_id,
                candidate_ordering_sha256=candidate_ordering_sha256,
                output_ordering_sha256=output_ordering_sha256,
                draw_id=expected_draw_id,
            )
            if record.seed != expected_child_seed:
                raise ValueError("rollout child seed differs from its frozen identity")
            expected_rollout_id = _rollout_id(
                namespace=self.rollout_namespace,
                branch_id=record.branch_id,
                policy_version=record.policy_version,
                rollout_index=rollout_index,
                draw_id=expected_draw_id,
                child_seed=expected_child_seed,
                context=record.context,
            )
            if record.rollout_id != expected_rollout_id:
                raise ValueError("rollout_id must bind branch, context, policy, and direction")
        object.__setattr__(self, "outcome_draws", draws)
        object.__setattr__(self, "preference_weights", preferences)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "output_ids", output_ids)
        object.__setattr__(self, "rollout_indices", rollout_indices)

    def draw_for_rollout(self, rollout_id: str) -> FloatArray:
        """Return the frozen draw assigned to ``rollout_id`` without resampling."""

        for position, record in enumerate(self.records):
            if record.rollout_id == rollout_id:
                return self.outcome_draws[position]
        raise KeyError(rollout_id)

    @property
    def candidate_ordering_sha256(self) -> str:
        """Identity of the exact ordered candidate semantics."""

        return _ordered_ids_sha256(
            self.candidate_ids,
            domain=b"amp-frozen-candidate-order-v1\0",
            name="candidate_ids",
        )[1]

    @property
    def output_ordering_sha256(self) -> str:
        """Identity of the exact ordered output semantics."""

        return _ordered_ids_sha256(
            self.output_ids,
            domain=b"amp-frozen-output-order-v1\0",
            name="output_ids",
        )[1]

    def verify_against_posterior(self, posterior: JointGaussianPosterior) -> None:
        """Recompute the snapshot and draws from their stable global seeds."""

        expected_snapshot = _posterior_snapshot_id(
            posterior,
            candidate_ordering_sha256=self.candidate_ordering_sha256,
            output_ordering_sha256=self.output_ordering_sha256,
        )
        if expected_snapshot != self.posterior_snapshot_id:
            raise ValueError("posterior does not match the frozen snapshot identity")
        draw_seeds = tuple(
            _stable_rollout_seed(
                domain=b"amp-frozen-thompson-draw-seed-v1\0",
                root_seed=self.root_seed,
                rollout_namespace=self.rollout_namespace,
                rollout_index=rollout_index,
                branch_id=record.branch_id,
                policy_version=record.policy_version,
                posterior_snapshot_id=self.posterior_snapshot_id,
                candidate_ordering_sha256=self.candidate_ordering_sha256,
                output_ordering_sha256=self.output_ordering_sha256,
            )
            for record, rollout_index in zip(
                self.records,
                self.rollout_indices,
                strict=True,
            )
        )
        expected_draws = posterior.sample_functions_from_seeds(draw_seeds)
        expected_bytes = np.asarray(expected_draws, dtype="<f8").tobytes(order="C")
        recorded_bytes = np.asarray(self.outcome_draws, dtype="<f8").tobytes(order="C")
        if expected_bytes != recorded_bytes:
            raise ValueError("frozen draws do not reproduce from the declared posterior and seeds")

    def to_manifest(self) -> dict[str, object]:
        """Serialize all frozen directions and replay identities without resampling."""

        return {
            "schema_version": FROZEN_ROLLOUT_MANIFEST_SCHEMA_VERSION,
            "rng_algorithm": FROZEN_ROLLOUT_RNG_ALGORITHM,
            "replay_implementation": FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION,
            "candidate_ids": list(self.candidate_ids),
            "candidate_ordering_sha256": self.candidate_ordering_sha256,
            "output_ids": list(self.output_ids),
            "output_ordering_sha256": self.output_ordering_sha256,
            "posterior_snapshot_id": self.posterior_snapshot_id,
            "root_seed": str(self.root_seed),
            "rollout_namespace": self.rollout_namespace,
            "rollout_indices": [str(value) for value in self.rollout_indices],
            "records": [
                {
                    "rollout_id": record.rollout_id,
                    "branch_id": record.branch_id,
                    "posterior_draw_id": record.posterior_draw_id,
                    "context": [[key, _manifest_value(value)] for key, value in record.context],
                    "policy_version": record.policy_version,
                    "seed": str(record.seed),
                    "draw_seed": record.context_value("draw_seed"),
                    "outcome_draw": self.outcome_draws[position].tolist(),
                    "preference_weights": self.preference_weights[position].tolist(),
                }
                for position, record in enumerate(self.records)
            ],
        }

    @classmethod
    def from_manifest(
        cls,
        payload: object,
        *,
        posterior: JointGaussianPosterior | None = None,
    ) -> FrozenRolloutBatch:
        """Validate and reconstruct a frozen batch from its durable manifest."""

        if not isinstance(payload, dict):
            raise TypeError("frozen rollout manifest must be a mapping")
        expected_keys = {
            "schema_version",
            "rng_algorithm",
            "replay_implementation",
            "candidate_ids",
            "candidate_ordering_sha256",
            "output_ids",
            "output_ordering_sha256",
            "posterior_snapshot_id",
            "root_seed",
            "rollout_namespace",
            "rollout_indices",
            "records",
        }
        if set(payload) != expected_keys:
            raise ValueError("frozen rollout manifest fields do not match the schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != FROZEN_ROLLOUT_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported frozen rollout manifest schema version")
        if payload["rng_algorithm"] != FROZEN_ROLLOUT_RNG_ALGORITHM:
            raise ValueError("unsupported frozen rollout RNG algorithm")
        if payload["replay_implementation"] != FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION:
            raise ValueError("unsupported frozen rollout replay implementation")
        raw_candidates = payload["candidate_ids"]
        raw_outputs = payload["output_ids"]
        if not isinstance(raw_candidates, list) or not all(
            isinstance(value, str) for value in raw_candidates
        ):
            raise TypeError("manifest candidate_ids must be a list of strings")
        if not isinstance(raw_outputs, list) or not all(
            isinstance(value, str) for value in raw_outputs
        ):
            raise TypeError("manifest output_ids must be a list of strings")
        candidates, candidate_hash = _ordered_ids_sha256(
            raw_candidates,
            domain=b"amp-frozen-candidate-order-v1\0",
            name="candidate_ids",
        )
        outputs, output_hash = _ordered_ids_sha256(
            raw_outputs,
            domain=b"amp-frozen-output-order-v1\0",
            name="output_ids",
        )
        if payload["candidate_ordering_sha256"] != candidate_hash:
            raise ValueError("manifest candidate ordering identity differs")
        if payload["output_ordering_sha256"] != output_hash:
            raise ValueError("manifest output ordering identity differs")
        raw_indices = payload["rollout_indices"]
        if not isinstance(raw_indices, list):
            raise TypeError("manifest rollout_indices must be a list")
        rollout_indices = tuple(
            _decimal_integer(value, name="manifest rollout index") for value in raw_indices
        )
        root_seed = _decimal_integer(payload["root_seed"], name="manifest root seed")
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("manifest records must be a non-empty list")
        records: list[RolloutRecord] = []
        draws: list[object] = []
        preferences: list[object] = []
        record_keys = {
            "rollout_id",
            "branch_id",
            "posterior_draw_id",
            "context",
            "policy_version",
            "seed",
            "draw_seed",
            "outcome_draw",
            "preference_weights",
        }
        for raw_record in raw_records:
            if not isinstance(raw_record, dict) or set(raw_record) != record_keys:
                raise ValueError("frozen rollout record fields do not match the schema")
            identifiers = {
                field: _canonical_identifier(
                    raw_record[field],
                    name=f"manifest rollout {field}",
                )
                for field in (
                    "rollout_id",
                    "branch_id",
                    "posterior_draw_id",
                    "policy_version",
                )
            }
            raw_context = raw_record["context"]
            if not isinstance(raw_context, list) or any(
                not isinstance(entry, list) or len(entry) != 2 for entry in raw_context
            ):
                raise TypeError("manifest rollout context must be a list of key-value pairs")
            context = tuple(
                (
                    _canonical_identifier(
                        entry[0],
                        name="manifest rollout context key",
                    ),
                    _immutable_manifest_value(entry[1]),
                )
                for entry in raw_context
            )
            try:
                context_draw_seed = dict(context)["draw_seed"]
            except KeyError as error:
                raise ValueError("manifest rollout context lacks draw_seed") from error
            if raw_record["draw_seed"] != context_draw_seed:
                raise ValueError("manifest record draw seed differs from its context")
            _validate_manifest_float_tree(
                raw_record["outcome_draw"],
                name="manifest outcome_draw",
                depth=2,
            )
            _validate_manifest_float_tree(
                raw_record["preference_weights"],
                name="manifest preference_weights",
                depth=1,
                canonical_zero=True,
            )
            records.append(
                RolloutRecord(
                    rollout_id=identifiers["rollout_id"],
                    branch_id=identifiers["branch_id"],
                    posterior_draw_id=identifiers["posterior_draw_id"],
                    context=context,
                    policy_version=identifiers["policy_version"],
                    seed=_decimal_integer(raw_record["seed"], name="manifest child seed"),
                )
            )
            draws.append(raw_record["outcome_draw"])
            preferences.append(raw_record["preference_weights"])
        batch = cls(
            records=tuple(records),
            outcome_draws=np.asarray(draws, dtype=np.float64),
            preference_weights=np.asarray(preferences, dtype=np.float64),
            candidate_ids=candidates,
            output_ids=outputs,
            posterior_snapshot_id=payload["posterior_snapshot_id"],
            root_seed=root_seed,
            rollout_namespace=payload["rollout_namespace"],
            rollout_indices=rollout_indices,
        )
        if posterior is not None:
            batch.verify_against_posterior(posterior)
        return batch


def sample_frozen_rollouts(
    posterior: JointGaussianPosterior,
    branch_ids: Sequence[str],
    *,
    candidate_ids: Sequence[str],
    output_ids: Sequence[str],
    rollout_namespace: str,
    rollout_indices: Sequence[int],
    contexts: Sequence[tuple[tuple[str, object], ...]],
    preference_weights: object,
    policy_version: str,
    root_seed: int,
) -> FrozenRolloutBatch:
    """Draw many coherent landscapes once and bind each to one rollout.

    Each draw and child seed is keyed by its caller-assigned global rollout
    index, so reordering or sharding a batch cannot change its randomness.
    Proposal microbatches retrieve the stored draw by rollout ID; they must
    never resample token- or edit-level hypotheses.
    """

    if isinstance(branch_ids, str):
        raise TypeError("branch_ids must be an ordered sequence of identifiers, not a string")
    branches = tuple(branch_ids)
    context_values = tuple(contexts)
    if not branches:
        raise ValueError("branch_ids cannot be empty")
    if any(
        not isinstance(branch, str) or not branch or branch.strip() != branch for branch in branches
    ):
        raise ValueError("branch_ids must contain non-empty canonical strings")
    if len(context_values) != len(branches):
        raise ValueError("contexts must have one entry per branch rollout")
    if (
        not isinstance(policy_version, str)
        or not policy_version
        or policy_version.strip() != policy_version
    ):
        raise ValueError("policy_version must be a non-empty canonical string")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int | np.integer):
        raise ValueError("root_seed must be an integer")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    parsed_root_seed = int(root_seed)
    global_indices = _rollout_indices(rollout_indices, rollout_count=len(branches))
    if (
        not isinstance(rollout_namespace, str)
        or not rollout_namespace
        or rollout_namespace.strip() != rollout_namespace
    ):
        raise ValueError("rollout_namespace must be a non-empty canonical string")
    preferences = _normalized_preferences(
        preference_weights,
        rollout_count=len(branches),
    )
    if preferences.shape[1] != posterior.n_outputs:
        raise ValueError("preference output dimension must match posterior outputs")
    candidates, candidate_ordering_sha256 = _ordered_ids_sha256(
        candidate_ids,
        domain=b"amp-frozen-candidate-order-v1\0",
        name="candidate_ids",
    )
    if len(candidates) != posterior.n_points:
        raise ValueError("candidate_ids must identify every posterior point in order")
    outputs, output_ordering_sha256 = _ordered_ids_sha256(
        output_ids,
        domain=b"amp-frozen-output-order-v1\0",
        name="output_ids",
    )
    if len(outputs) != posterior.n_outputs:
        raise ValueError("output_ids must identify every posterior output in order")
    posterior_snapshot_id = _posterior_snapshot_id(
        posterior,
        candidate_ordering_sha256=candidate_ordering_sha256,
        output_ordering_sha256=output_ordering_sha256,
    )
    base_contexts: list[tuple[tuple[str, object], ...]] = []
    for context, preference in zip(context_values, preferences, strict=True):
        if not isinstance(context, tuple):
            raise TypeError("each rollout context must be an immutable tuple")
        if any(not isinstance(entry, tuple) or len(entry) != 2 for entry in context):
            raise TypeError("each rollout context must contain key-value pairs")
        for key, value in context:
            _canonical_identifier(key, name="rollout context key")
            # Validate the reserved preference below against its normalized,
            # float-exact value first. This keeps every malformed preference on
            # the same deterministic API error path, including mutable lists.
            if key != "preference":
                _manifest_value(value)
        context_map = dict(context)
        if len(context_map) != len(context):
            raise ValueError("rollout context keys must be unique")
        reserved = {
            "posterior_snapshot_id",
            "candidate_ordering_sha256",
            "output_ids",
            "root_seed",
            "rollout_namespace",
            "rollout_index",
            "draw_seed",
            "rng_algorithm",
            "replay_implementation",
        }
        if reserved & set(context_map):
            raise ValueError("rollout context cannot override frozen provenance keys")
        normalized_preference = tuple(float(value) for value in preference)
        if "preference" in context_map:
            claimed_preference = context_map["preference"]
            if not _float_tuple_is_exactly_equal(
                claimed_preference,
                normalized_preference,
            ):
                raise ValueError("rollout context preference differs from normalized weights")
            context = tuple(entry for entry in context if entry[0] != "preference")
        base_contexts.append(context)

    draw_seeds = tuple(
        _stable_rollout_seed(
            domain=b"amp-frozen-thompson-draw-seed-v1\0",
            root_seed=parsed_root_seed,
            rollout_namespace=rollout_namespace,
            rollout_index=rollout_index,
            branch_id=branch_id,
            policy_version=policy_version,
            posterior_snapshot_id=posterior_snapshot_id,
            candidate_ordering_sha256=candidate_ordering_sha256,
            output_ordering_sha256=output_ordering_sha256,
        )
        for branch_id, rollout_index in zip(branches, global_indices, strict=True)
    )
    draws = posterior.sample_functions_from_seeds(draw_seeds)
    bound_contexts = tuple(
        (
            *context,
            ("preference", tuple(float(value) for value in preference)),
            ("posterior_snapshot_id", posterior_snapshot_id),
            ("candidate_ordering_sha256", candidate_ordering_sha256),
            ("output_ids", outputs),
            ("output_ordering_sha256", output_ordering_sha256),
            ("root_seed", str(parsed_root_seed)),
            ("rollout_namespace", rollout_namespace),
            ("rollout_index", str(rollout_index)),
            ("draw_seed", str(draw_seed)),
            ("rng_algorithm", FROZEN_ROLLOUT_RNG_ALGORITHM),
            ("replay_implementation", FROZEN_ROLLOUT_REPLAY_IMPLEMENTATION),
        )
        for context, preference, rollout_index, draw_seed in zip(
            base_contexts,
            preferences,
            global_indices,
            draw_seeds,
            strict=True,
        )
    )
    draw_ids = tuple(
        _draw_id(
            draw,
            preference,
            posterior_snapshot_id=posterior_snapshot_id,
            candidate_ordering_sha256=candidate_ordering_sha256,
            output_ordering_sha256=output_ordering_sha256,
            root_seed=parsed_root_seed,
            rollout_namespace=rollout_namespace,
            rollout_index=rollout_index,
            draw_seed=draw_seed,
        )
        for draw, preference, rollout_index, draw_seed in zip(
            draws,
            preferences,
            global_indices,
            draw_seeds,
            strict=True,
        )
    )
    child_seeds = tuple(
        _stable_rollout_seed(
            domain=b"amp-frozen-thompson-child-seed-v1\0",
            root_seed=parsed_root_seed,
            rollout_namespace=rollout_namespace,
            rollout_index=rollout_index,
            branch_id=branch_id,
            policy_version=policy_version,
            posterior_snapshot_id=posterior_snapshot_id,
            candidate_ordering_sha256=candidate_ordering_sha256,
            output_ordering_sha256=output_ordering_sha256,
            draw_id=draw_id,
        )
        for branch_id, rollout_index, draw_id in zip(
            branches,
            global_indices,
            draw_ids,
            strict=True,
        )
    )
    records = tuple(
        RolloutRecord(
            rollout_id=_rollout_id(
                namespace=rollout_namespace,
                branch_id=branch_id,
                policy_version=policy_version,
                rollout_index=rollout_index,
                draw_id=draw_id,
                child_seed=child_seed,
                context=context,
            ),
            branch_id=branch_id,
            posterior_draw_id=draw_id,
            context=context,
            policy_version=policy_version,
            seed=child_seed,
        )
        for branch_id, context, rollout_index, draw_id, child_seed in zip(
            branches,
            bound_contexts,
            global_indices,
            draw_ids,
            child_seeds,
            strict=True,
        )
    )
    return FrozenRolloutBatch(
        records=records,
        outcome_draws=draws,
        preference_weights=preferences,
        candidate_ids=candidates,
        output_ids=outputs,
        posterior_snapshot_id=posterior_snapshot_id,
        root_seed=parsed_root_seed,
        rollout_namespace=rollout_namespace,
        rollout_indices=global_indices,
    )


def thompson_substitution_weights_batch(
    parents: Sequence[str],
    alphabet: str,
    base_residue_probabilities: object,
    gains: object,
    *,
    scale: float,
) -> tuple[FloatArray, FloatArray]:
    """Factor exact Gibbs tilts for an equal-length batch of parent peptides."""

    parent_values = tuple(parent.strip().upper() for parent in parents)
    if not parent_values or any(not parent for parent in parent_values):
        raise ValueError("parents must be non-empty peptide strings")
    lengths = {len(parent) for parent in parent_values}
    if len(lengths) != 1:
        raise ValueError("parents must share one length bucket")
    if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must contain distinct residue tokens")
    if any(set(parent) - set(alphabet) for parent in parent_values):
        raise ValueError("every parent must use the declared alphabet")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("Thompson scale must be finite and non-negative")

    rollout_count = len(parent_values)
    length = lengths.pop()
    expected = (rollout_count, length, len(alphabet))
    probabilities = np.asarray(base_residue_probabilities, dtype=np.float64)
    if probabilities.shape == expected[1:]:
        probabilities = np.broadcast_to(probabilities, expected)
    if probabilities.shape != expected:
        raise ValueError(f"base_residue_probabilities must have shape {expected[1:]} or {expected}")
    gain_values = np.asarray(gains, dtype=np.float64)
    if gain_values.shape != expected:
        raise ValueError(f"gains must have shape {expected}")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("base residue probabilities must be finite and non-negative")
    if np.any(~np.isfinite(gain_values)):
        raise ValueError("substitution gains must be finite")

    token_indices = {token: index for index, token in enumerate(alphabet)}
    position_output = np.empty((rollout_count, length), dtype=np.float64)
    residue_output = np.zeros(expected, dtype=np.float64)
    for rollout, parent in enumerate(parent_values):
        admissible_gain = np.array(gain_values[rollout], copy=True)
        for position, token in enumerate(parent):
            admissible_gain[position, token_indices[token]] = -np.inf
            admissible_gain[position, probabilities[rollout, position] <= 0.0] = -np.inf
        global_maximum = float(np.max(admissible_gain))
        if not np.isfinite(global_maximum):
            raise ValueError("each rollout needs an admissible substitution with positive mass")
        log_position = np.empty(length, dtype=np.float64)
        for position, token in enumerate(parent):
            conditional = np.array(probabilities[rollout, position], copy=True)
            conditional[token_indices[token]] = 0.0
            conditional_scale = float(np.max(conditional))
            if conditional_scale <= 0.0:
                raise ValueError("every position needs a positive-probability alternative")
            conditional /= conditional_scale
            conditional /= float(np.sum(conditional))
            alternative = conditional > 0.0
            row_maximum = float(np.max(gain_values[rollout, position, alternative]))
            if scale == 0.0:
                centered_gain = np.zeros(np.count_nonzero(alternative), dtype=np.float64)
                position_gain = 0.0
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    centered_gain = scale * (
                        gain_values[rollout, position, alternative] - row_maximum
                    )
                    position_gain = scale * (row_maximum - global_maximum)
                if np.any(np.isnan(centered_gain)) or np.isnan(position_gain):
                    raise ValueError("relative Thompson gains became undefined")
            log_tilted = np.full(len(alphabet), -np.inf, dtype=np.float64)
            log_tilted[alternative] = np.log(conditional[alternative]) + centered_gain
            normalizer = float(logsumexp(log_tilted))
            if not np.isfinite(normalizer):
                raise FloatingPointError("Thompson substitution normalizer became invalid")
            log_position[position] = -np.log(length) + position_gain + normalizer
            residue_output[rollout, position] = np.exp(log_tilted - normalizer)
        position_output[rollout] = np.exp(log_position - logsumexp(log_position))

    position_output.setflags(write=False)
    residue_output.setflags(write=False)
    return position_output, residue_output

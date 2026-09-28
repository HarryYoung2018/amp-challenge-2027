"""Charged-only linear Gaussian acquisition model; never imports an oracle teacher.

This likelihood describes regression discrepancy for normalized MODEL_PROXY
targets, not assay noise or calibrated biological probabilities. Callers must
authenticate the history and feature receipts before supplying them here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.models.feature_posterior import FeatureGaussianPosterior
from amp_challenge.representations.peptide_esm import (
    NAMESPACE_RECEIPT_SHA256,
    canonical_json,
    digest,
    load_features,
)

CONFIG_SHA256 = "7ed8a5598bda401e993c350f5ba4e1cf5466c5b810a04cd22910d9f99c0f20ba"
KNOWN_FEATURE_SHA256 = "c6c49f570be0788496b04ce2c3248b23d5babeb560b046f6aef5b006d3b325f9"
QUERY_LAYOUT_SHA256 = "6a2d727734ad58dad01ac842d497dd10602479c984c0512b6d334cd0ccdf7871"
REPRESENTATIONS = {
    "esm320_plus_normalized_length": ("esm_length", 321),
    "esm320_plus_normalized_length_plus_spectral32": ("esm_length_spectral", 353),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    return type(value) is str and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _array(value, shape):
    raw = np.asarray(value)
    _require(raw.shape == shape and raw.dtype.kind in "fiu", "learner array shape/dtype differs")
    result = np.array(raw, dtype=np.float64, copy=True)
    _require(np.isfinite(result).all(), "learner array must be finite")
    result.setflags(write=False)
    return result


def _numerical_digest(*arrays):
    result = hashlib.sha256(b"amp/charged-gaussian/numerical-arrays/v1\0")
    for value in arrays:
        array = np.asarray(value, dtype="<f8", order="C")
        result.update(canonical_json({"shape": list(array.shape), "dtype": "<f8"}))
        result.update(array.tobytes())
    return result.hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratorFeatureTransform:
    representation: str
    mean: np.ndarray
    scale: np.ndarray
    source_sha256: str

    def __post_init__(self):
        _require(self.representation in REPRESENTATIONS, "unsupported learner representation")
        dimension = REPRESENTATIONS[self.representation][1]
        mean = _array(self.mean, (dimension,))
        scale = _array(self.scale, (dimension,))
        _require(np.all(scale > 0) and _pin(self.source_sha256), "invalid transform scale/source")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @property
    def sha256(self):
        # Source paths/receipts remain separate; semantic draws bind the actual transform.
        return digest(
            canonical_json(
                {
                    "config_sha256": CONFIG_SHA256,
                    "representation": self.representation,
                    "parameters_sha256": _numerical_digest(self.mean, self.scale),
                }
            )
        )

    @property
    def transform_sha256(self):
        return self.sha256

    def apply(self, raw):
        view = np.asarray(raw)
        _require(view.ndim == 2 and len(view) <= 8192, "feature batch exceeds learner bound")
        values = _array(view, (len(view), len(self.mean)))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                transformed = np.clip((values - self.mean) / self.scale, -8.0, 8.0)
            except FloatingPointError as error:
                raise ValueError("feature transform is not representable") from error
        result = np.column_stack((np.ones(len(view)), transformed))
        result.setflags(write=False)
        return result


def fit_generator_transform(rows, features, *, representation, source_sha256):
    """Bounded numerical seam; the real loader below authenticates membership."""
    _require(
        representation in REPRESENTATIONS and _pin(source_sha256), "transform identity differs"
    )
    _require(1 <= len(rows) <= 698, "generator transform row bound differs")
    _require(
        all(row["namespace"] == "generator" and _pin(row["sequence_id"]) for row in rows)
        and len({row["sequence_id"] for row in rows}) == len(rows)
        and all(
            type(row["union_component_id"]) is str and row["union_component_id"] for row in rows
        ),
        "only unique generator-namespace rows may determine the transform",
    )
    values = _array(features, (len(rows), REPRESENTATIONS[representation][1]))
    counts = Counter(row["union_component_id"] for row in rows)
    weights = np.asarray([1.0 / len(counts) / counts[row["union_component_id"]] for row in rows])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            mean = weights @ values
            scale = np.sqrt(weights @ np.square(values - mean))
        except FloatingPointError as error:
            raise ValueError("generator transform is not representable") from error
    # Only exact constants get the declared fallback. Underflow in a genuinely
    # varying column is a numerical failure, not authority to erase its scale.
    scale = np.where(np.any(values != values[0], axis=0), scale, 1.0)
    _require(np.all(scale > 0), "nonconstant generator variance is not representable")
    return GeneratorFeatureTransform(representation, mean, scale, source_sha256)


def load_generator_transform(
    *, namespace_receipt: Path, generator_corpus: Path, features: Path, representation: str
):
    """Authenticate 698 generator identities before selecting preprocessing rows."""
    _require(representation in REPRESENTATIONS, "unsupported learner representation")
    for path in (namespace_receipt, generator_corpus):
        _require(
            path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 16777216,
            "namespace input path/size differs",
        )
    receipt = namespace_receipt.read_bytes()
    _require(digest(receipt) == NAMESPACE_RECEIPT_SHA256, "namespace receipt pin differs")
    corpus = generator_corpus.read_bytes()
    expected = json.loads(receipt)["twins"][0]["semantic_sha256"]["generator_corpus.jsonl"]
    _require(digest(corpus) == expected, "generator corpus pin differs")
    rows = [json.loads(line) for line in corpus.splitlines()]
    _require(
        len(rows) == 698 and all(row["namespace"] == "generator" for row in rows),
        "generator census differs",
    )
    feature_rows, arrays, _ = load_features(features, KNOWN_FEATURE_SHA256)
    indexed = {row["sequence_id"]: index for index, row in enumerate(feature_rows)}
    indices = []
    for row in rows:
        index = indexed[row["sequence_id"]]
        _require(
            feature_rows[index]["sequence"] == row["sequence"], "generator feature identity differs"
        )
        indices.append(index)
    source = digest(
        canonical_json(
            {
                "namespace_receipt_sha256": NAMESPACE_RECEIPT_SHA256,
                "generator_corpus_sha256": expected,
                "feature_manifest_sha256": KNOWN_FEATURE_SHA256,
            }
        )
    )
    return fit_generator_transform(
        rows,
        arrays[REPRESENTATIONS[representation][0]][indices],
        representation=representation,
        source_sha256=source,
    )


def _prior(transform):
    raw_dimension = len(transform.mean)
    dimension = raw_dimension + 1
    mean = np.zeros((dimension, 2))
    mean[0] = 0.5
    basis = np.zeros((dimension, 2, 2 * dimension))
    for feature in range(dimension):
        scale = 0.25 if feature == 0 else 0.1 / math.sqrt(raw_dimension)
        basis[feature, 0, 2 * feature] = scale
        basis[feature, 1, 2 * feature] = 0.5 * scale
        basis[feature, 1, 2 * feature + 1] = math.sqrt(0.75) * scale
    return FeatureGaussianPosterior.prior(mean, basis)


def _scalar_prior(transform):
    """One scalar process with the historical per-output marginal prior scale."""
    raw_dimension = len(transform.mean)
    dimension = raw_dimension + 1
    mean = np.zeros((dimension, 1))
    mean[0, 0] = 0.5
    scales = np.full(dimension, 0.1 / math.sqrt(raw_dimension))
    scales[0] = 0.25
    return FeatureGaussianPosterior.prior(mean, np.diag(scales)[:, None, :])


def _lift_scalar_replicas(backend):
    """Duplicate output coordinates, not observations or latent information."""
    _require(backend.coefficient_mean.shape[1] == 1, "scalar lift needs one output")
    return FeatureGaussianPosterior(
        np.repeat(backend.coefficient_mean, 2, axis=1),
        np.repeat(backend.coefficient_factor, 2, axis=1),
        backend.latent_mean,
        backend.upper_precision,
        backend.observation_count,
    )


@dataclass(frozen=True, slots=True)
class GaussianLearnerSnapshot:
    backend: FeatureGaussianPosterior
    transform: GeneratorFeatureTransform
    history_sha256: str
    semantic_history_sha256: str
    feature_receipt_sha256: str
    fit_query_ids: tuple[str, ...]
    fit_sequence_ids: tuple[str, ...]
    objective_context_sha256: str
    max_fit_rows: int = 512
    scalar_replicas: bool = False

    def __post_init__(self):
        _require(
            type(self.backend) is FeatureGaussianPosterior
            and type(self.transform) is GeneratorFeatureTransform,
            "learner numerical snapshot types differ",
        )
        _require(
            all(
                _pin(value)
                for value in (
                    self.history_sha256,
                    self.semantic_history_sha256,
                    self.feature_receipt_sha256,
                    self.objective_context_sha256,
                )
            ),
            "learner source binding differs",
        )
        _require(
            type(self.max_fit_rows) is int
            and self.max_fit_rows > 0
            and type(self.fit_query_ids) is tuple
            and type(self.fit_sequence_ids) is tuple
            and all(type(value) is str and value for value in self.fit_query_ids)
            and len(self.fit_query_ids) == len(self.fit_sequence_ids) <= self.max_fit_rows
            and len(set(self.fit_query_ids)) == len(self.fit_query_ids)
            and len(set(self.fit_sequence_ids)) == len(self.fit_sequence_ids)
            and all(_pin(value) for value in self.fit_sequence_ids),
            "learner fitted-row inventory differs",
        )
        _require(
            self.backend.coefficient_mean.shape == (len(self.transform.mean) + 1, 2)
            and type(self.scalar_replicas) is bool
            and self.backend.observation_count
            == (1 if self.scalar_replicas else 2) * len(self.fit_query_ids),
            "learner dimensions/count differ",
        )
        if self.scalar_replicas:
            _require(
                np.array_equal(
                    self.backend.coefficient_mean[:, 0], self.backend.coefficient_mean[:, 1]
                )
                and np.array_equal(
                    self.backend.coefficient_factor[:, 0], self.backend.coefficient_factor[:, 1]
                ),
                "scalar replica outputs must share exactly one latent process",
            )

    @property
    def numerical_sha256(self):
        payload = {
            "config_sha256": CONFIG_SHA256,
            "transform_sha256": self.transform.sha256,
            "semantic_history_sha256": self.semantic_history_sha256,
            "arrays_sha256": _numerical_digest(
                self.backend.coefficient_mean,
                self.backend.coefficient_factor,
                self.backend.latent_mean,
                self.backend.upper_precision,
            ),
        }
        if self.max_fit_rows != 512:
            payload["max_fit_rows"] = self.max_fit_rows
        if self.scalar_replicas:
            payload["observation_semantics"] = (
                "one_scalar_observation_lifted_to_perfectly_correlated_coordinates"
            )
        return digest(canonical_json(payload))

    def clipped_point_means(self, raw):
        features = self.transform.apply(raw)
        # Mean-only consumers do not pay for triangular solves or covariance.
        coefficients = self.backend.coefficient_mean + np.einsum(
            "dor,r->do", self.backend.coefficient_factor, self.backend.latent_mean
        )
        result = np.clip(features @ coefficients, 0.0, 1.0)
        result.setflags(write=False)
        return result

    def joint(self, raw):
        view = np.asarray(raw)
        _require(view.ndim == 2 and 1 <= len(view) <= 256, "joint shortlist bound differs")
        features = self.transform.apply(raw)
        noise = np.repeat(self.observation_noise[None, :, :], len(features), axis=0)
        return self.backend.joint(features, observation_noise=noise)

    @property
    def observation_noise(self):
        return (np.ones((2, 2)) if self.scalar_replicas else np.eye(2)) * 0.01

    def draw_latent(self, *, seed: int, draw_ordinal: int, count: int = 1):
        _require(
            type(seed) is int
            and 0 <= seed < 2**63
            and type(draw_ordinal) is int
            and 0 <= draw_ordinal < 2**32,
            "learner semantic draw identity differs",
        )
        semantic = digest(
            canonical_json(
                {"snapshot": self.numerical_sha256, "seed": seed, "draw_ordinal": draw_ordinal}
            )
        )
        return self.backend.draw_latent(seed=int(semantic[:16], 16), count=count)


def fit_charged_learner(
    history,
    transform,
    *,
    feature_sequence_ids,
    raw_features,
    feature_receipt_sha256,
    eligible_query_ids,
    scalar_replicas: bool = False,
):
    """Fit only a complete already-authenticated charged prefix; no hidden query."""
    _require(
        type(history) is VerifiedHistorySnapshot and type(transform) is GeneratorFeatureTransform,
        "learner needs exact verified history and frozen transform",
    )
    history.__post_init__()
    _require(history.complete, "incomplete charged history cannot update the learner")
    _require(
        type(eligible_query_ids) is tuple
        and len(set(eligible_query_ids)) == len(eligible_query_ids)
        and all(type(value) is str for value in eligible_query_ids)
        and set(eligible_query_ids) <= {row.query_id for row in history.observations},
        "eligibility must refer only to this charged prefix",
    )
    _require(_pin(feature_receipt_sha256), "learner feature receipt differs")
    selected = [
        row
        for row in history.observations
        if row.status == "successful" and row.query_id in eligible_query_ids
    ]
    wanted = tuple(digest(row.sequence.encode("ascii")) for row in selected)
    _require(
        type(feature_sequence_ids) is tuple and feature_sequence_ids == wanted,
        "learner features must exactly follow successful eligible charged rows",
    )
    features = transform.apply(raw_features)
    _require(len(features) == len(selected), "learner feature/charged-row count differs")
    _require(type(scalar_replicas) is bool, "scalar replica fit mode must be explicit boolean")
    backend = _scalar_prior(transform) if scalar_replicas else _prior(transform)
    values = np.asarray([row.objectives for row in selected], dtype=np.float64).reshape(-1, 2)
    if scalar_replicas:
        _require(
            np.array_equal(values[:, 0], values[:, 1]),
            "scalar replica observations must be identical",
        )
    for start in range(0, len(selected), 128):
        part = features[start : start + 128]
        backend = backend.condition(
            part if scalar_replicas else np.repeat(part, 2, axis=0),
            [0] * len(part) if scalar_replicas else [0, 1] * len(part),
            values[start : start + 128, 0]
            if scalar_replicas
            else values[start : start + 128].reshape(-1),
            np.eye((1 if scalar_replicas else 2) * len(part)) * 0.01,
        )
    if scalar_replicas:
        backend = _lift_scalar_replicas(backend)
    semantic = digest(
        canonical_json(
            {
                "objective_context_sha256": history.objective_context_sha256,
                "history": [
                    {
                        "sequence": row.sequence,
                        "status": row.status,
                        "eligible": row.query_id in eligible_query_ids,
                        "objectives": row.objectives
                        if row.query_id in eligible_query_ids
                        else None,
                    }
                    for row in history.observations
                ],
            }
        )
    )
    if scalar_replicas:
        semantic = digest(
            canonical_json(
                {
                    "history_semantic_sha256": semantic,
                    "observation_semantics": "one_scalar_observation_lifted_to_perfectly_correlated_coordinates",
                }
            )
        )
    return GaussianLearnerSnapshot(
        backend,
        transform,
        history.sha256,
        semantic,
        feature_receipt_sha256,
        tuple(row.query_id for row in selected),
        wanted,
        history.objective_context_sha256,
        history.total_charge_budget,
        scalar_replicas,
    )

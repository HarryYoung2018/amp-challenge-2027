"""Reconstruct grouped feature scoring without calling its producer or refitting."""

from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
    CONFIG_SHA256,
    GroupedNativePosterior,
)
from amp_challenge.models.charged_probability_learner import GaussianLearnerSnapshot
from amp_challenge.representations.run_feature_cache_records import (
    canonical,
    json_object,
    pin,
    public_document,
    require,
    sequences_checked,
    sha256,
)


def verify_grouped_native_posterior(
    result,
    *,
    expected_binding,
    expected_configuration_sha256,
    expected_source_sha256,
    expected_sequences,
    expected_raw_feature_receipt,
    learner,
    feasibility,
    feasibility_source_sha256,
):
    """Bind exact group slices to externally verified feature bytes and learner.

    The caller authenticates the raw feature receipt against the feature ledger,
    the learner against charged history, and group sequences against native paths.
    This reader checks those connections and independently calculates eight-row
    means; it does not attest the underlying ESM execution or external timing.
    """
    require(type(result) is GroupedNativePosterior, "exact grouped result required")
    result.__post_init__()
    payload, batches = result.record_payload, canonical([asdict(g) for g in result.groups])
    require(
        type(expected_binding) is FrozenNativePosteriorBinding
        and type(learner) is GaussianLearnerSnapshot
        and expected_configuration_sha256 == CONFIG_SHA256
        and pin(expected_source_sha256)
        and pin(feasibility_source_sha256)
        and callable(feasibility)
        and getattr(feasibility, "source_sha256", None) == feasibility_source_sha256,
        "grouped scoring external identity differs",
    )
    expected_binding.__post_init__()
    learner.__post_init__()
    binding_bytes = canonical(asdict(expected_binding))
    learner_pin = learner.numerical_sha256
    require(
        learner_pin == expected_binding.posterior_sha256
        and learner.history_sha256 == expected_binding.history_sha256
        and learner.objective_context_sha256 == expected_binding.objective_context_sha256
        and learner.transform.representation == "esm320_plus_normalized_length",
        "grouped learner binding differs",
    )
    require(
        type(expected_sequences) is tuple
        and 1 <= len(expected_sequences) <= 10
        and all(type(group) is tuple and len(group) == 8 for group in expected_sequences),
        "grouped scoring requires original eight-row groups",
    )
    sequences = tuple(seq for group in expected_sequences for seq in group)
    sequences_checked(sequences)
    record = json_object(payload)
    require(
        set(record)
        == {
            "artifact",
            "configuration_sha256",
            "source_sha256",
            "binding",
            "raw_feature_receipt",
            "raw_feature_receipt_sha256",
            "raw_matrix",
            "groups",
        }
        and record["artifact"] == "native_tr2_grouped_posterior_v1"
        and record["configuration_sha256"] == expected_configuration_sha256
        and record["source_sha256"] == expected_source_sha256
        and canonical(record["binding"]) == binding_bytes,
        "grouped record scope differs",
    )
    raw_document = json_object(expected_raw_feature_receipt)
    raw_pin = sha256(expected_raw_feature_receipt)
    require(
        canonical(record["raw_feature_receipt"]) == expected_raw_feature_receipt
        and record["raw_feature_receipt_sha256"] == raw_pin,
        "grouped raw feature receipt differs",
    )
    matrix = np.asarray(record["raw_matrix"], dtype=np.float64)
    require(
        matrix.shape == (len(sequences), 321) and np.isfinite(matrix).all(),
        "grouped raw matrix differs",
    )
    row_pins = tuple(
        sha256(
            b"amp/run-feature-row/v1\0"
            + canonical({"dtype": "<f8", "width": 321})
            + np.asarray(row, dtype="<f8", order="C").tobytes()
        )
        for row in matrix
    )
    require(
        public_document(
            sequences,
            "esm_length",
            row_pins,
            context_sha256=expected_binding.objective_context_sha256,
            history_sha256=expected_binding.history_sha256,
        )
        == raw_document,
        "grouped raw rows/order differ from external feature receipt",
    )
    require(
        len(record["groups"]) == len(result.groups) == len(expected_sequences),
        "grouped slice count differs",
    )
    coefficients = learner.backend.coefficient_mean + np.einsum(
        "dor,r->do", learner.backend.coefficient_factor, learner.backend.latent_mean
    )
    for index, (sequences8, batch, saved) in enumerate(
        zip(expected_sequences, result.groups, record["groups"], strict=True)
    ):
        start, stop = index * 8, (index + 1) * 8
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            transformed = np.clip(
                (matrix[start:stop] - learner.transform.mean) / learner.transform.scale, -8.0, 8.0
            )
            means = np.clip(np.column_stack((np.ones(8), transformed)) @ coefficients, 0.0, 1.0)
        flags = feasibility(sequences8)
        require(
            type(flags) is tuple
            and len(flags) == 8
            and all(type(value) is bool for value in flags),
            "grouped feasibility differs",
        )
        scores = [
            {"objectives": list(map(float, mean)), "feasible": flag}
            for mean, flag in zip(means, flags, strict=True)
        ]
        ids = sequences_checked(sequences8)
        expected = {
            "artifact": "native_tr2_grouped_posterior_slice_v1",
            "configuration_sha256": expected_configuration_sha256,
            "source_sha256": expected_source_sha256,
            "binding": asdict(expected_binding),
            "raw_feature_receipt_sha256": raw_pin,
            "row_start": start,
            "row_stop": stop,
            "sequence_ids": list(ids),
            "scores": scores,
        }
        require(
            canonical(saved) == canonical(expected)
            and batch.sequence_ids == ids
            and canonical([asdict(score) for score in batch.scores]) == canonical(scores)
            and batch.receipt_sha256 == sha256(canonical(expected)),
            "grouped slice/score/receipt reconstruction differs",
        )
    require(
        result.record_payload == payload
        and canonical([asdict(g) for g in result.groups]) == batches
        and canonical(asdict(expected_binding)) == binding_bytes
        and learner.numerical_sha256 == learner_pin
        and getattr(feasibility, "source_sha256", None) == feasibility_source_sha256,
        "grouped reader input changed during reconstruction",
    )
    return {
        "reconstructed": True,
        "groups": len(result.groups),
        "rows": len(sequences),
        "record_sha256": sha256(payload),
        "raw_feature_receipt_sha256": raw_pin,
        "learner_independently_refitted": False,
        "external_timing_authenticated": False,
        "scientific_evidence_accepted": False,
    }

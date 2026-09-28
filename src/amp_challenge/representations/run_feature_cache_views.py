"""Thin native/evolution views over the real raw-feature bridge.

Model imports are deliberately isolated here, never in the historical audit.
No feasibility qualification, posterior fit, oracle route or terminal timing
reconciliation is supplied by these views.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import NormalizedObjectiveContext
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureBatch,
    EvolutionFeatureBinding,
)
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorBatch,
    NativePosteriorScore,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.models.charged_probability_learner import (
    GaussianLearnerSnapshot,
    GeneratorFeatureTransform,
)
from amp_challenge.representations.run_feature_cache_bridge import (
    FeatureRows,
    RunFeatureBridge,
    ScopedVisibleFeaturePort,
)
from amp_challenge.representations.run_feature_cache_records import (
    FeatureAssemblyBinding,
    FeatureIntent,
    canonical,
    json_object,
    legacy_history_document,
    pin,
    representation_checked,
    require,
    sequences_checked,
    sha256,
)


def native_evaluator_source_sha256(provider_sha256, feasibility_source_sha256):
    require(
        pin(provider_sha256) and pin(feasibility_source_sha256),
        "native evaluator source pins differ",
    )
    return sha256(
        canonical(
            {
                "artifact": "run_feature_native_mean_view_v1",
                "implementation_source_sha256": provider_sha256,
                "external_feasibility_source_sha256": feasibility_source_sha256,
            }
        )
    )


class NativeFeaturePosterior:
    """Existing clipped point means; externally fixed cheap feasibility only."""

    def __init__(self, port, learner, binding, context, *, feasibility, feasibility_source_sha256):
        require(
            type(port) is ScopedVisibleFeaturePort and type(learner) is GaussianLearnerSnapshot,
            "native feature view requires real scoped bridge and charged learner",
        )
        require(
            type(binding) is FrozenNativePosteriorBinding
            and type(context) is NormalizedObjectiveContext,
            "native feature view requires exact existing bindings",
        )
        self.binding, self.context = binding, context
        self.__port, self.__learner = port, learner
        self.__feasibility = feasibility
        self.__original_feasibility = feasibility
        self.__feasibility_source = feasibility_source_sha256
        self.__binding_payload = canonical(asdict(binding))
        self.__context_payload = canonical(asdict(context))
        self.__port_payload = port.binding_payload
        self.__learner_sha256 = learner.numerical_sha256
        self._guard()

    def _guard(self, *, external_source=True):
        require(
            type(self.binding) is FrozenNativePosteriorBinding
            and type(self.context) is NormalizedObjectiveContext,
            "native feature view binding type drifted",
        )
        self.binding.__post_init__()
        self.context.__post_init__()
        require(
            canonical(asdict(self.binding)) == self.__binding_payload
            and canonical(asdict(self.context)) == self.__context_payload,
            "native feature view binding drifted",
        )
        require(
            type(self.__port) is ScopedVisibleFeaturePort
            and self.__port.binding_payload == self.__port_payload,
            "native scoped feature provider drifted",
        )
        scope = json_object(self.__port_payload)
        learner = self.__learner
        require(type(learner) is GaussianLearnerSnapshot, "native learner type drifted")
        learner.__post_init__()
        require(
            learner.numerical_sha256 == self.__learner_sha256 == self.binding.posterior_sha256
            and learner.history_sha256 == scope["history_sha256"] == self.binding.history_sha256
            and learner.objective_context_sha256
            == scope["objective_context_sha256"]
            == self.binding.objective_context_sha256
            == self.context.context_sha256
            and learner.transform.representation == "esm320_plus_normalized_length"
            and self.binding.feature_source_sha256 == scope["feature_source_sha256"],
            "native learner/history/raw321 feature binding differs",
        )
        require(
            self.__feasibility is self.__original_feasibility
            and callable(self.__feasibility)
            and pin(self.__feasibility_source),
            "native external cheap-feasibility identity differs",
        )
        if external_source:
            require(
                getattr(self.__feasibility, "source_sha256", None) == self.__feasibility_source,
                "native external cheap-feasibility source differs",
            )
        require(
            self.binding.evaluator_source_sha256
            == native_evaluator_source_sha256(scope["provider_sha256"], self.__feasibility_source),
            "native evaluator implementation binding differs",
        )

    def evaluate(self, sequences):
        try:
            return self._evaluate(sequences)
        except BaseException as error:
            self.__port.abort(error)
            raise

    def evaluate_groups(self, groups, *, retain=None):
        """Acquire one TR2 iteration, retaining the original eight-row score calls."""
        from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
            CONFIG_SHA256,
            GroupedNativePosterior,
            batching_source,
        )

        try:
            require(
                type(groups) is tuple
                and 1 <= len(groups) <= 10
                and all(type(group) is tuple and len(group) == 8 for group in groups),
                "TR2 feature groups must retain eight rows per checkpoint",
            )
            sequences = tuple(sequence for group in groups for sequence in group)
            sequences_checked(sequences)
            self._guard()
            source = batching_source()
            raw = self.__port.raw_rows(sequences, "esm_length")
            require(type(raw) is FeatureRows, "native grouped feature result type differs")
            raw.__post_init__()
            raw_payload = raw.receipt_payload
            record = {
                "artifact": "native_tr2_grouped_posterior_v1",
                "configuration_sha256": CONFIG_SHA256,
                "source_sha256": source,
                "binding": asdict(self.binding),
                "raw_feature_receipt": json_object(raw_payload),
                "raw_feature_receipt_sha256": raw.receipt_sha256,
                "raw_matrix": raw.matrix.tolist(),
                "groups": [],
            }
            if retain is not None:
                retain("raw_features", canonical(record))
            results = []
            for index, sequences in enumerate(groups):
                self._guard()
                start, stop = 8 * index, 8 * (index + 1)
                means = self.__learner.clipped_point_means(raw.matrix[start:stop])
                feasible = self.__feasibility(sequences)
                require(
                    type(feasible) is tuple
                    and len(feasible) == 8
                    and all(type(value) is bool for value in feasible)
                    and means.shape == (8, 2)
                    and np.isfinite(means).all()
                    and np.all((means >= 0) & (means <= 1)),
                    "native grouped score/feasibility result differs",
                )
                scores = tuple(
                    NativePosteriorScore(tuple(map(float, mean)), feasible[row])
                    for row, mean in enumerate(means)
                )
                for score in scores:
                    score.validate(self.context)
                item = {
                    "artifact": "native_tr2_grouped_posterior_slice_v1",
                    "configuration_sha256": CONFIG_SHA256,
                    "source_sha256": source,
                    "binding": asdict(self.binding),
                    "raw_feature_receipt_sha256": raw.receipt_sha256,
                    "row_start": start,
                    "row_stop": stop,
                    "sequence_ids": list(sequences_checked(sequences)),
                    "scores": [asdict(score) for score in scores],
                }
                record["groups"].append(item)
                results.append(
                    NativePosteriorBatch(
                        sequences_checked(sequences), scores, sha256(canonical(item))
                    )
                )
                if retain is not None:
                    retain("scored_group", canonical(item))
                self._guard()
                self.__port.check()
                self._guard(external_source=False)
                raw.__post_init__()
                require(raw.receipt_payload == raw_payload, "native grouped raw result changed")
            require(batching_source() == source, "TR2 grouped-feature source changed")
            return GroupedNativePosterior(tuple(results), canonical(record))
        except BaseException as error:
            self.__port.abort(error)
            raise

    def _evaluate(self, sequences):
        sequences_checked(sequences)
        self._guard()
        raw = self.__port.raw_rows(sequences, "esm_length")
        require(type(raw) is FeatureRows, "native feature result type differs")
        raw.__post_init__()
        raw_payload = raw.receipt_payload
        means = self.__learner.clipped_point_means(raw.matrix)
        feasible = self.__feasibility(sequences)
        require(
            type(feasible) is tuple
            and len(feasible) == len(sequences)
            and all(type(value) is bool for value in feasible),
            "native cheap feasibility result differs",
        )
        require(
            means.shape == (len(sequences), 2)
            and np.isfinite(means).all()
            and np.all((means >= 0) & (means <= 1)),
            "native clipped means differ",
        )
        scores = tuple(
            NativePosteriorScore(tuple(map(float, mean)), feasible[index])
            for index, mean in enumerate(means)
        )
        for score in scores:
            score.validate(self.context)
        receipt = {
            "artifact": "run_feature_native_posterior_rows_v1",
            "binding": asdict(self.binding),
            "raw_receipt_sha256": raw.receipt_sha256,
            "sequence_ids": list(sequences_checked(sequences)),
            "scores": [asdict(score) for score in scores],
            "oracle_calls": 0,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }
        frozen_receipt = canonical(receipt)
        result = NativePosteriorBatch(sequences_checked(sequences), scores, sha256(frozen_receipt))
        # An external source property can itself perform work or mutate aliases.
        # Read it before the source/I/O + last clock boundary, never after it.
        self._guard()
        self.__port.check()
        self._guard(external_source=False)
        raw.__post_init__()
        require(
            raw.receipt_payload == raw_payload, "native raw result changed during posterior scoring"
        )
        require(
            type(result) is NativePosteriorBatch
            and type(result.sequence_ids) is tuple
            and result.sequence_ids == sequences_checked(sequences)
            and type(result.scores) is tuple
            and len(result.scores) == len(sequences)
            and all(type(score) is NativePosteriorScore for score in result.scores)
            and result.receipt_sha256 == sha256(frozen_receipt)
            and canonical(
                receipt
                | {
                    "sequence_ids": list(result.sequence_ids),
                    "scores": [asdict(score) for score in result.scores],
                }
            )
            == frozen_receipt,
            "native returned scores differ from original sealed result",
        )
        return result


class EvolutionRawFeatureProvider:
    """Raw321/raw353 only: existing evolution posterior applies its transform."""

    def __init__(self, port, binding, transform):
        require(
            type(port) is ScopedVisibleFeaturePort
            and type(binding) is EvolutionFeatureBinding
            and type(transform) is GeneratorFeatureTransform,
            "evolution raw feature view types differ",
        )
        self.binding = binding
        self.__port, self.__transform = port, transform
        self.__binding_payload = canonical(asdict(binding))
        self.__port_payload = port.binding_payload
        self._guard()

    def _guard(self):
        require(
            type(self.binding) is EvolutionFeatureBinding
            and canonical(asdict(self.binding)) == self.__binding_payload,
            "evolution feature binding drifted",
        )
        self.binding.__post_init__()
        require(
            type(self.__port) is ScopedVisibleFeaturePort
            and self.__port.binding_payload == self.__port_payload,
            "evolution scoped feature provider drifted",
        )
        scope = json_object(self.__port_payload)
        require(
            type(self.__transform) is GeneratorFeatureTransform
            and self.__transform.sha256 == self.binding.transform_sha256
            and self.__transform.representation == self.binding.representation
            and scope["feature_source_sha256"] == self.binding.feature_source_sha256
            and scope["provider_sha256"] == self.binding.provider_sha256,
            "evolution raw feature/source/transform binding differs",
        )

    def evaluate(self, sequences):
        try:
            return self._evaluate(sequences)
        except BaseException as error:
            self.__port.abort(error)
            raise

    def _evaluate(self, sequences):
        sequences_checked(sequences, unique=True)
        self._guard()
        name = "esm_length" if self.binding.width == 321 else "esm_length_spectral"
        representation_checked(name, self.binding.representation)
        raw = self.__port.raw_rows(sequences, name)
        require(type(raw) is FeatureRows, "evolution raw result type differs")
        raw.__post_init__()
        frozen = raw.receipt_payload
        result = EvolutionFeatureBatch(sequences, raw.matrix, raw.receipt_sha256, self.binding)
        self.__port.check()
        self._guard()
        raw.__post_init__()
        require(
            type(result) is EvolutionFeatureBatch
            and type(result.sequences) is tuple
            and result.sequences == sequences
            and type(result.binding) is EvolutionFeatureBinding
            and canonical(asdict(result.binding)) == self.__binding_payload
            and result.receipt_sha256 == raw.receipt_sha256
            and raw.receipt_payload == frozen
            and np.array_equal(result.features, raw.matrix),
            "evolution raw rows changed before return",
        )
        return result


def assemble_charged(bridge, history, transform, authority, intent, *, expected_authority):
    """Return exact existing learner kwargs, not a fit or terminal-clock waiver.

    This bridge deliberately requires eligibility already intersected with
    successful IDs, narrower than fit_charged_learner's general eligibility list.
    """
    try:
        return _assemble_charged(
            bridge, history, transform, authority, intent, expected_authority=expected_authority
        )
    except BaseException as error:
        if type(bridge) is RunFeatureBridge:
            bridge.abort(error, intent=intent)
        raise


def _assemble_charged(bridge, history, transform, authority, intent, *, expected_authority):
    require(
        type(bridge) is RunFeatureBridge
        and type(history) is VerifiedHistorySnapshot
        and type(transform) is GeneratorFeatureTransform
        and type(authority) is FeatureAssemblyBinding
        and type(expected_authority) is FeatureAssemblyBinding
        and type(intent) is FeatureIntent,
        "charged feature view exact types differ",
    )
    history.__post_init__()
    require(
        history.complete
        and canonical(legacy_history_document(asdict(history)))
        == canonical(legacy_history_document(json_object(authority.raw_history_payload)))
        and history.sha256 == authority.history_sha256
        and transform.sha256 == authority.transform_sha256,
        "charged feature view raw history/transform differs",
    )
    frozen_history = history.sha256
    frozen_transform = transform.sha256
    name = (
        "esm_length"
        if transform.representation == "esm320_plus_normalized_length"
        else "esm_length_spectral"
    )
    representation_checked(name, transform.representation)
    raw = bridge.assemble_rows(authority, name, intent, expected_authority=expected_authority)
    require(
        history.sha256 == frozen_history and transform.sha256 == frozen_transform,
        "charged feature history/transform changed during assembly",
    )
    raw.__post_init__()
    return {
        "feature_sequence_ids": sequences_checked(raw.sequences, empty=True, maximum=512),
        "raw_features": raw.matrix,
        "feature_receipt_sha256": raw.receipt_sha256,
        "eligible_query_ids": tuple(authority.eligible_query_ids),
    }

"""TR2 method seam for the existing charged 14+2 campaign loop.

The serial/default bridge retains 128 requests against a 2,800-call worst case.
Explicit grouped collection can opt into the separate 281-request capacity
profile. Full-lifecycle runtime remains unqualified.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.generators.diffusion.native_baseline_context_records import plain
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v4 import NativeTR2D2GuardedV4
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v4_records import source_v4
from amp_challenge.models.charged_probability_learner import fit_charged_learner
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_records import (
    FeatureIntent,
    canonical,
    require,
    sha256,
)
from amp_challenge.representations.run_feature_cache_views import (
    NativeFeaturePosterior,
    assemble_charged,
    native_evaluator_source_sha256,
)

TR2_ARM = "tr2d2_style_tree_offpolicy"


def tr2_contract(common):
    return {
        **{k: v for k, v in common.items() if k not in ("arm_variants", "connection_sha256")},
        "contract": "native_tr2_campaign_v1_20260914",
        "arm": TR2_ARM,
        "selection": "unchanged_tree_pareto_shortlist_first_14_privately_available",
        "connection_sha256": file_digest(Path(__file__)),
        "atomic_source_sha256": source_v4(),
        "full_feature_capacity_qualified": False,
    }


def tr2_state(ensemble):
    return sha256(
        canonical(
            {
                "policies": ensemble.policy_identities,
                "versions": ensemble.versions,
                "history": None
                if ensemble.tree._history is None
                else ensemble.tree._history.sha256,
                "generation": None if ensemble.generation is None else ensemble.generation.sha256,
                "update": None if ensemble.receipt is None else ensemble.receipt.sha256,
                "feature_batching_sha256": ensemble.feature_batching_sha256,
                "feature_batching_source_sha256": ensemble.feature_batching_source_sha256,
                "collection_operation_sha256": None
                if ensemble._replay.collection_operation is None
                else sha256(ensemble._replay.collection_operation),
            }
        )
    )


def tr2_failure(ensemble):
    """Serialize already-paid work without invoking a failed operator or model."""
    return plain(
        {
            "last_update": None
            if ensemble.receipt is None
            else json.loads(ensemble.receipt.record_json),
            "atomic_failure": ensemble.failure,
            "replay_failure": ensemble._replay.failure,
            "versions": ensemble.versions,
            "scientific_evidence_accepted": False,
        }
    )


@dataclass(frozen=True)
class TR2Preparation:
    status: str
    record_payload: bytes

    @property
    def sha256(self):
        return sha256(self.record_payload)


@dataclass(frozen=True)
class TR2Seats(TR2Preparation):
    method_sequences: tuple[str, ...]


class NativeTR2CampaignDriver:
    def __init__(
        self,
        ensemble,
        bridge,
        transform,
        *,
        history_provider_sha256,
        learner_source_sha256,
        eligibility_source_sha256,
        release_source_sha256,
        feasibility,
        feasibility_source_sha256,
        monotonic,
    ):
        require(type(ensemble) is NativeTR2D2GuardedV4, "exact guarded TR2 required")
        require(
            bridge.binding.arm_id == TR2_ARM
            and transform.representation == "esm320_plus_normalized_length",
            "TR2 raw321 representation required",
        )
        require(
            ensemble._replay.clock is monotonic
            and ensemble._replay.scientific_deadline == bridge.binding.original_deadline,
            "TR2 must retain original feature clock",
        )
        require(
            ensemble.matched_feasibility.predicate is feasibility
            and ensemble.matched_feasibility.predicate_sha256 == feasibility_source_sha256,
            "TR2 feasibility source differs",
        )
        self.ensemble, self.bridge, self.transform, self.clock = (
            ensemble,
            bridge,
            transform,
            monotonic,
        )
        self.history_source, self.eligibility_source = (
            history_provider_sha256,
            eligibility_source_sha256,
        )
        self.release_source = release_source_sha256
        self.feasibility, self.feasibility_source = feasibility, feasibility_source_sha256
        self.learner_path = (
            Path(bridge.binding.repository)
            / "src/amp_challenge/models/charged_probability_learner.py"
        )
        self.learner_source = learner_source_sha256
        self.source = file_digest(Path(__file__))
        self.prepared = self.history = None
        self._old_ids, self._old_eligible = frozenset(), frozenset()
        self._guard_state()

    def _guard_state(self):
        self.ensemble._check()
        profile = self.bridge.binding.tr2_grouped_capacity
        require(
            profile is None
            or self.ensemble.feature_batching_sha256
            == profile["feature_batching_configuration_sha256"],
            "TR2 capacity requires pinned grouped feature collection",
        )
        require(
            file_digest(Path(__file__)) == self.source
            and file_digest(self.learner_path) == self.learner_source
            and getattr(self.feasibility, "source_sha256", None) == self.feasibility_source,
            "TR2 driver source changed",
        )

    def _check(self):
        self._guard_state()
        if self.clock() >= self.deadline:
            raise TimeoutError("TR2 original wave deadline exhausted")

    def _intent(self, purpose):
        return FeatureIntent(
            purpose,
            self.history.sha256,
            self.history.objective_context_sha256,
            self.history.round_index,
            0,
            self.bridge.accepted_head,
            self.deadline,
        )

    def prepare_wave(
        self,
        history_callback,
        *,
        round_index,
        poll_ordinal,
        previous_wave_head_sha256,
        original_wave_started_at,
        assembly,
        expected_assembly,
        release,
        expected_release,
    ):
        require(
            self.prepared is None or round_index == self.history.round_index + 1,
            "TR2 preparation cannot repeat",
        )
        self.deadline = min(self.bridge.binding.original_deadline, original_wave_started_at + 180)
        self._check()
        require(
            history_callback.provider_sha256 == self.history_source and poll_ordinal == 0,
            "TR2 history authority differs",
        )
        history = history_callback(previous_wave_head_sha256, 64 + 16 * (round_index - 1))
        require(
            history.complete and history.round_index == round_index and round_index <= 28,
            "TR2 complete adaptive history required",
        )
        require(
            assembly == expected_assembly
            and assembly.history_sha256 == history.sha256
            and assembly.raw_history_payload == canonical(asdict(history))
            and assembly.transform_sha256 == self.transform.sha256
            and assembly.eligibility_source_sha256 == self.eligibility_source,
            "TR2 eligibility differs",
        )
        eligible = frozenset(assembly.eligible_query_ids)
        require(
            eligible <= {r.query_id for r in history.observations if r.status == "successful"}
            and eligible & self._old_ids == self._old_eligible,
            "TR2 successful/previous eligibility changed",
        )
        require(
            release == expected_release and release.source_sha256 == self.release_source,
            "TR2 release authority differs",
        )
        self.history = history
        self.bridge.release_private(
            release, self._intent("release"), expected_release=expected_release
        )
        data = assemble_charged(
            self.bridge,
            history,
            self.transform,
            assembly,
            self._intent("charged"),
            expected_authority=expected_assembly,
        )
        self._check()
        learner = fit_charged_learner(history, self.transform, **data)
        self._check()
        port = self.bridge.scoped_consumer(self._intent("candidate"))
        pins = json.loads(port.binding_payload)
        binding = FrozenNativePosteriorBinding(
            history.sha256,
            history.objective_context_sha256,
            learner.numerical_sha256,
            pins["feature_source_sha256"],
            native_evaluator_source_sha256(pins["provider_sha256"], self.feasibility_source),
        )
        posterior = NativeFeaturePosterior(
            port,
            learner,
            binding,
            self.ensemble.tree.context,
            feasibility=self.feasibility,
            feasibility_source_sha256=self.feasibility_source,
        )
        update = self.ensemble.advance(
            history,
            expected_previous_head_sha256=previous_wave_head_sha256,
            outer_deadline=self.deadline,
        )
        generation = self.ensemble.collect(
            posterior, expected_posterior_sha256=learner.numerical_sha256
        )
        self._check()
        self.prepared = TR2Preparation(
            "prepared",
            canonical(
                {
                    "artifact": "native_tr2_preparation_v1"
                    if self.ensemble.feature_batching_sha256 is None
                    else "native_tr2_preparation_grouped_v1",
                    "round_index": round_index,
                    "history_sha256": history.sha256,
                    "update": json.loads(update.record_json),
                    "generation": asdict(generation),
                    **(
                        {
                            "feature_batching_sha256": self.ensemble.feature_batching_sha256,
                            "feature_batching_source_sha256": self.ensemble.feature_batching_source_sha256,
                            "collection_operation": json.loads(
                                self.ensemble._replay.collection_operation
                            ),
                        }
                        if self.ensemble.feature_batching_sha256 is not None
                        else {}
                    ),
                    "learner": {
                        "fit_query_ids": learner.fit_query_ids,
                        "numerical_sha256": learner.numerical_sha256,
                    },
                    "effective_deadline": self.deadline,
                    "scientific_evidence_accepted": False,
                }
            ),
        )
        self._old_ids = frozenset(r.query_id for r in history.observations)
        self._old_eligible = eligible
        self._check()
        return self.prepared

    def select_wave(self, *, preparation_sha256, allowed_sequence_ids, external_filter_sha256):
        self._check()
        require(
            self.prepared is not None and preparation_sha256 == self.prepared.sha256,
            "TR2 preparation differs",
        )
        shortlist = self.ensemble.generation.collection.shortlisted_sequences
        require(
            allowed_sequence_ids <= frozenset(map(sequence_id, shortlist)),
            "TR2 private filter added rows",
        )
        sequences = tuple(s for s in shortlist if sequence_id(s) in allowed_sequence_ids)[:14]
        status = "selected" if len(sequences) == 14 else "underfilled"
        result = TR2Seats(
            status,
            canonical(
                {
                    "artifact": "native_tr2_seats_v1",
                    "status": status,
                    "preparation_sha256": preparation_sha256,
                    "external_filter_sha256": external_filter_sha256,
                    "method_sequences": sequences,
                    "scientific_evidence_accepted": False,
                }
            ),
            sequences,
        )
        self._check()
        return result

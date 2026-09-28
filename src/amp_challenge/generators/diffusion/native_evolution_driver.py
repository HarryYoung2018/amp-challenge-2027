"""Adaptation-only native full/ablation handoff; no oracle, terminal fit or dispatch.

This unqualified in-process adapter requires the exact callable stored in the
bridge's existing private ``_RunFeatureBridge__clock`` field. Distinct equivalent
wrappers are deliberately rejected. This compatibility dependency is neither a
public bridge API nor a privacy guarantee; native time.monotonic stays unchanged.

Canonical result timing excludes the final post-serialization checkpoint; it
does not authenticate completion time.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_evolution import NativeCounterfactualEnsemble
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureBinding,
    EvolutionFeatureCache,
    FrozenEvolutionPosterior,
)
from amp_challenge.generators.diffusion.native_evolution_records import (
    NO_COUNTERFACTUAL_CONFIG_SHA256,
    EvolutionVariant,
    EvolutionWave,
    evolution_configuration_sha256,
)
from amp_challenge.generators.diffusion.native_matched_feasibility import matched_feasibility_source
from amp_challenge.generators.search.verified_charged_history import read_verified_history
from amp_challenge.models import charged_probability_learner as learner_module
from amp_challenge.models.charged_probability_learner import (
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    LAYOUT_SHA256,
    FeatureAssemblyBinding,
    FeatureIntent,
    PrivateFeatureRelease,
    canonical,
    json_object,
    pin,
    require,
    sha256,
)
from amp_challenge.representations.run_feature_cache_views import (
    EvolutionRawFeatureProvider,
    assemble_charged,
)

MAX_RECORD_BYTES = 128 * 1024**2
MAX_RETAINED_BYTES = 512 * 1024**2
FAILURE_RESERVE_BYTES = 4096
MAX_POLLS = 128
FULL_ARM = "counterfactual_softkg_evolutionary_diffusion"
ARM_VARIANTS = (
    (FULL_ARM, "full"),
    ("ablation_no_spectral_representation", "no_spectral"),
    ("ablation_no_counterfactual_credit", "no_counterfactual"),
    ("ablation_singleton_kg", "singleton_kg"),
    ("ablation_no_endpoint_distillation", "no_endpoint"),
    ("ablation_no_kl_controls", "no_kl"),
)
ABLATION_CONNECTION_CONTRACT = canonical(
    {
        "schema": "amp/native-evolution/ablation-driver/v3",
        "matched_feasibility": "fixed_public_160_pairs_nine_scales_95_percent_per_update_drop_upper_at_most_0.05",
        "matched_feasibility_source_sha256": matched_feasibility_source(),
        "arm_variants": ARM_VARIANTS,
        "no_counterfactual": "neutral_credit_absolute_utility_replay_preserve_beta_yield",
        "no_counterfactual_amendment_sha256": NO_COUNTERFACTUAL_CONFIG_SHA256,
        "no_spectral": "raw321_generator_transform_keep_intercept",
        "singleton_kg": "singleton_actions_same_posterior_and_fantasy_checks",
        "no_endpoint": "skip_endpoint_training_preserve_branch_bookkeeping",
        "no_kl": "disable_kl_enforcement_preserve_diagnostics_and_other_guards",
        "global_successor_protocol_adopted": False,
        "scientific_evidence_accepted": False,
    }
)
ABLATION_CONNECTION_SHA256 = sha256(ABLATION_CONNECTION_CONTRACT)
LEARNER_PATH = "src/amp_challenge/models/charged_probability_learner.py"


@dataclass(frozen=True, slots=True)
class NativeEvolutionPreparation:
    round_index: int
    poll_ordinal: int
    status: str
    history_sha256: str | None
    wave_sha256: str | None
    record_payload: bytes

    @property
    def sha256(self):
        return sha256(b"amp/native-evolution/preparation/v1\0" + self.record_payload)


@dataclass(frozen=True, slots=True)
class NativeEvolutionSeats:
    round_index: int
    preparation_sha256: str
    status: str
    method_sequences: tuple[str, ...]
    record_payload: bytes

    @property
    def sha256(self):
        return sha256(b"amp/native-evolution/seats/v1\0" + self.record_payload)


def _optional(value, expected_type):
    require(value is None or type(value) is expected_type, "driver authority exact type differs")
    return None if value is None else value.document()


class NativeEvolutionDriver:
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
        monotonic=time.monotonic,
    ):
        require(
            type(ensemble) is NativeCounterfactualEnsemble
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform,
            "driver requires actual native ensemble, bridge and generator transform",
        )
        require(
            type(ensemble.variant) is EvolutionVariant
            and (bridge.binding.arm_id, ensemble.variant.name) in ARM_VARIANTS
            and transform.representation == ensemble.variant.representation
            and type(ensemble.cache) is EvolutionFeatureCache
            and type(ensemble.cache.binding) is EvolutionFeatureBinding,
            "driver arm/variant/representation or exact native cache differs",
        )
        require(
            all(
                value is None
                for value in (
                    ensemble.history,
                    ensemble.posterior,
                    ensemble.wave,
                    ensemble.selection,
                    ensemble.deadline,
                )
            )
            and not any(
                (
                    ensemble.branches,
                    ensemble.update_records,
                    ensemble.ingest_attempts,
                    ensemble._seen,
                    ensemble.cache.rows,
                    ensemble.cache.events,
                )
            )
            and not any(ensemble.versions.values())
            and ensemble.cache.requests == ensemble.cache.wave_requests == 0,
            "driver cannot reset a previously used native ensemble/cache",
        )
        self._ensemble, self._bridge, self._transform = ensemble, bridge, transform
        self._cache, self._feasibility, self._clock = ensemble.cache, feasibility, monotonic
        self._pins = (
            history_provider_sha256,
            learner_source_sha256,
            eligibility_source_sha256,
            release_source_sha256,
            feasibility_source_sha256,
        )
        require(all(pin(value) for value in self._pins), "driver source pin differs")
        require(callable(feasibility) and callable(monotonic), "driver callable differs")
        binding = bridge.binding
        require(
            binding.physical_run_limit == 128
            and (ensemble.run_id, ensemble.seed, ensemble.context.context_sha256)
            == (binding.run_id, binding.seed, binding.objective_context_sha256),
            "driver bridge arm/run/context differs",
        )
        expected_features = EvolutionFeatureBinding(
            transform.representation,
            sha256(
                canonical(
                    {
                        "warm_source": json_object(binding.warm_source_payload),
                        "runtime_sha256": sha256(binding.runtime_payload),
                        "model_sha256": binding.model_sha256,
                        "layout_sha256": LAYOUT_SHA256,
                    }
                )
            ),
            transform.sha256,
            sha256(binding.implementation_source_payload),
        )
        require(ensemble.cache.binding == expected_features, "driver feature/transform pins differ")
        require(
            json_object(binding.implementation_source_payload)[LEARNER_PATH] == self._pins[1],
            "driver learner pin differs from fixed bridge source",
        )
        self._identity = self._objects()
        self._binding = self._bindings()
        self._source = sha256(Path(__file__).read_bytes())
        self._last_time, self._deadline = binding.original_epoch, binding.original_deadline
        self._timing, self._active = [], None
        self._history, self._history_bytes = None, None
        self._last_rows, self._old_query_ids, self._old_eligible = [], frozenset(), frozenset()
        self._requests, self._preparations, self._selections, self._waves = {}, {}, {}, {}
        self._next_round, self._current, self._prior_seats, self._prior_head = 1, None, (), None
        self._stage, self._info, self._busy = "constructor", {}, False
        self._selection_wave = None
        self._numerical_work = {}
        self.stopped, self.retained_bytes, self.last_failure = False, 0, None
        self._guard()
        self._constructed_at = self._now("constructor")

    def _objects(self):
        return (
            self._ensemble,
            self._bridge,
            self._transform,
            self._ensemble.cache,
            self._cache,
            self._feasibility,
            self._clock,
        )

    def _bindings(self):
        return canonical(
            {
                "connection_sha256": ABLATION_CONNECTION_SHA256,
                "bridge": self._bridge.binding.document(),
                "pins": self._pins,
                "run": (self._ensemble.run_id, self._ensemble.seed, self._ensemble.bundle),
                "context": asdict(self._ensemble.context),
                "variant": asdict(self._ensemble.variant),
                "configuration_sha256": evolution_configuration_sha256(self._ensemble.variant.name),
                "features": asdict(self._ensemble.cache.binding),
                "transform": self._transform.sha256,
            }
        )

    def _guard_state(self):
        # No supplied callable or advertised-source getter belongs in this tail.
        require(
            all(left is right for left, right in zip(self._identity, self._objects(), strict=True))
            and self._binding == self._bindings(),
            "driver fixed identity/binding drift",
        )
        require(
            getattr(self._bridge, "_RunFeatureBridge__clock", None) is self._clock,
            "driver and bridge must use the same absolute clock callable",
        )
        require(
            sha256(Path(__file__).read_bytes()) == self._source
            and sha256(Path(learner_module.__file__).read_bytes()) == self._pins[1],
            "driver/learner source changed",
        )
        if self._history is not None:
            require(
                canonical(asdict(self._history)) == self._history_bytes,
                "driver charged history changed during work",
            )
        if self._active is not None:
            arguments, expected = self._active
            require(
                self._fingerprint(arguments) == expected, "driver authority changed during work"
            )
        if self._selection_wave is not None:
            wave, expected = self._selection_wave
            require(
                type(self._ensemble.wave) is EvolutionWave
                and self._ensemble.wave is wave
                and wave.sha256 == expected,
                "driver prepared native wave changed",
            )

    def _guard(self):
        self._guard_state()
        require(
            getattr(self._feasibility, "source_sha256", self._pins[4]) == self._pins[4],
            "driver advertised public-feasibility source changed",
        )
        self._guard_state()

    def _now(self, phase):
        before = time.monotonic()
        value = self._clock()
        self._guard()
        # The trusted native clock also charges the callback-free tail itself.
        after = time.monotonic()
        require(
            type(value) in (int, float)
            and math.isfinite(value)
            and before <= value <= after
            and value >= self._last_time,
            "driver clock must be finite, nondecreasing and in the actual absolute domain",
        )
        self._last_time = value
        self._timing.append((phase, value))
        if after >= self._deadline:
            raise TimeoutError("driver original scientific/wave deadline exhausted")
        return value

    def _checkpoint(self, stage):
        self._stage = stage
        self._guard()
        return self._now(stage)

    def _call(self, stage, function, *args, **kwargs):
        self._checkpoint("before_" + stage)
        result = function(*args, **kwargs)
        if stage in ("learner_fit", "posterior", "ingest", "collect", "selection"):
            self._numerical_work[stage] = result
        self._checkpoint("after_" + stage)
        return result

    @staticmethod
    def _fingerprint(arguments):
        round_index, poll, head, started, assembly, expected_assembly, release, expected_release = (
            arguments
        )
        return sha256(
            canonical(
                {
                    "round": round_index,
                    "poll": poll,
                    "head": head,
                    "started": started,
                    "assembly": _optional(assembly, FeatureAssemblyBinding),
                    "expected_assembly": _optional(expected_assembly, FeatureAssemblyBinding),
                    "release": _optional(release, PrivateFeatureRelease),
                    "expected_release": _optional(expected_release, PrivateFeatureRelease),
                }
            )
        )

    def _intent(self, purpose):
        return FeatureIntent(
            purpose,
            self._history.sha256,
            self._ensemble.context.context_sha256,
            self._history.round_index,
            self._bridge.counters.logical_opportunities,
            self._bridge.accepted_head,
            self._deadline,
        )

    def _payload(self, kind, status, **fields):
        return {
            "kind": kind,
            "connection_sha256": ABLATION_CONNECTION_SHA256,
            "arm_id": self._bridge.binding.arm_id,
            "variant": self._ensemble.variant.name,
            "configuration_sha256": evolution_configuration_sha256(self._ensemble.variant.name),
            "status": status,
            **self._info,
            **fields,
            "original_epoch": self._bridge.binding.original_epoch,
            "original_deadline": self._bridge.binding.original_deadline,
            "timing": tuple(self._timing),
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        }

    def _retain(self, kind, status, **fields):
        self._checkpoint("before_record_construction")
        payload = canonical(self._payload(kind, status, **fields))
        require(
            len(payload) <= MAX_RECORD_BYTES
            and self.retained_bytes + len(payload) <= MAX_RETAINED_BYTES - FAILURE_RESERVE_BYTES,
            "driver retained record budget exhausted",
        )
        self._checkpoint("after_record_construction")
        result = self._result(kind, status, payload, fields.get("method_sequences", ()))
        self.retained_bytes += len(payload)
        return result

    def _result(self, kind, status, payload, sequences=()):
        if kind == "preparation":
            return NativeEvolutionPreparation(
                self._info["round_index"],
                self._info["poll_ordinal"],
                status,
                self._info.get("history_sha256"),
                self._info.get("wave_sha256"),
                payload,
            )
        return NativeEvolutionSeats(
            self._info["round_index"], self._current.sha256, status, sequences, payload
        )

    def _failure(self, kind, error):
        self.stopped = True
        # Keep the original numerical objects on the stopped ensemble; never
        # truncate them into a purportedly complete canonical wave/selection.
        document = {
            "kind": kind,
            "status": "stopped_failure",
            "round_index": self._info["round_index"],
            "poll_ordinal": self._info.get("poll_ordinal"),
            "stage": self._stage,
            "error_type": type(error).__name__[:80],
            "error": str(error)[:256],
            "last_checked_monotonic": self._last_time,
            "effective_deadline": self._deadline,
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        }
        for name in (
            "history_sha256",
            "history_receipt_sha256",
            "previous_wave_head_sha256",
            "charged_count",
            "wave_sha256",
            "original_wave_started_at",
        ):
            if name in self._info:
                document[name] = self._info[name]
        if kind == "seats":
            document.update(preparation_sha256=self._current.sha256, method_sequences=())
        payload = canonical(document)
        require(
            len(payload) <= min(FAILURE_RESERVE_BYTES, MAX_RECORD_BYTES)
            and self.retained_bytes + len(payload) <= MAX_RETAINED_BYTES,
            "driver failure reserve cannot be exceeded",
        )
        self.retained_bytes += len(payload)
        self.last_failure = self._result(kind, "stopped_failure", payload)
        return self.last_failure

    def prepare_wave(
        self,
        history_callback,
        *,
        round_index,
        poll_ordinal,
        previous_wave_head_sha256,
        original_wave_started_at,
        assembly=None,
        expected_assembly=None,
        release=None,
        expected_release=None,
    ):
        require(
            type(round_index) is int and 1 <= round_index <= 28,
            "driver supports adaptive rounds 1 through 28 only",
        )
        require(
            type(poll_ordinal) is int and 0 <= poll_ordinal < MAX_POLLS,
            "driver admits at most 128 total polls per wave",
        )
        arguments = (
            round_index,
            poll_ordinal,
            previous_wave_head_sha256,
            original_wave_started_at,
            assembly,
            expected_assembly,
            release,
            expected_release,
        )
        fingerprint, key = self._fingerprint(arguments), (round_index, poll_ordinal)
        if key in self._requests:
            saved, callback, result = self._requests[key]
            require(
                saved == fingerprint and callback is history_callback,
                "changed duplicate preparation",
            )
            return result
        require(not self.stopped and not self._busy, "driver is stopped or busy")
        require(round_index == self._next_round, "driver round was skipped or already prepared")
        prior = self._waves.get(round_index)
        require(
            (prior is None and poll_ordinal == 0)
            or (
                prior is not None
                and poll_ordinal == prior[2] + 1
                and self._preparations[(round_index, prior[2])].status
                == "paused_incomplete_history"
            ),
            "driver poll must follow the last incomplete prefix contiguously",
        )
        self._busy, self._timing, self._active = True, [], (arguments, fingerprint)
        self._info = {
            "round_index": round_index,
            "poll_ordinal": poll_ordinal,
            "original_wave_started_at": original_wave_started_at,
        }
        try:
            self._deadline = self._bridge.binding.original_deadline if prior is None else prior[3]
            now = self._checkpoint("prepare_admission")
            require(
                type(original_wave_started_at) in (int, float)
                and math.isfinite(original_wave_started_at)
                and self._bridge.binding.original_epoch <= original_wave_started_at <= now
                and pin(previous_wave_head_sha256),
                "driver original wave epoch/head differs",
            )
            require(
                prior is None or prior[:2] == (original_wave_started_at, previous_wave_head_sha256),
                "driver resumed wave start/head changed",
            )
            self._deadline = min(self._deadline, original_wave_started_at + 180)
            self._info["effective_deadline"] = self._deadline
            require(
                prior is not None or self._prior_head != previous_wave_head_sha256,
                "next driver round lacks a new request-wave head",
            )
            history = self._call(
                "history",
                read_verified_history,
                history_callback,
                provider_sha256=self._pins[0],
                run_id=self._ensemble.run_id,
                seed=self._ensemble.seed,
                round_index=round_index,
                objective_context_sha256=self._ensemble.context.context_sha256,
                oracle_bundle_sha256=self._ensemble.bundle,
                previous_wave_head_sha256=previous_wave_head_sha256,
            )
            rows = json_object(canonical(asdict(history)))["observations"]
            require(
                len(rows) >= len(self._last_rows)
                and rows[: len(self._last_rows)] == self._last_rows,
                "driver charged prefix shortened or changed",
            )
            if round_index > 1:
                begin = 64 + 16 * (round_index - 2)
                observed = tuple(row["sequence"] for row in rows[begin : begin + 14])
                require(
                    len(self._prior_seats) == 14 and observed == self._prior_seats[: len(observed)],
                    "driver prior fourteen method-seat order changed",
                )
            self._history, self._history_bytes = history, canonical(asdict(history))
            self._last_rows = rows
            self._info.update(
                history_sha256=history.sha256,
                history_receipt_sha256=history.receipt_sha256,
                previous_wave_head_sha256=history.previous_wave_head_sha256,
                charged_count=len(rows),
            )
            self._waves[round_index] = (
                original_wave_started_at,
                previous_wave_head_sha256,
                poll_ordinal,
                self._deadline,
            )
            if not history.complete:
                result = self._retain("preparation", "paused_incomplete_history")
            else:
                result = self._prepare_complete(
                    assembly, expected_assembly, release, expected_release
                )
            if result.status != "paused_incomplete_history":
                self._current = result
                if result.status != "prepared":
                    self.stopped, self.last_failure = True, result
        except BaseException as error:
            result = self._failure("preparation", error)
            if not isinstance(error, Exception):
                self._requests[key] = fingerprint, history_callback, result
                self._preparations[key] = result
                raise
        finally:
            self._busy, self._active = False, None
        self._requests[key] = fingerprint, history_callback, result
        self._preparations[key] = result
        return result

    def _prepare_complete(self, assembly, expected_assembly, release, expected_release):
        self._checkpoint("complete_authorities")
        require(
            all(
                value is not None
                for value in (assembly, expected_assembly, release, expected_release)
            ),
            "complete driver prefix requires assembly and private release authorities",
        )
        history = self._history
        require(
            canonical(assembly.document()) == canonical(expected_assembly.document())
            and assembly.raw_history_payload == self._history_bytes
            and assembly.history_sha256 == history.sha256
            and assembly.transform_sha256 == self._transform.sha256
            and assembly.eligibility_source_sha256 == self._pins[2],
            "driver charged assembly authority differs",
        )
        eligible = frozenset(assembly.eligible_query_ids)
        require(
            eligible <= {row.query_id for row in history.observations if row.status == "successful"}
            and eligible & self._old_query_ids == self._old_eligible,
            "driver successful/previously consumed eligibility changed",
        )
        revealed = history.observations if history.round_index == 1 else history.observations[-2:]
        sequences = tuple(row.sequence for row in revealed)
        require(
            canonical(release.document()) == canonical(expected_release.document())
            and (
                release.run_id,
                release.history_sha256,
                release.objective_context_sha256,
                release.source_sha256,
                release.revealed_sequence_ids,
            )
            == (
                history.run_id,
                history.sha256,
                history.objective_context_sha256,
                self._pins[3],
                tuple(map(sequence_id, sequences)),
            ),
            "driver private release authority/subset differs",
        )
        if history.round_index > 1:
            require(
                all(seq in self._cache.rows for seq in self._prior_seats),
                "driver previously selected method features are missing",
            )
        self._call(
            "release",
            self._bridge.release_private,
            release,
            self._intent("release"),
            expected_release=expected_release,
        )
        purpose = "initial" if history.round_index == 1 else "charged"
        provider = EvolutionRawFeatureProvider(
            self._bridge.scoped_consumer(self._intent(purpose)),
            self._cache.binding,
            self._transform,
        )
        batch = self._call("released_features", provider.evaluate, sequences)
        self._call("cache_preload", self._cache.preload, batch)
        assembled = self._call(
            "assembly",
            assemble_charged,
            self._bridge,
            history,
            self._transform,
            assembly,
            self._intent(purpose),
            expected_authority=expected_assembly,
        )
        learner = self._call(
            "learner_fit", fit_charged_learner, history, self._transform, **assembled
        )
        posterior = self._call(
            "posterior",
            FrozenEvolutionPosterior,
            learner.backend,
            history_sha256=history.sha256,
            context_sha256=history.objective_context_sha256,
            learner_source_sha256=self._pins[1],
            observation_noise=np.eye(2) * 0.01,
            feature_binding=self._cache.binding,
            transform=learner.transform,
        )
        update_start = len(self._ensemble.update_records)
        self._call(
            "ingest",
            self._ensemble.ingest,
            history,
            posterior,
            expected_previous_head_sha256=history.previous_wave_head_sha256,
            eligible_charged_ids=frozenset(
                sequence_id(row.sequence)
                for row in history.observations
                if row.query_id in eligible
            ),
            outer_deadline=self._deadline,
            feasible=self._feasibility,
            feasibility_source_sha256=self._pins[4],
        )
        provider = EvolutionRawFeatureProvider(
            self._bridge.scoped_consumer(self._intent("candidate")),
            self._cache.binding,
            self._transform,
        )
        wave = self._call(
            "collect",
            self._ensemble.collect,
            provider,
            feasible=self._feasibility,
            feasibility_source_sha256=self._pins[4],
        )
        self._info["wave_sha256"] = wave.sha256
        result = self._retain(
            "preparation",
            "prepared" if wave.status == "complete" else wave.status,
            assembly_sha256=assembly.sha256,
            release_sha256=release.sha256,
            feature_accepted_head=self._bridge.accepted_head,
            feature_counters=self._bridge.counters.document(),
            learner={
                "numerical_sha256": learner.numerical_sha256,
                "feature_receipt_sha256": learner.feature_receipt_sha256,
                "fit_query_ids": learner.fit_query_ids,
                "fit_sequence_ids": learner.fit_sequence_ids,
            },
            policy_identities=self._ensemble.policy_identities,
            updates=self._ensemble.update_records[update_start:],
            wave=asdict(wave),
        )
        self._old_query_ids = frozenset(row.query_id for row in history.observations)
        self._old_eligible, self._prior_head = eligible, history.previous_wave_head_sha256
        return result

    def select_wave(self, *, preparation_sha256, allowed_sequence_ids, external_filter_sha256):
        require(
            pin(preparation_sha256)
            and pin(external_filter_sha256)
            and type(allowed_sequence_ids) is frozenset
            and all(pin(key) for key in allowed_sequence_ids),
            "driver selection authority shape differs",
        )
        fingerprint = sha256(
            canonical((preparation_sha256, sorted(allowed_sequence_ids), external_filter_sha256))
        )
        if preparation_sha256 in self._selections:
            expected, result = self._selections[preparation_sha256]
            require(expected == fingerprint, "changed duplicate selection")
            return result
        require(
            not self.stopped
            and not self._busy
            and self._current is not None
            and self._current.status == "prepared"
            and self._current.sha256 == preparation_sha256,
            "driver selection lacks its current complete preparation",
        )
        self._busy, self._timing = True, []
        self._selection_wave = self._ensemble.wave, self._current.wave_sha256
        try:
            self._checkpoint("selection_admission")
            selection = self._call(
                "selection",
                self._ensemble.select,
                allowed_sequence_ids=allowed_sequence_ids,
                external_filter_sha256=external_filter_sha256,
            )
            sequences = ()
            if selection.status == "selected":
                sequences = tuple(
                    self._ensemble.wave.attempts[index].trace.endpoint
                    for index in selection.selected_ordinals
                )
                require(
                    len(sequences) == len(set(sequences)) == 14,
                    "driver selection did not produce fourteen distinct method seats",
                )
            result = self._retain(
                "seats",
                selection.status,
                preparation_sha256=preparation_sha256,
                allowed_sequence_ids=tuple(sorted(allowed_sequence_ids)),
                external_filter_sha256=external_filter_sha256,
                selection=asdict(selection),
                method_sequences=sequences,
            )
            if sequences:
                self._prior_seats, self._next_round = sequences, self._current.round_index + 1
            else:
                self.stopped, self.last_failure = True, result
        except BaseException as error:
            result = self._failure("seats", error)
            if not isinstance(error, Exception):
                self._selections[preparation_sha256] = fingerprint, result
                raise
        finally:
            self._busy, self._selection_wave = False, None
        self._selections[preparation_sha256] = fingerprint, result
        return result

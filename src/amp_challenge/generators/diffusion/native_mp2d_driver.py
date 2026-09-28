"""Adaptive MP2D connection, not oracle qualification or a campaign controller.

The exact bridge private clock callable is an intentional compatibility
dependency, not a public API or privacy guarantee. Native monotonic time is
unchanged. Retained timing excludes the final serialization checkpoint and is
not an authenticated completion timestamp. Arbitrary worker exceptions cannot
recover its unavailable local event inventory; known returned objects survive.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from amp_challenge.generators.diffusion.model import NativeDenoiser, canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_initialization import (
    TRIPLES,
    AuditedNativeInitialization,
)
from amp_challenge.generators.diffusion.native_mp2d_equal_ten import (
    COMPLETE,
    CONFIG_SHA256,
    EqualTenMP2DWave,
    run_native_mp2d_equal_ten,
    source_identities,
)
from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.search.verified_charged_history import read_verified_history
from amp_challenge.models import charged_probability_learner as learner_module
from amp_challenge.models.charged_probability_learner import (
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    CANDIDATE_LIMITS,
    FeatureAssemblyBinding,
    FeatureIntent,
    PrivateFeatureRelease,
    canonical,
    finite_clock,
    json_object,
    pin,
    require,
    sha256,
)
from amp_challenge.representations.run_feature_cache_views import (
    NativeFeaturePosterior,
    assemble_charged,
    native_evaluator_source_sha256,
)

MAX_RECORD_BYTES = 128 * 1024**2
MAX_RETAINED_BYTES = 512 * 1024**2
FAILURE_RESERVE_BYTES = 4096
MAX_POLLS = 128
MP2D_ARM = "mp2d_style_inference_search"
LEARNER_PATH = "src/amp_challenge/models/charged_probability_learner.py"
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
WORKER_STOPS = frozenset(
    {
        "partial_assigned_attempt_cap",
        "partial_bootstrap_root_cap",
        "partial_fixed_score_admission_cap",
        "partial_deadline_stop",
        "partial_attempt_cap",
        "partial_posterior_cap",
    }
)


@dataclass(frozen=True, slots=True)
class NativeMP2DPreparation:
    round_index: int
    poll_ordinal: int
    status: str
    history_sha256: str | None
    wave_sha256: str | None
    record_payload: bytes

    @property
    def sha256(self):
        return sha256(b"amp/native-mp2d/preparation/v1\0" + self.record_payload)


@dataclass(frozen=True, slots=True)
class NativeMP2DSeats:
    round_index: int
    preparation_sha256: str
    status: str
    method_sequences: tuple[str, ...]
    record_payload: bytes

    @property
    def sha256(self):
        return sha256(b"amp/native-mp2d/seats/v1\0" + self.record_payload)


def _optional(value, exact_type):
    require(value is None or type(value) is exact_type, "driver authority exact type differs")
    return None if value is None else value.document()


class NativeMP2DDriver:
    def __init__(
        self,
        initializations,
        corpora,
        context,
        bridge,
        transform,
        *,
        history_provider_sha256,
        oracle_bundle_sha256,
        learner_source_sha256,
        eligibility_source_sha256,
        release_source_sha256,
        feasibility,
        feasibility_source_sha256,
        monotonic=time.monotonic,
    ):
        require(
            type(initializations) is tuple
            and len(initializations) == 10
            and all(type(item) is AuditedNativeInitialization for item in initializations)
            and tuple(item.triple for item in initializations) == TRIPLES
            and type(corpora) is tuple
            and len(corpora) == 10
            and all(type(corpus) is tuple for corpus in corpora)
            and type(context) is NormalizedObjectiveContext
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform,
            "driver requires ten ordered native initializations/corpora and actual bindings",
        )
        context.__post_init__()
        transform.__post_init__()
        require(
            transform.representation == "esm320_plus_normalized_length", "driver requires raw321"
        )
        for initialization, corpus in zip(initializations, corpora, strict=True):
            require(
                1 <= len(corpus) <= 1113
                and len(set(corpus)) == len(corpus)
                and all(
                    type(seq) is str and 8 <= len(seq) <= 50 and set(seq) <= ALPHABET
                    for seq in corpus
                )
                and tuple(sorted(map(sequence_id, corpus))) == initialization.training_sequence_ids,
                "driver training corpus/exclusion binding differs",
            )
        self._initializations, self._corpora, self._context = initializations, corpora, context
        self._bridge, self._transform, self._feasibility, self._clock = (
            bridge,
            transform,
            feasibility,
            monotonic,
        )
        self._pins = (
            history_provider_sha256,
            oracle_bundle_sha256,
            learner_source_sha256,
            eligibility_source_sha256,
            release_source_sha256,
            feasibility_source_sha256,
        )
        require(all(pin(value) for value in self._pins), "driver source pin differs")
        require(callable(feasibility) and callable(monotonic), "driver callable differs")
        binding = bridge.binding
        require(
            binding.arm_id == MP2D_ARM
            and CANDIDATE_LIMITS[MP2D_ARM] == (112, 4)
            and binding.objective_context_sha256 == context.context_sha256,
            "driver MP2D arm/context/candidate ceiling differs",
        )
        counters = bridge.counters
        # Existing private/visible acquisitions remain paid in the bridge ledger.
        # No release, charged assembly or candidate opportunity may be reset.
        require(
            counters.candidate_opportunities == counters.assembly_opportunities == 0
            and counters.released_rows == 0,
            "driver cannot reset an already consumed bridge",
        )
        require(
            json_object(binding.implementation_source_payload)[LEARNER_PATH] == self._pins[2],
            "driver learner source differs from bridge inventory",
        )
        self._identity, self._binding = self._objects(), self._bindings()
        self._source, self._worker_sources = (
            sha256(Path(__file__).read_bytes()),
            source_identities(),
        )
        self._model_state = self._models()
        self._last_time, self._deadline = binding.original_epoch, binding.original_deadline
        self._history, self._history_bytes = None, None
        self._last_rows, self._old_query_ids, self._old_eligible = [], frozenset(), frozenset()
        self._requests, self._preparations, self._selections, self._waves = {}, {}, {}, {}
        self._next_round, self._current, self._prior_seats, self._prior_head = 1, None, (), None
        self._wave, self._wave_identity, self._wave_seal = None, None, None
        self._timing, self._active, self._info, self._stage = [], None, {}, "constructor"
        self._numerical_work, self._busy = {}, False
        self.stopped, self.retained_bytes, self.last_failure = False, 0, None
        self._guard()
        self._now("constructor")

    def _objects(self):
        return (
            self._initializations,
            self._corpora,
            self._context,
            self._bridge,
            self._transform,
            self._feasibility,
            self._clock,
            *(item.model for item in self._initializations),
        )

    def _metadata(self):
        return [
            {
                field.name: getattr(item, field.name)
                for field in fields(item)
                if field.name != "model"
            }
            for item in self._initializations
        ]

    def _bindings(self):
        return canonical(
            {
                "bridge": self._bridge.binding.document(),
                "pins": self._pins,
                "context": asdict(self._context),
                "transform": self._transform.sha256,
                "initializations": self._metadata(),
                "corpora": self._corpora,
            }
        )

    def _models(self):
        result = []
        for item in self._initializations:
            require(type(item.model) is NativeDenoiser, "driver requires the exact native denoiser")
            require(
                canonical_model_logical_hash(item.model) == item.checkpoint_logical_sha256,
                "driver caller model weights changed",
            )
            result.append(
                (
                    tuple(
                        (name, id(parameter), parameter.requires_grad)
                        for name, parameter in item.model.named_parameters()
                    ),
                    tuple((name, module.training) for name, module in item.model.named_modules()),
                )
            )
        return tuple(result)

    @staticmethod
    def _seal(wave):
        require(type(wave) is EqualTenMP2DWave, "driver native wave exact type differs")
        return (
            wave.record_json,
            wave.sha256,
            wave.candidates,
            wave.stop_reason,
            wave.scientific_evidence_accepted,
            wave.production_eligible,
            wave.campaign_eligible,
        )

    def _guard_state(self):
        # Callback-free tail: no supplied clocks or advertised-source getters.
        require(
            all(left is right for left, right in zip(self._identity, self._objects(), strict=True))
            and self._binding == self._bindings()
            and self._model_state == self._models(),
            "driver fixed identity/binding/model drift",
        )
        require(
            getattr(self._bridge, "_RunFeatureBridge__clock", None) is self._clock,
            "driver and bridge require the same absolute clock callable",
        )
        require(
            sha256(Path(__file__).read_bytes()) == self._source
            and sha256(Path(learner_module.__file__).read_bytes()) == self._pins[2]
            and source_identities() == self._worker_sources,
            "driver/learner/worker source changed",
        )
        if self._history is not None:
            require(
                canonical(asdict(self._history)) == self._history_bytes,
                "driver charged history changed",
            )
        if self._active is not None:
            arguments, expected = self._active
            require(
                self._fingerprint(arguments) == expected, "driver authority changed during work"
            )
        if self._wave_identity is not None:
            require(
                self._wave is self._wave_identity and self._seal(self._wave) == self._wave_seal,
                "driver sealed native wave changed",
            )

    def _guard(self):
        self._guard_state()
        advertised = getattr(self._feasibility, "source_sha256", None)
        require(
            type(advertised) is str and advertised == self._pins[5],
            "driver advertised feasibility source changed",
        )
        self._guard_state()

    def _now(self, phase):
        before = time.monotonic()
        value = self._clock()
        self._guard()
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
        if stage in ("learner_fit", "posterior", "native_wave"):
            self._numerical_work[(self._info["round_index"], stage)] = result
        if stage == "history":
            self._history, self._history_bytes = result, canonical(asdict(result))
        if stage == "native_wave":
            self._validate_wave(result, kwargs["expected_binding"], kwargs["eligible_query_ids"])
            self._wave, self._wave_identity, self._wave_seal = result, result, self._seal(result)
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
            self._context.context_sha256,
            self._history.round_index,
            self._bridge.counters.logical_opportunities,
            self._bridge.accepted_head,
            self._deadline,
        )

    def _result(self, kind, status, payload, sequences=()):
        if kind == "preparation":
            return NativeMP2DPreparation(
                self._info["round_index"],
                self._info["poll_ordinal"],
                status,
                self._info.get("history_sha256"),
                self._info.get("wave_sha256"),
                payload,
            )
        return NativeMP2DSeats(
            self._info["round_index"], self._current.sha256, status, sequences, payload
        )

    def _retain(self, kind, status, **fields):
        self._checkpoint("before_record_construction")
        payload = canonical(
            {
                "kind": kind,
                "status": status,
                **self._info,
                **fields,
                "original_epoch": self._bridge.binding.original_epoch,
                "original_deadline": self._bridge.binding.original_deadline,
                "timing": tuple(self._timing),
                "scientific_evidence_accepted": False,
                "production_eligible": False,
                "campaign_eligible": False,
            }
        )
        require(
            len(payload) <= MAX_RECORD_BYTES
            and self.retained_bytes + len(payload) <= MAX_RETAINED_BYTES - FAILURE_RESERVE_BYTES,
            "driver retained record budget exhausted",
        )
        self._checkpoint("after_record_construction")
        result = self._result(kind, status, payload, fields.get("method_sequences", ()))
        self.retained_bytes += len(payload)
        return result

    def _failure(self, kind, error):
        self.stopped = True

        # No new callbacks, feature work or clock reads after an admitted failure.
        def bounded(text, limit):
            return text[:limit].encode("ascii", "backslashreplace")[:limit].decode("ascii")

        document = {
            "kind": kind,
            "status": "stopped_failure",
            "round_index": self._info["round_index"],
            "poll_ordinal": self._info.get("poll_ordinal"),
            "stage": self._stage,
            "error_type": bounded(type(error).__name__, 80),
            "error": bounded(str(error), 256),
            "last_checked_monotonic": self._last_time,
            "effective_deadline": self._deadline,
            "native_wave_available": (self._info["round_index"], "native_wave")
            in self._numerical_work,
            "feature_counters": asdict(self._bridge.counters),
            "scientific_evidence_accepted": False,
            "production_eligible": False,
            "campaign_eligible": False,
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
            "driver failure reserve exhausted",
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
            "driver admits at most 128 total polls",
        )
        finite_clock(original_wave_started_at)
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
            self._deadline = min(self._deadline, original_wave_started_at + 120)
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
                run_id=self._bridge.binding.run_id,
                seed=self._bridge.binding.seed,
                round_index=round_index,
                objective_context_sha256=self._context.context_sha256,
                oracle_bundle_sha256=self._pins[1],
                previous_wave_head_sha256=previous_wave_head_sha256,
            )
            rows = json.loads(canonical(asdict(history)))["observations"]
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
            self._history, self._history_bytes, self._last_rows = (
                history,
                canonical(asdict(history)),
                rows,
            )
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
            result = (
                self._retain("preparation", "paused_incomplete_history")
                if not history.complete
                else self._prepare_complete(assembly, expected_assembly, release, expected_release)
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
            "complete driver prefix requires assembly and release authorities",
        )
        history = self._history
        require(
            canonical(assembly.document()) == canonical(expected_assembly.document())
            and assembly.raw_history_payload == self._history_bytes
            and assembly.history_sha256 == history.sha256
            and assembly.transform_sha256 == self._transform.sha256
            and assembly.eligibility_source_sha256 == self._pins[3],
            "driver charged assembly authority differs",
        )
        eligible = frozenset(assembly.eligible_query_ids)
        require(
            eligible <= {row.query_id for row in history.observations if row.status == "successful"}
            and eligible & self._old_query_ids == self._old_eligible,
            "driver successful/consumed eligibility changed",
        )
        revealed = history.observations if history.round_index == 1 else history.observations[-2:]
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
                self._pins[4],
                tuple(sequence_id(row.sequence) for row in revealed),
            ),
            "driver private release authority/subset differs",
        )
        self._call(
            "release",
            self._bridge.release_private,
            release,
            self._intent("release"),
            expected_release=expected_release,
        )
        assembled = self._call(
            "assembly",
            assemble_charged,
            self._bridge,
            history,
            self._transform,
            assembly,
            self._intent("initial" if history.round_index == 1 else "charged"),
            expected_authority=expected_assembly,
        )
        learner = self._call(
            "learner_fit", fit_charged_learner, history, self._transform, **assembled
        )
        require(
            frozenset(learner.fit_query_ids) == eligible, "driver fitted query eligibility differs"
        )
        port = self._call("candidate_port", self._bridge.scoped_consumer, self._intent("candidate"))
        scope = json_object(port.binding_payload)
        binding = FrozenNativePosteriorBinding(
            history.sha256,
            self._context.context_sha256,
            learner.numerical_sha256,
            scope["feature_source_sha256"],
            native_evaluator_source_sha256(scope["provider_sha256"], self._pins[5]),
        )
        posterior = self._call(
            "posterior",
            NativeFeaturePosterior,
            port,
            learner,
            binding,
            self._context,
            feasibility=self._feasibility,
            feasibility_source_sha256=self._pins[5],
        )
        wave = self._call(
            "native_wave",
            run_native_mp2d_equal_ten,
            self._initializations,
            self._corpora,
            history,
            self._context,
            posterior,
            expected_binding=binding,
            eligible_query_ids=frozenset(learner.fit_query_ids),
            deadline=self._deadline,
            clock=self._clock,
        )
        self._info["wave_sha256"] = wave.sha256
        result = self._retain(
            "preparation",
            "prepared" if wave.stop_reason == COMPLETE else "stopped_native",
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
            wave=asdict(wave),
        )
        self._old_query_ids = frozenset(row.query_id for row in history.observations)
        self._old_eligible, self._prior_head = eligible, history.previous_wave_head_sha256
        return result

    def _validate_wave(self, wave, binding, fit_ids):
        self._seal(wave)
        require(
            type(wave.record_json) is str
            and len(wave.record_json.encode()) <= MAX_RECORD_BYTES
            and pin(wave.sha256)
            and sha256(wave.record_json.encode()) == wave.sha256
            and type(wave.candidates) is tuple
            and len(wave.candidates) <= 256
            and len(set(wave.candidates)) == len(wave.candidates)
            and all(
                type(seq) is str and 8 <= len(seq) <= 50 and set(seq) <= ALPHABET
                for seq in wave.candidates
            )
            and type(wave.stop_reason) is str
            and wave.stop_reason in WORKER_STOPS | {COMPLETE}
            and wave.scientific_evidence_accepted is False
            and wave.production_eligible is False
            and wave.campaign_eligible is False,
            "driver native wave fields/hash differ",
        )
        document = json.loads(wave.record_json)
        require(
            type(document) is dict
            and json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
            == wave.record_json,
            "driver native wave JSON is not the worker encoding",
        )
        expected_initializations = [
            {
                "triple": item.triple,
                "model_sha256": item.checkpoint_logical_sha256,
                "checkpoint_sha256": item.checkpoint_file_sha256,
                "audit_sha256": item.audit_sha256,
                "manifest_sha256": item.manifest_sha256,
                "training_ids_sha256": sha256(
                    json.dumps(
                        item.training_sequence_ids,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ),
            }
            for item in self._initializations
        ]
        require(
            document["artifact"] == "native_mp2d_equal_ten_wave_v2"
            and document["config_sha256"] == CONFIG_SHA256
            and document["history_sha256"] == self._history.sha256
            and document["posterior"] == asdict(binding)
            and document["eligible_query_ids"] == sorted(fit_ids)
            and document["source_identities"] == self._worker_sources
            and document["initializations"] == expected_initializations
            and document["candidates"] == list(wave.candidates)
            and document["stop_reason"] == wave.stop_reason
            and document["completed_balanced_wave"] is (wave.stop_reason == COMPLETE)
            and all(
                document[name] is False
                for name in (
                    "scientific_evidence_accepted",
                    "production_eligible",
                    "campaign_eligible",
                )
            )
            and (wave.stop_reason == COMPLETE or not wave.candidates),
            "driver native wave record binding differs",
        )

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
        try:
            self._checkpoint("selection_admission")
            require(
                self._wave.sha256 == self._current.wave_sha256
                and self._wave.stop_reason == COMPLETE,
                "driver selection wave differs from preparation",
            )
            require(
                allowed_sequence_ids <= frozenset(map(sequence_id, self._wave.candidates)),
                "driver filter includes IDs outside the sealed current pool",
            )
            eligible = tuple(
                seq for seq in self._wave.candidates if sequence_id(seq) in allowed_sequence_ids
            )
            sequences = eligible[:14] if len(eligible) >= 14 else ()
            result = self._retain(
                "seats",
                "selected" if sequences else "underfilled",
                preparation_sha256=preparation_sha256,
                allowed_sequence_ids=tuple(sorted(allowed_sequence_ids)),
                external_filter_sha256=external_filter_sha256,
                eligible_pool_count=len(eligible),
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
            self._busy = False
        self._selections[preparation_sha256] = fingerprint, result
        return result

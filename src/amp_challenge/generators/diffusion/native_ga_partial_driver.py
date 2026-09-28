"""Stateful eligible-GA/partial-native adaptation, never a campaign controller.

Private composition and selection authority remain controller-side. The exact
surviving bridge clock is deliberately coupled through its existing private
field; this is not a public bridge API or a privacy guarantee. Fine-grained
prefix resume, terminal fitting and whole-process restoration are unsupported.
Advertised source pins must be ordinary strings: descriptor/dynamic-only source
attributes are unsupported so the final integrity tail need not call them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from inspect import getattr_static
from pathlib import Path, PosixPath

from safetensors.torch import save

from amp_challenge.evaluation import sequential_v2_seals
from amp_challenge.generators.diffusion import native_ga_partial_driver_records as driver_records
from amp_challenge.generators.diffusion.model import NativeDenoiser, _canonical_model_state
from amp_challenge.generators.diffusion.native_baseline_operators import _NativeUnit
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ChargedEndpointEligibility,
    GAEndpointContext,
)
from amp_challenge.generators.diffusion.native_ga_matched_feasibility import (
    GAMatchedRequirement,
    ga_matched_source,
)
from amp_challenge.generators.diffusion.native_ga_partial_arm import (
    READY,
    compose_partial_ga_seats,
    propose_partial_ga_wave,
)
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    CONFIG_SHA256,
    PartialGAWave,
    encode_record,
    plain,
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_driver_v2 import PrivateGACollisions
from amp_challenge.generators.search.peptide_ga_eligible_v3 import generate_eligible_prefix
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    NEW_CONTRACT_SHA256,
    EligibleGAPrefix,
    GAContextEligibility,
    eligible_batch_digest,
    eligible_implementation_sha256,
    prepare_eligible_ga_input,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import canonical_sequence
from amp_challenge.generators.search.verified_charged_history import read_verified_history
from amp_challenge.models import charged_probability_learner as learner_module
from amp_challenge.models.charged_probability_learner import (
    GaussianLearnerSnapshot,
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    CANDIDATE_LIMITS,
    LAYOUT_SHA256,
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

ARM_ID = "ga_endpoint_distillation_no_kg"
MAX_TRANSACTIONS = 128
LEARNER_PATH = "src/amp_challenge/models/charged_probability_learner.py"


def _authority(value, exact_type):
    require(value is None or type(value) is exact_type, "driver authority exact type differs")
    if value is None:
        return None
    return plain(value.document() if hasattr(value, "document") else asdict(value))


def _destination(value):
    require(type(value) in (str, PosixPath), "driver destination must be an ordinary path")
    text = str(value)
    require(0 < len(text) <= 4096, "driver destination length differs")
    path = Path(text)
    require(
        path.is_absolute() and str(path) == text and ".." not in path.parts,
        "driver destination must be canonical and absolute",
    )
    return text


@dataclass(frozen=True, slots=True)
class GAPartialAttachment:
    """One-use live ownership transfer; saved phase bytes cannot construct it."""

    _owner: object
    _nonce: object
    seat_seal_sha256: str
    state_payload: bytes


class NativeGAPartialDriver:
    def __init__(
        self,
        units,
        context,
        bridge,
        transform,
        *,
        selection,
        learner_source_sha256,
        release_source_sha256,
        feasibility,
        feasibility_source_sha256,
        monotonic=time.monotonic,
    ):
        require(
            type(units) is tuple
            and len(units) == 10
            and all(type(unit) is _NativeUnit for unit in units)
            and tuple(unit.triple for unit in units) == TRIPLES
            and type(context) is GAEndpointContext
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform
            and type(selection) is driver_records.GASelectionAuthority,
            "driver requires actual ten units/context/bridge/transform",
        )
        context.__post_init__()
        transform.__post_init__()
        require(
            transform.representation == "esm320_plus_normalized_length",
            "driver requires the frozen raw321 transform",
        )
        require(
            all(
                pin(value)
                for value in (
                    learner_source_sha256,
                    release_source_sha256,
                    feasibility_source_sha256,
                )
            )
            and callable(feasibility)
            and callable(monotonic),
            "driver source/callable differs",
        )
        self._initial_units = self._units = units
        self._context, self._bridge, self._transform = context, bridge, transform
        self._selection, self._selection_binding = selection, None
        self._feasibility, self._clock = feasibility, monotonic
        self._pins = (learner_source_sha256, release_source_sha256, feasibility_source_sha256)
        binding = bridge.binding
        require(
            binding.arm_id == ARM_ID
            and CANDIDATE_LIMITS[ARM_ID] == (56, 2)
            and (binding.run_id, binding.seed, binding.objective_context_sha256)
            == (context.driver.run_id, context.driver.seed, context.objective.context_sha256),
            "driver arm/run/context/candidate binding differs",
        )
        counts = bridge.counters
        require(
            counts.candidate_opportunities == counts.assembly_opportunities == 0
            and counts.released_rows == 0,
            "driver cannot reset consumed feature work",
        )
        self._bridge_sources = json_object(binding.implementation_source_payload)
        self._source_root = Path(__file__).resolve().parents[4]
        require(
            Path(binding.repository) == self._source_root
            and self._bridge_sources[LEARNER_PATH] == learner_source_sha256
            and sha256(Path(learner_module.__file__).read_bytes()) == learner_source_sha256,
            "driver learner source differs from the bridge",
        )
        feature_source = sha256(
            canonical(
                {
                    "warm_source": json_object(binding.warm_source_payload),
                    "runtime_sha256": sha256(binding.runtime_payload),
                    "model_sha256": binding.model_sha256,
                    "layout_sha256": LAYOUT_SHA256,
                }
            )
        )
        require(
            context.feature_source_sha256 == feature_source
            and context.evaluator_source_sha256
            == native_evaluator_source_sha256(
                sha256(binding.implementation_source_payload), feasibility_source_sha256
            ),
            "driver feature/posterior source binding differs",
        )
        self._worker_sources, self._kernel_source = (
            source_identities(),
            eligible_implementation_sha256(),
        )
        require(
            context.driver.implementation_sha256 == _json_hash(self._worker_sources),
            "driver requires the current partial implementation, not the old endpoint arm",
        )
        require(
            set().union(*(unit.initialization.training_sequence_ids for unit in units))
            <= set(context.driver.training_sequence_keys),
            "driver training exclusions differ",
        )
        self._local_sources = {
            Path(name): sha256(Path(name).read_bytes())
            for name in (
                __file__,
                driver_records.__file__,
                sequential_v2_seals.__file__,
                learner_module.__file__,
            )
        }
        self._versions = dict.fromkeys(TRIPLES, 0)
        self._initial_signature = self._unit_signature(units, self._versions)
        self._unit_signature_saved = self._initial_signature
        self._identity = (units, context, bridge, transform, selection, feasibility, monotonic)
        self._binding_payload = self._bindings()
        self._last_time, self._deadline = binding.original_epoch, binding.original_deadline
        self._wave_clock = None
        self._history = self._history_bytes = None
        self._returned_history = self._learner_signature = None
        self._learner = None
        self._last_rows, self._old_queries, self._eligible = (), frozenset(), frozenset()
        self._origins, self._prior_seats, self._prior_head = {}, (), None
        self._native_ordinal, self._next_round, self._transactions = 0, 1, 0
        self._phase_head, self._current, self._pending = None, None, None
        self._requests, self._seat_requests, self._waves = {}, {}, {}
        self._prefix = self._prefix_bytes = self._wave = self._wave_signature = None
        self._pending_signature, self._candidate_units = None, None
        self._numerical_work, self._phase_paths = {}, {}
        self._active, self._active_callback, self._busy = None, None, False
        self._info, self._stage = {}, "constructor"
        self._last_seats = self._last_seat_path = self._next_state_payload = None
        self._completed_bridge = self._attachment = None
        self._bridge_expected = None
        self._current_signature = self._last_seats_signature = None
        self._functions_saved = self._functions()
        self._wave_open = False
        self.stopped, self.detached, self.last_failure = False, False, None
        self._remember_control()
        self._checkpoint("constructor")
        self._selection_binding = driver_records.validate_selection(
            selection, context, checkpoint=self._checkpoint
        )
        self._checkpoint("constructor_authenticated")

    @property
    def units(self):
        return self._units

    @property
    def behavior_versions(self):
        return self._versions.copy()

    @property
    def native_ordinal(self):
        return self._native_ordinal

    @property
    def round_index(self):
        return self._next_round

    @property
    def pending(self):
        return self._pending

    @property
    def phase_head(self):
        return self._phase_head

    @property
    def transactions(self):
        return self._transactions

    def _bindings(self):
        return canonical(
            {
                "context": asdict(self._context),
                "bridge": self._bridge.binding.document(),
                "transform": self._transform.sha256,
                "pins": self._pins,
            }
        )

    @staticmethod
    def _unit_signature(units, versions):
        require(
            type(units) is tuple
            and len(units) == 10
            and all(type(unit) is _NativeUnit for unit in units)
            and tuple(unit.triple for unit in units) == TRIPLES,
            "driver ten-unit order changed",
        )
        result = []
        for unit in units:
            models = (unit.model, unit.reference, unit.initialization.model)
            require(
                all(type(model) is NativeDenoiser for model in models),
                "driver requires exact native denoisers",
            )
            binding = unit_binding(unit, versions[unit.triple])
            result.append(
                (
                    id(unit),
                    id(unit.initialization),
                    binding,
                    tuple(
                        (
                            id(model),
                            tuple(
                                (
                                    name,
                                    id(value),
                                    value.requires_grad,
                                    str(value.dtype),
                                    str(value.device),
                                )
                                for name, value in model.named_parameters()
                            ),
                            tuple((name, child.training) for name, child in model.named_modules()),
                        )
                        for model in models
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _wave_state(wave):
        require(type(wave) is PartialGAWave, "driver returned wave exact type differs")
        return (
            wave.record_json,
            wave.sha256,
            wave.status,
            wave.ranked_sequences,
            wave.next_native_ordinal,
            wave.next_behavior_versions,
            wave.checkpoint_payloads,
            wave.campaign_eligible,
            wave.scientific_evidence_accepted,
            wave.production_eligible,
        )

    @staticmethod
    def _learner_state(learner):
        require(type(learner) is GaussianLearnerSnapshot, "driver actual learner type differs")
        learner.__post_init__()
        return (
            id(learner),
            learner.numerical_sha256,
            learner.history_sha256,
            learner.transform.sha256,
            learner.feature_receipt_sha256,
            learner.fit_query_ids,
            learner.fit_sequence_ids,
            learner.objective_context_sha256,
        )

    def _control(self):
        return (
            self._next_round,
            self._native_ordinal,
            tuple(self._versions.items()),
            self._phase_head,
            id(self._current),
            id(self._pending),
            self._prior_seats,
            self._prior_head,
            self._transactions,
            self.stopped,
            self.detached,
            self._wave_open,
            self._deadline,
            self._wave_clock,
            self._busy,
            id(self._attachment),
            tuple(sorted(self._eligible)),
            tuple(sorted(self._old_queries)),
            canonical({query: asdict(origin) for query, origin in sorted(self._origins.items())}),
        )

    @staticmethod
    def _functions():
        return (
            fit_charged_learner,
            generate_eligible_prefix,
            propose_partial_ga_wave,
            assemble_charged,
            compose_partial_ga_seats,
            read_verified_history,
            NativeFeaturePosterior,
            prepare_eligible_ga_input,
            driver_records.validate_selection,
            driver_records.selection_guard,
            driver_records.encode_document,
            driver_records.publish_preparation,
            driver_records.publish_seats,
            driver_records.readback_phase,
        )

    @staticmethod
    def _result_state(result):
        require(
            type(result)
            in (driver_records.NativeGAPartialPreparation, driver_records.NativeGAPartialSeats),
            "driver retained result exact type changed",
        )
        seal = result.phase_seal
        require(
            seal is None or type(seal) is sequential_v2_seals.PhaseSeal,
            "driver retained phase type changed",
        )
        phase = (
            None
            if seal is None
            else (
                id(seal),
                seal.artifact,
                seal.seal_sha256,
                seal.receipt_sha256,
                seal.predecessor_seals,
                seal.payload_sha256,
                seal.payload_bytes,
                seal.files,
                seal.metadata_json,
            )
        )
        common = (id(result), result.round_index, result.status, result.record_payload, phase)
        if type(result) is driver_records.NativeGAPartialPreparation:
            return (*common, result.poll_ordinal, result.history_sha256, result.wave_sha256)
        return (*common, result.preparation_sha256, result.method_sequences)

    def _remember_control(self):
        self._control_saved = self._control()

    def _guard_state(self):
        # This tail calls no supplied clock/authenticator/advertised-source getter.
        current = (
            self._initial_units,
            self._context,
            self._bridge,
            self._transform,
            self._selection,
            self._feasibility,
            self._clock,
        )
        require(
            all(left is right for left, right in zip(self._identity, current, strict=True))
            and self._bindings() == self._binding_payload
            and self._control() == self._control_saved,
            "driver fixed identity/binding/state changed",
        )
        require(
            all(
                left is right
                for left, right in zip(self._functions_saved, self._functions(), strict=True)
            ),
            "driver consumed callable binding changed",
        )
        expected_deadline = self._bridge.binding.original_deadline
        if self._wave_clock is not None:
            started = self._wave_clock[0]
            expected_deadline = min(
                expected_deadline, self._bridge.binding.original_epoch + 7200, started + 180
            )
            require(
                (
                    self._info.get("original_wave_started_at"),
                    self._info.get("round_index"),
                    self._info.get("poll_ordinal"),
                )
                == self._wave_clock,
                "driver reported original wave clock/round/poll changed",
            )
        require(self._deadline == expected_deadline, "driver effective deadline was renewed")
        require(
            getattr(self._bridge, "_RunFeatureBridge__clock", None) is self._clock,
            "driver and bridge must retain the same absolute clock object",
        )
        advertised = (
            (self._feasibility, "source_sha256", self._pins[2]),
            (
                self._selection.authenticator,
                "source_sha256",
                self._selection.authenticator_source_sha256,
            ),
        )
        if self._active_callback is not None:
            advertised = (
                *advertised,
                (
                    self._active_callback,
                    "provider_sha256",
                    self._context.driver.history_provider_sha256,
                ),
            )
        for target, name, expected in advertised:
            value = getattr_static(target, name, None)
            require(
                type(value) is str and value == expected,
                "driver requires unchanged ordinary advertised source strings, not descriptors",
            )
        require(
            self._unit_signature(self._initial_units, dict.fromkeys(TRIPLES, 0))
            == self._initial_signature
            and self._unit_signature(self._units, self._versions) == self._unit_signature_saved,
            "driver committed/original models changed",
        )
        if self._candidate_units is not None:
            require(
                self._unit_signature(self._candidate_units, dict(self._wave.next_behavior_versions))
                == self._pending_signature,
                "driver pending model binding changed",
            )
        require(
            all(sha256(path.read_bytes()) == value for path, value in self._local_sources.items())
            and source_identities() == self._worker_sources
            and eligible_implementation_sha256() == self._kernel_source
            and all(
                sha256((self._source_root / name).read_bytes()) == value
                for name, value in self._bridge_sources.items()
            ),
            "driver or consumed implementation source changed",
        )
        if self._selection_binding is not None:
            driver_records.selection_guard(self._selection_binding, external_source=False)
        if self._history is not None:
            require(
                canonical(asdict(self._history)) == self._history_bytes,
                "driver raw charged history changed",
            )
        if self._returned_history is not None:
            history, payload = self._returned_history
            require(
                canonical(asdict(history)) == payload,
                "driver just-returned authenticated history changed",
            )
        if self._learner_signature is not None:
            require(
                self._learner_state(self._learner) == self._learner_signature,
                "driver actual returned learner changed",
            )
        if self._active is not None:
            values, fingerprint = self._active
            require(self._fingerprint(values) == fingerprint, "driver active request changed")
        if self._prefix is not None:
            require(
                self._prefix.canonical_bytes() == self._prefix_bytes,
                "driver returned eligible prefix changed",
            )
        if self._wave is not None:
            require(
                self._wave_state(self._wave) == self._wave_signature,
                "driver returned partial wave changed",
            )
        if self._current is not None:
            require(
                self._result_state(self._current) == self._current_signature,
                "driver current preparation changed",
            )
        if self._last_seats is not None:
            require(
                self._result_state(self._last_seats) == self._last_seats_signature,
                "driver completed seats changed",
            )
            require(
                self._next_state_payload
                == self._last_seats.phase_seal.read_payload_bytes("next_state.json"),
                "driver completed live state differs from the sealed proposed state",
            )
        if self._bridge_expected is not None:
            require(
                (self._bridge.accepted_head, self._bridge.counters.document())
                == self._bridge_expected,
                "driver sealed paid feature state changed",
            )

    def _guard(self):
        self._guard_state()
        advertised = getattr(self._feasibility, "source_sha256", None)
        require(
            type(advertised) is str and advertised == self._pins[2],
            "driver advertised feasibility source changed",
        )
        if self._selection_binding is not None:
            driver_records.selection_guard(self._selection_binding)
        if self._active_callback is not None:
            advertised = getattr(self._active_callback, "provider_sha256", None)
            require(
                type(advertised) is str
                and advertised == self._context.driver.history_provider_sha256,
                "driver advertised history source changed",
            )
        self._guard_state()

    def _checkpoint(self, stage):
        self._stage = stage
        self._guard()
        before = time.monotonic()
        value = self._clock()
        self._guard()
        after = time.monotonic()
        finite_clock(value)
        require(
            before <= value <= after and value >= self._last_time,
            "driver clock must be nondecreasing in the actual absolute domain",
        )
        self._last_time = value
        if after >= self._deadline:
            raise TimeoutError("driver original scientific/wave deadline exhausted")
        return value

    def _call(self, stage, function, *args, **kwargs):
        self._checkpoint("before_" + stage)
        result = function(*args, **kwargs)
        if stage in ("history", "learner_fit", "eligible_prefix", "native_wave"):
            self._numerical_work[(self._info["round_index"], stage)] = result
        if stage == "history":
            self._returned_history = result, canonical(asdict(result))
        if stage == "learner_fit":
            self._learner, self._learner_signature = result, self._learner_state(result)
        if stage == "eligible_prefix":
            require(type(result) is EligibleGAPrefix, "driver eligible prefix type differs")
            self._prefix, self._prefix_bytes = result, result.canonical_bytes()
        if stage == "native_wave":
            require(
                type(result) is tuple and len(result) == 2, "driver native return shape differs"
            )
            self._wave, self._wave_signature = result[1], self._wave_state(result[1])
            self._candidate_units = result[0]
            self._pending_signature = self._unit_signature(
                result[0], dict(result[1].next_behavior_versions)
            )
        self._checkpoint("after_" + stage)
        return result

    @staticmethod
    def _fingerprint(values):
        if len(values) == 4:
            destination, preparation, collisions, receipt = values
            require(type(collisions) is PrivateGACollisions, "driver collision exact type differs")
            collisions.__post_init__()
            document = (destination, preparation, asdict(collisions), receipt)
        else:
            destination, round_index, poll, head, started, *authorities = values
            types = (
                GAContextEligibility,
                GAContextEligibility,
                ChargedEndpointEligibility,
                ChargedEndpointEligibility,
                FeatureAssemblyBinding,
                FeatureAssemblyBinding,
                PrivateFeatureRelease,
                PrivateFeatureRelease,
            )
            document = (
                destination,
                round_index,
                poll,
                head,
                started,
                tuple(
                    _authority(value, exact_type)
                    for value, exact_type in zip(authorities, types, strict=True)
                ),
            )
        payload = canonical(document)
        require(len(payload) <= driver_records.MAX_RECORD_BYTES, "driver request byte cap")
        return sha256(payload)

    def _intent(self, purpose):
        return FeatureIntent(
            purpose,
            self._history.sha256,
            self._context.objective.context_sha256,
            self._history.round_index,
            self._bridge.counters.logical_opportunities,
            self._bridge.accepted_head,
            self._deadline,
        )

    def _clock_document(self):
        return {
            "original_epoch": self._bridge.binding.original_epoch,
            "run_deadline": self._bridge.binding.original_deadline,
            "original_wave_started_at": self._info["original_wave_started_at"],
            "effective_deadline": self._deadline,
        }

    def _flags(self):
        return {
            "campaign_eligible": False,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }

    def _final_phase(self, destination, seal):
        actual = driver_records.readback_phase(
            destination,
            expected_seal=seal,
            run_root=self._bridge.binding.run_root,
            checkpoint=self._checkpoint,
            deadline=self._deadline,
        )
        if seal.artifact == driver_records.SEATS_ARTIFACT:
            preparation = self._current.phase_seal
            linked = sequential_v2_seals.verify_phase(
                self._phase_paths[preparation.seal_sha256],
                expected_artifact=driver_records.PREPARATION_ARTIFACT,
                expected_payload_paths=tuple(name for name, _ in preparation.payload_sha256),
                expected_predecessor_seals=dict(preparation.predecessor_seals),
                expected_seal_sha256=preparation.seal_sha256,
            )
            require(linked == preparation, "driver linked preparation/checkpoint phase changed")
        self._guard_state()
        if time.monotonic() >= self._deadline:
            raise TimeoutError("driver final phase/state readback exceeded original deadline")
        return actual

    def _failure(self, kind, error, *, status="stopped_failure"):
        self.stopped = True
        self._remember_control()

        def bounded(value, limit):
            return value[:limit].encode("ascii", "backslashreplace")[:limit].decode("ascii")

        document = {
            "schema_version": 1,
            "artifact": "native_ga_partial_driver_failure_v1",
            "status": status,
            "kind": kind,
            "round_index": self._info.get("round_index", self._next_round),
            "poll_ordinal": self._info.get("poll_ordinal"),
            "stage": self._stage,
            "error_type": bounded(type(error).__name__, 80),
            "error": bounded(str(error), 256),
            "original_wave_started_at": self._info.get("original_wave_started_at"),
            "effective_deadline": self._deadline,
            "last_checked_monotonic": self._last_time,
            "history_sha256": self._info.get("history_sha256"),
            "wave_sha256": self._info.get("wave_sha256"),
            "known_destination": bounded(self._info.get("destination", ""), 256),
            "native_wave_available": (self._info.get("round_index"), "native_wave")
            in self._numerical_work,
            "durable_bytes_alone_authorize_model_commit": False,
            **self._flags(),
        }
        if kind == "seats":
            document["preparation_sha256"] = self._current.sha256
            document["method_sequences"] = ()
        payload = canonical(document)
        require(
            len(payload) <= driver_records.FAILURE_RESERVE_BYTES,
            "driver bounded failure reserve exceeded",
        )
        seal = self._info.get("phase_seal")
        if kind == "preparation":
            result = driver_records.NativeGAPartialPreparation(
                document["round_index"],
                document["poll_ordinal"],
                status,
                document["history_sha256"],
                document["wave_sha256"],
                payload,
                seal,
            )
        else:
            result = driver_records.NativeGAPartialSeats(
                document["round_index"], self._current.sha256, status, (), payload, seal
            )
        self.last_failure = result
        self._numerical_work[(document["round_index"], "failure_exception")] = error
        return result

    def _admit(self):
        require(self._transactions < MAX_TRANSACTIONS, "driver aggregate transaction cap exhausted")
        self._transactions += 1
        self._remember_control()

    def prepare_wave(
        self,
        destination,
        *,
        round_index,
        poll_ordinal,
        history_callback,
        previous_wave_head_sha256,
        original_wave_started_at,
        eligibility=None,
        expected_eligibility=None,
        teacher_eligibility=None,
        expected_teacher_eligibility=None,
        assembly_binding=None,
        expected_assembly_binding=None,
        private_release=None,
        expected_private_release=None,
    ):
        require(not self.detached, "driver owner is detached")
        require(
            type(round_index) is int and 1 <= round_index <= 28,
            "driver supports adaptation rounds 1 through 28 only",
        )
        require(
            type(poll_ordinal) is int and 0 <= poll_ordinal < MAX_TRANSACTIONS,
            "driver poll ordinal exceeds the aggregate transaction bound",
        )
        finite_clock(original_wave_started_at)
        require(pin(previous_wave_head_sha256), "driver previous head differs")
        target = _destination(destination)
        values = (
            target,
            round_index,
            poll_ordinal,
            previous_wave_head_sha256,
            original_wave_started_at,
            eligibility,
            expected_eligibility,
            teacher_eligibility,
            expected_teacher_eligibility,
            assembly_binding,
            expected_assembly_binding,
            private_release,
            expected_private_release,
        )
        key, fingerprint = (round_index, poll_ordinal), self._fingerprint(values)
        if key in self._requests:
            expected, callback, result, state = self._requests[key]
            require(
                expected == fingerprint and callback is history_callback,
                "driver changed duplicate preparation",
            )
            require(self._result_state(result) == state, "driver cached preparation changed")
            return result
        require(
            not self.stopped and not self._busy and self._pending is None,
            "driver is stopped, busy or awaiting seats",
        )
        require(round_index == self._next_round, "driver round was skipped or already consumed")
        prior = self._waves.get(round_index)
        require(
            (prior is None and poll_ordinal == 0)
            or (
                prior is not None
                and poll_ordinal == prior[2] + 1
                and self._requests[(round_index, prior[2])][2].status == "paused_incomplete_history"
            ),
            "driver polls must continue the exact incomplete history",
        )
        self._busy, self._active, self._active_callback = (
            True,
            (values, fingerprint),
            history_callback,
        )
        self._info = {
            "round_index": round_index,
            "poll_ordinal": poll_ordinal,
            "original_wave_started_at": original_wave_started_at,
            "destination": target,
        }
        try:
            self._admit()
            self._bridge_expected = None
            self._deadline = min(
                self._bridge.binding.original_deadline,
                self._bridge.binding.original_epoch + 7200,
                original_wave_started_at + 180,
            )
            self._wave_clock = (original_wave_started_at, round_index, poll_ordinal)
            self._remember_control()
            now = self._checkpoint("prepare_admission")
            require(
                self._bridge.binding.original_epoch <= original_wave_started_at <= now,
                "driver original wave start differs",
            )
            require(
                prior is None or prior[:2] == (original_wave_started_at, previous_wave_head_sha256),
                "driver incomplete poll changed its original start/head",
            )
            require(
                prior is not None or self._prior_head != previous_wave_head_sha256,
                "driver next wave lacks a new authenticated controller head",
            )
            self._prefix = self._prefix_bytes = self._wave = self._wave_signature = None
            self._candidate_units = self._pending_signature = None
            history = self._call(
                "history",
                read_verified_history,
                history_callback,
                provider_sha256=self._context.driver.history_provider_sha256,
                run_id=self._context.driver.run_id,
                seed=self._context.driver.seed,
                round_index=round_index,
                objective_context_sha256=self._context.objective.context_sha256,
                oracle_bundle_sha256=self._context.driver.oracle_bundle_sha256,
                previous_wave_head_sha256=previous_wave_head_sha256,
            )
            rows = history.observations
            require(
                len(rows) >= len(self._last_rows)
                and rows[: len(self._last_rows)] == self._last_rows,
                "driver charged history shortened or changed",
            )
            require(
                not set(self._context.driver.training_sequence_keys).intersection(
                    sha256(row.sequence.encode("ascii")) for row in rows
                ),
                "driver history overlaps training",
            )
            if round_index > 1:
                begin = 64 + 16 * (round_index - 2)
                revealed = tuple(row.sequence for row in rows[begin : begin + 14])
                require(
                    len(self._prior_seats) == 14 and revealed == self._prior_seats[: len(revealed)],
                    "driver prior fourteen-seat order changed",
                )
            self._history, self._history_bytes, self._last_rows = (
                history,
                canonical(asdict(history)),
                rows,
            )
            self._info["history_sha256"] = history.sha256
            self._waves[round_index] = (
                original_wave_started_at,
                previous_wave_head_sha256,
                poll_ordinal,
            )
            self._wave_open = True
            self._remember_control()
            status = "paused_incomplete_history"
            if history.complete:
                status = self._prepare_complete(
                    eligibility,
                    expected_eligibility,
                    teacher_eligibility,
                    expected_teacher_eligibility,
                    assembly_binding,
                    expected_assembly_binding,
                    private_release,
                    expected_private_release,
                )
            self._bridge_expected = (self._bridge.accepted_head, self._bridge.counters.document())
            payload = self._call(
                "preparation_record",
                driver_records.encode_document,
                self._preparation_document(status),
            )
            seal = driver_records.publish_preparation(
                target,
                record_payload=payload,
                prefix=self._prefix,
                wave=self._wave,
                previous_phase_sha256=self._phase_head,
                run_root=self._bridge.binding.run_root,
                checkpoint=self._checkpoint,
                deadline=self._deadline,
            )
            self._info["phase_seal"] = seal
            seal = self._final_phase(target, seal)
            result = driver_records.NativeGAPartialPreparation(
                round_index,
                poll_ordinal,
                status,
                history.sha256,
                None if self._wave is None else self._wave.sha256,
                payload,
                seal,
            )
            self._phase_head = seal.seal_sha256
            self._phase_paths[seal.seal_sha256] = target
            if status != "paused_incomplete_history":
                self._current = result
                self._current_signature = self._result_state(result)
                if status == "prepared":
                    self._pending = (self._candidate_units, self._wave)
                else:
                    self.stopped, self.last_failure = True, result
            self._remember_control()
        except BaseException as error:
            result = self._failure("preparation", error)
            if not isinstance(error, Exception):
                self._requests[key] = (
                    fingerprint,
                    history_callback,
                    result,
                    self._result_state(result),
                )
                raise
        finally:
            self._busy, self._active, self._active_callback = False, None, None
            self._remember_control()
        self._requests[key] = fingerprint, history_callback, result, self._result_state(result)
        return result

    def _prepare_complete(
        self,
        eligibility,
        expected_eligibility,
        teacher,
        expected_teacher,
        assembly,
        expected_assembly,
        release,
        expected_release,
    ):
        self._checkpoint("complete_authorities")
        pairs = (
            (eligibility, expected_eligibility, GAContextEligibility),
            (teacher, expected_teacher, ChargedEndpointEligibility),
            (assembly, expected_assembly, FeatureAssemblyBinding),
            (release, expected_release, PrivateFeatureRelease),
        )
        for actual, expected, exact_type in pairs:
            require(
                type(actual) is exact_type
                and type(expected) is exact_type
                and canonical(_authority(actual, exact_type))
                == canonical(_authority(expected, exact_type)),
                "driver complete authority or independently held expectation differs",
            )
        history, context = self._history, self._context
        eligibility.__post_init__()
        teacher.validate(history, context)
        eligible = eligibility.query_ids
        require(
            eligible <= {row.query_id for row in history.observations if row.status == "successful"}
            and eligible & self._old_queries == self._eligible,
            "driver successful or old-row eligibility changed",
        )
        require(
            (
                eligibility.history_sha256,
                eligibility.objective_context_sha256,
                eligibility.source_sha256,
            )
            == (history.sha256, context.objective.context_sha256, context.eligibility_source_sha256)
            and frozenset(teacher.query_ids) == eligible
            and frozenset(assembly.eligible_query_ids) == eligible
            and assembly.history_sha256 == history.sha256
            and assembly.raw_history_payload == self._history_bytes
            and assembly.transform_sha256 == self._transform.sha256
            and assembly.eligibility_source_sha256 == context.eligibility_source_sha256,
            "driver eligibility/teacher/assembly cross-binding differs",
        )
        incoming = dict(zip(teacher.query_ids, teacher.origins, strict=True))
        require(
            all(
                query not in self._origins or self._origins[query] == origin
                for query, origin in incoming.items()
            ),
            "driver original endpoint origin changed",
        )
        disclosed = history.observations if history.round_index == 1 else history.observations[-2:]
        require(
            (
                release.run_id,
                release.history_sha256,
                release.objective_context_sha256,
                release.source_sha256,
                release.revealed_sequence_ids,
            )
            == (
                history.run_id,
                history.sha256,
                context.objective.context_sha256,
                self._pins[1],
                tuple(sha256(row.sequence.encode("ascii")) for row in disclosed),
            ),
            "driver private release is not the exact newly disclosed initial/reserve subset",
        )
        self._eligibility_record, self._teacher_record = eligibility, teacher
        self._authority_hashes = {
            "eligibility_sha256": sha256(canonical(eligibility.document())),
            "teacher_eligibility_sha256": sha256(canonical(asdict(teacher))),
            "assembly_sha256": assembly.sha256,
            "release_sha256": release.sha256,
        }
        self._call(
            "release",
            self._bridge.release_private,
            release,
            self._intent("release"),
            expected_release=expected_release,
        )
        purpose = "initial" if history.round_index == 1 else "charged"
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
        expected_fit = tuple(
            row.query_id for row in history.observations if row.query_id in eligible
        )
        require(
            learner.fit_query_ids == expected_fit
            and learner.fit_sequence_ids
            == tuple(
                sha256(row.sequence.encode("ascii"))
                for row in history.observations
                if row.query_id in eligible
            ),
            "driver actual learner eligibility/order differs",
        )
        self._learner = learner
        self._kernel = self._call(
            "eligible_input",
            prepare_eligible_ga_input,
            history,
            eligibility,
            configuration_id=context.driver.configuration_id,
            training_sequence_keys=context.driver.training_sequence_keys,
            expected_history_sha256=history.sha256,
            expected_objective_context_sha256=context.objective.context_sha256,
            expected_eligibility_source_sha256=context.eligibility_source_sha256,
            expected_eligible_query_ids=eligible,
        )
        prefix = self._call(
            "eligible_prefix",
            generate_eligible_prefix,
            self._kernel,
            expected_kernel_source_sha256=self._kernel_source,
            expected_eligibility_source_sha256=context.eligibility_source_sha256,
            expected_objective_context_sha256=context.objective.context_sha256,
            expected_history_sha256=history.sha256,
            expected_eligible_query_ids=eligible,
            deadline_monotonic=self._deadline,
            clock=self._clock,
            resume=None,
            max_new_attempts=None,
        )
        require(
            prefix.output_sha256 == eligible_batch_digest(prefix)
            and prefix.input_sha256 == self._kernel.sha256
            and prefix.source_sha256 == self._kernel_source
            and prefix.contract_sha256 == NEW_CONTRACT_SHA256
            and prefix.deadline_monotonic == self._deadline
            and prefix.prior_resume_sha256 is None
            and prefix.prior_resume_attempt_count == 0
            and not prefix.resume_reconstruction_completed,
            "driver prefix returned different source/history/deadline/resume bindings",
        )
        if prefix.status != "complete":
            require(
                prefix.status
                in ("stopped_deadline", "attempt_cap_exhausted", "abstained_no_eligible_parent"),
                "driver unexpected noncomplete prefix without chunking",
            )
            return prefix.status
        binding = FrozenNativePosteriorBinding(
            history.sha256,
            context.objective.context_sha256,
            learner.numerical_sha256,
            context.feature_source_sha256,
            context.evaluator_source_sha256,
        )
        port = self._bridge.scoped_consumer(self._intent("candidate"))
        evaluator = self._call(
            "posterior",
            NativeFeaturePosterior,
            port,
            learner,
            binding,
            context.objective,
            feasibility=self._feasibility,
            feasibility_source_sha256=self._pins[2],
        )
        candidate_units, wave = self._call(
            "native_wave",
            propose_partial_ga_wave,
            self._units,
            self._kernel,
            prefix,
            teacher,
            context=context,
            evaluator=evaluator,
            posterior_binding=binding,
            expected_history_sha256=history.sha256,
            expected_eligible_query_ids=eligible,
            expected_kernel_source_sha256=self._kernel_source,
            expected_eligibility_source_sha256=context.eligibility_source_sha256,
            expected_prefix_deadline=self._deadline,
            expected_behavior_versions=self._versions.copy(),
            native_ordinal=self._native_ordinal,
            deadline=self._deadline,
            enforce_kl=True,
            matched_feasibility=GAMatchedRequirement(
                self._feasibility,
                self._pins[2],
                context.objective.context_sha256,
                ga_matched_source(),
            ),
            clock=self._clock,
        )
        self._validate_wave(candidate_units, wave, binding)
        self._info["wave_sha256"] = wave.sha256
        require(
            self._unit_signature(candidate_units, dict(wave.next_behavior_versions))
            == self._pending_signature,
            "driver returned candidate metadata changed",
        )
        self._old_queries = frozenset(row.query_id for row in history.observations)
        self._eligible, self._origins = eligible, self._origins | incoming
        self._remember_control()
        return "prepared" if wave.status == READY else wave.status

    def _preparation_document(self, status):
        learner = getattr(self, "_learner", None) if self._history.complete else None
        return {
            "schema_version": 1,
            "artifact": driver_records.PREPARATION_ARTIFACT,
            "arm_id": ARM_ID,
            "run_id": self._context.driver.run_id,
            "seed": self._context.driver.seed,
            "round_index": self._info["round_index"],
            "poll_ordinal": self._info["poll_ordinal"],
            "status": status,
            "history": asdict(self._history),
            "history_sha256": self._history.sha256,
            "previous_wave_head_sha256": self._history.previous_wave_head_sha256,
            "previous_phase_sha256": self._phase_head,
            "binding": {
                "context_sha256": sha256(canonical(asdict(self._context))),
                "selection_sha256": self._selection_binding.selection_sha256,
                "driver_source_sha256": self._local_sources[Path(__file__)],
                "learner_source_sha256": self._pins[0],
                "release_source_sha256": self._pins[1],
                "feasibility_source_sha256": self._pins[2],
                "bridge_binding_sha256": self._bridge.binding.sha256,
            },
            "clock": self._clock_document(),
            "authorities": getattr(self, "_authority_hashes", None)
            if self._history.complete
            else None,
            "bridge": {
                "accepted_head": self._bridge.accepted_head,
                "counters": self._bridge.counters.document(),
            },
            "learner": None
            if learner is None
            else {
                "numerical_sha256": learner.numerical_sha256,
                "feature_receipt_sha256": learner.feature_receipt_sha256,
                "fit_query_ids": learner.fit_query_ids,
                "fit_sequence_ids": learner.fit_sequence_ids,
            },
            "units_before": tuple(
                unit_binding(unit, self._versions[unit.triple]) for unit in self._units
            ),
            "native_ordinal": self._native_ordinal,
            "behavior_versions": self._versions.copy(),
            "prefix_sha256": None if self._prefix is None else self._prefix.output_sha256,
            "wave_sha256": None if self._wave is None else self._wave.sha256,
            "checkpoint_sha256s": self._checkpoint_hashes(),
            "error": None,
            "durable_bytes_alone_authorize_model_commit": False,
            **self._flags(),
        }

    def _checkpoint_hashes(self):
        return (
            []
            if self._wave is None
            else [[name, sha256(data)] for name, data in self._wave.checkpoint_payloads]
        )

    def _validate_wave(self, candidate_units, wave, posterior_binding):
        require(
            type(wave) is PartialGAWave and type(wave.record_json) is str,
            "driver native result type differs",
        )
        payload = wave.record_json.encode("utf-8")
        require(len(payload) <= driver_records.MAX_RECORD_BYTES, "driver native wave byte cap")
        raw = json.loads(payload)
        require(
            raw.get("matched_feasibility_requirement")
            == GAMatchedRequirement(
                self._feasibility,
                self._pins[2],
                self._context.objective.context_sha256,
                ga_matched_source(),
            ).document(),
            "driver wave omitted its matched feasibility requirement",
        )
        require(
            type(raw) is dict
            and encode_record(raw) == (wave.record_json, wave.sha256)
            and raw.get("artifact") == "native_ga_partial_wave_v2"
            and raw.get("status") == wave.status
            and all(raw.get(key) is False and getattr(wave, key) is False for key in self._flags()),
            "driver native wave schema/hash/status/qualification differs",
        )
        require(
            wave.status
            in (
                READY,
                "stopped_kl_guard",
                "stopped_policy_guard",
                "stopped_deadline",
                "stopped_integrity_or_numerical_failure",
                "stopped_final_integrity",
                "stopped_diagnostic_byte_cap",
                "abstained_incomplete_native_branch",
                "abstained_insufficient_feasible_candidates",
            ),
            "driver unknown native status",
        )
        require(
            type(wave.next_native_ordinal) is int
            and self._native_ordinal
            <= wave.next_native_ordinal
            <= self._native_ordinal + 65536 - len(self._prefix.prefix.attempts)
            and wave.next_native_ordinal == raw["next_native_ordinal"]
            and type(wave.next_behavior_versions) is tuple
            and tuple(name for name, _ in wave.next_behavior_versions) == TRIPLES
            and dict(wave.next_behavior_versions) == raw["next_behavior_versions"],
            "driver native attempt/version inventory differs",
        )
        versions = dict(wave.next_behavior_versions)
        require(
            all(
                type(value) is int and self._versions[name] <= value <= self._versions[name] + 1
                for name, value in versions.items()
            ),
            "driver native versions are not actual increments",
        )
        if wave.status == "stopped_diagnostic_byte_cap":
            require(
                raw.get("prefix_sha256") == self._prefix.output_sha256,
                "driver compact stopped prefix differs",
            )
        else:
            require(
                raw["configuration_sha256"] == CONFIG_SHA256
                and raw["sources"] == self._worker_sources
                and raw["context"] == plain(asdict(self._context))
                and raw["kernel_input"] == plain(asdict(self._kernel))
                and raw["eligible_prefix"] == plain(asdict(self._prefix))
                and raw["native_ordinal"] == self._native_ordinal
                and raw["old_units"]
                == plain(
                    tuple(unit_binding(unit, self._versions[unit.triple]) for unit in self._units)
                )
                and raw["enforce_kl"] is True
                and raw["selected_tuning_result_available"] is False,
                "driver native source/context/input/old-model bindings differ",
            )
        require(
            type(candidate_units) is tuple
            and len(candidate_units) == 10
            and all(type(unit) is _NativeUnit for unit in candidate_units)
            and tuple(unit.triple for unit in candidate_units) == TRIPLES,
            "driver returned candidate unit inventory differs",
        )
        checkpoints = wave.checkpoint_payloads
        require(
            type(checkpoints) is tuple
            and all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is bytes
                for row in checkpoints
            )
            and sum(len(data) for _, data in checkpoints) <= driver_records.MAX_CHECKPOINT_BYTES,
            "driver native checkpoint payload cap/type differs",
        )
        hashes = [[name, sha256(data)] for name, data in checkpoints]
        if wave.status != "stopped_diagnostic_byte_cap":
            require(raw["checkpoints"] == hashes, "driver raw checkpoint identities differ")
        if wave.status != READY:
            require(
                wave.ranked_sequences == ()
                and raw["ranked_positions"] == []
                and versions == self._versions
                and all(
                    left is right for left, right in zip(candidate_units, self._units, strict=True)
                ),
                "driver stopped wave releases seats or changed committed models",
            )
            return
        pool, positions = raw["pool"], raw["ranked_positions"]
        require(
            type(pool) is list
            and len(pool) == 256
            and all(type(row) is dict and canonical_sequence(row.get("sequence")) for row in pool),
            "driver complete native pool shape differs",
        )
        sequences = tuple(row["sequence"] for row in pool)
        forbidden = set(self._context.driver.training_sequence_keys) | {
            sha256(row.sequence.encode("ascii")) for row in self._history.observations
        }
        require(
            len(set(sequences)) == 256
            and sequences[:128] == self._prefix.accepted_sequences[:128]
            and not forbidden.intersection(sha256(seq.encode("ascii")) for seq in sequences)
            and type(positions) is list
            and 14 <= len(positions) <= 256
            and all(type(index) is int and 0 <= index < 256 for index in positions)
            and len(set(positions)) == len(positions)
            and type(wave.ranked_sequences) is tuple
            and wave.ranked_sequences == tuple(sequences[index] for index in positions),
            "driver sealed eligible pool/ranking differs",
        )
        require(
            raw["posterior_binding"] == plain(asdict(posterior_binding))
            and raw["teacher_eligibility"] == plain(asdict(self._teacher_record))
            and raw["working_models"]
            == [[unit.triple, unit.policy_sha256] for unit in candidate_units]
            and tuple(name for name, _ in checkpoints) == TRIPLES,
            "driver ready posterior/teacher/model/checkpoint binding differs",
        )
        # This is byte/model binding, not the offline arithmetic reconstruction.
        for unit, (_, expected) in zip(candidate_units, checkpoints, strict=True):
            unit_binding(unit, versions[unit.triple])
            require(
                save(_canonical_model_state(unit.model), metadata=None) == expected,
                "driver returned checkpoint bytes differ from actual pending model",
            )

    def commit_method_seats(
        self, destination, *, preparation_sha256, private_collisions, composition_receipt_sha256
    ):
        require(not self.detached, "driver owner is detached")
        require(
            pin(preparation_sha256) and pin(composition_receipt_sha256),
            "driver composition/preparation pin differs",
        )
        target = _destination(destination)
        values = (target, preparation_sha256, private_collisions, composition_receipt_sha256)
        fingerprint = self._fingerprint(values)
        if preparation_sha256 in self._seat_requests:
            expected, result, state = self._seat_requests[preparation_sha256]
            require(expected == fingerprint, "driver changed duplicate seat request")
            require(self._result_state(result) == state, "driver cached seat result changed")
            return result
        require(
            not self.stopped
            and not self._busy
            and self._current is not None
            and self._current.status == "prepared"
            and self._current.sha256 == preparation_sha256
            and self._pending is not None,
            "driver seats lack a current complete preparation",
        )
        self._busy, self._active = True, (values, fingerprint)
        self._info["destination"] = target
        self._info.pop("phase_seal", None)
        try:
            self._admit()
            self._checkpoint("seat_admission")
            sequences = self._call(
                "private_composition",
                compose_partial_ga_seats,
                self._wave,
                self._history,
                private_collisions,
            )
            if len(sequences) != 14:
                result = self._failure(
                    "seats",
                    ValueError("private composition underfilled fourteen seats"),
                    status="stopped_private_underfill",
                )
            else:
                positions = tuple(self._wave.method_pool.index(seq) for seq in sequences)
                require(
                    len(set(sequences)) == 14 and positions == tuple(sorted(set(positions))),
                    "driver private composition changed the sealed rank order",
                )
                document = {
                    "schema_version": 1,
                    "artifact": driver_records.SEATS_ARTIFACT,
                    "arm_id": ARM_ID,
                    "run_id": self._context.driver.run_id,
                    "seed": self._context.driver.seed,
                    "round_index": self._current.round_index,
                    "preparation_sha256": preparation_sha256,
                    "preparation_seal_sha256": self._current.phase_seal.seal_sha256,
                    "history_sha256": self._history.sha256,
                    "status": "selected",
                    "method_sequences": sequences,
                    "ranked_positions": positions,
                    "composition_receipt_sha256": composition_receipt_sha256,
                    "method_seats": 14,
                    "private_reserve_seats": 2,
                    "clock": self._clock_document(),
                    "durable_bytes_alone_authorize_model_commit": False,
                    **self._flags(),
                }
                payload = self._call("seat_record", driver_records.encode_document, document)
                next_state = self._call(
                    "next_state_record",
                    driver_records.encode_document,
                    self._next_state_document(sequences),
                )
                seal = driver_records.publish_seats(
                    target,
                    record_payload=payload,
                    next_state_payload=next_state,
                    preparation_seal_sha256=self._current.phase_seal.seal_sha256,
                    run_root=self._bridge.binding.run_root,
                    checkpoint=self._checkpoint,
                    deadline=self._deadline,
                )
                self._info["phase_seal"] = seal
                seal = self._final_phase(target, seal)
                result = driver_records.NativeGAPartialSeats(
                    self._current.round_index,
                    preparation_sha256,
                    "selected",
                    sequences,
                    payload,
                    seal,
                )
                # No external callback follows the fresh phase read and pure guard.
                self._units, self._versions = (
                    self._candidate_units,
                    dict(self._wave.next_behavior_versions),
                )
                self._unit_signature_saved = self._pending_signature
                self._native_ordinal = self._wave.next_native_ordinal
                self._prior_seats, self._prior_head = (
                    sequences,
                    self._history.previous_wave_head_sha256,
                )
                self._next_round = self._current.round_index + 1
                self._phase_head = seal.seal_sha256
                self._phase_paths[seal.seal_sha256] = target
                self._last_seats, self._last_seat_path, self._next_state_payload = (
                    result,
                    target,
                    next_state,
                )
                self._last_seats_signature = self._result_state(result)
                self._completed_bridge = self._bridge_expected
                self._pending, self._wave_open = None, False
                self._remember_control()
        except BaseException as error:
            result = self._failure("seats", error)
            if not isinstance(error, Exception):
                self._seat_requests[preparation_sha256] = (
                    fingerprint,
                    result,
                    self._result_state(result),
                )
                raise
        finally:
            self._busy, self._active = False, None
            self._remember_control()
        self._seat_requests[preparation_sha256] = fingerprint, result, self._result_state(result)
        return result

    def _next_state_document(self, sequences):
        versions = dict(self._wave.next_behavior_versions)
        return {
            "schema_version": 1,
            "artifact": driver_records.NEXT_STATE_ARTIFACT,
            "state_kind": "proposed_next_state",
            "completed_round_index": self._current.round_index,
            "next_round_index": self._current.round_index + 1,
            "history_sha256": self._history.sha256,
            "previous_wave_head_sha256": self._history.previous_wave_head_sha256,
            "prior_method_sequences": sequences,
            "eligible_query_ids": sorted(self._eligible),
            "origins": [[query, asdict(origin)] for query, origin in sorted(self._origins.items())],
            "native_ordinal": self._wave.next_native_ordinal,
            "behavior_versions": versions,
            "unit_bindings": [
                unit_binding(unit, versions[unit.triple]) for unit in self._candidate_units
            ],
            "checkpoint_phase_sha256": self._current.phase_seal.seal_sha256,
            "checkpoint_sha256s": self._checkpoint_hashes(),
            "bridge_binding_sha256": self._bridge.binding.sha256,
            "feature_accepted_head": self._bridge.accepted_head,
            "feature_counters": self._bridge.counters.document(),
            "original_epoch": self._bridge.binding.original_epoch,
            "run_deadline": self._bridge.binding.original_deadline,
            "selection_sha256": self._selection_binding.selection_sha256,
            "context_sha256": self._context.objective.context_sha256,
            "requires_same_live_completion_capability": True,
            **self._flags(),
        }

    def detach_completed(self, *, expected_seat_seal_sha256):
        require(
            not self.detached
            and not self.stopped
            and not self._busy
            and not self._wave_open
            and self._pending is None
            and self._last_seats is not None
            and self._attachment is None
            and pin(expected_seat_seal_sha256)
            and self._last_seats.phase_seal.seal_sha256 == expected_seat_seal_sha256,
            "driver can detach only its latest completed live seat handoff",
        )
        self._busy = True
        self._remember_control()
        try:
            self._checkpoint("detach_admission")
            require(
                (self._bridge.accepted_head, self._bridge.counters.document())
                == self._completed_bridge,
                "driver bridge advanced since completed handoff",
            )
            self._final_phase(self._last_seat_path, self._last_seats.phase_seal)
            token = GAPartialAttachment(
                self, object(), expected_seat_seal_sha256, self._next_state_payload
            )
            self._attachment, self.detached = token, True
            self._remember_control()
            return token
        except BaseException as error:
            self._failure("seats", error)
            raise
        finally:
            self._busy = False
            self._remember_control()

    @classmethod
    def reattach(cls, attachment, *, bridge, monotonic, expected_seat_seal_sha256):
        require(
            cls is NativeGAPartialDriver and type(attachment) is GAPartialAttachment,
            "driver reattachment requires its exact live capability",
        )
        owner = attachment._owner
        require(
            type(owner) is cls
            and owner.detached
            and not owner.stopped
            and not owner._busy
            and owner._attachment is attachment
            and bridge is owner._bridge
            and monotonic is owner._clock
            and pin(expected_seat_seal_sha256)
            and attachment.seat_seal_sha256
            == expected_seat_seal_sha256
            == owner._last_seats.phase_seal.seal_sha256
            and attachment.state_payload == owner._next_state_payload,
            "driver reattachment capability/owner/bridge/clock/phase differs",
        )
        owner._busy = True
        owner._remember_control()
        try:
            owner._checkpoint("reattach_admission")
            require(
                (bridge.accepted_head, bridge.counters.document()) == owner._completed_bridge,
                "driver reattachment would rewind paid feature work",
            )
            owner._final_phase(owner._last_seat_path, owner._last_seats.phase_seal)
            require(
                owner._attachment is attachment
                and attachment._owner is owner
                and attachment.seat_seal_sha256 == expected_seat_seal_sha256
                and attachment.state_payload == owner._next_state_payload,
                "driver live attachment changed before consumption",
            )
            result = object.__new__(cls)
            result.__dict__ = owner.__dict__.copy()
            for name in (
                "_requests",
                "_seat_requests",
                "_waves",
                "_numerical_work",
                "_phase_paths",
                "_versions",
                "_origins",
                "_info",
            ):
                setattr(result, name, getattr(owner, name).copy())
            owner._attachment = None
            result.detached, result._attachment, result._busy = False, None, False
            result._remember_control()
            return result
        except BaseException as error:
            owner._attachment = None
            owner._failure("seats", error)
            raise
        finally:
            owner._busy = False
            owner._remember_control()

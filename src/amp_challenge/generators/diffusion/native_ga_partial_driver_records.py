"""Private selection and durable partial-GA handoffs, not scientific authority.

The caller supplies the actual selection issuer and the plain-GA common anchor.
Phase seals authenticate captured bytes, not a tuning experiment or its issuer.
Publication and readback are structural only; no numerical verifier is called.
Returned phase bytes alone never authorize model adoption or process recovery.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseBuilder,
    PhaseSeal,
    canonical_json_bytes,
    checksum_manifest_bytes,
    verify_phase,
    verify_phase_capability,
)
from amp_challenge.generators.diffusion.native_ga_endpoint_records import GAEndpointContext
from amp_challenge.generators.diffusion.native_ga_partial_records import PartialGAWave
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3_records import (
    EligibleGADriverContext,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import EligibleGAPrefix
from amp_challenge.generators.search.peptide_ga_records import (
    canonical_json_bytes as kernel_json_bytes,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    CONTRACT_SHA256,
    canonical_sequence,
    digest,
    hash_string,
    parameters,
    require,
)
from amp_challenge.representations.run_feature_cache_records import finite_clock
from amp_challenge.representations.run_feature_cache_verify import inspect_feature_tree

SELECTION_ARTIFACT = "ga_tuning_selection_authority_v1"
PREPARATION_ARTIFACT = "native_ga_partial_preparation_v1"
SEATS_ARTIFACT = "native_ga_partial_seats_v1"
NEXT_STATE_ARTIFACT = "native_ga_partial_next_state_v1"
PROTOCOL_SHA256 = "579d071fac1d70bf514761ff98e6d2019730fbdad484e548f9b672fda1a001af"
MAX_RECORD_BYTES = 128 * 1024**2
MAX_CHECKPOINT_BYTES = 256 * 1024**2
MAX_PHASE_BYTES = 512 * 1024**2
MAX_RUN_BYTES = 5 * 1024**3
FAILURE_RESERVE_BYTES = 4096
_FLAGS = ("campaign_eligible", "scientific_evidence_accepted", "production_eligible")


def encode_document(document):
    """Outer records use phase canonical JSON, including the final LF."""
    payload = canonical_json_bytes(document)
    require(len(payload) <= MAX_RECORD_BYTES, "partial driver diagnostic record cap exceeded")
    return payload


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "partial driver JSON contains duplicate keys")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"partial driver JSON contains non-finite {value}")


def _document(payload, *, maximum, phase_json=True):
    require(
        type(payload) is bytes and 0 < len(payload) <= maximum,
        "partial driver payload type or byte cap differs",
    )
    value = json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)
    encoder = canonical_json_bytes if phase_json else kernel_json_bytes
    require(
        type(value) is dict and encoder(value) == payload,
        "partial driver JSON is not an exact canonical object",
    )
    return value


@dataclass(frozen=True, slots=True)
class GASelectionAuthority:
    """Caller-held common selection evidence; construction does not authenticate it."""

    seal: PhaseSeal
    expected_selection_seal_sha256: str
    expected_study_seal_sha256: str
    plain_ga_selection_seal_sha256: str
    authenticator: Callable
    authenticator_source_sha256: str


@dataclass(frozen=True, slots=True)
class SelectionBinding:
    """A private captured binding, not a transferable issuer certificate."""

    authority: GASelectionAuthority
    context: GAEndpointContext | EligibleGADriverContext
    selection_sha256: str
    expected_semantics: bytes
    _authority_state: tuple
    _seal_state: tuple
    _context_payload: bytes
    _authenticator: Callable
    _seal: PhaseSeal


def _authority_state(authority):
    require(type(authority) is GASelectionAuthority, "GA selection authority exact type differs")
    pins = (
        authority.expected_selection_seal_sha256,
        authority.expected_study_seal_sha256,
        authority.plain_ga_selection_seal_sha256,
        authority.authenticator_source_sha256,
    )
    require(
        all(hash_string(value) for value in pins) and callable(authority.authenticator),
        "GA selection caller pins or authenticator differ",
    )
    require(pins[0] == pins[2], "GA selection differs from the common plain-GA selection")
    return pins


def _selection_seal_state(seal):
    # Bound every supplied payload before the generic capability checker hashes it.
    require(
        type(seal) is PhaseSeal
        and type(seal.payload_bytes) is tuple
        and len(seal.payload_bytes) == 2,
        "GA selection phase exact type/inventory differs",
    )
    expected = (("authority_receipt.bin", 64 * 1024), ("selection.json", 16 * 1024))
    for row, (name, maximum) in zip(seal.payload_bytes, expected, strict=True):
        require(
            type(row) is tuple
            and len(row) == 2
            and type(row[0]) is str
            and row[0] == name
            and type(row[1]) is bytes
            and 0 < len(row[1]) <= maximum,
            "GA selection payload inventory or byte cap differs",
        )
    require(
        type(seal.metadata_json) is bytes and seal.metadata_json == b"{}\n",
        "GA selection metadata must be empty",
    )
    require(
        type(seal.artifact) is str
        and seal.artifact == SELECTION_ARTIFACT
        and hash_string(seal.seal_sha256)
        and hash_string(seal.receipt_sha256)
        and type(seal.predecessor_seals) is tuple
        and len(seal.predecessor_seals) == 1
        and type(seal.payload_sha256) is tuple
        and len(seal.payload_sha256) == 2
        and all(
            type(row) is tuple and len(row) == 2 and type(row[0]) is str and hash_string(row[1])
            for row in (*seal.predecessor_seals, *seal.payload_sha256)
        )
        and type(seal.files) is tuple
        and len(seal.files) == 4
        and all(type(name) is str for name in seal.files),
        "GA selection seal scalar or inventory types differ",
    )
    return (
        seal.artifact,
        seal.seal_sha256,
        seal.receipt_sha256,
        seal.predecessor_seals,
        seal.payload_sha256,
        seal.payload_bytes,
        seal.files,
        seal.metadata_json,
    )


def _selection_expected(authority, context):
    require(
        type(context) in (GAEndpointContext, EligibleGADriverContext),
        "GA selection context exact type differs",
    )
    context.__post_init__()
    driver = context.driver if type(context) is GAEndpointContext else context
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact": SELECTION_ARTIFACT,
            "successor_protocol_sha256": PROTOCOL_SHA256,
            "tuning_contract_sha256": CONTRACT_SHA256,
            "study_seal_sha256": authority.expected_study_seal_sha256,
            "objective_context_sha256": driver.objective_context_sha256,
            "oracle_bundle_sha256": driver.oracle_bundle_sha256,
            "consumer_method_ids": ["tuned_peptide_ga", "ga_endpoint_distillation_no_kg"],
            "selected_configuration_id": driver.configuration_id,
            "selected_parameters": asdict(parameters(driver.configuration_id)),
        }
    )


def selection_guard(binding, *, external_source=True):
    """Check captured state; optional advertised-source getter ends with a pure tail."""
    require(type(binding) is SelectionBinding, "GA selection binding exact type differs")
    authority = binding.authority
    require(
        _authority_state(authority) == binding._authority_state
        and authority.authenticator is binding._authenticator
        and authority.seal is binding._seal
        and _selection_seal_state(authority.seal) == binding._seal_state
        and _selection_expected(authority, binding.context) == binding.expected_semantics
        and canonical_json_bytes(asdict(binding.context)) == binding._context_payload
        and authority.expected_selection_seal_sha256 == binding.selection_sha256,
        "GA selection captured identity or context changed",
    )
    if external_source:
        source = getattr(binding._authenticator, "source_sha256", None)
        require(
            type(source) is str and source == binding._authority_state[3],
            "GA selection authenticator source changed",
        )
        selection_guard(binding, external_source=False)


def validate_selection(selection, context, *, checkpoint):
    """Authenticate once against caller pins; every external call is bracketed."""
    state = _authority_state(selection)
    seal_state = _selection_seal_state(selection.seal)
    expected = _selection_expected(selection, context)
    binding = SelectionBinding(
        selection,
        context,
        state[0],
        expected,
        state,
        seal_state,
        canonical_json_bytes(asdict(context)),
        selection.authenticator,
        selection.seal,
    )
    checkpoint("before_selection_validation")
    selection_guard(binding)
    seal = verify_phase_capability(
        selection.seal,
        expected_artifact=SELECTION_ARTIFACT,
        expected_payload_paths=("authority_receipt.bin", "selection.json"),
        expected_predecessor_seals={"tuning_study": state[1]},
        expected_seal_sha256=state[0],
    )
    payload = seal.read_payload_bytes("selection.json")
    _document(payload, maximum=16 * 1024)
    require(payload == expected, "GA selection schema/configuration/study/context differs")
    receipt = seal.read_payload_bytes("authority_receipt.bin")
    checkpoint("before_selection_authentication")
    selection_guard(binding)
    accepted = binding._authenticator(payload, receipt, expected)
    selection_guard(binding)
    checkpoint("after_selection_authentication")
    selection_guard(binding)
    require(accepted is None, "GA selection authenticator did not accept")
    return binding


@dataclass(frozen=True, slots=True)
class NativeGAPartialPreparation:
    round_index: int
    poll_ordinal: int
    status: str
    history_sha256: str | None
    wave_sha256: str | None
    record_payload: bytes
    phase_seal: PhaseSeal | None

    @property
    def sha256(self):
        return digest(b"amp/native-ga-partial/preparation/v1\0" + self.record_payload)


@dataclass(frozen=True, slots=True)
class NativeGAPartialSeats:
    round_index: int
    preparation_sha256: str
    status: str
    method_sequences: tuple[str, ...]
    record_payload: bytes
    phase_seal: PhaseSeal | None

    @property
    def sha256(self):
        return digest(b"amp/native-ga-partial/seats/v1\0" + self.record_payload)


def _outer(payload, artifact):
    document = _document(payload, maximum=MAX_RECORD_BYTES)
    require(
        type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and document.get("artifact") == artifact
        and all(document.get(flag) is False for flag in _FLAGS),
        "partial driver outer artifact/version/eligibility differs",
    )
    if artifact == NEXT_STATE_ARTIFACT:
        require(
            document.get("state_kind") == "proposed_next_state"
            and document.get("requires_same_live_completion_capability") is True,
            "partial driver next state must require the same live completion capability",
        )
    else:
        require(
            document.get("durable_bytes_alone_authorize_model_commit") is False,
            "partial driver durable bytes cannot authorize adoption",
        )
    return document


def _paths(destination, run_root):
    root, target = Path(run_root), Path(destination)
    require(
        root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(),
        "partial driver run root must be an existing canonical directory",
    )
    require(
        target.is_absolute()
        and target != root
        and target.is_relative_to(root)
        and target.parent.resolve(strict=True) == target.parent,
        "partial driver publication must be a canonical descendant of the bridge run root",
    )
    root_stat, parent_stat = root.stat(), target.parent.stat()
    return (
        root,
        target,
        (root_stat.st_dev, root_stat.st_ino, parent_stat.st_dev, parent_stat.st_ino),
    )


def _path_guard(destination, run_root, identity):
    require(
        _paths(destination, run_root)[2] == identity,
        "partial driver publication root or parent changed",
    )


def _phase_size(artifact, payloads, predecessors):
    hashes = {name: digest(payload) for name, payload in payloads.items()}
    receipt = canonical_json_bytes(
        {
            "artifact": artifact,
            "metadata": {},
            "payloads": hashes,
            "predecessor_seals": predecessors,
            "schema_version": 1,
            "status": "sealed",
        }
    )
    manifest = checksum_manifest_bytes({**hashes, "receipt.json": digest(receipt)})
    return sum(len(payload) for payload in payloads.values()) + len(receipt) + len(manifest)


def _run_admission(root, additional):
    inventory = inspect_feature_tree(root)
    require(
        inventory["regular_bytes"] + additional <= MAX_RUN_BYTES,
        "partial driver shared run-root byte cap exceeded",
    )
    return inventory


def _readback_payloads(seal):
    require(
        type(seal) is PhaseSeal
        and type(seal.artifact) is str
        and seal.artifact in (PREPARATION_ARTIFACT, SEATS_ARTIFACT)
        and type(seal.payload_bytes) is tuple
        and len(seal.payload_bytes) <= 13
        and type(seal.metadata_json) is bytes
        and seal.metadata_json == b"{}\n",
        "partial driver expected phase exact type/inventory differs",
    )
    payloads = {}
    checkpoint_bytes = 0
    allowed = (
        {"preparation.json", "prefix.json", "wave.json"}
        | {f"working_models/{name}.safetensors" for name in TRIPLES}
        if seal.artifact == PREPARATION_ARTIFACT
        else {"seats.json", "next_state.json"}
    )
    for row in seal.payload_bytes:
        require(
            type(row) is tuple
            and len(row) == 2
            and type(row[0]) is str
            and row[0] in allowed
            and row[0] not in payloads
            and type(row[1]) is bytes,
            "partial driver expected phase payload types differ",
        )
        name, payload = row
        if name.startswith("working_models/"):
            checkpoint_bytes += len(payload)
            require(
                len(payload) > 0 and checkpoint_bytes <= MAX_CHECKPOINT_BYTES,
                "partial driver expected checkpoint byte cap exceeded",
            )
        else:
            require(
                0 < len(payload) <= MAX_RECORD_BYTES,
                "partial driver expected diagnostic byte cap exceeded",
            )
        payloads[name] = payload
    if seal.artifact == SEATS_ARTIFACT:
        require(set(payloads) == allowed, "partial driver seat payload inventory differs")
    else:
        require(
            "preparation.json" in payloads and (not checkpoint_bytes or "wave.json" in payloads),
            "partial driver preparation payload inventory differs",
        )
    return payloads


def readback_phase(destination, *, expected_seal, run_root, checkpoint, deadline):
    """Exact readback only; a successful result is not a model-adoption capability."""
    finite_clock(deadline)
    checkpoint("before_phase_readback")
    root, target, identity = _paths(destination, run_root)
    payloads = _readback_payloads(expected_seal)
    verify_phase_capability(expected_seal)
    require(
        _phase_size(expected_seal.artifact, payloads, dict(expected_seal.predecessor_seals))
        <= MAX_PHASE_BYTES,
        "partial driver complete phase byte cap exceeded",
    )
    _run_admission(root, 0)
    actual = verify_phase(
        target,
        expected_artifact=expected_seal.artifact,
        expected_payload_paths=payloads,
        expected_predecessor_seals=dict(expected_seal.predecessor_seals),
        expected_seal_sha256=expected_seal.seal_sha256,
    )
    require(actual == expected_seal, "partial driver published bytes differ from captured phase")
    checkpoint("after_phase_readback")
    _path_guard(target, root, identity)
    # No supplied callback follows this fresh read. The trusted closing clock
    # charges its actual cost; this is a finite observation, not a filesystem lock.
    _run_admission(root, 0)
    final = verify_phase(
        target,
        expected_artifact=expected_seal.artifact,
        expected_payload_paths=payloads,
        expected_predecessor_seals=dict(expected_seal.predecessor_seals),
        expected_seal_sha256=expected_seal.seal_sha256,
    )
    require(final == expected_seal, "partial driver phase changed after the final callback")
    _path_guard(target, root, identity)
    if time.monotonic() >= deadline:
        raise TimeoutError("partial driver final structural readback exceeded original deadline")
    return final


def _publish(destination, *, artifact, payloads, predecessors, run_root, checkpoint, deadline):
    finite_clock(deadline)
    checkpoint("before_publication_inventory")
    root, target, identity = _paths(destination, run_root)
    phase_bytes = _phase_size(artifact, payloads, predecessors)
    require(phase_bytes <= MAX_PHASE_BYTES, "partial driver complete phase byte cap exceeded")
    # Count both possible publication aliases during the accepted relocation path.
    _run_admission(root, 2 * phase_bytes + FAILURE_RESERVE_BYTES)
    checkpoint("after_publication_inventory")
    _path_guard(target, root, identity)
    builder = PhaseBuilder(target, artifact=artifact, predecessor_seals=predecessors, metadata={})
    # Deliberately retain known staging paths on failure instead of invoking the
    # context manager's tree cleanup. An interrupted write may itself be removed
    # by PhaseBuilder; the outer driver retains only evidence actually available.
    staging = None
    try:
        builder.__enter__()
        staging = builder.staging_dir
        for name, payload in payloads.items():
            checkpoint("before_phase_payload")
            _path_guard(target, root, identity)
            builder.write_bytes(name, payload)
            checkpoint("after_phase_payload")
            _path_guard(target, root, identity)
        checkpoint("before_phase_publication")
        _path_guard(target, root, identity)
        # Staged payloads are already inside the shared tree. Keep room for the
        # complete destination alias and the not-yet-written receipt/manifest.
        _run_admission(
            root,
            2 * phase_bytes
            + FAILURE_RESERVE_BYTES
            - sum(len(payload) for payload in payloads.values()),
        )
        checkpoint("after_staging_inventory")
        _path_guard(target, root, identity)
        seal = builder.publish(expected_payload_paths=payloads)
        checkpoint("after_phase_publication")
        _path_guard(target, root, identity)
        actual = readback_phase(
            target, expected_seal=seal, run_root=root, checkpoint=checkpoint, deadline=deadline
        )
        require(
            actual.payload_bytes == tuple(sorted(payloads.items())),
            "partial driver readback payload bytes differ from actual work",
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("partial driver final publication exceeded original deadline")
        return actual
    except BaseException as error:
        # Known names are not claims that publication, or an individual partial
        # write, completed. The same live driver retains this private exception.
        BaseException.add_note(error, f"partial driver attempted destination: {target}")
        if staging is not None:
            BaseException.add_note(error, f"partial driver known staging path: {staging}")
        raise


def publish_preparation(
    destination,
    *,
    record_payload,
    prefix,
    wave,
    previous_phase_sha256,
    run_root,
    checkpoint,
    deadline,
):
    """Publish only actual returned prefix/wave/model bytes, including stopped work."""
    checkpoint("before_preparation_payloads")
    document = _outer(record_payload, PREPARATION_ARTIFACT)
    payloads = {"preparation.json": record_payload}
    checkpoint_hashes = []
    if prefix is not None:
        require(type(prefix) is EligibleGAPrefix, "partial driver prefix exact type differs")
        prefix.__post_init__()
        payloads["prefix.json"] = prefix.canonical_bytes()
        require(
            len(payloads["prefix.json"]) <= MAX_RECORD_BYTES,
            "partial driver prefix diagnostic byte cap exceeded",
        )
    if wave is not None:
        require(
            type(wave) is PartialGAWave and type(wave.record_json) is str,
            "partial driver wave exact type differs",
        )
        raw_wave = wave.record_json.encode("utf-8")
        raw = _document(raw_wave, maximum=MAX_RECORD_BYTES, phase_json=False)
        require(
            digest(raw_wave) == wave.sha256
            and raw.get("status") == wave.status
            and raw.get("artifact") == "native_ga_partial_wave_v2"
            and all(raw.get(flag) is False for flag in _FLAGS)
            and all(getattr(wave, flag) is False for flag in _FLAGS),
            "partial driver actual wave identity or eligibility differs",
        )
        compact = wave.status == "stopped_diagnostic_byte_cap"
        require(prefix is not None, "partial driver actual wave lacks its returned prefix")
        if compact:
            require(
                set(raw)
                == {
                    "artifact",
                    "status",
                    "ranked_positions",
                    "next_native_ordinal",
                    "next_behavior_versions",
                    "work",
                    "prefix_sha256",
                    *_FLAGS,
                }
                and raw["prefix_sha256"] == prefix.output_sha256
                and raw["ranked_positions"] == [],
                "partial driver compact stopped wave schema or prefix differs",
            )
        else:
            require(
                raw.get("selected_tuning_result_available") is False
                and kernel_json_bytes(raw.get("eligible_prefix")) == prefix.canonical_bytes(),
                "partial driver actual wave/prefix binding differs",
            )
        checkpoints = wave.checkpoint_payloads
        require(
            type(checkpoints) is tuple
            and len(checkpoints) <= len(TRIPLES)
            and all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and row[0] in TRIPLES
                and type(row[1]) is bytes
                and len(row[1]) > 0
                for row in checkpoints
            ),
            "partial driver checkpoint inventory type differs",
        )
        names = tuple(name for name, _ in checkpoints)
        checkpoint_hashes = [[name, digest(data)] for name, data in checkpoints]
        require(
            names == tuple(name for name in TRIPLES if name in names)
            and sum(len(data) for _, data in checkpoints) <= MAX_CHECKPOINT_BYTES
            and (compact or raw.get("checkpoints") == checkpoint_hashes),
            "partial driver checkpoint hashes/order/byte cap differ",
        )
        if wave.status == "ready_private_composition_required":
            require(
                names == TRIPLES and prefix.status == "complete",
                "partial driver READY lacks the complete prefix/ten checkpoints",
            )
        payloads["wave.json"] = raw_wave
        payloads.update({f"working_models/{name}.safetensors": data for name, data in checkpoints})
    require(
        previous_phase_sha256 is None or hash_string(previous_phase_sha256),
        "partial driver previous phase hash differs",
    )
    require(
        document.get("previous_phase_sha256") == previous_phase_sha256
        and document.get("prefix_sha256") == (None if prefix is None else prefix.output_sha256)
        and document.get("wave_sha256") == (None if wave is None else wave.sha256)
        and document.get("checkpoint_sha256s") == checkpoint_hashes,
        "partial driver outer preparation differs from actual returned payloads",
    )
    predecessors = {} if previous_phase_sha256 is None else {"previous": previous_phase_sha256}
    return _publish(
        destination,
        artifact=PREPARATION_ARTIFACT,
        payloads=payloads,
        predecessors=predecessors,
        run_root=run_root,
        checkpoint=checkpoint,
        deadline=deadline,
    )


def publish_seats(
    destination,
    *,
    record_payload,
    next_state_payload,
    preparation_seal_sha256,
    run_root,
    checkpoint,
    deadline,
):
    """Seal proposed state only; the same live core must finish its final guards."""
    checkpoint("before_seat_payloads")
    seats = _outer(record_payload, SEATS_ARTIFACT)
    state = _outer(next_state_payload, NEXT_STATE_ARTIFACT)
    require(hash_string(preparation_seal_sha256), "partial driver preparation seal differs")
    sequences = seats.get("method_sequences")
    positions = seats.get("ranked_positions")
    require(
        type(sequences) is list
        and len(sequences) == 14
        and all(canonical_sequence(value) for value in sequences)
        and len(set(sequences)) == 14
        and type(positions) is list
        and len(positions) == 14
        and all(type(value) is int and 0 <= value < 256 for value in positions)
        and len(set(positions)) == 14
        and positions == sorted(positions)
        and seats.get("status") == "selected"
        and hash_string(seats.get("composition_receipt_sha256"))
        and type(seats.get("method_seats")) is int
        and seats["method_seats"] == 14
        and type(seats.get("private_reserve_seats")) is int
        and seats["private_reserve_seats"] == 2,
        "partial driver seat publication requires exactly fourteen ordered method seats",
    )
    require(
        type(seats.get("round_index")) is int
        and 1 <= seats["round_index"] <= 28
        and hash_string(seats.get("preparation_sha256"))
        and hash_string(seats.get("history_sha256"))
        and seats.get("preparation_seal_sha256") == preparation_seal_sha256
        and state.get("completed_round_index") == seats["round_index"]
        and type(state.get("completed_round_index")) is int
        and state.get("next_round_index") == seats["round_index"] + 1
        and type(state.get("next_round_index")) is int
        and state.get("history_sha256") == seats["history_sha256"]
        and state.get("prior_method_sequences") == sequences
        and state.get("checkpoint_phase_sha256") == preparation_seal_sha256,
        "partial driver seats and proposed next-state bindings differ",
    )
    return _publish(
        destination,
        artifact=SEATS_ARTIFACT,
        payloads={"seats.json": record_payload, "next_state.json": next_state_payload},
        predecessors={"preparation": preparation_seal_sha256},
        run_root=run_root,
        checkpoint=checkpoint,
        deadline=deadline,
    )

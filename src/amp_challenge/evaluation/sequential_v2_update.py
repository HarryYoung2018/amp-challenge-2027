"""Pure update and label-free outer-projection logic for sequential v2.

This module deliberately keeps the three computations inside the logical
``update`` phase separate:

* a policy update can see base labels and exactly one selected pool reveal;
* outer diversity components can see only post-update, label-free metadata;
* an outer projection can see one frozen model state and label-free metadata.

Sealed publication and campaign assembly are layered on these immutable values
in a later boundary module.  Keeping the numerical core free of filesystem
handles makes its leakage and determinism properties directly testable.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from amp_challenge.acquisition.sequential_v2_selector import OBJECTIVES, OuterMeanCandidate
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    BaseUpdateCapability,
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
)
from amp_challenge.evaluation.sequential_v2_primitives import (
    OBJECTIVES as MODEL_OBJECTIVES,
)
from amp_challenge.evaluation.sequential_v2_primitives import (
    PROBABILITY_CLIP,
    TARGET_GRAM,
    TARGETS,
    DescriptorLogisticState,
    cluster_diversity_components,
    diversity_component_id,
    fit_descriptor_logistic,
    predict_contexts,
    predict_target_objectives,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_CONTEXTS_BY_FOLD,
    EXPECTED_SUPPORT_BY_FOLD,
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    policy_run_by_track_id,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    RevealCampaignCapability,
    SelectedPoolRevealCapability,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    sha256_bytes,
)
from amp_challenge.evaluation.sequential_v2_stage import StageManifestCapability
from amp_challenge.evaluation.sequential_v2_staging import (
    LabelFreeContext,
    OuterMetadataCapability,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

if TYPE_CHECKING:
    from amp_challenge.evaluation.sequential_v2_update_state import UpdateStateAttestation
else:
    # Keep runtime annotation introspection cycle-free.  The lazy boundary
    # performs the exact concrete-type validation after importing its module.
    UpdateStateAttestation = object

SCHEMA_VERSION = 1
UPDATE_STATE_ARTIFACT = "sequential_v2_update_state_v1"
OUTER_COMPONENTS_ARTIFACT = "sequential_v2_update_outer_components_v1"
OUTER_VIEW_ARTIFACT = "sequential_v2_update_outer_view_v1"
OUTER_EVIDENCE_ARTIFACT = "sequential_v2_update_outer_evidence_v1"
UPDATE_CAMPAIGN_ARTIFACT = "sequential_v2_update_campaign_barrier_v1"

UPDATE_STATE_PAYLOAD_PATHS = (
    "training-example-ids.jsonl",
    "update-summary.json",
    "updated-model.json",
)
OUTER_COMPONENT_PAYLOAD_PATHS = (
    "components-summary.json",
    "outer-components.jsonl",
)
OUTER_VIEW_PAYLOAD_PATHS = ("candidates.jsonl", "view-summary.json")
OUTER_EVIDENCE_PAYLOAD_PATHS = (
    "outer-context-predictions.jsonl",
    "outer-evidence-summary.json",
    "outer-sequence-predictions.jsonl",
)
UPDATE_CAMPAIGN_PAYLOAD_PATHS = ("update-index.jsonl", "update-summary.json")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID = re.compile(r"seqv2-div70:[0-9a-f]{64}\Z")
if OBJECTIVES != MODEL_OBJECTIVES:
    raise RuntimeError("selector and model objective orders have diverged")
_POSITIVE_TARGET_INDICES = tuple(
    index for index, target in enumerate(TARGETS) if TARGET_GRAM[target] == "positive"
)
_NEGATIVE_TARGET_INDICES = tuple(
    index for index, target in enumerate(TARGETS) if TARGET_GRAM[target] == "negative"
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    if value != rotation_by_id(value.rotation_id):
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if type(value.policy) is not str or (value.seed is not None and type(value.seed) is not int):
        raise TypeError(f"{label} policy and seed must use exact scalar types")
    if value != policy_run_by_track_id(value.track_id):
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _identifier_stream(
    value: object,
    *,
    label: str,
    allow_empty: bool,
    sha256_identifiers: bool,
    require_sorted: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise TypeError(f"{label} must be an exact immutable text tuple")
    identifiers = tuple(value)
    if (not identifiers and not allow_empty) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} must be nonempty when required and unique")
    if require_sorted and identifiers != tuple(sorted(identifiers)):
        raise ValueError(f"{label} must be sorted")
    if any(not item or "\n" in item or "\r" in item for item in identifiers):
        raise ValueError(f"{label} contains an invalid identifier")
    if sha256_identifiers and any(_SHA256.fullmatch(item) is None for item in identifiers):
        raise ValueError(f"{label} must contain lowercase SHA-256 values")
    return identifiers


def id_stream_sha256(values: Sequence[str], *, allow_empty: bool = False) -> str:
    """Hash one ordered canonical SHA-256 ID stream with frozen LF framing."""

    if type(values) not in {tuple, list}:
        raise TypeError("ID stream must be an exact tuple or list")
    if type(allow_empty) is not bool:
        raise TypeError("ID stream allow_empty must be an exact Boolean")
    identifiers = tuple(values)
    _identifier_stream(
        identifiers,
        label="ID stream",
        allow_empty=allow_empty,
        sha256_identifiers=True,
        require_sorted=False,
    )
    return hashlib.sha256("".join(f"{item}\n" for item in identifiers).encode("ascii")).hexdigest()


def _validate_outer_metadata(capability: OuterMetadataCapability) -> OuterMetadataCapability:
    if type(capability) is not OuterMetadataCapability:
        raise TypeError("outer computation requires an exact OuterMetadataCapability")
    spec = _require_frozen_rotation(capability.spec, label="outer metadata rotation")
    if type(capability.contexts) is not tuple or any(
        type(row) is not LabelFreeContext for row in capability.contexts
    ):
        raise TypeError("outer metadata contexts must be exact immutable label-free rows")
    contexts = capability.contexts
    example_ids = tuple(row.example_id for row in contexts)
    if (
        len(contexts) != EXPECTED_CONTEXTS_BY_FOLD[spec.outer_fold]
        or example_ids != tuple(sorted(set(example_ids)))
        or type(capability.allowed_example_ids) is not tuple
        or capability.allowed_example_ids != example_ids
    ):
        raise ValueError("outer metadata differs from the complete ordered outer fold")
    for row in contexts:
        if (
            type(row.example_id) is not str
            or type(row.assay_context_id) is not str
            or type(row.sequence_id) is not str
            or type(row.sequence) is not str
            or type(row.target) is not str
            or type(row.gram) is not str
            or type(row.fold) is not int
            or _SHA256.fullmatch(row.example_id) is None
            or row.assay_context_id != row.example_id
            or _SHA256.fullmatch(row.sequence_id) is None
            or row.fold != spec.outer_fold
            or row.target not in TARGET_GRAM
            or row.gram != TARGET_GRAM[row.target]
            or canonicalize_sequence(row.sequence) != row.sequence
            or canonical_sequence_id(row.sequence) != row.sequence_id
        ):
            raise ValueError("outer metadata contains a noncanonical context")
    support_ids = _identifier_stream(
        capability.support_sequence_ids,
        label="outer support sequence IDs",
        allow_empty=False,
        sha256_identifiers=True,
        require_sorted=True,
    )
    if len(support_ids) != EXPECTED_SUPPORT_BY_FOLD[spec.outer_fold]:
        raise ValueError("outer support census differs from the frozen fold")
    available_ids = {row.sequence_id for row in contexts}
    if not set(support_ids).issubset(available_ids):
        raise ValueError("outer support IDs are not a subset of outer metadata")
    sequence_by_id: dict[str, str] = {}
    grams_by_id: dict[str, set[str]] = {}
    for row in contexts:
        prior = sequence_by_id.setdefault(row.sequence_id, row.sequence)
        if prior != row.sequence:
            raise ValueError("outer metadata maps one sequence ID to multiple sequences")
        grams_by_id.setdefault(row.sequence_id, set()).add(row.gram)
    expected_support = tuple(
        sorted(
            sequence_id
            for sequence_id, grams in grams_by_id.items()
            if grams == {"positive", "negative"}
        )
    )
    if support_ids != expected_support:
        raise ValueError("outer support IDs differ from the exact two-Gram support rule")
    return capability


@dataclass(frozen=True, slots=True)
class PolicyUpdateState:
    """One frozen policy model plus digest-only training membership evidence."""

    run: PolicyRunSpec
    model: DescriptorLogisticState
    base_example_ids: tuple[str, ...]
    revealed_example_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    base_update_seal_sha256: str
    reveal_leaf_seal_sha256: str

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="policy update state run")
        if type(self.model) is not DescriptorLogisticState:
            raise TypeError("policy update state requires an exact descriptor-logistic model")
        base = _identifier_stream(
            self.base_example_ids,
            label="base example IDs",
            allow_empty=False,
            sha256_identifiers=True,
            require_sorted=True,
        )
        revealed = _identifier_stream(
            self.revealed_example_ids,
            label="revealed example IDs",
            allow_empty=True,
            sha256_identifiers=True,
            require_sorted=True,
        )
        training = _identifier_stream(
            self.training_example_ids,
            label="training example IDs",
            allow_empty=False,
            sha256_identifiers=True,
            require_sorted=True,
        )
        if set(base).intersection(revealed) or training != tuple(sorted((*base, *revealed))):
            raise ValueError("training IDs must be the disjoint sorted base/reveal union")
        if (run.policy == NO_QUERY) != (not revealed):
            raise ValueError("only no-query may have an empty selected reveal")
        if self.model.training_contexts != len(training):
            raise ValueError("updated model training census differs from training IDs")
        _sha256(self.base_update_seal_sha256, label="base-update leaf seal")
        _sha256(self.reveal_leaf_seal_sha256, label="reveal leaf seal")

    @property
    def refit(self) -> bool:
        return self.run.refit

    def training_id_documents(self) -> tuple[dict[str, object], ...]:
        return tuple({"example_id": example_id} for example_id in self.training_example_ids)


@dataclass(frozen=True, slots=True)
class UpdateStateCapability:
    """Raw-label-free projection input decoded from one authenticated state leaf.

    Constructing this value directly is not authority.  The sealed update
    boundary must rederive it from the controller-authoritative state-leaf
    digest before launching a projection worker.
    """

    run: PolicyRunSpec
    model: DescriptorLogisticState
    training_example_ids: tuple[str, ...]
    state_leaf_seal_sha256: str
    updated_model_payload_sha256: str

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="update state capability run")
        if type(self.model) is not DescriptorLogisticState:
            raise TypeError("update state capability requires an exact model state")
        training = _identifier_stream(
            self.training_example_ids,
            label="update state capability training IDs",
            allow_empty=False,
            sha256_identifiers=True,
            require_sorted=True,
        )
        if self.model.training_contexts != len(training):
            raise ValueError("update state capability model and training census disagree")
        _sha256(self.state_leaf_seal_sha256, label="update state capability leaf seal")
        model_digest = _sha256(
            self.updated_model_payload_sha256,
            label="update state capability model payload",
        )
        if model_digest != sha256_bytes(canonical_json_bytes(self.model.document())):
            raise ValueError("update state capability model payload digest changed")


def fit_policy_update_state(
    *,
    run: PolicyRunSpec,
    base_update: BaseUpdateCapability,
    selected_reveal: SelectedPoolRevealCapability,
) -> PolicyUpdateState:
    """Fit one update on exactly base plus its own selected reveal.

    The no-query branch intentionally never invokes the fitter and returns the
    exact accepted base-model object.
    """

    frozen_run = _require_frozen_run(run, label="policy update run")
    if type(base_update) is not BaseUpdateCapability:
        raise TypeError("policy update requires an exact BaseUpdateCapability")
    if type(selected_reveal) is not SelectedPoolRevealCapability:
        raise TypeError("policy update requires an exact SelectedPoolRevealCapability")
    if base_update.spec != frozen_run.rotation or selected_reveal.run != frozen_run:
        raise ValueError("update inputs do not match the requested policy run")
    base_contexts = base_update.contexts
    revealed_contexts = selected_reveal.contexts
    if any(row.fold not in frozen_run.rotation.base_folds for row in base_contexts):
        raise ValueError("update base context belongs outside the exact three base folds")
    if any(row.fold != frozen_run.rotation.pool_fold for row in revealed_contexts):
        raise ValueError("selected reveal context belongs outside the acquisition fold")
    base_ids = tuple(row.example_id for row in base_contexts)
    revealed_ids = tuple(row.example_id for row in revealed_contexts)
    if base_ids != tuple(sorted(set(base_ids))) or revealed_ids != tuple(sorted(set(revealed_ids))):
        raise ValueError("update contexts must be ordered by unique example ID")
    if base_update.allowed_example_ids != base_ids:
        raise ValueError("base-update capability allows a different example-ID stream")
    if selected_reveal.allowed_example_ids != revealed_ids:
        raise ValueError("selected reveal allows a different example-ID stream")
    if set(base_ids).intersection(revealed_ids):
        raise ValueError("base and selected reveal example IDs overlap")
    selected_ids = {row.sequence_id for row in revealed_contexts}
    if selected_ids != set(selected_reveal.selected_sequence_ids):
        raise ValueError("revealed contexts differ from the selected sequence-ID set")
    training_contexts = tuple(
        sorted((*base_contexts, *revealed_contexts), key=lambda row: row.example_id)
    )
    if frozen_run.policy == NO_QUERY:
        if revealed_contexts or selected_reveal.selected_sequence_ids:
            raise ValueError("no-query update must have an exactly empty reveal")
        model = base_update.base_model
    else:
        if not revealed_contexts:
            raise ValueError("a refitted policy requires at least one revealed context")
        model = fit_descriptor_logistic(training_contexts)
    return PolicyUpdateState(
        run=frozen_run,
        model=model,
        base_example_ids=base_ids,
        revealed_example_ids=revealed_ids,
        training_example_ids=tuple(row.example_id for row in training_contexts),
        base_update_seal_sha256=base_update.base_update_seal_sha256,
        reveal_leaf_seal_sha256=selected_reveal.reveal_leaf_seal_sha256,
    )


@dataclass(frozen=True, slots=True)
class OuterComponent:
    """One complete outer-fold 70%-identity component."""

    component_id: str
    sequence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.component_id) is not str or _COMPONENT_ID.fullmatch(self.component_id) is None:
            raise ValueError("outer component ID violates the sequential-v2 namespace")
        _identifier_stream(
            self.sequence_ids,
            label="outer component sequence IDs",
            allow_empty=False,
            sha256_identifiers=True,
            require_sorted=True,
        )

    def document(self, *, spec: RotationSpec) -> dict[str, object]:
        frozen = _require_frozen_rotation(spec, label="outer component document rotation")
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation_id": frozen.rotation_id,
            "role": "outer",
            "fold": frozen.outer_fold,
            "diversity_component_id": self.component_id,
            "sequence_ids": list(self.sequence_ids),
        }


@dataclass(frozen=True, slots=True)
class OuterComponentSet:
    """The one rotation-scoped partition shared by all eleven policy tracks."""

    spec: RotationSpec
    support_sequence_ids: tuple[str, ...]
    components: tuple[OuterComponent, ...]

    def __post_init__(self) -> None:
        spec = _require_frozen_rotation(self.spec, label="outer component set rotation")
        support = _identifier_stream(
            self.support_sequence_ids,
            label="outer component support IDs",
            allow_empty=False,
            sha256_identifiers=True,
            require_sorted=True,
        )
        if len(support) != EXPECTED_SUPPORT_BY_FOLD[spec.outer_fold]:
            raise ValueError("outer component support census differs from the frozen fold")
        if type(self.components) is not tuple or any(
            type(item) is not OuterComponent for item in self.components
        ):
            raise TypeError("outer components must be an exact immutable component tuple")
        if self.components != tuple(sorted(self.components, key=lambda item: item.component_id)):
            raise ValueError("outer components must be ordered by component ID")
        members = tuple(
            sorted(item for component in self.components for item in component.sequence_ids)
        )
        if members != support:
            raise ValueError("outer components must partition every support sequence exactly once")
        for component in self.components:
            expected = diversity_component_id(
                component.sequence_ids,
                role="outer",
                fold=spec.outer_fold,
            )
            if component.component_id != expected:
                raise ValueError("outer component ID differs from its role/fold/content hash")

    @property
    def component_by_sequence_id(self) -> Mapping[str, str]:
        return {
            sequence_id: component.component_id
            for component in self.components
            for sequence_id in component.sequence_ids
        }

    def documents(self) -> tuple[dict[str, object], ...]:
        return tuple(component.document(spec=self.spec) for component in self.components)


def build_outer_component_set(capability: OuterMetadataCapability) -> OuterComponentSet:
    """Build the shared rotation component partition from label-free metadata only."""

    metadata = _validate_outer_metadata(capability)
    sequence_by_id = {row.sequence_id: row.sequence for row in metadata.contexts}
    support_sequences = tuple(
        sequence_by_id[sequence_id] for sequence_id in metadata.support_sequence_ids
    )
    mapping, member_sets = cluster_diversity_components(
        support_sequences,
        role="outer",
        fold=metadata.spec.outer_fold,
    )
    components = tuple(
        sorted(
            (
                OuterComponent(component_id=mapping[members[0]], sequence_ids=members)
                for members in member_sets
            ),
            key=lambda item: item.component_id,
        )
    )
    return OuterComponentSet(
        spec=metadata.spec,
        support_sequence_ids=metadata.support_sequence_ids,
        components=components,
    )


@dataclass(frozen=True, slots=True)
class OuterContextPrediction:
    """One label-free prediction for one untouched outer assay context."""

    run: PolicyRunSpec
    context: LabelFreeContext
    probability: float

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="outer context prediction run")
        if type(self.context) is not LabelFreeContext:
            raise TypeError("outer context prediction requires exact label-free metadata")
        if type(self.probability) is not float or not math.isfinite(self.probability):
            raise TypeError("outer context probability must be an exact finite float")
        if not PROBABILITY_CLIP <= self.probability <= 1.0 - PROBABILITY_CLIP:
            raise ValueError("outer context probability is outside the accepted model range")
        if self.context.fold != run.rotation.outer_fold:
            raise ValueError("outer context prediction belongs to the wrong fold")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "example_id": self.context.example_id,
            "sequence_id": self.context.sequence_id,
            "target": self.context.target,
            "gram": self.context.gram,
            "fold": self.context.fold,
            "probability_hex": self.probability.hex(),
        }


@dataclass(frozen=True, slots=True)
class OuterSequencePrediction:
    """Bit-exact seven-target and three-objective outer support prediction."""

    run: PolicyRunSpec
    sequence_id: str
    target_probabilities: tuple[float, ...]
    objective_probabilities: tuple[float, float, float]

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="outer sequence prediction run")
        _sha256(self.sequence_id, label="outer sequence prediction sequence ID")
        if (
            type(self.target_probabilities) is not tuple
            or type(self.objective_probabilities) is not tuple
        ):
            raise TypeError("outer sequence probabilities must be exact immutable tuples")
        if len(self.target_probabilities) != len(TARGETS) or len(
            self.objective_probabilities
        ) != len(OBJECTIVES):
            raise ValueError("outer sequence probability dimensions changed")
        values = (*self.target_probabilities, *self.objective_probabilities)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise TypeError("outer sequence probabilities must be exact finite floats")
        if any(not PROBABILITY_CLIP <= value <= 1.0 - PROBABILITY_CLIP for value in values):
            raise ValueError("outer sequence probability is outside the accepted model range")
        target = np.asarray(self.target_probabilities, dtype=np.float64)
        positive = np.asarray(_POSITIVE_TARGET_INDICES, dtype=np.int64)
        negative = np.asarray(_NEGATIVE_TARGET_INDICES, dtype=np.int64)
        reduced = (
            float(np.mean(target, dtype=np.float64)),
            float(np.mean(target[positive], dtype=np.float64)),
            float(np.mean(target[negative], dtype=np.float64)),
        )
        if self.objective_probabilities != reduced:
            raise ValueError("outer objectives differ from the frozen seven-target reductions")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "sequence_id": self.sequence_id,
            "target_probabilities_hex": {
                target: probability.hex()
                for target, probability in zip(TARGETS, self.target_probabilities, strict=True)
            },
            "objective_probabilities_hex": {
                objective: probability.hex()
                for objective, probability in zip(
                    OBJECTIVES,
                    self.objective_probabilities,
                    strict=True,
                )
            },
        }


def outer_mean_candidate_document(candidate: OuterMeanCandidate) -> dict[str, object]:
    """Serialize exactly the six selector-visible outer candidate fields."""

    if type(candidate) is not OuterMeanCandidate:
        raise TypeError("outer candidate document requires an exact OuterMeanCandidate")
    return {
        "rotation_id": candidate.rotation_id,
        "sequence_id": candidate.sequence_id,
        "sequence": candidate.sequence,
        "objective_probabilities": {
            objective: probability
            for objective, probability in zip(
                OBJECTIVES,
                candidate.objective_probabilities,
                strict=True,
            )
        },
        "diversity_component_id": candidate.diversity_component_id,
        "eligible": candidate.eligible,
    }


@dataclass(frozen=True, slots=True)
class OuterProjection:
    """One model's complete label-free outer evidence and isolated selector view."""

    run: PolicyRunSpec
    component_set: OuterComponentSet
    context_predictions: tuple[OuterContextPrediction, ...]
    sequence_predictions: tuple[OuterSequencePrediction, ...]
    candidates: tuple[OuterMeanCandidate, ...]

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="outer projection run")
        if type(self.component_set) is not OuterComponentSet:
            raise TypeError("outer projection requires an exact OuterComponentSet")
        if self.component_set.spec != run.rotation:
            raise ValueError("outer projection component set has the wrong rotation")
        if type(self.context_predictions) is not tuple or any(
            type(item) is not OuterContextPrediction for item in self.context_predictions
        ):
            raise TypeError("outer context predictions must be an exact immutable tuple")
        if type(self.sequence_predictions) is not tuple or any(
            type(item) is not OuterSequencePrediction for item in self.sequence_predictions
        ):
            raise TypeError("outer sequence predictions must be an exact immutable tuple")
        if type(self.candidates) is not tuple or any(
            type(item) is not OuterMeanCandidate for item in self.candidates
        ):
            raise TypeError("outer candidates must be an exact immutable tuple")
        context_ids = tuple(item.context.example_id for item in self.context_predictions)
        if (
            len(context_ids) != EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold]
            or context_ids != tuple(sorted(set(context_ids)))
            or any(item.run != run for item in self.context_predictions)
        ):
            raise ValueError("outer context predictions do not cover the complete ordered fold")
        sequence_ids = tuple(item.sequence_id for item in self.sequence_predictions)
        candidate_ids = tuple(item.sequence_id for item in self.candidates)
        if (
            sequence_ids != self.component_set.support_sequence_ids
            or candidate_ids != sequence_ids
            or any(item.run != run for item in self.sequence_predictions)
            or any(item.rotation_id != run.rotation.rotation_id for item in self.candidates)
        ):
            raise ValueError("outer sequence evidence and candidate view are not aligned")
        component_by_id = self.component_set.component_by_sequence_id
        for prediction, candidate in zip(self.sequence_predictions, self.candidates, strict=True):
            if (
                candidate.objective_probabilities != prediction.objective_probabilities
                or candidate.diversity_component_id != component_by_id[prediction.sequence_id]
                or candidate.eligible is not True
            ):
                raise ValueError("outer candidate differs from its isolated prediction evidence")

    def context_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(item.document() for item in self.context_predictions)

    def sequence_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(item.document() for item in self.sequence_predictions)

    def candidate_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(outer_mean_candidate_document(item) for item in self.candidates)


def build_outer_projection(
    *,
    run: PolicyRunSpec,
    state: UpdateStateCapability,
    outer_metadata: OuterMetadataCapability,
    component_set: OuterComponentSet,
) -> OuterProjection:
    """Predict every outer context and the support-only next-round view."""

    frozen_run = _require_frozen_run(run, label="outer projection run")
    if type(state) is not UpdateStateCapability or state.run != frozen_run:
        raise ValueError("outer projection requires the matching exact update state")
    metadata = _validate_outer_metadata(outer_metadata)
    if metadata.spec != frozen_run.rotation:
        raise ValueError("outer metadata has the wrong rotation")
    if type(component_set) is not OuterComponentSet or component_set.spec != frozen_run.rotation:
        raise ValueError("outer components have the wrong rotation")
    if component_set.support_sequence_ids != metadata.support_sequence_ids:
        raise ValueError("outer components and metadata bind different support IDs")

    context_probabilities = predict_contexts(
        state.model,
        tuple(row.sequence for row in metadata.contexts),
        tuple(row.target for row in metadata.contexts),
        tuple(row.gram for row in metadata.contexts),
    )
    if (
        type(context_probabilities) is not np.ndarray
        or context_probabilities.dtype != np.float64
        or context_probabilities.shape != (len(metadata.contexts),)
    ):
        raise ValueError("outer context predictions must be an exact binary64 vector")
    context_predictions = tuple(
        OuterContextPrediction(frozen_run, row, float(probability))
        for row, probability in zip(metadata.contexts, context_probabilities, strict=True)
    )

    sequence_by_id: dict[str, str] = {}
    for row in metadata.contexts:
        prior = sequence_by_id.setdefault(row.sequence_id, row.sequence)
        if prior != row.sequence:
            raise ValueError("outer metadata maps one sequence ID to multiple sequences")
    support_sequences = tuple(
        sequence_by_id[sequence_id] for sequence_id in metadata.support_sequence_ids
    )
    target_matrix, objective_matrix = predict_target_objectives(state.model, support_sequences)
    sequence_count = len(metadata.support_sequence_ids)
    if (
        type(target_matrix) is not np.ndarray
        or target_matrix.dtype != np.float64
        or target_matrix.shape != (sequence_count, len(TARGETS))
        or type(objective_matrix) is not np.ndarray
        or objective_matrix.dtype != np.float64
        or objective_matrix.shape != (sequence_count, len(OBJECTIVES))
    ):
        raise ValueError("outer sequence predictions must be exact binary64 matrices")
    sequence_predictions = tuple(
        OuterSequencePrediction(
            run=frozen_run,
            sequence_id=sequence_id,
            target_probabilities=tuple(map(float, target_matrix[index])),
            objective_probabilities=tuple(map(float, objective_matrix[index])),
        )
        for index, sequence_id in enumerate(metadata.support_sequence_ids)
    )
    component_by_id = component_set.component_by_sequence_id
    candidates = tuple(
        OuterMeanCandidate(
            rotation_id=frozen_run.rotation.rotation_id,
            sequence_id=sequence_id,
            sequence=sequence_by_id[sequence_id],
            objective_probabilities=sequence_predictions[index].objective_probabilities,
            diversity_component_id=component_by_id[sequence_id],
            eligible=True,
        )
        for index, sequence_id in enumerate(metadata.support_sequence_ids)
    )
    return OuterProjection(
        run=frozen_run,
        component_set=component_set,
        context_predictions=context_predictions,
        sequence_predictions=sequence_predictions,
        candidates=candidates,
    )


def update_state_from_authorities(
    state_seal: PhaseSeal,
    *,
    attestation: UpdateStateAttestation,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
) -> UpdateStateCapability:
    """Lazily dispatch to the sealed state boundary without an import cycle."""

    from amp_challenge.evaluation.sequential_v2_update_state import (
        update_state_from_authorities as materialize,
    )

    return materialize(
        state_seal,
        attestation=attestation,
        run=run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
    )


__all__ = [
    "OUTER_COMPONENTS_ARTIFACT",
    "OUTER_COMPONENT_PAYLOAD_PATHS",
    "OUTER_EVIDENCE_ARTIFACT",
    "OUTER_EVIDENCE_PAYLOAD_PATHS",
    "OUTER_VIEW_ARTIFACT",
    "OUTER_VIEW_PAYLOAD_PATHS",
    "UPDATE_CAMPAIGN_ARTIFACT",
    "UPDATE_CAMPAIGN_PAYLOAD_PATHS",
    "UPDATE_STATE_ARTIFACT",
    "UPDATE_STATE_PAYLOAD_PATHS",
    "OuterComponent",
    "OuterComponentSet",
    "OuterContextPrediction",
    "OuterProjection",
    "OuterSequencePrediction",
    "PolicyUpdateState",
    "UpdateStateCapability",
    "build_outer_component_set",
    "build_outer_projection",
    "fit_policy_update_state",
    "id_stream_sha256",
    "outer_mean_candidate_document",
    "update_state_from_authorities",
]

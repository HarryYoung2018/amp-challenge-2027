"""Portable selected-target records, not oracle authenticity or IS authority."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    canonical_sequence,
    digest,
    hash_string,
    sequence_key,
)

ENDPOINT_CONFIG_SHA256 = "d2c1ce3d8ae774bcfd068ac5d5671245d648a38ef9e5d978df22a4dbfca338c7"
REFERENCE_MEASURE = "FULLMASK_GENERATOR_PATH"
TRIPLES = ("012", "013", "014", "023", "024", "034", "123", "124", "134", "234")


@dataclass(frozen=True, slots=True)
class EndpointOrigin:
    """Preserve genuine original versions; unknown is None, never invented zero.

    source_record_sha256 binds caller-owned immutable lineage/charged evidence.
    This record does not assert that the caller's source is genuine oracle truth.
    Target rebuild generation is separate from every original source version.
    """

    source_record_sha256: str
    origin_kind: str
    generation: int | None
    checkpoint_triple: str | None
    behavior_model_sha256: str | None
    behavior_version: int | None

    def __post_init__(self):
        if not hash_string(self.source_record_sha256) or self.origin_kind not in (
            "ga_endpoint",
            "native_endpoint",
            "external_charged",
            "lineage_parent",
        ):
            raise ValueError("endpoint origin source/kind differs")
        for version in (self.generation, self.behavior_version):
            if version is not None and (type(version) is not int or not 0 <= version <= 512):
                raise ValueError("endpoint original version differs")
        if self.checkpoint_triple is not None and self.checkpoint_triple not in TRIPLES:
            raise ValueError("endpoint origin checkpoint differs")
        if self.behavior_model_sha256 is not None and not hash_string(self.behavior_model_sha256):
            raise ValueError("endpoint original behavior identity differs")
        if (self.behavior_model_sha256 is None) != (self.behavior_version is None):
            raise ValueError("endpoint original behavior version/hash must be jointly known")
        if self.behavior_model_sha256 is not None and self.checkpoint_triple is None:
            raise ValueError("endpoint known behavior lacks checkpoint identity")
        if self.origin_kind == "native_endpoint" and self.behavior_model_sha256 is None:
            raise ValueError("native endpoint lacks actual original behavior")


@dataclass(frozen=True, slots=True)
class EndpointTarget:
    sequence: str
    role: str
    log_weight: float
    origin: EndpointOrigin

    def __post_init__(self):
        if not canonical_sequence(self.sequence) or self.role not in (
            "charged_endpoint",
            "positive_child",
            "zero_advantage_parent",
        ):
            raise ValueError("endpoint target support/role differs")
        if type(self.log_weight) is not float or not math.isfinite(self.log_weight):
            raise ValueError("endpoint target logweight must be a finite float")
        if type(self.origin) is not EndpointOrigin:
            raise TypeError("endpoint target origin differs")
        self.origin.__post_init__()

    @property
    def sequence_id(self):
        return sequence_key(self.sequence)


@dataclass(frozen=True, slots=True)
class EndpointTeacher:
    """Immutable selected-target supervision; the private oracle is NOT supplied.

    An empty/short source is representable for explicit no-update admission.
    Protected rows are derived by role, so no selected child/charged target can
    be silently excluded from its source-specific concentration diagnostics.
    """

    targets: tuple[EndpointTarget, ...]
    protected_role: str
    objective_context_sha256: str
    target_source_sha256: str
    rebuild_generation: int
    weighting_semantics: str = "fresh_forward_corruption_selected_endpoint_supervision_not_IS"
    max_generations: int = 28
    prospective_protocol_sha256: str | None = None

    def __post_init__(self):
        if (
            type(self.targets) is not tuple
            or len(self.targets) > 64
            or self.protected_role
            not in (
                "charged_endpoint",
                "positive_child",
            )
        ):
            raise ValueError("endpoint teacher inventory/protected source differs")
        if not hash_string(self.objective_context_sha256) or not hash_string(
            self.target_source_sha256
        ):
            raise ValueError("endpoint teacher context/source binding differs")
        if (
            type(self.max_generations) is not int
            or self.max_generations <= 0
            or type(self.rebuild_generation) is not int
            or not 0 <= self.rebuild_generation <= self.max_generations
            or (self.max_generations != 28 and not hash_string(self.prospective_protocol_sha256))
            or (
                self.prospective_protocol_sha256 is not None
                and not hash_string(self.prospective_protocol_sha256)
            )
        ):
            raise ValueError("endpoint rebuild generation differs")
        if (
            self.weighting_semantics
            != "fresh_forward_corruption_selected_endpoint_supervision_not_IS"
        ):
            raise ValueError("endpoint supervision must not be relabeled as trajectory IS")
        for target in self.targets:
            if type(target) is not EndpointTarget:
                raise TypeError("endpoint teacher target record differs")
            target.__post_init__()
            if target.role not in (self.protected_role, "zero_advantage_parent"):
                raise ValueError("endpoint teacher mixes protected source roles")
        if len({target.sequence_id for target in self.targets}) != len(self.targets):
            raise ValueError("endpoint teacher duplicates cannot increase effective sample size")
        if sum(target.role == "zero_advantage_parent" for target in self.targets) > 8:
            raise ValueError("endpoint teacher exceeds eight zero-advantage parents")
        if (
            self.targets
            and max(row.log_weight for row in self.targets)
            - min(row.log_weight for row in self.targets)
            > math.log(2) + 1e-12
        ):
            raise ValueError("endpoint teacher exceeds predeclared logweight span")

    @property
    def protected_indices(self):
        return tuple(
            index for index, row in enumerate(self.targets) if row.role == self.protected_role
        )

    @property
    def sha256(self):
        payload = asdict(self)
        if self.max_generations == 28 and self.prospective_protocol_sha256 is None:
            payload.pop("max_generations")
            payload.pop("prospective_protocol_sha256")
        return digest(b"native-shared-endpoint-teacher-v1\0" + canonical_json_bytes(payload))

    @property
    def semantic_sha256(self):
        payload = [
            ENDPOINT_CONFIG_SHA256,
            self.objective_context_sha256,
            self.rebuild_generation,
            [(row.sequence, row.role, row.log_weight) for row in self.targets],
        ]
        if self.max_generations != 28 or self.prospective_protocol_sha256 is not None:
            payload.append([self.max_generations, self.prospective_protocol_sha256])
        return digest(canonical_json_bytes(payload))


def weight_summary(weights):
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("endpoint weight diagnostics need positive finite unique rows")
    values = values / values.sum()
    ess = float(1 / np.square(values).sum())
    result = {
        "rows": len(values),
        "ess": ess,
        "ess_fraction": ess / len(values),
        "maximum_weight": float(values.max()),
    }
    result["passed"] = result["ess_fraction"] >= 0.2 and result["maximum_weight"] <= 0.05 + 1e-15
    return result


def teacher_admission(teacher: EndpointTeacher):
    teacher.__post_init__()
    if len(teacher.protected_indices) < 40:
        return {
            "admitted": False,
            "reason": "insufficient_distinct_protected_targets",
            "protected_rows": len(teacher.protected_indices),
        }, None
    logits = np.asarray([row.log_weight for row in teacher.targets])
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    summary = {
        "protected_targets": weight_summary(weights[list(teacher.protected_indices)]),
        "all_targets": weight_summary(weights),
        "anchors": weight_summary(np.ones(64)),
        "combined": weight_summary(np.concatenate([np.full(64, 0.5 / 64), 0.5 * weights])),
    }
    if not all(value["passed"] for value in summary.values()):
        raise ValueError("endpoint source-specific ESS/concentration guard failed")
    weights.setflags(write=False)
    return {"admitted": True, "reason": "source_specific_weight_guards_passed", **summary}, weights

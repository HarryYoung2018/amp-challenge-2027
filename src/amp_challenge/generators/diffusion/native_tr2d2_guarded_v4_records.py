"""Prospective atomic TR2 feasibility configuration and caller-pinned requirement."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_tr2_matched_feasibility import (
    SCOPE,
    tr2_matched_source,
)
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v3 import CONTRACT as V3_CONTRACT
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v3 import guarded_source_sha256
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string

CONFIG_SHA256 = "35ea36d05764f41a139680bc8bc775de8844d179885ea9e6eae344295e2e758a"
MAXIMUM_BYTES = 128 * 1024**2
CONTRACT = {
    **V3_CONTRACT,
    "version": "native-tr2-guarded-replay-v4",
    "configuration_sha256": CONFIG_SHA256,
    "feasibility_change_enforced": True,
    "native_update_work_counted": True,
}


def source_v4():
    root = Path(__file__).resolve().parents[4]
    config = root / "configs/search/native_tr2_matched_atomic_v4.toml"
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("TR2 v4 declared configuration changed")
    return _json_hash(
        {
            "guarded_v3": guarded_source_sha256(),
            "matched": tr2_matched_source(),
            "configuration": CONFIG_SHA256,
            "component": {
                name: hashlib.sha256(
                    (Path(__file__).parent / (name + ".py")).read_bytes()
                ).hexdigest()
                for name in (
                    "native_tr2d2_guarded_v4",
                    "native_tr2d2_guarded_v4_records",
                    "native_tr2d2_guarded_v4_verify",
                    "native_shared_endpoint_verify",
                    "native_ga_partial_verify",
                )
            },
        }
    )


@dataclass(frozen=True, slots=True)
class TR2MatchedRequirement:
    predicate: object
    predicate_sha256: str
    context_sha256: str
    source_sha256: str

    def document(self):
        if (
            not callable(self.predicate)
            or not all(
                hash_string(v)
                for v in (
                    self.predicate_sha256,
                    self.context_sha256,
                    self.source_sha256,
                )
            )
            or getattr(self.predicate, "source_sha256", None) != self.predicate_sha256
            or self.source_sha256 != tr2_matched_source()
        ):
            raise ValueError("TR2 v4 public requirement changed")
        return dict(
            configuration_sha256=CONFIG_SHA256,
            predicate_sha256=self.predicate_sha256,
            context_sha256=self.context_sha256,
            source_sha256=self.source_sha256,
            scope=SCOPE,
        )


@dataclass(frozen=True, slots=True)
class GuardedReplayAdvanceV4:
    record_json: str
    sha256: str

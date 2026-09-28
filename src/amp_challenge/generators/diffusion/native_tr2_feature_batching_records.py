"""Explicit opt-in feature batching; the native tree law remains unchanged."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.generators.diffusion.native_search_posterior import NativePosteriorBatch
from amp_challenge.representations.run_feature_cache_records import canonical, json_object, sha256

CONFIGURATION = "configs/search/native_tr2_feature_batching_v1.toml"
CONFIG_SHA256 = "45be987de4a6e5805e10328bcad2d25f95fca47d024504c0c52a8d0b2f30993e"
MAXIMUM_BYTES = 8 * 1024 * 1024


def check_batching_configuration(expected_sha256):
    root = Path(__file__).resolve().parents[4]
    if (
        expected_sha256 != CONFIG_SHA256
        or hashlib.sha256((root / CONFIGURATION).read_bytes()).hexdigest() != CONFIG_SHA256
    ):
        raise ValueError("TR2 grouped-feature configuration differs")
    return CONFIG_SHA256


def batching_source():
    check_batching_configuration(CONFIG_SHA256)
    root = Path(__file__).resolve().parents[4]
    paths = [
        CONFIGURATION,
        "src/amp_challenge/representations/run_feature_cache_views.py",
        *(
            "src/amp_challenge/generators/diffusion/" + name + ".py"
            for name in (
                "native_tr2_feature_batching_records",
                "native_tr2d2",
                "native_tr2d2_replay_v2",
                "native_tr2d2_guarded_v4",
                "native_tr2_campaign_driver",
            )
        ),
    ]
    return sha256(canonical({name: sha256((root / name).read_bytes()) for name in paths}))


@dataclass(frozen=True, slots=True)
class GroupedNativePosterior:
    groups: tuple[NativePosteriorBatch, ...]
    record_payload: bytes

    def __post_init__(self):
        if (
            type(self.groups) is not tuple
            or not 1 <= len(self.groups) <= 10
            or any(type(group) is not NativePosteriorBatch for group in self.groups)
            or type(self.record_payload) is not bytes
            or len(self.record_payload) > MAXIMUM_BYTES
            or canonical(json_object(self.record_payload)) != self.record_payload
        ):
            raise ValueError("TR2 grouped posterior record differs")

    @property
    def sha256(self):
        self.__post_init__()
        return sha256(self.record_payload)

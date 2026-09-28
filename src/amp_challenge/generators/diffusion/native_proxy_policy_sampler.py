"""Exact five-percent mixtures of frozen native denoising trajectory policies.

Bound scope: raw denoising law conditional on a fixed parent, start level and
kernel configuration. Changing parents, rejection filters, acquisition selection
or a complete search policy is outside this guarantee. Native trace likelihoods
remain conditional on the selected component, not marginalized mixture scores.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from amp_challenge.generators.diffusion.distribution_policy_mixture import (
    FrozenPolicyMixture,
    accept_mixture_update,
)
from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals


class NativeProxyPolicySampler:
    """Own immutable copies, retain every component and sample once per trajectory."""

    def __init__(self, units, *, storage_root=None, cache_size=10):
        from amp_challenge.generators.diffusion.checkpoint_cache import DiskModelStore, ModelCache

        units = self._units(units)
        self.storage_backed = storage_root is not None
        cache = ModelCache(cache_size) if storage_root is not None else None
        self._models = {}
        self._mixtures = {}
        self._latest = {}
        self._updates = []
        self._choices = []
        self._kernel_sha256 = _json_hash(asdict(NATIVE_ENDPOINT_DEFAULTS))
        for unit in units:
            frozen = copy.deepcopy(unit.model).eval()
            if storage_root is None:
                self._models[unit.triple] = {unit.policy_sha256: frozen}
            else:
                store = DiskModelStore(
                    Path(storage_root) / unit.triple,
                    cache,
                    device=next(unit.model.parameters()).device,
                )
                store[unit.policy_sha256] = frozen
                self._models[unit.triple] = store
            self._mixtures[unit.triple] = FrozenPolicyMixture.singleton(unit.policy_sha256)
            self._latest[unit.triple] = unit.policy_sha256
        self.check()

    @staticmethod
    def _units(units):
        units = tuple(units)
        if tuple(unit.triple for unit in units) != TRIPLES:
            raise ValueError("policy mixtures require all ten ordered students")
        for unit in units:
            unit.check()
        if len({unit.model.config for unit in units}) != 1:
            raise ValueError("all native model configurations must agree")
        return units

    def check(self):
        """Check inventory and resident models; disk models authenticate on reload."""
        self._check_models(TRIPLES)

    def _check_models(self, checked_triples):
        if any(triple not in TRIPLES for triple in checked_triples):
            raise ValueError("unknown native mixture student")
        if _json_hash(asdict(NATIVE_ENDPOINT_DEFAULTS)) != self._kernel_sha256:
            raise ValueError("native mixture kernel configuration changed")
        for triple in TRIPLES:
            if set(self._models[triple]) != set(self._mixtures[triple].component_ids):
                raise ValueError("frozen component inventory does not match exact mixture")
            if triple not in checked_triples:
                continue
            if hasattr(self._models[triple], "check_resident"):
                self._models[triple].check_resident()
                continue
            for identity, model in self._models[triple].items():
                if model.training or canonical_model_logical_hash(model) != identity:
                    raise ValueError("frozen native policy component mutated")

    def accept(self, new_units):
        """Atomically accept ten proposals with exactly 1/20 new component mass.

        The update adapter owns fitting and its metric checks. This adapter owns
        immutable generation components and the separate distribution guarantee.
        No old component is pruned, rounded, compressed or modified.
        """
        self.check()
        units = self._units(new_units)
        models = {
            triple: self._models[triple].clone()
            if hasattr(self._models[triple], "clone")
            else dict(self._models[triple])
            for triple in TRIPLES
        }
        mixtures, latest, records = {}, {}, []
        for unit in units:
            reference = self._models[unit.triple][self._latest[unit.triple]]
            if unit.model.config != reference.config:
                raise ValueError("proposal changes frozen native configuration")
            identity = unit.policy_sha256
            if identity not in models[unit.triple]:
                frozen = copy.deepcopy(unit.model).eval()
                if canonical_model_logical_hash(frozen) != identity:
                    raise ValueError("proposal mutated while freezing")
                models[unit.triple][identity] = frozen
            update = accept_mixture_update(self._mixtures[unit.triple], identity)
            mixtures[unit.triple] = update.accepted
            latest[unit.triple] = identity
            record = update.record()
            record.update(
                triple=unit.triple,
                native_sampler_connected=True,
                bound_scope="raw_denoising_trajectory_conditional_fixed_parent_start_level_kernel",
                excludes="changed_parent_distribution_filtering_acquisition_and_whole_search",
            )
            records.append(record)
        self.check()
        self._models, self._mixtures, self._latest = models, mixtures, latest
        self._updates.append(records)
        return copy.deepcopy(records)

    def sample(self, unit, parents, *, start_levels, seed, ordinals):
        """Choose one frozen component per ordinal, then execute its complete path."""
        # Every component this batch could use is authenticated before and after
        # sampling. Other students are not read here. Full-ensemble checks still
        # precede policy acceptance, recorded publication and checkpoint saving.
        self._check_models((unit.triple,))
        unit.check()
        if unit.triple not in self._latest or unit.policy_sha256 != self._latest[unit.triple]:
            raise ValueError("sampling unit does not match last accepted proposal")
        parents, start_levels, ordinals = tuple(parents), tuple(start_levels), tuple(ordinals)
        if (
            type(seed) is not int
            or not 0 <= seed < 2**63
            or not 1 <= len(parents) <= NATIVE_ENDPOINT_DEFAULTS.maximum_replay_rows
            or len(start_levels) != len(parents)
            or len(ordinals) != len(parents)
            or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
            or len(set(ordinals)) != len(ordinals)
        ):
            raise ValueError("mixture sampling needs valid seed and aligned unique ordinals")
        mixture = self._mixtures[unit.triple]
        mixture_sha256 = _json_hash(mixture.record())
        groups, choices = {}, []
        for index, ordinal in enumerate(ordinals):
            choice_seed = int(
                _json_hash(["native-proxy-mixture-v1", seed, unit.triple, ordinal])[:32], 16
            )
            rng = np.random.Generator(np.random.PCG64DXSM(choice_seed))
            component = mixture.sample_component(rng)
            groups.setdefault(component, []).append(index)
            choices.append(component)
        traces = [None] * len(parents)
        for component, indices in groups.items():
            sampled = sample_native_proposals(
                self._models[unit.triple][component],
                tuple(parents[index] for index in indices),
                start_levels=tuple(start_levels[index] for index in indices),
                seed=seed,
                ordinals=tuple(ordinals[index] for index in indices),
            )
            for index, trace in zip(indices, sampled, strict=True):
                if trace.model_sha256 != component:
                    raise ValueError("native trace differs from selected frozen component")
                traces[index] = trace
        self._check_models((unit.triple,))
        probabilities = dict(zip(mixture.component_ids, mixture.weights, strict=True))
        for trace, component in zip(traces, choices, strict=True):
            probability = probabilities[component]
            self._choices.append(
                {
                    "triple": unit.triple,
                    "seed": seed,
                    "ordinal": trace.ordinal,
                    "parent_sha256": trace.parent_sha256,
                    "start_level": trace.start_level,
                    "mixture_sha256": mixture_sha256,
                    "component_sha256": component,
                    "component_probability_numerator": probability.numerator,
                    "component_probability_denominator": probability.denominator,
                    "component_conditional_path_log_probability": trace.augmented_path_log_probability,
                    "component_and_path_joint_log_probability": (
                        math.log(probability.numerator)
                        - math.log(probability.denominator)
                        + trace.augmented_path_log_probability
                    ),
                    "trace_sha256": _json_hash(asdict(trace)),
                }
            )
        return tuple(traces)

    def model_for_trace(self, triple, trace):
        """Private copy for independent replay; never expose a mutable frozen model."""
        self.check()
        if triple not in self._models or trace.model_sha256 not in self._models[triple]:
            raise ValueError("unknown native mixture component")
        return copy.deepcopy(self._models[triple][trace.model_sha256])

    def record(self):
        self.check()
        record = {
            "artifact": "native_proxy_policy_sampler_v1",
            "kernel_sha256": self._kernel_sha256,
            "mixtures": {triple: mixture.record() for triple, mixture in self._mixtures.items()},
            "latest_proposals": dict(self._latest),
            "updates": copy.deepcopy(self._updates),
            "choices": copy.deepcopy(self._choices),
            "bound_scope": "raw_denoising_trajectory_conditional_fixed_parent_start_level_kernel",
            "filtered_candidate_or_whole_search_bound_claimed": False,
            "trace_log_probability_scope": "conditional_selected_component_not_marginal_mixture",
        }
        record["sha256"] = _json_hash(record)
        return record

    def persist(self, outputdir):
        """Save every immutable component and a content-addressed replay index.

        Existing files are verified, never overwritten. Snapshots can be added
        after later updates or samples; older snapshots remain independently
        usable. Torch payloads contain only tensors and primitive metadata, and
        existing payloads are loaded with weights_only=True.
        """
        self.check()
        output = Path(outputdir)
        output.mkdir(parents=True, exist_ok=True)
        snapshot = self.record()
        artifacts = []
        for triple in TRIPLES:
            for identity, model in self._models[triple].items():
                target = output / f"{triple}-{identity}.pt"
                if target.is_symlink():
                    raise ValueError("component checkpoint must not be a symbolic link")
                payload = {
                    "artifact": "native_proxy_frozen_component_v1",
                    "triple": triple,
                    "model_sha256": identity,
                    "config": asdict(model.config),
                    "state_dict": {
                        name: value.detach().cpu().contiguous().clone()
                        for name, value in model.state_dict().items()
                    },
                }
                try:
                    with target.open("xb") as stream:
                        torch.save(payload, stream)
                except FileExistsError:
                    pass
                loaded = torch.load(target, map_location="cpu", weights_only=True)
                if (
                    loaded.get("artifact") != payload["artifact"]
                    or loaded.get("triple") != triple
                    or loaded.get("model_sha256") != identity
                    or loaded.get("config") != payload["config"]
                ):
                    raise ValueError("existing frozen component metadata differs")
                # Reconstruct actual model identity, not merely supplied metadata.
                # Model initialization must not advance the experiment RNG.
                with torch.random.fork_rng(devices=[]):
                    restored = NativeDenoiser(NativeDenoiserConfig(**loaded["config"]))
                    dtype = next(
                        value.dtype
                        for value in loaded["state_dict"].values()
                        if value.is_floating_point()
                    )
                    restored = restored.to(dtype=dtype).eval()
                    restored.load_state_dict(loaded["state_dict"], strict=True)
                if canonical_model_logical_hash(restored) != identity:
                    raise ValueError("existing frozen component actual model identity differs")
                artifacts.append(
                    {
                        "triple": triple,
                        "model_sha256": identity,
                        "file": target.name,
                        "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "bytes": target.stat().st_size,
                    }
                )
        self.check()
        if self.record()["sha256"] != snapshot["sha256"]:
            raise ValueError("sampler changed while persisting checkpoint snapshot")
        index = {
            "artifact": "native_proxy_policy_replay_index_v1",
            "sampler": snapshot,
            "checkpoints": artifacts,
        }
        index["sha256"] = _json_hash(index)
        encoded = json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        target = output / f"index-{index['sha256']}.json"
        if target.is_symlink():
            raise ValueError("replay index must not be a symbolic link")
        try:
            with target.open("xb") as stream:
                stream.write(encoded)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ValueError("existing replay index bytes differ") from None
        return index

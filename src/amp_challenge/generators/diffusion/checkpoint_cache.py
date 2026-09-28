"""Disk-backed model inventory with a shared least-recently-used RAM cache."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import asdict
from pathlib import Path

import torch

from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
)


class ModelCache:
    def __init__(self, capacity=10):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.resident = OrderedDict()

    def put(self, key, model):
        self.resident[key] = model
        self.resident.move_to_end(key)
        while len(self.resident) > self.capacity:
            self.resident.popitem(last=False)


class DiskModelStore(MutableMapping):
    def __init__(self, root, cache, *, device="cpu"):
        self.root, self.cache, self.device = Path(root), cache, device
        self.root.mkdir(parents=True, exist_ok=True)
        self.identities = set()

    def __getitem__(self, identity):
        if identity not in self.identities:
            raise KeyError(identity)
        key = (str(self.root), identity)
        if key in self.cache.resident:
            self.cache.resident.move_to_end(key)
            return self.cache.resident[key]
        payload = torch.load(self.root / f"{identity}.pt", map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = NativeDenoiser(NativeDenoiserConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(self.device).eval()
        if canonical_model_logical_hash(model) != identity:
            raise ValueError("cached checkpoint identity differs")
        self.cache.put(key, model)
        return model

    def __setitem__(self, identity, model):
        if model.training or canonical_model_logical_hash(model) != identity:
            raise ValueError("cache insertion requires an unchanged evaluation model")
        target = self.root / f"{identity}.pt"
        if identity not in self.identities:
            with target.open("xb") as stream:
                torch.save(
                    {
                        "config": asdict(model.config),
                        "state_dict": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                    },
                    stream,
                )
        self.identities.add(identity)
        self.cache.put((str(self.root), identity), model)

    def __delitem__(self, identity):
        raise TypeError("mixture components must not be deleted")

    def __iter__(self):
        return iter(sorted(self.identities))

    def __len__(self):
        return len(self.identities)

    def __contains__(self, identity):
        return identity in self.identities

    def clone(self):
        result = DiskModelStore(self.root, self.cache, device=self.device)
        result.identities = set(self.identities)
        return result

    def check_resident(self):
        for (root, identity), model in self.cache.resident.items():
            if root == str(self.root) and (
                model.training or canonical_model_logical_hash(model) != identity
            ):
                raise ValueError("resident policy component mutated")

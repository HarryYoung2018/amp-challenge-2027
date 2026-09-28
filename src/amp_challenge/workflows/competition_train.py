"""Train the ten six-layer generators using the existing fold assignments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from amp_challenge.generators.diffusion.data import namespaced_seed
from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
    configure_deterministic_runtime,
    load_safetensors_checkpoint,
    save_safetensors_checkpoint,
)
from amp_challenge.generators.diffusion.namespace_checkpoint_contract import (
    learning_rate,
)
from amp_challenge.generators.diffusion.namespace_checkpoint_train import (
    corrupted_batch,
    objective_for_batch,
    optimizer_for_model,
    training_levels,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.sequences import canonical_sequence_id

BASE = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/data/runs/"
    "generator-namespace-checkpoints-v1-20260909-cnRHxX"
)


def six_layer_config():
    return NativeDenoiserConfig(layers=6, hidden_dim=128, attention_heads=4, ffn_dim=384)


@dataclass(frozen=True)
class TrainedNativeInitialization:
    """A trained model, not a claim of independent scientific qualification."""

    model: NativeDenoiser
    triple: str
    training_sequence_ids: tuple[str, ...]
    checkpoint_logical_sha256: str
    qualification: str = "six_layer_training_not_independent_biological_validation"


def train_member(output, ordinal, *, base=BASE, steps=1000, device="cuda"):
    output = Path(output) / f"{ordinal:02d}"
    output.mkdir(parents=True, exist_ok=False)
    projection = (Path(base) / f"{ordinal:02d}" / "training_projection.jsonl").read_bytes()
    rows = [json.loads(line) for line in projection.splitlines()]
    seed = 530000 + int(TRIPLES[ordinal])
    configure_deterministic_runtime(namespaced_seed(seed, "initialization", "model"))
    torch.set_num_threads(1)
    model = NativeDenoiser(six_layer_config()).to(device)
    optimizer = optimizer_for_model(model)
    weights = np.asarray([row["sampling_weight"] for row in rows])
    weights /= weights.sum()
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    started = time.monotonic()
    model.train()
    with (output / "training_trace.jsonl").open("x") as trace:
        for step in range(1, steps + 1):
            indices = rng.choice(len(rows), size=64, p=weights)
            batch_rows = [rows[index] for index in indices]
            start = (step - 1) * 64
            levels = training_levels(seed, start, 64)
            ids = [row["sequence_id"] for row in batch_rows]
            seeds = tuple(
                namespaced_seed(seed, "corruption", start + offset, identity, int(levels[offset]))
                for offset, identity in enumerate(ids)
            )
            batch = corrupted_batch([row["sequence"] for row in batch_rows], ids, levels, seeds)
            torch.manual_seed(namespaced_seed(seed, "dropout", step))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(min(step, 1000))
            optimizer.zero_grad(set_to_none=True)
            objective = objective_for_batch(model, batch, levels, torch.device(device))
            if not torch.isfinite(objective.loss):
                raise ValueError("nonfinite training loss")
            objective.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = {
                "step": step,
                "loss": float(objective.loss.detach()),
                "seconds": time.monotonic() - started,
            }
            trace.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps({"member": ordinal, **record}), flush=True)
    model.eval()
    hashes = save_safetensors_checkpoint(model, output / "checkpoint.safetensors")
    (output / "training_projection.jsonl").write_bytes(projection)
    manifest = {
        "architecture": asdict(model.config),
        "triple": TRIPLES[ordinal],
        "seed": seed,
        "steps": steps,
        "rows": len(rows),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hashes": asdict(hashes),
        "model_sha256": canonical_model_logical_hash(model),
        "elapsed_seconds": time.monotonic() - started,
        "sampling": "component_weighted_with_replacement_pcg64dxsm",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_trained_units(root, *, device="cpu", check=lambda: None):
    initializations, corpora = [], []
    for ordinal, triple in enumerate(TRIPLES):
        check()
        directory = Path(root) / f"{ordinal:02d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["triple"] != triple or manifest["architecture"] != asdict(six_layer_config()):
            raise ValueError("six-layer checkpoint architecture or member differs")
        model = NativeDenoiser(six_layer_config())
        load_safetensors_checkpoint(
            model,
            directory / "checkpoint.safetensors",
            **{
                "expected_file_sha256": manifest["hashes"]["file_sha256"],
                "expected_logical_state_sha256": manifest["hashes"]["logical_state_sha256"],
            },
        )
        model.to(device).eval()
        if canonical_model_logical_hash(model) != manifest["model_sha256"]:
            raise ValueError("trained model identity differs")
        corpus = tuple(
            json.loads(line)["sequence"]
            for line in (directory / "training_projection.jsonl").read_text().splitlines()
        )
        corpora.append(corpus)
        initializations.append(
            TrainedNativeInitialization(
                model,
                triple,
                tuple(sorted(map(canonical_sequence_id, corpus))),
                manifest["model_sha256"],
            )
        )
    return tuple(initializations), tuple(corpora)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    for ordinal in args.members:
        print(json.dumps(train_member(args.output, ordinal, steps=args.steps)), flush=True)


if __name__ == "__main__":
    main()

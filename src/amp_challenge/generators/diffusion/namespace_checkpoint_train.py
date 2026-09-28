"""Bounded label-free training of the ten generator-namespace fold checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import safetensors
import torch

from amp_challenge.generators.diffusion.categorical import (
    AbsorbingDiffusion,
    CosineMaskSchedule,
    PeptideVocabulary,
)
from amp_challenge.generators.diffusion.data import namespaced_seed
from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    canonical_model_logical_hash,
    configure_deterministic_runtime,
    masked_token_objective,
    save_safetensors_checkpoint,
)
from amp_challenge.generators.diffusion.namespace_checkpoint_contract import (
    ARTIFACT_FILES,
    TRIPLES,
    CheckpointContract,
    authenticate_generator_rows,
    canonical,
    learning_rate,
    load_contract,
    projection_bytes,
    read_regular,
    require,
    sha,
    training_distribution,
)
from amp_challenge.generators.diffusion.sampling import sample_unconditional_v0


def source_inventory(repository: Path, expected_commit: str) -> dict[str, str]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repository), *args])

    require(
        git("rev-parse", "HEAD").decode().strip() == expected_commit, "repository commit changed"
    )
    require(
        not git("status", "--porcelain=v1", "--untracked-files=all"),
        "checkpoint producer requires clean committed source",
    )
    paths = (
        git(
            "ls-files",
            "-z",
            "src/amp_challenge",
            "pyproject.toml",
            "uv.lock",
            "configs/diffusion/generator_namespace_checkpoints_v1.toml",
        )
        .decode()
        .split("\0")
    )
    return {path: sha(read_regular(repository / path)) for path in sorted(paths) if path}


def training_levels(seed: int, start: int, count: int) -> np.ndarray:
    return np.asarray(
        [
            1 + namespaced_seed(seed, "timestep", ordinal, 0) % 64
            for ordinal in range(start, start + count)
        ],
        dtype=np.int64,
    )


def corrupted_batch(
    sequences: list[str], sequence_ids: list[str], levels: np.ndarray, seeds: tuple[int, ...]
):
    vocabulary = PeptideVocabulary("ACDEFGHIKLMNPQRSTVWY")
    encoded = vocabulary.encode(sequences, max_length=50)
    diffusion = AbsorbingDiffusion(vocabulary, CosineMaskSchedule(0.008))
    noisy, selected = diffusion.corrupt_fixed_count(
        encoded.tokens, encoded.attention_mask, levels, total_levels=64, row_seeds=seeds
    )
    require(len(sequence_ids) == len(sequences), "sequence identity count differs")
    return encoded.tokens, noisy, selected, encoded.attention_mask


def objective_for_batch(model: NativeDenoiser, batch, levels: np.ndarray, device: torch.device):
    clean, noisy, selected, mask = batch
    clean_tensor, noisy_tensor, selected_tensor, mask_tensor = [
        torch.from_numpy(array.copy()).to(device) for array in (clean, noisy, selected, mask)
    ]
    logits = model(
        noisy_tensor,
        mask_tensor,
        torch.from_numpy(levels.copy()).to(device),
        mask_tensor.sum(dim=1, dtype=torch.long),
    )
    return masked_token_objective(
        logits, clean_tensor, selected_tensor, mask_tensor, corrupted_tokens=noisy_tensor
    )


def optimizer_for_model(model: NativeDenoiser) -> torch.optim.AdamW:
    # Same numerical grouping as the native trainer: embeddings, norms and biases
    # have no weight decay; matrix projection weights get the declared decay.
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding | torch.nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    decay, no_decay = [], []
    for name, parameter in sorted(model.named_parameters()):
        (no_decay if name.endswith("bias") or id(parameter) in no_decay_ids else decay).append(
            parameter
        )
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)


def train_checkpoint(
    contract: CheckpointContract,
    *,
    repository: Path,
    expected_commit: str,
    run_root: Path,
    fit_index: int,
) -> Path:
    started = time.monotonic()
    require(type(fit_index) is int and 0 <= fit_index <= 10, "fit index differs")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "training requires a Slurm allocation")
    require(
        os.environ.get("SLURM_JOB_PARTITION") == "gpumid"
        and os.environ.get("SLURM_JOB_ACCOUNT") == "bio",
        "training allocation differs",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "training requires exactly one visible GPU",
    )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    require("A100" in properties.name, "training allocation is not an A100")
    ordinal = 0 if fit_index == 10 else fit_index
    triple = TRIPLES[ordinal]
    seed = contract.document["fits"]["seeds"][ordinal]
    run_root = run_root.resolve()
    require(run_root.is_dir(), "run root must be prepared by the submitter")
    output = run_root / f"{fit_index:02d}"
    output.mkdir(mode=0o700)
    inventory = source_inventory(repository, expected_commit)
    selected_twin = 1 if fit_index == 10 else 0
    rows = authenticate_generator_rows(contract, selected_twin=selected_twin)
    distribution = training_distribution(rows, triple)
    require(
        len(distribution.rows) == contract.document["fits"]["training_row_counts"][ordinal],
        "training row count differs",
    )
    projection = projection_bytes(distribution)
    write_new(output / "config.toml", contract.payload)
    code_bytes = "".join(f"{digest}  {path}\n" for path, digest in inventory.items()).encode()
    write_new(output / "CODE_SHA256SUMS", code_bytes)
    write_new(output / "training_projection.jsonl", projection)
    initialization_seed = namespaced_seed(seed, "initialization", "model")
    torch.set_num_threads(1)
    controls = configure_deterministic_runtime(initialization_seed)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = NativeDenoiser(contract.model_config).to(device)
    require(
        sum(parameter.numel() for parameter in model.parameters()) == 354068,
        "model parameter count differs",
    )
    initial_model_sha = canonical_model_logical_hash(model)
    optimizer = optimizer_for_model(model)
    trace = []
    consumed_ids = set()

    def check_budget() -> None:
        require(time.monotonic() - started < 720, "worker wall-time budget exhausted")
        require(
            torch.cuda.max_memory_reserved(device) <= 12 * 1024**3, "GPU memory budget exhausted"
        )

    model.train()
    for step in range(1, 1001):
        check_budget()
        start = (step - 1) * 64
        minibatch = distribution.draw(root_seed=seed, draw_start=start, draw_count=64)
        ids = [row.sequence_id for row in minibatch]
        sequences = [row.sequence for row in minibatch]
        consumed_ids.update(ids)
        levels = training_levels(seed, start, 64)
        corruption_seeds = tuple(
            namespaced_seed(seed, "corruption", start + offset, identity, int(levels[offset]))
            for offset, identity in enumerate(ids)
        )
        batch = corrupted_batch(sequences, ids, levels, corruption_seeds)
        torch.manual_seed(namespaced_seed(seed, "dropout", step))
        rate = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        objective = objective_for_batch(model, batch, levels, device)
        objective.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
        loss = float(objective.loss.detach().item())
        gradient_norm = float(norm.item())
        require(math.isfinite(loss) and math.isfinite(gradient_norm), "nonfinite training metric")
        trace.append(
            canonical(
                {
                    "step": step,
                    "sequence_ids": ids,
                    "levels": levels.tolist(),
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "learning_rate": rate,
                    "selected_tokens": int(objective.selected_counts.sum().item()),
                }
            )
        )
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "triple": triple,
                        "loss": loss,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    check_budget()
    write_new(output / "training_trace.jsonl", b"".join(trace))
    hashes = save_safetensors_checkpoint(model, output / "checkpoint.safetensors")
    require(hashes.logical_state_sha256 != initial_model_sha, "training did not change the model")
    model.eval()
    holdout_records = []
    fold_values: dict[int, list[float]] = {}
    with torch.inference_mode():
        for fold in range(5):
            if str(fold) in triple:
                continue
            heldout = [row for row in rows if row["generator_fold"] == fold]
            fold_values[fold] = []
            for level in contract.document["diagnostics"]["holdout_levels"]:
                for start in range(0, len(heldout), 64):
                    check_budget()
                    part = heldout[start : start + 64]
                    ids = [row["sequence_id"] for row in part]
                    sequences = [row["sequence"] for row in part]
                    levels = np.full(len(part), level, dtype=np.int64)
                    seeds = tuple(
                        namespaced_seed(640000, "holdout-corruption", identity, level)
                        for identity in ids
                    )
                    objective = objective_for_batch(
                        model, corrupted_batch(sequences, ids, levels, seeds), levels, device
                    )
                    values = objective.row_losses.detach().cpu().tolist()
                    for identity, loss in zip(ids, values, strict=True):
                        require(math.isfinite(loss), "nonfinite heldout loss")
                        fold_values[fold].append(loss)
                        holdout_records.append(
                            canonical(
                                {
                                    "fold": fold,
                                    "sequence_id": identity,
                                    "level": level,
                                    "loss": loss,
                                }
                            )
                        )
    write_new(output / "heldout_losses.jsonl", b"".join(holdout_records))

    def provider(tokens, mask, levels, lengths):
        check_budget()
        with torch.inference_mode():
            tensors = [
                torch.from_numpy(value.copy()).to(device)
                for value in (tokens, mask, levels, lengths)
            ]
            return model(*tensors).detach().cpu().numpy().copy()

    sample_seed = namespaced_seed(650000, "diagnostic-sampling", triple)
    lengths = distribution.length_prior.draw(root_seed=sample_seed, draw_start=0, draw_count=128)
    result = sample_unconditional_v0(
        provider,
        lengths,
        checkpoint_logical_sha256=hashes.logical_state_sha256,
        seed=sample_seed,
        contract_sha256=contract.sha256,
        batch_size=64,
        require_locked_count=False,
    )
    require(len(result.candidates) == 128, "raw sample count differs")
    write_new(output / "samples.jsonl", result.canonical_jsonl_bytes())
    training_ids = {row.sequence_id for row in distribution.rows}
    sample_ids = [row.sequence_id for row in result.candidates]
    metrics = {
        "diagnostic_only": True,
        "checkpoint_selection_allowed": False,
        "final_step": 1000,
        "training_draw_count": 64000,
        "final_training_loss": json.loads(trace[-1])["loss"],
        "heldout_fold_losses": {
            str(fold): math.fsum(values) / len(values) for fold, values in fold_values.items()
        },
        "raw_sample_count": 128,
        "valid_count": 128,
        "valid_rate": 1.0,
        "unique_valid_count": len(set(sample_ids)),
        "unique_valid_rate": len(set(sample_ids)) / 128,
        "exact_training_overlap_count": sum(identity in training_ids for identity in sample_ids),
        "exact_training_overlap_rate": sum(identity in training_ids for identity in sample_ids)
        / 128,
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "worker_elapsed_seconds": time.monotonic() - started,
    }
    write_new(output / "metrics.json", canonical(metrics))
    require(
        source_inventory(repository, expected_commit) == inventory,
        "source inventory changed during fit",
    )
    check_budget()
    artifacts = {
        name: {"sha256": sha(read_regular(output / name)), "bytes": (output / name).stat().st_size}
        for name in sorted(ARTIFACT_FILES - {"manifest.json", "COMPLETE.json"})
    }
    manifest = {
        "schema_version": 1,
        "artifact": "generator_namespace_native_checkpoint_v1",
        "fit_index": fit_index,
        "triple": triple,
        "seed": seed,
        "duplicate_of": 0 if fit_index == 10 else None,
        "config_sha256": contract.sha256,
        "namespace_receipt_sha256": contract.document["input"]["receipt_sha256"],
        "generator_corpus_sha256": contract.document["input"]["corpus_sha256"],
        "selected_generator_twin": selected_twin,
        "training_folds": [int(value) for value in triple],
        "training_sequence_ids": sorted(training_ids),
        "consumed_sequence_ids": sorted(consumed_ids),
        "training_row_count": len(distribution.rows),
        "training_union_component_ids": sorted(
            {row["union_component_id"] for row in rows if str(row["generator_fold"]) in triple}
        ),
        "initial_model_logical_sha256": initial_model_sha,
        "checkpoint_logical_sha256": hashes.logical_state_sha256,
        "checkpoint_file_sha256": hashes.file_sha256,
        "model_config": asdict(contract.model_config),
        "model_parameters": 354068,
        "final_step": 1000,
        "training_draw_count": 64000,
        "git_commit": expected_commit,
        "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "node": platform.node(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "safetensors": safetensors.__version__,
            "gpu_name": properties.name,
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "controls": controls,
        },
        "artifacts": artifacts,
        "oracle_calls": 0,
        "production_eligible": False,
        "scientific_superiority_claim": False,
        "independent_verification_passed": False,
    }
    manifest_bytes = canonical(manifest)
    write_new(output / "manifest.json", manifest_bytes)
    require(
        set(path.name for path in output.iterdir()) == ARTIFACT_FILES - {"COMPLETE.json"},
        "checkpoint artifact inventory differs",
    )
    write_new(
        output / "COMPLETE.json",
        canonical(
            {
                "artifact": "generator_namespace_checkpoint_complete_v1",
                "manifest_sha256": sha(manifest_bytes),
            }
        ),
    )
    os.chmod(output, 0o555)
    print(
        json.dumps(
            {
                "completed": str(output),
                "manifest_sha256": sha(manifest_bytes),
                "checkpoint_logical_sha256": hashes.logical_state_sha256,
            }
        ),
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-index", type=int, required=True)
    args = parser.parse_args()
    train_checkpoint(
        load_contract(args.config, args.expected_config_sha256),
        repository=args.repository,
        expected_commit=args.expected_git_commit,
        run_root=args.run_root,
        fit_index=args.fit_index,
    )


if __name__ == "__main__":
    main()

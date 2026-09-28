"""Safe bridge to the specifically audited generator-only direct-logit inits.

This consumer does not infer checkpoint semantics from tensor shapes. It only
accepts the frozen producer/config and independently audited receipt below.
Selection is by an explicit training-fold triple, never held-out diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch

from amp_challenge.generators.diffusion.model import (
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
    load_safetensors_checkpoint,
)

PRODUCER_COMMIT = "b62a1e2ee8e19e117cdd6bb5948a3d3120799634"
EVIDENCE_COMMIT = "ac934586999435c5db162f45f0015ef80a579505"
AUDIT_SHA256 = "badc7d1bdb94f2dcc4c8395b26120be41cfe4dcd43fb09285691aeb71dae6cf5"
CONFIG_SHA256 = "17ff52f007bbd38ab149153dae7b7ace7dbe741f15725aba572febc51cbe22f3"
NAMESPACE_SHA256 = "c430c8704c853a90fd325765251ede189e0edfdb7226922f5234c1d095c09662"
TRIPLES = ("012", "013", "014", "023", "024", "034", "123", "124", "134", "234")
_REPO = Path(__file__).resolve().parents[4]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path, *, limit: int = 16 * 1024**2, immutable: bool = True) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit or before.st_nlink != 1:
            raise ValueError("invalid bounded regular initialization asset")
        if immutable and stat.S_IMODE(before.st_mode) & 0o222:
            raise ValueError("initialization asset must be read-only")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(payload) != before.st_size:
            raise ValueError("initialization asset changed while reading")
        return payload
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class AuditedNativeInitialization:
    model: NativeDenoiser
    triple: str
    training_folds: tuple[int, ...]
    training_sequence_ids: tuple[str, ...]
    training_union_component_ids: tuple[str, ...]
    audit_sha256: str
    manifest_sha256: str
    checkpoint_file_sha256: str
    checkpoint_logical_sha256: str
    producer_commit: str
    qualification: str = "generator_namespace_direct_logit_initialization_only"
    production_input_eligible: bool = False
    scientific_evidence_accepted: bool = False


def _new_cpu_model(config: NativeDenoiserConfig) -> NativeDenoiser:
    # Explicit device prevents ambient CUDA defaults from spending GPU budget or
    # consuming CUDA RNG, which the deliberately CPU-only fork does not save.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        return NativeDenoiser(config)


def load_audited_native_initialization(
    run_root: Path, audit_receipt: Path, *, triple: str
) -> AuditedNativeInitialization:
    """Load one explicitly requested accepted fold-triple checkpoint on CPU.

    Returns authenticated training inventories for downstream namespace exclusion.
    No assay, oracle, held-out-loss, or sample-diagnostic file is read. Caller RNG is
    restored after temporary module initialization; parameter payload uses the
    existing exact-schema, physical+logical SHA-256 safetensors loader.
    """
    if triple not in TRIPLES:
        raise ValueError("requested initialization triple is not in the accepted declaration")
    receipt_bytes = _read(Path(audit_receipt))
    if _sha(receipt_bytes) != AUDIT_SHA256:
        raise ValueError("initialization independent audit receipt pin mismatch")
    receipt = json.loads(receipt_bytes)
    if (
        receipt["artifact"] != "generator_namespace_checkpoint_verification_v1"
        or receipt["git_commit"] != PRODUCER_COMMIT
        or receipt["config_sha256"] != CONFIG_SHA256
        or receipt["namespace_receipt_sha256"] != NAMESPACE_SHA256
        or receipt["independent_execution_passed"] is not True
        or receipt["exact_duplicate_passed"] is not True
        or receipt["oracle_calls"] != 0
        or receipt["production_eligible"] is not False
        or receipt["scientific_superiority_claim"] is not False
    ):
        raise ValueError(
            "initialization audit is not the accepted non-promoting direct-model asset"
        )
    ordinal = TRIPLES.index(triple)
    fits = [
        fit for fit in receipt["fits"] if fit["fit_index"] == ordinal and fit["triple"] == triple
    ]
    if len(fits) != 1:
        raise ValueError("ambiguous initialization identity")
    accepted = fits[0]
    root = Path(run_root) / f"{ordinal:02d}"
    manifest_bytes = _read(root / "manifest.json")
    manifest_sha = _sha(manifest_bytes)
    if manifest_sha != accepted["manifest_sha256"]:
        raise ValueError("initialization manifest pin mismatch")
    manifest = json.loads(manifest_bytes)
    if json.loads(_read(root / "COMPLETE.json")) != {
        "artifact": "generator_namespace_checkpoint_complete_v1",
        "manifest_sha256": manifest_sha,
    }:
        raise ValueError("initialization publication is incomplete")
    if (
        manifest["artifact"] != "generator_namespace_native_checkpoint_v1"
        or manifest["git_commit"] != PRODUCER_COMMIT
        or manifest["config_sha256"] != CONFIG_SHA256
        or manifest["namespace_receipt_sha256"] != NAMESPACE_SHA256
        or manifest["triple"] != triple
        or manifest["fit_index"] != ordinal
        or manifest["training_folds"] != [int(value) for value in triple]
        or manifest["duplicate_of"] is not None
        or manifest["oracle_calls"] != 0
        or manifest["production_eligible"] is not False
        or manifest["scientific_superiority_claim"] is not False
    ):
        raise ValueError("initialization source/fold/qualification mismatch")
    if {path.name for path in root.iterdir()} != set(manifest["artifacts"]) | {
        "manifest.json",
        "COMPLETE.json",
    }:
        raise ValueError("initialization artifact inventory mismatch")

    def bound(name: str) -> bytes:
        payload = _read(root / name)
        expected = manifest["artifacts"][name]
        if len(payload) != expected["bytes"] or _sha(payload) != expected["sha256"]:
            raise ValueError(f"initialization payload binding mismatch: {name}")
        return payload

    config_payload = bound("config.toml")
    if _sha(config_payload) != CONFIG_SHA256:
        raise ValueError("initialization configuration pin mismatch")
    config = tomllib.loads(config_payload.decode())
    model_config = {
        key: value
        for key, value in config["model"].items()
        if key != "expected_trainable_parameters"
    }
    if (
        model_config != manifest["model_config"]
        or manifest["model_parameters"] != config["model"]["expected_trainable_parameters"]
    ):
        raise ValueError("initialization direct-native architecture mismatch")
    source_inventory = {}
    for line in bound("CODE_SHA256SUMS").decode().splitlines():
        digest, name = line.split("  ", 1)
        path = PurePosixPath(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path.is_absolute()
            or ".." in path.parts
            or name in source_inventory
        ):
            raise ValueError("invalid initialization source inventory")
        source_inventory[name] = digest
    for relative in (
        "configs/diffusion/generator_namespace_checkpoints_v1.toml",
        "src/amp_challenge/generators/diffusion/namespace_checkpoint_train.py",
        "src/amp_challenge/generators/diffusion/model.py",
        "src/amp_challenge/generators/diffusion/categorical.py",
    ):
        committed = subprocess.check_output(
            ["git", "show", f"{PRODUCER_COMMIT}:{relative}"], cwd=_REPO
        )
        if source_inventory.get(relative) != _sha(committed):
            raise ValueError("initialization producer source bytes mismatch")
        if relative.endswith(("/model.py", "/categorical.py")) and _sha(
            _read(_REPO / relative, immutable=False)
        ) != _sha(committed):
            raise ValueError(
                "loaded native numerical implementation differs from the accepted initialization"
            )
    projection = [json.loads(line) for line in bound("training_projection.jsonl").splitlines()]
    ids = []
    for row in projection:
        if (
            set(row) != {"sequence_id", "sequence", "sampling_weight"}
            or not isinstance(row["sequence"], str)
            or set(row["sequence"]) - set("ACDEFGHIKLMNPQRSTVWY")
            or not 8 <= len(row["sequence"]) <= 50
            or row["sequence_id"] != _sha(row["sequence"].encode("ascii"))
            or isinstance(row["sampling_weight"], bool)
            or not math.isfinite(row["sampling_weight"])
            or row["sampling_weight"] <= 0
        ):
            raise ValueError("invalid label-free initialization training projection")
        ids.append(row["sequence_id"])
    if (
        len(ids) != len(set(ids))
        or sorted(ids) != manifest["training_sequence_ids"]
        or len(ids) != accepted["training_rows"]
        or len(ids) != manifest["training_row_count"]
    ):
        raise ValueError("initialization training inventory reconstruction mismatch")
    checkpoint_payload = bound("checkpoint.safetensors")
    physical = _sha(checkpoint_payload)
    if (
        physical != accepted["checkpoint_file_sha256"]
        or physical != manifest["checkpoint_file_sha256"]
        or accepted["checkpoint_logical_sha256"] != manifest["checkpoint_logical_sha256"]
    ):
        raise ValueError("initialization checkpoint identity mismatch")
    model = _new_cpu_model(NativeDenoiserConfig(**model_config))
    loaded = load_safetensors_checkpoint(
        model,
        root / "checkpoint.safetensors",
        expected_file_sha256=physical,
        expected_logical_state_sha256=accepted["checkpoint_logical_sha256"],
    )
    model.eval()
    if canonical_model_logical_hash(model) != loaded.logical_state_sha256:
        raise ValueError("loaded initialization logical identity mismatch")
    return AuditedNativeInitialization(
        model,
        triple,
        tuple(int(value) for value in triple),
        tuple(sorted(ids)),
        tuple(manifest["training_union_component_ids"]),
        AUDIT_SHA256,
        manifest_sha,
        physical,
        loaded.logical_state_sha256,
        PRODUCER_COMMIT,
    )

"""Strict preregistration contract for native unconditional diffusion v0.

This module deliberately contains no model or training implementation.  It
turns the reviewed TOML preregistration into immutable typed values and rejects
any schema, type, value, or byte-level drift before an experiment can start.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_SHA256 = "8242e589db8f35710e44fd63343444c3d5ce0f30626933be37561c4cbb49d764"
ARTIFACT = "native_categorical_diffusion_unconditional_v0"
CORPUS_SHA256 = "c03595a8650b732307ada7e030f996d65345238efaea55d25925ac433bec8bc7"
TRAINING_PROJECTION_SHA256 = "127a0eb88c5dc10c94904dcc5a3e98ff75a55890dc29af807e99f3f93c61ae46"
CORPUS_MANIFEST_SHA256 = "8f546c93fcdfe0a6fd28fe7d7c0bea7d4e64ede17fac296fd9b6f9c0cb17d280"
CORPUS_RECEIPT_SHA256 = "6ae50c58098fbfeae3ec28035cf409df14f7818d978b160e68d2326e46c3e2c5"

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "evidence_doc",
        "status",
        "input",
        "model",
        "diffusion",
        "training",
        "determinism",
        "environment",
        "evaluation",
        "baselines",
        "sampling",
        "gates",
        "compute",
        "artifacts",
        "leakage",
    }
)


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect diffusion contract path: {candidate}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"diffusion contract path traverses a symlink: {candidate}")


def _read_contract_bytes(path: Path) -> bytes:
    _reject_symlink_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open diffusion contract: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("diffusion contract must be a non-symlink regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.lstat(path)
        except OSError as error:
            raise ValueError("diffusion contract changed while it was read") from error
        _reject_symlink_chain(path)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _fingerprint(before) != _fingerprint(after)
        or _fingerprint(before) != _fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
        or len(payload) != before.st_size
    ):
        raise ValueError("diffusion contract changed while it was read")
    return payload


@dataclass(frozen=True, slots=True)
class StatusContract:
    before_execution: str
    evidence_invalid: str
    reproducible_no_go: str
    candidate_generator_only: str


@dataclass(frozen=True, slots=True)
class InputContract:
    corpus_artifact: str
    corpus_config_sha256: str
    corpus_sha256: str
    training_projection_sha256: str
    summary_sha256: str
    manifest_sha256: str
    independent_outer_receipt_sha256: str
    organizer_reference_sha256: str
    organizer_reference_records: int
    organizer_reference_role: str
    expected_train_sequences: int
    expected_validation_sequences: int
    train_folds: tuple[int, ...]
    validation_fold: int
    component_weighting: str


@dataclass(frozen=True, slots=True)
class ModelContract:
    kind: str
    alphabet: str
    min_length: int
    max_length: int
    special_tokens: tuple[str, ...]
    layers: int
    hidden_dim: int
    attention_heads: int
    ffn_dim: int
    activation: str
    dropout: float
    layer_norm_epsilon: float
    token_embedding: str
    position_embedding: str
    length_embedding: str
    timestep_embedding: str
    tie_residue_input_output_weights: bool
    attention_projection_bias: bool
    ffn_bias: bool
    final_layer_norm: bool
    output_bias: bool
    prediction_classes: int
    expected_trainable_parameters: int
    initialization: str
    property_conditioning: bool
    classifier_free_guidance: bool
    self_conditioning: bool


@dataclass(frozen=True, slots=True)
class DiffusionProcessContract:
    kind: str
    levels: int
    schedule: str
    cosine_offset: float
    timestep_sampling: str
    mask_count: str
    mask_position_sampling: str
    prediction_target: str
    loss_positions: str
    loss_reduction: str


@dataclass(frozen=True, slots=True)
class TrainingContract:
    seeds: tuple[int, ...]
    reproducibility_seed: int
    batch_sequences: int
    sample_with_replacement: bool
    sampling_weight_application: str
    max_steps: int
    optimizer: str
    learning_rate: float
    betas: tuple[float, ...]
    epsilon: float
    weight_decay: float
    exclude_from_weight_decay: tuple[str, ...]
    warmup_steps: int
    lr_schedule: str
    final_learning_rate: float
    gradient_clip_norm: float
    precision: str
    gradient_accumulation_steps: int
    early_stopping: bool
    checkpoint_selection: str
    validation_during_training: bool
    training_log_interval_steps: int
    ema: bool
    amp: bool
    tf32: bool
    torch_compile: bool


@dataclass(frozen=True, slots=True)
class DeterminismContract:
    rng_derivation: str
    rng_namespaces: tuple[str, ...]
    corruption_rng: str
    validation_rng: str
    proposal_rng: str
    deterministic_algorithms: bool
    math_sdpa_only: bool
    mha_fastpath: bool
    cublas_workspace_config: str
    cudnn_benchmark: bool
    dataloader_workers: int
    pytorch_allocator: str
    require_distinct_node_seed42_twin: bool
    require_exact_seed42_twin: bool
    twin_environment_equal_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    python: str
    numpy: str
    torch: str
    torch_cuda: str
    safetensors: str
    triton: str
    nvidia_cudnn_cu13: str
    gpu_name: str
    compute_capability: tuple[int, ...]
    checkpoint_format: str
    allow_pickle: bool


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    levels: int
    replicates_per_sequence_level: int
    expected_validation_sequences: int
    expected_corruption_cases: int
    evaluation_seed: int
    batch_sequences: int
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    timestep_bins: tuple[str, ...]
    bootstrap_unit: str
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_confidence: float


@dataclass(frozen=True, slots=True)
class BaselineContract:
    names: tuple[str, ...]
    count_pseudocount: float
    count_weighting: str
    transition_count_weighting: str
    bidirectional_markov_combination: str
    relative_position_length_bins: tuple[int, ...]
    relative_position_bins: int
    relative_position_prior_mass: float
    strongest_baseline_rule: str
    generator_controls: tuple[str, ...]
    memorization_control: str


@dataclass(frozen=True, slots=True)
class SamplingContract:
    checkpoint: str
    length_distribution: str
    shared_length_plan_across_methods: bool
    reverse_steps: int
    reverse_target_mask_count: str
    reveal_order: str
    reveal_tie_break: str
    categorical_draw_order: str
    temperature: float
    top_k: int
    top_p: float
    rejection_retries: int
    raw_proposals_per_seed: int
    batch_sequences: int
    report_before_filtering: bool
    length_rng: str
    control_rng: str
    distribution_reference: str
    distribution_candidate_pool: str
    ngram_orders: tuple[int, ...]
    ngram_reference_weighting: str
    descriptor_features: tuple[str, ...]
    descriptor_standardization: str
    descriptor_energy_distance: str
    top_reference_ratio: float
    identity_threshold: float


@dataclass(frozen=True, slots=True)
class DenoisingGateContract:
    require_every_seed_beats_strongest_baseline: bool
    minimum_mean_relative_nll_improvement: float
    minimum_bootstrap_lower_bound_improvement: float
    bootstrap_lower_bound_comparator: str
    bootstrap_seed_aggregation: str
    high_noise_timestep_bin: str
    minimum_high_noise_relative_nll_improvement: float
    high_noise_seed_aggregation: str
    maximum_timestep_bin_relative_nll_regression: float
    timestep_bin_seed_aggregation: str
    maximum_ece: float
    maximum_ece_regression: float
    ece_seed_rule: str


@dataclass(frozen=True, slots=True)
class SamplingGateContract:
    minimum_canonical_valid_fraction_each_seed: float
    minimum_raw_unique_fraction_each_seed: float
    maximum_exact_train_overlap_fraction_each_seed: float
    minimum_common_funnel_yield_fraction_each_seed: float
    minimum_top_reference_safe_count_each_seed: int
    minimum_hill2_effective_70pct_clusters_each_seed: float
    maximum_largest_70pct_cluster_fraction_each_seed: float
    maximum_common_funnel_yield_deficit_vs_best_control: float
    maximum_ngram_jsd_regression_bits_vs_best_control: float
    maximum_descriptor_energy_distance_ratio_vs_best_control: float
    minimum_improvement_in_3mer_jsd_or_descriptor_distance: float
    relative_control_seed_aggregation: str


@dataclass(frozen=True, slots=True)
class ComputeContract:
    account: str
    gpu_partition: str
    cpu_partition: str
    gpu_type: str
    gpus_per_job: int
    cpus_per_task: int
    memory_gib: int
    maximum_gpu_hours_per_training_run: float
    maximum_total_gpu_hours: float
    maximum_peak_gpu_memory_gib: float
    cpu_audit_time_hours: float
    scratch_root_env: str
    run_subdir: str
    execution_command: str


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    schema_version: int
    canonical_json: bool
    reject_nonfinite_json: bool
    file_mode: str
    directory_mode: str
    manifest_published_last: bool
    semantic_manifests_path_free: bool
    independent_verifier_imports_producer: bool
    evaluation_scope: str
    evaluation_seed_order: tuple[int, ...]
    training_bundle_binding_labels: tuple[str, ...]
    proposal_method_order: tuple[str, ...]
    validation_token_stats_method_order: tuple[str, ...]
    npz_archive_format: str
    validation_corruptions_npz_schema: tuple[str, ...]
    validation_token_stats_npz_schema: tuple[str, ...]
    training_bundle_files: tuple[str, ...]
    evaluation_bundle_files: tuple[str, ...]
    independent_receipt_file: str
    training_manifest_fields: tuple[str, ...]
    evaluation_manifest_fields: tuple[str, ...]
    operational_receipt_fields: tuple[str, ...]
    independent_receipt_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeakageContract:
    trainer_allowed_roles: tuple[str, ...]
    trainer_allowed_fields: tuple[str, ...]
    validation_visible_to_trainer: bool
    validation_used_for_early_stopping: bool
    labels_allowed: bool
    provenance_allowed: bool
    study_keys_allowed: bool
    oracle_predictions_allowed: bool
    structures_allowed: bool
    organizer_reference_allowed_during_training: bool
    organizer_reference_use: str
    future_fold4_informed_changes_require_new_contract: bool


@dataclass(frozen=True, slots=True)
class NativeDiffusionContract:
    config_sha256: str
    schema_version: int
    artifact: str
    evidence_doc: str
    status: StatusContract
    input: InputContract
    model: ModelContract
    diffusion: DiffusionProcessContract
    training: TrainingContract
    determinism: DeterminismContract
    environment: EnvironmentContract
    evaluation: EvaluationContract
    baselines: BaselineContract
    sampling: SamplingContract
    denoising_gates: DenoisingGateContract
    sampling_gates: SamplingGateContract
    compute: ComputeContract
    artifacts: ArtifactContract
    leakage: LeakageContract


def _table(
    parent: Mapping[str, Any],
    name: str,
    fields: frozenset[str],
    *,
    label: str | None = None,
) -> Mapping[str, Any]:
    value = parent.get(name)
    section = name if label is None else label
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a TOML table")
    if set(value) != fields:
        raise ValueError(
            f"{section} schema mismatch: "
            f"missing={sorted(fields - set(value))}, extra={sorted(set(value) - fields)}"
        )
    return value


def _string(table: Mapping[str, Any], field: str, section: str) -> str:
    value = table[field]
    if type(value) is not str or not value:
        raise ValueError(f"{section}.{field} must be a non-empty string")
    return value


def _integer(table: Mapping[str, Any], field: str, section: str) -> int:
    value = table[field]
    if type(value) is not int:
        raise ValueError(f"{section}.{field} must be an integer")
    return value


def _number(table: Mapping[str, Any], field: str, section: str) -> float:
    value = table[field]
    if type(value) is not float:
        raise ValueError(f"{section}.{field} must be a TOML float")
    return value


def _boolean(table: Mapping[str, Any], field: str, section: str) -> bool:
    value = table[field]
    if type(value) is not bool:
        raise ValueError(f"{section}.{field} must be a Boolean")
    return value


def _string_tuple(table: Mapping[str, Any], field: str, section: str) -> tuple[str, ...]:
    value = table[field]
    if not isinstance(value, list) or not value or any(type(item) is not str for item in value):
        raise ValueError(f"{section}.{field} must be a non-empty string array")
    return tuple(value)


def _integer_tuple(table: Mapping[str, Any], field: str, section: str) -> tuple[int, ...]:
    value = table[field]
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{section}.{field} must be a non-empty integer array")
    return tuple(value)


def _number_tuple(table: Mapping[str, Any], field: str, section: str) -> tuple[float, ...]:
    value = table[field]
    if not isinstance(value, list) or not value or any(type(item) is not float for item in value):
        raise ValueError(f"{section}.{field} must be a non-empty float array")
    return tuple(value)


def _parse_contract(document: Mapping[str, Any], *, config_sha256: str) -> NativeDiffusionContract:
    if set(document) != _TOP_FIELDS:
        raise ValueError(
            "top-level schema mismatch: "
            f"missing={sorted(_TOP_FIELDS - set(document))}, "
            f"extra={sorted(set(document) - _TOP_FIELDS)}"
        )
    schema_version = _integer(document, "schema_version", "root")
    artifact = _string(document, "artifact", "root")
    evidence_doc = _string(document, "evidence_doc", "root")

    status_fields = frozenset(
        {"before_execution", "evidence_invalid", "reproducible_no_go", "candidate_generator_only"}
    )
    status_raw = _table(document, "status", status_fields)
    status = StatusContract(
        **{field: _string(status_raw, field, "status") for field in status_fields}
    )

    input_fields = frozenset(
        {
            "corpus_artifact",
            "corpus_config_sha256",
            "corpus_sha256",
            "training_projection_sha256",
            "summary_sha256",
            "manifest_sha256",
            "independent_outer_receipt_sha256",
            "organizer_reference_sha256",
            "organizer_reference_records",
            "organizer_reference_role",
            "expected_train_sequences",
            "expected_validation_sequences",
            "train_folds",
            "validation_fold",
            "component_weighting",
        }
    )
    input_raw = _table(document, "input", input_fields)
    input_contract = InputContract(
        corpus_artifact=_string(input_raw, "corpus_artifact", "input"),
        corpus_config_sha256=_string(input_raw, "corpus_config_sha256", "input"),
        corpus_sha256=_string(input_raw, "corpus_sha256", "input"),
        training_projection_sha256=_string(input_raw, "training_projection_sha256", "input"),
        summary_sha256=_string(input_raw, "summary_sha256", "input"),
        manifest_sha256=_string(input_raw, "manifest_sha256", "input"),
        independent_outer_receipt_sha256=_string(
            input_raw, "independent_outer_receipt_sha256", "input"
        ),
        organizer_reference_sha256=_string(input_raw, "organizer_reference_sha256", "input"),
        organizer_reference_records=_integer(input_raw, "organizer_reference_records", "input"),
        organizer_reference_role=_string(input_raw, "organizer_reference_role", "input"),
        expected_train_sequences=_integer(input_raw, "expected_train_sequences", "input"),
        expected_validation_sequences=_integer(input_raw, "expected_validation_sequences", "input"),
        train_folds=_integer_tuple(input_raw, "train_folds", "input"),
        validation_fold=_integer(input_raw, "validation_fold", "input"),
        component_weighting=_string(input_raw, "component_weighting", "input"),
    )

    model_fields = frozenset(ModelContract.__dataclass_fields__)
    model_raw = _table(document, "model", model_fields)
    model = ModelContract(
        kind=_string(model_raw, "kind", "model"),
        alphabet=_string(model_raw, "alphabet", "model"),
        min_length=_integer(model_raw, "min_length", "model"),
        max_length=_integer(model_raw, "max_length", "model"),
        special_tokens=_string_tuple(model_raw, "special_tokens", "model"),
        layers=_integer(model_raw, "layers", "model"),
        hidden_dim=_integer(model_raw, "hidden_dim", "model"),
        attention_heads=_integer(model_raw, "attention_heads", "model"),
        ffn_dim=_integer(model_raw, "ffn_dim", "model"),
        activation=_string(model_raw, "activation", "model"),
        dropout=_number(model_raw, "dropout", "model"),
        layer_norm_epsilon=_number(model_raw, "layer_norm_epsilon", "model"),
        token_embedding=_string(model_raw, "token_embedding", "model"),
        position_embedding=_string(model_raw, "position_embedding", "model"),
        length_embedding=_string(model_raw, "length_embedding", "model"),
        timestep_embedding=_string(model_raw, "timestep_embedding", "model"),
        tie_residue_input_output_weights=_boolean(
            model_raw, "tie_residue_input_output_weights", "model"
        ),
        attention_projection_bias=_boolean(model_raw, "attention_projection_bias", "model"),
        ffn_bias=_boolean(model_raw, "ffn_bias", "model"),
        final_layer_norm=_boolean(model_raw, "final_layer_norm", "model"),
        output_bias=_boolean(model_raw, "output_bias", "model"),
        prediction_classes=_integer(model_raw, "prediction_classes", "model"),
        expected_trainable_parameters=_integer(model_raw, "expected_trainable_parameters", "model"),
        initialization=_string(model_raw, "initialization", "model"),
        property_conditioning=_boolean(model_raw, "property_conditioning", "model"),
        classifier_free_guidance=_boolean(model_raw, "classifier_free_guidance", "model"),
        self_conditioning=_boolean(model_raw, "self_conditioning", "model"),
    )

    diffusion_fields = frozenset(DiffusionProcessContract.__dataclass_fields__)
    diffusion_raw = _table(document, "diffusion", diffusion_fields)
    diffusion = DiffusionProcessContract(
        kind=_string(diffusion_raw, "kind", "diffusion"),
        levels=_integer(diffusion_raw, "levels", "diffusion"),
        schedule=_string(diffusion_raw, "schedule", "diffusion"),
        cosine_offset=_number(diffusion_raw, "cosine_offset", "diffusion"),
        timestep_sampling=_string(diffusion_raw, "timestep_sampling", "diffusion"),
        mask_count=_string(diffusion_raw, "mask_count", "diffusion"),
        mask_position_sampling=_string(diffusion_raw, "mask_position_sampling", "diffusion"),
        prediction_target=_string(diffusion_raw, "prediction_target", "diffusion"),
        loss_positions=_string(diffusion_raw, "loss_positions", "diffusion"),
        loss_reduction=_string(diffusion_raw, "loss_reduction", "diffusion"),
    )

    training_fields = frozenset(TrainingContract.__dataclass_fields__)
    training_raw = _table(document, "training", training_fields)
    training = TrainingContract(
        seeds=_integer_tuple(training_raw, "seeds", "training"),
        reproducibility_seed=_integer(training_raw, "reproducibility_seed", "training"),
        batch_sequences=_integer(training_raw, "batch_sequences", "training"),
        sample_with_replacement=_boolean(training_raw, "sample_with_replacement", "training"),
        sampling_weight_application=_string(
            training_raw, "sampling_weight_application", "training"
        ),
        max_steps=_integer(training_raw, "max_steps", "training"),
        optimizer=_string(training_raw, "optimizer", "training"),
        learning_rate=_number(training_raw, "learning_rate", "training"),
        betas=_number_tuple(training_raw, "betas", "training"),
        epsilon=_number(training_raw, "epsilon", "training"),
        weight_decay=_number(training_raw, "weight_decay", "training"),
        exclude_from_weight_decay=_string_tuple(
            training_raw, "exclude_from_weight_decay", "training"
        ),
        warmup_steps=_integer(training_raw, "warmup_steps", "training"),
        lr_schedule=_string(training_raw, "lr_schedule", "training"),
        final_learning_rate=_number(training_raw, "final_learning_rate", "training"),
        gradient_clip_norm=_number(training_raw, "gradient_clip_norm", "training"),
        precision=_string(training_raw, "precision", "training"),
        gradient_accumulation_steps=_integer(
            training_raw, "gradient_accumulation_steps", "training"
        ),
        early_stopping=_boolean(training_raw, "early_stopping", "training"),
        checkpoint_selection=_string(training_raw, "checkpoint_selection", "training"),
        validation_during_training=_boolean(training_raw, "validation_during_training", "training"),
        training_log_interval_steps=_integer(
            training_raw, "training_log_interval_steps", "training"
        ),
        ema=_boolean(training_raw, "ema", "training"),
        amp=_boolean(training_raw, "amp", "training"),
        tf32=_boolean(training_raw, "tf32", "training"),
        torch_compile=_boolean(training_raw, "torch_compile", "training"),
    )

    determinism_fields = frozenset(DeterminismContract.__dataclass_fields__)
    determinism_raw = _table(document, "determinism", determinism_fields)
    determinism = DeterminismContract(
        rng_derivation=_string(determinism_raw, "rng_derivation", "determinism"),
        rng_namespaces=_string_tuple(determinism_raw, "rng_namespaces", "determinism"),
        corruption_rng=_string(determinism_raw, "corruption_rng", "determinism"),
        validation_rng=_string(determinism_raw, "validation_rng", "determinism"),
        proposal_rng=_string(determinism_raw, "proposal_rng", "determinism"),
        deterministic_algorithms=_boolean(
            determinism_raw, "deterministic_algorithms", "determinism"
        ),
        math_sdpa_only=_boolean(determinism_raw, "math_sdpa_only", "determinism"),
        mha_fastpath=_boolean(determinism_raw, "mha_fastpath", "determinism"),
        cublas_workspace_config=_string(determinism_raw, "cublas_workspace_config", "determinism"),
        cudnn_benchmark=_boolean(determinism_raw, "cudnn_benchmark", "determinism"),
        dataloader_workers=_integer(determinism_raw, "dataloader_workers", "determinism"),
        pytorch_allocator=_string(determinism_raw, "pytorch_allocator", "determinism"),
        require_distinct_node_seed42_twin=_boolean(
            determinism_raw, "require_distinct_node_seed42_twin", "determinism"
        ),
        require_exact_seed42_twin=_boolean(
            determinism_raw, "require_exact_seed42_twin", "determinism"
        ),
        twin_environment_equal_fields=_string_tuple(
            determinism_raw, "twin_environment_equal_fields", "determinism"
        ),
    )

    environment_fields = frozenset(EnvironmentContract.__dataclass_fields__)
    environment_raw = _table(document, "environment", environment_fields)
    environment = EnvironmentContract(
        python=_string(environment_raw, "python", "environment"),
        numpy=_string(environment_raw, "numpy", "environment"),
        torch=_string(environment_raw, "torch", "environment"),
        torch_cuda=_string(environment_raw, "torch_cuda", "environment"),
        safetensors=_string(environment_raw, "safetensors", "environment"),
        triton=_string(environment_raw, "triton", "environment"),
        nvidia_cudnn_cu13=_string(environment_raw, "nvidia_cudnn_cu13", "environment"),
        gpu_name=_string(environment_raw, "gpu_name", "environment"),
        compute_capability=_integer_tuple(environment_raw, "compute_capability", "environment"),
        checkpoint_format=_string(environment_raw, "checkpoint_format", "environment"),
        allow_pickle=_boolean(environment_raw, "allow_pickle", "environment"),
    )

    evaluation_fields = frozenset(EvaluationContract.__dataclass_fields__)
    evaluation_raw = _table(document, "evaluation", evaluation_fields)
    evaluation = EvaluationContract(
        levels=_integer(evaluation_raw, "levels", "evaluation"),
        replicates_per_sequence_level=_integer(
            evaluation_raw, "replicates_per_sequence_level", "evaluation"
        ),
        expected_validation_sequences=_integer(
            evaluation_raw, "expected_validation_sequences", "evaluation"
        ),
        expected_corruption_cases=_integer(
            evaluation_raw, "expected_corruption_cases", "evaluation"
        ),
        evaluation_seed=_integer(evaluation_raw, "evaluation_seed", "evaluation"),
        batch_sequences=_integer(evaluation_raw, "batch_sequences", "evaluation"),
        primary_metric=_string(evaluation_raw, "primary_metric", "evaluation"),
        secondary_metrics=_string_tuple(evaluation_raw, "secondary_metrics", "evaluation"),
        timestep_bins=_string_tuple(evaluation_raw, "timestep_bins", "evaluation"),
        bootstrap_unit=_string(evaluation_raw, "bootstrap_unit", "evaluation"),
        bootstrap_replicates=_integer(evaluation_raw, "bootstrap_replicates", "evaluation"),
        bootstrap_seed=_integer(evaluation_raw, "bootstrap_seed", "evaluation"),
        bootstrap_confidence=_number(evaluation_raw, "bootstrap_confidence", "evaluation"),
    )

    baseline_fields = frozenset(BaselineContract.__dataclass_fields__)
    baseline_raw = _table(document, "baselines", baseline_fields)
    baselines = BaselineContract(
        names=_string_tuple(baseline_raw, "names", "baselines"),
        count_pseudocount=_number(baseline_raw, "count_pseudocount", "baselines"),
        count_weighting=_string(baseline_raw, "count_weighting", "baselines"),
        transition_count_weighting=_string(baseline_raw, "transition_count_weighting", "baselines"),
        bidirectional_markov_combination=_string(
            baseline_raw, "bidirectional_markov_combination", "baselines"
        ),
        relative_position_length_bins=_integer_tuple(
            baseline_raw, "relative_position_length_bins", "baselines"
        ),
        relative_position_bins=_integer(baseline_raw, "relative_position_bins", "baselines"),
        relative_position_prior_mass=_number(
            baseline_raw, "relative_position_prior_mass", "baselines"
        ),
        strongest_baseline_rule=_string(baseline_raw, "strongest_baseline_rule", "baselines"),
        generator_controls=_string_tuple(baseline_raw, "generator_controls", "baselines"),
        memorization_control=_string(baseline_raw, "memorization_control", "baselines"),
    )

    sampling_fields = frozenset(SamplingContract.__dataclass_fields__)
    sampling_raw = _table(document, "sampling", sampling_fields)
    sampling = SamplingContract(
        checkpoint=_string(sampling_raw, "checkpoint", "sampling"),
        length_distribution=_string(sampling_raw, "length_distribution", "sampling"),
        shared_length_plan_across_methods=_boolean(
            sampling_raw, "shared_length_plan_across_methods", "sampling"
        ),
        reverse_steps=_integer(sampling_raw, "reverse_steps", "sampling"),
        reverse_target_mask_count=_string(sampling_raw, "reverse_target_mask_count", "sampling"),
        reveal_order=_string(sampling_raw, "reveal_order", "sampling"),
        reveal_tie_break=_string(sampling_raw, "reveal_tie_break", "sampling"),
        categorical_draw_order=_string(sampling_raw, "categorical_draw_order", "sampling"),
        temperature=_number(sampling_raw, "temperature", "sampling"),
        top_k=_integer(sampling_raw, "top_k", "sampling"),
        top_p=_number(sampling_raw, "top_p", "sampling"),
        rejection_retries=_integer(sampling_raw, "rejection_retries", "sampling"),
        raw_proposals_per_seed=_integer(sampling_raw, "raw_proposals_per_seed", "sampling"),
        batch_sequences=_integer(sampling_raw, "batch_sequences", "sampling"),
        report_before_filtering=_boolean(sampling_raw, "report_before_filtering", "sampling"),
        length_rng=_string(sampling_raw, "length_rng", "sampling"),
        control_rng=_string(sampling_raw, "control_rng", "sampling"),
        distribution_reference=_string(sampling_raw, "distribution_reference", "sampling"),
        distribution_candidate_pool=_string(
            sampling_raw, "distribution_candidate_pool", "sampling"
        ),
        ngram_orders=_integer_tuple(sampling_raw, "ngram_orders", "sampling"),
        ngram_reference_weighting=_string(sampling_raw, "ngram_reference_weighting", "sampling"),
        descriptor_features=_string_tuple(sampling_raw, "descriptor_features", "sampling"),
        descriptor_standardization=_string(sampling_raw, "descriptor_standardization", "sampling"),
        descriptor_energy_distance=_string(sampling_raw, "descriptor_energy_distance", "sampling"),
        top_reference_ratio=_number(sampling_raw, "top_reference_ratio", "sampling"),
        identity_threshold=_number(sampling_raw, "identity_threshold", "sampling"),
    )

    gates_raw = _table(document, "gates", frozenset({"denoising", "sampling"}))
    denoising_fields = frozenset(DenoisingGateContract.__dataclass_fields__)
    denoising_raw = _table(gates_raw, "denoising", denoising_fields, label="gates.denoising")
    denoising_gates = DenoisingGateContract(
        require_every_seed_beats_strongest_baseline=_boolean(
            denoising_raw, "require_every_seed_beats_strongest_baseline", "gates.denoising"
        ),
        minimum_mean_relative_nll_improvement=_number(
            denoising_raw, "minimum_mean_relative_nll_improvement", "gates.denoising"
        ),
        minimum_bootstrap_lower_bound_improvement=_number(
            denoising_raw,
            "minimum_bootstrap_lower_bound_improvement",
            "gates.denoising",
        ),
        bootstrap_lower_bound_comparator=_string(
            denoising_raw,
            "bootstrap_lower_bound_comparator",
            "gates.denoising",
        ),
        bootstrap_seed_aggregation=_string(
            denoising_raw,
            "bootstrap_seed_aggregation",
            "gates.denoising",
        ),
        high_noise_timestep_bin=_string(
            denoising_raw, "high_noise_timestep_bin", "gates.denoising"
        ),
        minimum_high_noise_relative_nll_improvement=_number(
            denoising_raw,
            "minimum_high_noise_relative_nll_improvement",
            "gates.denoising",
        ),
        high_noise_seed_aggregation=_string(
            denoising_raw,
            "high_noise_seed_aggregation",
            "gates.denoising",
        ),
        maximum_timestep_bin_relative_nll_regression=_number(
            denoising_raw,
            "maximum_timestep_bin_relative_nll_regression",
            "gates.denoising",
        ),
        timestep_bin_seed_aggregation=_string(
            denoising_raw,
            "timestep_bin_seed_aggregation",
            "gates.denoising",
        ),
        maximum_ece=_number(denoising_raw, "maximum_ece", "gates.denoising"),
        maximum_ece_regression=_number(denoising_raw, "maximum_ece_regression", "gates.denoising"),
        ece_seed_rule=_string(denoising_raw, "ece_seed_rule", "gates.denoising"),
    )
    sampling_gate_fields = frozenset(SamplingGateContract.__dataclass_fields__)
    sampling_gate_raw = _table(gates_raw, "sampling", sampling_gate_fields, label="gates.sampling")
    sampling_gates = SamplingGateContract(
        minimum_canonical_valid_fraction_each_seed=_number(
            sampling_gate_raw,
            "minimum_canonical_valid_fraction_each_seed",
            "gates.sampling",
        ),
        minimum_raw_unique_fraction_each_seed=_number(
            sampling_gate_raw, "minimum_raw_unique_fraction_each_seed", "gates.sampling"
        ),
        maximum_exact_train_overlap_fraction_each_seed=_number(
            sampling_gate_raw,
            "maximum_exact_train_overlap_fraction_each_seed",
            "gates.sampling",
        ),
        minimum_common_funnel_yield_fraction_each_seed=_number(
            sampling_gate_raw,
            "minimum_common_funnel_yield_fraction_each_seed",
            "gates.sampling",
        ),
        minimum_top_reference_safe_count_each_seed=_integer(
            sampling_gate_raw,
            "minimum_top_reference_safe_count_each_seed",
            "gates.sampling",
        ),
        minimum_hill2_effective_70pct_clusters_each_seed=_number(
            sampling_gate_raw,
            "minimum_hill2_effective_70pct_clusters_each_seed",
            "gates.sampling",
        ),
        maximum_largest_70pct_cluster_fraction_each_seed=_number(
            sampling_gate_raw,
            "maximum_largest_70pct_cluster_fraction_each_seed",
            "gates.sampling",
        ),
        maximum_common_funnel_yield_deficit_vs_best_control=_number(
            sampling_gate_raw,
            "maximum_common_funnel_yield_deficit_vs_best_control",
            "gates.sampling",
        ),
        maximum_ngram_jsd_regression_bits_vs_best_control=_number(
            sampling_gate_raw,
            "maximum_ngram_jsd_regression_bits_vs_best_control",
            "gates.sampling",
        ),
        maximum_descriptor_energy_distance_ratio_vs_best_control=_number(
            sampling_gate_raw,
            "maximum_descriptor_energy_distance_ratio_vs_best_control",
            "gates.sampling",
        ),
        minimum_improvement_in_3mer_jsd_or_descriptor_distance=_number(
            sampling_gate_raw,
            "minimum_improvement_in_3mer_jsd_or_descriptor_distance",
            "gates.sampling",
        ),
        relative_control_seed_aggregation=_string(
            sampling_gate_raw,
            "relative_control_seed_aggregation",
            "gates.sampling",
        ),
    )

    compute_fields = frozenset(ComputeContract.__dataclass_fields__)
    compute_raw = _table(document, "compute", compute_fields)
    compute = ComputeContract(
        account=_string(compute_raw, "account", "compute"),
        gpu_partition=_string(compute_raw, "gpu_partition", "compute"),
        cpu_partition=_string(compute_raw, "cpu_partition", "compute"),
        gpu_type=_string(compute_raw, "gpu_type", "compute"),
        gpus_per_job=_integer(compute_raw, "gpus_per_job", "compute"),
        cpus_per_task=_integer(compute_raw, "cpus_per_task", "compute"),
        memory_gib=_integer(compute_raw, "memory_gib", "compute"),
        maximum_gpu_hours_per_training_run=_number(
            compute_raw, "maximum_gpu_hours_per_training_run", "compute"
        ),
        maximum_total_gpu_hours=_number(compute_raw, "maximum_total_gpu_hours", "compute"),
        maximum_peak_gpu_memory_gib=_number(compute_raw, "maximum_peak_gpu_memory_gib", "compute"),
        cpu_audit_time_hours=_number(compute_raw, "cpu_audit_time_hours", "compute"),
        scratch_root_env=_string(compute_raw, "scratch_root_env", "compute"),
        run_subdir=_string(compute_raw, "run_subdir", "compute"),
        execution_command=_string(compute_raw, "execution_command", "compute"),
    )

    artifact_fields = frozenset(ArtifactContract.__dataclass_fields__)
    artifact_raw = _table(document, "artifacts", artifact_fields)
    artifacts = ArtifactContract(
        schema_version=_integer(artifact_raw, "schema_version", "artifacts"),
        canonical_json=_boolean(artifact_raw, "canonical_json", "artifacts"),
        reject_nonfinite_json=_boolean(artifact_raw, "reject_nonfinite_json", "artifacts"),
        file_mode=_string(artifact_raw, "file_mode", "artifacts"),
        directory_mode=_string(artifact_raw, "directory_mode", "artifacts"),
        manifest_published_last=_boolean(artifact_raw, "manifest_published_last", "artifacts"),
        semantic_manifests_path_free=_boolean(
            artifact_raw, "semantic_manifests_path_free", "artifacts"
        ),
        independent_verifier_imports_producer=_boolean(
            artifact_raw, "independent_verifier_imports_producer", "artifacts"
        ),
        evaluation_scope=_string(artifact_raw, "evaluation_scope", "artifacts"),
        evaluation_seed_order=_integer_tuple(artifact_raw, "evaluation_seed_order", "artifacts"),
        training_bundle_binding_labels=_string_tuple(
            artifact_raw, "training_bundle_binding_labels", "artifacts"
        ),
        proposal_method_order=_string_tuple(artifact_raw, "proposal_method_order", "artifacts"),
        validation_token_stats_method_order=_string_tuple(
            artifact_raw, "validation_token_stats_method_order", "artifacts"
        ),
        npz_archive_format=_string(artifact_raw, "npz_archive_format", "artifacts"),
        validation_corruptions_npz_schema=_string_tuple(
            artifact_raw, "validation_corruptions_npz_schema", "artifacts"
        ),
        validation_token_stats_npz_schema=_string_tuple(
            artifact_raw, "validation_token_stats_npz_schema", "artifacts"
        ),
        training_bundle_files=_string_tuple(artifact_raw, "training_bundle_files", "artifacts"),
        evaluation_bundle_files=_string_tuple(artifact_raw, "evaluation_bundle_files", "artifacts"),
        independent_receipt_file=_string(artifact_raw, "independent_receipt_file", "artifacts"),
        training_manifest_fields=_string_tuple(
            artifact_raw, "training_manifest_fields", "artifacts"
        ),
        evaluation_manifest_fields=_string_tuple(
            artifact_raw, "evaluation_manifest_fields", "artifacts"
        ),
        operational_receipt_fields=_string_tuple(
            artifact_raw, "operational_receipt_fields", "artifacts"
        ),
        independent_receipt_fields=_string_tuple(
            artifact_raw, "independent_receipt_fields", "artifacts"
        ),
    )

    leakage_fields = frozenset(LeakageContract.__dataclass_fields__)
    leakage_raw = _table(document, "leakage", leakage_fields)
    leakage = LeakageContract(
        trainer_allowed_roles=_string_tuple(leakage_raw, "trainer_allowed_roles", "leakage"),
        trainer_allowed_fields=_string_tuple(leakage_raw, "trainer_allowed_fields", "leakage"),
        validation_visible_to_trainer=_boolean(
            leakage_raw, "validation_visible_to_trainer", "leakage"
        ),
        validation_used_for_early_stopping=_boolean(
            leakage_raw, "validation_used_for_early_stopping", "leakage"
        ),
        labels_allowed=_boolean(leakage_raw, "labels_allowed", "leakage"),
        provenance_allowed=_boolean(leakage_raw, "provenance_allowed", "leakage"),
        study_keys_allowed=_boolean(leakage_raw, "study_keys_allowed", "leakage"),
        oracle_predictions_allowed=_boolean(leakage_raw, "oracle_predictions_allowed", "leakage"),
        structures_allowed=_boolean(leakage_raw, "structures_allowed", "leakage"),
        organizer_reference_allowed_during_training=_boolean(
            leakage_raw, "organizer_reference_allowed_during_training", "leakage"
        ),
        organizer_reference_use=_string(leakage_raw, "organizer_reference_use", "leakage"),
        future_fold4_informed_changes_require_new_contract=_boolean(
            leakage_raw,
            "future_fold4_informed_changes_require_new_contract",
            "leakage",
        ),
    )

    return NativeDiffusionContract(
        config_sha256=config_sha256,
        schema_version=schema_version,
        artifact=artifact,
        evidence_doc=evidence_doc,
        status=status,
        input=input_contract,
        model=model,
        diffusion=diffusion,
        training=training,
        determinism=determinism,
        environment=environment,
        evaluation=evaluation,
        baselines=baselines,
        sampling=sampling,
        denoising_gates=denoising_gates,
        sampling_gates=sampling_gates,
        compute=compute,
        artifacts=artifacts,
        leakage=leakage,
    )


def _assert_hard_pins(contract: NativeDiffusionContract) -> None:
    """Make critical scientific boundaries explicit in addition to the byte pin."""

    expected = {
        "schema_version": (contract.schema_version, 1),
        "artifact": (contract.artifact, ARTIFACT),
        "corpus_sha256": (contract.input.corpus_sha256, CORPUS_SHA256),
        "training_projection_sha256": (
            contract.input.training_projection_sha256,
            TRAINING_PROJECTION_SHA256,
        ),
        "corpus_manifest_sha256": (
            contract.input.manifest_sha256,
            CORPUS_MANIFEST_SHA256,
        ),
        "corpus_receipt_sha256": (
            contract.input.independent_outer_receipt_sha256,
            CORPUS_RECEIPT_SHA256,
        ),
        "organizer_reference_sha256": (
            contract.input.organizer_reference_sha256,
            "cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9",
        ),
        "organizer_reference_records": (
            contract.input.organizer_reference_records,
            39_448,
        ),
        "organizer_reference_role": (
            contract.input.organizer_reference_role,
            "post_generation_compliance_audit_only",
        ),
        "train_folds": (contract.input.train_folds, (0, 1, 2, 3)),
        "validation_fold": (contract.input.validation_fold, 4),
        "layers": (contract.model.layers, 4),
        "hidden_dim": (contract.model.hidden_dim, 256),
        "attention_heads": (contract.model.attention_heads, 8),
        "ffn_dim": (contract.model.ffn_dim, 1024),
        "expected_trainable_parameters": (
            contract.model.expected_trainable_parameters,
            3_205_652,
        ),
        "diffusion_levels": (contract.diffusion.levels, 64),
        "seeds": (contract.training.seeds, (42, 43, 44)),
        "max_steps": (contract.training.max_steps, 10_000),
        "precision": (contract.training.precision, "float32"),
        "python": (contract.environment.python, "3.11.14"),
        "numpy": (contract.environment.numpy, "2.4.6"),
        "torch": (contract.environment.torch, "2.14.0"),
        "torch_cuda": (contract.environment.torch_cuda, "13.0"),
        "safetensors": (contract.environment.safetensors, "0.6.2"),
        "triton": (contract.environment.triton, "3.8.0"),
        "nvidia_cudnn_cu13": (
            contract.environment.nvidia_cudnn_cu13,
            "9.24.0.43",
        ),
        "gpu_name": (
            contract.environment.gpu_name,
            "NVIDIA A100-SXM4-80GB",
        ),
        "compute_capability": (contract.environment.compute_capability, (8, 0)),
        "pytorch_allocator": (
            contract.determinism.pytorch_allocator,
            "backend:native",
        ),
        "validation_cases": (contract.evaluation.expected_corruption_cases, 50_944),
        "evaluation_batch_sequences": (contract.evaluation.batch_sequences, 256),
        "raw_proposals_per_seed": (contract.sampling.raw_proposals_per_seed, 2_048),
        "sampling_batch_sequences": (contract.sampling.batch_sequences, 256),
        "bootstrap_lower_bound_comparator": (
            contract.denoising_gates.bootstrap_lower_bound_comparator,
            "strictly_greater_than",
        ),
        "bootstrap_seed_aggregation": (
            contract.denoising_gates.bootstrap_seed_aggregation,
            "arithmetic_mean_relative_improvement",
        ),
        "high_noise_seed_aggregation": (
            contract.denoising_gates.high_noise_seed_aggregation,
            "arithmetic_mean_nll",
        ),
        "timestep_bin_seed_aggregation": (
            contract.denoising_gates.timestep_bin_seed_aggregation,
            "arithmetic_mean_nll",
        ),
        "ece_seed_rule": (
            contract.denoising_gates.ece_seed_rule,
            "every_seed",
        ),
        "evaluation_scope": (
            contract.artifacts.evaluation_scope,
            "cohort_seed_order_42_43_44",
        ),
        "evaluation_seed_order": (
            contract.artifacts.evaluation_seed_order,
            (42, 43, 44),
        ),
        "training_bundle_binding_labels": (
            contract.artifacts.training_bundle_binding_labels,
            ("seed-42-primary", "seed-42-twin", "seed-43", "seed-44"),
        ),
    }
    for label, (observed, required) in expected.items():
        if type(observed) is not type(required) or observed != required:
            raise ValueError(f"{label} must equal the hard-pinned v0 value {required!r}")

    prohibited = (
        contract.model.property_conditioning,
        contract.model.classifier_free_guidance,
        contract.model.self_conditioning,
        contract.training.ema,
        contract.training.amp,
        contract.training.tf32,
        contract.training.torch_compile,
        contract.training.early_stopping,
        contract.training.validation_during_training,
        contract.determinism.mha_fastpath,
        contract.environment.allow_pickle,
        contract.leakage.validation_visible_to_trainer,
        contract.leakage.labels_allowed,
        contract.leakage.provenance_allowed,
        contract.leakage.study_keys_allowed,
        contract.leakage.oracle_predictions_allowed,
        contract.leakage.structures_allowed,
        contract.leakage.organizer_reference_allowed_during_training,
        contract.artifacts.independent_verifier_imports_producer,
    )
    if any(prohibited):
        raise ValueError("unconditional v0 enables a prohibited feature or leakage path")
    if contract.evaluation.expected_corruption_cases != (
        contract.input.expected_validation_sequences
        * contract.evaluation.levels
        * contract.evaluation.replicates_per_sequence_level
    ):
        raise ValueError("evaluation corruption-case census is inconsistent")


def load_unconditional_v0_contract(path: str | Path) -> NativeDiffusionContract:
    """Read and validate the one accepted byte-exact v0 preregistration."""

    source = Path(os.path.abspath(os.fspath(path)))
    payload = _read_contract_bytes(source)
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("diffusion contract must be LF-terminated TOML")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("diffusion contract is not valid UTF-8 TOML") from error
    contract = _parse_contract(document, config_sha256=digest)
    _assert_hard_pins(contract)
    if digest != CONFIG_SHA256:
        raise ValueError(
            f"diffusion contract SHA-256 mismatch: expected {CONFIG_SHA256}, observed {digest}"
        )
    return contract

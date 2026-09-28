"""Frozen execution contract for the native categorical-diffusion-v1 pilot.

This module authenticates and types the immutable parent and child documents.
It deliberately contains no trainer, evaluator, projection producer, or
GPU-launching code and never semantically merges the two contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from amp_challenge.generators.diffusion.v1.contract import (
    ARTIFACT as PARENT_ARTIFACT,
)
from amp_challenge.generators.diffusion.v1.contract import (
    NativeDiffusionV1Contract,
)
from amp_challenge.generators.diffusion.v1.contract import (
    _parse_contract as _parse_parent_contract,
)

CONFIG_SHA256 = "9499e40d348ce2542f99f4363f34b790f7c09a75b2ebcb92d40fe4aad6e9457d"
ARTIFACT = "native_categorical_diffusion_v1_r128_seed42_pilot_execution"
PARENT_CONFIG_SHA256 = "4a13952a606e154b652f8bbc1e8f698b5993ee1a41181867c979729709573d40"
PROJECTION_ARTIFACT = "native_categorical_diffusion_v1_development_projection"
PROJECTION_GIT_COMMIT = "68dd1c70e0277090ad1a3d4a02c7593238855915"
PROJECTION_TOP_SHA256 = "4c0f627990f7f4fa43ebbe969a8422089dbc45a3317869a28035a0af24596657"
PROJECTION_TREE_SHA256 = "c693be7465c5ab2d227bc46378b10c5a407f5f4f5068e2a6046a5f13c1e3e75c"
PROJECTION_MANIFEST_SHA256 = "421241b4fc01387df32f19f14900ff926a41aa3fc5c0d498f77250758035dd81"
PROJECTION_SUMMARY_SHA256 = "696125cc8b7c4c9654fecc92d0390b05e518d50b8cbc9c8a602c620153a85769"
PROJECTION_RECEIPT_SHA256 = "9f4de72ed96de76de1537303ef7b48d351ea1e1a816481ac940a2ad007b12bfd"
PROJECTION_OPERATIONAL_RECEIPT_SHA256 = (
    "df98d251a4bcdcfeec1bd984525cd06c7a64c22a6498a3ab1bdd0338c87275b4"
)

_MAX_CONTRACT_BYTES = 262144

_INHERITANCE_SCHEME = "native_diffusion_v1_authenticated_parent_child_no_merge_v1"
_PARENT_TABLES = (
    "status",
    "prior_evidence",
    "input",
    "development",
    "model",
    "models",
    "diffusion",
    "training",
    "calibration",
    "recipe_selection",
    "baselines",
    "count_prior",
    "sampling",
    "evaluation",
    "gates",
    "confirmation",
    "determinism",
    "environment",
    "compute",
    "telemetry",
    "artifacts",
    "leakage",
)
_RELATION_IDS = (
    "child_parent_identity",
    "r128_variant_and_architecture",
    "seed42_four_outer_fit_scope",
    "checkpoint_grid_and_fixed_horizon",
    "single_corruption_replicate",
    "parent_diffusion_and_loss_reductions",
    "training_recipe",
    "count_prior_algorithm",
    "louco_calibration_and_grids",
    "parent_calibration_zero_lambda_ties_and_ece",
    "parent_bootstrap",
    "pilot_gate",
    "parent_rng_environment_and_determinism",
    "resource_caps",
    "parent_artifact_canonicalization_and_independent_audit",
    "artifact_and_leakage_policy",
)
_PILOT_NARROWINGS = (
    "select_R128_from_parent_pilot_variants",
    "select_parent_seed42",
    "execute_parent_outer_folds_0_through_3_once_each",
    "select_parent_pilot_single_corruption_replicate",
)
_DEFERRED_PARENT_SCOPES = (
    "models.R96",
    "models.D128",
    "development.full_variants",
    "development.full_corruption_replicates_per_sequence_level",
    "sampling",
    "evaluation",
    "gates.denoising",
    "gates.sampling",
    "confirmation",
    "recipe_selection.final_seed42_distinct_node_twins",
    "recipe_selection.fold4_evaluation_twins",
    "determinism.require_exact_seed42_twin",
    "determinism.require_distinct_node_seed42_twin",
    "determinism.training_twin_equality_scope",
    "determinism.evaluation_twin_equality_scope",
)

_HEX = frozenset("0123456789abcdef")
_FOLDS = (0, 1, 2, 3)
_CHECKPOINTS = (250, 500, 1000, 2000, 4000)
_CHECKPOINT_RECEIPT_KEYS = ("000250", "000500", "001000", "002000", "004000")
_FOLD_RECEIPT_KEYS = ("0", "1", "2", "3")
_DIGEST_VALUE_FIELDS = (
    "checkpoint_file_sha256",
    "checkpoint_logical_state_sha256",
    "checkpoint_metadata_sha256",
)
_REINFERENCE_VALUE_FIELDS = (
    "archived_residual_logit_slice_sha256",
    "reinferred_residual_logit_slice_sha256",
    "byte_equal",
)
_FOLD_BUNDLE_VALUE_FIELDS = (
    "trainer_bundle_sha256",
    "evaluator_bundle_sha256",
)
_TREE_ENTRY_FIELDS = ("type", "mode", "size", "sha256", "link_count")
_NPZ_SCHEMA_NAMES = frozenset(
    {
        "count_prior_npz_schema",
        "score_corruptions_npz_schema",
        "score_residual_logits_npz_schema",
        "pilot_bootstrap_npz_schema",
    }
)
_CHECKPOINT_PATHS = (
    "checkpoints/step_000250.safetensors",
    "checkpoints/step_000500.safetensors",
    "checkpoints/step_001000.safetensors",
    "checkpoints/step_002000.safetensors",
    "checkpoints/step_004000.safetensors",
)
_CHECKPOINT_METADATA_PATHS = (
    "checkpoints/step_000250.metadata.json",
    "checkpoints/step_000500.metadata.json",
    "checkpoints/step_001000.metadata.json",
    "checkpoints/step_002000.metadata.json",
    "checkpoints/step_004000.metadata.json",
)
_CHECKPOINT_BUNDLE_PATHS = tuple(
    path
    for pair in zip(_CHECKPOINT_PATHS, _CHECKPOINT_METADATA_PATHS, strict=True)
    for path in pair
)
_MODEL_CONFIG_FIELDS = (
    "kind",
    "alphabet",
    "min_length",
    "max_length",
    "special_tokens",
    "layers",
    "hidden_dim",
    "attention_heads",
    "ffn_dim",
    "expected_trainable_parameters",
    "dropout",
    "layer_norm_epsilon",
    "activation",
    "tie_residue_input_output_weights",
    "prediction_classes",
    "initialization",
    "residual_training_logits",
    "residual_training_lambda",
    "residual_training_temperature",
    "property_conditioning",
    "self_conditioning",
    "geometry_conditioning",
)
_R128_TENSOR_SCHEMA = (
    ("encoder.layers.0.linear1.bias", "F32", (384,)),
    ("encoder.layers.0.linear1.weight", "F32", (384, 128)),
    ("encoder.layers.0.linear2.bias", "F32", (128,)),
    ("encoder.layers.0.linear2.weight", "F32", (128, 384)),
    ("encoder.layers.0.norm1.bias", "F32", (128,)),
    ("encoder.layers.0.norm1.weight", "F32", (128,)),
    ("encoder.layers.0.norm2.bias", "F32", (128,)),
    ("encoder.layers.0.norm2.weight", "F32", (128,)),
    ("encoder.layers.0.self_attn.in_proj_bias", "F32", (384,)),
    ("encoder.layers.0.self_attn.in_proj_weight", "F32", (384, 128)),
    ("encoder.layers.0.self_attn.out_proj.bias", "F32", (128,)),
    ("encoder.layers.0.self_attn.out_proj.weight", "F32", (128, 128)),
    ("encoder.layers.1.linear1.bias", "F32", (384,)),
    ("encoder.layers.1.linear1.weight", "F32", (384, 128)),
    ("encoder.layers.1.linear2.bias", "F32", (128,)),
    ("encoder.layers.1.linear2.weight", "F32", (128, 384)),
    ("encoder.layers.1.norm1.bias", "F32", (128,)),
    ("encoder.layers.1.norm1.weight", "F32", (128,)),
    ("encoder.layers.1.norm2.bias", "F32", (128,)),
    ("encoder.layers.1.norm2.weight", "F32", (128,)),
    ("encoder.layers.1.self_attn.in_proj_bias", "F32", (384,)),
    ("encoder.layers.1.self_attn.in_proj_weight", "F32", (384, 128)),
    ("encoder.layers.1.self_attn.out_proj.bias", "F32", (128,)),
    ("encoder.layers.1.self_attn.out_proj.weight", "F32", (128, 128)),
    ("encoder.norm.bias", "F32", (128,)),
    ("encoder.norm.weight", "F32", (128,)),
    ("length_embedding.weight", "F32", (43, 128)),
    ("position_embedding.weight", "F32", (50, 128)),
    ("residue_output_bias", "F32", (20,)),
    ("timestep_embedding.weight", "F32", (65, 128)),
    ("token_embedding.weight", "F32", (22, 128)),
)
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "evidence_doc",
        "parent",
        "inheritance",
        "contract_io",
        "status",
        "projection",
        "folds",
        "pilot",
        "model",
        "training",
        "count_prior",
        "evaluation",
        "gate",
        "dtypes",
        "rng",
        "checkpoints",
        "checkpoint_metadata",
        "barrier",
        "resources",
        "audit",
        "outputs",
        "leakage",
        "acquisition_exclusions",
    }
)
_TABLE_KEYS = {
    "parent": {
        "artifact",
        "config_path",
        "config_sha256",
        "status_before_pilot",
    },
    "inheritance": {
        "scheme",
        "entire_parent_document_normative",
        "normative_parent_tables",
        "type_exact_relation_ids",
        "omitted_parent_execution_fields",
        "allowed_parent_overrides",
        "pilot_phase_narrowing_ids",
        "inactive_deferred_parent_scopes",
        "narrowings_are_not_parent_overrides",
        "deferred_twin_scope",
    },
    "contract_io": {
        "maximum_bytes",
        "require_regular_file",
        "require_single_link",
        "reject_symlink_ancestors",
        "digest_before_toml_parse",
    },
    "status": {
        "before_execution",
        "fold_complete",
        "evidence_invalid",
        "pilot_no_go",
        "pilot_continue",
        "missing_or_incomplete_fold",
        "evidence_invalid_is_scientific_no_go",
        "scientific_no_go_requires_valid_complete_pilot",
        "continuation_grants_candidate_status",
        "continuation_grants_proposal_or_library_rights",
    },
    "projection": {
        "artifact",
        "producer_job_id",
        "audit_job_id",
        "execution_git_commit",
        "scratch_root_env",
        "canonical_twin_slot",
        "verified_twin_slots",
        "canonical_bundle_relative_path",
        "independent_receipt_relative_path",
        "operational_receipt_relative_path",
        "bundle_top_manifest_sha256",
        "bundle_tree_sha256",
        "manifest_sha256",
        "summary_sha256",
        "independent_receipt_sha256",
        "operational_receipt_sha256",
        "bundle_file_mode",
        "bundle_directory_mode",
        "fold4_sequence_rows",
    },
    "pilot": {
        "variant",
        "output_mode",
        "selection_seed",
        "outer_folds",
        "fit_count",
        "replicas_per_outer_fold",
        "corruption_replicates_per_sequence_level",
        "diffusion_levels",
        "checkpoint_steps",
        "max_steps",
        "checkpoint_selection_rule",
        "all_four_outer_fits_required",
        "reuse_exact_fold_bundles_in_full_matrix",
        "exact_seed42_twins_required_for_pilot",
        "exact_seed42_twins_scope",
        "stops_before_sampling",
        "proposal_sampling_enabled",
        "expected_proposal_count",
    },
    "model": {
        "kind",
        "alphabet",
        "min_length",
        "max_length",
        "special_tokens",
        "layers",
        "hidden_dim",
        "attention_heads",
        "ffn_dim",
        "expected_trainable_parameters",
        "dropout",
        "layer_norm_epsilon",
        "activation",
        "tie_residue_input_output_weights",
        "prediction_classes",
        "initialization",
        "residual_training_logits",
        "residual_training_lambda",
        "residual_training_temperature",
        "property_conditioning",
        "self_conditioning",
        "geometry_conditioning",
    },
    "training": {
        "batch_sequences",
        "sample_with_replacement",
        "sampling_weight_application",
        "optimizer",
        "learning_rate",
        "betas",
        "epsilon",
        "weight_decay",
        "adamw_fused",
        "adamw_foreach",
        "adamw_amsgrad",
        "adamw_capturable",
        "adamw_differentiable",
        "adamw_maximize",
        "weight_decay_includes",
        "exclude_from_weight_decay",
        "parameter_group_order",
        "warmup_steps",
        "lr_schedule",
        "final_learning_rate",
        "learning_rate_step_index",
        "learning_rate_equation",
        "learning_rate_assignment",
        "schedule_digest_domain",
        "schedule_digest_step_encoding",
        "schedule_digest_rate_encoding",
        "gradient_clip_norm",
        "gradient_clip_operation",
        "label_smoothing",
        "visible_context_dropout",
        "visible_context_dropout_changes_loss_mask",
        "gradient_accumulation_steps",
        "zero_grad_set_to_none",
        "optimizer_step_order",
        "validation_during_training",
        "training_log_interval_steps",
        "ema",
        "amp",
        "tf32",
        "torch_compile",
        "resume_from_checkpoint_allowed",
        "partial_or_unsealed_checkpoint_promotion_allowed",
    },
    "count_prior": {
        "name",
        "input_path",
        "input_sha256",
        "algorithm_source_commit",
        "algorithm_source_path",
        "algorithm_source_sha256",
        "fit_rows",
        "effective_count_scale",
        "row_weight",
        "residue_contribution",
        "unigram_pseudocount_per_residue",
        "unigram_normalization",
        "length_edges",
        "length_bin",
        "relative_position_bins",
        "relative_position_bin",
        "relative_position_prior_mass",
        "relative_position_prior",
        "relative_position_normalization",
        "log_floor",
        "log_floor_rule",
        "relative_position_probability_output",
        "log_relative_position_probability",
        "production_order",
        "trainer_output_path",
        "evaluator_reuse",
        "recomputation_during_evaluation_allowed",
        "training_bridge",
    },
    "evaluation": {
        "evaluation_seed",
        "batch_sequences",
        "levels",
        "replicates_per_sequence_level",
        "timestep_bins",
        "calibration_crossfit",
        "residual_lambda_grid",
        "temperature_grid",
        "primary_metric",
        "outer_fold_aggregation",
        "bootstrap_unit",
        "bootstrap_replicates",
        "bootstrap_seed",
        "bootstrap_rng",
        "bootstrap_standard_error",
        "calibration_choices_are",
        "score_rows_may_select_checkpoint",
        "score_rows_may_change_training",
    },
    "gate": {
        "candidate",
        "comparator",
        "minimum_mean_relative_nll_improvement",
        "minimum_bootstrap_lower_bound_improvement",
        "bootstrap_lower_bound_comparator",
        "maximum_fold_relative_nll_regression",
        "maximum_timestep_bin_relative_nll_regression",
        "maximum_ece",
        "maximum_ece_regression",
        "ece_scope",
        "require_all_four_outer_fits",
        "on_failure",
        "on_pass",
    },
    "dtypes": {
        "model_parameter",
        "model_activation",
        "model_logit",
        "loss_and_gradient",
        "encoded_token",
        "attention_mask",
        "sequence_length",
        "timestep",
        "scheduled_mask",
        "target_token",
        "numpy_sampling_probability",
        "numpy_metric_accumulator",
        "checkpoint_tensor",
        "score_count_log_probability",
        "score_residual_logit_storage",
        "score_residual_logit_calibration_compute",
        "score_residual_logit_cast_rule",
        "count_prior_archive_log_probability",
        "count_prior_bridge_source_tensor",
        "count_prior_training_tensor",
        "count_prior_gathered_logit",
        "count_prior_residual_addition",
        "allow_implicit_dtype_coercion",
        "allow_nonfinite",
    },
    "rng": {
        "derivation",
        "training_root_seed",
        "evaluation_root_seed",
        "bootstrap_root_seed",
        "training_namespaces",
        "initialization_key",
        "minibatch_key",
        "timestep_key",
        "corruption_key",
        "context_dropout_key",
        "model_dropout_key",
        "validation_namespace",
        "validation_key",
        "bootstrap_namespace",
        "bootstrap_key",
        "fit_identity_fields",
        "fit_identity_serialization",
        "global_draw_ordinal",
        "minibatch_rng",
        "timestep_rng",
        "corruption_rng",
        "context_dropout_rng",
        "model_dropout_rng",
        "validation_rng",
        "checkpoint_restores_mutable_rng_state",
        "seed_domain_utf8_without_terminal_nul",
        "seed_domain_terminal_nul",
        "seed_hash",
        "seed_framing_length",
        "root_seed_encoding",
        "namespace_encoding",
        "string_part_encoding",
        "integer_part_encoding",
        "seed_output",
        "uniform_conversion",
        "minibatch_probability_normalization",
        "minibatch_cumulative_sum",
        "minibatch_search",
        "bounded_integer_rejection",
        "corruption_consumption",
        "context_dropout_consumption",
        "model_dropout_consumption",
        "validation_row_seed",
        "case_id_domain_utf8_without_terminal_nul",
        "case_id_domain_terminal_nul",
        "case_id_fields",
        "case_id_framing",
        "case_id_output",
    },
    "checkpoints": {
        "steps",
        "relative_paths",
        "metadata_relative_paths",
        "format",
        "tensor_dtype",
        "save_moment",
        "state_scope",
        "tensor_name_order",
        "optimizer_state_serialized",
        "rng_state_serialized",
        "pickle_allowed",
        "evaluate_in_ascending_step_order",
        "all_checkpoints_sealed_before_score_open",
        "producer_gpu_reinference",
        "producer_gpu_reinference_comparison",
        "producer_gpu_reinference_timing",
    },
    "checkpoint_metadata": {
        "schema_version",
        "fields",
        "checkpoint_step_key_order",
        "optimizer_step_completed",
        "count_prior_binding",
        "model_binding_fields",
        "model_config_fields",
        "model_config_binding",
        "bindings",
        "physical_hash",
        "logical_hash",
        "tensor_order",
        "tensor_dtype",
        "tensor_count",
        "tensors",
    },
    "barrier": {
        "execution_topology",
        "training_stage_receives_score_paths",
        "trainer_input_staging",
        "trainer_environment_exposes_projection_root",
        "readiness_condition",
        "readiness_receipt_relative_path",
        "readiness_receipt_fields",
        "checkpoint_digest_key_order",
        "checkpoint_digest_value_fields",
        "readiness_checkpoint_digest_map_schema",
        "required_readiness_receipts",
        "coordinator_verifies_distinct_outer_folds",
        "coordinator_verifies_all_checkpoint_hashes",
        "score_release_condition",
        "score_release_relative_path",
        "score_release_fields",
        "release_digest_key_order",
        "release_readiness_digest_map_schema",
        "digest_map_serialization",
        "digest_map_sha256_rule",
        "score_stage_inputs_materialized_after_release",
        "score_stage_opens_only_own_outer_fold",
        "score_stage_starts_after_release",
        "barrier_failure_status",
    },
    "resources": {
        "account",
        "partition",
        "gpu_type",
        "allocation_nodes",
        "worker_tasks",
        "maximum_concurrent_fit_tasks",
        "nodes_per_fit",
        "tasks_per_node",
        "gpus_per_task",
        "cpus_per_task",
        "memory_gib_per_task",
        "wall_minutes_per_fit",
        "maximum_pilot_a100_hours",
        "maximum_peak_allocated_memory_gib",
        "maximum_total_allocated_gpu_seconds",
        "maximum_peak_allocated_memory_bytes",
        "bare_exclusive_allowed",
        "execution_command",
        "run_subdir",
        "require_clean_synchronized_commit",
    },
    "audit": {
        "account",
        "partition",
        "nodes",
        "tasks",
        "gpus",
        "cpus_per_task",
        "memory_gib_per_task",
        "wall_minutes",
        "third_node_required",
        "excluded_nodes",
        "producer_gpu_reinference_required",
        "producer_gpu_reinference_scope",
        "cpu_exact_reconstruction",
        "cpu_checkpoint_verification",
        "cpu_neural_forward_pass",
        "independent_verifier_imports_producer",
        "independent_receipt_file",
        "operational_receipt_file",
        "independent_receipt_fields",
        "operational_receipt_fields",
        "fold_digest_map_schema",
        "fold_digest_map_fields",
        "checkpoint_digest_by_fold_and_step_schema",
        "producer_gpu_reinference_schema",
        "receipt_map_serialization",
        "receipt_map_sha256_rule",
        "receipt_file_mode",
        "receipt_directory_mode",
    },
    "outputs": {
        "schema_version",
        "canonical_json",
        "reject_nonfinite_json",
        "file_mode",
        "directory_mode",
        "file_link_count",
        "manifest_published_last",
        "publication",
        "semantic_manifests_path_free",
        "bundled_contract_files",
        "bundled_contract_hashes",
        "trainer_bundle_relative_path",
        "evaluator_bundle_relative_path",
        "pilot_bundle_relative_path",
        "trainer_bundle_files",
        "trainer_bundle_directories",
        "evaluator_bundle_files",
        "evaluator_bundle_directories",
        "pilot_bundle_files",
        "pilot_bundle_directories",
        "trainer_manifest_fields",
        "evaluator_manifest_fields",
        "pilot_manifest_fields",
        "npz_archive_format",
        "npz_allow_pickle",
        "npz_member_order",
        "comparator_index_order",
        "score_row_order",
        "score_case_order",
        "score_selected_token_order",
        "checkpoint_axis_order",
        "residue_axis_order",
        "count_prior_relative_position_axis_order",
        "count_prior_npz_schema",
        "score_corruptions_npz_schema",
        "score_residual_logits_npz_schema",
        "pilot_bootstrap_npz_schema",
        "operational_telemetry_excluded_from_semantic_bundles",
        "independent_verifier_imports_producer",
        "independent_receipt_file",
        "operational_receipt_file",
        "post_execution_third_node_audit_required",
        "bundle_identity",
        "sha256_sidecars",
    },
    "leakage": {
        "trainer_allowed_roles",
        "trainer_allowed_fields",
        "trainer_process_receives_score_path",
        "trainer_process_receives_projection_root",
        "score_opened_only_after_all_checkpoints_sealed",
        "validation_used_for_early_stopping",
        "fold4_visible",
        "fold4_path_allowed",
        "fold4_sequence_rows_allowed",
        "labels_allowed",
        "provenance_allowed",
        "study_keys_allowed",
        "oracle_predictions_allowed",
        "structures_allowed",
        "organizer_reference_allowed",
        "proposal_sampling_allowed",
        "post_fold4_changes_require_new_version",
    },
    "acquisition_exclusions": {
        "acquisition_configs_allowed",
        "acquisition_outputs_allowed",
        "start_ids_allowed",
        "rollout_outputs_allowed",
        "reward_mean_allowed",
        "reward_uncertainty_allowed",
        "reward_ucb_allowed",
        "generator_uncertainty_allowed",
        "wet_lab_results_allowed",
        "memory_or_history_allowed",
        "ensemble_predictions_allowed",
        "excluded_scope",
    },
}
_FOLD_KEYS = {
    "outer_fold",
    "fit_folds",
    "train_path",
    "train_sha256",
    "train_rows",
    "train_homology_components",
    "train_union_components",
    "score_path",
    "score_sha256",
    "score_rows",
    "score_homology_components",
    "score_union_components",
    "score_cases",
    "score_selected_tokens",
    "score_selected_tokens_by_timestep_bin",
    "fit_identity_sha256",
}


@dataclass(frozen=True, slots=True)
class PilotFoldContract:
    """Exact input and census for one held-out development fold."""

    outer_fold: int
    fit_folds: tuple[int, ...]
    train_path: str
    train_sha256: str
    train_rows: int
    train_homology_components: int
    train_union_components: int
    score_path: str
    score_sha256: str
    score_rows: int
    score_homology_components: int
    score_union_components: int
    score_cases: int
    score_selected_tokens: int
    score_selected_tokens_by_timestep_bin: tuple[int, ...]
    fit_identity_sha256: str


@dataclass(frozen=True, slots=True)
class NpzArrayContract:
    """One exact, endian-qualified array in a pickle-free NPZ archive."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CheckpointTensorContract:
    """One exact tensor in the R128 logical checkpoint state."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeDiffusionV1PilotContract:
    """Typed pins plus separate recursively immutable parent and child views."""

    config_sha256: str
    parent_config_sha256: str
    projection_top_sha256: str
    projection_tree_sha256: str
    projection_receipt_sha256: str
    seed: int
    folds: tuple[PilotFoldContract, ...]
    checkpoint_steps: tuple[int, ...]
    checkpoint_paths: tuple[str, ...]
    maximum_pilot_a100_hours: float
    trainer_bundle_files: tuple[str, ...]
    evaluator_bundle_files: tuple[str, ...]
    pilot_bundle_files: tuple[str, ...]
    checkpoint_tensors: tuple[CheckpointTensorContract, ...]
    parent_contract: NativeDiffusionV1Contract
    document: Mapping[str, object]

    def revalidate(self) -> NativeDiffusionV1PilotContract:
        """Reparse both frozen documents and reject any in-memory relabeling."""

        if type(self) is not NativeDiffusionV1PilotContract:
            raise TypeError("pilot contract must be an exact NativeDiffusionV1PilotContract")
        if self.config_sha256 != CONFIG_SHA256:
            raise ValueError("pilot contract child digest was relabeled")
        if self.parent_config_sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("pilot contract parent digest was relabeled")
        parent_document = _thaw(self.parent_contract.document)
        child_document = _thaw(self.document)
        if not isinstance(parent_document, dict) or not isinstance(child_document, dict):
            raise ValueError("pilot contract documents lost their frozen mapping structure")
        rebuilt_parent = _parse_parent_contract(
            parent_document,
            config_sha256=PARENT_CONFIG_SHA256,
        )
        rebuilt = _parse_contract(
            child_document,
            config_sha256=CONFIG_SHA256,
            parent_contract=rebuilt_parent,
        )
        if self != rebuilt:
            raise ValueError("pilot contract typed fields differ from its authenticated documents")
        return self

    def table(self, name: str) -> Mapping[str, object]:
        value = self.document[name]
        if not isinstance(value, Mapping):  # pragma: no cover - construction invariant
            raise RuntimeError(f"pilot contract field {name!r} is not a table")
        return value

    def parent_table(self, name: str) -> Mapping[str, object]:
        """Return one immutable table from the authenticated parent document."""

        return self.parent_contract.table(name)

    def fold(self, outer_fold: int) -> PilotFoldContract:
        """Return the unique contract for ``outer_fold``."""

        if type(outer_fold) is not int:
            raise TypeError("outer_fold must be an exact integer")
        for item in self.folds:
            if item.outer_fold == outer_fold:
                return item
        raise KeyError(outer_fold)

    def fit_identity_document(self, outer_fold: int) -> Mapping[str, object]:
        """Build the exact path-free parent-protocol fit identity."""

        item = self.fold(outer_fold)
        pilot = self.table("pilot")
        return MappingProxyType(
            {
                "fit_folds": item.fit_folds,
                "fit_projection_sha256": item.train_sha256,
                "output_mode": pilot["output_mode"],
                "parent_contract_sha256": self.parent_config_sha256,
                "seed": self.seed,
                "variant": pilot["variant"],
            }
        )

    def fit_identity_bytes(self, outer_fold: int) -> bytes:
        """Serialize a fit identity using the parent's frozen canonical JSON rule."""

        document = dict(self.fit_identity_document(outer_fold))
        document["fit_folds"] = list(document["fit_folds"])
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def fit_identity_sha256(self, outer_fold: int) -> str:
        return hashlib.sha256(self.fit_identity_bytes(outer_fold)).hexdigest()

    def readiness_checkpoint_digest_map_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate and serialize one fold's exact five-checkpoint digest map."""

        return _canonical_digest_map_bytes(
            value,
            exact_keys=_CHECKPOINT_RECEIPT_KEYS,
            value_fields=_DIGEST_VALUE_FIELDS,
            label="readiness checkpoint digest map",
        )

    def release_readiness_digest_map_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate and serialize the exact four-fold readiness receipt map."""

        return _canonical_digest_map_bytes(
            value,
            exact_keys=_FOLD_RECEIPT_KEYS,
            value_fields=None,
            label="score-release readiness digest map",
        )

    def audit_fold_digest_map_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate and serialize a four-fold flat digest map from the audit receipt."""

        return _canonical_digest_map_bytes(
            value,
            exact_keys=_FOLD_RECEIPT_KEYS,
            value_fields=None,
            label="audit fold digest map",
        )

    def audit_checkpoint_digest_map_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate and serialize the audit's exact four-by-five checkpoint map."""

        return _canonical_fold_step_digest_map_bytes(
            value,
            value_fields=_DIGEST_VALUE_FIELDS,
            label="audit checkpoint digest map",
        )

    def producer_gpu_reinference_receipt_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate the producer's exact four-by-five byte-equality receipt."""

        if not isinstance(value, Mapping):
            raise TypeError("producer GPU reinference receipt must be a mapping")
        _exact_keys(
            value,
            {"by_fold_and_step", "all_fold_step_pairs_byte_equal"},
            label="producer GPU reinference receipt",
        )
        if value["all_fold_step_pairs_byte_equal"] is not True:
            raise ValueError("producer GPU reinference aggregate must be exact true")
        by_fold = value["by_fold_and_step"]
        if not isinstance(by_fold, Mapping):
            raise ValueError("producer GPU reinference by-fold value must be a mapping")
        document = _fold_step_digest_map_document(
            by_fold,
            value_fields=_REINFERENCE_VALUE_FIELDS,
            label="producer GPU reinference by-fold map",
            exact_true_fields=frozenset({"byte_equal"}),
        )
        for fold in _FOLD_RECEIPT_KEYS:
            fold_document = document[fold]
            if not isinstance(fold_document, dict):  # pragma: no cover - invariant
                raise RuntimeError("reinference fold map is not a dictionary")
            for step in _CHECKPOINT_RECEIPT_KEYS:
                item = fold_document[step]
                if not isinstance(item, dict):  # pragma: no cover - invariant
                    raise RuntimeError("reinference step map is not a dictionary")
                if (
                    item["archived_residual_logit_slice_sha256"]
                    != item["reinferred_residual_logit_slice_sha256"]
                ):
                    raise ValueError(
                        f"producer GPU reinference digest mismatch at fold {fold} step {step}"
                    )
        return _canonical_json_bytes(
            {
                "all_fold_step_pairs_byte_equal": True,
                "by_fold_and_step": document,
            }
        )

    def fold_bundle_digest_map_bytes(self, value: Mapping[str, object]) -> bytes:
        """Validate the exact trainer/evaluator bundle digest map for all folds."""

        return _canonical_digest_map_bytes(
            value,
            exact_keys=_FOLD_RECEIPT_KEYS,
            value_fields=_FOLD_BUNDLE_VALUE_FIELDS,
            label="fold bundle digest map",
        )

    def bundle_tree_map_bytes(self, bundle_kind: str, entries: Mapping[str, object]) -> bytes:
        """Validate and serialize one path-free exact semantic-bundle tree map."""

        if type(bundle_kind) is not str or bundle_kind not in {
            "trainer",
            "evaluator",
            "pilot",
        }:
            raise ValueError("bundle_kind must be exactly trainer, evaluator, or pilot")
        if not isinstance(entries, Mapping):
            raise TypeError("bundle tree entries must be a mapping")
        files = tuple(getattr(self, f"{bundle_kind}_bundle_files"))
        directories_raw = self.table("outputs")[f"{bundle_kind}_bundle_directories"]
        if not isinstance(directories_raw, tuple):  # pragma: no cover - parse invariant
            raise RuntimeError("frozen bundle directory inventory is not a tuple")
        directories = tuple(directories_raw)
        expected_paths = (*directories, *files)
        _exact_keys(entries, set(expected_paths), label=f"{bundle_kind} bundle tree entries")
        entry_types = {
            path: "directory" if path in directories else "file" for path in expected_paths
        }
        document_entries: dict[str, object] = {}
        for path in expected_paths:
            raw = entries[path]
            if not isinstance(raw, Mapping):
                raise ValueError(f"bundle tree entry {path!r} must be an object")
            _exact_keys(raw, set(_TREE_ENTRY_FIELDS), label=f"bundle tree entry {path!r}")
            if entry_types[path] == "file":
                expected = {
                    "type": "file",
                    "mode": "0444",
                    "size": _integer(raw["size"], label=f"bundle tree {path}.size"),
                    "sha256": _sha256(raw["sha256"], label=f"bundle tree {path}.sha256"),
                    "link_count": 1,
                }
            else:
                child_types = _direct_child_types(path, entry_types)
                expected = {
                    "type": "directory",
                    "mode": "0555",
                    "size": len(child_types),
                    "sha256": hashlib.sha256(_canonical_json_bytes(child_types)).hexdigest(),
                    "link_count": 2
                    + sum(child_type == "directory" for child_type in child_types.values()),
                }
            _require_exact(raw, expected, label=f"bundle tree entry {path!r}")
            document_entries[path] = dict(expected)
        return _canonical_json_bytes({"entries": document_entries, "schema_version": 1})

    @staticmethod
    def canonical_map_sha256(payload: bytes) -> str:
        """Hash exact canonical JSON map bytes without accepting text coercions."""

        if type(payload) is not bytes:
            raise TypeError("canonical map payload must be exact bytes")
        try:
            decoded = json.loads(
                payload.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("canonical map payload is not valid finite UTF-8 JSON") from error
        if not isinstance(decoded, dict) or _canonical_json_bytes(decoded) != payload:
            raise ValueError("canonical map payload is not in the frozen byte encoding")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def sha256_sidecar_bytes(digest: str) -> bytes:
        """Serialize a lowercase SHA-256 as exactly 64 ASCII bytes plus LF."""

        return (_sha256(digest, label="sidecar digest") + "\n").encode("ascii")

    def resolved_npz_schema(
        self, name: str, outer_fold: int | None = None
    ) -> tuple[NpzArrayContract, ...]:
        """Resolve one frozen NPZ schema to concrete, path-independent shapes."""

        if type(name) is not str or name not in _NPZ_SCHEMA_NAMES:
            raise KeyError(name)
        outputs = self.table("outputs")
        raw = outputs.get(name)
        if not isinstance(raw, tuple) or any(not isinstance(item, Mapping) for item in raw):
            raise KeyError(name)
        values: Mapping[str, int] = {}
        if outer_fold is not None:
            fold = self.fold(outer_fold)
            values = {
                "score_rows": fold.score_rows,
                "score_cases": fold.score_cases,
                "score_cases_plus_one": fold.score_cases + 1,
                "score_selected_tokens": fold.score_selected_tokens,
            }
        result: list[NpzArrayContract] = []
        for item in raw:
            shape = item.get("shape")
            if not isinstance(shape, tuple):  # pragma: no cover - parser invariant
                raise RuntimeError("frozen NPZ shape is not a tuple")
            dimensions: list[int] = []
            for dimension in shape:
                if type(dimension) is not str:  # pragma: no cover - parser invariant
                    raise RuntimeError("frozen NPZ dimension is not a string")
                try:
                    resolved = dimension.format_map(values)
                except KeyError as error:
                    raise ValueError(f"{name} requires an outer fold") from error
                dimensions.append(int(resolved))
            result.append(
                NpzArrayContract(
                    name=str(item["name"]),
                    dtype=str(item["dtype"]),
                    shape=tuple(dimensions),
                )
            )
        return tuple(result)


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect pilot contract path: {candidate}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"pilot contract path traverses a symlink: {candidate}")


def _read_contract_bytes(path: Path) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect v1 pilot contract: {source}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError("v1 pilot contract must be a non-symlink regular file")
    if named_before.st_nlink != 1:
        raise ValueError("v1 pilot contract must have exactly one hard link")
    if named_before.st_size <= 0 or named_before.st_size > _MAX_CONTRACT_BYTES:
        raise ValueError(f"v1 pilot contract size must be 1..{_MAX_CONTRACT_BYTES} bytes")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open v1 pilot contract: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("v1 pilot contract must be a non-symlink regular file")
        if before.st_nlink != 1:
            raise ValueError("v1 pilot contract must have exactly one hard link")
        if before.st_size <= 0 or before.st_size > _MAX_CONTRACT_BYTES:
            raise ValueError(f"v1 pilot contract size must be 1..{_MAX_CONTRACT_BYTES} bytes")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(65536, _MAX_CONTRACT_BYTES + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_CONTRACT_BYTES:
                raise ValueError(
                    f"v1 pilot contract size must be at most {_MAX_CONTRACT_BYTES} bytes"
                )
        after = os.fstat(descriptor)
        try:
            named = os.lstat(source)
        except OSError as error:
            raise ValueError("v1 pilot contract changed while it was read") from error
        _reject_symlink_chain(source)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _fingerprint(named_before) != _fingerprint(before)
        or before.st_nlink != 1
        or after.st_nlink != 1
        or named.st_nlink != 1
        or _fingerprint(before) != _fingerprint(after)
        or _fingerprint(before) != _fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
        or len(payload) != before.st_size
    ):
        raise ValueError("v1 pilot contract changed while it was read")
    return payload


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    """Recreate TOML-shaped mutable containers for integrity reparsing."""

    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strict_equal(left: object, right: object) -> bool:
    """Compare recursively without Python's bool/int/float equivalences."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if len(left) != len(right):
            return False
        right_keys = tuple(right.keys())
        for left_key, left_value in left.items():
            if not any(
                type(left_key) is type(right_key) and left_key == right_key
                for right_key in right_keys
            ):
                return False
            right_key = next(
                right_key
                for right_key in right_keys
                if type(left_key) is type(right_key) and left_key == right_key
            )
            if not _strict_equal(left_value, right[right_key]):
                return False
        return True
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if not _strict_equal(value, expected):
        raise ValueError(f"{label} is invalid or has a non-exact TOML type")


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _exact_keys(table: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if any(type(key) is not str for key in table):
        raise ValueError(f"{label} contains a non-string key")
    observed = set(table)
    if observed != expected:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite TOML float")
    return value


def _tuple_of(value: object, kind: type, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value or any(type(item) is not kind for item in value):
        raise ValueError(f"{label} must be a non-empty {kind.__name__} array")
    return tuple(value)


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    _validate_recursive(value, label="canonical JSON value")
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value cannot be serialized as canonical UTF-8 JSON") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _canonical_digest_map_bytes(
    value: Mapping[str, object],
    *,
    exact_keys: tuple[str, ...],
    value_fields: tuple[str, ...] | None,
    label: str,
) -> bytes:
    return _canonical_json_bytes(
        _digest_map_document(
            value,
            exact_keys=exact_keys,
            value_fields=value_fields,
            label=label,
        )
    )


def _digest_map_document(
    value: Mapping[str, object],
    *,
    exact_keys: tuple[str, ...],
    value_fields: tuple[str, ...] | None,
    label: str,
    exact_true_fields: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _exact_keys(value, set(exact_keys), label=label)
    document: dict[str, object] = {}
    for key in exact_keys:
        raw = value[key]
        if value_fields is None:
            document[key] = _sha256(raw, label=f"{label}.{key}")
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}.{key} must be a digest object")
        _exact_keys(raw, set(value_fields), label=f"{label}.{key}")
        item: dict[str, object] = {}
        for field in value_fields:
            if field in exact_true_fields:
                if raw[field] is not True:
                    raise ValueError(f"{label}.{key}.{field} must be exact true")
                item[field] = True
            else:
                item[field] = _sha256(raw[field], label=f"{label}.{key}.{field}")
        document[key] = item
    return document


def _fold_step_digest_map_document(
    value: Mapping[str, object],
    *,
    value_fields: tuple[str, ...],
    label: str,
    exact_true_fields: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _exact_keys(value, set(_FOLD_RECEIPT_KEYS), label=label)
    return {
        fold: _digest_map_document(
            value[fold],
            exact_keys=_CHECKPOINT_RECEIPT_KEYS,
            value_fields=value_fields,
            label=f"{label}.{fold}",
            exact_true_fields=exact_true_fields,
        )
        for fold in _FOLD_RECEIPT_KEYS
    }


def _canonical_fold_step_digest_map_bytes(
    value: Mapping[str, object],
    *,
    value_fields: tuple[str, ...],
    label: str,
) -> bytes:
    return _canonical_json_bytes(
        _fold_step_digest_map_document(
            value,
            value_fields=value_fields,
            label=label,
        )
    )


def _direct_child_types(directory: str, entry_types: Mapping[str, str]) -> dict[str, str]:
    parent = PurePosixPath(directory)
    result: dict[str, str] = {}
    for path, entry_type in entry_types.items():
        if path == directory:
            continue
        candidate = PurePosixPath(path)
        if candidate.parent == parent:
            result[candidate.name] = entry_type
    return result


def _npz_schema(value: object, *, label: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array of inline tables")
    allowed_dtypes = {"|S64", "|u1", "|b1", "<u2", "<u8", "<f4", "<f8"}
    allowed_placeholders = {
        "{score_rows}",
        "{score_cases}",
        "{score_cases_plus_one}",
        "{score_selected_tokens}",
    }
    result: list[tuple[str, str, tuple[str, ...]]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"{label}[{index}] must be an inline table")
        _exact_keys(raw, {"name", "dtype", "shape"}, label=f"{label}[{index}]")
        name = raw["name"]
        dtype = raw["dtype"]
        shape = raw["shape"]
        if type(name) is not str or not name.isidentifier() or name.lower() != name:
            raise ValueError(f"{label}[{index}].name must be a lowercase identifier")
        if type(dtype) is not str or dtype not in allowed_dtypes:
            raise ValueError(f"{label}[{index}].dtype is not exact or portable")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                type(item) is not str
                or not (
                    (item.isascii() and item.isdigit() and not item.startswith("0"))
                    or item in allowed_placeholders
                )
                for item in shape
            )
        ):
            raise ValueError(f"{label}[{index}].shape is invalid")
        result.append((name, dtype, tuple(shape)))
    names = tuple(item[0] for item in result)
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate member names")
    return tuple(result)


def _relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _validate_recursive(value: object, *, label: str) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_recursive(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{label} contains an invalid key")
            _validate_recursive(item, label=f"{label}.{key}")
        return
    raise ValueError(f"{label} contains an unsupported TOML value")


def _require_fields_exact(
    table: Mapping[str, object], expected: Mapping[str, object], *, label: str
) -> None:
    observed = {field: table.get(field) for field in expected}
    _require_exact(observed, expected, label=label)


def _require_same_fields(
    child: Mapping[str, object],
    parent: Mapping[str, object],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    child_values = {field: _freeze(child.get(field)) for field in fields}
    parent_values = {field: parent.get(field) for field in fields}
    _require_exact(child_values, parent_values, label=label)


def _resolve_parent_scope(parent: NativeDiffusionV1Contract, path: str) -> object:
    current: object = parent.document
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"declared deferred parent scope does not exist: {path}")
        current = current[component]
    return current


def _validate_inheritance(
    document: Mapping[str, Any], parent_contract: NativeDiffusionV1Contract
) -> None:
    parent_binding = _table(document, "parent")
    if (
        parent_contract.config_sha256 != PARENT_CONFIG_SHA256
        or parent_contract.document.get("artifact") != PARENT_ARTIFACT
    ):
        raise ValueError("authenticated parent contract identity is invalid")
    _require_exact(
        _freeze(parent_binding["artifact"]),
        parent_contract.document["artifact"],
        label="child parent artifact assertion",
    )
    _require_exact(
        parent_binding["config_sha256"],
        parent_contract.config_sha256,
        label="child parent SHA-256 assertion",
    )

    inheritance = _table(document, "inheritance")
    expected_inheritance = {
        "scheme": _INHERITANCE_SCHEME,
        "entire_parent_document_normative": True,
        "normative_parent_tables": list(_PARENT_TABLES),
        "type_exact_relation_ids": list(_RELATION_IDS),
        "omitted_parent_execution_fields": "normative_from_exact_authenticated_parent_bytes",
        "allowed_parent_overrides": [],
        "pilot_phase_narrowing_ids": list(_PILOT_NARROWINGS),
        "inactive_deferred_parent_scopes": list(_DEFERRED_PARENT_SCOPES),
        "narrowings_are_not_parent_overrides": True,
        "deferred_twin_scope": "final_all_development_fit_and_fold4_evaluation_only",
    }
    _require_exact(
        inheritance,
        expected_inheritance,
        label="authenticated no-merge inheritance declaration",
    )
    for table_name in _PARENT_TABLES:
        if not isinstance(parent_contract.document.get(table_name), Mapping):
            raise ValueError(f"normative parent table is unavailable: {table_name}")
    for scope in _DEFERRED_PARENT_SCOPES:
        _resolve_parent_scope(parent_contract, scope)

    pilot = _table(document, "pilot")
    model = _table(document, "model")
    training = _table(document, "training")
    evaluation = _table(document, "evaluation")
    gate = _table(document, "gate")
    rng = _table(document, "rng")
    resources = _table(document, "resources")
    outputs = _table(document, "outputs")
    leakage = _table(document, "leakage")
    status = _table(document, "status")

    parent_status = parent_contract.table("status")
    _require_exact(
        {
            "before_execution": status["before_execution"],
            "evidence_invalid": status["evidence_invalid"],
            "pilot_no_go": status["pilot_no_go"],
        },
        {
            "before_execution": parent_status["before_execution"],
            "evidence_invalid": parent_status["evidence_invalid"],
            "pilot_no_go": parent_status["pilot_no_go"],
        },
        label="child-parent status relations",
    )

    parent_development = parent_contract.table("development")
    parent_training = parent_contract.table("training")
    parent_model = parent_contract.table("model")
    parent_models = parent_contract.table("models")
    r128 = parent_models.get("R128")
    if not isinstance(r128, Mapping):
        raise ValueError("authenticated parent R128 model is unavailable")
    _require_exact(
        _freeze([pilot["variant"]]),
        parent_development["pilot_variants"],
        label="parent pilot variant selection",
    )
    _require_exact(
        _freeze(pilot["outer_folds"]),
        parent_development["outer_folds"],
        label="parent development outer folds",
    )
    _require_exact(
        pilot["selection_seed"],
        parent_training["selection_seed"],
        label="parent selection seed",
    )
    _require_exact(
        _freeze(pilot["checkpoint_steps"]),
        parent_training["checkpoint_steps"],
        label="parent checkpoint grid",
    )
    _require_exact(pilot["max_steps"], parent_training["max_steps"], label="parent fixed horizon")
    _require_exact(
        pilot["checkpoint_selection_rule"],
        parent_development["pilot_checkpoint_rule"],
        label="parent pilot checkpoint rule",
    )
    _require_exact(
        pilot["corruption_replicates_per_sequence_level"],
        parent_development["pilot_corruption_replicates_per_sequence_level"],
        label="parent pilot corruption replicate count",
    )
    _require_exact(
        pilot["stops_before_sampling"],
        parent_development["pilot_stops_before_sampling"],
        label="parent pilot sampling boundary",
    )
    _require_exact(
        pilot["reuse_exact_fold_bundles_in_full_matrix"],
        parent_development["pilot_r128_bundles_reused_in_full_matrix"],
        label="parent R128 bundle reuse",
    )
    if (
        parent_contract.table("determinism")["require_exact_seed42_twin"] is not True
        or parent_contract.table("recipe_selection")["final_seed42_distinct_node_twins"] is not True
        or pilot["replicas_per_outer_fold"] != 1
        or type(pilot["replicas_per_outer_fold"]) is not int
        or pilot["exact_seed42_twins_required_for_pilot"] is not False
    ):
        raise ValueError("pilot replica narrowing or deferred final twin scope is invalid")

    common_model_fields = (
        "kind",
        "alphabet",
        "min_length",
        "max_length",
        "special_tokens",
        "activation",
        "dropout",
        "layer_norm_epsilon",
        "tie_residue_input_output_weights",
        "prediction_classes",
        "initialization",
        "property_conditioning",
        "self_conditioning",
        "geometry_conditioning",
    )
    _require_same_fields(
        model, parent_model, common_model_fields, label="parent common model fields"
    )
    _require_fields_exact(
        parent_model,
        {
            "token_embedding": "learned",
            "position_embedding": "learned_absolute_0_to_49",
            "length_embedding": "learned_8_to_50",
            "timestep_embedding": "learned_0_to_64_row_zero_reserved",
            "attention_projection_bias": True,
            "ffn_bias": True,
            "final_layer_norm": True,
            "output_bias": True,
            "classifier_free_guidance": False,
        },
        label="inherited parent embedding, bias, normalization, and CFG model fields",
    )
    _require_same_fields(
        model,
        r128,
        ("layers", "hidden_dim", "attention_heads", "ffn_dim", "expected_trainable_parameters"),
        label="parent R128 architecture",
    )
    _require_exact(pilot["output_mode"], r128["output"], label="parent R128 output mode")
    _require_same_fields(
        model,
        parent_training,
        ("residual_training_logits", "residual_training_lambda", "residual_training_temperature"),
        label="parent residual training head",
    )

    parent_diffusion = parent_contract.table("diffusion")
    _require_fields_exact(
        parent_diffusion,
        {
            "kind": "absorbing_mask_fixed_count",
            "levels": 64,
            "schedule": "cosine_alpha_bar",
            "cosine_offset": 0.008,
            "timestep_sampling": "uniform_integer_1_to_64",
            "mask_count": "ceil_length_times_mask_probability",
            "mask_position_sampling": "uniform_without_replacement",
            "prediction_target": "clean_residue",
            "loss_positions": "scheduled_masked_valid_positions_only",
            "loss_reduction": "masked_mean_per_sequence_then_batch_mean",
            "visible_context_dropout_changes_loss_mask": False,
        },
        label="inherited parent diffusion and loss reductions",
    )
    _require_exact(pilot["diffusion_levels"], parent_diffusion["levels"], label="pilot levels")
    _require_exact(evaluation["levels"], parent_diffusion["levels"], label="score levels")
    _require_exact(
        training["visible_context_dropout_changes_loss_mask"],
        parent_diffusion["visible_context_dropout_changes_loss_mask"],
        label="visible context loss-mask rule",
    )
    inherited_training_fields = (
        "batch_sequences",
        "sample_with_replacement",
        "sampling_weight_application",
        "optimizer",
        "learning_rate",
        "betas",
        "epsilon",
        "weight_decay",
        "weight_decay_includes",
        "exclude_from_weight_decay",
        "warmup_steps",
        "lr_schedule",
        "final_learning_rate",
        "gradient_clip_norm",
        "label_smoothing",
        "visible_context_dropout",
        "gradient_accumulation_steps",
        "validation_during_training",
        "training_log_interval_steps",
        "ema",
        "amp",
        "tf32",
        "torch_compile",
    )
    _require_same_fields(
        training,
        parent_training,
        inherited_training_fields,
        label="parent training recipe",
    )
    _require_fields_exact(
        parent_training,
        {
            "visible_context_dropout_rule": (
                "independently_mask_visible_valid_residues_after_scheduled_corruption"
            ),
            "visible_context_dropout_rng": (
                "numpy_pcg64_one_uniform_per_valid_position_in_ascending_position_order"
            ),
            "visible_context_dropout_seed_key": (
                "sha256_namespace_to_uint64_v1_root_seed_context_dropout_fit_id_"
                "global_draw_ordinal_sequence_id_level"
            ),
            "model_dropout_rng": (
                "torch_global_generator_reseeded_immediately_before_each_training_forward"
            ),
            "model_dropout_seed_key": (
                "sha256_namespace_to_uint64_v1_root_seed_model_dropout_fit_id_optimizer_step"
            ),
            "training_count_prior_scope": ("fit_on_exact_outer_training_projection_only"),
            "cross_entropy_label_smoothing_definition": (
                "pytorch_cross_entropy_uniform_over_20_classes"
            ),
            "precision": "float32",
        },
        label="inherited parent dropout, count-prior scope, and loss semantics",
    )

    parent_calibration = parent_contract.table("calibration")
    _require_exact(
        _freeze(evaluation["timestep_bins"]),
        tuple(parent_calibration["timestep_bins"][:4]),
        label="parent pilot timestep bins",
    )
    _require_exact(
        _freeze(evaluation["residual_lambda_grid"]),
        parent_calibration["residual_lambda_grid"],
        label="parent residual lambda grid",
    )
    _require_exact(
        _freeze(evaluation["temperature_grid"]),
        parent_calibration["temperature_grid"],
        label="parent temperature grid",
    )
    _require_exact(
        evaluation["calibration_crossfit"],
        parent_development["calibration_crossfit"],
        label="parent LOUCO scope",
    )
    _require_fields_exact(
        parent_calibration,
        {
            "count_probability_floor": 0.000000000001,
            "residual_formula": ("softmax((log(p0)+lambda_bin*residual_logits)/temperature_bin)"),
            "count_calibrated_formula": "softmax(log(p0)/temperature_bin)",
            "zero_lambda_report_only": True,
            "zero_lambda_selected_decision": "development_no_go_v1",
            "case_metric_reduction": "mean_nll_over_scheduled_masked_tokens",
            "sequence_metric_reduction": "equal_mean_over_levels_and_replicates",
            "outer_fold_metric_reduction": "equal_arithmetic_mean",
            "selection_tie_break": (
                "lower_lambda_then_temperature_closest_to_one_then_lower_temperature"
            ),
            "strongest_count_control_rule": "lower_cross_calibrated_nll_of_C0_and_C0T",
            "strongest_count_control_tie_break": "C0",
            "strongest_count_control_scope": (
                "one_global_method_reused_for_fold_timestep_bin_ece_and_bootstrap_gates"
            ),
            "ece_bins": 15,
            "ece_prediction": ("maximum_probability_with_lowest_residue_index_on_exact_tie"),
            "ece_confidence": "maximum_residue_probability",
            "ece_bin_index": "minimum_14_floor_15_times_confidence",
            "ece_bin_intervals": "left_closed_right_open_except_final_right_closed",
            "ece_case_weighting": "equal_scheduled_masked_token_weight_within_case",
            "ece_sequence_weighting": "equal_case_weight_within_sequence",
            "ece_fold_weighting": "homology_component_equal_row_weight",
            "ece_aggregate": (
                "ece_from_one_quarter_weighted_sum_of_each_folds_bin_sufficient_statistics"
            ),
            "ece_empty_bin": "zero_contribution",
        },
        label="inherited parent calibration, tie, zero-lambda, and ECE rules",
    )
    _require_fields_exact(
        parent_development,
        {
            "calibration_exclusion_unit": "union_component_id",
            "calibration_sequence_case_reduction": (
                "ascending_level_then_replicate_math_fsum_divided_by_case_count"
            ),
            "calibration_component_reduction": (
                "ascending_sequence_id_math_fsum_for_weighted_nll_numerator_and_sampling_mass"
            ),
            "calibration_total_reduction": "ascending_union_component_id_math_fsum",
            "calibration_leaveout_reduction": ("math_fsum_total_then_negative_heldout_component"),
            "no_scored_token_selects_its_calibration": True,
            "bootstrap_unit": "union_component_id_stratified_by_fold",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260905,
            "bootstrap_fold_aggregation": "equal_arithmetic_mean_over_four_outer_folds",
            "bootstrap_recalibrates_each_draw": False,
            "bootstrap_candidate_comparator_draws_shared": True,
            "bootstrap_standard_error": (
                "sample_standard_deviation_ddof1_of_bootstrap_mean_nll_without_"
                "division_by_sqrt_replicates"
            ),
            "bootstrap_rng": "numpy_pcg64dxsm_2.4.6",
            "bootstrap_component_order": "ascending_union_component_id",
            "bootstrap_confidence_interval": ("numpy_quantile_0p025_and_0p975_method_linear"),
        },
        label="inherited parent LOUCO and bootstrap reductions",
    )
    _require_fields_exact(
        parent_development,
        {
            "pilot_checkpoint_standard_error": (
                "same_best_mean_plus_best_bootstrap_sample_standard_deviation_ddof1_"
                "rule_as_full_selection"
            ),
            "pilot_gate_uses_selected_checkpoint": True,
        },
        label="inherited parent pilot checkpoint selection semantics",
    )
    _require_fields_exact(
        parent_contract.table("recipe_selection"),
        {
            "checkpoint_candidates": (250, 500, 1000, 2000, 4000),
            "rule": "one_standard_error_of_best_cross_calibrated_mean_nll",
            "standard_error_source": "stratified_union_component_bootstrap",
            "best_choice_rule": (
                "lowest_equal_fold_mean_nll_then_candidate_order_then_earliest_checkpoint"
            ),
            "standard_error_definition": (
                "sample_standard_deviation_ddof1_of_10000_best_choice_bootstrap_mean_nll_values"
            ),
            "eligibility_threshold": "best_observed_mean_nll_plus_standard_error_of_best_choice",
            "eligibility_comparator": "mean_nll_less_than_or_equal_to_threshold",
            "complexity_tie_break": "candidate_order_then_earliest_checkpoint",
        },
        label="inherited parent one-standard-error selection and tie rules",
    )
    _require_exact(
        {
            "bootstrap_unit": evaluation["bootstrap_unit"],
            "bootstrap_replicates": evaluation["bootstrap_replicates"],
            "bootstrap_seed": evaluation["bootstrap_seed"],
            "bootstrap_rng": evaluation["bootstrap_rng"],
            "bootstrap_standard_error": evaluation["bootstrap_standard_error"],
            "calibration_choices_are": evaluation["calibration_choices_are"],
        },
        {
            "bootstrap_unit": parent_development["bootstrap_unit"],
            "bootstrap_replicates": parent_development["bootstrap_replicates"],
            "bootstrap_seed": parent_development["bootstrap_seed"],
            "bootstrap_rng": parent_development["bootstrap_rng"],
            "bootstrap_standard_error": parent_development["bootstrap_standard_error"],
            "calibration_choices_are": parent_development["calibration_choices_are"],
        },
        label="child-parent score calibration and bootstrap assertions",
    )
    parent_pilot_gate = parent_contract.table("gates").get("pilot")
    if not isinstance(parent_pilot_gate, Mapping):
        raise ValueError("authenticated parent pilot gate is unavailable")
    inherited_gate_fields = (
        "candidate",
        "comparator",
        "minimum_mean_relative_nll_improvement",
        "minimum_bootstrap_lower_bound_improvement",
        "bootstrap_lower_bound_comparator",
        "maximum_fold_relative_nll_regression",
        "maximum_timestep_bin_relative_nll_regression",
        "maximum_ece",
        "maximum_ece_regression",
        "ece_scope",
        "require_all_four_outer_fits",
    )
    _require_same_fields(gate, parent_pilot_gate, inherited_gate_fields, label="parent pilot gate")

    parent_determinism = parent_contract.table("determinism")
    _require_exact(rng["derivation"], parent_determinism["rng_derivation"], label="RNG derivation")
    _require_exact(
        _freeze(rng["fit_identity_fields"]),
        parent_determinism["fit_identity_fields"],
        label="fit identity fields",
    )
    _require_same_fields(
        rng,
        parent_determinism,
        ("fit_identity_serialization", "global_draw_ordinal", "corruption_rng", "validation_rng"),
        label="parent RNG semantics",
    )
    _require_fields_exact(
        parent_determinism,
        {
            "deterministic_algorithms": True,
            "math_sdpa_only": True,
            "mha_fastpath": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "dataloader_workers": 0,
            "pytorch_allocator": "backend:native",
            "require_exact_seed42_twin": True,
            "require_distinct_node_seed42_twin": True,
            "training_twin_equality_scope": "path_free_semantic_bundle_bytes_and_modes",
            "evaluation_twin_equality_scope": "path_free_semantic_bundle_bytes_and_modes",
            "operational_telemetry_is_outside_semantic_twin_bundle": True,
        },
        label="inherited parent deterministic runtime",
    )
    _require_fields_exact(
        parent_contract.table("environment"),
        {
            "python": "3.11.14",
            "numpy": "2.4.6",
            "torch": "2.14.0",
            "torch_cuda": "13.0",
            "safetensors": "0.6.2",
            "packaging": "26.3",
            "triton": "3.8.0",
            "nvidia_cudnn_cu13": "9.24.0.43",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "compute_capability": (8, 0),
            "checkpoint_format": "safetensors",
            "allow_pickle": False,
        },
        label="inherited parent environment",
    )
    _require_fields_exact(
        parent_contract.table("telemetry"),
        {
            "required_per_worker": (
                "torch_cuda_max_memory_allocated",
                "torch_cuda_max_memory_reserved",
                "cuda_device_uuid",
                "process_pid",
                "in_allocation_nvidia_smi_process_memory_samples",
                "slurm_step_accounting",
            ),
            "allocated_memory_is_frozen_gate": True,
            "reserved_memory_is_report_only": True,
            "nvidia_smi_process_memory_is_report_only": True,
            "slurm_epilog_memory_is_report_only": True,
        },
        label="inherited parent telemetry semantics",
    )

    parent_compute = parent_contract.table("compute")
    resource_relations = {
        "account": parent_compute["account"],
        "partition": parent_compute["gpu_partition"],
        "gpu_type": parent_compute["gpu_type"],
        "gpus_per_task": parent_compute["gpus_per_task"],
        "cpus_per_task": parent_compute["cpus_per_task"],
        "memory_gib_per_task": parent_compute["memory_gib_per_task"],
        "maximum_pilot_a100_hours": parent_compute["pilot_maximum_a100_hours"],
        "maximum_peak_allocated_memory_gib": parent_compute["maximum_peak_allocated_memory_gib"],
        "bare_exclusive_allowed": parent_compute["bare_exclusive_allowed"],
        "execution_command": parent_compute["execution_command"],
    }
    _require_exact(
        {field: resources[field] for field in resource_relations},
        resource_relations,
        label="parent compute resource relations",
    )

    parent_artifacts = parent_contract.table("artifacts")
    _require_exact(
        {
            "schema_version": outputs["schema_version"],
            "canonical_json": outputs["canonical_json"],
            "reject_nonfinite_json": outputs["reject_nonfinite_json"],
            "file_mode": outputs["file_mode"],
            "directory_mode": outputs["directory_mode"],
            "manifest_published_last": outputs["manifest_published_last"],
            "semantic_manifests_path_free": outputs["semantic_manifests_path_free"],
            "independent_verifier_imports_producer": outputs[
                "independent_verifier_imports_producer"
            ],
        },
        {
            field: parent_artifacts[field]
            for field in (
                "schema_version",
                "canonical_json",
                "reject_nonfinite_json",
                "file_mode",
                "directory_mode",
                "manifest_published_last",
                "semantic_manifests_path_free",
                "independent_verifier_imports_producer",
            )
        },
        label="parent artifact canonicalization and audit isolation",
    )
    parent_leakage = parent_contract.table("leakage")
    _require_exact(
        {
            "trainer_allowed_roles": _freeze(leakage["trainer_allowed_roles"]),
            "trainer_allowed_fields": _freeze(leakage["trainer_allowed_fields"]),
            "labels_allowed": leakage["labels_allowed"],
            "provenance_allowed": leakage["provenance_allowed"],
            "study_keys_allowed": leakage["study_keys_allowed"],
            "oracle_predictions_allowed": leakage["oracle_predictions_allowed"],
            "structures_allowed": leakage["structures_allowed"],
            "post_fold4_changes_require_new_version": leakage[
                "post_fold4_changes_require_new_version"
            ],
        },
        {
            field: parent_leakage[field]
            for field in (
                "trainer_allowed_roles",
                "trainer_allowed_fields",
                "labels_allowed",
                "provenance_allowed",
                "study_keys_allowed",
                "oracle_predictions_allowed",
                "structures_allowed",
                "post_fold4_changes_require_new_version",
            )
        },
        label="parent leakage policy",
    )


def _validate_count_prior(
    document: Mapping[str, Any], parent_contract: NativeDiffusionV1Contract
) -> None:
    count_prior = _table(document, "count_prior")
    expected = {
        "name": "C0_length_relative_position_frequency",
        "input_path": "fold_contract.train_path_exact_folds/{outer_fold}/train.jsonl",
        "input_sha256": "fold_contract.train_sha256_verified_before_jsonl_parse",
        "algorithm_source_commit": "3e81822a1e2fa2c5b0bfcbd6527d1e9ea31a8474",
        "algorithm_source_path": "src/amp_challenge/generators/diffusion/evaluation.py",
        "algorithm_source_sha256": (
            "96d7e0c644242ffe9b87fb1bddc3d357a67318631a3ac01ba10ded8c36e735c1"
        ),
        "fit_rows": "exact_outer_training_projection_sorted_by_sequence_id",
        "effective_count_scale": "number_of_fit_rows",
        "row_weight": "1/(fit_homology_component_count*fit_component_size)",
        "residue_contribution": (
            "effective_count_scale_times_row_weight_divided_by_sequence_length"
        ),
        "unigram_pseudocount_per_residue": 0.5,
        "unigram_normalization": "normalize_20_counts_after_pseudocount",
        "length_edges": [8, 15, 20, 25, 33, 51],
        "length_bin": "searchsorted_right_minus_one",
        "relative_position_bins": 10,
        "relative_position_bin": ("min_9_floor_10_times_zero_based_position_divided_by_length"),
        "relative_position_prior_mass": 20.0,
        "relative_position_prior": "prior_mass_times_normalized_unigram_probability",
        "relative_position_normalization": ("normalize_each_length_position_cell_over_20_residues"),
        "log_floor": 0.000000000001,
        "log_floor_rule": ("elementwise_max_with_floor_then_renormalize_over_20_residues"),
        "relative_position_probability_output": (
            "post_log_floor_rule_clipped_and_renormalized_probability_float64"
        ),
        "log_relative_position_probability": (
            "numpy_log_of_post_floor_renormalized_relative_position_probability_float64"
        ),
        "production_order": (
            "before_model_initialization_optimizer_creation_and_first_minibatch_draw"
        ),
        "trainer_output_path": "count_prior.npz",
        "evaluator_reuse": ("read_exact_trainer_bundle_bytes_verified_by_physical_sha256"),
        "recomputation_during_evaluation_allowed": False,
        "training_bridge": {
            "source_member": "log_relative_position_probability",
            "source_dtype": "<f8",
            "source_shape": [5, 10, 20],
            "source_order": ("C_contiguous_length_bin_relative_position_bin_residue_index"),
            "load_allow_pickle": False,
            "validation": (
                "exact_dtype_shape_order_finite_and_sha256_bound_to_checkpoint_before_conversion"
            ),
            "numpy_to_torch": ("torch.from_numpy_exact_loaded_array_without_intermediate_cast"),
            "torch_conversion": (
                "tensor.to(device=training_device,dtype=torch.float32,non_blocking=false,"
                "copy=true,memory_format=torch.contiguous_format)"
            ),
            "float64_to_float32_rounding": "IEEE_754_roundTiesToEven_once_per_fit",
            "conversion_count_per_fit": 1,
            "conversion_timing": (
                "after_count_prior_seal_before_model_initialization_optimizer_creation_"
                "and_first_forward"
            ),
            "gather_index_order": [
                "length_bin",
                "relative_position_bin",
                "residue_index",
            ],
            "gathered_count_logit_dtype": "torch.float32",
            "residual_logit_dtype": "torch.float32",
            "addition_dtype": "torch.float32_no_autocast",
            "loss_dtype": "torch.float32_no_autocast",
        },
    }
    _require_exact(count_prior, expected, label="fold-local count-prior contract")
    parent_count_prior = parent_contract.table("count_prior")
    _require_exact(
        {field: _freeze(count_prior[field]) for field in parent_count_prior},
        parent_count_prior,
        label="parent count-prior algorithm assertion",
    )


def _validate_fixed_tables(document: Mapping[str, Any]) -> None:
    parent = _table(document, "parent")
    _require_exact(
        parent,
        {
            "artifact": "native_categorical_diffusion_unconditional_v1",
            "config_path": "configs/diffusion/unconditional_v1.toml",
            "config_sha256": PARENT_CONFIG_SHA256,
            "status_before_pilot": "predeclared_not_run",
        },
        label="pilot parent protocol binding",
    )

    status = _table(document, "status")
    expected_status = {
        "before_execution": "predeclared_not_run",
        "fold_complete": "pilot_fold_complete_pending_selection",
        "evidence_invalid": "invalid_run",
        "pilot_no_go": "development_no_go_v1_pilot",
        "pilot_continue": "development_continue_v1_full_matrix_authorized",
        "missing_or_incomplete_fold": "invalid_run",
        "evidence_invalid_is_scientific_no_go": False,
        "scientific_no_go_requires_valid_complete_pilot": True,
        "continuation_grants_candidate_status": False,
        "continuation_grants_proposal_or_library_rights": False,
    }
    _require_exact(status, expected_status, label="pilot status vocabulary")

    model = _table(document, "model")
    expected_model = {
        "kind": "bidirectional_pre_layer_norm_transformer",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "min_length": 8,
        "max_length": 50,
        "special_tokens": ["PAD", "MASK"],
        "layers": 2,
        "hidden_dim": 128,
        "attention_heads": 4,
        "ffn_dim": 384,
        "expected_trainable_parameters": 354068,
        "dropout": 0.20,
        "layer_norm_epsilon": 0.00001,
        "activation": "gelu",
        "tie_residue_input_output_weights": True,
        "prediction_classes": 20,
        "initialization": "normal_std_0.02_bias_zero_norm_scale_one",
        "residual_training_logits": ("log(clipped_renormalized_fold_local_C0)+raw_residual_logits"),
        "residual_training_lambda": 1.0,
        "residual_training_temperature": 1.0,
        "property_conditioning": False,
        "self_conditioning": False,
        "geometry_conditioning": False,
    }
    _require_exact(model, expected_model, label="pilot R128 model binding")

    dtypes = _table(document, "dtypes")
    expected_dtypes = {
        "model_parameter": "torch.float32",
        "model_activation": "torch.float32",
        "model_logit": "torch.float32",
        "loss_and_gradient": "torch.float32",
        "encoded_token": "torch.int64",
        "attention_mask": "torch.bool",
        "sequence_length": "torch.int64",
        "timestep": "torch.int64",
        "scheduled_mask": "torch.bool",
        "target_token": "torch.int64",
        "numpy_sampling_probability": "<f8",
        "numpy_metric_accumulator": "<f8",
        "checkpoint_tensor": "F32",
        "score_count_log_probability": "<f8",
        "score_residual_logit_storage": "<f4",
        "score_residual_logit_calibration_compute": "<f8",
        "score_residual_logit_cast_rule": (
            "lossless_float32_to_float64_before_softmax_calibration_and_metric_reductions"
        ),
        "count_prior_archive_log_probability": "<f8",
        "count_prior_bridge_source_tensor": "torch.float64",
        "count_prior_training_tensor": "torch.float32",
        "count_prior_gathered_logit": "torch.float32",
        "count_prior_residual_addition": "torch.float32",
        "allow_implicit_dtype_coercion": False,
        "allow_nonfinite": False,
    }
    _require_exact(dtypes, expected_dtypes, label="pilot tensor dtype contract")

    gate = _table(document, "gate")
    expected_gate = {
        "candidate": "R128",
        "comparator": "strongest_of_C0_and_C0T",
        "minimum_mean_relative_nll_improvement": 0.02,
        "minimum_bootstrap_lower_bound_improvement": 0.0,
        "bootstrap_lower_bound_comparator": "strictly_greater_than",
        "maximum_fold_relative_nll_regression": 0.01,
        "maximum_timestep_bin_relative_nll_regression": 0.01,
        "maximum_ece": 0.10,
        "maximum_ece_regression": 0.02,
        "ece_scope": "every_outer_fold_and_equal_fold_aggregate",
        "require_all_four_outer_fits": True,
        "on_failure": "development_no_go_v1_pilot",
        "on_pass": "development_continue_v1_full_matrix_authorized",
    }
    _require_exact(gate, expected_gate, label="pilot continuation gate")


def _parse_folds(document: Mapping[str, Any]) -> tuple[PilotFoldContract, ...]:
    folds_table = _table(document, "folds")
    _exact_keys(folds_table, {str(fold) for fold in _FOLDS}, label="folds")
    expected = {
        0: (
            (1, 2, 3),
            "03dcf8b53e266ee6142e1892dc769f40c858328ae63c98fd5382d970f4db6787",
            609,
            340,
            170,
            "b810118d92fed2ea11aae84fa405ad042037708328e8c77497b2f703679a59ae",
            305,
            131,
            51,
            272385,
            (10338, 45080, 92494, 124473),
            "5a3461fce78e34fa0a08b739e7f32d0a830c8cafa775e3d22ab4911fb902b4a8",
        ),
        1: (
            (0, 2, 3),
            "50540f534dfffab9993aabfeb46c23b8c3d6654b4e970c25a77a8b908b88856b",
            705,
            333,
            165,
            "c73890843038b34a152449ae4086bc01244ee1c3e6e53b1a9e84feb0238a5d22",
            209,
            138,
            56,
            163489,
            (6457, 27123, 55465, 74444),
            "8d7eeae81688d7a1c118e981a360ac8a2ccdcf3f43f2bfb7d6b71c18d7a87a69",
        ),
        2: (
            (0, 1, 3),
            "e8e31f98222ffc7982fbdeebc8764abf8d8ca5781db43c0d1d5db70bcbb0a74b",
            715,
            384,
            164,
            "551f0a2e6bb9aa414a9b07a2e46590f0533da39842e2f13e0377e380da4780fb",
            199,
            87,
            57,
            134208,
            (5551, 22377, 45458, 60822),
            "fedd90011393ad3b3a5122c2adc0b75ba3808de3fe7b122b6b97b3ba3f947627",
        ),
        3: (
            (0, 1, 2),
            "6198eff45d57559dedec5c9acb138cbb9666cd26216cb4710da28d3e81eb30a6",
            713,
            356,
            164,
            "f5665608235132b45a6ded5850751653d42a1667a82078ce02ba722938480d76",
            201,
            115,
            57,
            144102,
            (5808, 23981, 48847, 65466),
            "dcf45e9941c5836382c22a0f8ddcfe2ecfa5d18cbae48ee7aa66c69434982a04",
        ),
    }
    result: list[PilotFoldContract] = []
    for fold in _FOLDS:
        raw = _table(folds_table, str(fold))
        _exact_keys(raw, _FOLD_KEYS, label=f"folds.{fold}")
        fit_folds = _tuple_of(raw.get("fit_folds"), int, label=f"folds.{fold}.fit_folds")
        train_path = _relative_path(raw.get("train_path"), label=f"folds.{fold}.train_path")
        score_path = _relative_path(raw.get("score_path"), label=f"folds.{fold}.score_path")
        train_sha256 = _sha256(raw.get("train_sha256"), label=f"folds.{fold}.train_sha256")
        score_sha256 = _sha256(raw.get("score_sha256"), label=f"folds.{fold}.score_sha256")
        score_bins = _tuple_of(
            raw.get("score_selected_tokens_by_timestep_bin"),
            int,
            label=f"folds.{fold}.score_selected_tokens_by_timestep_bin",
        )
        fit_identity_sha256 = _sha256(
            raw.get("fit_identity_sha256"), label=f"folds.{fold}.fit_identity_sha256"
        )
        values = tuple(
            _integer(raw.get(field), label=f"folds.{fold}.{field}", minimum=1)
            for field in (
                "train_rows",
                "train_homology_components",
                "train_union_components",
                "score_rows",
                "score_homology_components",
                "score_union_components",
                "score_cases",
                "score_selected_tokens",
            )
        )
        item = PilotFoldContract(
            outer_fold=_integer(raw.get("outer_fold"), label=f"folds.{fold}.outer_fold"),
            fit_folds=fit_folds,
            train_path=train_path,
            train_sha256=train_sha256,
            train_rows=values[0],
            train_homology_components=values[1],
            train_union_components=values[2],
            score_path=score_path,
            score_sha256=score_sha256,
            score_rows=values[3],
            score_homology_components=values[4],
            score_union_components=values[5],
            score_cases=values[6],
            score_selected_tokens=values[7],
            score_selected_tokens_by_timestep_bin=score_bins,
            fit_identity_sha256=fit_identity_sha256,
        )
        pin = expected[fold]
        observed = (
            item.fit_folds,
            item.train_sha256,
            item.train_rows,
            item.train_homology_components,
            item.train_union_components,
            item.score_sha256,
            item.score_rows,
            item.score_homology_components,
            item.score_union_components,
            item.score_selected_tokens,
            item.score_selected_tokens_by_timestep_bin,
            item.fit_identity_sha256,
        )
        if (
            item.outer_fold != fold
            or item.train_path != f"folds/{fold}/train.jsonl"
            or item.score_path != f"folds/{fold}/score.jsonl"
            or observed != pin
            or item.score_cases != item.score_rows * 64
            or len(item.score_selected_tokens_by_timestep_bin) != 4
            or sum(item.score_selected_tokens_by_timestep_bin) != item.score_selected_tokens
        ):
            raise ValueError(f"fold {fold} projection pin or census is invalid")
        result.append(item)
    return tuple(result)


def _validate_projection(document: Mapping[str, Any]) -> None:
    projection = _table(document, "projection")
    digest_fields = {
        "bundle_top_manifest_sha256": PROJECTION_TOP_SHA256,
        "bundle_tree_sha256": PROJECTION_TREE_SHA256,
        "manifest_sha256": PROJECTION_MANIFEST_SHA256,
        "summary_sha256": PROJECTION_SUMMARY_SHA256,
        "independent_receipt_sha256": PROJECTION_RECEIPT_SHA256,
        "operational_receipt_sha256": PROJECTION_OPERATIONAL_RECEIPT_SHA256,
    }
    for field, expected in digest_fields.items():
        if _sha256(projection.get(field), label=f"projection.{field}") != expected:
            raise ValueError("accepted projection evidence SHA-256 changed")
    expected_execution = {
        "artifact": PROJECTION_ARTIFACT,
        "producer_job_id": 224105,
        "audit_job_id": 224106,
        "execution_git_commit": PROJECTION_GIT_COMMIT,
        "scratch_root_env": "AMP_CHALLENGE_SCRATCH",
        "canonical_twin_slot": 0,
        "verified_twin_slots": [0, 1],
        "bundle_file_mode": "0444",
        "bundle_directory_mode": "0555",
        "fold4_sequence_rows": 0,
    }
    _require_exact(
        {field: projection.get(field) for field in expected_execution},
        expected_execution,
        label="accepted projection execution binding",
    )
    expected_paths = {
        "canonical_bundle_relative_path": (
            "diffusion/native-categorical-unconditional-v1/development-projections/224105/0"
        ),
        "independent_receipt_relative_path": (
            "diffusion/native-categorical-unconditional-v1/"
            "development-projection-audits/224105/224106/independent-verification.json"
        ),
        "operational_receipt_relative_path": (
            "diffusion/native-categorical-unconditional-v1/"
            "development-projection-audits/224105/224106/operational-receipt.json"
        ),
    }
    for field, expected in expected_paths.items():
        if _relative_path(projection.get(field), label=f"projection.{field}") != expected:
            raise ValueError("accepted projection relative path is invalid")


def _validate_pilot(document: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    pilot = _table(document, "pilot")
    checkpoints = _tuple_of(pilot.get("checkpoint_steps"), int, label="pilot checkpoints")
    expected = {
        "variant": "R128",
        "output_mode": "residual_over_fold_local_length_relative_position_log_probability",
        "selection_seed": 42,
        "outer_folds": [0, 1, 2, 3],
        "fit_count": 4,
        "replicas_per_outer_fold": 1,
        "corruption_replicates_per_sequence_level": 1,
        "diffusion_levels": 64,
        "checkpoint_steps": [250, 500, 1000, 2000, 4000],
        "max_steps": 4000,
        "checkpoint_selection_rule": (
            "earliest_checkpoint_within_one_bootstrap_standard_error_of_best_"
            "R128_cross_calibrated_nll"
        ),
        "all_four_outer_fits_required": True,
        "reuse_exact_fold_bundles_in_full_matrix": True,
        "exact_seed42_twins_required_for_pilot": False,
        "exact_seed42_twins_scope": "final_all_development_fit_and_fold4_evaluation_only",
        "stops_before_sampling": True,
        "proposal_sampling_enabled": False,
        "expected_proposal_count": 0,
    }
    _require_exact(pilot, expected, label="four-fold R128 seed-42 pilot scope")
    if checkpoints != _CHECKPOINTS:  # pragma: no cover - implied by exact assertion
        raise RuntimeError("pilot checkpoint parser invariant failed")
    return 42, checkpoints


def _validate_training_and_evaluation(document: Mapping[str, Any]) -> None:
    training = _table(document, "training")
    expected_training = {
        "batch_sequences": 128,
        "sample_with_replacement": True,
        "sampling_weight_application": "weighted_draw_only",
        "optimizer": "adamw_unfused",
        "learning_rate": 0.0002,
        "betas": [0.90, 0.95],
        "epsilon": 0.00000001,
        "weight_decay": 0.05,
        "adamw_fused": False,
        "adamw_foreach": False,
        "adamw_amsgrad": False,
        "adamw_capturable": False,
        "adamw_differentiable": False,
        "adamw_maximize": False,
        "weight_decay_includes": ["matrices", "all_embeddings", "tied_residue_embedding"],
        "exclude_from_weight_decay": ["bias", "normalization"],
        "parameter_group_order": ["ascending_name_decay", "ascending_name_zero_decay"],
        "warmup_steps": 200,
        "lr_schedule": "linear_warmup_cosine_decay",
        "final_learning_rate": 0.00002,
        "learning_rate_step_index": "one_indexed_optimizer_step_s_in_1_through_4000",
        "learning_rate_equation": (
            "s_le_200:0.0002*s/200;else:0.00002+(0.0002-0.00002)*0.5*(1+cos(pi*(s-200)/(4000-200)))"
        ),
        "learning_rate_assignment": (
            "assign_all_parameter_group_lr_immediately_before_zero_grad_and_forward_for_step_s"
        ),
        "schedule_digest_domain": (
            "utf8_amp-native-diffusion-learning-rate-schedule-v1_then_terminal_nul"
        ),
        "schedule_digest_step_encoding": "uint64_big_endian_for_steps_1_through_4000",
        "schedule_digest_rate_encoding": (
            "python_float_hex_ascii_framed_by_uint64_big_endian_length"
        ),
        "gradient_clip_norm": 1.0,
        "gradient_clip_operation": (
            "torch.nn.utils.clip_grad_norm_parameter_iteration_order_max_norm_1_"
            "error_if_nonfinite_true_foreach_false_after_backward_before_step"
        ),
        "label_smoothing": 0.05,
        "visible_context_dropout": 0.15,
        "visible_context_dropout_changes_loss_mask": False,
        "gradient_accumulation_steps": 1,
        "zero_grad_set_to_none": True,
        "optimizer_step_order": [
            "assign_learning_rate",
            "zero_grad_set_to_none",
            "forward_float32",
            "loss_float32",
            "backward",
            "clip_gradient_norm",
            "adamw_step",
            "seal_checkpoint_if_scheduled",
        ],
        "validation_during_training": False,
        "training_log_interval_steps": 50,
        "ema": False,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "resume_from_checkpoint_allowed": False,
        "partial_or_unsealed_checkpoint_promotion_allowed": False,
    }
    _require_exact(
        training,
        expected_training,
        label="pilot training recipe differs from the parent protocol",
    )

    evaluation = _table(document, "evaluation")
    expected_evaluation = {
        "evaluation_seed": 20260905,
        "batch_sequences": 256,
        "levels": 64,
        "replicates_per_sequence_level": 1,
        "timestep_bins": ["1_to_16", "17_to_32", "33_to_48", "49_to_64"],
        "calibration_crossfit": "leave_one_union_component_out_within_each_outer_score_fold",
        "residual_lambda_grid": [0.125, 0.25, 0.5, 0.75, 1.0],
        "temperature_grid": [1.0, 1.25, 1.5, 2.0, 3.0],
        "primary_metric": "component_balanced_masked_token_nll_nats",
        "outer_fold_aggregation": "equal_arithmetic_mean",
        "bootstrap_unit": "union_component_id_stratified_by_fold",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260905,
        "bootstrap_rng": "numpy_pcg64dxsm_2.4.6",
        "bootstrap_standard_error": (
            "sample_standard_deviation_ddof1_of_bootstrap_mean_nll_without_"
            "division_by_sqrt_replicates"
        ),
        "calibration_choices_are": "score_only_not_deployed_or_averaged",
        "score_rows_may_select_checkpoint": True,
        "score_rows_may_change_training": False,
    }
    _require_exact(
        evaluation,
        expected_evaluation,
        label="pilot evaluation and calibration contract",
    )


def _validate_rng_and_checkpoints(document: Mapping[str, Any]) -> tuple[str, ...]:
    rng = _table(document, "rng")
    expected_rng = {
        "derivation": "sha256_namespace_to_uint64_v1",
        "training_root_seed": 42,
        "evaluation_root_seed": 20260905,
        "bootstrap_root_seed": 20260905,
        "training_namespaces": [
            "initialization",
            "minibatch",
            "timestep",
            "corruption",
            "context_dropout",
            "model_dropout",
        ],
        "initialization_key": ["fit_identity_sha256", "model"],
        "minibatch_key": ["fit_identity_sha256", "global_draw_ordinal"],
        "timestep_key": [
            "fit_identity_sha256",
            "global_draw_ordinal",
            "rejection_counter",
        ],
        "corruption_key": [
            "fit_identity_sha256",
            "global_draw_ordinal",
            "sequence_id",
            "level",
        ],
        "context_dropout_key": [
            "fit_identity_sha256",
            "global_draw_ordinal",
            "sequence_id",
            "level",
        ],
        "model_dropout_key": ["fit_identity_sha256", "optimizer_step"],
        "validation_namespace": "validation",
        "validation_key": [
            "parent_contract_sha256",
            "sequence_id",
            "level",
            "replicate",
        ],
        "bootstrap_namespace": "bootstrap",
        "bootstrap_key": ["parent_contract_sha256", "replicate", "outer_fold"],
        "fit_identity_fields": [
            "parent_contract_sha256",
            "variant",
            "output_mode",
            "seed",
            "fit_folds",
            "fit_projection_sha256",
        ],
        "fit_identity_serialization": (
            "utf8_canonical_json_sorted_keys_compact_separators_allow_nan_false_single_lf"
        ),
        "global_draw_ordinal": (
            "zero_based_optimizer_step_minus_one_times_batch_sequences_plus_"
            "zero_based_batch_row_index"
        ),
        "minibatch_rng": "sha256_counter_uniform_weighted_draw_v1",
        "timestep_rng": "sha256_rejection_bounded_integer_v1",
        "corruption_rng": "numpy_pcg64_2.4.6",
        "context_dropout_rng": (
            "numpy_pcg64_one_uniform_per_valid_position_in_ascending_position_order"
        ),
        "model_dropout_rng": (
            "torch_global_generator_reseeded_immediately_before_each_training_forward"
        ),
        "validation_rng": "sha256_counter_by_contract_sequence_timestep_replicate_v1",
        "checkpoint_restores_mutable_rng_state": False,
        "seed_domain_utf8_without_terminal_nul": (
            "amp-challenge/native-categorical-diffusion/stateless-seed/v1"
        ),
        "seed_domain_terminal_nul": True,
        "seed_hash": "sha256",
        "seed_framing_length": "uint64_big_endian_byte_length",
        "root_seed_encoding": "tag_r_then_uint64_big_endian",
        "namespace_encoding": "tag_n_then_utf8",
        "string_part_encoding": "tag_s_then_utf8",
        "integer_part_encoding": "tag_i_then_canonical_base10_ascii",
        "seed_output": "first_8_sha256_digest_bytes_as_unsigned_uint64_big_endian",
        "uniform_conversion": ("binary64_(uint64_right_shift_11)/2**53_in_half_open_0_1"),
        "minibatch_probability_normalization": (
            "python_math_fsum_then_ascending_sequence_id_float64_division"
        ),
        "minibatch_cumulative_sum": (
            "numpy_cumsum_float64_ascending_sequence_id_then_force_last_to_1"
        ),
        "minibatch_search": ("numpy_searchsorted_side_right_then_clamp_to_last_index"),
        "bounded_integer_rejection": (
            "accept_uint64_below_2**64-(2**64_mod_upper_bound)_then_modulo;"
            "append_zero_based_retry_as_final_integer_part"
        ),
        "corruption_consumption": (
            "for_each_row_construct_numpy_Generator_PCG64_row_seed_then_choice_"
            "ascending_valid_positions_size_exact_mask_count_replace_false"
        ),
        "context_dropout_consumption": (
            "one_PCG64_float64_uniform_for_every_valid_position_in_ascending_position_"
            "order;mask_only_scheduled_visible_positions_with_u_lt_0.15;scheduled_loss_"
            "mask_unchanged"
        ),
        "model_dropout_consumption": (
            "reseed_torch_global_generator_from_model_dropout_key_immediately_before_"
            "each_training_forward"
        ),
        "validation_row_seed": (
            "namespaced_seed(evaluation_root_seed,validation,parent_contract_sha256,"
            "sequence_id,level,replicate)"
        ),
        "case_id_domain_utf8_without_terminal_nul": (
            "amp-challenge/native-categorical-diffusion/validation-case/v1"
        ),
        "case_id_domain_terminal_nul": True,
        "case_id_fields": [
            "parent_contract_sha256_ascii",
            "sequence_id_ascii",
            "level_uint16_big_endian",
            "replicate_uint16_big_endian",
            "row_seed_uint64_big_endian",
        ],
        "case_id_framing": "each_field_prefixed_by_uint64_big_endian_byte_length",
        "case_id_output": "lowercase_sha256_hex",
    }
    _require_exact(rng, expected_rng, label="pilot RNG implementation contract")

    checkpoints = _table(document, "checkpoints")
    paths = _tuple_of(checkpoints.get("relative_paths"), str, label="checkpoint paths")
    expected_checkpoints = {
        "steps": list(_CHECKPOINTS),
        "relative_paths": list(_CHECKPOINT_PATHS),
        "metadata_relative_paths": list(_CHECKPOINT_METADATA_PATHS),
        "format": "safetensors",
        "tensor_dtype": "F32",
        "save_moment": "after_completed_optimizer_step_before_next_minibatch_draw",
        "state_scope": "model_state_dict_only",
        "tensor_name_order": "ascending_lexicographic",
        "optimizer_state_serialized": False,
        "rng_state_serialized": False,
        "pickle_allowed": False,
        "evaluate_in_ascending_step_order": True,
        "all_checkpoints_sealed_before_score_open": True,
        "producer_gpu_reinference": (
            "reload_each_sealed_checkpoint_and_byte_compare_archived_float32_residual_logits"
        ),
        "producer_gpu_reinference_comparison": (
            "rtol_0_atol_0_and_identical_little_endian_float32_bytes"
        ),
        "producer_gpu_reinference_timing": (
            "after_score_logits_archived_before_producer_A100_allocation_ends"
        ),
    }
    _require_exact(
        checkpoints,
        expected_checkpoints,
        label="pilot checkpoint inventory or timing",
    )
    for path in (*paths, *_CHECKPOINT_METADATA_PATHS):
        _relative_path(path, label="checkpoint path")
    return paths


def _validate_checkpoint_metadata(
    document: Mapping[str, Any],
) -> tuple[CheckpointTensorContract, ...]:
    metadata = _table(document, "checkpoint_metadata")
    expected_fields = [
        "schema_version",
        "artifact",
        "child_contract_sha256",
        "parent_contract_sha256",
        "fit_identity_sha256",
        "outer_fold",
        "checkpoint_step",
        "variant",
        "output_mode",
        "model_config",
        "count_prior_file_sha256",
        "optimizer_step_completed",
        "checkpoint_file",
        "checkpoint_file_sha256",
        "checkpoint_logical_state_sha256",
        "tensors",
    ]
    expected_header = {
        "schema_version": 1,
        "fields": expected_fields,
        "checkpoint_step_key_order": list(_CHECKPOINTS),
        "optimizer_step_completed": ("must_equal_checkpoint_step_after_optimizer_step_returns"),
        "count_prior_binding": "physical_sha256_of_sealed_trainer_count_prior_npz",
        "model_binding_fields": ["variant", "output_mode", "model_config"],
        "model_config_fields": list(_MODEL_CONFIG_FIELDS),
        "model_config_binding": (
            "recursive_type_exact_canonical_json_object_equal_to_child_model_table"
        ),
        "bindings": [
            {"field": "artifact", "source": "child.artifact"},
            {"field": "child_contract_sha256", "source": "child.CONFIG_SHA256"},
            {"field": "parent_contract_sha256", "source": "parent.CONFIG_SHA256"},
            {
                "field": "fit_identity_sha256",
                "source": "folds.{outer_fold}.fit_identity_sha256",
            },
            {"field": "outer_fold", "source": "assigned_worker_outer_fold_exact_integer"},
            {
                "field": "checkpoint_step",
                "source": "checkpoints.steps_and_relative_path_step",
            },
            {"field": "variant", "source": "pilot.variant"},
            {"field": "output_mode", "source": "pilot.output_mode"},
            {"field": "model_config", "source": "child.model_recursive_type_exact"},
            {
                "field": "count_prior_file_sha256",
                "source": "sha256_of_sealed_trainer_count_prior.npz",
            },
            {
                "field": "optimizer_step_completed",
                "source": "checkpoint_step_after_optimizer_step_returns",
            },
            {
                "field": "checkpoint_file",
                "source": "checkpoints.relative_paths_for_checkpoint_step",
            },
            {
                "field": "checkpoint_file_sha256",
                "source": "sha256_of_complete_safetensors_bytes",
            },
            {
                "field": "checkpoint_logical_state_sha256",
                "source": "checkpoint_metadata.logical_hash",
            },
            {
                "field": "tensors",
                "source": "checkpoint_metadata.tensors_exact_order",
            },
        ],
        "physical_hash": "lowercase_sha256_of_complete_safetensors_file_bytes",
        "logical_hash": (
            "canonical_model_logical_hash_over_parent_R128_config_and_sorted_logical_tensor_state"
        ),
        "tensor_order": "ascending_lexicographic_name",
        "tensor_dtype": "F32",
        "tensor_count": 31,
    }
    _require_exact(
        {field: metadata.get(field) for field in expected_header},
        expected_header,
        label="checkpoint metadata header",
    )
    raw_tensors = metadata.get("tensors")
    if not isinstance(raw_tensors, list) or len(raw_tensors) != 31:
        raise ValueError("checkpoint tensor schema must contain exactly 31 tensors")
    observed: list[tuple[str, str, tuple[int, ...]]] = []
    contracts: list[CheckpointTensorContract] = []
    for index, raw in enumerate(raw_tensors):
        if not isinstance(raw, dict):
            raise ValueError(f"checkpoint tensor {index} must be an inline table")
        _exact_keys(raw, {"name", "dtype", "shape"}, label=f"checkpoint tensor {index}")
        name = raw.get("name")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        if (
            type(name) is not str
            or not name
            or name.startswith(".")
            or name.endswith(".")
            or ".." in name
            or any(not (character.isalnum() or character in "._") for character in name)
        ):
            raise ValueError(f"checkpoint tensor {index} has an invalid name")
        if dtype != "F32" or type(dtype) is not str:
            raise ValueError(f"checkpoint tensor {index} must have exact F32 dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(f"checkpoint tensor {index} has an invalid shape")
        item = (name, dtype, tuple(shape))
        observed.append(item)
        contracts.append(CheckpointTensorContract(name=name, dtype=dtype, shape=tuple(shape)))
    if tuple(observed) != _R128_TENSOR_SCHEMA:
        raise ValueError("checkpoint tensor schema differs from exact parent R128 state")
    if tuple(item[0] for item in observed) != tuple(sorted(item[0] for item in observed)):
        raise ValueError("checkpoint tensor names are not ascending lexicographic")
    if sum(math.prod(shape) for _, _, shape in observed) != 354068:
        raise ValueError("checkpoint tensor schema scalar count is not 354068")
    if any(name == "residue_output_weight" for name, _, _ in observed):
        raise ValueError("tied residue output weight must be absent from checkpoint state")
    return tuple(contracts)


def _validate_barrier(document: Mapping[str, Any]) -> None:
    barrier = _table(document, "barrier")
    expected = {
        "execution_topology": ("single_allocation_one_srun_four_gpu_tasks_on_four_distinct_nodes"),
        "training_stage_receives_score_paths": False,
        "trainer_input_staging": "node_local_train_only_copy_verified_by_sha256",
        "trainer_environment_exposes_projection_root": False,
        "readiness_condition": (
            "trainer_bundle_manifest_and_all_five_checkpoints_sealed_and_hashes_receipted"
        ),
        "readiness_receipt_relative_path": "control/checkpoint-ready/{outer_fold}.json",
        "readiness_receipt_fields": [
            "schema_version",
            "artifact",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "outer_fold",
            "fit_identity_sha256",
            "trainer_bundle_sha256",
            "count_prior_file_sha256",
            "checkpoint_digest_by_step",
            "checkpoint_digest_map_sha256",
            "node_name",
            "device_uuid",
        ],
        "checkpoint_digest_key_order": list(_CHECKPOINT_RECEIPT_KEYS),
        "checkpoint_digest_value_fields": [
            "checkpoint_file_sha256",
            "checkpoint_logical_state_sha256",
            "checkpoint_metadata_sha256",
        ],
        "readiness_checkpoint_digest_map_schema": {
            "exact_keys": list(_CHECKPOINT_RECEIPT_KEYS),
            "additional_keys": False,
            "value_type": "object",
            "value_fields": list(_DIGEST_VALUE_FIELDS),
            "digest_fields": list(_DIGEST_VALUE_FIELDS),
            "digest_type": "lowercase_ascii_sha256_hex_64",
            "value_additional_fields": False,
        },
        "required_readiness_receipts": 4,
        "coordinator_verifies_distinct_outer_folds": True,
        "coordinator_verifies_all_checkpoint_hashes": True,
        "score_release_condition": ("all_four_valid_readiness_receipts_observed_and_rechecked"),
        "score_release_relative_path": "control/score-release.json",
        "score_release_fields": [
            "schema_version",
            "artifact",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "readiness_receipt_sha256_by_fold",
            "readiness_receipt_digest_map_sha256",
        ],
        "release_digest_key_order": list(_FOLD_RECEIPT_KEYS),
        "release_readiness_digest_map_schema": {
            "exact_keys": list(_FOLD_RECEIPT_KEYS),
            "additional_keys": False,
            "value_type": "lowercase_ascii_sha256_hex_64",
        },
        "digest_map_serialization": (
            "utf8_canonical_json_sorted_keys_compact_separators_allow_nan_false_single_lf"
        ),
        "digest_map_sha256_rule": "lowercase_sha256_of_complete_canonical_digest_map_bytes",
        "score_stage_inputs_materialized_after_release": True,
        "score_stage_opens_only_own_outer_fold": True,
        "score_stage_starts_after_release": True,
        "barrier_failure_status": "invalid_run",
    }
    _require_exact(barrier, expected, label="pilot four-fit training-to-scoring barrier")
    _relative_path(
        barrier["readiness_receipt_relative_path"], label="readiness receipt path template"
    )
    _relative_path(barrier["score_release_relative_path"], label="score release path")


def _validate_resources(document: Mapping[str, Any]) -> float:
    resources = _table(document, "resources")
    maximum_hours = _number(
        resources.get("maximum_pilot_a100_hours"), label="maximum pilot A100 hours"
    )
    expected = {
        "account": "bio",
        "partition": "gpumid",
        "gpu_type": "A100_80GB",
        "allocation_nodes": 4,
        "worker_tasks": 4,
        "maximum_concurrent_fit_tasks": 4,
        "nodes_per_fit": 1,
        "tasks_per_node": 1,
        "gpus_per_task": 1,
        "cpus_per_task": 8,
        "memory_gib_per_task": 32,
        "wall_minutes_per_fit": 60,
        "maximum_pilot_a100_hours": 4.0,
        "maximum_peak_allocated_memory_gib": 16.0,
        "maximum_total_allocated_gpu_seconds": 14400,
        "maximum_peak_allocated_memory_bytes": 17179869184,
        "bare_exclusive_allowed": False,
        "execution_command": "uv run --locked --no-sync",
        "run_subdir": "diffusion/native-categorical-unconditional-v1/pilot-executions",
        "require_clean_synchronized_commit": True,
    }
    _require_exact(resources, expected, label="pilot resource envelope")
    if (
        resources["worker_tasks"]
        * resources["gpus_per_task"]
        * resources["wall_minutes_per_fit"]
        * 60
        != resources["maximum_total_allocated_gpu_seconds"]
        or resources["maximum_total_allocated_gpu_seconds"] / 3600 != maximum_hours
        or resources["maximum_peak_allocated_memory_gib"] * 1024**3
        != resources["maximum_peak_allocated_memory_bytes"]
    ):
        raise ValueError("pilot allocation wall limit exceeds the A100-hour cap")
    return maximum_hours


def _validate_contract_io(document: Mapping[str, Any]) -> None:
    _require_exact(
        _table(document, "contract_io"),
        {
            "maximum_bytes": _MAX_CONTRACT_BYTES,
            "require_regular_file": True,
            "require_single_link": True,
            "reject_symlink_ancestors": True,
            "digest_before_toml_parse": True,
        },
        label="contract I/O policy",
    )


def _validate_audit(
    document: Mapping[str, Any], parent_contract: NativeDiffusionV1Contract
) -> None:
    audit = _table(document, "audit")
    expected = {
        "account": "bio",
        "partition": "standard",
        "nodes": 1,
        "tasks": 1,
        "gpus": 0,
        "cpus_per_task": 8,
        "memory_gib_per_task": 32,
        "wall_minutes": 120,
        "third_node_required": True,
        "excluded_nodes": "all_four_gpu_producer_nodes",
        "producer_gpu_reinference_required": True,
        "producer_gpu_reinference_scope": (
            "all_four_folds_all_five_sealed_checkpoints_exact_float32_logit_bytes"
        ),
        "cpu_exact_reconstruction": [
            "count_prior_from_train_jsonl",
            "corruption_ledgers_from_score_jsonl",
            "calibration_sufficient_statistics",
            "louco_choices",
            "bootstrap_draws",
            "metrics",
            "checkpoint_selection",
            "pilot_gate_decision",
        ],
        "cpu_checkpoint_verification": [
            "physical_sha256",
            "logical_state_sha256",
            "exact_31_tensor_schema",
            "finite_float32_tensor_values",
            "checkpoint_metadata_bindings",
        ],
        "cpu_neural_forward_pass": "not_performed_no_cross_backend_numeric_or_argmax_claim",
        "independent_verifier_imports_producer": False,
        "independent_receipt_file": "independent-verification.json",
        "operational_receipt_file": "operational-receipt.json",
        "independent_receipt_fields": [
            "schema_version",
            "artifact",
            "decision_status",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "projection_evidence",
            "trainer_bundle_sha256_by_fold",
            "evaluator_bundle_sha256_by_fold",
            "pilot_bundle_sha256",
            "count_prior_sha256_by_fold",
            "checkpoint_digest_by_fold_and_step",
            "producer_gpu_reinference",
            "cpu_reconstruction",
            "metrics",
            "gates",
            "checks",
            "limitations",
        ],
        "operational_receipt_fields": [
            "schema_version",
            "artifact",
            "decision_status",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "producer_job",
            "producer_nodes",
            "audit_job",
            "audit_node",
            "slurm_accounting",
            "environment",
            "resources",
            "independent_receipt_sha256",
            "checks",
        ],
        "fold_digest_map_schema": {
            "exact_keys": list(_FOLD_RECEIPT_KEYS),
            "additional_keys": False,
            "value_type": "lowercase_ascii_sha256_hex_64",
        },
        "fold_digest_map_fields": [
            "trainer_bundle_sha256_by_fold",
            "evaluator_bundle_sha256_by_fold",
            "count_prior_sha256_by_fold",
        ],
        "checkpoint_digest_by_fold_and_step_schema": {
            "exact_fold_keys": list(_FOLD_RECEIPT_KEYS),
            "fold_additional_keys": False,
            "exact_step_keys": list(_CHECKPOINT_RECEIPT_KEYS),
            "step_additional_keys": False,
            "value_fields": list(_DIGEST_VALUE_FIELDS),
            "value_additional_fields": False,
            "digest_fields": list(_DIGEST_VALUE_FIELDS),
            "digest_type": "lowercase_ascii_sha256_hex_64",
        },
        "producer_gpu_reinference_schema": {
            "object_fields": ["by_fold_and_step", "all_fold_step_pairs_byte_equal"],
            "object_additional_fields": False,
            "exact_fold_keys": list(_FOLD_RECEIPT_KEYS),
            "fold_additional_keys": False,
            "exact_step_keys": list(_CHECKPOINT_RECEIPT_KEYS),
            "step_additional_keys": False,
            "value_fields": list(_REINFERENCE_VALUE_FIELDS),
            "value_additional_fields": False,
            "digest_fields": [
                "archived_residual_logit_slice_sha256",
                "reinferred_residual_logit_slice_sha256",
            ],
            "digest_type": "lowercase_ascii_sha256_hex_64",
            "digest_pair_must_match": True,
            "equality_field": "byte_equal",
            "equality_required_value": True,
            "slice_bytes": ("C_contiguous_little_endian_float32_selected_token_then_residue_order"),
            "aggregate_field": "all_fold_step_pairs_byte_equal",
            "aggregate_required_value": True,
        },
        "receipt_map_serialization": (
            "utf8_canonical_json_sorted_keys_compact_separators_allow_nan_false_single_lf"
        ),
        "receipt_map_sha256_rule": (
            "lowercase_sha256_of_complete_canonical_map_bytes_when_a_map_digest_is_named"
        ),
        "receipt_file_mode": "0444",
        "receipt_directory_mode": "0555",
    }
    _require_exact(audit, expected, label="independent CPU audit contract")
    for field in ("independent_receipt_file", "operational_receipt_file"):
        _relative_path(audit[field], label=f"audit.{field}")
    parent_compute = parent_contract.table("compute")
    _require_exact(
        {
            "account": audit["account"],
            "partition": audit["partition"],
            "cpus_per_task": audit["cpus_per_task"],
            "memory_gib_per_task": audit["memory_gib_per_task"],
        },
        {
            "account": parent_compute["account"],
            "partition": parent_compute["cpu_partition"],
            "cpus_per_task": parent_compute["cpus_per_task"],
            "memory_gib_per_task": parent_compute["memory_gib_per_task"],
        },
        label="parent CPU audit resources",
    )
    if parent_contract.table("recipe_selection")["independent_third_node_audit"] is not True:
        raise ValueError("parent does not authorize the required independent third-node audit")


def _validate_outputs(
    document: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    outputs = _table(document, "outputs")
    trainer_files = _tuple_of(
        outputs.get("trainer_bundle_files"), str, label="trainer bundle inventory"
    )
    evaluator_files = _tuple_of(
        outputs.get("evaluator_bundle_files"), str, label="evaluator bundle inventory"
    )
    pilot_files = _tuple_of(outputs.get("pilot_bundle_files"), str, label="pilot bundle inventory")
    expected_trainer_files = (
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "pilot_execution_v1.toml",
        "unconditional_v1.toml",
        "count_prior.npz",
        "environment.json",
        "fit_identity.json",
        "rng.json",
        "training_schedule.sha256",
        "training_trace.jsonl",
        "train_metrics.json",
        *_CHECKPOINT_BUNDLE_PATHS,
        "manifest.json",
    )
    expected_evaluator_files = (
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "pilot_execution_v1.toml",
        "unconditional_v1.toml",
        "environment.json",
        "rng.json",
        "trainer_bundle.sha256",
        "count_prior.sha256",
        "score_corruptions.npz",
        "score_residual_logits.npz",
        "fold_metrics.json",
        "manifest.json",
    )
    expected_pilot_files = (
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "pilot_execution_v1.toml",
        "unconditional_v1.toml",
        "fold_bundles.sha256",
        "pilot_bootstrap.npz",
        "pilot_metrics.json",
        "decision.json",
        "manifest.json",
    )
    if (
        trainer_files != expected_trainer_files
        or evaluator_files != expected_evaluator_files
        or pilot_files != expected_pilot_files
    ):
        raise ValueError("pilot semantic bundle inventory is invalid")
    for inventory in (trainer_files, evaluator_files, pilot_files):
        if len(set(inventory)) != len(inventory) or inventory[-1] != "manifest.json":
            raise ValueError("pilot output inventory is duplicated or not manifest-committed")
        for path in inventory:
            _relative_path(path, label="output inventory path")
    fixed = {
        "schema_version": 1,
        "canonical_json": True,
        "reject_nonfinite_json": True,
        "file_mode": "0444",
        "directory_mode": "0555",
        "file_link_count": 1,
        "manifest_published_last": True,
        "publication": "atomic_no_replace",
        "semantic_manifests_path_free": True,
        "bundled_contract_files": ["pilot_execution_v1.toml", "unconditional_v1.toml"],
        "bundled_contract_hashes": ["child_contract_sha256", "parent_contract_sha256"],
        "trainer_bundle_relative_path": "trainers/{outer_fold}",
        "evaluator_bundle_relative_path": "evaluators/{outer_fold}",
        "pilot_bundle_relative_path": "pilot",
        "trainer_bundle_directories": [".", "checkpoints"],
        "evaluator_bundle_directories": ["."],
        "pilot_bundle_directories": ["."],
        "trainer_manifest_fields": [
            "schema_version",
            "artifact",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "outer_fold",
            "fit_identity",
            "projection",
            "model",
            "rng",
            "training",
            "checkpoints",
            "artifacts",
        ],
        "evaluator_manifest_fields": [
            "schema_version",
            "artifact",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "outer_fold",
            "fit_identity_sha256",
            "trainer_bundle_sha256",
            "projection",
            "rng",
            "count_prior",
            "score",
            "status",
            "artifacts",
        ],
        "pilot_manifest_fields": [
            "schema_version",
            "artifact",
            "child_contract_sha256",
            "parent_contract_sha256",
            "git_commit",
            "fold_bundle_sha256",
            "checkpoint_selection",
            "comparator_selection",
            "bootstrap",
            "gates",
            "decision_status",
            "artifacts",
        ],
        "npz_archive_format": "zip_stored_npy_v1_0_dos_epoch_1980_member_order_v1",
        "npz_allow_pickle": False,
        "npz_member_order": "schema_array_order",
        "comparator_index_order": ["C0", "C0T"],
        "score_row_order": "ascending_sequence_id",
        "score_case_order": "ascending_sequence_id_then_level_1_to_64_then_replicate_0",
        "score_selected_token_order": "score_case_order_then_ascending_position",
        "checkpoint_axis_order": "ascending_checkpoint_steps",
        "residue_axis_order": "alphabet_ACDEFGHIKLMNPQRSTVWY",
        "count_prior_relative_position_axis_order": [
            "length_bin",
            "relative_position_bin",
            "residue_index",
        ],
        "operational_telemetry_excluded_from_semantic_bundles": True,
        "independent_verifier_imports_producer": False,
        "independent_receipt_file": "independent-verification.json",
        "operational_receipt_file": "operational-receipt.json",
        "post_execution_third_node_audit_required": True,
        "bundle_identity": {
            "schema": "path_free_exact_inventory_canonical_tree_map_v1",
            "document_fields": ["schema_version", "entries"],
            "document_additional_fields": False,
            "schema_version": 1,
            "entries_key": "entries",
            "exact_paths": "bundle_kind_files_union_directories_including_dot_root",
            "additional_paths": False,
            "path_format": ("normalized_relative_posix_path_with_dot_as_the_only_root_spelling"),
            "entry_fields": list(_TREE_ENTRY_FIELDS),
            "entry_additional_fields": False,
            "allowed_types": ["file", "directory"],
            "file_mode": "0444",
            "file_size": "exact_complete_file_byte_length_nonnegative_integer",
            "file_sha256": "lowercase_sha256_of_complete_file_bytes",
            "file_link_count": 1,
            "directory_mode": "0555",
            "directory_size": "exact_direct_child_entry_count_nonnegative_integer",
            "directory_sha256": (
                "lowercase_sha256_of_canonical_direct_child_name_to_type_map_bytes"
            ),
            "directory_link_count": (
                "2_plus_exact_direct_subdirectory_count_and_equal_to_lstat_st_nlink"
            ),
            "directory_child_map": ("exact_direct_children_only_name_to_file_or_directory_type"),
            "symlinks_allowed": False,
            "special_files_allowed": False,
            "canonical_json_bytes": (
                "utf8_sort_keys_true_separators_comma_colon_ensure_ascii_false_"
                "allow_nan_false_single_lf"
            ),
            "tree_sha256": ("lowercase_sha256_of_complete_canonical_tree_document_bytes"),
            "manifest_is_included_file": True,
            "snapshot_rule": (
                "publish_manifest_last_atomic_no_replace_then_lstat_before_and_after_"
                "hashing_complete_immutable_tree_including_manifest"
            ),
        },
        "sha256_sidecars": {
            "content": "exactly_64_lowercase_ascii_sha256_hex_bytes_plus_single_lf",
            "byte_length": 65,
            "additional_whitespace_allowed": False,
            "training_schedule_sha256_target": (
                "canonical_learning_rate_schedule_transcript_bytes"
            ),
            "trainer_bundle_sha256_target": ("canonical_sealed_trainer_bundle_tree_document_bytes"),
            "count_prior_sha256_target": ("exact_sealed_trainer_count_prior_npz_file_bytes"),
            "fold_bundles_sha256_target": (
                "canonical_exact_four_fold_trainer_evaluator_bundle_digest_map_bytes"
            ),
            "fold_bundle_map_exact_keys": list(_FOLD_RECEIPT_KEYS),
            "fold_bundle_map_additional_keys": False,
            "fold_bundle_value_fields": list(_FOLD_BUNDLE_VALUE_FIELDS),
            "fold_bundle_value_additional_fields": False,
            "fold_bundle_digest_type": "lowercase_ascii_sha256_hex_64",
            "fold_bundle_map_serialization": (
                "utf8_sort_keys_true_separators_comma_colon_ensure_ascii_false_"
                "allow_nan_false_single_lf"
            ),
        },
    }
    _require_exact(
        {field: outputs.get(field) for field in fixed},
        fixed,
        label="pilot output publication contract",
    )
    expected_schemas = {
        "count_prior_npz_schema": (
            ("effective_count_scale", "<f8", ("1",)),
            ("unigram_probability", "<f8", ("20",)),
            ("length_edges", "|u1", ("6",)),
            ("relative_position_probability", "<f8", ("5", "10", "20")),
            ("log_relative_position_probability", "<f8", ("5", "10", "20")),
        ),
        "score_corruptions_npz_schema": (
            ("sequence_id", "|S64", ("{score_rows}",)),
            ("homology_component_id", "|S64", ("{score_rows}",)),
            ("union_component_id", "|S64", ("{score_rows}",)),
            ("length", "|u1", ("{score_rows}",)),
            ("sampling_weight", "<f8", ("{score_rows}",)),
            ("clean_tokens", "|u1", ("{score_rows}", "50")),
            ("attention_mask", "|b1", ("{score_rows}", "50")),
            ("case_id", "|S64", ("{score_cases}",)),
            ("row_index", "<u2", ("{score_cases}",)),
            ("level", "|u1", ("{score_cases}",)),
            ("replicate", "|u1", ("{score_cases}",)),
            ("row_seed", "<u8", ("{score_cases}",)),
            ("mask_count", "|u1", ("{score_cases}",)),
            ("corrupted_tokens", "|u1", ("{score_cases}", "50")),
            ("selected_mask", "|b1", ("{score_cases}", "50")),
        ),
        "score_residual_logits_npz_schema": (
            ("case_id", "|S64", ("{score_cases}",)),
            ("case_offsets", "<u8", ("{score_cases_plus_one}",)),
            ("position", "|u1", ("{score_selected_tokens}",)),
            ("target_token", "|u1", ("{score_selected_tokens}",)),
            ("count_log_probability", "<f8", ("{score_selected_tokens}", "20")),
            ("checkpoint_step", "<u2", ("5",)),
            ("residual_logit", "<f4", ("5", "{score_selected_tokens}", "20")),
        ),
        "pilot_bootstrap_npz_schema": (
            ("checkpoint_step", "<u2", ("5",)),
            ("r128_mean_nll", "<f8", ("5",)),
            ("c0_mean_nll", "<f8", ("1",)),
            ("c0t_mean_nll", "<f8", ("1",)),
            ("selected_comparator_index", "|u1", ("1",)),
            ("r128_bootstrap_mean_nll", "<f8", ("5", "10000")),
            ("comparator_bootstrap_mean_nll", "<f8", ("10000",)),
            ("relative_improvement", "<f8", ("5", "10000")),
        ),
    }
    for field, expected in expected_schemas.items():
        if _npz_schema(outputs.get(field), label=f"outputs.{field}") != expected:
            raise ValueError("pilot NPZ schema is invalid")
    return trainer_files, evaluator_files, pilot_files


def _validate_leakage(document: Mapping[str, Any]) -> None:
    leakage = _table(document, "leakage")
    expected = {
        "trainer_allowed_roles": ["development_train"],
        "trainer_allowed_fields": ["sequence_id", "sequence", "sampling_weight"],
        "trainer_process_receives_score_path": False,
        "trainer_process_receives_projection_root": False,
        "score_opened_only_after_all_checkpoints_sealed": True,
        "validation_used_for_early_stopping": False,
        "fold4_visible": False,
        "fold4_path_allowed": False,
        "fold4_sequence_rows_allowed": 0,
        "labels_allowed": False,
        "provenance_allowed": False,
        "study_keys_allowed": False,
        "oracle_predictions_allowed": False,
        "structures_allowed": False,
        "organizer_reference_allowed": False,
        "proposal_sampling_allowed": False,
        "post_fold4_changes_require_new_version": True,
    }
    _require_exact(leakage, expected, label="pilot leakage and score chronology boundary")


def _validate_acquisition_exclusions(document: Mapping[str, Any]) -> None:
    _require_exact(
        _table(document, "acquisition_exclusions"),
        {
            "acquisition_configs_allowed": False,
            "acquisition_outputs_allowed": False,
            "start_ids_allowed": False,
            "rollout_outputs_allowed": False,
            "reward_mean_allowed": False,
            "reward_uncertainty_allowed": False,
            "reward_ucb_allowed": False,
            "generator_uncertainty_allowed": False,
            "wet_lab_results_allowed": False,
            "memory_or_history_allowed": False,
            "ensemble_predictions_allowed": False,
            "excluded_scope": "trainer_evaluator_checkpoint_selection_and_pilot_gate",
        },
        label="pilot acquisition exclusions",
    )


def _parse_contract(
    document: Mapping[str, Any],
    *,
    config_sha256: str,
    parent_contract: NativeDiffusionV1Contract,
) -> NativeDiffusionV1PilotContract:
    if _sha256(config_sha256, label="config_sha256") != CONFIG_SHA256:
        raise ValueError("v1 pilot parser received an unauthenticated config digest")
    if any(type(key) is not str for key in document):
        raise ValueError("v1 pilot top-level schema contains a non-string key")
    if set(document) != _TOP_FIELDS:
        raise ValueError(
            "v1 pilot top-level schema mismatch: "
            f"missing={sorted(_TOP_FIELDS - set(document))}, "
            f"extra={sorted(set(document) - _TOP_FIELDS)}"
        )
    _validate_recursive(document, label="pilot contract")
    if document.get("schema_version") != 1 or type(document.get("schema_version")) is not int:
        raise ValueError("v1 pilot schema_version must equal 1")
    if document.get("artifact") != ARTIFACT:
        raise ValueError("v1 pilot artifact identity is invalid")
    if document.get("evidence_doc") != "docs/benchmarks/native_categorical_diffusion_v1.md":
        raise ValueError("v1 pilot evidence document is invalid")
    for name, keys in _TABLE_KEYS.items():
        _exact_keys(_table(document, name), keys, label=name)
    _validate_contract_io(document)
    _validate_fixed_tables(document)
    _validate_inheritance(document, parent_contract)
    _validate_projection(document)
    folds = _parse_folds(document)
    seed, checkpoints = _validate_pilot(document)
    _validate_training_and_evaluation(document)
    _validate_count_prior(document, parent_contract)
    checkpoint_paths = _validate_rng_and_checkpoints(document)
    checkpoint_tensors = _validate_checkpoint_metadata(document)
    _validate_barrier(document)
    maximum_hours = _validate_resources(document)
    _validate_audit(document, parent_contract)
    trainer_files, evaluator_files, pilot_files = _validate_outputs(document)
    _validate_leakage(document)
    _validate_acquisition_exclusions(document)

    frozen = _freeze(document)
    if not isinstance(frozen, Mapping):  # pragma: no cover - construction invariant
        raise RuntimeError("frozen pilot contract is not a mapping")
    contract = NativeDiffusionV1PilotContract(
        config_sha256=config_sha256,
        parent_config_sha256=PARENT_CONFIG_SHA256,
        projection_top_sha256=PROJECTION_TOP_SHA256,
        projection_tree_sha256=PROJECTION_TREE_SHA256,
        projection_receipt_sha256=PROJECTION_RECEIPT_SHA256,
        seed=seed,
        folds=folds,
        checkpoint_steps=checkpoints,
        checkpoint_paths=checkpoint_paths,
        maximum_pilot_a100_hours=maximum_hours,
        trainer_bundle_files=trainer_files,
        evaluator_bundle_files=evaluator_files,
        pilot_bundle_files=pilot_files,
        checkpoint_tensors=checkpoint_tensors,
        parent_contract=parent_contract,
        document=frozen,
    )
    if any(
        contract.fit_identity_sha256(fold.outer_fold) != fold.fit_identity_sha256
        for fold in contract.folds
    ):
        raise ValueError("pilot fit identity SHA-256 pin is inconsistent")
    return contract


def load_pilot_execution_v1_contract(
    path: str | os.PathLike[str],
    *,
    parent_path: str | os.PathLike[str] | None = None,
) -> NativeDiffusionV1PilotContract:
    """Authenticate, separately parse, validate, and freeze parent and child."""

    child_source = Path(path)
    payload = _read_contract_bytes(child_source)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise ValueError(f"v1 pilot contract SHA-256 mismatch: {digest}")
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("v1 pilot contract is not valid UTF-8 TOML") from error

    parent_source = (
        Path(parent_path)
        if parent_path is not None
        else child_source.parent / "unconditional_v1.toml"
    )
    parent_payload = _read_contract_bytes(parent_source)
    parent_digest = hashlib.sha256(parent_payload).hexdigest()
    if parent_digest != PARENT_CONFIG_SHA256:
        raise ValueError(f"v1 parent contract SHA-256 mismatch: {parent_digest}")
    try:
        parent_document = tomllib.loads(parent_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("v1 parent contract is not valid UTF-8 TOML") from error
    parent_contract = _parse_parent_contract(
        parent_document,
        config_sha256=parent_digest,
    )
    if (
        parent_contract.config_sha256 != parent_digest
        or parent_contract.document.get("artifact") != PARENT_ARTIFACT
    ):
        raise ValueError("v1 parent contract identity mismatch")
    return _parse_contract(
        document,
        config_sha256=digest,
        parent_contract=parent_contract,
    )

"""Machine-checked contract for the evolutionary/KL diffusion research campaign.

This module plans runs, validates budgets, and applies pure fail-closed research
gates. It cannot reveal outcomes, evaluate an oracle, train a model, or authorize
the blocked research protocol.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FROZEN_PROTOCOL_SHA256 = "ccbc55e5f20d851c3495413ff1f26f8ed2e0e0e818b076fde76641937c346463"

SCREEN_SEEDS = (17, 42, 91, 137, 271)
PUBLISHED_DEVELOPMENT_CONFIRMATION_SEEDS = (314, 577, 811, 1217, 2027)
# Backward-compatible planning name. These published seeds are not confirmatory.
CONFIRMATION_SEEDS = PUBLISHED_DEVELOPMENT_CONFIRMATION_SEEDS
METHOD_IDS = (
    "tuned_peptide_ga",
    "categorical_diffusion_posthoc",
    "diffusion_reward_kl_no_search",
    "arcadiamp_style_iterative_d3pm",
    "tr2d2_style_tree_offpolicy",
    "mp2d_style_inference_search",
    "ga_endpoint_distillation_no_kg",
    "counterfactual_softkg_evolutionary_diffusion",
)
ABLATION_IDS = (
    "ablation_no_spectral_representation",
    "ablation_no_counterfactual_credit",
    "ablation_singleton_kg",
    "ablation_no_endpoint_distillation",
    "ablation_no_kl_controls",
)
SCREEN_CONFIGURATION_IDS = METHOD_IDS + ABLATION_IDS
CONFIRMATION_METHOD_IDS = (
    "tuned_peptide_ga",
    "tr2d2_style_tree_offpolicy",
    "counterfactual_softkg_evolutionary_diffusion",
)
PRIMARY_COMPARATOR_IDS = (
    "tuned_peptide_ga",
    "tr2d2_style_tree_offpolicy",
)
_SCREEN_PREREQUISITES = (
    "accepted_base_fold_native_policy_checkpoints",
    "accepted_de_novo_checkpoint_aggregation_contract",
    "accepted_training_homology_exclusion_contract",
    "accepted_generator_oracle_training_provenance_separation",
    "accepted_generated_sequence_oracle_adapter",
    "accepted_common_initial_oracle_resource_accounting",
    "accepted_content_pinned_hidden_oracle_contract",
    "accepted_oracle_objective_constraint_semantics",
    "accepted_oracle_missing_censoring_replicate_semantics",
    "accepted_terminal_evaluation_contract",
    "accepted_organizer_reference_set_receipt",
    "accepted_calibrated_joint_posterior",
    "accepted_generated_sequence_esm_contact_producer",
    "accepted_persistent_search_ledger",
    "accepted_all_eight_method_adapters",
    "accepted_independent_verifier",
    "accepted_manifest_hashed_telemetry_pipeline",
    "accepted_external_trusted_receipt_issuer",
)
_CONFIRMATORY_PREREQUISITES = (
    "accepted_hidden_confirmation_seed_reveal_contract",
    "accepted_sequestered_confirmation_oracle_access_control",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact",
    "status",
    "decision_date",
    "automatic_production_eligible",
    "execution_authorized",
    "evidence_class",
    "biological_superiority_claim_allowed",
    "screen_seeds",
    "confirmation_seeds",
    "confirmation_seed_status",
    "screen_method_ids",
    "confirmation_method_ids",
    "support",
    "cohorts",
    "design",
    "oracle_query_contract",
    "resources",
    "batching",
    "metrics",
    "uncertainty",
    "terminal_evaluation",
    "stopping",
    "statistics",
    "promotion",
    "prerequisites",
    "methods",
    "ablations",
    "provenance",
    "evidence_boundaries",
}
_SUPPORT_KEYS = {
    "alphabet",
    "min_length",
    "max_length",
    "linear_unmodified_free_termini",
    "canonical_sequence_function",
    "exact_training_overlap_forbidden",
    "organizer_reference_role",
    "organizer_reference_may_shape_generation",
    "organizer_reference_may_shape_model_fit_or_search_score",
    "organizer_reference_compliance_may_veto_shadow_or_production",
    "organizer_reference_validator_commit",
    "organizer_reference_similarity",
    "organizer_reference_max_similarity",
    "organizer_reference_set_receipt_status",
    "training_homology_exclusion_status",
    "generator_oracle_training_provenance_separation_status",
}
_COHORT_KEYS = {
    "screen_unit",
    "confirmation_unit",
    "de_novo_rotation_dimension",
    "de_novo_rotation_count",
    "de_novo_checkpoint_aggregation_status",
    "retrospective_fixed_pool_protocol",
    "retrospective_fixed_pool_rotations",
    "retrospective_results_may_enter_de_novo_screen_or_confirmation",
    "retrospective_calls_in_de_novo_budget",
    "published_confirmation_seeds_may_support_confirmatory_or_promotion_claim",
    "successor_hidden_confirmation_seed_count",
    "confirmatory_seed_source",
    "confirmation_oracle_access",
}
_DESIGN_KEYS = {
    "initial_design_unique_calls",
    "adaptive_batches",
    "unique_calls_per_batch",
    "method_controlled_seats_per_batch",
    "random_reserve_seats_per_batch",
    "total_unique_calls",
    "proposal_attempt_cap",
    "kg_shortlist_cap",
    "kg_joint_q_cap",
    "kg_exact_pool_cap",
    "kg_max_combinations",
    "kg_fantasies",
    "oracle_service_batch_cap",
    "terminal_rule",
    "common_initial_design_within_seed",
    "common_random_reserve_within_seed",
    "common_initial_responses_delivered_before_scientific_clock",
    "common_initial_calls_charged_per_run",
    "cross_method_cache_is_charged_logically",
    "exact_same_run_identity_cache_replay_is_not_charged",
    "random_reserve_is_fixed_before_method_selection",
    "scheduled_common_reserve_scope",
    "scheduled_common_reserve_proposal_exclusion",
    "scheduled_common_reserve_submission_rule",
    "timed_cross_method_cache_release_policy",
    "method_controlled_duplicate_rule",
    "method_controlled_exhaustion_rule",
    "overflow_reserve_scope",
    "overflow_reserve_attempt_cap_per_batch",
    "overflow_reserve_collision_rule",
    "overflow_reserve_exhaustion_rule",
}
_ORACLE_QUERY_KEYS = {
    "status",
    "query_identity_fields",
    "logical_call_unit",
    "all_required_endpoints_return_atomically",
    "cross_method_cache_hit_is_charged",
    "exact_same_run_identity_cache_replay_is_not_a_submission",
    "same_identity_resubmission_after_any_submission_is_forbidden",
    "transport_retry_may_only_poll_existing_submission",
    "new_replicate_id_is_new_identity_and_is_charged",
    "failed_submitted_call_is_charged",
    "missing_submitted_response_is_charged",
    "censored_submitted_response_is_charged",
    "partial_submitted_response_is_charged_and_ineligible",
    "timeout_after_submission_is_charged",
    "failed_or_ineligible_call_may_be_replaced_without_charge",
    "oracle_contract_sha256_status",
    "objective_constraint_semantics_status",
    "missing_censoring_semantics_status",
    "timed_cross_method_physical_cache_latency_advantage_allowed",
    "timed_cross_method_physical_cache_reuse_allowed",
    "generator_oracle_training_provenance_status",
}
_RESOURCE_KEYS = {
    "slurm_account",
    "cpu_partition",
    "gpu_partition",
    "gpu_type",
    "nodes_per_run",
    "gpus_per_gpu_run",
    "cpus_per_run",
    "host_memory_gib",
    "max_peak_gpu_memory_gib",
    "scientific_wall_seconds",
    "outer_allowance_seconds",
    "scientific_clock_source",
    "scientific_clock_start",
    "scientific_clock_stop",
    "scientific_elapsed_accounting",
    "outer_allowance_elapsed_accounting",
    "resume_clock_rule",
    "timing_receipt_authentication",
    "scientific_clock_includes",
    "outer_allowance_scope",
    "outer_allowance_execution",
    "outer_allowance_accelerator_hours",
    "scientific_schedule",
    "scientific_schedule_seed",
    "scientific_schedule_key",
    "scientific_schedule_wave_launch",
    "scientific_schedule_wave_size",
    "array_concurrency_cap",
    "max_output_gib_per_run",
    "scratch_ceiling_gib",
    "screen_runs",
    "screen_logical_unique_calls",
    "screen_a100_hour_ceiling",
    "confirmation_runs",
    "confirmation_logical_unique_calls",
    "confirmation_a100_hour_ceiling",
    "scientific_runs",
    "scientific_logical_unique_calls",
    "scientific_a100_hour_ceiling",
    "reproduction_runs",
    "reproduction_methods",
    "reproduction_seed_count",
    "reproduction_seed_status",
    "reproduction_logical_unique_calls",
    "reproduction_scientific_wall_seconds_per_run",
    "reproduction_a100_hour_ceiling",
    "reproduction_results_may_enter_screen_confirmation_or_promotion",
    "hard_evidence_and_reproduction_a100_hour_ceiling",
    "common_initial_oracle_production_accelerator_accounting_status",
    "adapter_or_tuning_work_authorized_by_v1",
    "adapter_or_tuning_a100_hours_in_v1",
    "adapter_or_tuning_logical_calls_in_v1",
    "unallocated_resource_pool_allowed",
}
_BATCHING_KEYS = {
    "profile",
    "rollout_batch_size_cap",
    "proposal_batch_size_cap",
    "surrogate_batch_size_cap",
    "kg_candidate_chunk_size_cap",
    "kg_fantasy_chunk_size_cap",
    "oracle_batch_size_cap",
    "replay_sequence_batch_cap",
    "replay_token_batch_cap",
    "gradient_accumulation_steps",
    "record_realized_sizes",
    "seed_order_invariant_to_batching",
}
_METRIC_KEYS = {
    "primary",
    "objectives",
    "objective_bounds",
    "hypervolume_reference",
    "hypervolume_auc_denominator_calls",
    "primary_auc_quadrature",
    "primary_checkpoint_input",
    "primary_checkpoint_values",
    "primary_value_source",
    "primary_archive_domain",
    "broad_spectrum_role",
    "pairwise_sequence_diversity_estimand",
    "pairwise_embedding_diversity_estimand",
    "pairwise_diversity_fewer_than_two_value",
    "pairwise_embedding_nonfinite_or_zero_norm_value",
    "scalar_utility",
    "no_action_utility",
    "call_checkpoint_start",
    "call_checkpoint_step",
    "call_checkpoint_count",
    "wall_checkpoint_minutes",
    "secondary",
    "top10_feasible_mean_utility_checkpoint_estimand",
    "top10_feasible_mean_utility_auc_estimand",
    "wall_time_auc_estimand",
    "wall_time_checkpoint_assignment",
    "terminal_abstention_rate_estimand",
    "terminal_abstention_error_estimand",
    "parent_child_contrast_coverage_estimand",
    "parent_child_contrast_accuracy_estimand",
}
_UNCERTAINTY_KEYS = {
    "ece_equal_mass_bins",
    "coverage_levels",
    "paired_bootstrap_samples",
    "paired_bootstrap_seed",
    "de_novo_paired_bootstrap_unit",
    "paired_bootstrap_statistic",
    "paired_bootstrap_interval_level",
    "paired_bootstrap_interval_type",
    "paired_bootstrap_interval_quantiles",
    "paired_bootstrap_quantile_convention",
    "retrospective_bootstrap_unit",
    "retrospective_bootstrap_samples",
    "kg_fantasy_sensitivity_fantasies",
    "kg_rank_stability_kendall_tau_min",
    "kg_monte_carlo_se_fraction_of_top_two_gap_max",
    "kg_rank_correlation",
    "kg_top_two_gap_requirement",
    "kg_zero_or_tied_top_two_gap",
    "kg_nonfinite_score_se_or_rank_statistic",
    "coverage_reporting_scope",
    "coverage_interval_convention",
    "coverage_missing_nonfinite_or_wrong_level_count",
}
_TERMINAL_EVALUATION_KEYS = {
    "candidate_domain",
    "point_estimate",
    "feasibility",
    "oracle_constraint_extension_at_runtime_allowed",
    "candidate_truth_eligibility",
    "recommendation_tie_break",
    "minimum_eligible_candidates",
    "abstain_if",
    "regret_domain",
    "regret_oracle_utility",
    "regret",
    "empty_feasible_domain_oracle_utility",
    "abstention_regret_uses_no_action_utility",
    "missing_censored_or_nonfinite_is_ineligible_not_imputed",
}
_STOPPING_KEYS = {
    "stop_at_unique_calls_or_wall_seconds_whichever_first",
    "discard_unsealed_partial_batch",
    "carry_last_sealed_incumbent_to_later_checkpoints",
    "required_terminal_eligible_candidates",
    "infrastructure_reruns_before_first_oracle_response",
    "result_driven_reruns",
    "resume_after_response_from_last_authenticated_round_only",
    "algorithmic_failure_is_retained",
    "max_local_complete_transition_kl_mean",
    "max_local_complete_transition_kl_p99",
    "max_frozen_reference_path_kl_mean",
    "max_frozen_reference_transition_kl_p99",
    "min_replay_ess_fraction",
    "max_normalized_replay_weight",
    "max_policy_version_lag",
    "max_validity_drop_fraction",
    "nonfinite_or_psd_support_failure",
    "forbidden_support_or_overlap",
    "unsealed_oracle_response",
    "no_kl_ablation_ignores_only_kl_thresholds",
    "kg_tie_or_numerical_instability",
}
_STATISTIC_KEYS = {
    "screen_is_descriptive_only",
    "screen_two_sided_exact_sign_test_minimum_p",
    "run_primary_missing_nonfinite_or_no_initial_checkpoint",
    "early_stop_primary_metric",
    "screen_effect",
    "screen_full_mean",
    "screen_full_unique_highest",
    "screen_comparison_absolute_tolerance",
    "screen_pair_success",
    "screen_positive_pairs_required",
    "screen_median_additive_margin",
    "screen_core_ablation_gate",
    "confirmation_primary_comparators",
    "confirmation_effect",
    "confirmation_additive_materiality_margin",
    "confirmation_comparison_absolute_tolerance",
    "confirmation_pair_success",
    "confirmation_missing_nonfinite_or_tie",
    "confirmation_one_sided_alpha",
    "confirmation_exact_sign_test_minimum_p",
    "confirmation_pairs_required_above_margin",
    "confirmation_comparator_gate",
    "confirmation_global_iut_gate",
    "global_claim",
    "multiple_testing_adjustment_for_intersection_union",
    "combine_screen_and_confirmation_for_p_value",
    "report_hodges_lehmann",
    "hodges_lehmann_estimand",
    "median_convention",
    "bootstrap_interval_is_descriptive",
    "ablations_are_exploratory",
    "ablation_p_values_are_confirmatory",
    "ablation_claim_requires_fresh_seed_preregistration",
}
_PROMOTION_KEYS = {
    "screen_requires_highest_mean_primary_metric_among_eight_methods",
    "screen_requires_each_core_ablation_positive_pairs",
    "screen_requires_each_core_ablation_median_above_additive_margin",
    "confirmation_requires_all_pairs_above_additive_margin",
    "primary_missing_nonfinite_or_tied_required_gate_is_failure",
    "full_method_kl_replay_numerical_integrity_result_required",
    "yield_and_diversity_gate_comparators",
    "yield_and_diversity_pairwise_gate_scope",
    "max_valid_unique_reference_safe_yield_additive_loss",
    "max_hill2_effective_cluster_loss_fraction",
    "max_largest_cluster_share_additive_increase",
    "secondary_gate_comparison_absolute_tolerance",
    "valid_unique_reference_safe_yield_estimand",
    "yield_zero_charged_identity_value",
    "identity70_metric",
    "identity70_threshold",
    "identity70_linkage",
    "hill2_estimand",
    "hill2_zero_valid_sequence_value",
    "hill2_zero_comparator_gate",
    "largest_cluster_share_estimand",
    "largest_cluster_share_zero_valid_sequence_value",
    "yield_diversity_comparison_rule",
    "max_ece",
    "max_ece_additive_degradation",
    "calibration_gate_scope",
    "calibration_target",
    "ece_estimand",
    "ece_equal_mass_bin_rule",
    "ece_empty_or_incomplete_value",
    "coverage90_estimand",
    "coverage90_empty_or_incomplete_value",
    "calibration_comparison_rule",
    "calibration_per_seed_report_required",
    "coverage90_lower",
    "coverage90_upper",
    "independent_chronological_or_assay_result_required",
    "surrogate_screen_max_shadow_generator_mixture_quota",
    "surrogate_screen_max_production_generator_mixture_quota",
    "shadow_candidates_may_enter_submission_or_top100",
    "production_requires_confirmation_and_independent_result",
    "production_quota_requires_new_content_pinned_promotion_protocol",
    "surrogate_pass_guarantees_final_library_or_top100_seat",
    "no_go_on_failed_confirmation",
    "screen_failure_is_no_go",
    "unsequestered_confirmation_is_no_go",
    "independent_result_missing_or_failure_is_no_go",
    "any_required_gate_failure_is_no_go",
    "passing_research_gates_authorizes_v1_production",
}
_PREREQUISITE_KEYS = {
    *_SCREEN_PREREQUISITES,
    *_CONFIRMATORY_PREREQUISITES,
    "required_base_fold_policy_checkpoints",
    "independent_biological_or_chronological_panel_available",
    "native_v0_is_usable_prior",
    "native_v1_currently_has_accepted_checkpoint",
    "current_activity_scorer_supplies_calibrated_uncertainty",
    "current_activity_scorer_supplies_toxicity_or_selectivity",
}
_PROVENANCE_KEYS = {
    "semantic_format",
    "manifest",
    "file_mode",
    "directory_mode",
    "all_artifacts_including_operational_telemetry_manifest_hashed",
    "operational_telemetry_outside_semantic_equivalence_hashes_only",
    "log_all_rejected_and_unevaluated_proposals",
    "separate_query_and_terminal_recommendation_ids",
    "excluded_node_independent_verifier",
    "verifier_may_import_producer_or_search_implementation",
    "external_trusted_receipt_required",
    "trusted_receipt_location",
    "producer_may_write_or_replace_trusted_receipt",
    "trusted_receipt_binds",
}
_EVIDENCE_BOUNDARY_KEYS = {
    "toy_smoke_is_scientific_evidence",
    "computational_oracle_can_establish_biological_superiority",
    "retrospective_fixed_pool_can_establish_de_novo_quality",
    "equiformer_v3_validates_laplacian_representation",
    "counterfactual_means_biological_causal_effect",
}


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One of the eight matched method arms."""

    method_id: str
    role: str
    implementation: str
    requires_diffusion: bool
    adaptive_querying: bool
    uses_endpoint_distillation: bool


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One frozen component removal from the full proposal."""

    ablation_id: str
    base_method: str
    change: str


@dataclass(frozen=True, slots=True)
class CampaignRun:
    """One seed/configuration pair in a named evidence cohort."""

    ordinal: int
    phase: Literal["screen", "confirmation"]
    configuration_id: str
    seed: int
    unique_oracle_call_budget: int
    scientific_wall_seconds: int
    schedule_block: int
    schedule_wave: int
    schedule_slot: int


@dataclass(frozen=True, slots=True)
class CampaignBudgetSummary:
    """Exact logical-call and upper-bound accelerator accounting."""

    screen_runs: int
    confirmation_runs: int
    scientific_runs: int
    screen_logical_unique_calls: int
    confirmation_logical_unique_calls: int
    scientific_logical_unique_calls: int
    screen_a100_hour_ceiling: float
    confirmation_a100_hour_ceiling: float
    scientific_a100_hour_ceiling: float


@dataclass(frozen=True, slots=True)
class EvidenceCohortContract:
    """Disjoint de novo and retrospective evidence units."""

    screen_unit: str
    confirmation_unit: str
    de_novo_rotation_dimension: str
    de_novo_rotation_count: int
    de_novo_checkpoint_aggregation_status: str
    retrospective_fixed_pool_protocol: str
    retrospective_fixed_pool_rotations: int
    retrospective_results_may_enter_de_novo_screen_or_confirmation: bool
    retrospective_calls_in_de_novo_budget: int
    published_confirmation_seeds_may_support_confirmatory_or_promotion_claim: bool
    successor_hidden_confirmation_seed_count: int
    confirmatory_seed_source: str
    confirmation_oracle_access: str


@dataclass(frozen=True, slots=True)
class OracleQueryContract:
    """Logical-call identity and fail-closed charging semantics."""

    status: str
    query_identity_fields: tuple[str, ...]
    logical_call_unit: str
    all_required_endpoints_return_atomically: bool
    cross_method_cache_hit_is_charged: bool
    exact_same_run_identity_cache_replay_is_not_a_submission: bool
    same_identity_resubmission_after_any_submission_is_forbidden: bool
    transport_retry_may_only_poll_existing_submission: bool
    new_replicate_id_is_new_identity_and_is_charged: bool
    failed_submitted_call_is_charged: bool
    missing_submitted_response_is_charged: bool
    censored_submitted_response_is_charged: bool
    partial_submitted_response_is_charged_and_ineligible: bool
    timeout_after_submission_is_charged: bool
    failed_or_ineligible_call_may_be_replaced_without_charge: bool
    oracle_contract_sha256_status: str
    objective_constraint_semantics_status: str
    missing_censoring_semantics_status: str
    timed_cross_method_physical_cache_latency_advantage_allowed: bool
    timed_cross_method_physical_cache_reuse_allowed: bool
    generator_oracle_training_provenance_status: str


@dataclass(frozen=True, slots=True)
class SearchResourceLimits:
    """Hard per-run and campaign planning ceilings."""

    slurm_account: str
    cpu_partition: str
    gpu_partition: str
    gpu_type: str
    nodes_per_run: int
    gpus_per_gpu_run: int
    proposal_attempt_cap: int
    kg_shortlist_cap: int
    kg_joint_q_cap: int
    kg_exact_pool_cap: int
    kg_max_combinations: int
    kg_fantasies: int
    oracle_service_batch_cap: int
    cpus_per_run: int
    host_memory_gib: int
    max_peak_gpu_memory_gib: int
    scientific_wall_seconds: int
    outer_allowance_seconds: int
    scientific_clock_source: str
    scientific_clock_start: str
    scientific_clock_stop: str
    scientific_elapsed_accounting: str
    outer_allowance_elapsed_accounting: str
    resume_clock_rule: str
    timing_receipt_authentication: str
    scientific_clock_includes: tuple[str, ...]
    outer_allowance_scope: str
    outer_allowance_execution: str
    outer_allowance_accelerator_hours: float
    scientific_schedule: str
    scientific_schedule_seed: int
    scientific_schedule_key: str
    scientific_schedule_wave_launch: str
    scientific_schedule_wave_size: int
    array_concurrency_cap: int
    max_output_gib_per_run: int
    scratch_ceiling_gib: int
    reproduction_runs: int
    reproduction_methods: tuple[str, ...]
    reproduction_seed_count: int
    reproduction_seed_status: str
    reproduction_logical_unique_calls: int
    reproduction_scientific_wall_seconds_per_run: int
    reproduction_a100_hour_ceiling: float
    reproduction_results_may_enter_screen_confirmation_or_promotion: bool
    hard_evidence_and_reproduction_a100_hour_ceiling: float
    common_initial_oracle_production_accelerator_accounting_status: str
    adapter_or_tuning_work_authorized_by_v1: bool
    adapter_or_tuning_a100_hours_in_v1: float
    adapter_or_tuning_logical_calls_in_v1: int
    unallocated_resource_pool_allowed: bool


@dataclass(frozen=True, slots=True)
class BatchingLimits:
    """Every launcher-visible batching and accumulation ceiling."""

    profile: str
    rollout_batch_size_cap: int
    proposal_batch_size_cap: int
    surrogate_batch_size_cap: int
    kg_candidate_chunk_size_cap: int
    kg_fantasy_chunk_size_cap: int
    oracle_batch_size_cap: int
    replay_sequence_batch_cap: int
    replay_token_batch_cap: int
    gradient_accumulation_steps: int
    record_realized_sizes: bool
    seed_order_invariant_to_batching: bool


@dataclass(frozen=True, slots=True)
class StatisticalReportingContract:
    """Complete, outcome-blind interval and secondary-metric definitions."""

    paired_bootstrap_statistic: str
    paired_bootstrap_interval_level: float
    paired_bootstrap_interval_type: str
    paired_bootstrap_interval_quantiles: tuple[float, ...]
    paired_bootstrap_quantile_convention: str
    hodges_lehmann_estimand: str
    median_convention: str
    coverage_reporting_scope: str
    coverage_interval_convention: str
    coverage_missing_nonfinite_or_wrong_level_count: str
    top10_feasible_mean_utility_checkpoint_estimand: str
    top10_feasible_mean_utility_auc_estimand: str
    wall_time_auc_estimand: str
    wall_time_checkpoint_assignment: str
    terminal_abstention_rate_estimand: str
    terminal_abstention_error_estimand: str
    parent_child_contrast_coverage_estimand: str
    parent_child_contrast_accuracy_estimand: str


@dataclass(frozen=True, slots=True)
class SearchStoppingRules:
    """Outcome-blind stopping and numerical-integrity gates."""

    stop_at_unique_calls_or_wall_seconds_whichever_first: bool
    discard_unsealed_partial_batch: bool
    carry_last_sealed_incumbent_to_later_checkpoints: bool
    required_terminal_eligible_candidates: int
    infrastructure_reruns_before_first_oracle_response: int
    result_driven_reruns: int
    resume_after_response_from_last_authenticated_round_only: bool
    algorithmic_failure_is_retained: bool
    max_local_complete_transition_kl_mean: float
    max_local_complete_transition_kl_p99: float
    max_frozen_reference_path_kl_mean: float
    max_frozen_reference_transition_kl_p99: float
    min_replay_ess_fraction: float
    max_normalized_replay_weight: float
    max_policy_version_lag: int
    max_validity_drop_fraction: float
    nonfinite_or_psd_support_failure: str
    forbidden_support_or_overlap: str
    unsealed_oracle_response: str
    no_kl_ablation_ignores_only_kl_thresholds: bool
    kg_tie_or_numerical_instability: str


@dataclass(frozen=True, slots=True)
class TerminalEvaluationContract:
    """Attainable recommendation and regret estimands."""

    candidate_domain: str
    point_estimate: str
    feasibility: str
    oracle_constraint_extension_at_runtime_allowed: bool
    candidate_truth_eligibility: str
    recommendation_tie_break: str
    minimum_eligible_candidates: int
    abstain_if: str
    regret_domain: str
    regret_oracle_utility: str
    regret: str
    empty_feasible_domain_oracle_utility: float
    abstention_regret_uses_no_action_utility: bool
    missing_censored_or_nonfinite_is_ineligible_not_imputed: bool


@dataclass(frozen=True, slots=True)
class PromotionGate:
    """One-way gate from research evidence to a bounded generator quota."""

    screen_requires_highest_mean_primary_metric_among_eight_methods: bool
    screen_requires_each_core_ablation_positive_pairs: int
    screen_requires_each_core_ablation_median_above_additive_margin: bool
    confirmation_requires_all_pairs_above_additive_margin: bool
    primary_missing_nonfinite_or_tied_required_gate_is_failure: bool
    full_method_kl_replay_numerical_integrity_result_required: bool
    yield_and_diversity_gate_comparators: str
    yield_and_diversity_pairwise_gate_scope: str
    max_valid_unique_reference_safe_yield_additive_loss: float
    max_hill2_effective_cluster_loss_fraction: float
    max_largest_cluster_share_additive_increase: float
    secondary_gate_comparison_absolute_tolerance: float
    valid_unique_reference_safe_yield_estimand: str
    yield_zero_charged_identity_value: str
    identity70_metric: str
    identity70_threshold: float
    identity70_linkage: str
    hill2_estimand: str
    hill2_zero_valid_sequence_value: float
    hill2_zero_comparator_gate: str
    largest_cluster_share_estimand: str
    largest_cluster_share_zero_valid_sequence_value: float
    yield_diversity_comparison_rule: str
    max_ece: float
    max_ece_additive_degradation: float
    calibration_gate_scope: str
    calibration_target: str
    ece_estimand: str
    ece_equal_mass_bin_rule: str
    ece_empty_or_incomplete_value: str
    coverage90_estimand: str
    coverage90_empty_or_incomplete_value: str
    calibration_comparison_rule: str
    calibration_per_seed_report_required: bool
    coverage90_lower: float
    coverage90_upper: float
    independent_chronological_or_assay_result_required: bool
    surrogate_screen_max_shadow_generator_mixture_quota: float
    surrogate_screen_max_production_generator_mixture_quota: float
    shadow_candidates_may_enter_submission_or_top100: bool
    production_requires_confirmation_and_independent_result: bool
    production_quota_requires_new_content_pinned_promotion_protocol: bool
    surrogate_pass_guarantees_final_library_or_top100_seat: bool
    no_go_on_failed_confirmation: bool
    screen_failure_is_no_go: bool
    unsequestered_confirmation_is_no_go: bool
    independent_result_missing_or_failure_is_no_go: bool
    any_required_gate_failure_is_no_go: bool
    passing_research_gates_authorizes_v1_production: bool


@dataclass(frozen=True, slots=True)
class TimingSegmentReceipt:
    """One externally authenticated link in a run's cumulative timing chain."""

    protocol_sha256: str
    run_id: str
    segment_index: int
    phase: Literal["scientific", "outer"]
    slurm_job_id: str
    node_id: str
    predecessor_receipt_sha256: str
    segment_receipt_sha256: str
    external_trusted_receipt_sha256: str
    segment_elapsed_nanoseconds: int
    cumulative_scientific_elapsed_nanoseconds: int
    cumulative_outer_elapsed_nanoseconds: int


@dataclass(frozen=True, slots=True)
class TimingBudgetDecision:
    """Fail-closed validation result for one initial/resumed Slurm timing chain."""

    passed: bool
    no_go: bool
    cumulative_scientific_elapsed_nanoseconds: int
    cumulative_outer_elapsed_nanoseconds: int
    terminal_receipt_sha256: str | None
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.no_go) is not bool:
            raise ValueError("timing decision flags must be exact booleans")
        if self.no_go is self.passed:
            raise ValueError("timing no-go must be the complement of passed")
        if self.passed and self.failures:
            raise ValueError("a passing timing decision cannot contain failures")
        if not self.passed and not self.failures:
            raise ValueError("a failed timing decision must explain its failure")
        if (
            type(self.cumulative_scientific_elapsed_nanoseconds) is not int
            or self.cumulative_scientific_elapsed_nanoseconds < 0
            or type(self.cumulative_outer_elapsed_nanoseconds) is not int
            or self.cumulative_outer_elapsed_nanoseconds < 0
            or type(self.failures) is not tuple
        ):
            raise ValueError("timing decision counters or failures have invalid types")


@dataclass(frozen=True, slots=True)
class FullMethodIntegrityEvidenceReceipt:
    """Trusted receipt binding the full method's integrity verdict to evidence."""

    protocol_sha256: str
    configuration_id: str
    screen_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    evidence_manifest_sha256: str
    timing_manifest_sha256: str
    external_trusted_receipt_sha256: str
    acceptance_status: Literal["accepted", "rejected"]
    verdict: Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class IndependentResultEvidenceReceipt:
    """Trusted receipt for a genuinely independent chronological/assay result."""

    protocol_sha256: str
    evidence_class: Literal["independent_chronological", "independent_assay"]
    subject_configuration_id: str
    panel_id: str
    result_manifest_sha256: str
    independence_declaration_sha256: str
    external_trusted_receipt_sha256: str
    acceptance_status: Literal["accepted", "rejected"]
    verdict: Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class ScreenGateDecision:
    """Deterministic descriptive-screen decision; never a p-value claim."""

    passed: bool
    no_go: bool
    full_unique_highest: bool
    ablation_passes: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool for value in (self.passed, self.no_go, self.full_unique_highest)
        ):
            raise ValueError("screen decision flags must be exact booleans")
        if self.no_go is self.passed:
            raise ValueError("screen no-go must be the complement of passed")
        valid_ablation_rows = type(self.ablation_passes) is tuple and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            for item in self.ablation_passes
        )
        if (
            not valid_ablation_rows
            or tuple(item[0] for item in self.ablation_passes) != ABLATION_IDS
        ):
            raise ValueError("screen decision must contain each exact ablation decision")
        if (
            type(self.failures) is not tuple
            or any(type(item) is not str or not item for item in self.failures)
            or len(set(self.failures)) != len(self.failures)
        ):
            raise ValueError("screen decision failures must be unique non-empty strings")
        mechanically_passed = (
            self.full_unique_highest
            and all(value for _, value in self.ablation_passes)
            and not self.failures
        )
        if self.passed is not mechanically_passed:
            raise ValueError("screen decision flags contradict its component evidence")


@dataclass(frozen=True, slots=True)
class ComparatorConfirmationDecision:
    """One fixed comparator's five-pair confirmation decision."""

    comparator_id: str
    successes: int
    exact_one_sided_p: float
    passed: bool

    def __post_init__(self) -> None:
        if self.comparator_id not in PRIMARY_COMPARATOR_IDS:
            raise ValueError("unknown primary comparator decision")
        if type(self.successes) is not int or not 0 <= self.successes <= len(CONFIRMATION_SEEDS):
            raise ValueError("confirmation successes are outside the frozen cohort")
        expected_p = _one_sided_exact_sign_p(
            successes=self.successes,
            pairs=len(CONFIRMATION_SEEDS),
        )
        if (
            isinstance(self.exact_one_sided_p, bool)
            or not isinstance(self.exact_one_sided_p, int | float)
            or not math.isclose(float(self.exact_one_sided_p), expected_p, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("confirmation p-value contradicts its success count")
        expected_pass = self.successes == len(CONFIRMATION_SEEDS) and expected_p <= 0.05
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ValueError("confirmation comparator flag contradicts its exact test")


@dataclass(frozen=True, slots=True)
class ConfirmationGateDecision:
    """Intersection-union decision across both frozen primary comparators."""

    development_check_passed: bool
    confirmatory_claim_allowed: bool
    passed: bool
    no_go: bool
    comparators: tuple[ComparatorConfirmationDecision, ...]
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.development_check_passed,
                self.confirmatory_claim_allowed,
                self.passed,
                self.no_go,
            )
        ):
            raise ValueError("confirmation decision flags must be exact booleans")
        if (
            type(self.comparators) is not tuple
            or any(type(item) is not ComparatorConfirmationDecision for item in self.comparators)
            or tuple(item.comparator_id for item in self.comparators) != PRIMARY_COMPARATOR_IDS
        ):
            raise ValueError("confirmation decision comparator order differs from the frozen IUT")
        if self.passed is not (self.development_check_passed and self.confirmatory_claim_allowed):
            raise ValueError("confirmation pass flag contradicts development/claim flags")
        if self.no_go is self.passed:
            raise ValueError("confirmation no-go must be the complement of passed")
        if self.development_check_passed and not all(item.passed for item in self.comparators):
            raise ValueError("confirmation development pass contradicts a comparator failure")
        if (
            type(self.failures) is not tuple
            or any(type(item) is not str or not item for item in self.failures)
            or len(set(self.failures)) != len(self.failures)
        ):
            raise ValueError("confirmation failures must be unique non-empty strings")


@dataclass(frozen=True, slots=True)
class PairedSecondaryGateMetrics:
    """Non-authoritative raw counts used only by the pure secondary gate math.

    This deliberately omits method, seed, run, and evidence identities.  It
    must be constructed internally from an exact confirmation cohort bound to
    a controller-supplied expected digest, never treated as an evidence or
    authorization type.  That trust input is not independent raw-byte replay.
    """

    full_valid_unique_count: int
    full_charged_submitted_identity_count: int
    full_identity70_cluster_sizes: tuple[int, ...]
    comparator_valid_unique_count: int
    comparator_charged_submitted_identity_count: int
    comparator_identity70_cluster_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalibrationSeedMetrics:
    """ECE and all four frozen marginal coverage values for one report unit."""

    ece: object
    coverage_by_level: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CalibrationGateMetrics:
    """Pooled and per-seed calibration reports for every confirmation method."""

    pooled: Mapping[str, CalibrationSeedMetrics]
    per_seed: Mapping[str, Mapping[int, CalibrationSeedMetrics]]


@dataclass(frozen=True, slots=True)
class ResearchGateDecision:
    """All-gate research decision; passing never authorizes v1 production."""

    development_checks_passed: bool
    passed: bool
    no_go: bool
    production_authorized: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.development_checks_passed,
                self.passed,
                self.no_go,
                self.production_authorized,
            )
        ):
            raise ValueError("research decision flags must be exact booleans")
        if self.no_go is self.passed:
            raise ValueError("research no-go must be the complement of passed")
        if self.passed and not self.development_checks_passed:
            raise ValueError("research pass requires all development checks")
        if self.production_authorized:
            raise ValueError("blocked v1 can never authorize production")
        if (
            type(self.failures) is not tuple
            or any(type(item) is not str or not item for item in self.failures)
            or len(set(self.failures)) != len(self.failures)
        ):
            raise ValueError("research failures must be unique non-empty strings")


@dataclass(frozen=True, slots=True)
class EvolutionaryKLProtocol:
    """Validated, outcome-blind research campaign contract."""

    artifact: str
    status: str
    automatic_production_eligible: bool
    execution_authorized: bool
    biological_superiority_claim_allowed: bool
    evidence_class: str
    confirmation_seed_status: str
    support_alphabet: str
    support_min_length: int
    support_max_length: int
    exact_training_overlap_forbidden: bool
    training_homology_exclusion_status: str
    generator_oracle_training_provenance_separation_status: str
    organizer_reference_validator_commit: str
    organizer_reference_similarity: str
    organizer_reference_max_similarity: float
    organizer_reference_may_shape_generation: bool
    organizer_reference_may_shape_model_fit_or_search_score: bool
    organizer_reference_compliance_may_veto_shadow_or_production: bool
    screen_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    screen_configuration_ids: tuple[str, ...]
    confirmation_method_ids: tuple[str, ...]
    methods: tuple[MethodSpec, ...]
    ablations: tuple[AblationSpec, ...]
    initial_design_unique_calls: int
    adaptive_batches: int
    unique_calls_per_batch: int
    method_controlled_seats_per_batch: int
    random_reserve_seats_per_batch: int
    total_unique_calls: int
    common_initial_responses_delivered_before_scientific_clock: bool
    common_initial_calls_charged_per_run: bool
    scheduled_common_reserve_scope: str
    scheduled_common_reserve_proposal_exclusion: str
    scheduled_common_reserve_submission_rule: str
    timed_cross_method_cache_release_policy: str
    method_controlled_duplicate_rule: str
    method_controlled_exhaustion_rule: str
    overflow_reserve_scope: str
    overflow_reserve_attempt_cap_per_batch: int
    overflow_reserve_collision_rule: str
    overflow_reserve_exhaustion_rule: str
    scientific_wall_seconds: int
    primary_metric: str
    primary_objectives: tuple[str, ...]
    objective_bounds: tuple[float, ...]
    hypervolume_reference: tuple[float, ...]
    hypervolume_auc_denominator_calls: int
    primary_auc_quadrature: str
    primary_checkpoint_input: str
    primary_checkpoint_values: str
    wall_checkpoint_minutes: tuple[int, ...]
    primary_value_source: str
    primary_archive_domain: str
    broad_spectrum_role: str
    pairwise_sequence_diversity_estimand: str
    pairwise_embedding_diversity_estimand: str
    pairwise_diversity_fewer_than_two_value: float
    pairwise_embedding_nonfinite_or_zero_norm_value: str
    ece_equal_mass_bins: int
    coverage_levels: tuple[float, ...]
    paired_bootstrap_samples: int
    paired_bootstrap_seed: int
    de_novo_paired_bootstrap_unit: str
    run_primary_missing_nonfinite_or_no_initial_checkpoint: str
    early_stop_primary_metric: str
    screen_effect: str
    screen_full_mean: str
    screen_full_unique_highest: str
    screen_comparison_absolute_tolerance: float
    screen_pair_success: str
    screen_positive_pairs_required: int
    screen_median_additive_margin: float
    screen_core_ablation_gate: str
    confirmation_effect: str
    confirmation_additive_materiality_margin: float
    confirmation_comparison_absolute_tolerance: float
    confirmation_pair_success: str
    confirmation_missing_nonfinite_or_tie: str
    confirmation_one_sided_alpha: float
    confirmation_exact_sign_test_minimum_p: float
    confirmation_pairs_required_above_margin: int
    confirmation_comparator_gate: str
    confirmation_global_iut_gate: str
    primary_comparator_ids: tuple[str, ...]
    screen_prerequisite_states: tuple[tuple[str, bool], ...]
    confirmatory_prerequisite_states: tuple[tuple[str, bool], ...]
    ablations_are_exploratory: bool
    configured_budget_summary: CampaignBudgetSummary
    cohort_contract: EvidenceCohortContract
    oracle_query_contract: OracleQueryContract
    resource_limits: SearchResourceLimits
    batching_limits: BatchingLimits
    statistical_reporting: StatisticalReportingContract
    stopping_rules: SearchStoppingRules
    terminal_evaluation: TerminalEvaluationContract
    promotion_gate: PromotionGate

    @property
    def call_checkpoints(self) -> tuple[int, ...]:
        """Unique-call checkpoints, including the common initial design."""

        return tuple(
            self.initial_design_unique_calls + index * self.unique_calls_per_batch
            for index in range(self.adaptive_batches + 1)
        )

    @property
    def execution_blockers(self) -> tuple[str, ...]:
        """Unaccepted prerequisites that currently prevent a screen campaign."""

        return tuple(name for name, accepted in self.screen_prerequisite_states if not accepted)

    @property
    def confirmatory_claim_blockers(self) -> tuple[str, ...]:
        """Unaccepted controls that prohibit confirmatory or promotion claims."""

        return tuple(
            name for name, accepted in self.confirmatory_prerequisite_states if not accepted
        )

    @property
    def budget_summary(self) -> CampaignBudgetSummary:
        """Recompute campaign totals without trusting declared aggregate fields."""

        screen_runs = len(self.screen_configuration_ids) * len(self.screen_seeds)
        confirmation_runs = len(self.confirmation_method_ids) * len(self.confirmation_seeds)
        scientific_runs = screen_runs + confirmation_runs
        hours_per_run = self.scientific_wall_seconds / 3600.0
        return CampaignBudgetSummary(
            screen_runs=screen_runs,
            confirmation_runs=confirmation_runs,
            scientific_runs=scientific_runs,
            screen_logical_unique_calls=screen_runs * self.total_unique_calls,
            confirmation_logical_unique_calls=(confirmation_runs * self.total_unique_calls),
            scientific_logical_unique_calls=scientific_runs * self.total_unique_calls,
            screen_a100_hour_ceiling=screen_runs * hours_per_run,
            confirmation_a100_hour_ceiling=confirmation_runs * hours_per_run,
            scientific_a100_hour_ceiling=scientific_runs * hours_per_run,
        )

    def runs(self, phase: Literal["screen", "confirmation"]) -> tuple[CampaignRun, ...]:
        """Return deterministic seed blocks with hash-randomized simultaneous waves."""

        if phase == "screen":
            configuration_ids = self.screen_configuration_ids
            seeds = self.screen_seeds
        elif phase == "confirmation":
            configuration_ids = self.confirmation_method_ids
            seeds = self.confirmation_seeds
        else:  # pragma: no cover - guarded by the Literal type for typed callers
            raise ValueError(f"unsupported campaign phase: {phase!r}")
        schedule_seed = self.resource_limits.scientific_schedule_seed
        wave_size = self.resource_limits.scientific_schedule_wave_size
        planned: list[CampaignRun] = []
        for block, seed in enumerate(seeds):
            ordered = sorted(
                configuration_ids,
                key=lambda configuration_id: (
                    hashlib.sha256(
                        (f"{schedule_seed}\0{phase}\0{seed}\0{configuration_id}").encode()
                    ).digest(),
                    configuration_id,
                ),
            )
            for position, configuration_id in enumerate(ordered):
                planned.append(
                    CampaignRun(
                        ordinal=len(planned),
                        phase=phase,
                        configuration_id=configuration_id,
                        seed=seed,
                        unique_oracle_call_budget=self.total_unique_calls,
                        scientific_wall_seconds=self.scientific_wall_seconds,
                        schedule_block=block,
                        schedule_wave=position // wave_size,
                        schedule_slot=position % wave_size,
                    )
                )
        return tuple(planned)

    def require_screen_authority(self) -> None:
        """Always reject execution from the immutable, blocked v1 document."""

        blockers = ", ".join(self.execution_blockers)
        raise RuntimeError(
            "blocked evolutionary/KL v1 cannot authorize execution; freeze a new "
            f"version after independent prerequisite acceptance (current blockers: {blockers})"
        )


def normalized_primary_hypervolume_auc(
    protocol: EvolutionaryKLProtocol,
    sealed_hypervolume_by_checkpoint: Mapping[int, object],
) -> float | None:
    """Compute the frozen primary summary, carrying a sealed prefix after stop.

    The input must contain a nonempty contiguous prefix of the 29 fixed call
    checkpoints, starting at 64. Values are authenticated incumbent
    hypervolumes, so Boolean, non-finite, out-of-range, or decreasing values
    fail closed. A valid prefix is extended with its last value and integrated
    by the frozen trapezoidal rule.
    """

    if not isinstance(sealed_hypervolume_by_checkpoint, Mapping):
        return None
    keys = tuple(sealed_hypervolume_by_checkpoint)
    if not keys or any(type(key) is not int for key in keys):
        return None
    checkpoints = protocol.call_checkpoints
    checkpoint_positions = {checkpoint: index for index, checkpoint in enumerate(checkpoints)}
    if any(key not in checkpoint_positions for key in keys):
        return None
    last_position = max(checkpoint_positions[key] for key in keys)
    required_prefix = checkpoints[: last_position + 1]
    if set(keys) != set(required_prefix) or len(keys) != len(required_prefix):
        return None

    values: list[float] = []
    for checkpoint in required_prefix:
        value = _finite_unit_interval(sealed_hypervolume_by_checkpoint[checkpoint])
        if value is None or (values and value < values[-1]):
            return None
        values.append(value)
    values.extend([values[-1]] * (len(checkpoints) - len(values)))
    area = sum(
        (right_checkpoint - left_checkpoint) * (left_value + right_value) / 2.0
        for left_checkpoint, right_checkpoint, left_value, right_value in zip(
            checkpoints[:-1],
            checkpoints[1:],
            values[:-1],
            values[1:],
            strict=True,
        )
    )
    return area / protocol.hypervolume_auc_denominator_calls


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_timing_budget(
    protocol: EvolutionaryKLProtocol,
    receipts: tuple[TimingSegmentReceipt, ...],
    *,
    expected_external_trusted_receipt_sha256s: tuple[str, ...],
) -> TimingBudgetDecision:
    """Validate one complete, externally authenticated initial/resume timing chain.

    Segment elapsed time is integer nanoseconds. Each resume carries both
    cumulative clocks, so neither the scientific cap nor the CPU-only sealing
    allowance can restart with a new Slurm job.
    """

    failures: list[str] = []
    if not isinstance(receipts, tuple) or not receipts:
        return TimingBudgetDecision(
            passed=False,
            no_go=True,
            cumulative_scientific_elapsed_nanoseconds=0,
            cumulative_outer_elapsed_nanoseconds=0,
            terminal_receipt_sha256=None,
            failures=("timing receipt chain must be a non-empty tuple",),
        )
    if (
        not isinstance(expected_external_trusted_receipt_sha256s, tuple)
        or len(expected_external_trusted_receipt_sha256s) != len(receipts)
        or any(not _is_sha256(digest) for digest in expected_external_trusted_receipt_sha256s)
    ):
        failures.append("authoritative timing trusted-receipt inventory is invalid")
    run_id: str | None = None
    predecessor = "0" * 64
    computed_scientific = 0
    computed_outer = 0
    outer_started = False
    seen_job_ids: set[str] = set()
    seen_segment_receipts: set[str] = set()
    terminal_receipt: str | None = None
    for expected_index, receipt in enumerate(receipts):
        if type(receipt) is not TimingSegmentReceipt:
            failures.append(f"timing segment {expected_index} has the wrong receipt type")
            continue
        if receipt.protocol_sha256 != FROZEN_PROTOCOL_SHA256:
            failures.append(f"timing segment {expected_index} binds the wrong protocol")
        if not receipt.run_id:
            failures.append(f"timing segment {expected_index} has an empty run ID")
        elif run_id is None:
            run_id = receipt.run_id
        elif receipt.run_id != run_id:
            failures.append(f"timing segment {expected_index} changes run ID")
        if type(receipt.segment_index) is not int or receipt.segment_index != expected_index:
            failures.append(f"timing segment {expected_index} has a non-contiguous index")
        if receipt.phase not in ("scientific", "outer"):
            failures.append(f"timing segment {expected_index} has an invalid phase")
        if not receipt.slurm_job_id or receipt.slurm_job_id in seen_job_ids:
            failures.append(f"timing segment {expected_index} has an empty or reused Slurm job ID")
        else:
            seen_job_ids.add(receipt.slurm_job_id)
        if not receipt.node_id:
            failures.append(f"timing segment {expected_index} has an empty node ID")
        for field_name, digest in (
            ("predecessor", receipt.predecessor_receipt_sha256),
            ("segment", receipt.segment_receipt_sha256),
            ("external trusted", receipt.external_trusted_receipt_sha256),
        ):
            if not _is_sha256(digest):
                failures.append(
                    f"timing segment {expected_index} has an invalid {field_name} receipt digest"
                )
        if (
            expected_index >= len(expected_external_trusted_receipt_sha256s)
            or receipt.external_trusted_receipt_sha256
            != expected_external_trusted_receipt_sha256s[expected_index]
        ):
            failures.append(
                f"timing segment {expected_index} does not match the authoritative trusted receipt"
            )
        if receipt.predecessor_receipt_sha256 != predecessor:
            failures.append(f"timing segment {expected_index} breaks the receipt chain")
        if receipt.segment_receipt_sha256 in seen_segment_receipts:
            failures.append(f"timing segment {expected_index} reuses a segment receipt")
        seen_segment_receipts.add(receipt.segment_receipt_sha256)
        terminal_receipt = receipt.segment_receipt_sha256
        predecessor = receipt.segment_receipt_sha256
        if (
            type(receipt.segment_elapsed_nanoseconds) is not int
            or receipt.segment_elapsed_nanoseconds < 0
            or type(receipt.cumulative_scientific_elapsed_nanoseconds) is not int
            or receipt.cumulative_scientific_elapsed_nanoseconds < 0
            or type(receipt.cumulative_outer_elapsed_nanoseconds) is not int
            or receipt.cumulative_outer_elapsed_nanoseconds < 0
        ):
            failures.append(f"timing segment {expected_index} has invalid nanosecond counters")
            continue
        if receipt.phase == "scientific":
            if outer_started:
                failures.append("scientific timing cannot resume after the outer allowance starts")
            computed_scientific += receipt.segment_elapsed_nanoseconds
        else:
            outer_started = True
            computed_outer += receipt.segment_elapsed_nanoseconds
        if (
            receipt.cumulative_scientific_elapsed_nanoseconds != computed_scientific
            or receipt.cumulative_outer_elapsed_nanoseconds != computed_outer
        ):
            failures.append(
                f"timing segment {expected_index} resets or misstates a cumulative clock"
            )
    if receipts and type(receipts[0]) is TimingSegmentReceipt and receipts[0].phase != "scientific":
        failures.append("the timing chain must begin with a scientific segment")
    scientific_cap = protocol.scientific_wall_seconds * 1_000_000_000
    outer_cap = protocol.resource_limits.outer_allowance_seconds * 1_000_000_000
    if computed_scientific > scientific_cap:
        failures.append("cumulative scientific elapsed time exceeds the frozen cap")
    if computed_outer > outer_cap:
        failures.append("cumulative outer elapsed time exceeds the frozen allowance")
    unique_failures = tuple(dict.fromkeys(failures))
    return TimingBudgetDecision(
        passed=not unique_failures,
        no_go=bool(unique_failures),
        cumulative_scientific_elapsed_nanoseconds=computed_scientific,
        cumulative_outer_elapsed_nanoseconds=computed_outer,
        terminal_receipt_sha256=terminal_receipt,
        failures=unique_failures,
    )


def confirmation_pair_is_success(
    protocol: EvolutionaryKLProtocol,
    *,
    full_metric: object,
    comparator_metric: object,
) -> bool:
    """Apply the frozen strict additive-margin rule to one paired seed.

    Missing, non-numeric, Boolean, non-finite, out-of-range, tied, and
    within-tolerance values all fail closed and never count as sign-test
    successes.
    """

    values: list[float] = []
    for value in (full_metric, comparator_metric):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            return False
        values.append(parsed)
    effect = values[0] - values[1]
    threshold = (
        protocol.confirmation_additive_materiality_margin
        + protocol.confirmation_comparison_absolute_tolerance
    )
    return effect > threshold


def _finite_unit_interval(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _yield_diversity_estimands(
    protocol: EvolutionaryKLProtocol,
    *,
    valid_unique_count: object,
    charged_submitted_identity_count: object,
    identity70_cluster_sizes: object,
) -> tuple[float, float, float] | None:
    if (
        type(valid_unique_count) is not int
        or type(charged_submitted_identity_count) is not int
        or valid_unique_count < 0
        or charged_submitted_identity_count < protocol.initial_design_unique_calls
        or charged_submitted_identity_count > protocol.total_unique_calls
        or valid_unique_count > charged_submitted_identity_count
        or not isinstance(identity70_cluster_sizes, tuple)
        or any(type(size) is not int or size <= 0 for size in identity70_cluster_sizes)
        or sum(identity70_cluster_sizes) != valid_unique_count
    ):
        return None
    yield_value = valid_unique_count / charged_submitted_identity_count
    if valid_unique_count == 0:
        return yield_value, 0.0, 0.0
    proportions = tuple(size / valid_unique_count for size in identity70_cluster_sizes)
    hill2 = 1.0 / sum(proportion * proportion for proportion in proportions)
    largest_share = max(identity70_cluster_sizes) / valid_unique_count
    return yield_value, hill2, largest_share


def _cohort_metric_vector(
    metrics: Mapping[str, Mapping[int, object]],
    *,
    configuration_id: str,
    seeds: tuple[int, ...],
    failures: list[str],
) -> tuple[float, ...] | None:
    by_seed = metrics.get(configuration_id)
    if not isinstance(by_seed, Mapping):
        failures.append(f"missing metrics for {configuration_id}")
        return None
    if any(type(seed) is not int for seed in by_seed) or set(by_seed) != set(seeds):
        failures.append(f"seed keys differ for {configuration_id}")
        return None
    parsed: list[float] = []
    for seed in seeds:
        value = _finite_unit_interval(by_seed.get(seed))
        if value is None:
            failures.append(f"invalid primary metric for {configuration_id} seed {seed}")
        else:
            parsed.append(value)
    if len(parsed) != len(seeds):
        return None
    return tuple(parsed)


def screen_gate_decision(
    protocol: EvolutionaryKLProtocol,
    metrics: Mapping[str, Mapping[int, object]],
) -> ScreenGateDecision:
    """Apply the frozen descriptive screen and five paired ablation gates."""

    if not isinstance(metrics, Mapping):
        return ScreenGateDecision(
            passed=False,
            no_go=True,
            full_unique_highest=False,
            ablation_passes=tuple((ablation_id, False) for ablation_id in ABLATION_IDS),
            failures=("screen metrics must be a mapping",),
        )
    failures: list[str] = []
    if any(type(key) is not str for key in metrics) or set(metrics) != set(
        protocol.screen_configuration_ids
    ):
        failures.append("screen configuration keys differ from the frozen cohort")
    vectors = {
        configuration_id: _cohort_metric_vector(
            metrics,
            configuration_id=configuration_id,
            seeds=protocol.screen_seeds,
            failures=failures,
        )
        for configuration_id in protocol.screen_configuration_ids
    }
    full_id = "counterfactual_softkg_evolutionary_diffusion"
    full = vectors[full_id]
    tolerance = protocol.screen_comparison_absolute_tolerance
    full_unique_highest = full is not None
    if full is not None:
        full_mean = sum(full) / len(full)
        for method_id in METHOD_IDS:
            if method_id == full_id:
                continue
            comparator = vectors[method_id]
            if comparator is None:
                full_unique_highest = False
                continue
            comparator_mean = sum(comparator) / len(comparator)
            if not full_mean > comparator_mean + tolerance:
                full_unique_highest = False
                failures.append(f"full method is not uniquely highest versus {method_id}")
    if not full_unique_highest and not any(
        message.startswith("full method is not uniquely highest") for message in failures
    ):
        failures.append("full-method screen mean is unavailable")

    ablation_passes: list[tuple[str, bool]] = []
    for ablation_id in ABLATION_IDS:
        ablation = vectors[ablation_id]
        passed = False
        if full is not None and ablation is not None:
            effects = tuple(left - right for left, right in zip(full, ablation, strict=True))
            positive = sum(effect > tolerance for effect in effects)
            median = sorted(effects)[len(effects) // 2]
            passed = (
                positive >= protocol.screen_positive_pairs_required
                and median > protocol.screen_median_additive_margin + tolerance
            )
        if not passed:
            failures.append(f"core ablation gate failed for {ablation_id}")
        ablation_passes.append((ablation_id, passed))

    passed = full_unique_highest and all(value for _, value in ablation_passes) and not failures
    return ScreenGateDecision(
        passed=passed,
        no_go=not passed,
        full_unique_highest=full_unique_highest,
        ablation_passes=tuple(ablation_passes),
        failures=tuple(dict.fromkeys(failures)),
    )


def _one_sided_exact_sign_p(*, successes: int, pairs: int) -> float:
    return sum(math.comb(pairs, count) for count in range(successes, pairs + 1)) / (2**pairs)


def confirmation_gate_decision(
    protocol: EvolutionaryKLProtocol,
    metrics: Mapping[str, Mapping[int, object]],
) -> ConfirmationGateDecision:
    """Apply both fixed five-pair tests and their intersection-union gate."""

    if not isinstance(metrics, Mapping):
        pair_count = len(protocol.confirmation_seeds)
        return ConfirmationGateDecision(
            development_check_passed=False,
            confirmatory_claim_allowed=False,
            passed=False,
            no_go=True,
            comparators=tuple(
                ComparatorConfirmationDecision(
                    comparator_id=comparator_id,
                    successes=0,
                    exact_one_sided_p=_one_sided_exact_sign_p(
                        successes=0,
                        pairs=pair_count,
                    ),
                    passed=False,
                )
                for comparator_id in protocol.primary_comparator_ids
            ),
            failures=("confirmation metrics must be a mapping",),
        )
    failures: list[str] = []
    if any(type(key) is not str for key in metrics) or set(metrics) != set(
        protocol.confirmation_method_ids
    ):
        failures.append("confirmation method keys differ from the frozen cohort")
    vectors = {
        method_id: _cohort_metric_vector(
            metrics,
            configuration_id=method_id,
            seeds=protocol.confirmation_seeds,
            failures=failures,
        )
        for method_id in protocol.confirmation_method_ids
    }
    full = vectors["counterfactual_softkg_evolutionary_diffusion"]
    decisions: list[ComparatorConfirmationDecision] = []
    pair_count = len(protocol.confirmation_seeds)
    for comparator_id in protocol.primary_comparator_ids:
        comparator = vectors[comparator_id]
        successes = 0
        if full is not None and comparator is not None:
            successes = sum(
                confirmation_pair_is_success(
                    protocol,
                    full_metric=left,
                    comparator_metric=right,
                )
                for left, right in zip(full, comparator, strict=True)
            )
        p_value = _one_sided_exact_sign_p(successes=successes, pairs=pair_count)
        passed = (
            successes == protocol.confirmation_pairs_required_above_margin
            and p_value <= protocol.confirmation_one_sided_alpha
        )
        if not passed:
            failures.append(f"confirmation gate failed versus {comparator_id}")
        decisions.append(
            ComparatorConfirmationDecision(
                comparator_id=comparator_id,
                successes=successes,
                exact_one_sided_p=p_value,
                passed=passed,
            )
        )
    development_check_passed = all(decision.passed for decision in decisions) and not any(
        message.startswith("invalid primary metric")
        or message.startswith("missing metrics")
        or message.startswith("seed keys differ")
        or message.startswith("confirmation method keys differ")
        for message in failures
    )
    cohort = protocol.cohort_contract
    confirmatory_claim_allowed = (
        cohort.published_confirmation_seeds_may_support_confirmatory_or_promotion_claim
        and not protocol.execution_blockers
        and not protocol.confirmatory_claim_blockers
    )
    passed = development_check_passed and confirmatory_claim_allowed
    if development_check_passed and not confirmatory_claim_allowed:
        failures.append("published confirmation seeds are unsequestered development-only")
    return ConfirmationGateDecision(
        development_check_passed=development_check_passed,
        confirmatory_claim_allowed=confirmatory_claim_allowed,
        passed=passed,
        no_go=not passed,
        comparators=tuple(decisions),
        failures=tuple(dict.fromkeys(failures)),
    )


def _calibration_values(
    protocol: EvolutionaryKLProtocol,
    row: object,
) -> tuple[float, tuple[float, ...]] | None:
    if type(row) is not CalibrationSeedMetrics or not isinstance(row.coverage_by_level, tuple):
        return None
    if len(row.coverage_by_level) != len(protocol.coverage_levels):
        return None
    ece = _finite_unit_interval(row.ece)
    coverage = tuple(_finite_unit_interval(value) for value in row.coverage_by_level)
    if ece is None or any(value is None for value in coverage):
        return None
    return ece, tuple(value for value in coverage if value is not None)


def _accepted_full_method_integrity_evidence(
    protocol: EvolutionaryKLProtocol,
    evidence: object,
    *,
    expected_external_trusted_receipt_sha256: object,
) -> bool:
    return (
        type(evidence) is FullMethodIntegrityEvidenceReceipt
        and _is_sha256(expected_external_trusted_receipt_sha256)
        and evidence.external_trusted_receipt_sha256 == expected_external_trusted_receipt_sha256
        and evidence.protocol_sha256 == FROZEN_PROTOCOL_SHA256
        and evidence.configuration_id == "counterfactual_softkg_evolutionary_diffusion"
        and evidence.screen_seeds == protocol.screen_seeds
        and evidence.confirmation_seeds == protocol.confirmation_seeds
        and _is_sha256(evidence.evidence_manifest_sha256)
        and _is_sha256(evidence.timing_manifest_sha256)
        and evidence.acceptance_status == "accepted"
        and evidence.verdict == "pass"
    )


def _accepted_independent_result_evidence(
    evidence: object,
    *,
    expected_external_trusted_receipt_sha256: object,
) -> bool:
    return (
        type(evidence) is IndependentResultEvidenceReceipt
        and _is_sha256(expected_external_trusted_receipt_sha256)
        and evidence.external_trusted_receipt_sha256 == expected_external_trusted_receipt_sha256
        and evidence.protocol_sha256 == FROZEN_PROTOCOL_SHA256
        and evidence.evidence_class in ("independent_chronological", "independent_assay")
        and evidence.subject_configuration_id == "counterfactual_softkg_evolutionary_diffusion"
        and isinstance(evidence.panel_id, str)
        and bool(evidence.panel_id.strip())
        and _is_sha256(evidence.result_manifest_sha256)
        and _is_sha256(evidence.independence_declaration_sha256)
        and evidence.acceptance_status == "accepted"
        and evidence.verdict == "pass"
    )


def research_gate_decision(
    protocol: EvolutionaryKLProtocol,
    *,
    screen_metrics: Mapping[str, Mapping[int, object]],
    confirmation_metrics: Mapping[str, Mapping[int, object]],
    secondary: Mapping[str, Mapping[int, PairedSecondaryGateMetrics]],
    calibration: CalibrationGateMetrics,
    full_method_integrity_evidence: FullMethodIntegrityEvidenceReceipt | None,
    expected_integrity_trusted_receipt_sha256: str | None,
    independent_result_evidence: IndependentResultEvidenceReceipt | None,
    expected_independent_trusted_receipt_sha256: str | None,
) -> ResearchGateDecision:
    """Recompute raw primary gates and apply every frozen pure-math gate.

    Consuming raw metric maps prevents a caller from injecting a constructible
    passing decision object. Integrity and independent-result verdicts must bind
    to separately supplied, controller-authoritative trusted-receipt digests.

    The ``secondary`` mapping is a legacy, identity-free arithmetic interface;
    calling this function directly is not authenticated secondary evidence.
    Evidence-bearing consumers must use
    ``research_gate_decision_from_authenticated_secondary_cohort`` from the
    secondary-evidence module, which derives this mapping internally.
    """

    screen = screen_gate_decision(protocol, screen_metrics)
    confirmation = confirmation_gate_decision(protocol, confirmation_metrics)
    if type(calibration) is not CalibrationGateMetrics:
        return ResearchGateDecision(
            development_checks_passed=False,
            passed=False,
            no_go=True,
            production_authorized=False,
            failures=("calibration input must use the frozen type",),
        )
    failures: list[str] = []
    if not screen.passed:
        failures.append("screen gate failed")
    if not confirmation.development_check_passed:
        failures.append("confirmation intersection-union gate failed")
    if not _accepted_full_method_integrity_evidence(
        protocol,
        full_method_integrity_evidence,
        expected_external_trusted_receipt_sha256=(expected_integrity_trusted_receipt_sha256),
    ):
        failures.append(
            "full-method KL, replay, numerical, and timing integrity evidence is unaccepted"
        )

    gate = protocol.promotion_gate
    tolerance = gate.secondary_gate_comparison_absolute_tolerance
    if not isinstance(secondary, Mapping):
        secondary = {}
        failures.append("secondary metrics must be a mapping")
    if any(type(key) is not str for key in secondary) or set(secondary) != set(
        protocol.primary_comparator_ids
    ):
        failures.append("secondary comparator keys differ from the frozen gate")
    full_rows_by_seed: dict[int, tuple[object, ...]] = {}
    for comparator_id in protocol.primary_comparator_ids:
        by_seed = secondary.get(comparator_id)
        if (
            not isinstance(by_seed, Mapping)
            or any(type(seed) is not int for seed in by_seed)
            or set(by_seed) != set(protocol.confirmation_seeds)
        ):
            failures.append(f"secondary seed keys differ for {comparator_id}")
            continue
        for seed in protocol.confirmation_seeds:
            row = by_seed.get(seed)
            if type(row) is not PairedSecondaryGateMetrics:
                failures.append(f"missing secondary metrics for {comparator_id} seed {seed}")
                continue
            full = _yield_diversity_estimands(
                protocol,
                valid_unique_count=row.full_valid_unique_count,
                charged_submitted_identity_count=row.full_charged_submitted_identity_count,
                identity70_cluster_sizes=row.full_identity70_cluster_sizes,
            )
            comparator = _yield_diversity_estimands(
                protocol,
                valid_unique_count=row.comparator_valid_unique_count,
                charged_submitted_identity_count=row.comparator_charged_submitted_identity_count,
                identity70_cluster_sizes=row.comparator_identity70_cluster_sizes,
            )
            if full is None or comparator is None:
                failures.append(f"invalid secondary metrics for {comparator_id} seed {seed}")
                continue
            full_row = (
                row.full_valid_unique_count,
                row.full_charged_submitted_identity_count,
                row.full_identity70_cluster_sizes,
            )
            if seed in full_rows_by_seed and full_rows_by_seed[seed] != full_row:
                failures.append(f"full secondary row differs across comparators for seed {seed}")
            else:
                full_rows_by_seed[seed] = full_row
            full_yield, full_hill2, full_largest = full
            comparator_yield, comparator_hill2, comparator_largest = comparator
            if (
                full_yield + gate.max_valid_unique_reference_safe_yield_additive_loss + tolerance
                < comparator_yield
            ):
                failures.append(f"yield gate failed for {comparator_id} seed {seed}")
            hill2_floor = comparator_hill2 * (1.0 - gate.max_hill2_effective_cluster_loss_fraction)
            if full_hill2 + tolerance < hill2_floor:
                failures.append(f"Hill-2 gate failed for {comparator_id} seed {seed}")
            if full_largest > (
                comparator_largest + gate.max_largest_cluster_share_additive_increase + tolerance
            ):
                failures.append(f"largest-cluster gate failed for {comparator_id} seed {seed}")

    pooled_input = calibration.pooled
    if not isinstance(pooled_input, Mapping):
        pooled_input = {}
    if any(type(key) is not str for key in pooled_input) or set(pooled_input) != set(
        protocol.confirmation_method_ids
    ):
        failures.append("pooled calibration method keys differ from the frozen cohort")
    pooled = {
        method_id: _calibration_values(protocol, pooled_input.get(method_id))
        for method_id in protocol.confirmation_method_ids
    }
    if any(value is None for value in pooled.values()):
        failures.append("missing or invalid pooled calibration report")
    else:
        full_report = pooled["counterfactual_softkg_evolutionary_diffusion"]
        assert full_report is not None
        full_ece, full_coverages = full_report
        coverage90_index = protocol.coverage_levels.index(0.90)
        full_coverage90 = full_coverages[coverage90_index]
        if full_ece > gate.max_ece + tolerance:
            failures.append("absolute ECE gate failed")
        for comparator_id in protocol.primary_comparator_ids:
            comparator_report = pooled[comparator_id]
            assert comparator_report is not None
            comparator_ece = comparator_report[0]
            if full_ece > comparator_ece + gate.max_ece_additive_degradation + tolerance:
                failures.append(f"ECE degradation gate failed versus {comparator_id}")
        if not (
            gate.coverage90_lower - tolerance
            <= full_coverage90
            <= gate.coverage90_upper + tolerance
        ):
            failures.append("90-percent coverage gate failed")
    per_seed_input = calibration.per_seed
    if not isinstance(per_seed_input, Mapping):
        per_seed_input = {}
    if any(type(key) is not str for key in per_seed_input) or set(per_seed_input) != set(
        protocol.confirmation_method_ids
    ):
        failures.append("per-seed calibration method keys differ from the frozen cohort")
    for method_id in protocol.confirmation_method_ids:
        by_seed = per_seed_input.get(method_id)
        if (
            not isinstance(by_seed, Mapping)
            or any(type(seed) is not int for seed in by_seed)
            or set(by_seed) != set(protocol.confirmation_seeds)
        ):
            failures.append(f"per-seed calibration keys differ for {method_id}")
            continue
        for seed in protocol.confirmation_seeds:
            if _calibration_values(protocol, by_seed.get(seed)) is None:
                failures.append(
                    f"missing or invalid per-seed calibration for {method_id} seed {seed}"
                )
    if not _accepted_independent_result_evidence(
        independent_result_evidence,
        expected_external_trusted_receipt_sha256=(expected_independent_trusted_receipt_sha256),
    ):
        failures.append(
            "independent chronological or assay evidence is missing, unaccepted, or failed"
        )

    development_checks_passed = not failures
    cohort = protocol.cohort_contract
    if not cohort.published_confirmation_seeds_may_support_confirmatory_or_promotion_claim:
        failures.append(
            "published v1 confirmation seeds are unsequestered and cannot support promotion"
        )
    if protocol.execution_blockers:
        failures.append("screen execution prerequisites are unaccepted")
    if protocol.confirmatory_claim_blockers:
        failures.append("hidden confirmation seed and oracle controls are unaccepted")
    passed = not failures
    return ResearchGateDecision(
        development_checks_passed=development_checks_passed,
        passed=passed,
        no_go=not passed,
        production_authorized=False,
        failures=tuple(dict.fromkeys(failures)),
    )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table")
    return value


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, Any]:
    mapping = _mapping(value, name=name)
    if set(mapping) != keys:
        missing = sorted(keys - set(mapping))
        unexpected = sorted(set(mapping) - keys)
        raise ValueError(
            f"{name} keys differ from the frozen schema; missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_exact(value: object, expected: object, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} differs from the frozen v1 contract")


def _require_frozen_fields(
    mapping: Mapping[str, Any],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    for key, value in expected.items():
        _require_exact(mapping.get(key), value, name=f"{name}.{key}")


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _integer(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer greater than or equal to {minimum}")
    return value


def _number(mapping: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{key} must be finite and at least {minimum}")
    return parsed


def _string_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array")
    parsed = tuple(value)
    if any(not isinstance(item, str) or not item for item in parsed):
        raise ValueError(f"{key} must contain non-empty strings")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{key} must be unique")
    return parsed


def _number_tuple(mapping: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty numeric array")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{key} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{key} must contain only finite numbers")
        parsed.append(number)
    return tuple(parsed)


def _integer_tuple(mapping: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty integer array")
    parsed = tuple(value)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in parsed):
        raise ValueError(f"{key} must contain only integers")
    return parsed


def _seed_tuple(mapping: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array")
    parsed = tuple(value)
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in parsed):
        raise ValueError(f"{key} must contain non-negative integers")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{key} must be unique")
    return parsed


def _method_specs(raw: object) -> tuple[MethodSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("methods must be a non-empty array of tables")
    parsed: list[MethodSpec] = []
    sourced = {
        "arcadiamp_style_iterative_d3pm",
        "tr2d2_style_tree_offpolicy",
        "mp2d_style_inference_search",
    }
    for index, item in enumerate(raw):
        preliminary = _mapping(item, name=f"method {index}")
        method_id = _string(preliminary, "id")
        keys = {
            "id",
            "role",
            "implementation",
            "requires_diffusion",
            "adaptive_querying",
            "uses_endpoint_distillation",
        }
        if method_id in sourced:
            keys.add("source")
        method = _exact_mapping(preliminary, name=f"method {index}", keys=keys)
        if method_id in sourced:
            _string(method, "source")
        parsed.append(
            MethodSpec(
                method_id=method_id,
                role=_string(method, "role"),
                implementation=_string(method, "implementation"),
                requires_diffusion=_boolean(method, "requires_diffusion"),
                adaptive_querying=_boolean(method, "adaptive_querying"),
                uses_endpoint_distillation=_boolean(method, "uses_endpoint_distillation"),
            )
        )
    methods = tuple(parsed)
    if len({method.method_id for method in methods}) != len(methods):
        raise ValueError("method IDs must be unique")
    return methods


def _ablation_specs(raw: object) -> tuple[AblationSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("ablations must be a non-empty array of tables")
    parsed: list[AblationSpec] = []
    for index, item in enumerate(raw):
        ablation = _exact_mapping(
            item,
            name=f"ablation {index}",
            keys={"id", "base_method", "change"},
        )
        parsed.append(
            AblationSpec(
                ablation_id=_string(ablation, "id"),
                base_method=_string(ablation, "base_method"),
                change=_string(ablation, "change"),
            )
        )
    ablations = tuple(parsed)
    if len({ablation.ablation_id for ablation in ablations}) != len(ablations):
        raise ValueError("ablation IDs must be unique")
    return ablations


def _budget_summary(resources: Mapping[str, Any]) -> CampaignBudgetSummary:
    return CampaignBudgetSummary(
        screen_runs=_integer(resources, "screen_runs", minimum=1),
        confirmation_runs=_integer(resources, "confirmation_runs", minimum=1),
        scientific_runs=_integer(resources, "scientific_runs", minimum=1),
        screen_logical_unique_calls=_integer(resources, "screen_logical_unique_calls", minimum=1),
        confirmation_logical_unique_calls=_integer(
            resources, "confirmation_logical_unique_calls", minimum=1
        ),
        scientific_logical_unique_calls=_integer(
            resources, "scientific_logical_unique_calls", minimum=1
        ),
        screen_a100_hour_ceiling=_number(resources, "screen_a100_hour_ceiling"),
        confirmation_a100_hour_ceiling=_number(resources, "confirmation_a100_hour_ceiling"),
        scientific_a100_hour_ceiling=_number(resources, "scientific_a100_hour_ceiling"),
    )


def _cohort_contract(cohorts: Mapping[str, Any]) -> EvidenceCohortContract:
    return EvidenceCohortContract(
        screen_unit=_string(cohorts, "screen_unit"),
        confirmation_unit=_string(cohorts, "confirmation_unit"),
        de_novo_rotation_dimension=_string(cohorts, "de_novo_rotation_dimension"),
        de_novo_rotation_count=_integer(cohorts, "de_novo_rotation_count", minimum=0),
        de_novo_checkpoint_aggregation_status=_string(
            cohorts, "de_novo_checkpoint_aggregation_status"
        ),
        retrospective_fixed_pool_protocol=_string(cohorts, "retrospective_fixed_pool_protocol"),
        retrospective_fixed_pool_rotations=_integer(
            cohorts, "retrospective_fixed_pool_rotations", minimum=1
        ),
        retrospective_results_may_enter_de_novo_screen_or_confirmation=_boolean(
            cohorts,
            "retrospective_results_may_enter_de_novo_screen_or_confirmation",
        ),
        retrospective_calls_in_de_novo_budget=_integer(
            cohorts, "retrospective_calls_in_de_novo_budget", minimum=0
        ),
        published_confirmation_seeds_may_support_confirmatory_or_promotion_claim=(
            _boolean(
                cohorts,
                "published_confirmation_seeds_may_support_confirmatory_or_promotion_claim",
            )
        ),
        successor_hidden_confirmation_seed_count=_integer(
            cohorts, "successor_hidden_confirmation_seed_count", minimum=1
        ),
        confirmatory_seed_source=_string(cohorts, "confirmatory_seed_source"),
        confirmation_oracle_access=_string(cohorts, "confirmation_oracle_access"),
    )


def _oracle_query_contract(
    oracle_query: Mapping[str, Any],
) -> OracleQueryContract:
    return OracleQueryContract(
        status=_string(oracle_query, "status"),
        query_identity_fields=_string_tuple(oracle_query, "query_identity_fields"),
        logical_call_unit=_string(oracle_query, "logical_call_unit"),
        all_required_endpoints_return_atomically=_boolean(
            oracle_query, "all_required_endpoints_return_atomically"
        ),
        cross_method_cache_hit_is_charged=_boolean(
            oracle_query, "cross_method_cache_hit_is_charged"
        ),
        exact_same_run_identity_cache_replay_is_not_a_submission=_boolean(
            oracle_query,
            "exact_same_run_identity_cache_replay_is_not_a_submission",
        ),
        same_identity_resubmission_after_any_submission_is_forbidden=_boolean(
            oracle_query,
            "same_identity_resubmission_after_any_submission_is_forbidden",
        ),
        transport_retry_may_only_poll_existing_submission=_boolean(
            oracle_query, "transport_retry_may_only_poll_existing_submission"
        ),
        new_replicate_id_is_new_identity_and_is_charged=_boolean(
            oracle_query, "new_replicate_id_is_new_identity_and_is_charged"
        ),
        failed_submitted_call_is_charged=_boolean(oracle_query, "failed_submitted_call_is_charged"),
        missing_submitted_response_is_charged=_boolean(
            oracle_query, "missing_submitted_response_is_charged"
        ),
        censored_submitted_response_is_charged=_boolean(
            oracle_query, "censored_submitted_response_is_charged"
        ),
        partial_submitted_response_is_charged_and_ineligible=_boolean(
            oracle_query,
            "partial_submitted_response_is_charged_and_ineligible",
        ),
        timeout_after_submission_is_charged=_boolean(
            oracle_query, "timeout_after_submission_is_charged"
        ),
        failed_or_ineligible_call_may_be_replaced_without_charge=_boolean(
            oracle_query,
            "failed_or_ineligible_call_may_be_replaced_without_charge",
        ),
        oracle_contract_sha256_status=_string(oracle_query, "oracle_contract_sha256_status"),
        objective_constraint_semantics_status=_string(
            oracle_query, "objective_constraint_semantics_status"
        ),
        missing_censoring_semantics_status=_string(
            oracle_query, "missing_censoring_semantics_status"
        ),
        timed_cross_method_physical_cache_latency_advantage_allowed=_boolean(
            oracle_query,
            "timed_cross_method_physical_cache_latency_advantage_allowed",
        ),
        timed_cross_method_physical_cache_reuse_allowed=_boolean(
            oracle_query, "timed_cross_method_physical_cache_reuse_allowed"
        ),
        generator_oracle_training_provenance_status=_string(
            oracle_query, "generator_oracle_training_provenance_status"
        ),
    )


def _resource_limits(
    design: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> SearchResourceLimits:
    return SearchResourceLimits(
        slurm_account=_string(resources, "slurm_account"),
        cpu_partition=_string(resources, "cpu_partition"),
        gpu_partition=_string(resources, "gpu_partition"),
        gpu_type=_string(resources, "gpu_type"),
        nodes_per_run=_integer(resources, "nodes_per_run", minimum=1),
        gpus_per_gpu_run=_integer(resources, "gpus_per_gpu_run", minimum=1),
        proposal_attempt_cap=_integer(design, "proposal_attempt_cap", minimum=1),
        kg_shortlist_cap=_integer(design, "kg_shortlist_cap", minimum=1),
        kg_joint_q_cap=_integer(design, "kg_joint_q_cap", minimum=1),
        kg_exact_pool_cap=_integer(design, "kg_exact_pool_cap", minimum=1),
        kg_max_combinations=_integer(design, "kg_max_combinations", minimum=1),
        kg_fantasies=_integer(design, "kg_fantasies", minimum=2),
        oracle_service_batch_cap=_integer(design, "oracle_service_batch_cap", minimum=1),
        cpus_per_run=_integer(resources, "cpus_per_run", minimum=1),
        host_memory_gib=_integer(resources, "host_memory_gib", minimum=1),
        max_peak_gpu_memory_gib=_integer(resources, "max_peak_gpu_memory_gib", minimum=1),
        scientific_wall_seconds=_integer(resources, "scientific_wall_seconds", minimum=1),
        outer_allowance_seconds=_integer(resources, "outer_allowance_seconds", minimum=0),
        scientific_clock_source=_string(resources, "scientific_clock_source"),
        scientific_clock_start=_string(resources, "scientific_clock_start"),
        scientific_clock_stop=_string(resources, "scientific_clock_stop"),
        scientific_elapsed_accounting=_string(resources, "scientific_elapsed_accounting"),
        outer_allowance_elapsed_accounting=_string(resources, "outer_allowance_elapsed_accounting"),
        resume_clock_rule=_string(resources, "resume_clock_rule"),
        timing_receipt_authentication=_string(resources, "timing_receipt_authentication"),
        scientific_clock_includes=_string_tuple(resources, "scientific_clock_includes"),
        outer_allowance_scope=_string(resources, "outer_allowance_scope"),
        outer_allowance_execution=_string(resources, "outer_allowance_execution"),
        outer_allowance_accelerator_hours=_number(resources, "outer_allowance_accelerator_hours"),
        scientific_schedule=_string(resources, "scientific_schedule"),
        scientific_schedule_seed=_integer(resources, "scientific_schedule_seed", minimum=0),
        scientific_schedule_key=_string(resources, "scientific_schedule_key"),
        scientific_schedule_wave_launch=_string(resources, "scientific_schedule_wave_launch"),
        scientific_schedule_wave_size=_integer(
            resources, "scientific_schedule_wave_size", minimum=1
        ),
        array_concurrency_cap=_integer(resources, "array_concurrency_cap", minimum=1),
        max_output_gib_per_run=_integer(resources, "max_output_gib_per_run", minimum=1),
        scratch_ceiling_gib=_integer(resources, "scratch_ceiling_gib", minimum=1),
        reproduction_runs=_integer(resources, "reproduction_runs", minimum=1),
        reproduction_methods=_string_tuple(resources, "reproduction_methods"),
        reproduction_seed_count=_integer(resources, "reproduction_seed_count", minimum=1),
        reproduction_seed_status=_string(resources, "reproduction_seed_status"),
        reproduction_logical_unique_calls=_integer(
            resources, "reproduction_logical_unique_calls", minimum=1
        ),
        reproduction_scientific_wall_seconds_per_run=_integer(
            resources,
            "reproduction_scientific_wall_seconds_per_run",
            minimum=1,
        ),
        reproduction_a100_hour_ceiling=_number(resources, "reproduction_a100_hour_ceiling"),
        reproduction_results_may_enter_screen_confirmation_or_promotion=_boolean(
            resources,
            "reproduction_results_may_enter_screen_confirmation_or_promotion",
        ),
        hard_evidence_and_reproduction_a100_hour_ceiling=_number(
            resources, "hard_evidence_and_reproduction_a100_hour_ceiling"
        ),
        common_initial_oracle_production_accelerator_accounting_status=_string(
            resources,
            "common_initial_oracle_production_accelerator_accounting_status",
        ),
        adapter_or_tuning_work_authorized_by_v1=_boolean(
            resources, "adapter_or_tuning_work_authorized_by_v1"
        ),
        adapter_or_tuning_a100_hours_in_v1=_number(resources, "adapter_or_tuning_a100_hours_in_v1"),
        adapter_or_tuning_logical_calls_in_v1=_integer(
            resources, "adapter_or_tuning_logical_calls_in_v1", minimum=0
        ),
        unallocated_resource_pool_allowed=_boolean(resources, "unallocated_resource_pool_allowed"),
    )


def _batching_limits(batching: Mapping[str, Any]) -> BatchingLimits:
    return BatchingLimits(
        profile=_string(batching, "profile"),
        rollout_batch_size_cap=_integer(batching, "rollout_batch_size_cap", minimum=1),
        proposal_batch_size_cap=_integer(batching, "proposal_batch_size_cap", minimum=1),
        surrogate_batch_size_cap=_integer(batching, "surrogate_batch_size_cap", minimum=1),
        kg_candidate_chunk_size_cap=_integer(batching, "kg_candidate_chunk_size_cap", minimum=1),
        kg_fantasy_chunk_size_cap=_integer(batching, "kg_fantasy_chunk_size_cap", minimum=1),
        oracle_batch_size_cap=_integer(batching, "oracle_batch_size_cap", minimum=1),
        replay_sequence_batch_cap=_integer(batching, "replay_sequence_batch_cap", minimum=1),
        replay_token_batch_cap=_integer(batching, "replay_token_batch_cap", minimum=1),
        gradient_accumulation_steps=_integer(batching, "gradient_accumulation_steps", minimum=1),
        record_realized_sizes=_boolean(batching, "record_realized_sizes"),
        seed_order_invariant_to_batching=_boolean(batching, "seed_order_invariant_to_batching"),
    )


def _statistical_reporting_contract(
    *,
    metrics: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> StatisticalReportingContract:
    return StatisticalReportingContract(
        paired_bootstrap_statistic=_string(uncertainty, "paired_bootstrap_statistic"),
        paired_bootstrap_interval_level=_number(uncertainty, "paired_bootstrap_interval_level"),
        paired_bootstrap_interval_type=_string(uncertainty, "paired_bootstrap_interval_type"),
        paired_bootstrap_interval_quantiles=_number_tuple(
            uncertainty, "paired_bootstrap_interval_quantiles"
        ),
        paired_bootstrap_quantile_convention=_string(
            uncertainty, "paired_bootstrap_quantile_convention"
        ),
        hodges_lehmann_estimand=_string(statistics, "hodges_lehmann_estimand"),
        median_convention=_string(statistics, "median_convention"),
        coverage_reporting_scope=_string(uncertainty, "coverage_reporting_scope"),
        coverage_interval_convention=_string(uncertainty, "coverage_interval_convention"),
        coverage_missing_nonfinite_or_wrong_level_count=_string(
            uncertainty, "coverage_missing_nonfinite_or_wrong_level_count"
        ),
        top10_feasible_mean_utility_checkpoint_estimand=_string(
            metrics, "top10_feasible_mean_utility_checkpoint_estimand"
        ),
        top10_feasible_mean_utility_auc_estimand=_string(
            metrics, "top10_feasible_mean_utility_auc_estimand"
        ),
        wall_time_auc_estimand=_string(metrics, "wall_time_auc_estimand"),
        wall_time_checkpoint_assignment=_string(metrics, "wall_time_checkpoint_assignment"),
        terminal_abstention_rate_estimand=_string(metrics, "terminal_abstention_rate_estimand"),
        terminal_abstention_error_estimand=_string(metrics, "terminal_abstention_error_estimand"),
        parent_child_contrast_coverage_estimand=_string(
            metrics, "parent_child_contrast_coverage_estimand"
        ),
        parent_child_contrast_accuracy_estimand=_string(
            metrics, "parent_child_contrast_accuracy_estimand"
        ),
    )


def _stopping_rules(stopping: Mapping[str, Any]) -> SearchStoppingRules:
    return SearchStoppingRules(
        stop_at_unique_calls_or_wall_seconds_whichever_first=_boolean(
            stopping, "stop_at_unique_calls_or_wall_seconds_whichever_first"
        ),
        discard_unsealed_partial_batch=_boolean(stopping, "discard_unsealed_partial_batch"),
        carry_last_sealed_incumbent_to_later_checkpoints=_boolean(
            stopping, "carry_last_sealed_incumbent_to_later_checkpoints"
        ),
        required_terminal_eligible_candidates=_integer(
            stopping, "required_terminal_eligible_candidates", minimum=1
        ),
        infrastructure_reruns_before_first_oracle_response=_integer(
            stopping, "infrastructure_reruns_before_first_oracle_response", minimum=0
        ),
        result_driven_reruns=_integer(stopping, "result_driven_reruns", minimum=0),
        resume_after_response_from_last_authenticated_round_only=_boolean(
            stopping, "resume_after_response_from_last_authenticated_round_only"
        ),
        algorithmic_failure_is_retained=_boolean(stopping, "algorithmic_failure_is_retained"),
        max_local_complete_transition_kl_mean=_number(
            stopping, "max_local_complete_transition_kl_mean"
        ),
        max_local_complete_transition_kl_p99=_number(
            stopping, "max_local_complete_transition_kl_p99"
        ),
        max_frozen_reference_path_kl_mean=_number(stopping, "max_frozen_reference_path_kl_mean"),
        max_frozen_reference_transition_kl_p99=_number(
            stopping, "max_frozen_reference_transition_kl_p99"
        ),
        min_replay_ess_fraction=_number(stopping, "min_replay_ess_fraction"),
        max_normalized_replay_weight=_number(stopping, "max_normalized_replay_weight"),
        max_policy_version_lag=_integer(stopping, "max_policy_version_lag", minimum=0),
        max_validity_drop_fraction=_number(stopping, "max_validity_drop_fraction"),
        nonfinite_or_psd_support_failure=_string(stopping, "nonfinite_or_psd_support_failure"),
        forbidden_support_or_overlap=_string(stopping, "forbidden_support_or_overlap"),
        unsealed_oracle_response=_string(stopping, "unsealed_oracle_response"),
        no_kl_ablation_ignores_only_kl_thresholds=_boolean(
            stopping, "no_kl_ablation_ignores_only_kl_thresholds"
        ),
        kg_tie_or_numerical_instability=_string(stopping, "kg_tie_or_numerical_instability"),
    )


def _terminal_evaluation_contract(
    terminal: Mapping[str, Any],
) -> TerminalEvaluationContract:
    return TerminalEvaluationContract(
        candidate_domain=_string(terminal, "candidate_domain"),
        point_estimate=_string(terminal, "point_estimate"),
        feasibility=_string(terminal, "feasibility"),
        oracle_constraint_extension_at_runtime_allowed=_boolean(
            terminal, "oracle_constraint_extension_at_runtime_allowed"
        ),
        candidate_truth_eligibility=_string(terminal, "candidate_truth_eligibility"),
        recommendation_tie_break=_string(terminal, "recommendation_tie_break"),
        minimum_eligible_candidates=_integer(terminal, "minimum_eligible_candidates", minimum=1),
        abstain_if=_string(terminal, "abstain_if"),
        regret_domain=_string(terminal, "regret_domain"),
        regret_oracle_utility=_string(terminal, "regret_oracle_utility"),
        regret=_string(terminal, "regret"),
        empty_feasible_domain_oracle_utility=_number(
            terminal, "empty_feasible_domain_oracle_utility"
        ),
        abstention_regret_uses_no_action_utility=_boolean(
            terminal, "abstention_regret_uses_no_action_utility"
        ),
        missing_censored_or_nonfinite_is_ineligible_not_imputed=_boolean(
            terminal,
            "missing_censored_or_nonfinite_is_ineligible_not_imputed",
        ),
    )


def _promotion_gate(promotion: Mapping[str, Any]) -> PromotionGate:
    return PromotionGate(
        screen_requires_highest_mean_primary_metric_among_eight_methods=_boolean(
            promotion,
            "screen_requires_highest_mean_primary_metric_among_eight_methods",
        ),
        screen_requires_each_core_ablation_positive_pairs=_integer(
            promotion,
            "screen_requires_each_core_ablation_positive_pairs",
            minimum=1,
        ),
        screen_requires_each_core_ablation_median_above_additive_margin=_boolean(
            promotion,
            "screen_requires_each_core_ablation_median_above_additive_margin",
        ),
        confirmation_requires_all_pairs_above_additive_margin=_boolean(
            promotion, "confirmation_requires_all_pairs_above_additive_margin"
        ),
        primary_missing_nonfinite_or_tied_required_gate_is_failure=_boolean(
            promotion, "primary_missing_nonfinite_or_tied_required_gate_is_failure"
        ),
        full_method_kl_replay_numerical_integrity_result_required=_boolean(
            promotion,
            "full_method_kl_replay_numerical_integrity_result_required",
        ),
        yield_and_diversity_gate_comparators=_string(
            promotion, "yield_and_diversity_gate_comparators"
        ),
        yield_and_diversity_pairwise_gate_scope=_string(
            promotion, "yield_and_diversity_pairwise_gate_scope"
        ),
        max_valid_unique_reference_safe_yield_additive_loss=_number(
            promotion, "max_valid_unique_reference_safe_yield_additive_loss"
        ),
        max_hill2_effective_cluster_loss_fraction=_number(
            promotion, "max_hill2_effective_cluster_loss_fraction"
        ),
        max_largest_cluster_share_additive_increase=_number(
            promotion, "max_largest_cluster_share_additive_increase"
        ),
        secondary_gate_comparison_absolute_tolerance=_number(
            promotion, "secondary_gate_comparison_absolute_tolerance"
        ),
        valid_unique_reference_safe_yield_estimand=_string(
            promotion, "valid_unique_reference_safe_yield_estimand"
        ),
        yield_zero_charged_identity_value=_string(promotion, "yield_zero_charged_identity_value"),
        identity70_metric=_string(promotion, "identity70_metric"),
        identity70_threshold=_number(promotion, "identity70_threshold"),
        identity70_linkage=_string(promotion, "identity70_linkage"),
        hill2_estimand=_string(promotion, "hill2_estimand"),
        hill2_zero_valid_sequence_value=_number(promotion, "hill2_zero_valid_sequence_value"),
        hill2_zero_comparator_gate=_string(promotion, "hill2_zero_comparator_gate"),
        largest_cluster_share_estimand=_string(promotion, "largest_cluster_share_estimand"),
        largest_cluster_share_zero_valid_sequence_value=_number(
            promotion, "largest_cluster_share_zero_valid_sequence_value"
        ),
        yield_diversity_comparison_rule=_string(promotion, "yield_diversity_comparison_rule"),
        max_ece=_number(promotion, "max_ece"),
        max_ece_additive_degradation=_number(promotion, "max_ece_additive_degradation"),
        calibration_gate_scope=_string(promotion, "calibration_gate_scope"),
        calibration_target=_string(promotion, "calibration_target"),
        ece_estimand=_string(promotion, "ece_estimand"),
        ece_equal_mass_bin_rule=_string(promotion, "ece_equal_mass_bin_rule"),
        ece_empty_or_incomplete_value=_string(promotion, "ece_empty_or_incomplete_value"),
        coverage90_estimand=_string(promotion, "coverage90_estimand"),
        coverage90_empty_or_incomplete_value=_string(
            promotion, "coverage90_empty_or_incomplete_value"
        ),
        calibration_comparison_rule=_string(promotion, "calibration_comparison_rule"),
        calibration_per_seed_report_required=_boolean(
            promotion, "calibration_per_seed_report_required"
        ),
        coverage90_lower=_number(promotion, "coverage90_lower"),
        coverage90_upper=_number(promotion, "coverage90_upper"),
        independent_chronological_or_assay_result_required=_boolean(
            promotion, "independent_chronological_or_assay_result_required"
        ),
        surrogate_screen_max_shadow_generator_mixture_quota=_number(
            promotion, "surrogate_screen_max_shadow_generator_mixture_quota"
        ),
        surrogate_screen_max_production_generator_mixture_quota=_number(
            promotion, "surrogate_screen_max_production_generator_mixture_quota"
        ),
        shadow_candidates_may_enter_submission_or_top100=_boolean(
            promotion, "shadow_candidates_may_enter_submission_or_top100"
        ),
        production_requires_confirmation_and_independent_result=_boolean(
            promotion, "production_requires_confirmation_and_independent_result"
        ),
        production_quota_requires_new_content_pinned_promotion_protocol=_boolean(
            promotion,
            "production_quota_requires_new_content_pinned_promotion_protocol",
        ),
        surrogate_pass_guarantees_final_library_or_top100_seat=_boolean(
            promotion, "surrogate_pass_guarantees_final_library_or_top100_seat"
        ),
        no_go_on_failed_confirmation=_boolean(promotion, "no_go_on_failed_confirmation"),
        screen_failure_is_no_go=_boolean(promotion, "screen_failure_is_no_go"),
        unsequestered_confirmation_is_no_go=_boolean(
            promotion, "unsequestered_confirmation_is_no_go"
        ),
        independent_result_missing_or_failure_is_no_go=_boolean(
            promotion, "independent_result_missing_or_failure_is_no_go"
        ),
        any_required_gate_failure_is_no_go=_boolean(
            promotion, "any_required_gate_failure_is_no_go"
        ),
        passing_research_gates_authorizes_v1_production=_boolean(
            promotion, "passing_research_gates_authorizes_v1_production"
        ),
    )


def _validate_frozen_tables(
    *,
    raw: Mapping[str, Any],
    support: Mapping[str, Any],
    cohorts: Mapping[str, Any],
    design: Mapping[str, Any],
    oracle_query: Mapping[str, Any],
    resources: Mapping[str, Any],
    batching: Mapping[str, Any],
    metrics: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    terminal: Mapping[str, Any],
    stopping: Mapping[str, Any],
    statistics: Mapping[str, Any],
    promotion: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evidence_boundaries: Mapping[str, Any],
) -> None:
    """Check the safety-bearing fields even though the full file is content-pinned."""

    _require_frozen_fields(
        raw,
        {
            "decision_date": "2026-09-08",
            "execution_authorized": False,
            "automatic_production_eligible": False,
            "evidence_class": "de_novo_hidden_computational_oracle_surrogate_only",
            "biological_superiority_claim_allowed": False,
            "confirmation_seed_status": (
                "published_unsequestered_development_only_not_confirmatory"
            ),
        },
        name="root",
    )
    _require_frozen_fields(
        support,
        {
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "min_length": 8,
            "max_length": 50,
            "linear_unmodified_free_termini": True,
            "canonical_sequence_function": "amp_challenge.sequences.canonicalize_sequence",
            "exact_training_overlap_forbidden": True,
            "organizer_reference_role": "post_generation_compliance_only",
            "organizer_reference_may_shape_generation": False,
            "organizer_reference_may_shape_model_fit_or_search_score": False,
            "organizer_reference_compliance_may_veto_shadow_or_production": True,
            "organizer_reference_validator_commit": ("5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e"),
            "organizer_reference_similarity": "Levenshtein.ratio",
            "organizer_reference_max_similarity": 0.8,
            "organizer_reference_set_receipt_status": "missing_execution_blocking",
            "training_homology_exclusion_status": "missing_execution_blocking",
            "generator_oracle_training_provenance_separation_status": (
                "missing_execution_blocking"
            ),
        },
        name="support",
    )
    _require_frozen_fields(
        cohorts,
        {
            "screen_unit": "one_method_seed_de_novo_hidden_oracle_run",
            "confirmation_unit": (
                "one_method_published_development_seed_de_novo_hidden_oracle_run"
            ),
            "de_novo_rotation_dimension": "none",
            "de_novo_rotation_count": 0,
            "de_novo_checkpoint_aggregation_status": "undefined_execution_blocking",
            "retrospective_fixed_pool_protocol": ("separate_content_pinned_protocol_required"),
            "retrospective_fixed_pool_rotations": 20,
            "retrospective_results_may_enter_de_novo_screen_or_confirmation": False,
            "retrospective_calls_in_de_novo_budget": 0,
            "published_confirmation_seeds_may_support_confirmatory_or_promotion_claim": (False),
            "successor_hidden_confirmation_seed_count": 5,
            "confirmatory_seed_source": (
                "successor_hidden_independent_reveal_after_code_model_protocol_freeze"
            ),
            "confirmation_oracle_access": (
                "successor_sequestered_independent_access_control_required"
            ),
        },
        name="cohorts",
    )
    _require_frozen_fields(
        design,
        {
            "proposal_attempt_cap": 65536,
            "kg_shortlist_cap": 256,
            "kg_joint_q_cap": 14,
            "kg_exact_pool_cap": 20,
            "kg_max_combinations": 65536,
            "kg_fantasies": 512,
            "oracle_service_batch_cap": 32,
            "terminal_rule": "posterior_mean_feasible_real_candidate_or_abstain",
            "common_initial_design_within_seed": True,
            "common_random_reserve_within_seed": True,
            "common_initial_responses_delivered_before_scientific_clock": True,
            "common_initial_calls_charged_per_run": True,
            "cross_method_cache_is_charged_logically": True,
            "exact_same_run_identity_cache_replay_is_not_charged": True,
            "random_reserve_is_fixed_before_method_selection": True,
            "scheduled_common_reserve_scope": (
                "all_28_batches_times_2_seats_prefrozen_per_seed_before_any_"
                "method_specific_load_or_compute"
            ),
            "scheduled_common_reserve_proposal_exclusion": (
                "all_56_scheduled_query_identities_excluded_from_every_arm_"
                "method_controlled_proposal_and_overflow_candidate"
            ),
            "scheduled_common_reserve_submission_rule": (
                "submit_the_same_two_prefrozen_query_identities_in_the_same_seat_"
                "order_for_every_arm_at_each_batch"
            ),
            "timed_cross_method_cache_release_policy": (
                "physical_cross_arm_cache_disabled_during_timed_adaptive_rounds"
            ),
            "method_controlled_duplicate_rule": (
                "skip_exact_query_identity_without_call_or_seat_advance_then_continue_"
                "frozen_proposal_order"
            ),
            "method_controlled_exhaustion_rule": (
                "fill_unfilled_method_seats_from_separate_frozen_overflow_reserve_"
                "after_scheduled_common_reserve_seats"
            ),
            "overflow_reserve_scope": (
                "separate_prefrozen_per_seed_stream_disjoint_from_all_56_"
                "scheduled_common_reserve_identities"
            ),
            "overflow_reserve_attempt_cap_per_batch": 65536,
            "overflow_reserve_collision_rule": (
                "skip_previously_submitted_or_current_batch_query_identity_in_frozen_order"
            ),
            "overflow_reserve_exhaustion_rule": (
                "hard_stop_algorithmic_failure_no_unfrozen_refill"
            ),
        },
        name="design",
    )
    _require_frozen_fields(
        oracle_query,
        {
            "status": "identity_frozen_oracle_semantics_missing_execution_blocking",
            "query_identity_fields": [
                "canonical_sequence_id",
                "oracle_contract_sha256",
                "evaluator_sha256",
                "checkpoint_sha256",
                "endpoint_context_sha256",
                "transform_sha256",
                "replicate_id",
            ],
            "logical_call_unit": ("first_submission_of_one_exact_query_identity_within_one_run"),
            "all_required_endpoints_return_atomically": True,
            "cross_method_cache_hit_is_charged": True,
            "exact_same_run_identity_cache_replay_is_not_a_submission": True,
            "same_identity_resubmission_after_any_submission_is_forbidden": True,
            "transport_retry_may_only_poll_existing_submission": True,
            "new_replicate_id_is_new_identity_and_is_charged": True,
            "failed_submitted_call_is_charged": True,
            "missing_submitted_response_is_charged": True,
            "censored_submitted_response_is_charged": True,
            "partial_submitted_response_is_charged_and_ineligible": True,
            "timeout_after_submission_is_charged": True,
            "failed_or_ineligible_call_may_be_replaced_without_charge": False,
            "oracle_contract_sha256_status": "missing_execution_blocking",
            "objective_constraint_semantics_status": "missing_execution_blocking",
            "missing_censoring_semantics_status": "missing_execution_blocking",
            "timed_cross_method_physical_cache_latency_advantage_allowed": False,
            "timed_cross_method_physical_cache_reuse_allowed": False,
            "generator_oracle_training_provenance_status": ("missing_execution_blocking"),
        },
        name="oracle_query_contract",
    )
    _require_frozen_fields(
        resources,
        {
            "slurm_account": "bio",
            "cpu_partition": "standard",
            "gpu_partition": "gpumid",
            "gpu_type": "A100",
            "nodes_per_run": 1,
            "gpus_per_gpu_run": 1,
            "cpus_per_run": 8,
            "host_memory_gib": 32,
            "max_peak_gpu_memory_gib": 16,
            "scientific_wall_seconds": 7200,
            "outer_allowance_seconds": 900,
            "scientific_clock_source": "monotonic_elapsed_seconds",
            "scientific_clock_start": (
                "after_common_inputs_and_64_charged_initial_responses_authenticated_"
                "and_delivered_before_method_specific_load_or_compute"
            ),
            "scientific_clock_stop": (
                "at_7200_seconds_or_terminal_status_of_512th_charged_submission_whichever_first"
            ),
            "scientific_elapsed_accounting": (
                "sum_authenticated_segment_elapsed_nanoseconds_across_initial_and_"
                "all_resumed_scientific_slurm_jobs_without_reset"
            ),
            "outer_allowance_elapsed_accounting": (
                "sum_authenticated_segment_elapsed_nanoseconds_across_all_sealing_"
                "slurm_jobs_without_reset"
            ),
            "resume_clock_rule": (
                "each_receipt_chains_to_the_prior_receipt_and_carries_exact_"
                "cumulative_scientific_and_outer_elapsed_nanoseconds"
            ),
            "timing_receipt_authentication": (
                "external_trusted_receipt_required_for_every_initial_or_resumed_slurm_job_segment"
            ),
            "scientific_clock_includes": [
                "method_specific_model_loading",
                "generation",
                "training_and_distillation",
                "proposal_and_feature_computation",
                "posterior_and_acquisition_computation",
                "adaptive_oracle_queue_service_and_status_poll_time",
            ],
            "outer_allowance_scope": (
                "sealing_integrity_checks_and_receipt_handoff_only_no_search_or_oracle"
            ),
            "outer_allowance_execution": ("cpu_only_followup_after_accelerator_release"),
            "outer_allowance_accelerator_hours": 0,
            "scientific_schedule": ("seed_blocked_sha256_randomized_simultaneous_waves"),
            "scientific_schedule_seed": 20260908,
            "scientific_schedule_key": (
                "sha256_utf8_decimal_schedule_seed_nul_phase_nul_decimal_run_seed_"
                "nul_configuration_id_ascending_then_configuration_id"
            ),
            "scientific_schedule_wave_launch": (
                "one_seed_block_at_a_time_each_wave_single_barrier_release"
            ),
            "scientific_schedule_wave_size": 4,
            "array_concurrency_cap": 4,
            "max_output_gib_per_run": 5,
            "scratch_ceiling_gib": 1024,
            "reproduction_a100_hour_ceiling": 6,
            "reproduction_runs": 3,
            "reproduction_methods": [
                "tuned_peptide_ga",
                "tr2d2_style_tree_offpolicy",
                "counterfactual_softkg_evolutionary_diffusion",
            ],
            "reproduction_seed_count": 1,
            "reproduction_seed_status": (
                "successor_hidden_independent_reveal_after_scientific_campaign_sealed"
            ),
            "reproduction_logical_unique_calls": 1536,
            "reproduction_scientific_wall_seconds_per_run": 7200,
            "reproduction_results_may_enter_screen_confirmation_or_promotion": False,
            "hard_evidence_and_reproduction_a100_hour_ceiling": 166,
            "common_initial_oracle_production_accelerator_accounting_status": (
                "successor_separate_content_pinned_budget_required_outside_166_method_job_ceiling"
            ),
            "adapter_or_tuning_work_authorized_by_v1": False,
            "adapter_or_tuning_a100_hours_in_v1": 0,
            "adapter_or_tuning_logical_calls_in_v1": 0,
            "unallocated_resource_pool_allowed": False,
        },
        name="resources",
    )
    _require_frozen_fields(
        batching,
        {
            "profile": "cluster_batch_first_v1",
            "rollout_batch_size_cap": 128,
            "proposal_batch_size_cap": 65536,
            "surrogate_batch_size_cap": 8192,
            "kg_candidate_chunk_size_cap": 256,
            "kg_fantasy_chunk_size_cap": 512,
            "oracle_batch_size_cap": 32,
            "replay_sequence_batch_cap": 1024,
            "replay_token_batch_cap": 32768,
            "gradient_accumulation_steps": 8,
            "record_realized_sizes": True,
            "seed_order_invariant_to_batching": True,
        },
        name="batching",
    )
    _require_frozen_fields(
        metrics,
        {
            "primary": (
                "normalized_feasible_two_objective_hypervolume_auc_by_submitted_query_identity"
            ),
            "objectives": [
                "gram_positive_activity",
                "gram_negative_activity",
            ],
            "objective_bounds": [0.0, 1.0],
            "hypervolume_reference": [0.0, 0.0],
            "hypervolume_auc_denominator_calls": 448,
            "primary_auc_quadrature": (
                "trapezoidal_over_fixed_call_checkpoints_divided_by_call_span"
            ),
            "primary_checkpoint_input": (
                "nonempty_contiguous_authenticated_sealed_prefix_starting_at_call_64"
            ),
            "primary_checkpoint_values": (
                "finite_unit_interval_nondecreasing_incumbent_hypervolume"
            ),
            "primary_value_source": ("authenticated_hidden_oracle_response_never_method_posterior"),
            "primary_archive_domain": (
                "same_run_submitted_identities_passing_canonical_support_exact_"
                "training_overlap_exclusion_accepted_successor_training_homology_"
                "exclusion_and_complete_finite_uncensored_oracle_objectives_and_"
                "all_frozen_constraints_observed_finite_and_pass"
            ),
            "broad_spectrum_role": (
                "secondary_derived_three_sevenths_gram_positive_plus_four_sevenths_gram_negative"
            ),
            "pairwise_sequence_diversity_estimand": (
                "mean_one_minus_global_sequence_identity_over_unordered_pairs_"
                "of_valid_unique_reference_safe_sequences"
            ),
            "pairwise_embedding_diversity_estimand": (
                "mean_one_minus_cosine_similarity_over_unordered_pairs_of_"
                "content_pinned_embeddings_for_same_sequence_set"
            ),
            "pairwise_diversity_fewer_than_two_value": 0.0,
            "pairwise_embedding_nonfinite_or_zero_norm_value": (
                "missing_not_imputed_descriptive_only"
            ),
            "scalar_utility": ("equal_arithmetic_mean_of_gram_positive_and_gram_negative"),
            "no_action_utility": 0.0,
            "call_checkpoint_start": 64,
            "call_checkpoint_step": 16,
            "call_checkpoint_count": 29,
            "top10_feasible_mean_utility_checkpoint_estimand": (
                "arithmetic_mean_of_up_to_ten_highest_oracle_scalar_utilities_in_"
                "the_primary_eligible_archive_at_each_call_checkpoint_or_no_action_"
                "utility_if_empty"
            ),
            "top10_feasible_mean_utility_auc_estimand": (
                "trapezoidal_integral_over_the_same_29_call_checkpoints_divided_by_"
                "448_calls_with_last_sealed_checkpoint_carried_after_early_stop"
            ),
            "wall_time_auc_estimand": (
                "trapezoidal_integral_of_sealed_primary_incumbent_hypervolume_at_"
                "fixed_wall_checkpoints_divided_by_120_minutes"
            ),
            "wall_time_checkpoint_assignment": (
                "include_only_authenticated_complete_batches_sealed_at_or_before_"
                "each_cumulative_scientific_elapsed_checkpoint_use_zero_before_"
                "first_eligible_archive_value_and_carry_last_value_after_stop"
            ),
            "terminal_abstention_rate_estimand": (
                "number_of_abstaining_runs_divided_by_all_exact_required_cohort_"
                "runs_with_any_missing_terminal_record_a_required_gate_failure"
            ),
            "terminal_abstention_error_estimand": (
                "among_abstaining_runs_fraction_whose_terminal_regret_domain_"
                "contains_a_feasible_oracle_utility_strictly_above_no_action_"
                "utility_zero_if_no_abstentions"
            ),
            "parent_child_contrast_coverage_estimand": (
                "submitted_adaptive_child_identities_with_exactly_one_ledger_bound_"
                "parent_complete_finite_uncensored_eligible_parent_and_child_oracle_"
                "utility_and_finite_presubmission_predicted_contrast_divided_by_all_"
                "such_children_with_eligible_parent_and_child_truth"
            ),
            "parent_child_contrast_accuracy_estimand": (
                "covered_pairs_with_predicted_and_realized_child_minus_parent_"
                "utility_signs_both_strictly_outside_1e_minus_12_and_equal_divided_"
                "by_covered_pairs_with_zero_or_tied_sign_counted_incorrect_missing_"
                "if_no_covered_pairs"
            ),
        },
        name="metrics",
    )
    _require_frozen_fields(
        uncertainty,
        {
            "paired_bootstrap_samples": 10000,
            "paired_bootstrap_seed": 20260908,
            "de_novo_paired_bootstrap_unit": (
                "seed_block_resample_with_same_seed_index_shared_across_all_compared_methods"
            ),
            "paired_bootstrap_statistic": (
                "arithmetic_mean_of_five_paired_full_minus_comparator_seed_effects"
            ),
            "paired_bootstrap_interval_level": 0.95,
            "paired_bootstrap_interval_type": "percentile_two_sided",
            "paired_bootstrap_interval_quantiles": [0.025, 0.975],
            "paired_bootstrap_quantile_convention": (
                "sorted_B_replicates_linear_type7_h_equals_B_minus_1_times_p_"
                "interpolate_between_floor_and_ceil"
            ),
            "retrospective_bootstrap_unit": "homology_study_union_component",
            "retrospective_bootstrap_samples": 10000,
            "kg_rank_stability_kendall_tau_min": 0.95,
            "kg_monte_carlo_se_fraction_of_top_two_gap_max": 0.05,
            "kg_rank_correlation": "kendall_tau_b",
            "kg_top_two_gap_requirement": "strictly_positive_and_finite",
            "kg_zero_or_tied_top_two_gap": "hard_stop_no_arbitrary_ranking",
            "kg_nonfinite_score_se_or_rank_statistic": "hard_stop",
            "coverage_reporting_scope": (
                "pooled_and_each_seed_for_all_three_confirmation_methods_at_all_four_frozen_levels"
            ),
            "coverage_interval_convention": (
                "presubmission_equal_tailed_marginal_predictive_interval_with_"
                "inclusive_endpoints_for_each_query_objective_pair"
            ),
            "coverage_missing_nonfinite_or_wrong_level_count": ("missing_required_gate_failure"),
        },
        name="uncertainty",
    )
    _require_frozen_fields(
        terminal,
        {
            "candidate_domain": (
                "authenticated_oracle_responded_candidates_submitted_by_same_run_only"
            ),
            "point_estimate": (
                "final_sealed_same_run_posterior_arithmetic_mean_of_gram_positive_"
                "and_gram_negative_means"
            ),
            "feasibility": (
                "all_successor_content_pinned_oracle_constraints_observed_finite_and_pass"
            ),
            "oracle_constraint_extension_at_runtime_allowed": False,
            "candidate_truth_eligibility": (
                "canonical_support_exact_training_overlap_exclusion_accepted_"
                "successor_training_homology_exclusion_both_objectives_and_all_"
                "constraints_complete_finite_uncensored"
            ),
            "recommendation_tie_break": "ascending_canonical_sequence_id",
            "minimum_eligible_candidates": 100,
            "abstain_if": "fewer_than_minimum_eligible_candidates",
            "regret_domain": (
                "same_run_authenticated_oracle_responded_candidates_passing_"
                "candidate_truth_eligibility_only"
            ),
            "regret_oracle_utility": ("arithmetic_mean_of_oracle_gram_positive_and_gram_negative"),
            "regret": (
                "max_no_action_and_best_feasible_oracle_utility_minus_"
                "recommendation_or_no_action_utility"
            ),
            "empty_feasible_domain_oracle_utility": 0.0,
            "abstention_regret_uses_no_action_utility": True,
            "missing_censored_or_nonfinite_is_ineligible_not_imputed": True,
        },
        name="terminal_evaluation",
    )
    _require_frozen_fields(
        stopping,
        {
            "stop_at_unique_calls_or_wall_seconds_whichever_first": True,
            "discard_unsealed_partial_batch": True,
            "carry_last_sealed_incumbent_to_later_checkpoints": True,
            "required_terminal_eligible_candidates": 100,
            "infrastructure_reruns_before_first_oracle_response": 1,
            "result_driven_reruns": 0,
            "resume_after_response_from_last_authenticated_round_only": True,
            "algorithmic_failure_is_retained": True,
            "max_local_complete_transition_kl_mean": 0.01,
            "max_local_complete_transition_kl_p99": 0.02,
            "max_frozen_reference_path_kl_mean": 0.08,
            "max_frozen_reference_transition_kl_p99": 0.02,
            "min_replay_ess_fraction": 0.20,
            "max_normalized_replay_weight": 0.05,
            "max_policy_version_lag": 1,
            "max_validity_drop_fraction": 0.05,
            "nonfinite_or_psd_support_failure": "hard_stop",
            "forbidden_support_or_overlap": "hard_stop",
            "unsealed_oracle_response": "hard_stop",
            "no_kl_ablation_ignores_only_kl_thresholds": True,
            "kg_tie_or_numerical_instability": "hard_stop",
        },
        name="stopping",
    )
    _require_frozen_fields(
        statistics,
        {
            "screen_is_descriptive_only": True,
            "screen_two_sided_exact_sign_test_minimum_p": 0.0625,
            "run_primary_missing_nonfinite_or_no_initial_checkpoint": ("required_gate_failure"),
            "early_stop_primary_metric": (
                "carry_last_sealed_hypervolume_through_remaining_call_checkpoints"
            ),
            "screen_effect": ("full_normalized_hv_auc_minus_comparator_normalized_hv_auc"),
            "screen_full_mean": ("arithmetic_mean_over_exact_five_finite_seed_metrics"),
            "screen_full_unique_highest": (
                "full_mean_strictly_greater_than_each_other_method_mean_plus_tolerance"
            ),
            "screen_comparison_absolute_tolerance": 1e-12,
            "screen_pair_success": ("finite_effect_strictly_greater_than_zero_plus_tolerance"),
            "screen_positive_pairs_required": 4,
            "screen_median_additive_margin": 0.05,
            "screen_core_ablation_gate": (
                "for_each_ablation_at_least_four_pair_successes_and_median_effect_"
                "strictly_greater_than_margin_plus_tolerance"
            ),
            "confirmation_effect": ("full_normalized_hv_auc_minus_comparator_normalized_hv_auc"),
            "confirmation_additive_materiality_margin": 0.05,
            "confirmation_comparison_absolute_tolerance": 1e-12,
            "confirmation_pair_success": (
                "finite_effect_strictly_greater_than_additive_margin_plus_tolerance"
            ),
            "confirmation_missing_nonfinite_or_tie": (
                "missing_nonfinite_or_effect_at_or_below_margin_plus_tolerance_is_pair_failure"
            ),
            "confirmation_one_sided_alpha": 0.05,
            "confirmation_exact_sign_test_minimum_p": 0.03125,
            "confirmation_pairs_required_above_margin": 5,
            "confirmation_comparator_gate": (
                "all_five_fixed_seed_pairs_are_successes_and_one_sided_exact_sign_p_at_most_alpha"
            ),
            "confirmation_global_iut_gate": ("both_primary_comparator_gates_must_pass"),
            "global_claim": "intersection_union_full_beats_both_primary_comparators",
            "multiple_testing_adjustment_for_intersection_union": "none",
            "combine_screen_and_confirmation_for_p_value": False,
            "report_hodges_lehmann": True,
            "hodges_lehmann_estimand": (
                "median_of_all_15_walsh_averages_d_i_plus_d_j_over_2_for_1_less_"
                "than_or_equal_to_i_less_than_or_equal_to_j_less_than_or_equal_to_5"
            ),
            "median_convention": (
                "arithmetic_mean_of_two_central_sorted_values_when_even_otherwise_"
                "central_sorted_value"
            ),
            "bootstrap_interval_is_descriptive": True,
            "ablations_are_exploratory": True,
            "ablation_p_values_are_confirmatory": False,
            "ablation_claim_requires_fresh_seed_preregistration": True,
        },
        name="statistics",
    )
    _require_frozen_fields(
        promotion,
        {
            "screen_requires_highest_mean_primary_metric_among_eight_methods": True,
            "screen_requires_each_core_ablation_positive_pairs": 4,
            "screen_requires_each_core_ablation_median_above_additive_margin": True,
            "confirmation_requires_all_pairs_above_additive_margin": True,
            "primary_missing_nonfinite_or_tied_required_gate_is_failure": True,
            "full_method_kl_replay_numerical_integrity_result_required": True,
            "yield_and_diversity_gate_comparators": "both_primary_comparators",
            "yield_and_diversity_pairwise_gate_scope": (
                "all_five_confirmation_seed_pairs_for_each_comparator"
            ),
            "max_valid_unique_reference_safe_yield_additive_loss": 0.02,
            "max_hill2_effective_cluster_loss_fraction": 0.10,
            "max_largest_cluster_share_additive_increase": 0.02,
            "secondary_gate_comparison_absolute_tolerance": 1e-12,
            "valid_unique_reference_safe_yield_estimand": (
                "distinct_canonical_sequences_passing_support_training_overlap_"
                "successor_homology_and_reference_rules_with_complete_finite_"
                "uncensored_truth_divided_by_charged_submitted_identities"
            ),
            "yield_zero_charged_identity_value": "missing_required_gate_failure",
            "identity70_metric": ("amp_challenge.similarity.global_sequence_identity"),
            "identity70_threshold": 0.70,
            "identity70_linkage": (
                "connected_components_single_linkage_edges_at_or_above_threshold"
            ),
            "hill2_estimand": (
                "inverse_sum_squared_cluster_proportions_over_valid_unique_reference_safe_sequences"
            ),
            "hill2_zero_valid_sequence_value": 0.0,
            "hill2_zero_comparator_gate": ("pass_if_full_is_finite_and_nonnegative"),
            "largest_cluster_share_estimand": (
                "largest_identity70_component_size_divided_by_valid_unique_"
                "reference_safe_sequence_count"
            ),
            "largest_cluster_share_zero_valid_sequence_value": 0.0,
            "yield_diversity_comparison_rule": (
                "yield_full_at_least_comparator_minus_additive_loss_hill2_full_at_"
                "least_comparator_times_one_minus_fractional_loss_largest_share_"
                "full_at_most_comparator_plus_additive_increase_all_with_tolerance"
            ),
            "max_ece": 0.10,
            "max_ece_additive_degradation": 0.02,
            "calibration_gate_scope": (
                "pooled_authenticated_confirmation_queries_reported_also_per_seed"
            ),
            "calibration_target": (
                "pre_submission_joint_feasibility_probability_against_authenticated_"
                "complete_binary_constraint_truth"
            ),
            "ece_estimand": (
                "query_weighted_sum_over_equal_mass_bins_of_bin_fraction_times_"
                "absolute_mean_probability_minus_mean_truth"
            ),
            "ece_equal_mass_bin_rule": (
                "sort_by_probability_then_query_identity_assign_zero_based_rank_"
                "times_ten_floor_divided_by_total"
            ),
            "ece_empty_or_incomplete_value": "missing_required_gate_failure",
            "coverage90_estimand": (
                "query_objective_pair_weighted_fraction_of_authenticated_truth_"
                "inside_pre_submission_equal_tailed_marginal_90_percent_predictive_"
                "interval"
            ),
            "coverage90_empty_or_incomplete_value": "missing_required_gate_failure",
            "calibration_comparison_rule": (
                "full_ece_at_most_absolute_max_and_each_comparator_ece_plus_"
                "additive_degradation_and_full_coverage90_within_closed_bounds_all_"
                "with_tolerance"
            ),
            "calibration_per_seed_report_required": True,
            "coverage90_lower": 0.85,
            "coverage90_upper": 0.95,
            "independent_chronological_or_assay_result_required": True,
            "surrogate_screen_max_shadow_generator_mixture_quota": 0.10,
            "surrogate_screen_max_production_generator_mixture_quota": 0.0,
            "shadow_candidates_may_enter_submission_or_top100": False,
            "production_requires_confirmation_and_independent_result": True,
            "production_quota_requires_new_content_pinned_promotion_protocol": True,
            "surrogate_pass_guarantees_final_library_or_top100_seat": False,
            "no_go_on_failed_confirmation": True,
            "screen_failure_is_no_go": True,
            "unsequestered_confirmation_is_no_go": True,
            "independent_result_missing_or_failure_is_no_go": True,
            "any_required_gate_failure_is_no_go": True,
            "passing_research_gates_authorizes_v1_production": False,
        },
        name="promotion",
    )
    _require_frozen_fields(
        prerequisites,
        {
            **{name: False for name in _SCREEN_PREREQUISITES},
            **{name: False for name in _CONFIRMATORY_PREREQUISITES},
            "required_base_fold_policy_checkpoints": 10,
            "independent_biological_or_chronological_panel_available": False,
            "native_v0_is_usable_prior": False,
            "native_v1_currently_has_accepted_checkpoint": False,
            "current_activity_scorer_supplies_calibrated_uncertainty": False,
            "current_activity_scorer_supplies_toxicity_or_selectivity": False,
        },
        name="prerequisites",
    )
    _require_frozen_fields(
        provenance,
        {
            "semantic_format": "canonical_utf8_lf_jsonl",
            "manifest": "SHA256SUMS_written_last",
            "file_mode": "0444",
            "directory_mode": "0555",
            "all_artifacts_including_operational_telemetry_manifest_hashed": True,
            "operational_telemetry_outside_semantic_equivalence_hashes_only": True,
            "log_all_rejected_and_unevaluated_proposals": True,
            "separate_query_and_terminal_recommendation_ids": True,
            "excluded_node_independent_verifier": True,
            "verifier_may_import_producer_or_search_implementation": False,
            "external_trusted_receipt_required": True,
            "trusted_receipt_location": "outside_producer_writable_artifact_tree",
            "producer_may_write_or_replace_trusted_receipt": False,
            "trusted_receipt_binds": [
                "manifest_sha256",
                "git_commit",
                "protocol_sha256",
                "environment_lock_sha256",
                "input_receipt_sha256s",
                "oracle_contract_sha256",
                "job_id",
                "node_id",
                "timing_chain_terminal_sha256",
                "cumulative_scientific_elapsed_nanoseconds",
                "cumulative_outer_elapsed_nanoseconds",
            ],
        },
        name="provenance",
    )
    _require_frozen_fields(
        evidence_boundaries,
        {name: False for name in _EVIDENCE_BOUNDARY_KEYS},
        name="evidence_boundaries",
    )


def load_evolutionary_kl_protocol(path: str | Path) -> EvolutionaryKLProtocol:
    """Load and validate a frozen research protocol without opening any data artifact."""

    config_path = Path(path)
    payload = config_path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            f"evolutionary/KL v1 protocol bytes differ from the frozen SHA-256: {observed_sha256}"
        )
    raw = _exact_mapping(
        tomllib.loads(payload.decode("utf-8")),
        name="root",
        keys=_TOP_LEVEL_KEYS,
    )
    if _integer(raw, "schema_version", minimum=1) != 1:
        raise ValueError("unsupported evolutionary/KL protocol schema")
    support = _exact_mapping(raw.get("support"), name="support", keys=_SUPPORT_KEYS)
    cohorts = _exact_mapping(raw.get("cohorts"), name="cohorts", keys=_COHORT_KEYS)
    design = _exact_mapping(raw.get("design"), name="design", keys=_DESIGN_KEYS)
    oracle_query = _exact_mapping(
        raw.get("oracle_query_contract"),
        name="oracle_query_contract",
        keys=_ORACLE_QUERY_KEYS,
    )
    resources = _exact_mapping(raw.get("resources"), name="resources", keys=_RESOURCE_KEYS)
    batching = _exact_mapping(raw.get("batching"), name="batching", keys=_BATCHING_KEYS)
    metrics = _exact_mapping(raw.get("metrics"), name="metrics", keys=_METRIC_KEYS)
    uncertainty = _exact_mapping(
        raw.get("uncertainty"),
        name="uncertainty",
        keys=_UNCERTAINTY_KEYS,
    )
    terminal = _exact_mapping(
        raw.get("terminal_evaluation"),
        name="terminal_evaluation",
        keys=_TERMINAL_EVALUATION_KEYS,
    )
    stopping = _exact_mapping(raw.get("stopping"), name="stopping", keys=_STOPPING_KEYS)
    statistics = _exact_mapping(
        raw.get("statistics"),
        name="statistics",
        keys=_STATISTIC_KEYS,
    )
    promotion = _exact_mapping(raw.get("promotion"), name="promotion", keys=_PROMOTION_KEYS)
    prerequisites = _exact_mapping(
        raw.get("prerequisites"),
        name="prerequisites",
        keys=_PREREQUISITE_KEYS,
    )
    provenance = _exact_mapping(
        raw.get("provenance"),
        name="provenance",
        keys=_PROVENANCE_KEYS,
    )
    evidence_boundaries = _exact_mapping(
        raw.get("evidence_boundaries"),
        name="evidence_boundaries",
        keys=_EVIDENCE_BOUNDARY_KEYS,
    )
    methods = _method_specs(raw.get("methods"))
    ablations = _ablation_specs(raw.get("ablations"))
    _validate_frozen_tables(
        raw=raw,
        support=support,
        cohorts=cohorts,
        design=design,
        oracle_query=oracle_query,
        resources=resources,
        batching=batching,
        metrics=metrics,
        uncertainty=uncertainty,
        terminal=terminal,
        stopping=stopping,
        statistics=statistics,
        promotion=promotion,
        prerequisites=prerequisites,
        provenance=provenance,
        evidence_boundaries=evidence_boundaries,
    )
    protocol = EvolutionaryKLProtocol(
        artifact=_string(raw, "artifact"),
        status=_string(raw, "status"),
        automatic_production_eligible=_boolean(raw, "automatic_production_eligible"),
        execution_authorized=_boolean(raw, "execution_authorized"),
        biological_superiority_claim_allowed=_boolean(raw, "biological_superiority_claim_allowed"),
        evidence_class=_string(raw, "evidence_class"),
        confirmation_seed_status=_string(raw, "confirmation_seed_status"),
        support_alphabet=_string(support, "alphabet"),
        support_min_length=_integer(support, "min_length", minimum=1),
        support_max_length=_integer(support, "max_length", minimum=1),
        exact_training_overlap_forbidden=_boolean(support, "exact_training_overlap_forbidden"),
        training_homology_exclusion_status=_string(support, "training_homology_exclusion_status"),
        generator_oracle_training_provenance_separation_status=_string(
            support, "generator_oracle_training_provenance_separation_status"
        ),
        organizer_reference_validator_commit=_string(
            support, "organizer_reference_validator_commit"
        ),
        organizer_reference_similarity=_string(support, "organizer_reference_similarity"),
        organizer_reference_max_similarity=_number(support, "organizer_reference_max_similarity"),
        organizer_reference_may_shape_generation=_boolean(
            support, "organizer_reference_may_shape_generation"
        ),
        organizer_reference_may_shape_model_fit_or_search_score=_boolean(
            support, "organizer_reference_may_shape_model_fit_or_search_score"
        ),
        organizer_reference_compliance_may_veto_shadow_or_production=_boolean(
            support,
            "organizer_reference_compliance_may_veto_shadow_or_production",
        ),
        screen_seeds=_seed_tuple(raw, "screen_seeds"),
        confirmation_seeds=_seed_tuple(raw, "confirmation_seeds"),
        screen_configuration_ids=_string_tuple(raw, "screen_method_ids"),
        confirmation_method_ids=_string_tuple(raw, "confirmation_method_ids"),
        methods=methods,
        ablations=ablations,
        initial_design_unique_calls=_integer(design, "initial_design_unique_calls", minimum=1),
        adaptive_batches=_integer(design, "adaptive_batches", minimum=1),
        unique_calls_per_batch=_integer(design, "unique_calls_per_batch", minimum=1),
        method_controlled_seats_per_batch=_integer(
            design, "method_controlled_seats_per_batch", minimum=1
        ),
        random_reserve_seats_per_batch=_integer(
            design, "random_reserve_seats_per_batch", minimum=1
        ),
        total_unique_calls=_integer(design, "total_unique_calls", minimum=1),
        common_initial_responses_delivered_before_scientific_clock=_boolean(
            design, "common_initial_responses_delivered_before_scientific_clock"
        ),
        common_initial_calls_charged_per_run=_boolean(
            design, "common_initial_calls_charged_per_run"
        ),
        scheduled_common_reserve_scope=_string(design, "scheduled_common_reserve_scope"),
        scheduled_common_reserve_proposal_exclusion=_string(
            design, "scheduled_common_reserve_proposal_exclusion"
        ),
        scheduled_common_reserve_submission_rule=_string(
            design, "scheduled_common_reserve_submission_rule"
        ),
        timed_cross_method_cache_release_policy=_string(
            design, "timed_cross_method_cache_release_policy"
        ),
        method_controlled_duplicate_rule=_string(design, "method_controlled_duplicate_rule"),
        method_controlled_exhaustion_rule=_string(design, "method_controlled_exhaustion_rule"),
        overflow_reserve_scope=_string(design, "overflow_reserve_scope"),
        overflow_reserve_attempt_cap_per_batch=_integer(
            design, "overflow_reserve_attempt_cap_per_batch", minimum=1
        ),
        overflow_reserve_collision_rule=_string(design, "overflow_reserve_collision_rule"),
        overflow_reserve_exhaustion_rule=_string(design, "overflow_reserve_exhaustion_rule"),
        scientific_wall_seconds=_integer(resources, "scientific_wall_seconds", minimum=1),
        primary_metric=_string(metrics, "primary"),
        primary_objectives=_string_tuple(metrics, "objectives"),
        objective_bounds=_number_tuple(metrics, "objective_bounds"),
        hypervolume_reference=_number_tuple(metrics, "hypervolume_reference"),
        hypervolume_auc_denominator_calls=_integer(
            metrics, "hypervolume_auc_denominator_calls", minimum=1
        ),
        primary_auc_quadrature=_string(metrics, "primary_auc_quadrature"),
        primary_checkpoint_input=_string(metrics, "primary_checkpoint_input"),
        primary_checkpoint_values=_string(metrics, "primary_checkpoint_values"),
        wall_checkpoint_minutes=_integer_tuple(metrics, "wall_checkpoint_minutes"),
        primary_value_source=_string(metrics, "primary_value_source"),
        primary_archive_domain=_string(metrics, "primary_archive_domain"),
        broad_spectrum_role=_string(metrics, "broad_spectrum_role"),
        pairwise_sequence_diversity_estimand=_string(
            metrics, "pairwise_sequence_diversity_estimand"
        ),
        pairwise_embedding_diversity_estimand=_string(
            metrics, "pairwise_embedding_diversity_estimand"
        ),
        pairwise_diversity_fewer_than_two_value=_number(
            metrics, "pairwise_diversity_fewer_than_two_value"
        ),
        pairwise_embedding_nonfinite_or_zero_norm_value=_string(
            metrics, "pairwise_embedding_nonfinite_or_zero_norm_value"
        ),
        ece_equal_mass_bins=_integer(uncertainty, "ece_equal_mass_bins", minimum=1),
        coverage_levels=_number_tuple(uncertainty, "coverage_levels"),
        paired_bootstrap_samples=_integer(uncertainty, "paired_bootstrap_samples", minimum=1),
        paired_bootstrap_seed=_integer(uncertainty, "paired_bootstrap_seed", minimum=0),
        de_novo_paired_bootstrap_unit=_string(uncertainty, "de_novo_paired_bootstrap_unit"),
        run_primary_missing_nonfinite_or_no_initial_checkpoint=_string(
            statistics, "run_primary_missing_nonfinite_or_no_initial_checkpoint"
        ),
        early_stop_primary_metric=_string(statistics, "early_stop_primary_metric"),
        screen_effect=_string(statistics, "screen_effect"),
        screen_full_mean=_string(statistics, "screen_full_mean"),
        screen_full_unique_highest=_string(statistics, "screen_full_unique_highest"),
        screen_comparison_absolute_tolerance=_number(
            statistics, "screen_comparison_absolute_tolerance"
        ),
        screen_pair_success=_string(statistics, "screen_pair_success"),
        screen_positive_pairs_required=_integer(
            statistics, "screen_positive_pairs_required", minimum=1
        ),
        screen_median_additive_margin=_number(statistics, "screen_median_additive_margin"),
        screen_core_ablation_gate=_string(statistics, "screen_core_ablation_gate"),
        confirmation_effect=_string(statistics, "confirmation_effect"),
        confirmation_additive_materiality_margin=_number(
            statistics, "confirmation_additive_materiality_margin"
        ),
        confirmation_comparison_absolute_tolerance=_number(
            statistics, "confirmation_comparison_absolute_tolerance"
        ),
        confirmation_pair_success=_string(statistics, "confirmation_pair_success"),
        confirmation_missing_nonfinite_or_tie=_string(
            statistics, "confirmation_missing_nonfinite_or_tie"
        ),
        confirmation_one_sided_alpha=_number(statistics, "confirmation_one_sided_alpha"),
        confirmation_exact_sign_test_minimum_p=_number(
            statistics, "confirmation_exact_sign_test_minimum_p"
        ),
        confirmation_pairs_required_above_margin=_integer(
            statistics, "confirmation_pairs_required_above_margin", minimum=1
        ),
        confirmation_comparator_gate=_string(statistics, "confirmation_comparator_gate"),
        confirmation_global_iut_gate=_string(statistics, "confirmation_global_iut_gate"),
        primary_comparator_ids=_string_tuple(statistics, "confirmation_primary_comparators"),
        screen_prerequisite_states=tuple(
            (name, _boolean(prerequisites, name)) for name in _SCREEN_PREREQUISITES
        ),
        confirmatory_prerequisite_states=tuple(
            (name, _boolean(prerequisites, name)) for name in _CONFIRMATORY_PREREQUISITES
        ),
        ablations_are_exploratory=_boolean(statistics, "ablations_are_exploratory"),
        configured_budget_summary=_budget_summary(resources),
        cohort_contract=_cohort_contract(cohorts),
        oracle_query_contract=_oracle_query_contract(oracle_query),
        resource_limits=_resource_limits(design, resources),
        batching_limits=_batching_limits(batching),
        statistical_reporting=_statistical_reporting_contract(
            metrics=metrics,
            uncertainty=uncertainty,
            statistics=statistics,
        ),
        stopping_rules=_stopping_rules(stopping),
        terminal_evaluation=_terminal_evaluation_contract(terminal),
        promotion_gate=_promotion_gate(promotion),
    )
    _validate_protocol(protocol)
    return protocol


def _validate_protocol(protocol: EvolutionaryKLProtocol) -> None:
    if protocol.artifact != "evolutionary_kl_research_protocol_v1":
        raise ValueError("unexpected evolutionary/KL protocol artifact")
    if protocol.status != "predeclared_blocked_on_prerequisites":
        raise ValueError("v1 protocol must remain explicitly blocked until a new version is frozen")
    if protocol.automatic_production_eligible:
        raise ValueError("research protocol cannot be automatically production eligible")
    if protocol.execution_authorized:
        raise ValueError("blocked v1 can never authorize execution; freeze a new version")
    if protocol.biological_superiority_claim_allowed:
        raise ValueError("surrogate protocol cannot authorize a biological-superiority claim")
    if protocol.evidence_class != "de_novo_hidden_computational_oracle_surrogate_only":
        raise ValueError("v1 evidence class changed")
    if (
        protocol.confirmation_seed_status
        != "published_unsequestered_development_only_not_confirmatory"
    ):
        raise ValueError("published v1 seeds cannot be described as confirmatory")
    if (
        protocol.support_alphabet != "ACDEFGHIKLMNPQRSTVWY"
        or protocol.support_min_length != 8
        or protocol.support_max_length != 50
        or not protocol.exact_training_overlap_forbidden
        or protocol.training_homology_exclusion_status != "missing_execution_blocking"
        or protocol.generator_oracle_training_provenance_separation_status
        != "missing_execution_blocking"
        or protocol.organizer_reference_validator_commit
        != "5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e"
        or protocol.organizer_reference_similarity != "Levenshtein.ratio"
        or protocol.organizer_reference_max_similarity != 0.8
        or protocol.organizer_reference_may_shape_generation
        or protocol.organizer_reference_may_shape_model_fit_or_search_score
        or not protocol.organizer_reference_compliance_may_veto_shadow_or_production
    ):
        raise ValueError("peptide support or leakage boundary changed")
    cohorts = protocol.cohort_contract
    if not (
        cohorts.de_novo_rotation_dimension == "none"
        and cohorts.de_novo_rotation_count == 0
        and cohorts.de_novo_checkpoint_aggregation_status == "undefined_execution_blocking"
        and cohorts.retrospective_fixed_pool_rotations == 20
        and not cohorts.retrospective_results_may_enter_de_novo_screen_or_confirmation
        and cohorts.retrospective_calls_in_de_novo_budget == 0
        and not cohorts.published_confirmation_seeds_may_support_confirmatory_or_promotion_claim
        and cohorts.successor_hidden_confirmation_seed_count == 5
        and cohorts.confirmatory_seed_source
        == "successor_hidden_independent_reveal_after_code_model_protocol_freeze"
        and cohorts.confirmation_oracle_access
        == "successor_sequestered_independent_access_control_required"
    ):
        raise ValueError("de novo and retrospective evidence cohorts are not separated")
    if protocol.screen_seeds != SCREEN_SEEDS:
        raise ValueError("screen seeds differ from the frozen cohort")
    if protocol.confirmation_seeds != CONFIRMATION_SEEDS:
        raise ValueError("confirmation seeds differ from the frozen cohort")
    if set(protocol.screen_seeds) & set(protocol.confirmation_seeds):
        raise ValueError("screen and confirmation seed cohorts must be disjoint")
    if tuple(method.method_id for method in protocol.methods) != METHOD_IDS:
        raise ValueError("the exact eight method arms must be preserved in order")
    if tuple(ablation.ablation_id for ablation in protocol.ablations) != ABLATION_IDS:
        raise ValueError("the exact five core ablations must be preserved in order")
    if any(
        ablation.base_method != "counterfactual_softkg_evolutionary_diffusion"
        for ablation in protocol.ablations
    ):
        raise ValueError("every core ablation must inherit the frozen full method")
    if protocol.screen_configuration_ids != SCREEN_CONFIGURATION_IDS:
        raise ValueError("screen must contain all eight methods and all five core ablations")
    if protocol.confirmation_method_ids != CONFIRMATION_METHOD_IDS:
        raise ValueError("confirmation must contain only full, GA, and TR2-D2-style methods")
    if protocol.primary_comparator_ids != PRIMARY_COMPARATOR_IDS:
        raise ValueError("primary confirmation comparators must remain GA and TR2-D2-style")
    if (
        protocol.primary_metric
        != "normalized_feasible_two_objective_hypervolume_auc_by_submitted_query_identity"
        or protocol.primary_objectives != ("gram_positive_activity", "gram_negative_activity")
        or protocol.objective_bounds != (0.0, 1.0)
        or protocol.hypervolume_reference != (0.0, 0.0)
        or protocol.hypervolume_auc_denominator_calls != 448
        or protocol.primary_auc_quadrature
        != "trapezoidal_over_fixed_call_checkpoints_divided_by_call_span"
        or protocol.primary_checkpoint_input
        != "nonempty_contiguous_authenticated_sealed_prefix_starting_at_call_64"
        or protocol.primary_checkpoint_values
        != "finite_unit_interval_nondecreasing_incumbent_hypervolume"
        or protocol.wall_checkpoint_minutes != (0, 15, 30, 45, 60, 75, 90, 105, 120)
        or protocol.primary_value_source
        != "authenticated_hidden_oracle_response_never_method_posterior"
        or protocol.primary_archive_domain
        != (
            "same_run_submitted_identities_passing_canonical_support_exact_"
            "training_overlap_exclusion_accepted_successor_training_homology_"
            "exclusion_and_complete_finite_uncensored_oracle_objectives_and_all_"
            "frozen_constraints_observed_finite_and_pass"
        )
        or protocol.broad_spectrum_role
        != ("secondary_derived_three_sevenths_gram_positive_plus_four_sevenths_gram_negative")
        or protocol.pairwise_sequence_diversity_estimand
        != (
            "mean_one_minus_global_sequence_identity_over_unordered_pairs_of_"
            "valid_unique_reference_safe_sequences"
        )
        or protocol.pairwise_embedding_diversity_estimand
        != (
            "mean_one_minus_cosine_similarity_over_unordered_pairs_of_content_"
            "pinned_embeddings_for_same_sequence_set"
        )
        or protocol.pairwise_diversity_fewer_than_two_value != 0.0
        or protocol.pairwise_embedding_nonfinite_or_zero_norm_value
        != "missing_not_imputed_descriptive_only"
    ):
        raise ValueError("primary two-objective hidden-oracle estimand changed")
    if (
        protocol.method_controlled_seats_per_batch + protocol.random_reserve_seats_per_batch
        != protocol.unique_calls_per_batch
    ):
        raise ValueError("method-controlled and random-reserve seats must fill each batch exactly")
    expected_total = (
        protocol.initial_design_unique_calls
        + protocol.adaptive_batches * protocol.unique_calls_per_batch
    )
    if protocol.total_unique_calls != expected_total:
        raise ValueError("total unique-call budget does not match initial design plus batches")
    if not (
        protocol.common_initial_responses_delivered_before_scientific_clock
        and protocol.common_initial_calls_charged_per_run
        and protocol.scheduled_common_reserve_scope
        == (
            "all_28_batches_times_2_seats_prefrozen_per_seed_before_any_"
            "method_specific_load_or_compute"
        )
        and protocol.scheduled_common_reserve_proposal_exclusion
        == (
            "all_56_scheduled_query_identities_excluded_from_every_arm_method_"
            "controlled_proposal_and_overflow_candidate"
        )
        and protocol.scheduled_common_reserve_submission_rule
        == (
            "submit_the_same_two_prefrozen_query_identities_in_the_same_seat_order_"
            "for_every_arm_at_each_batch"
        )
        and protocol.timed_cross_method_cache_release_policy
        == "physical_cross_arm_cache_disabled_during_timed_adaptive_rounds"
        and protocol.method_controlled_duplicate_rule
        == (
            "skip_exact_query_identity_without_call_or_seat_advance_then_continue_"
            "frozen_proposal_order"
        )
        and protocol.method_controlled_exhaustion_rule
        == (
            "fill_unfilled_method_seats_from_separate_frozen_overflow_reserve_"
            "after_scheduled_common_reserve_seats"
        )
        and protocol.overflow_reserve_scope
        == (
            "separate_prefrozen_per_seed_stream_disjoint_from_all_56_scheduled_"
            "common_reserve_identities"
        )
        and protocol.overflow_reserve_attempt_cap_per_batch == 65536
        and protocol.overflow_reserve_collision_rule
        == "skip_previously_submitted_or_current_batch_query_identity_in_frozen_order"
        and protocol.overflow_reserve_exhaustion_rule
        == "hard_stop_algorithmic_failure_no_unfrozen_refill"
    ):
        raise ValueError("initial response timing or deterministic refill policy changed")
    if protocol.call_checkpoints[0] != 64 or protocol.call_checkpoints[-1] != 512:
        raise ValueError("call checkpoints must span the frozen 64-to-512 design")
    if len(protocol.call_checkpoints) != 29:
        raise ValueError("call checkpoint count must include one initial and 28 adaptive points")
    if protocol.scientific_wall_seconds != 7200:
        raise ValueError("scientific wall-time cap must remain exactly two hours")
    limits = protocol.resource_limits
    if not (
        limits.slurm_account == "bio"
        and limits.cpu_partition == "standard"
        and limits.gpu_partition == "gpumid"
        and limits.gpu_type == "A100"
        and limits.nodes_per_run == 1
        and limits.gpus_per_gpu_run == 1
        and limits.cpus_per_run == 8
        and limits.host_memory_gib == 32
        and limits.max_peak_gpu_memory_gib == 16
        and limits.scientific_wall_seconds == protocol.scientific_wall_seconds == 7200
        and limits.outer_allowance_seconds == 900
        and limits.scientific_elapsed_accounting
        == (
            "sum_authenticated_segment_elapsed_nanoseconds_across_initial_and_all_"
            "resumed_scientific_slurm_jobs_without_reset"
        )
        and limits.outer_allowance_elapsed_accounting
        == (
            "sum_authenticated_segment_elapsed_nanoseconds_across_all_sealing_"
            "slurm_jobs_without_reset"
        )
        and limits.resume_clock_rule
        == (
            "each_receipt_chains_to_the_prior_receipt_and_carries_exact_cumulative_"
            "scientific_and_outer_elapsed_nanoseconds"
        )
        and limits.timing_receipt_authentication
        == ("external_trusted_receipt_required_for_every_initial_or_resumed_slurm_job_segment")
        and limits.outer_allowance_execution == "cpu_only_followup_after_accelerator_release"
        and limits.outer_allowance_accelerator_hours == 0.0
        and limits.scientific_schedule == "seed_blocked_sha256_randomized_simultaneous_waves"
        and limits.scientific_schedule_seed == 20260908
        and limits.scientific_schedule_key
        == (
            "sha256_utf8_decimal_schedule_seed_nul_phase_nul_decimal_run_seed_"
            "nul_configuration_id_ascending_then_configuration_id"
        )
        and limits.scientific_schedule_wave_launch
        == "one_seed_block_at_a_time_each_wave_single_barrier_release"
        and limits.scientific_schedule_wave_size == limits.array_concurrency_cap == 4
        and limits.reproduction_runs == 3
        and limits.reproduction_methods == CONFIRMATION_METHOD_IDS
        and limits.reproduction_seed_count == 1
        and limits.reproduction_seed_status
        == "successor_hidden_independent_reveal_after_scientific_campaign_sealed"
        and limits.reproduction_logical_unique_calls
        == limits.reproduction_runs * protocol.total_unique_calls
        and limits.reproduction_scientific_wall_seconds_per_run == protocol.scientific_wall_seconds
        and not limits.reproduction_results_may_enter_screen_confirmation_or_promotion
        and limits.common_initial_oracle_production_accelerator_accounting_status
        == ("successor_separate_content_pinned_budget_required_outside_166_method_job_ceiling")
    ):
        raise ValueError("timed scheduling, CPU sealing, or reproduction allocation changed")
    batching = protocol.batching_limits
    if not (
        batching.profile == "cluster_batch_first_v1"
        and batching.rollout_batch_size_cap == 128
        and batching.proposal_batch_size_cap == limits.proposal_attempt_cap == 65536
        and batching.surrogate_batch_size_cap == 8192
        and batching.kg_candidate_chunk_size_cap == limits.kg_shortlist_cap == 256
        and batching.kg_fantasy_chunk_size_cap == limits.kg_fantasies == 512
        and batching.oracle_batch_size_cap == limits.oracle_service_batch_cap == 32
        and batching.replay_sequence_batch_cap == 1024
        and batching.replay_token_batch_cap == 32768
        and batching.gradient_accumulation_steps == 8
        and batching.record_realized_sizes
        and batching.seed_order_invariant_to_batching
    ):
        raise ValueError("launcher batching and accumulation limits changed")
    if not (
        limits.kg_joint_q_cap == protocol.method_controlled_seats_per_batch
        and limits.kg_joint_q_cap <= limits.kg_exact_pool_cap <= limits.kg_shortlist_cap
        and math.comb(limits.kg_exact_pool_cap, limits.kg_joint_q_cap) <= limits.kg_max_combinations
        and math.comb(limits.kg_exact_pool_cap + 1, limits.kg_joint_q_cap)
        > limits.kg_max_combinations
    ):
        raise ValueError("exact joint-KG frontier does not match its combination guard")
    expected_confirmation_p = 1.0 / (2 ** len(protocol.confirmation_seeds))
    if not math.isclose(
        protocol.confirmation_exact_sign_test_minimum_p,
        expected_confirmation_p,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("confirmation exact sign-test minimum p-value is inconsistent")
    if protocol.confirmation_pairs_required_above_margin != len(protocol.confirmation_seeds):
        raise ValueError("every confirmation pair must clear the materiality margin")
    if (
        protocol.confirmation_effect != "full_normalized_hv_auc_minus_comparator_normalized_hv_auc"
        or protocol.confirmation_additive_materiality_margin != 0.05
        or protocol.confirmation_comparison_absolute_tolerance != 1e-12
        or protocol.confirmation_pair_success
        != "finite_effect_strictly_greater_than_additive_margin_plus_tolerance"
        or protocol.confirmation_missing_nonfinite_or_tie
        != ("missing_nonfinite_or_effect_at_or_below_margin_plus_tolerance_is_pair_failure")
    ):
        raise ValueError("confirmation additive effect or strict failure rule changed")
    if protocol.confirmation_one_sided_alpha != 0.05:
        raise ValueError("confirmation alpha must remain 0.05")
    if protocol.configured_budget_summary != protocol.budget_summary:
        raise ValueError("declared campaign totals do not match recomputed budgets")
    if (
        limits.hard_evidence_and_reproduction_a100_hour_ceiling
        != protocol.budget_summary.scientific_a100_hour_ceiling
        + limits.reproduction_a100_hour_ceiling
    ):
        raise ValueError("hard A100-hour ceiling must equal evidence plus reproduction")
    if (
        limits.adapter_or_tuning_work_authorized_by_v1
        or limits.adapter_or_tuning_a100_hours_in_v1 != 0.0
        or limits.adapter_or_tuning_logical_calls_in_v1 != 0
        or limits.unallocated_resource_pool_allowed
    ):
        raise ValueError("blocked v1 cannot allocate adapter, tuning, or discretionary resources")
    query = protocol.oracle_query_contract
    if not (
        query.status == "identity_frozen_oracle_semantics_missing_execution_blocking"
        and query.query_identity_fields
        == (
            "canonical_sequence_id",
            "oracle_contract_sha256",
            "evaluator_sha256",
            "checkpoint_sha256",
            "endpoint_context_sha256",
            "transform_sha256",
            "replicate_id",
        )
        and query.all_required_endpoints_return_atomically
        and query.cross_method_cache_hit_is_charged
        and query.exact_same_run_identity_cache_replay_is_not_a_submission
        and query.same_identity_resubmission_after_any_submission_is_forbidden
        and query.transport_retry_may_only_poll_existing_submission
        and query.new_replicate_id_is_new_identity_and_is_charged
        and query.failed_submitted_call_is_charged
        and query.missing_submitted_response_is_charged
        and query.censored_submitted_response_is_charged
        and query.partial_submitted_response_is_charged_and_ineligible
        and query.timeout_after_submission_is_charged
        and not query.failed_or_ineligible_call_may_be_replaced_without_charge
        and query.oracle_contract_sha256_status == "missing_execution_blocking"
        and query.objective_constraint_semantics_status == "missing_execution_blocking"
        and query.missing_censoring_semantics_status == "missing_execution_blocking"
        and not query.timed_cross_method_physical_cache_latency_advantage_allowed
        and not query.timed_cross_method_physical_cache_reuse_allowed
        and query.generator_oracle_training_provenance_status == "missing_execution_blocking"
    ):
        raise ValueError("oracle identity, charging, or blocking contract changed")
    stopping = protocol.stopping_rules
    if not (
        stopping.stop_at_unique_calls_or_wall_seconds_whichever_first
        and stopping.discard_unsealed_partial_batch
        and stopping.carry_last_sealed_incumbent_to_later_checkpoints
        and stopping.infrastructure_reruns_before_first_oracle_response == 1
        and stopping.result_driven_reruns == 0
        and stopping.resume_after_response_from_last_authenticated_round_only
        and stopping.algorithmic_failure_is_retained
        and stopping.nonfinite_or_psd_support_failure == "hard_stop"
        and stopping.forbidden_support_or_overlap == "hard_stop"
        and stopping.unsealed_oracle_response == "hard_stop"
        and stopping.no_kl_ablation_ignores_only_kl_thresholds
        and stopping.kg_tie_or_numerical_instability == "hard_stop"
    ):
        raise ValueError("outcome-blind stopping or rerun policy changed")
    terminal = protocol.terminal_evaluation
    if not (
        terminal.candidate_domain
        == "authenticated_oracle_responded_candidates_submitted_by_same_run_only"
        and terminal.point_estimate
        == (
            "final_sealed_same_run_posterior_arithmetic_mean_of_gram_positive_"
            "and_gram_negative_means"
        )
        and terminal.feasibility
        == "all_successor_content_pinned_oracle_constraints_observed_finite_and_pass"
        and not terminal.oracle_constraint_extension_at_runtime_allowed
        and terminal.candidate_truth_eligibility
        == (
            "canonical_support_exact_training_overlap_exclusion_accepted_successor_"
            "training_homology_exclusion_both_objectives_and_all_constraints_"
            "complete_finite_uncensored"
        )
        and terminal.minimum_eligible_candidates == stopping.required_terminal_eligible_candidates
        and terminal.minimum_eligible_candidates == 100
        and terminal.abstain_if == "fewer_than_minimum_eligible_candidates"
        and terminal.regret_domain
        == (
            "same_run_authenticated_oracle_responded_candidates_passing_candidate_"
            "truth_eligibility_only"
        )
        and terminal.regret_oracle_utility
        == "arithmetic_mean_of_oracle_gram_positive_and_gram_negative"
        and terminal.regret
        == (
            "max_no_action_and_best_feasible_oracle_utility_minus_"
            "recommendation_or_no_action_utility"
        )
        and terminal.empty_feasible_domain_oracle_utility == 0.0
        and terminal.abstention_regret_uses_no_action_utility
        and terminal.missing_censored_or_nonfinite_is_ineligible_not_imputed
    ):
        raise ValueError("terminal recommendation, feasibility, or regret domain changed")
    if not protocol.ablations_are_exploratory:
        raise ValueError("v1 ablations cannot be treated as confirmatory")
    if not (
        protocol.ece_equal_mass_bins == 10
        and protocol.coverage_levels == (0.50, 0.80, 0.90, 0.95)
        and protocol.paired_bootstrap_samples == 10000
        and protocol.paired_bootstrap_seed == 20260908
        and protocol.de_novo_paired_bootstrap_unit
        == ("seed_block_resample_with_same_seed_index_shared_across_all_compared_methods")
        and protocol.run_primary_missing_nonfinite_or_no_initial_checkpoint
        == "required_gate_failure"
        and protocol.early_stop_primary_metric
        == "carry_last_sealed_hypervolume_through_remaining_call_checkpoints"
        and protocol.screen_effect == "full_normalized_hv_auc_minus_comparator_normalized_hv_auc"
        and protocol.screen_full_mean == "arithmetic_mean_over_exact_five_finite_seed_metrics"
        and protocol.screen_full_unique_highest
        == "full_mean_strictly_greater_than_each_other_method_mean_plus_tolerance"
        and protocol.screen_comparison_absolute_tolerance == 1e-12
        and protocol.screen_pair_success
        == "finite_effect_strictly_greater_than_zero_plus_tolerance"
        and protocol.screen_positive_pairs_required == 4
        and protocol.screen_median_additive_margin == 0.05
        and protocol.screen_core_ablation_gate
        == (
            "for_each_ablation_at_least_four_pair_successes_and_median_effect_"
            "strictly_greater_than_margin_plus_tolerance"
        )
        and protocol.confirmation_comparator_gate
        == ("all_five_fixed_seed_pairs_are_successes_and_one_sided_exact_sign_p_at_most_alpha")
        and protocol.confirmation_global_iut_gate == "both_primary_comparator_gates_must_pass"
    ):
        raise ValueError("screen, confirmation, or bootstrap decision contract changed")
    reporting = protocol.statistical_reporting
    if not (
        reporting.paired_bootstrap_statistic
        == "arithmetic_mean_of_five_paired_full_minus_comparator_seed_effects"
        and reporting.paired_bootstrap_interval_level == 0.95
        and reporting.paired_bootstrap_interval_type == "percentile_two_sided"
        and reporting.paired_bootstrap_interval_quantiles == (0.025, 0.975)
        and reporting.paired_bootstrap_quantile_convention
        == (
            "sorted_B_replicates_linear_type7_h_equals_B_minus_1_times_p_"
            "interpolate_between_floor_and_ceil"
        )
        and reporting.hodges_lehmann_estimand
        == (
            "median_of_all_15_walsh_averages_d_i_plus_d_j_over_2_for_1_less_"
            "than_or_equal_to_i_less_than_or_equal_to_j_less_than_or_equal_to_5"
        )
        and reporting.median_convention
        == ("arithmetic_mean_of_two_central_sorted_values_when_even_otherwise_central_sorted_value")
        and reporting.coverage_reporting_scope
        == ("pooled_and_each_seed_for_all_three_confirmation_methods_at_all_four_frozen_levels")
        and reporting.coverage_interval_convention
        == (
            "presubmission_equal_tailed_marginal_predictive_interval_with_"
            "inclusive_endpoints_for_each_query_objective_pair"
        )
        and reporting.coverage_missing_nonfinite_or_wrong_level_count
        == "missing_required_gate_failure"
        and reporting.top10_feasible_mean_utility_checkpoint_estimand
        == (
            "arithmetic_mean_of_up_to_ten_highest_oracle_scalar_utilities_in_the_"
            "primary_eligible_archive_at_each_call_checkpoint_or_no_action_utility_"
            "if_empty"
        )
        and reporting.top10_feasible_mean_utility_auc_estimand
        == (
            "trapezoidal_integral_over_the_same_29_call_checkpoints_divided_by_448_"
            "calls_with_last_sealed_checkpoint_carried_after_early_stop"
        )
        and reporting.wall_time_auc_estimand
        == (
            "trapezoidal_integral_of_sealed_primary_incumbent_hypervolume_at_fixed_"
            "wall_checkpoints_divided_by_120_minutes"
        )
        and reporting.wall_time_checkpoint_assignment
        == (
            "include_only_authenticated_complete_batches_sealed_at_or_before_each_"
            "cumulative_scientific_elapsed_checkpoint_use_zero_before_first_"
            "eligible_archive_value_and_carry_last_value_after_stop"
        )
        and reporting.terminal_abstention_rate_estimand
        == (
            "number_of_abstaining_runs_divided_by_all_exact_required_cohort_runs_"
            "with_any_missing_terminal_record_a_required_gate_failure"
        )
        and reporting.terminal_abstention_error_estimand
        == (
            "among_abstaining_runs_fraction_whose_terminal_regret_domain_contains_"
            "a_feasible_oracle_utility_strictly_above_no_action_utility_zero_if_no_"
            "abstentions"
        )
        and reporting.parent_child_contrast_coverage_estimand
        == (
            "submitted_adaptive_child_identities_with_exactly_one_ledger_bound_"
            "parent_complete_finite_uncensored_eligible_parent_and_child_oracle_"
            "utility_and_finite_presubmission_predicted_contrast_divided_by_all_"
            "such_children_with_eligible_parent_and_child_truth"
        )
        and reporting.parent_child_contrast_accuracy_estimand
        == (
            "covered_pairs_with_predicted_and_realized_child_minus_parent_utility_"
            "signs_both_strictly_outside_1e_minus_12_and_equal_divided_by_covered_"
            "pairs_with_zero_or_tied_sign_counted_incorrect_missing_if_no_covered_"
            "pairs"
        )
    ):
        raise ValueError("statistical interval or secondary metric definitions changed")
    promotion = protocol.promotion_gate
    if not (
        promotion.screen_requires_highest_mean_primary_metric_among_eight_methods
        and promotion.screen_requires_each_core_ablation_positive_pairs == 4
        and promotion.screen_requires_each_core_ablation_median_above_additive_margin
        and promotion.confirmation_requires_all_pairs_above_additive_margin
        and promotion.primary_missing_nonfinite_or_tied_required_gate_is_failure
        and promotion.full_method_kl_replay_numerical_integrity_result_required
        and promotion.yield_and_diversity_gate_comparators == "both_primary_comparators"
        and promotion.yield_and_diversity_pairwise_gate_scope
        == "all_five_confirmation_seed_pairs_for_each_comparator"
        and promotion.max_valid_unique_reference_safe_yield_additive_loss == 0.02
        and promotion.max_hill2_effective_cluster_loss_fraction == 0.10
        and promotion.max_largest_cluster_share_additive_increase == 0.02
        and promotion.secondary_gate_comparison_absolute_tolerance == 1e-12
        and promotion.yield_zero_charged_identity_value == "missing_required_gate_failure"
        and promotion.identity70_metric == "amp_challenge.similarity.global_sequence_identity"
        and promotion.identity70_threshold == 0.70
        and promotion.identity70_linkage
        == "connected_components_single_linkage_edges_at_or_above_threshold"
        and promotion.hill2_zero_valid_sequence_value == 0.0
        and promotion.hill2_zero_comparator_gate == "pass_if_full_is_finite_and_nonnegative"
        and promotion.largest_cluster_share_zero_valid_sequence_value == 0.0
        and promotion.max_ece == 0.10
        and promotion.max_ece_additive_degradation == 0.02
        and promotion.calibration_gate_scope
        == "pooled_authenticated_confirmation_queries_reported_also_per_seed"
        and promotion.ece_empty_or_incomplete_value == "missing_required_gate_failure"
        and promotion.coverage90_empty_or_incomplete_value == "missing_required_gate_failure"
        and promotion.calibration_per_seed_report_required
        and promotion.coverage90_lower == 0.85
        and promotion.coverage90_upper == 0.95
        and promotion.independent_chronological_or_assay_result_required
        and promotion.surrogate_screen_max_shadow_generator_mixture_quota == 0.10
        and promotion.surrogate_screen_max_production_generator_mixture_quota == 0.0
        and not promotion.shadow_candidates_may_enter_submission_or_top100
        and promotion.production_requires_confirmation_and_independent_result
        and promotion.production_quota_requires_new_content_pinned_promotion_protocol
        and not promotion.surrogate_pass_guarantees_final_library_or_top100_seat
        and promotion.no_go_on_failed_confirmation
        and promotion.screen_failure_is_no_go
        and promotion.unsequestered_confirmation_is_no_go
        and promotion.independent_result_missing_or_failure_is_no_go
        and promotion.any_required_gate_failure_is_no_go
        and not promotion.passing_research_gates_authorizes_v1_production
    ):
        raise ValueError("promotion or no-go boundary changed")

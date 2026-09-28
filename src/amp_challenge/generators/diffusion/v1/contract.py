"""Strict immutable contract for native categorical diffusion v1.

V1 is intentionally isolated from the accepted v0 implementation.  This
module only authenticates and types the preregistration; it performs no model
fitting, calibration, sampling, or fold-4 access.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

CONFIG_SHA256 = "4a13952a606e154b652f8bbc1e8f698b5993ee1a41181867c979729709573d40"
ARTIFACT = "native_categorical_diffusion_unconditional_v1"
EVIDENCE_DOC = "docs/benchmarks/native_categorical_diffusion_v1.md"

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "evidence_doc",
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
    }
)
_TABLE_FIELDS = frozenset(_TOP_FIELDS - {"schema_version", "artifact", "evidence_doc"})
_MODEL_NAMES = ("R96", "R128", "D128")
_GATE_NAMES = ("pilot", "denoising", "sampling")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One exact neural architecture in the v1 development matrix."""

    name: str
    output: str
    layers: int
    hidden_dim: int
    attention_heads: int
    ffn_dim: int
    expected_trainable_parameters: int


@dataclass(frozen=True, slots=True)
class NativeDiffusionV1Contract:
    """Typed core values plus a recursively immutable complete document."""

    config_sha256: str
    development_folds: tuple[int, ...]
    locked_holdout_fold: int
    models: tuple[ModelVariant, ...]
    checkpoint_steps: tuple[int, ...]
    residual_lambda_grid: tuple[float, ...]
    temperature_grid: tuple[float, ...]
    sampling_backoff_grid: tuple[float, ...]
    target_slots_per_method_seed: int
    attempts_per_slot: int
    proposal_method_order: tuple[str, ...]
    proposal_seed_order: tuple[int, ...]
    absolute_maximum_a100_hours: float
    document: Mapping[str, object]

    def table(self, name: str) -> Mapping[str, object]:
        """Return one immutable top-level table."""

        value = self.document[name]
        if not isinstance(value, Mapping):  # pragma: no cover - construction invariant
            raise RuntimeError(f"contract field {name!r} is not a table")
        return value


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
            raise ValueError(f"cannot inspect v1 contract path: {candidate}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"v1 contract path traverses a symlink: {candidate}")


def _read_contract_bytes(path: Path) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open v1 diffusion contract: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("v1 diffusion contract must be a non-symlink regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.lstat(source)
        except OSError as error:
            raise ValueError("v1 diffusion contract changed while it was read") from error
        _reject_symlink_chain(source)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _fingerprint(before) != _fingerprint(after)
        or _fingerprint(before) != _fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
        or len(payload) != before.st_size
    ):
        raise ValueError("v1 diffusion contract changed while it was read")
    return payload


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _exact_keys(table: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(table)
    if observed != expected:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" at least {minimum}"
        raise ValueError(f"{label} must be an integer{suffix}")
    return value


def _number(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite TOML float")
    return value


def _tuple_of(
    value: object,
    expected_type: type,
    *,
    label: str,
) -> tuple[Any, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not expected_type for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty {expected_type.__name__} array")
    return tuple(value)


def _integer_matrix(
    value: object,
    *,
    label: str,
    rows: int,
    columns: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} must contain exactly {rows} rows")
    result: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(value):
        if (
            not isinstance(raw_row, list)
            or len(raw_row) != columns
            or any(type(item) is not int or item <= 0 for item in raw_row)
        ):
            raise ValueError(
                f"{label}[{row_index}] must contain exactly {columns} positive integers"
            )
        result.append(tuple(raw_row))
    return tuple(result)


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parameter_count(*, layers: int, hidden: int, ffn: int) -> int:
    embeddings = 180 * hidden
    encoder_layer = 4 * hidden**2 + 2 * hidden * ffn + 9 * hidden + ffn
    return embeddings + layers * encoder_layer + 2 * hidden + 20


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


def _parse_contract(
    document: Mapping[str, Any],
    *,
    config_sha256: str,
) -> NativeDiffusionV1Contract:
    if set(document) != _TOP_FIELDS:
        raise ValueError(
            "v1 top-level schema mismatch: "
            f"missing={sorted(_TOP_FIELDS - set(document))}, "
            f"extra={sorted(set(document) - _TOP_FIELDS)}"
        )
    _validate_recursive(document, label="contract")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("v1 schema_version must equal 1")
    if document["artifact"] != ARTIFACT or document["evidence_doc"] != EVIDENCE_DOC:
        raise ValueError("v1 artifact identity is invalid")
    for name in _TABLE_FIELDS:
        _table(document, name)

    input_table = _table(document, "input")
    development_folds = _tuple_of(
        input_table.get("development_folds"), int, label="input.development_folds"
    )
    holdout = _integer(input_table.get("locked_holdout_fold"), label="locked holdout")
    fold_counts = _tuple_of(
        input_table.get("expected_sequences_by_fold"),
        int,
        label="input.expected_sequences_by_fold",
    )
    input_censuses = {
        field: _integer(input_table.get(field), label=f"input.{field}", minimum=1)
        for field in (
            "expected_sequences",
            "expected_development_sequences",
            "expected_locked_holdout_sequences",
            "expected_development_homology_components",
            "expected_locked_holdout_homology_components",
            "expected_development_union_components",
            "expected_locked_holdout_union_components",
        )
    }
    if development_folds != (0, 1, 2, 3) or holdout != 4 or len(fold_counts) != 5:
        raise ValueError("v1 requires development folds 0..3 and locked fold 4")
    if sum(fold_counts[:4]) != input_censuses["expected_development_sequences"]:
        raise ValueError("development fold counts do not match their census")
    if fold_counts[4] != input_censuses["expected_locked_holdout_sequences"]:
        raise ValueError("locked fold count does not match its census")
    if (
        sum(fold_counts) != input_censuses["expected_sequences"]
        or input_censuses["expected_development_sequences"]
        + input_censuses["expected_locked_holdout_sequences"]
        != input_censuses["expected_sequences"]
    ):
        raise ValueError("v1 sequence censuses are inconsistent")
    for field in (
        "corpus_sha256",
        "training_projection_sha256",
        "union_assignments_sha256",
        "organizer_reference_sha256",
    ):
        _sha256(input_table.get(field), label=f"input.{field}")

    models_table = _table(document, "models")
    _exact_keys(models_table, set(_MODEL_NAMES), label="models")
    variants: list[ModelVariant] = []
    variant_fields = {
        "output",
        "layers",
        "hidden_dim",
        "attention_heads",
        "ffn_dim",
        "expected_trainable_parameters",
    }
    for name in _MODEL_NAMES:
        raw = _table(models_table, name)
        _exact_keys(raw, variant_fields, label=f"models.{name}")
        output = raw["output"]
        if type(output) is not str or not output:
            raise ValueError(f"models.{name}.output must be a non-empty string")
        layers = _integer(raw["layers"], label=f"models.{name}.layers", minimum=1)
        hidden = _integer(raw["hidden_dim"], label=f"models.{name}.hidden_dim", minimum=1)
        heads = _integer(raw["attention_heads"], label=f"models.{name}.heads", minimum=1)
        ffn = _integer(raw["ffn_dim"], label=f"models.{name}.ffn_dim", minimum=1)
        expected = _integer(
            raw["expected_trainable_parameters"],
            label=f"models.{name}.expected_trainable_parameters",
            minimum=1,
        )
        if hidden % heads or expected != _parameter_count(layers=layers, hidden=hidden, ffn=ffn):
            raise ValueError(f"models.{name} architecture or parameter census is invalid")
        variants.append(ModelVariant(name, output, layers, hidden, heads, ffn, expected))

    development = _table(document, "development")
    if tuple(development.get("outer_folds", ())) != development_folds:
        raise ValueError("development outer folds differ from the input boundary")
    score_sequences = _tuple_of(
        development.get("expected_outer_score_sequences"),
        int,
        label="development score sequence census",
    )
    training_sequences = _tuple_of(
        development.get("expected_outer_training_sequences"),
        int,
        label="development training sequence census",
    )
    score_homology = _tuple_of(
        development.get("expected_outer_score_homology_components"),
        int,
        label="development score homology census",
    )
    training_homology = _tuple_of(
        development.get("expected_outer_training_homology_components"),
        int,
        label="development training homology census",
    )
    score_union = _tuple_of(
        development.get("expected_outer_score_union_components"),
        int,
        label="development score union census",
    )
    training_union = _tuple_of(
        development.get("expected_outer_training_union_components"),
        int,
        label="development training union census",
    )
    if (
        score_sequences != fold_counts[:4]
        or training_sequences
        != tuple(
            input_censuses["expected_development_sequences"] - item for item in score_sequences
        )
        or score_homology != (131, 138, 87, 115)
        or training_homology
        != tuple(
            input_censuses["expected_development_homology_components"] - item
            for item in score_homology
        )
        or score_union != (51, 56, 57, 57)
        or training_union
        != tuple(
            input_censuses["expected_development_union_components"] - item for item in score_union
        )
    ):
        raise ValueError("v1 outer development censuses are inconsistent")
    if development.get("calibration_crossfit") != (
        "leave_one_union_component_out_within_each_outer_score_fold"
    ):
        raise ValueError("v1 requires held-out-fold LOUCO calibration")
    expected_calibration_values = {
        "calibration_exclusion_unit": "union_component_id",
        "calibration_component_statistics": (
            "weighted_nll_numerator_and_sampling_mass_by_union_component_timestep_bin_grid_point"
        ),
        "calibration_row_metric": (
            "homology_component_equal_row_weight_times_equal_case_mean_"
            "masked_token_nll_within_timestep_bin"
        ),
        "calibration_sequence_case_reduction": (
            "ascending_level_then_replicate_math_fsum_divided_by_case_count"
        ),
        "calibration_component_reduction": (
            "ascending_sequence_id_math_fsum_for_weighted_nll_numerator_and_sampling_mass"
        ),
        "calibration_total_reduction": "ascending_union_component_id_math_fsum",
        "calibration_leaveout_reduction": "math_fsum_total_then_negative_heldout_component",
        "calibration_leaveout_objective": (
            "leaveout_weighted_nll_numerator_divided_by_leaveout_sampling_mass"
        ),
        "calibration_choices_are": "score_only_not_deployed_or_averaged",
    }
    if any(
        development.get(field) != expected
        for field, expected in expected_calibration_values.items()
    ):
        raise ValueError("v1 LOUCO calibration semantics are not frozen")
    if development.get("no_scored_token_selects_its_calibration") is not True:
        raise ValueError("calibration must not score its own fitting tokens")
    selected_token_totals = _tuple_of(
        development.get("expected_score_selected_tokens_per_replicate_by_fold"),
        int,
        label="development selected-token fold census",
    )
    if selected_token_totals != (
        272385,
        163489,
        134208,
        144102,
    ):
        raise ValueError("v1 selected-token fold census is invalid")
    selected_token_bins = _integer_matrix(
        development.get("expected_score_selected_tokens_per_replicate_by_fold_bin"),
        label="development selected-token bin census",
        rows=4,
        columns=4,
    )
    if (
        selected_token_bins
        != (
            (10338, 45080, 92494, 124473),
            (6457, 27123, 55465, 74444),
            (5551, 22377, 45458, 60822),
            (5808, 23981, 48847, 65466),
        )
        or tuple(map(sum, selected_token_bins)) != selected_token_totals
    ):
        raise ValueError("v1 selected-token bin census is invalid")
    expected_louco_support = {
        "minimum_louco_union_components_by_fold": (50, 55, 56, 56),
        "minimum_louco_homology_components_by_fold": (59, 123, 82, 103),
        "minimum_louco_selected_tokens_per_replicate_by_fold": (
            56806,
            149167,
            111083,
            126086,
        ),
    }
    for field, expected in expected_louco_support.items():
        observed = _tuple_of(
            development.get(field),
            int,
            label=f"development {field}",
        )
        if observed != expected:
            raise ValueError("v1 minimum LOUCO support census is invalid")
    minimum_louco_bins = _integer_matrix(
        development.get("minimum_louco_selected_tokens_per_replicate_by_fold_bin"),
        label="development minimum LOUCO selected-token bin census",
        rows=4,
        columns=4,
    )
    if (
        minimum_louco_bins
        != (
            (2173, 9399, 19288, 25946),
            (5902, 24746, 50605, 67914),
            (4546, 18508, 37643, 50386),
            (5032, 20971, 42751, 57332),
        )
        or tuple(map(sum, minimum_louco_bins))
        != expected_louco_support["minimum_louco_selected_tokens_per_replicate_by_fold"]
    ):
        raise ValueError("v1 minimum LOUCO selected-token bin census is invalid")
    expected_selected_token_rules = {
        "selected_token_count_definition": (
            "sum_levels_1_to_64_of_fixed_cosine_mask_count_per_sequence_per_replicate"
        ),
        "selected_token_mask_probability": (
            "clip_one_minus_cosine_alpha_bar_offset_0p008_at_level_divided_by_64"
        ),
        "selected_token_mask_count": (
            "minimum_length_maximum_one_ceil_length_times_mask_probability"
        ),
    }
    if any(
        development.get(field) != expected
        for field, expected in expected_selected_token_rules.items()
    ):
        raise ValueError("v1 selected-token definition is invalid")
    expected_bootstrap_values = {
        "bootstrap_draw_rule": (
            "resample_observed_union_component_count_independently_with_"
            "replacement_within_each_outer_fold"
        ),
        "bootstrap_row_weight_rule": (
            "multiply_frozen_full_fold_row_weight_by_union_component_draw_"
            "multiplicity_then_renormalize_within_fold"
        ),
        "bootstrap_fold_aggregation": "equal_arithmetic_mean_over_four_outer_folds",
        "bootstrap_standard_error": (
            "sample_standard_deviation_ddof1_of_bootstrap_mean_nll_without_"
            "division_by_sqrt_replicates"
        ),
        "bootstrap_rng": "numpy_pcg64dxsm_2.4.6",
        "bootstrap_rng_key": (
            "sha256_namespace_to_uint64_v1_bootstrap_seed_parent_contract_"
            "sha256_replicate_outer_fold"
        ),
        "bootstrap_component_order": "ascending_union_component_id",
        "bootstrap_draw_sharing": (
            "same_outer_fold_draws_shared_across_candidates_comparator_checkpoints_and_seeds"
        ),
        "bootstrap_relative_improvement": (
            "comparator_nll_minus_candidate_nll_divided_by_comparator_nll"
        ),
        "bootstrap_confidence_interval": "numpy_quantile_0p025_and_0p975_method_linear",
    }
    if any(
        development.get(field) != expected for field, expected in expected_bootstrap_values.items()
    ):
        raise ValueError("v1 stratified bootstrap semantics are not frozen")
    if development.get("bootstrap_recalibrates_each_draw") is not False:
        raise ValueError("v1 bootstrap must keep LOUCO choices fixed")
    if development.get("bootstrap_candidate_comparator_draws_shared") is not True:
        raise ValueError("v1 candidate and comparator must share bootstrap draws")
    if development.get("bootstrap_unit") != "union_component_id_stratified_by_fold":
        raise ValueError("v1 bootstrap unit is invalid")
    if (
        _integer(
            development.get("bootstrap_replicates"),
            label="development.bootstrap_replicates",
            minimum=1,
        )
        != 10000
    ):
        raise ValueError("v1 bootstrap replicate count is invalid")
    if (
        _integer(
            development.get("bootstrap_seed"),
            label="development.bootstrap_seed",
            minimum=0,
        )
        != 20260905
    ):
        raise ValueError("v1 bootstrap seed is invalid")
    if (
        _integer(
            development.get("pilot_corruption_replicates_per_sequence_level"),
            label="development.pilot_corruption_replicates_per_sequence_level",
            minimum=1,
        )
        != 1
        or _integer(
            development.get("full_corruption_replicates_per_sequence_level"),
            label="development.full_corruption_replicates_per_sequence_level",
            minimum=1,
        )
        != 4
    ):
        raise ValueError("v1 corruption replicate counts are invalid")

    training = _table(document, "training")
    checkpoints = _tuple_of(
        training.get("checkpoint_steps"), int, label="training.checkpoint_steps"
    )
    if checkpoints != (250, 500, 1000, 2000, 4000):
        raise ValueError("v1 checkpoint grid is invalid")
    if training.get("max_steps") != checkpoints[-1]:
        raise ValueError("training max_steps must equal the final checkpoint")
    if training.get("residual_training_lambda") != 1.0:
        raise ValueError("residual checkpoints must train at lambda one")
    if training.get("residual_training_temperature") != 1.0:
        raise ValueError("residual checkpoints must train at temperature one")
    expected_rng_keys = {
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
    }
    if any(training.get(field) != expected for field, expected in expected_rng_keys.items()):
        raise ValueError("v1 dropout RNG policy is not frozen")

    calibration = _table(document, "calibration")
    lambdas = _tuple_of(
        calibration.get("residual_lambda_grid"), float, label="residual lambda grid"
    )
    temperatures = _tuple_of(calibration.get("temperature_grid"), float, label="temperature grid")
    if lambdas != (0.125, 0.25, 0.5, 0.75, 1.0) or any(value <= 0.0 for value in lambdas):
        raise ValueError("eligible residual scales must be the frozen positive grid")
    if temperatures != (1.0, 1.25, 1.5, 2.0, 3.0):
        raise ValueError("temperature grid is invalid")
    if calibration.get("zero_lambda_report_only") is not True:
        raise ValueError("zero residual must remain report-only")
    expected_metric_values = {
        "selection_metric": "component_balanced_masked_token_nll_nats",
        "case_metric_reduction": "mean_nll_over_scheduled_masked_tokens",
        "sequence_metric_reduction": "equal_mean_over_levels_and_replicates",
        "calibration_fit_weighting": (
            "accepted_outer_score_row_weights_renormalized_after_union_component_exclusion"
        ),
        "cross_calibrated_score_weighting": ("homology_component_equal_full_outer_score_fold"),
        "outer_fold_metric_reduction": "equal_arithmetic_mean",
        "selection_tie_break": (
            "lower_lambda_then_temperature_closest_to_one_then_lower_temperature"
        ),
        "strongest_count_control_rule": "lower_cross_calibrated_nll_of_C0_and_C0T",
        "final_calibration_fit": (
            "pooled_seed42_four_fold_oof_after_model_selection_before_recipe_freeze"
        ),
        "final_calibration_fit_weighting": (
            "equal_outer_fold_then_homology_component_equal_within_fold"
        ),
        "ece_prediction": "maximum_probability_with_lowest_residue_index_on_exact_tie",
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
    }
    if any(
        calibration.get(field) != expected for field, expected in expected_metric_values.items()
    ):
        raise ValueError("v1 calibration metric or reduction semantics are not frozen")
    if calibration.get("strongest_count_control_tie_break") != "C0":
        raise ValueError("v1 count-control tie break must select C0")
    if calibration.get("ece_bins") != 15:
        raise ValueError("v1 ECE must use exactly 15 bins")
    if calibration.get("strongest_count_control_scope") != (
        "one_global_method_reused_for_fold_timestep_bin_ece_and_bootstrap_gates"
    ):
        raise ValueError("v1 must reuse one global count comparator for every gate")
    if calibration.get("louco_choices_contribute_to_final_calibration") is not False:
        raise ValueError("LOUCO choices must not become deployable calibration")
    if calibration.get("fold4_may_change_calibration") is not False:
        raise ValueError("fold 4 must not change calibration")

    recipe = _table(document, "recipe_selection")
    expected_freeze_order = (
        "cross_calibrated_model_and_checkpoint_selection",
        "pooled_seed42_oof_calibration_fit",
        "seed42_oof_sampling_backoff_selection",
        "sealed_recipe_freeze_receipt",
        "seed43_44_confirmation",
    )
    if tuple(recipe.get("freeze_order", ())) != expected_freeze_order:
        raise ValueError("recipe freeze chronology is invalid")
    if (
        _tuple_of(
            recipe.get("final_fit_folds"),
            int,
            label="recipe_selection.final_fit_folds",
        )
        != development_folds
    ):
        raise ValueError("v1 final fit folds differ from development folds")
    if recipe.get("eligibility_threshold") != (
        "best_observed_mean_nll_plus_standard_error_of_best_choice"
    ):
        raise ValueError("one-standard-error threshold is not frozen")

    sampling = _table(document, "sampling")
    backoff = _tuple_of(
        sampling.get("probability_backoff_epsilon_grid"),
        float,
        label="sampling backoff grid",
    )
    if backoff != (0.0, 0.02, 0.05):
        raise ValueError("sampling backoff grid is invalid")
    slots = _integer(
        sampling.get("target_slots_per_method_seed"), label="sampling target slots", minimum=1
    )
    attempts = _integer(
        sampling.get("attempts_per_slot"), label="sampling attempts per slot", minimum=1
    )
    attempt_indices = _tuple_of(
        sampling.get("attempt_indices"), int, label="sampling attempt indices"
    )
    if attempt_indices != tuple(range(attempts)):
        raise ValueError("sampling attempt indices must be contiguous from zero")
    if sampling.get("attempt_grid_sequences_per_method_seed") != slots * attempts:
        raise ValueError("sampling attempt-grid census is invalid")
    methods = _tuple_of(
        sampling.get("proposal_method_order"), str, label="sampling proposal methods"
    )
    seeds = _tuple_of(sampling.get("proposal_seed_order"), int, label="sampling seed order")
    if methods != (
        "native_categorical_diffusion_v1",
        "component_weighted_unigram",
        "component_weighted_forward_markov",
    ) or seeds != (42, 43, 44):
        raise ValueError("sampling method or seed order is invalid")
    if sampling.get("organizer_reference_may_select_backoff") is not False:
        raise ValueError("organizer reference must not select sampling backoff")
    if sampling.get("seal_all_method_seed_attempt_grids_before_reference_parse") is not True:
        raise ValueError("all proposal grids must be sealed before reference parsing")
    if sampling.get("acceptance_pool_scope") != "independent_within_each_proposal_method_and_seed":
        raise ValueError("bounded-fill pool scope is invalid")

    gates = _table(document, "gates")
    _exact_keys(gates, set(_GATE_NAMES), label="gates")
    pilot_gate = _table(gates, "pilot")
    if pilot_gate.get("ece_scope") != "every_outer_fold_and_equal_fold_aggregate":
        raise ValueError("pilot ECE scope must include every fold and the aggregate")
    denoising_gate = _table(gates, "denoising")
    if denoising_gate.get("bootstrap_seed_aggregation") != (
        "arithmetic_mean_of_per_seed_relative_nll_improvements"
    ):
        raise ValueError("denoising seed aggregation is invalid")
    if denoising_gate.get("high_noise_seed_aggregation") != "arithmetic_mean_nll":
        raise ValueError("high-noise seed aggregation is invalid")
    if denoising_gate.get("timestep_bin_seed_aggregation") != "arithmetic_mean_nll":
        raise ValueError("timestep-bin seed aggregation is invalid")

    confirmation = _table(document, "confirmation")
    if confirmation.get("may_change_recipe") is not False:
        raise ValueError("confirmation must not retune the recipe")
    if tuple(confirmation.get("seeds", ())) != (43, 44):
        raise ValueError("confirmation seeds are invalid")

    leakage = _table(document, "leakage")
    forbidden_true = (
        "fold4_visible_before_unlock",
        "fold4_used_for_architecture",
        "fold4_used_for_horizon",
        "fold4_used_for_calibration",
        "fold4_used_for_sampler",
        "fold4_used_for_seed_choice",
        "labels_allowed",
        "provenance_allowed",
        "study_keys_allowed",
        "oracle_predictions_allowed",
        "structures_allowed",
        "organizer_reference_allowed_during_training",
        "organizer_reference_allowed_during_attempt_grid_generation",
    )
    if any(leakage.get(field) is not False for field in forbidden_true):
        raise ValueError("v1 leakage boundary enables a forbidden input")
    if leakage.get("post_fold4_changes_require_new_version") is not True:
        raise ValueError("post-fold-4 changes must require a new version")

    compute = _table(document, "compute")
    pilot_hours = _number(compute.get("pilot_maximum_a100_hours"), label="pilot hours")
    development_hours = _number(
        compute.get("development_and_confirmation_maximum_a100_hours"),
        label="development hours",
    )
    final_hours = _number(
        compute.get("final_fit_and_evaluation_maximum_a100_hours"), label="final hours"
    )
    total_hours = _number(compute.get("absolute_maximum_a100_hours"), label="absolute A100 hours")
    if not math.isclose(pilot_hours + development_hours + final_hours, total_hours):
        raise ValueError("v1 phase GPU budgets do not equal the absolute cap")
    if compute.get("bare_exclusive_allowed") is not False:
        raise ValueError("bare Slurm exclusivity must remain forbidden")

    determinism = _table(document, "determinism")
    if _tuple_of(
        determinism.get("rng_namespaces"),
        str,
        label="determinism.rng_namespaces",
    ) != (
        "initialization",
        "minibatch",
        "timestep",
        "corruption",
        "context_dropout",
        "model_dropout",
        "bootstrap",
        "proposal",
        "attempt",
    ):
        raise ValueError("v1 RNG namespaces are incomplete or reordered")
    if _tuple_of(
        determinism.get("fit_identity_fields"),
        str,
        label="determinism.fit_identity_fields",
    ) != (
        "parent_contract_sha256",
        "variant",
        "output_mode",
        "seed",
        "fit_folds",
        "fit_projection_sha256",
    ):
        raise ValueError("v1 fit identity fields are not frozen")
    expected_identity_values = {
        "fit_identity_fold_scope": ("ascending_integer_array_of_exact_trainer_visible_folds"),
        "fit_identity_serialization": (
            "utf8_canonical_json_sorted_keys_compact_separators_allow_nan_false_single_lf"
        ),
        "fit_identity_digest": "lowercase_sha256",
        "global_draw_ordinal": (
            "zero_based_optimizer_step_minus_one_times_batch_sequences_plus_"
            "zero_based_batch_row_index"
        ),
    }
    if any(
        determinism.get(field) != expected for field, expected in expected_identity_values.items()
    ):
        raise ValueError("v1 fit identity or draw ordinal is not frozen")
    if determinism.get("operational_telemetry_is_outside_semantic_twin_bundle") is not True:
        raise ValueError("node telemetry must remain outside semantic twin equality")

    frozen = _freeze(document)
    if not isinstance(frozen, Mapping):  # pragma: no cover - construction invariant
        raise RuntimeError("frozen contract is not a mapping")
    return NativeDiffusionV1Contract(
        config_sha256=config_sha256,
        development_folds=development_folds,
        locked_holdout_fold=holdout,
        models=tuple(variants),
        checkpoint_steps=checkpoints,
        residual_lambda_grid=lambdas,
        temperature_grid=temperatures,
        sampling_backoff_grid=backoff,
        target_slots_per_method_seed=slots,
        attempts_per_slot=attempts,
        proposal_method_order=methods,
        proposal_seed_order=seeds,
        absolute_maximum_a100_hours=total_hours,
        document=frozen,
    )


def load_unconditional_v1_contract(path: str | os.PathLike[str]) -> NativeDiffusionV1Contract:
    """Authenticate, parse, validate, and freeze the exact v1 preregistration."""

    payload = _read_contract_bytes(Path(path))
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("v1 diffusion contract is not valid UTF-8 TOML") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise ValueError(f"v1 diffusion contract SHA-256 mismatch: {digest}")
    return _parse_contract(document, config_sha256=digest)

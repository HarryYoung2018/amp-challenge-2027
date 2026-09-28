"""Actual-native evolutionary collection, charged credit and endpoint updates.

The outer controller authenticates history, exclusions, timing and eligibility.
There is deliberately no oracle evaluator, network transport or production gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    _NativeUnit,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_endpoint import _json_hash, _seed
from amp_challenge.generators.diffusion.native_evolution_acquisition import (
    EvolutionSelection,
    select_evolution_queries,
)
from amp_challenge.generators.diffusion.native_evolution_math import (
    allocate_branches,
    contrast_credit_value,
    contrast_from_moments,
    operator_plan,
    paired_credit,
)
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureCache,
    FrozenEvolutionPosterior,
)
from amp_challenge.generators.diffusion.native_evolution_records import (
    DIRECTIONS,
    EVOLUTION_CONFIG_SHA256,
    NO_COUNTERFACTUAL_CONFIG_PATH,
    NO_COUNTERFACTUAL_CONFIG_SHA256,
    OPERATORS,
    EvolutionAllocation,
    EvolutionAttempt,
    EvolutionBranch,
    EvolutionBudgetExceeded,
    EvolutionDeadline,
    EvolutionNoEligibleParent,
    EvolutionVariant,
    EvolutionWave,
    evolution_configuration_sha256,
    semantic_history,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def evolution_source():
    root = Path(__file__).resolve().parents[4]
    names = [
        "configs/diffusion/native_counterfactual_softkg_v1.toml",
        NO_COUNTERFACTUAL_CONFIG_PATH,
    ] + [
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            "native_evolution",
            "native_evolution_records",
            "native_evolution_math",
            "native_evolution_posterior",
            "native_evolution_acquisition",
            "native_acquisition_precision",
            "native_evolution_operator_kl",
            "native_evolution_teacher",
            "native_replay_teacher",
            "native_evolution_verify",
            "native_evolution_teacher_verify",
            "native_evolution_acquisition_verify",
            "native_evolution_driver",
            "native_proposals",
            "native_endpoint",
            "native_baseline_operators",
            "native_weighted_training",
            "model",
            "categorical",
            "replay",
            "subset_kernel",
            "native_shared_endpoint",
            "native_shared_endpoint_records",
            "native_shared_endpoint_verify",
            "native_matched_feasibility",
            "native_matched_feasibility_verify",
            "native_proxy_distance_update",
            "native_proxy_policy_sampler",
            "distribution_policy_mixture",
            "distribution_distance_study",
        )
    ]
    names.extend(
        "src/amp_challenge/" + name
        for name in (
            "acquisition/soft_kg.py",
            "acquisition/counterfactual.py",
            "models/feature_posterior.py",
            "models/charged_probability_learner.py",
            "workflows/peptide_distance_calibration.py",
            "workflows/native_proxy_evolution.py",
            "workflows/cpu_proxy_features.py",
            "representations/cpu_native_features.py",
            "generators/search/verified_charged_history.py",
            "generators/search/native_evolution_campaign.py",
            "generators/search/peptide_ga_tunable_v2_records.py",
        )
    )
    values = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    if values[names[0]] != EVOLUTION_CONFIG_SHA256:
        raise ValueError("native evolution predeclared configuration changed")
    if values[NO_COUNTERFACTUAL_CONFIG_PATH] != NO_COUNTERFACTUAL_CONFIG_SHA256:
        raise ValueError("native no-counterfactual amendment changed")
    return values


class NativeCounterfactualEnsemble:
    def __init__(
        self,
        initializations,
        corpora,
        context: NormalizedObjectiveContext,
        *,
        run_id: str,
        seed: int,
        oracle_bundle_sha256: str,
        cache: EvolutionFeatureCache,
        variant: EvolutionVariant | None = None,
        prospective_protocol_sha256: str | None = None,
        max_rounds: int = 28,
        proxy_distance: tuple[str, float] | None = None,
        fixed_budget: bool = False,
        model_updates_disabled: bool = False,
        replay_teacher: bool = False,
        precision_recheck: bool = False,
        policy_storage=None,
        allow_small_teacher: bool = False,
    ):
        variant = EvolutionVariant() if variant is None else variant
        if type(model_updates_disabled) is not bool or (
            model_updates_disabled
            and (
                variant.name != "full"
                or proxy_distance is None
                or not hash_string(prospective_protocol_sha256)
            )
        ):
            raise ValueError("disabled model updates require the prospective full distance method")
        self.model_updates_disabled = model_updates_disabled
        self.allow_small_teacher = allow_small_teacher
        if type(fixed_budget) is not bool or (
            fixed_budget
            and (variant.name != "full" or not hash_string(prospective_protocol_sha256))
        ):
            raise ValueError("fixed-budget allocation requires a bound prospective full method")
        self.fixed_budget = fixed_budget
        if type(precision_recheck) is not bool or (precision_recheck and not replay_teacher):
            raise ValueError("precision recheck requires the explicit replay-teacher intervention")
        self.precision_recheck = precision_recheck
        if type(replay_teacher) is not bool or (
            replay_teacher
            and (not fixed_budget or proxy_distance is None or variant.name != "full")
        ):
            raise ValueError("replay teacher requires the prospective fixed-budget distance method")
        self.replay_teacher = None
        if replay_teacher:
            from amp_challenge.generators.diffusion.native_replay_teacher import (
                RequalifiedReplayTeacher,
            )

            self.replay_teacher = RequalifiedReplayTeacher()
        if (
            max_rounds != 28
            or proxy_distance is not None
            or prospective_protocol_sha256 is not None
        ) and (max_rounds != 64 or not hash_string(prospective_protocol_sha256)):
            raise ValueError("prospective native changes require a bound 64-round protocol")
        if proxy_distance is not None and variant.name != "full":
            raise ValueError("prospective distance treatment requires the full native variant")
        self.max_rounds = max_rounds
        self.prospective_protocol_sha256 = prospective_protocol_sha256
        self.proxy_distance = proxy_distance
        if (
            len(initializations) != 10
            or len(corpora) != 10
            or tuple(item.triple for item in initializations) != TRIPLES
        ):
            raise ValueError("full native evolution requires all ten ordered initializations")
        if (
            not run_id
            or type(seed) is not int
            or not 0 <= seed < 2**63
            or not hash_string(oracle_bundle_sha256)
            or type(context) is not NormalizedObjectiveContext
        ):
            raise ValueError("native evolution run/context binding differs")
        if (
            type(cache) is not EvolutionFeatureCache
            or cache.binding.representation != variant.representation
        ):
            raise ValueError("evolution variant feature representation differs")
        self.units = tuple(
            _NativeUnit(init, corpus) for init, corpus in zip(initializations, corpora, strict=True)
        )
        if len({unit.model.config for unit in self.units}) != 1:
            raise ValueError("all ten native architectures must agree")
        self.context, self.run_id, self.seed, self.bundle = (
            context,
            run_id,
            seed,
            oracle_bundle_sha256,
        )
        self.cache, self.variant = cache, variant
        self.training_ids = frozenset(
            key for unit in self.units for key in unit.initialization.training_sequence_ids
        )
        self.history = self.posterior = self.wave = self.selection = self.deadline = None
        self.branches = ()
        self.versions = {triple: 0 for triple in TRIPLES}
        self.source = evolution_source()
        self.update_records = []
        self.ingest_attempts = {}
        self._seen = {}
        self.policy_sampler = None
        if proxy_distance is not None:
            from amp_challenge.generators.diffusion.native_proxy_policy_sampler import (
                NativeProxyPolicySampler,
            )

            self.policy_sampler = NativeProxyPolicySampler(self.units, storage_root=policy_storage)

    def _check(self):
        if evolution_source() != self.source:
            raise ValueError("evolution numerical source changed")
        for unit in self.units:
            unit.check()
        if self.policy_sampler is not None:
            self.policy_sampler.check()

    @property
    def policy_identities(self):
        return tuple((unit.triple, unit.policy_sha256) for unit in self.units)

    def _sample_proposals(self, unit, parents, *, start_levels, ordinals):
        """Keep the legacy sampler unchanged; prospective runs bind real mixtures."""
        if self.policy_sampler is None:
            return sample_native_proposals(
                unit.model, parents, start_levels=start_levels, seed=self.seed, ordinals=ordinals
            )
        return self.policy_sampler.sample(
            unit, parents, start_levels=start_levels, seed=self.seed, ordinals=ordinals
        )

    def ingest(
        self,
        history,
        posterior,
        *,
        expected_previous_head_sha256,
        eligible_charged_ids: frozenset[str],
        outer_deadline: float,
        feasible=None,
        feasibility_source_sha256=None,
    ):
        """Consume one already authenticated prefix; incomplete history pauses.

        eligible_charged_ids are controller-authenticated archive eligibility,
        not a method-posterior prediction or an independently issued authority.
        outer_deadline must include the original controller wave start, e.g.
        min(scientific_deadline, controller_wave_started + 180). It must not be
        reset after the upstream learner fit or a partial method handoff.
        """
        self._check()
        if (feasible is None) != (feasibility_source_sha256 is None) or (
            feasible is not None
            and (
                not callable(feasible)
                or not hash_string(feasibility_source_sha256)
                or getattr(feasible, "source_sha256", None) != feasibility_source_sha256
            )
        ):
            raise ValueError("ingest matched public feasibility identity differs")
        if (
            type(history) is not VerifiedHistorySnapshot
            or type(posterior) is not FrozenEvolutionPosterior
        ):
            raise ValueError("explicit verified history and frozen learner required")
        history.__post_init__()
        expected_initial = 512 if self.prospective_protocol_sha256 is not None else 64
        if (history.initial_charge_count, history.max_rounds, history.charges_per_round) != (
            expected_initial,
            self.max_rounds,
            16,
        ):
            raise ValueError("native history differs from bound initialization and adaptive budget")
        if history.max_rounds != self.max_rounds:
            raise ValueError("history and native ensemble round budgets differ")
        if (
            history.run_id,
            history.seed,
            history.oracle_bundle_sha256,
            history.objective_context_sha256,
            history.previous_wave_head_sha256,
        ) != (
            self.run_id,
            self.seed,
            self.bundle,
            self.context.context_sha256,
            expected_previous_head_sha256,
        ):
            raise ValueError("evolution history run/source/context/head differs")
        if (
            posterior.history_sha256 != history.sha256
            or posterior.context_sha256 != self.context.context_sha256
            or posterior.feature_binding != self.cache.binding
        ):
            raise ValueError("frozen learner does not bind current charged prefix/features")
        if type(eligible_charged_ids) is not frozenset or not eligible_charged_ids <= {
            sequence_id(row.sequence) for row in history.observations if row.status == "successful"
        }:
            raise ValueError("charged eligibility must be an explicit successful-prefix subset")
        if not history.complete:
            return "paused_incomplete_charged_prefix"
        if self.history is not None:
            if history.sha256 == self.history.sha256:
                if (
                    posterior.sha256 != self.posterior.sha256
                    or eligible_charged_ids != self.eligible_charged_ids
                ):
                    raise ValueError("same charged prefix cannot replace frozen learner")
                return "already_ingested"
            if (
                history.round_index != self.history.round_index + 1
                or history.observations[: len(self.history.observations)]
                != self.history.observations
            ):
                raise ValueError("evolution history is not next contiguous terminal wave")
            if self.wave is None or self.selection is None or self.selection.status != "selected":
                raise ValueError("new charged prefix lacks sealed method selection")
            submitted = {sequence_id(row.sequence) for row in history.observations[-16:]}
            selected = {
                sequence_id(self.wave.attempts[index].trace.endpoint)
                for index in self.selection.selected_ordinals
            }
            if not selected <= submitted:
                raise ValueError("new charges do not contain all fourteen selected identities")
        elif history.round_index != 1:
            raise ValueError(
                "initial evolution history must contain only the bound shared initialization"
            )
        deadline = EvolutionDeadline(outer_deadline)
        deadline.check("before_ingest")
        posterior.check()
        ingest_key = semantic_history(history, eligible_charged_ids)
        if ingest_key in self.ingest_attempts:
            raise ValueError("interrupted ingest cannot restart its neural/resource budget")
        attempt_record = {
            "history_sha256": history.sha256,
            "semantic_sha256": ingest_key,
            "status": "started",
            "deadline": deadline.deadline,
        }
        self.ingest_attempts[ingest_key] = attempt_record
        replacement_units, replacement_versions, replacement_branches = (
            self.units,
            self.versions,
            self.branches,
        )
        try:
            if self.history is not None:
                replacement_units, replacement_versions, replacement_branches = (
                    self._post_response_update(
                        history,
                        posterior,
                        eligible_charged_ids,
                        deadline,
                        feasible,
                        feasibility_source_sha256,
                    )
                )
            self._check()
            posterior.check()
            deadline.check("before_ingest_atomic_commit")
        except (TimeoutError, ValueError, FloatingPointError, RuntimeError, TypeError) as error:
            attempt_record.update(
                status="stopped_no_commit", failure=f"{type(error).__name__}: {error}"[:512]
            )
            raise
        self.units, self.versions, self.branches = (
            replacement_units,
            replacement_versions,
            replacement_branches,
        )
        self.history, self.posterior, self.deadline = history, posterior, deadline
        self.wave = self.selection = None
        self.cache.begin_wave()
        self.eligible_charged_ids = eligible_charged_ids
        attempt_record["status"] = "committed"
        return "terminal_history" if history.round_index == self.max_rounds + 1 else "ready"

    def _post_response_update(
        self, history, posterior, eligible, deadline, feasible=None, feasibility_source_sha256=None
    ):
        by_sequence = {
            attempt.trace.endpoint: attempt
            for attempt in self.wave.attempts
            if attempt.rejection is None
        }
        branches = {
            (branch.triple, branch.branch): replace(branch, credit=1 + (branch.credit - 1) * 0.9)
            for branch in self.branches
        }
        for observed in history.observations[-16:]:
            attempt = by_sequence.get(observed.sequence)
            if attempt is None:
                continue
            key = (attempt.triple, attempt.branch)
            branch = branches[key]
            useful = (
                observed.status == "successful"
                and sequence_id(observed.sequence) in eligible
                and self.context.scalarize(observed.objectives) > 0.5
            )
            credit = 1.0
            if self.variant.name != "no_counterfactual":
                contrast = paired_credit(
                    posterior.joint(self.cache.matrix((observed.sequence, attempt.lineage_parent))),
                    feasible=bool(useful),
                )
                credit = float(
                    np.clip(
                        0.9 * branch.credit + 0.1 * contrast_credit_value(contrast.advantage),
                        0.25,
                        4,
                    )
                )
            branches[key] = replace(
                branch,
                charged_descendants=branch.charged_descendants + 1,
                useful_descendants=branch.useful_descendants + int(useful),
                credit=credit,
            )
        replacement_units, replacement_versions = self.units, self.versions
        # Do not train an unused terminal policy after the last adaptive charge.
        if history.round_index <= self.max_rounds and self.variant.name != "no_endpoint":
            from amp_challenge.generators.diffusion.native_evolution_operator_kl import (
                EvolutionOperatorGuard,
            )
            from amp_challenge.generators.diffusion.native_evolution_teacher import (
                build_evolution_teacher,
            )
            from amp_challenge.generators.diffusion.native_shared_endpoint import (
                update_shared_endpoints,
            )

            teacher_builder = (
                build_evolution_teacher
                if self.replay_teacher is None
                else self.replay_teacher.build
            )
            teacher, target_record = teacher_builder(
                self.wave,
                posterior,
                self.cache,
                generation=history.round_index - 1,
                max_generations=self.max_rounds,
                prospective_protocol_sha256=self.prospective_protocol_sha256,
            )
            if self.model_updates_disabled:
                self._record_disabled_model_update(teacher, target_record)
                deadline.check("after_disabled_endpoint_decision")
                return (
                    replacement_units,
                    replacement_versions,
                    tuple(
                        branches[(unit.triple, index)] for unit in self.units for index in range(4)
                    ),
                )
            guard = (
                EvolutionOperatorGuard(self.units, self.wave, seed=self.seed, deadline=deadline)
                if self.proxy_distance is None
                else None
            )
            from amp_challenge.generators.diffusion.native_matched_feasibility import (
                MatchedFeasibilityRequirement,
                matched_feasibility_source,
            )

            feasibility_requirement = (
                None
                if feasible is None or self.proxy_distance is not None
                else MatchedFeasibilityRequirement(
                    self.wave,
                    feasible,
                    feasibility_source_sha256,
                    self.context.context_sha256,
                    matched_feasibility_source(),
                )
            )
            if self.proxy_distance is not None:
                from amp_challenge.generators.diffusion.native_proxy_distance_update import (
                    update_proxy_distance_endpoints,
                )

                units, update = update_proxy_distance_endpoints(
                    self.units,
                    teacher,
                    seed=self.seed,
                    deadline=deadline.deadline,
                    metric=self.proxy_distance[0],
                    metric_limit=self.proxy_distance[1],
                    allow_small_teacher=self.allow_small_teacher,
                )
            else:
                units, update = update_shared_endpoints(
                    self.units,
                    teacher,
                    seed=self.seed,
                    deadline=deadline.deadline,
                    enforce_kl=self.variant.kl_enforced,
                    operator_guard=guard,
                    expected_operator_source=guard.source_sha256,
                    expected_operator_plan=guard.plan_sha256,
                    matched_feasibility=feasibility_requirement,
                )
            self.update_records.append(
                {
                    "teacher": target_record,
                    "update": asdict(update),
                    "operator_records": guard.records if guard is not None else [],
                    "operator_equal_checkpoint_mixtures": guard.mixture_summaries()
                    if guard is not None
                    else [],
                }
            )
            deadline.check("after_endpoint_update")
            if update.accepted:
                if self.policy_sampler is not None:
                    self.policy_sampler.accept(units)
                    self.update_records[-1]["whole_trajectory_mixture"] = (
                        self.policy_sampler.record()
                    )
                replacement_units = units
                replacement_versions = {
                    triple: value + 1 for triple, value in self.versions.items()
                }
            elif update.status == "partial_deadline_no_commit":
                raise EvolutionBudgetExceeded("shared endpoint update did not seal")
            elif update.status not in (
                "insufficient_targets_no_update",
                "all_backtracks_rejected_ten_students_unchanged",
            ):
                raise ValueError("shared endpoint numerical/operator integrity failure")
        replacement_branches = tuple(
            branches[(unit.triple, index)] for unit in self.units for index in range(4)
        )
        return replacement_units, replacement_versions, replacement_branches

    def _record_disabled_model_update(self, teacher, target_record):
        """Keep teacher diagnostics, but do not train, probe distances or consume RNG.

        This is an endpoint-only control, not the historical ``no_endpoint``
        variant: teacher construction and all other full-method behavior remain.
        A concentration-integrity exception still fails closed as in the trainer.
        """
        from amp_challenge.generators.diffusion.native_shared_endpoint import SharedEndpointUpdate
        from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
            teacher_admission,
        )

        for unit in self.units:
            unit.check()
        training_ids = set().union(
            *(set(unit.initialization.training_sequence_ids) for unit in self.units)
        )
        if training_ids.intersection(target.sequence_id for target in teacher.targets):
            raise ValueError("teacher targets overlap generator training namespace")
        if self.allow_small_teacher:
            from amp_challenge.generators.diffusion.native_proxy_distance_update import (
                competition_teacher_weights,
            )

            admission, _ = competition_teacher_weights(teacher)
        else:
            admission, _ = teacher_admission(teacher)
        payload = {
            "artifact": "native_proxy_model_updates_disabled_v1",
            "teacher_sha256": teacher.sha256,
            "teacher_semantic_sha256": teacher.semantic_sha256,
            "seed": self.seed,
            "metric": self.proxy_distance[0],
            "metric_limit": self.proxy_distance[1],
            "model_updates_disabled": True,
            "old_models": self.policy_identities,
            "new_models": self.policy_identities,
            "admission": admission,
            "training": [],
            "candidates": [],
            "training_executed": False,
            "distance_checks_executed": False,
            "accepted": False,
            "status": "model_updates_disabled",
            "backtracks": None,
            "error": None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        update = SharedEndpointUpdate(
            encoded,
            hashlib.sha256(encoded.encode()).hexdigest(),
            "model_updates_disabled",
            False,
            None,
            False,
            False,
        )
        self.update_records.append(
            {
                "teacher": target_record,
                "update": asdict(update),
                "operator_records": [],
                "operator_equal_checkpoint_mixtures": [],
            }
        )

    def collect(self, provider, *, feasible, feasibility_source_sha256: str):
        """feasible is a fixed public/context check, never organizer/oracle truth.

        It receives only already generated sequences, returns bools, and cannot
        see reserve inventories. The outer caller authenticates its source.
        """
        if self.history is None or self.history.round_index > self.max_rounds:
            raise ValueError("no active adaptive evolution wave")
        if self.wave is not None:
            return self.wave
        if not hash_string(feasibility_source_sha256):
            raise ValueError("fixed context-feasibility source required")
        self._check()
        self.posterior.check()
        history, posterior, deadline = self.history, self.posterior, self.deadline
        semantic = semantic_history(history, self.eligible_charged_ids)
        draws = []
        attempts, allocations, shortlist = [], [], ()
        feasibility_by_id = {}
        status = "complete"
        failure = None
        prior = {(branch.triple, branch.branch): branch for branch in self.branches}
        new_branches = []
        charged = {sequence_id(row.sequence) for row in history.observations}
        roots = tuple(
            row.sequence
            for row in history.observations
            if sequence_id(row.sequence) in self.eligible_charged_ids
        )
        try:
            deadline.check("before_frozen_branch_functions")
            if not roots:
                raise EvolutionNoEligibleParent("no eligible charged parent root")
            for index in range(40):
                draw_seed = int(
                    _json_hash([semantic, "evolution-frozen-coefficient", index])[:16], 16
                )
                draws.append(
                    tuple(map(float, posterior.backend.draw_latent(seed=draw_seed, count=1)[0]))
                )
            root_features = self.cache.matrix(roots)
            all_directions = np.array(DIRECTIONS * 10)
            root_scores = posterior.direction_scores(root_features, draws, all_directions)
            for unit_index, unit in enumerate(self.units):
                for branch_index in range(4):
                    values = root_scores[:, 4 * unit_index + branch_index]
                    parent = roots[
                        min(
                            range(len(roots)),
                            key=lambda index: (-values[index], sequence_id(roots[index])),
                        )
                    ]
                    old = prior.get((unit.triple, branch_index))
                    new_branches.append(
                        EvolutionBranch(unit.triple, branch_index, parent)
                        if old is None
                        else replace(old, parent=parent)
                    )
            for stage in range(2):
                stage_begin = len(attempts)
                offset = int(_json_hash([self.seed, "evolution-round-robin"])[:16], 16) % 10
                for slot in range(10):
                    deadline.check("before_native_proposals")
                    unit_index = (slot + offset) % 10
                    unit = self.units[unit_index]
                    branches = tuple(new_branches[4 * unit_index : 4 * unit_index + 4])
                    scores, counts = allocate_branches(
                        branches, counterfactual=self.variant.name != "no_counterfactual"
                    )
                    allocations.append(
                        EvolutionAllocation(
                            stage,
                            unit.triple,
                            branches,
                            scores,
                            counts,
                            tuple(sequence_id(branch.parent) for branch in branches),
                        )
                    )
                    branch_rows = tuple(
                        branch
                        for branch, count in zip(branches, counts, strict=True)
                        for _ in range(count)
                    )
                    plans = []
                    for branch in branch_rows:
                        ordinal = (history.round_index - 1) * 480 + len(attempts) + len(plans)
                        operator = OPERATORS[
                            int(_seed(self.seed, ordinal, "evolution-operator", 0).integers(4))
                        ]
                        template, level, length_logp = operator_plan(
                            unit, branch.parent, operator, seed=self.seed, ordinal=ordinal
                        )
                        plans.append((ordinal, branch, operator, template, level, length_logp))
                    traces = self._sample_proposals(
                        unit,
                        tuple(plan[3] for plan in plans),
                        start_levels=tuple(plan[4] for plan in plans),
                        ordinals=tuple(plan[0] for plan in plans),
                    )
                    for plan, trace in zip(plans, traces, strict=True):
                        ordinal, branch, operator, _, _, length_logp = plan
                        key = trace.endpoint_sha256
                        first = self._seen.get(key, ordinal)
                        rejection = (
                            "training_overlap"
                            if key in self.training_ids
                            else "already_charged"
                            if key in charged
                            else "duplicate"
                            if key in self._seen
                            else "identity_noop"
                            if trace.endpoint == branch.parent
                            else None
                        )
                        self._seen.setdefault(key, ordinal)
                        attempts.append(
                            EvolutionAttempt(
                                ordinal=ordinal,
                                stage=stage,
                                triple=unit.triple,
                                branch=branch.branch,
                                lineage_parent=branch.parent,
                                operator=operator,
                                operator_log_probability=-math.log(4),
                                length_log_probability=length_logp,
                                trace=trace,
                                behavior_version=self.versions[unit.triple],
                                rejection=rejection,
                                first_seen_ordinal=first,
                            )
                        )
                    deadline.check("after_native_proposals")
                # 240 endpoints/stage need at mosttwo128-row requests.
                self.cache.ensure(
                    tuple(attempt.trace.endpoint for attempt in attempts[stage_begin:]),
                    provider,
                    deadline,
                )
                sequences = tuple(attempt.trace.endpoint for attempt in attempts[stage_begin:])
                flags = []
                for start in range(0, len(sequences), 128):
                    deadline.check("before_public_feasibility")
                    result = feasible(sequences[start : start + 128])
                    if (
                        type(result) is not tuple
                        or len(result) != len(sequences[start : start + 128])
                        or any(type(flag) is not bool for flag in result)
                    ):
                        raise ValueError("fixed public feasibility result differs")
                    flags.extend(result)
                for seq, flag in zip(sequences, flags, strict=True):
                    key = sequence_id(seq)
                    if key in feasibility_by_id and feasibility_by_id[key] != flag:
                        raise ValueError("fixed public feasibility changed for the same sequence")
                    feasibility_by_id[key] = flag
                self.cache.events.append(
                    {
                        "kind": "public_feasibility",
                        "source_sha256": feasibility_source_sha256,
                        "sequence_ids": [sequence_id(seq) for seq in sequences],
                        "flags": flags,
                    }
                )
                child_matrix = self.cache.matrix(sequences)
                parent_matrix = self.cache.matrix(
                    tuple(attempt.lineage_parent for attempt in attempts[stage_begin:])
                )
                pair_means, pair_covariances = posterior.pairs(child_matrix, parent_matrix)
                stage_scores = posterior.direction_scores(child_matrix, draws, all_directions)
                for local, index in enumerate(range(stage_begin, len(attempts))):
                    attempt = attempts[index]
                    contrast = contrast_from_moments(
                        pair_means[local], pair_covariances[local], feasible=flags[local]
                    )
                    unit_index = TRIPLES.index(attempt.triple)
                    value = float(stage_scores[local, 4 * unit_index + attempt.branch])
                    rejection = attempt.rejection or (None if flags[local] else "public_infeasible")
                    attempts[index] = replace(
                        attempt,
                        rejection=rejection,
                        contrast=contrast,
                        posterior_mean=tuple(map(float, pair_means[local, 0])),
                        thompson_value=value,
                        max_thompson_value=float(stage_scores[local].max()),
                    )
                # Only the same frozen function moves each second-stage parent.
                parent_scores = posterior.direction_scores(
                    self.cache.matrix(tuple(branch.parent for branch in new_branches)),
                    draws,
                    all_directions,
                )
                for index, branch in enumerate(new_branches):
                    values = [(parent_scores[index, index], branch.parent)]
                    values.extend(
                        (attempt.thompson_value, attempt.trace.endpoint)
                        for attempt in attempts[stage_begin:]
                        if attempt.triple == branch.triple
                        and attempt.branch == branch.branch
                        and attempt.rejection is None
                    )
                    parent = min(values, key=lambda row: (-row[0], sequence_id(row[1])))[1]
                    new_branches[index] = replace(branch, parent=parent)
                deadline.check("after_stage_posterior")
            eligible = [
                index for index, attempt in enumerate(attempts) if attempt.rejection is None
            ]
            mean_rank = sorted(
                eligible,
                key=lambda index: (
                    -sum(attempts[index].posterior_mean) / 2,
                    attempts[index].trace.endpoint_sha256,
                ),
            )
            draw_rank = sorted(
                eligible,
                key=lambda index: (
                    -attempts[index].max_thompson_value,
                    attempts[index].trace.endpoint_sha256,
                ),
            )
            selected = []
            for pair in zip(mean_rank, draw_rank, strict=True):
                for index in pair:
                    if index not in selected and len(selected) < 256:
                        selected.append(index)
            shortlist = tuple(selected)
        except EvolutionNoEligibleParent as error:
            status = "stopped_no_eligible_parent"
            failure = str(error)
        except EvolutionBudgetExceeded as error:
            status = "stopped_deadline_partial_wave"
            failure = str(error)[:512]
        except (ValueError, FloatingPointError, TypeError, RuntimeError) as error:
            status = "stopped_numerical_integrity_partial_wave"
            failure = f"{type(error).__name__}: {error}"[:512]
        self._check()
        posterior.check()
        wave = EvolutionWave(
            history.round_index,
            self.variant.name,
            history.sha256,
            semantic,
            posterior.sha256,
            self.cache.binding.sha256,
            _json_hash(self.source),
            self.policy_identities,
            tuple(draws),
            tuple(allocations),
            tuple(attempts),
            shortlist,
            tuple(copy.deepcopy(self.cache.events)),
            status,
            tuple(deadline.checkpoints),
            failure,
            configuration_sha256=evolution_configuration_sha256(self.variant.name),
            max_rounds=self.max_rounds,
            prospective_protocol_sha256=self.prospective_protocol_sha256,
        )
        wave.check()
        # The bounded serialization check above is part of scientific work.
        # Retain completed numerical work but never publish it as a usable wave
        # if its final seal crosses the original deadline.
        if status == "complete":
            try:
                deadline.check("after_wave_serialization")
            except EvolutionBudgetExceeded as error:
                status = "stopped_deadline_partial_wave"
                wave = replace(wave, status=status, shortlist_ordinals=(), failure=str(error))
        wave = replace(wave, timing=tuple(deadline.checkpoints))
        self.wave = wave
        if status == "complete":
            self.branches = tuple(new_branches)
        return wave

    def select(self, *, allowed_sequence_ids, external_filter_sha256):
        if self.wave is None:
            raise ValueError("selection requires a sealed proposal wave")
        if self.selection is not None:
            return self.selection
        try:
            result = select_evolution_queries(
                self.wave,
                self.posterior,
                self.cache,
                allowed_sequence_ids=allowed_sequence_ids,
                external_filter_sha256=external_filter_sha256,
                deadline=self.deadline,
                fixed_budget=self.fixed_budget,
                precision_recheck=self.precision_recheck,
            )
            _ = result.sha256
            self.deadline.check("after_selection_serialization")
        except (TimeoutError, ValueError, FloatingPointError, RuntimeError, TypeError) as error:
            result = EvolutionSelection(
                self.wave.sha256,
                self.posterior.sha256,
                external_filter_sha256,
                (),
                (),
                "stopped_kg_deadline"
                if isinstance(error, TimeoutError)
                else "stopped_kg_numerical_integrity",
                {
                    "failure": f"{type(error).__name__}: {error}"[:512],
                    "completed_phase_checkpoints": tuple(self.deadline.checkpoints),
                },
                (),
                _json_hash([]),
            )
        self.selection = result
        return self.selection

    def terminal(self, *, eligible_submitted_ids: frozenset[str]):
        if self.history is None:
            raise ValueError("terminal recommendation requires sealed charged history")
        submitted = {
            sequence_id(row.sequence): row.sequence
            for row in self.history.observations
            if row.status == "successful"
        }
        if (
            type(eligible_submitted_ids) is not frozenset
            or not eligible_submitted_ids <= submitted.keys()
        ):
            raise ValueError("terminal eligibility is not a real successful charged subset")
        base = {
            "history_sha256": self.history.sha256,
            "posterior_sha256": self.posterior.sha256,
            "eligible_count": len(eligible_submitted_ids),
            "eligible_ids_sha256": _json_hash(sorted(eligible_submitted_ids)),
            "feature_binding_sha256": self.cache.binding.sha256,
            "selected_sequence_id": None,
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        }
        if len(eligible_submitted_ids) < 100:
            return {**base, "status": "abstained_fewer_than100_eligible"}
        try:
            self.deadline.check("before_terminal_mean_recommendation")
        except EvolutionBudgetExceeded:
            return {**base, "status": "abstained_scientific_deadline"}
        ids = tuple(sorted(eligible_submitted_ids))
        feature_event_start = len(self.cache.events)
        values = self.posterior.means(
            self.cache.matrix(tuple(submitted[key] for key in ids))
        ) @ np.array([0.5, 0.5])
        best = min(range(len(ids)), key=lambda index: (-values[index], ids[index]))
        self.posterior.check()
        try:
            self.deadline.check("after_terminal_mean_recommendation")
        except EvolutionBudgetExceeded:
            return {**base, "status": "abstained_scientific_deadline"}
        return {
            **base,
            "status": "recommended" if values[best] > 0 else "abstained_outside_option",
            "selected_sequence_id": ids[best] if values[best] > 0 else None,
            "ranked_ids": tuple(
                ids[index]
                for index in sorted(range(len(ids)), key=lambda index: (-values[index], ids[index]))
            ),
            "posterior_mean_utilities": tuple(map(float, values)),
            "feature_cache_hits": tuple(copy.deepcopy(self.cache.events[feature_event_start:])),
            "scope": "unclipped_posterior_mean_real_charged_candidates_not_acquisition",
        }

"""Actual native evolutionary provider for the prospective frozen-oracle study.

Only disclosed controller history enters fitting. This module has no oracle
handle. Real checkpoint loading and real ESM extraction are explicit resources;
an algorithmic stop is preserved, never replaced with another search method.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

from amp_challenge.evaluation.peptide_proxy_campaign import ProviderCapabilities
from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint
from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureBatch,
    EvolutionFeatureBinding,
    FrozenEvolutionPosterior,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.models.charged_probability_learner import (
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.candidate_features import (
    CandidateFeatureRequest,
    validate_candidate_features,
)
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.warm_candidate_features import WarmCandidateFeatureSession


class NativeMethodStopped(RuntimeError):
    """A retained native stop, not permission to substitute random proposals."""


def verify_component_traces(attempts, sampler, *, replay=False):
    """Check real component bindings; optionally replay offline without oracle access.

    This does not claim a full independent acquisition/learner audit. In
    particular, latest proposal weights are not used to replay old components.
    """
    from amp_challenge.generators.diffusion.native_endpoint import _json_hash
    from amp_challenge.generators.diffusion.native_proposals import replay_native_trace

    record = sampler.record()
    by_key = {}
    for choice in record["choices"]:
        key = (choice["triple"], choice["ordinal"])
        if key in by_key:
            raise ValueError("duplicate recorded native component choice")
        by_key[key] = choice
    checked = []
    for attempt in attempts:
        trace = attempt.trace
        key = (attempt.triple, trace.ordinal)
        if key not in by_key:
            raise ValueError("native trace lacks actual mixture component choice")
        choice = by_key[key]
        if (
            choice["component_sha256"] != trace.model_sha256
            or choice["trace_sha256"] != _json_hash(asdict(trace))
            or choice["parent_sha256"] != trace.parent_sha256
            or choice["start_level"] != trace.start_level
        ):
            raise ValueError("native trace differs from recorded component")
        if replay:
            value, _ = replay_native_trace(
                sampler.model_for_trace(attempt.triple, trace),
                trace,
                authenticate_sampling=True,
            )
            if not math.isclose(
                value, trace.augmented_path_log_probability, abs_tol=1e-6, rel_tol=1e-8
            ):
                raise ValueError("actual frozen component replay likelihood differs")
        checked.append(choice["trace_sha256"])
    return {
        "trace_count": len(checked),
        "trace_inventory_sha256": _json_hash(checked),
        "actual_component_replay": replay,
        "sampler_sha256": record["sha256"],
        "full_method_audit_claimed": False,
    }


def charged_snapshot(
    history,
    *,
    run_id,
    seed,
    context_sha256,
    oracle_sha256,
    previous_head_sha256,
    initial_size=512,
    additional=1024,
    batch_size=16,
):
    if initial_size != 512 or additional != 1024 or batch_size != 16:
        raise ValueError("prospective native provider requires the declared 512+1024 budget")
    if not initial_size <= len(history) <= initial_size + additional:
        raise ValueError("charged history length outside prospective budget")
    if (len(history) - initial_size) % batch_size:
        raise ValueError("native learning requires a complete sixteen-response batch")
    observations = []
    for index, row in enumerate(history):
        if row.get("phase") != ("initial" if index < initial_size else "adaptive"):
            raise ValueError("charged phase differs from frozen initial prefix")
        if index >= initial_size and row.get("evaluation_index") != index - initial_size:
            raise ValueError("adaptive evaluation order differs")
        successful = row.get("status") == "success" and not row.get("late", False)
        objectives = None
        if successful:
            # The native Gaussian interface has two coordinates. Both are the
            # SAME paid scalar, explicitly not independent Gram labels.
            score = float(row["oracle_score"])
            objectives = (score, score)
        observations.append(
            ChargedObservation(
                index,
                f"charge-{index:04d}",
                row["sequence"],
                fingerprint(row),
                "successful" if successful else "timed_out" if row.get("late") else "failed",
                objectives,
            )
        )
    return VerifiedHistorySnapshot(
        run_id,
        seed,
        1 + (len(history) - initial_size) // batch_size,
        context_sha256,
        oracle_sha256,
        previous_head_sha256,
        tuple(observations),
        fingerprint(list(history)),
        initial_charge_count=initial_size,
        max_rounds=additional // batch_size,
        charges_per_round=batch_size,
    )


class WarmEvolutionFeatures:
    """Receipt-checked fixed-shape ESM; rotate bounded sessions without resetting clock."""

    def __init__(self, *, repository, commit, bundle, output_root, run_id, deadline):
        self.repository, self.commit, self.bundle = Path(repository), commit, Path(bundle)
        self.root, self.run_id, self.deadline = Path(output_root), run_id, deadline
        self.root.mkdir(exist_ok=False)
        self.session = None
        self.sessions = []
        self.binding = None
        self.requests = 0

    def check(self):
        if monotonic() >= self.deadline:
            raise TimeoutError("original prospective campaign deadline exhausted")

    def _raw(self, sequences):
        self.check()
        if self.session is None or self.session.ordinal == 128:
            if self.session is not None:
                self.session.close()
            session_id = f"s{len(self.sessions)}"
            self.session = WarmCandidateFeatureSession(
                run_id=self.run_id,
                session_id=session_id,
                repository=self.repository,
                expected_commit=self.commit,
                bundle=self.bundle,
                output_root=self.root / session_id,
                timeout_seconds=min(7200.0, self.deadline - monotonic()),
                maximum_requests=128,
                fixed_shape=True,
            )
            self.sessions.append(self.session)
        request = CandidateFeatureRequest(self.run_id, f"b{self.requests:05d}", tuple(sequences))
        ordinal = self.session.ordinal
        receipt = self.session.request(request)
        directory = self.session.output_root / f"batch-{ordinal:04d}" / "features"
        rows, arrays, _ = validate_candidate_features(
            directory,
            file_digest(directory / "manifest.json"),
            request,
            self.session.source,
            self.session.job_id,
        )
        if tuple(row["sequence"] for row in rows) != tuple(sequences):
            raise ValueError("real feature row order changed")
        self.requests += 1
        self.check()
        return np.asarray(arrays["esm_length_spectral"], dtype=np.float64), fingerprint(receipt)

    def fit_generator_transform(self, sequences):
        """Fit label-free normalization using generator training sequences only."""
        sequences = tuple(sorted(set(sequences), key=sequence_id))
        if not sequences:
            raise ValueError("generator corpus is empty")
        matrices, receipts = [], []
        for start in range(0, len(sequences), 128):
            matrix, receipt = self._raw(sequences[start : start + 128])
            matrices.append(matrix)
            receipts.append(receipt)
        matrix = np.vstack(matrices)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        transform = GeneratorFeatureTransform(
            "esm320_plus_normalized_length_plus_spectral32",
            matrix.mean(axis=0),
            scale,
            fingerprint({"generator_sequences": sequences, "feature_receipts": receipts}),
        )
        self.binding = EvolutionFeatureBinding(
            transform.representation,
            fingerprint(self.session.source),
            transform.sha256,
            file_digest(Path(__file__)),
        )
        return transform

    def evaluate(self, sequences):
        if self.binding is None:
            raise ValueError("generator-only transform must be frozen before candidate extraction")
        matrix, receipt = self._raw(sequences)
        return EvolutionFeatureBatch(tuple(sequences), matrix, receipt, self.binding)

    def close(self):
        if self.session is not None and not self.session.closed:
            self.session.close()


class PublicFeasibility:
    def __init__(self, excluded_sequences):
        self.excluded = frozenset(excluded_sequences)
        self.source_sha256 = fingerprint(
            {
                "canonical_min": 8,
                "canonical_max": 50,
                "excluded": sorted(self.excluded),
                "source": file_digest(Path(__file__)),
            }
        )

    def __call__(self, sequences):
        return tuple(
            8 <= len(seq) <= 50
            and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
            and seq not in self.excluded
            for seq in sequences
        )


class NativeProxyEvolutionProvider:
    """Controller adapter preserving native ingest, collection and knowledge-gain selection."""

    def __init__(
        self,
        protocol,
        *,
        arm_name,
        ensemble,
        features,
        transform,
        reserves,
        excluded_sequences,
        output_root,
        deadline,
        admission_evidence,
    ):
        arm = next(arm for arm in protocol.arms if arm.name == arm_name)
        if (
            protocol.initial_size != 512
            or protocol.additional_evaluations != 1024
            or protocol.batch_size != 16
        ):
            raise ValueError("native prospective provider budget mismatch")
        if len(reserves) != 128 or len(set(reserves)) != 128:
            raise ValueError("sixty-four frozen two-seat reserve batches required")
        if (
            features.binding != ensemble.cache.binding
            or transform.sha256 != features.binding.transform_sha256
        ):
            raise ValueError("ensemble, feature source and transform differ")
        self.protocol, self.ensemble, self.features, self.transform = (
            protocol,
            ensemble,
            features,
            transform,
        )
        self.reserves, self.deadline = tuple(reserves), deadline
        self.feasible = PublicFeasibility(excluded_sequences)
        if not all(self.feasible(self.reserves)):
            raise ValueError("reserve inventory violates common eligibility")
        self.excluded_ids = frozenset(map(sequence_id, excluded_sequences))
        self.root = Path(output_root)
        self.root.mkdir(exist_ok=False)
        self.previous_head = "0" * 64
        self.last_history_size = None
        self.capabilities = ProviderCapabilities(
            protocol.protocol_sha256,
            arm_name,
            protocol.initial_size,
            protocol.additional_evaluations,
            protocol.constraint_scope,
            protocol.ground_cost_id,
            protocol.clip_ratio_width,
            protocol.per_transition_total_variation_limit,
            arm.threshold,
            arm.units,
            admission_evidence,
            file_digest(Path(__file__)),
        )

    def _save(self, name, value):
        from amp_challenge.generators.diffusion.native_evolution_acquisition import (
            numerical_document,
        )

        with (self.root / name).open("x") as stream:
            json.dump(numerical_document(value), stream, sort_keys=True, allow_nan=False)

    def _ingest(self, history):
        if self.last_history_size == len(history):
            raise ValueError("provider cannot replay an already consumed charged prefix")
        snapshot = charged_snapshot(
            history,
            run_id=self.ensemble.run_id,
            seed=self.ensemble.seed,
            context_sha256=self.ensemble.context.context_sha256,
            oracle_sha256=self.ensemble.bundle,
            previous_head_sha256=self.previous_head,
        )
        successful = tuple(
            row
            for row in snapshot.observations
            if row.status == "successful" and self.feasible((row.sequence,))[0]
        )
        sequences = tuple(row.sequence for row in successful)
        unseen = tuple(seq for seq in sequences if seq not in self.ensemble.cache.rows)
        for start in range(0, len(unseen), 128):
            self.ensemble.cache.preload(self.features.evaluate(unseen[start : start + 128]))
        learner = fit_charged_learner(
            snapshot,
            self.transform,
            feature_sequence_ids=tuple(map(sequence_id, sequences)),
            raw_features=self.ensemble.cache.matrix(sequences),
            feature_receipt_sha256=fingerprint(
                [self.ensemble.cache.rows[seq][1] for seq in sequences]
            ),
            eligible_query_ids=tuple(row.query_id for row in successful),
            scalar_replicas=True,
        )
        posterior = FrozenEvolutionPosterior(
            learner.backend,
            history_sha256=snapshot.sha256,
            context_sha256=snapshot.objective_context_sha256,
            learner_source_sha256=file_digest(Path(__file__)),
            observation_noise=learner.observation_noise,
            feature_binding=self.features.binding,
            transform=self.transform,
        )
        status = self.ensemble.ingest(
            snapshot,
            posterior,
            expected_previous_head_sha256=self.previous_head,
            eligible_charged_ids=frozenset(map(sequence_id, sequences)),
            outer_deadline=self.deadline,
            feasible=self.feasible,
            feasibility_source_sha256=self.feasible.source_sha256,
        )
        self._save(f"history-{snapshot.round_index:02d}.json", asdict(snapshot))
        self._save(
            f"ingest-{snapshot.round_index:02d}.json",
            {
                "status": status,
                "learner_sha256": learner.numerical_sha256,
                "updates": self.ensemble.update_records,
                "objective_semantics": "identical_frozen_activity_proxy_scalar_replicas_not_independent_objectives",
            },
        )
        self.last_history_size = len(history)
        return snapshot, sequences

    def propose(self, history, batch_size):
        if batch_size != 16:
            raise ValueError("native method requires fourteen method and two reserved seats")
        snapshot, _ = self._ingest(history)
        if set(row["sequence"] for row in history[:512]) & set(self.reserves):
            raise ValueError("initial dataset overlaps frozen reserves")
        wave = self.ensemble.collect(
            self.features,
            feasible=self.feasible,
            feasibility_source_sha256=self.feasible.source_sha256,
        )
        if self.ensemble.policy_sampler is not None:
            sampler = self.ensemble.policy_sampler
            self._save(
                f"component-bindings-{snapshot.round_index:02d}.json",
                verify_component_traces(wave.attempts, sampler),
            )
            self._save(f"policy-mixture-{snapshot.round_index:02d}.json", sampler.record())
            if not sampler.storage_backed:
                sampler.persist(self.root / "policies")
        self._save(f"wave-{snapshot.round_index:02d}.json", asdict(wave))
        excluded = (
            self.excluded_ids
            | frozenset(map(sequence_id, self.reserves))
            | frozenset(sequence_id(row["sequence"]) for row in history)
        )
        allowed = (
            frozenset(sequence_id(wave.attempts[i].trace.endpoint) for i in wave.shortlist_ordinals)
            - excluded
        )
        selection = self.ensemble.select(
            allowed_sequence_ids=allowed, external_filter_sha256=fingerprint(sorted(excluded))
        )
        self._save(f"selection-{snapshot.round_index:02d}.json", asdict(selection))
        if selection.status != "selected" or len(selection.selected_ordinals) != 14:
            raise NativeMethodStopped(f"Native selection preserved: {selection.status}")
        method = tuple(wave.attempts[i].trace.endpoint for i in selection.selected_ordinals)
        offset = 2 * (snapshot.round_index - 1)
        self.previous_head = selection.sha256
        return method + self.reserves[offset : offset + 2]

    def finalize(self, history):
        snapshot, sequences = self._ingest(history)
        if snapshot.round_index != 65 or len(history) != 1536:
            raise ValueError("final selection requires all1536 charged observations")
        result = self.ensemble.terminal(
            eligible_submitted_ids=frozenset(map(sequence_id, sequences))
        )
        if self.ensemble.policy_sampler is not None and self.ensemble.policy_sampler.storage_backed:
            self.ensemble.policy_sampler.persist(self.root / "policies")
        return result


def load_native_proxy_provider(
    protocol,
    *,
    arm_name,
    run_id,
    seed,
    oracle_sha256,
    context_sha256,
    checkpoints,
    audit,
    repository,
    commit,
    bundle,
    output_root,
    reserves,
    excluded_sequences,
    deadline,
    device="cuda",
    admission_evidence,
    model_updates_disabled=False,
    replay_teacher=False,
    precision_recheck=False,
    trained_release=None,
    feature_provider=None,
    policy_storage=None,
    allow_small_teacher=False,
):
    """Load all ten audited checkpoints; setup consumes the caller's original clock.

    Caller must close ``provider.features`` on both success and failure. No
    checkpoint, outcome, reserve, or training-namespace fallback is permitted.
    """
    from amp_challenge.generators.diffusion.native_baseline_operators import (
        NormalizedObjectiveContext,
    )
    from amp_challenge.generators.diffusion.native_evolution import NativeCounterfactualEnsemble
    from amp_challenge.generators.diffusion.native_evolution_posterior import EvolutionFeatureCache
    from amp_challenge.generators.search.native_static_campaign import _load_audited_units

    root = Path(output_root)
    root.mkdir(exist_ok=False)

    def check():
        if monotonic() >= deadline:
            raise TimeoutError("original campaign deadline exhausted during provisioning")

    if trained_release is None:
        initializations, corpora = _load_audited_units(
            checkpoints=checkpoints, audit=audit, device=device, check=check
        )
    else:
        from amp_challenge.workflows.competition_train import load_trained_units

        initializations, corpora = load_trained_units(trained_release, device=device, check=check)
    training = frozenset(seq for corpus in corpora for seq in corpus)
    if not training <= frozenset(excluded_sequences):
        raise ValueError("common exclusions omit actual generator training sequences")
    if device not in {"cpu", "cuda"}:
        raise ValueError("native prospective device must identify the CPU or CUDA cohort")
    feature_class = WarmEvolutionFeatures
    if device == "cpu":
        from amp_challenge.workflows.cpu_proxy_features import CpuEvolutionFeatures

        feature_class = CpuEvolutionFeatures
    features = feature_provider or feature_class(
        repository=repository,
        commit=commit,
        bundle=bundle,
        output_root=root / "features",
        run_id=run_id,
        deadline=deadline,
    )
    try:
        transform = features.fit_generator_transform(training)
        context = NormalizedObjectiveContext(
            context_sha256,
            objective_names=("activity_proxy_replica_1", "activity_proxy_replica_2"),
            value_semantics="identical_frozen_activity_proxy_scalar_replicas_not_independent_objectives",
        )
        arm = next(arm for arm in protocol.arms if arm.name == arm_name)
        ensemble = NativeCounterfactualEnsemble(
            initializations,
            corpora,
            context,
            run_id=run_id,
            seed=seed,
            oracle_bundle_sha256=oracle_sha256,
            cache=EvolutionFeatureCache(features.binding, maximum_requests=256),
            prospective_protocol_sha256=protocol.protocol_sha256,
            max_rounds=64,
            proxy_distance=(arm.constraint, arm.threshold),
            fixed_budget=protocol.fixed_budget_selection,
            model_updates_disabled=model_updates_disabled,
            replay_teacher=replay_teacher,
            precision_recheck=precision_recheck,
            policy_storage=policy_storage,
            allow_small_teacher=allow_small_teacher,
        )
        provider = NativeProxyEvolutionProvider(
            protocol,
            arm_name=arm_name,
            ensemble=ensemble,
            features=features,
            transform=transform,
            reserves=reserves,
            excluded_sequences=excluded_sequences,
            output_root=root / "native",
            deadline=deadline,
            admission_evidence=admission_evidence,
        )
        provider._save(
            "provisioned.json",
            {
                "checkpoint_policy_identities": ensemble.policy_identities,
                "transform_mean": transform.mean.tolist(),
                "transform_scale": transform.scale.tolist(),
                "transform_sha256": transform.sha256,
                "transform_source_sha256": transform.source_sha256,
                "feature_binding": asdict(features.binding),
                "context": asdict(context),
                "excluded_sequence_sha256": fingerprint(sorted(excluded_sequences)),
                "reserve_sequence_sha256": fingerprint(list(reserves)),
                "initial_size": 512,
                "additional_evaluations": 1024,
                "model_updates_disabled": model_updates_disabled,
                "replay_teacher": replay_teacher,
                "precision_recheck": precision_recheck,
                "model_update_control_scope": "endpoint_training_only_full_method_teacher_and_acquisition_retained",
                "oracle_access": "none_provider_receives_only_controller_disclosed_scalar_history",
            },
        )
        check()
        return provider
    except BaseException:
        if features.session is not None and not features.session.closed:
            # This is teardown of a failed run, never a restarted feature request.
            features.close()
        raise

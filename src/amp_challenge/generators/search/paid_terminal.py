"""Paid final-history fit and recommendation, with no oracle or clock renewal.

Requires controller-authenticated inputs and its surviving bridge. This additive
transition implements a prospective successor clock; historical v2 is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureBinding,
    FrozenEvolutionPosterior,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.models.charged_probability_learner import (
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.candidate_features import _exclusive_write
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    LAYOUT_SHA256,
    FeatureAssemblyBinding,
    FeatureIntent,
    canonical,
    finite_clock,
    json_object,
    pin,
    require,
    sha256,
)
from amp_challenge.representations.run_feature_cache_views import assemble_charged

CONTRACT = "paid_terminal_v2_20260914"
TIMING_AMENDMENT_PATH = "configs/search/paid_terminal_timing_v2.toml"
TIMING_AMENDMENT_SHA256 = "f249a99a443970e7fa578f0f630982dabefba9c482cf2f07531c081b4c8bd189"
LEARNER_SOURCE = "src/amp_challenge/models/charged_probability_learner.py"


def timing_amendment_sha256(repository=None):
    """Check the exact prospective declaration in loaded and supplied repositories."""
    loaded_root = Path(__file__).resolve().parents[4]
    roots = (loaded_root,) if repository is None else (loaded_root, Path(repository))
    for root in roots:
        require(
            file_digest(root / TIMING_AMENDMENT_PATH) == TIMING_AMENDMENT_SHA256,
            "paid-terminal timing amendment changed",
        )
    return TIMING_AMENDMENT_SHA256


def terminal_timing_summary(
    *,
    original_epoch,
    original_deadline,
    query_closed_at,
    decision_completed_at,
    terminal_returned_at,
):
    """Local phase measurements; callers authenticate the enclosing statement.

    This reports timing only. It supplies no new truth observations, hypervolume
    points, external resource attestation or scientific completion authority.
    """
    epoch, deadline, closure, decision, returned = map(
        finite_clock,
        (
            original_epoch,
            original_deadline,
            query_closed_at,
            decision_completed_at,
            terminal_returned_at,
        ),
    )
    require(
        0 < deadline - epoch <= 7200 and epoch <= closure <= decision <= returned < deadline,
        "terminal phase times differ from original deadline or order",
    )
    return {
        "timing_amendment_sha256": timing_amendment_sha256(),
        "original_epoch": epoch,
        "original_deadline": deadline,
        "query_closed_at": closure,
        "decision_completed_at": decision,
        "terminal_returned_at": returned,
        "query_collection_seconds": closure - epoch,
        "terminal_decision_seconds": decision - closure,
        "post_decision_return_seconds": returned - decision,
        "elapsed_to_terminal_return_seconds": returned - epoch,
        "remaining_original_seconds": deadline - returned,
    }


def complete_paid_terminal(
    *,
    history,
    expected_history_sha256,
    bridge,
    transform,
    assembly,
    expected_assembly,
    terminal_query_ids,
    terminal_eligibility_source_sha256,
    terminal_eligibility_receipt_sha256,
    query_closed_at,
    query_closure_receipt_sha256,
    learner_source_sha256,
    output_root,
    monotonic=time.monotonic,
):
    """Fit the final 512-charge history, rank submitted rows and close the worker.

    A successful return includes the final live completion time. The caller must
    authenticate/retain that return; staged files alone are not timely authority.
    No whole-process restoration or repeat invocation is supported.
    """
    require(
        type(history) is VerifiedHistorySnapshot and type(bridge) is RunFeatureBridge,
        "exact history and surviving feature bridge required",
    )
    require(
        type(transform) is GeneratorFeatureTransform
        and type(assembly) is FeatureAssemblyBinding
        and type(expected_assembly) is FeatureAssemblyBinding,
        "exact transform and independently pinned assembly required",
    )
    history.__post_init__()
    require(
        history.complete and history.round_index == 29 and len(history.observations) == 512,
        "terminal requires exactly 512 complete charged outcomes",
    )
    require(history.sha256 == expected_history_sha256, "final history identity differs")
    require(
        bridge._RunFeatureBridge__clock is monotonic,
        "terminal must retain the surviving original clock callable",
    )
    require(
        type(terminal_query_ids) is frozenset and all(type(q) is str for q in terminal_query_ids),
        "terminal eligibility must be explicit query IDs",
    )
    require(
        all(
            pin(p)
            for p in (
                terminal_eligibility_source_sha256,
                terminal_eligibility_receipt_sha256,
                query_closure_receipt_sha256,
                learner_source_sha256,
            )
        ),
        "terminal external source/receipt identities required",
    )
    binding = bridge.binding
    timing_amendment_sha256(binding.repository)
    require(
        (history.run_id, history.seed, history.objective_context_sha256)
        == (binding.run_id, binding.seed, binding.objective_context_sha256),
        "terminal run/seed/context differs from surviving bridge",
    )
    require(
        assembly == expected_assembly and assembly.history_sha256 == history.sha256,
        "terminal charged assembly authority differs",
    )
    require(
        terminal_query_ids <= frozenset(assembly.eligible_query_ids),
        "terminal candidates must belong to fitted eligibility",
    )
    closure = finite_clock(query_closed_at)
    require(
        binding.original_epoch <= closure < binding.original_deadline,
        "query closure must be inside original scientific clock",
    )
    source = Path(binding.repository) / LEARNER_SOURCE
    require(file_digest(source) == learner_source_sha256, "actual charged learner source differs")
    fixed = (
        history.sha256,
        binding.sha256,
        transform.sha256,
        assembly.sha256,
        expected_assembly.sha256,
    )
    root = Path(output_root)
    require(root.is_absolute(), "absolute terminal output directory required")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    timings = []
    phase = "entry"
    last = closure
    base = {
        "contract": CONTRACT,
        "timing_amendment_sha256": TIMING_AMENDMENT_SHA256,
        "history_sha256": expected_history_sha256,
        "run_id": history.run_id,
        "seed": history.seed,
        "logical_charges": 512,
        "query_closed_at": closure,
        "query_closure_receipt_sha256": query_closure_receipt_sha256,
        "original_epoch": binding.original_epoch,
        "original_deadline": binding.original_deadline,
        "terminal_eligibility_source_sha256": terminal_eligibility_source_sha256,
        "terminal_eligibility_receipt_sha256": terminal_eligibility_receipt_sha256,
        "terminal_query_ids": sorted(terminal_query_ids),
        "learner_source_sha256": learner_source_sha256,
        "oracle_calls": 0,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }

    def save(name, document):
        _exclusive_write(root / name, canonical(document))

    def check(name):
        nonlocal last, phase
        phase = name
        timing_amendment_sha256(binding.repository)
        now = finite_clock(monotonic())
        require(now >= last, "terminal original monotonic clock moved backwards")
        if now >= binding.original_deadline:
            raise TimeoutError("terminal fit/decision exceeded the original deadline")
        last = now
        require(
            (
                history.sha256,
                bridge.binding.sha256,
                transform.sha256,
                assembly.sha256,
                expected_assembly.sha256,
            )
            == fixed,
            "terminal history/feature/eligibility drift",
        )
        require(file_digest(source) == learner_source_sha256, "terminal learner source changed")
        timings.append({"phase": name, "at_monotonic": now})
        return now

    def intent(purpose):
        return FeatureIntent(
            purpose,
            expected_history_sha256,
            history.objective_context_sha256,
            29,
            0,
            bridge.accepted_head,
            binding.original_deadline,
        )

    try:
        save("started.json", base)
        check("before_final_feature_assembly")
        data = assemble_charged(
            bridge,
            history,
            transform,
            assembly,
            intent("terminal"),
            expected_authority=expected_assembly,
        )
        check("before_final_history_fit")
        learner = fit_charged_learner(history, transform, **data)
        check("after_final_history_fit")
        feature_binding = EvolutionFeatureBinding(
            representation=transform.representation,
            feature_source_sha256=sha256(
                canonical(
                    {
                        "warm_source": json_object(binding.warm_source_payload),
                        "runtime_sha256": sha256(binding.runtime_payload),
                        "model_sha256": binding.model_sha256,
                        "layout_sha256": LAYOUT_SHA256,
                    }
                )
            ),
            transform_sha256=transform.sha256,
            provider_sha256=sha256(binding.implementation_source_payload),
        )
        posterior = FrozenEvolutionPosterior(
            learner.backend,
            history_sha256=history.sha256,
            context_sha256=history.objective_context_sha256,
            learner_source_sha256=learner_source_sha256,
            observation_noise=np.eye(2) * 0.01,
            feature_binding=feature_binding,
            transform=transform,
        )
        save(
            "learner.json",
            {
                "history_sha256": learner.history_sha256,
                "semantic_history_sha256": learner.semantic_history_sha256,
                "numerical_sha256": learner.numerical_sha256,
                "fit_query_ids": learner.fit_query_ids,
                "feature_receipt_sha256": learner.feature_receipt_sha256,
                "feature_sequence_ids": learner.fit_sequence_ids,
                "transform_sha256": transform.sha256,
            },
        )
        require(
            terminal_query_ids <= frozenset(learner.fit_query_ids),
            "terminal candidate lacks a successful fitted observation",
        )
        selected_rows = [row for row in history.observations if row.query_id in terminal_query_ids]
        positions = {key: i for i, key in enumerate(data["feature_sequence_ids"])}
        ranked = []
        # Final features are already assembled. Never request a candidate-search
        # feature opportunity or compute a covariance/Thompson draw here.
        if selected_rows:
            raw = data["raw_features"][
                [positions[sequence_id(row.sequence)] for row in selected_rows]
            ]
            values = posterior.means(raw)
            utilities = values @ np.array([0.5, 0.5])
            ranked = sorted(
                (
                    {
                        "query_id": row.query_id,
                        "sequence": row.sequence,
                        "sequence_id": sequence_id(row.sequence),
                        "objectives": list(map(float, values[i])),
                        "utility": float(utilities[i]),
                    }
                    for i, row in enumerate(selected_rows)
                ),
                key=lambda row: (-row["utility"], row["sequence_id"]),
            )
        posterior.check()
        check("after_final_mean_ranking")
        status = (
            "abstained_fewer_than100_eligible"
            if len(ranked) < 100
            else "abstained_outside_option"
            if ranked[0]["utility"] <= 0
            else "recommended"
        )
        bridge.close(intent("close"))
        check("after_worker_close")
        decision = {
            **base,
            "status": status,
            "eligible_count": len(ranked),
            "selected_sequence_id": ranked[0]["sequence_id"] if status == "recommended" else None,
            "ranked_rows": ranked,
            "posterior_sha256": posterior.sha256,
            "learner_numerical_sha256": learner.numerical_sha256,
            "feature_final_head": bridge.evidence_head,
            "feature_counters": asdict(bridge.counters),
            "timings": timings,
            "scope": "unclipped_final_history_mean_of_real_submitted_eligible_candidates",
        }
        save("decision.json", decision)
        decision_sha = file_digest(root / "decision.json")
        finished = check("after_decision_publication")
        return {
            **decision,
            "scientific_completed_at": finished,
            "decision_sha256": decision_sha,
            "query_collection_seconds": closure - binding.original_epoch,
            "terminal_decision_seconds": finished - closure,
        }

    except BaseException as error:
        # A late failure may follow successful worker closure. Do not read an
        # accepted-head property that intentionally rejects a closed bridge.
        bridge.abort(error)
        save(
            "failure.json",
            {
                **base,
                "status": "failed_terminal",
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "timings": timings,
                "bridge_failure": bridge.last_failure,
            },
        )
        raise

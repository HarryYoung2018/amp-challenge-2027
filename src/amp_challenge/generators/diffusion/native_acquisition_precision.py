"""One bounded, opt-in verification of an already selected fixed-budget batch.

Two fresh streams avoid reusing the selection-optimized estimate as a reference.
This is an explicit numerical-policy amendment, not a rescue of historical runs
or a guarantee that repeated agreement checks control a study-wide error rate.
"""

from __future__ import annotations

import math
from time import monotonic

from amp_challenge.acquisition.soft_kg import GaussianSoftKG
from amp_challenge.generators.diffusion.native_endpoint import _json_hash

SPEC = {
    "version": "single_fixed_batch_precision_recheck_v1",
    "streams": 2,
    "fantasies_per_stream": 2048,
    "maximum_rechecks": 1,
    "maximum_seconds": 5.0,
    "standard_error_limit": 0.02,
    "agreement_standard_errors": 3,
    "agreement_absolute_slack": 1e-6,
    "rerank": False,
}


def recheck_fixed_batch(
    problem, belief, batch, group, history_sha256, *, deadline, clock=monotonic
):
    if (
        len(group) != 14
        or len(set(group)) != 14
        or tuple(sorted(group)) != tuple(group)
        or len(history_sha256) != 64
        or set(history_sha256) - set("0123456789abcdef")
        or not math.isfinite(deadline)
    ):
        raise ValueError("bounded recheck requires one fixed fourteen-point batch and history")
    started = clock()
    end = min(deadline, started + SPEC["maximum_seconds"])

    def check_clock():
        if clock() >= end:
            raise TimeoutError("bounded acquisition precision recheck exhausted its clock")

    estimates = []
    for stream in range(2):
        check_clock()
        seed = int(_json_hash([history_sha256, SPEC["version"], stream])[:16], 16)
        engine = GaussianSoftKG(
            problem,
            temperature=0.25,
            observed_outputs=(0, 1),
            n_fantasies=2048,
            standard_error_multiplier=1,
            seed=seed,
            relative_eigenvalue_cutoff=1e-10,
            candidate_chunk_size=64,
            fantasy_chunk_size=512,
        )
        result = engine.score_joint_groups(belief, batch, [tuple(group)], max_groups=1)
        check_clock()
        estimates.append(
            {
                "seed": seed,
                "mean": float(result.estimate[0]),
                "standard_error": float(result.standard_error[0]),
            }
        )
    left, right = estimates
    valid = all(
        math.isfinite(row["mean"])
        and math.isfinite(row["standard_error"])
        and 0 <= row["standard_error"] <= 0.02
        for row in estimates
    )
    tolerance = (
        3 * math.hypot(left["standard_error"], right["standard_error"]) + 1e-6 if valid else None
    )
    return {
        "spec": SPEC.copy(),
        "spec_sha256": _json_hash(SPEC),
        "group": list(group),
        "estimates": estimates,
        "agreement_tolerance": tolerance,
        "passed": valid and abs(left["mean"] - right["mean"]) <= tolerance,
        "elapsed_seconds": clock() - started,
        "fixed_budget_positivity_is_not_a_veto": True,
    }

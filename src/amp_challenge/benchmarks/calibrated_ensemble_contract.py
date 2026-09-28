"""Frozen non-authorizing contract for the controller-private ensemble study."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

CONFIG = "configs/benchmarks/oracle_calibrated_ensemble_v1.toml"
CONFIG_SHA = "67cec93d3ed7c431fbf5910e4bb2d5676165623caa54fd3b98617336f85c78c0"
SEEDS = (310013, 310019, 310033, 310049, 310063)
VIEWS = ("descriptors_context", "esm_context")
PREDICTORS = ("prior", "raw", "raw_ood", "calibrated", "calibrated_ood")
SOLVER = {"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 50}
REPO = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_protocol(repository=REPO):
    payload = (Path(repository) / CONFIG).read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == CONFIG_SHA,
        "frozen ensemble configuration pin differs",
    )
    return tomllib.loads(payload.decode())


def private_teacher_claims():
    return {
        "teacher_controller_private": True,
        "teacher_moments_available_to_adaptive_search": False,
        "generated_label_kind": "MODEL_PROXY",
        "generated_query_count_kind": "unique_surrogate_oracle_calls_not_real_measurements",
        "adaptive_training_inputs": "charged_revealed_records_and_explicitly_allowed_label_free_priors_only",
        "final_all_oracle_refit_authorized": False,
        "final_query_mapping_authorized": False,
        "production_input_eligible": False,
        "search_superiority_accepted": False,
        "learned_HC50_hard_constraint_allowed": False,
        "oracle_calls": 0,
    }

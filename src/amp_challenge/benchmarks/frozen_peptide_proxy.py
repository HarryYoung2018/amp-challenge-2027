"""Frozen, measured-data-trained activity proxy; never biological ground truth.

This adapter reuses the accepted all-data descriptor fit, not the failed
calibrated teacher. Search algorithms receive only charged scores through their
runner; this service must not expose its training labels to adaptive learners.
QuickVina is deliberately not substituted: no docking target was specified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from amp_challenge.models.oracle_baselines import OracleInput
from amp_challenge.workflows.candidate_activity_scoring import (
    import_descriptor_model_states,
)

ACCEPTED_MODEL_PATH = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/scoring/"
    "candidate-activity-v1/production/225374/0/model_states.json"
)
ACCEPTED_MODEL_SHA256 = "98bdb74cd26b7652d18feb476f926eadfb9e4ac80369bb92b0c2608420473ab6"
ACCEPTED_EXAMPLES_SHA256 = "d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520"
TARGET_PANEL = (
    ("acinetobacter_baumannii", "negative"),
    ("enterococcus_faecalis", "positive"),
    ("enterococcus_faecium", "positive"),
    ("escherichia_coli", "negative"),
    ("klebsiella_pneumoniae", "negative"),
    ("pseudomonas_aeruginosa", "negative"),
    ("staphylococcus_aureus", "positive"),
)


class FrozenPeptideProxy:
    """Black-box mean activity reward in [0, 1], maximized by the search.

    No fitting occurs at construction or evaluation. The file digest must be
    provided explicitly for non-production models so alternate releases cannot be
    silently substituted. All seven target predictions have equal weight.
    """

    def __init__(
        self,
        model_path: Path | str = ACCEPTED_MODEL_PATH,
        *,
        expected_sha256: str = ACCEPTED_MODEL_SHA256,
    ) -> None:
        payload = Path(model_path).read_bytes()
        self.model_sha256 = hashlib.sha256(payload).hexdigest()
        if self.model_sha256 != expected_sha256:
            raise ValueError("frozen oracle model fingerprint mismatch")
        document = json.loads(payload)
        if document["target_order"] != [name for name, _ in TARGET_PANEL]:
            raise ValueError("frozen oracle target panel mismatch")
        models = import_descriptor_model_states(document)
        self._model = models[-1]
        self._payload = payload
        self._document = document
        if self._model._strains != tuple(name for name, _ in TARGET_PANEL):
            raise ValueError("all-data oracle does not cover the declared target panel")
        # These arrays are never trained or mutated by this service.
        for array in (self._model._mean, self._model._scale, self._model._coefficient):
            if array is not None:
                array.flags.writeable = False

    def score(self, sequences: Sequence[str]) -> np.ndarray:
        """Return one deterministic reward per canonical peptide, in order."""
        if isinstance(sequences, str):
            raise TypeError("score expects a sequence of peptide strings, not one string")
        rows = []
        for sequence in sequences:
            if not isinstance(sequence, str) or not 8 <= len(sequence) <= 50:
                raise ValueError("proxy peptides must be canonical sequences of length 8 to 50")
            rows.extend(OracleInput(sequence, name, gram) for name, gram in TARGET_PANEL)
        if not rows:
            return np.empty(0, dtype=np.float64)
        values = self._model.predict_proba(rows).reshape(len(sequences), len(TARGET_PANEL))
        rewards = values.mean(axis=1)
        if not np.all(np.isfinite(rewards)) or np.any((rewards < 0) | (rewards > 1)):
            raise ValueError("frozen oracle returned an invalid reward")
        return rewards

    def __call__(self, sequence: str) -> float:
        return float(self.score([sequence])[0])

    def manifest(self) -> dict:
        """Describe exactly which fixed computational landscape is optimized."""
        state = self._document["states"][-1]
        return {
            "schema_version": 1,
            "artifact": "frozen_peptide_activity_proxy_v1",
            "model_sha256": self.model_sha256,
            "training_examples_sha256": (
                ACCEPTED_EXAMPLES_SHA256 if self.model_sha256 == ACCEPTED_MODEL_SHA256 else None
            ),
            "training_examples": state["training_examples"],
            "training_sequences": state["training_sequences"],
            "training_example_ids_sha256": state["training_example_ids_sha256"],
            "training_scope": "all_designated_accepted_activity_contexts",
            "model_member": "all_data_deployment",
            "feature_order": self._document["feature_order"],
            "descriptor_settings": self._document["descriptor_settings"],
            "target_panel": [{"name": name, "gram": gram} for name, gram in TARGET_PANEL],
            "reward": "equal_target_mean_raw_logistic_probability_maximize",
            "reward_range": [0, 1],
            "fit_during_query": False,
            "evaluation_scope": "computational_proxy_only_not_biological_ground_truth",
            "independently_validated_biological_oracle": False,
            "prior_calibrated_oracle_failed_qualification": True,
            "prior_failure_waived": False,
            "docking_status": "not_configured_target_and_peptide_docking_recipe_required",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "implementation_sha256": {
                relative: hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / relative).read_bytes()
                ).hexdigest()
                for relative in (
                    "models/oracle_baselines.py",
                    "descriptors.py",
                    "sequences.py",
                    "workflows/candidate_activity_scoring.py",
                )
            },
        }

    def freeze(self, output: Path | str) -> Path:
        """Write a new immutable-content release; refuse existing destinations."""
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "model_states.json").write_bytes(self._payload)
        (destination / "manifest.json").write_text(
            json.dumps(self.manifest(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ACCEPTED_MODEL_PATH)
    parser.add_argument("--expected-sha256", default=ACCEPTED_MODEL_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proxy = FrozenPeptideProxy(args.model, expected_sha256=args.expected_sha256)
    proxy.freeze(args.output)
    print(json.dumps(proxy.manifest(), sort_keys=True))


if __name__ == "__main__":
    main()

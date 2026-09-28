"""In-process protein features for the competition training pipeline."""

from pathlib import Path

import numpy as np

from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint
from amp_challenge.generators.diffusion.native_evolution_posterior import (
    EvolutionFeatureBatch,
    EvolutionFeatureBinding,
)
from amp_challenge.models.charged_probability_learner import GeneratorFeatureTransform
from amp_challenge.representations.fixed_shape_esm import extract_fixed_shape
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.sequences import canonical_sequence_id


class CompetitionFeatures:
    def __init__(self, embedder):
        self.embedder = embedder
        self.rows = {}
        self.binding = None
        self.session = None

    def matrix(self, sequences):
        sequences = tuple(sequences)
        missing = tuple(dict.fromkeys(seq for seq in sequences if seq not in self.rows))
        for start in range(0, len(missing), 128):
            batch = missing[start : start + 128]
            rows = sorted(
                ({"sequence": seq, "sequence_id": canonical_sequence_id(seq)} for seq in batch),
                key=lambda row: row["sequence_id"],
            )
            arrays = extract_fixed_shape(
                self.embedder.model,
                self.embedder.alphabet,
                self.embedder.torch,
                rows,
                device=self.embedder.device,
            )
            for row, vector, mean in zip(
                rows, arrays["esm_length_spectral"], arrays["esm_mean"], strict=True
            ):
                self.rows[row["sequence"]] = vector
                self.embedder.cache[row["sequence"]] = mean
        return np.asarray([self.rows[seq] for seq in sequences], dtype=np.float64)

    def fit_generator_transform(self, sequences):
        sequences = tuple(sorted(set(sequences)))
        matrix = self.matrix(sequences)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1
        transform = GeneratorFeatureTransform(
            "esm320_plus_normalized_length_plus_spectral32",
            matrix.mean(axis=0),
            scale,
            fingerprint({"sequences": sequences, "embedding": self.embedder.sha256}),
        )
        self.binding = EvolutionFeatureBinding(
            transform.representation,
            self.embedder.sha256,
            transform.sha256,
            file_digest(Path(__file__)),
        )
        return transform

    def evaluate(self, sequences):
        matrix = self.matrix(sequences)
        return EvolutionFeatureBatch(
            tuple(sequences),
            matrix,
            fingerprint({"sequences": sequences, "matrix": matrix.tolist()}),
            self.binding,
        )

    def close(self):
        pass

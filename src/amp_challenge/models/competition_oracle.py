"""Target-specific embedding ensemble and honest held-out prediction metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

DEPENDENCIES = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/competition-20260920/dependencies"
)
CHECKPOINT = Path(
    "/lustre/scratch/users/yonghan.yang/amp_challenge/models/ampdiffusion-1a862af9/cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
)


def enable_dependencies():
    # Isolated optional dependencies; never modify the environment of another job.
    missing = any(importlib.util.find_spec(name) is None for name in ("sklearn", "lightgbm", "esm"))
    if missing and DEPENDENCIES.is_dir() and str(DEPENDENCIES) not in sys.path:
        sys.path.insert(0, str(DEPENDENCIES))


class PeptideEmbedder:
    def __init__(self, checkpoint=CHECKPOINT, *, device="cuda", cache=None):
        enable_dependencies()
        import esm
        import torch

        self.torch, self.device = torch, device
        self.checkpoint = Path(checkpoint)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with torch.serialization.safe_globals([argparse.Namespace]):
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                str(checkpoint)
            )
        self.model.eval().requires_grad_(False).to(device)
        self.converter = self.alphabet.get_batch_converter()
        self.cache = {} if cache is None else cache

    def encode(self, sequences):
        sequences = tuple(sequences)
        if any(
            not 8 <= len(seq) <= 50 or set(seq) - set("ACDEFGHIKLMNPQRSTVWY") for seq in sequences
        ):
            raise ValueError("embedding input must be a canonical peptide of length 8 to 50")
        missing = list(dict.fromkeys(seq for seq in sequences if seq not in self.cache))
        with self.torch.inference_mode():
            for start in range(0, len(missing), 128):
                batch = missing[start : start + 128]
                inputs = [(str(i), seq) for i, seq in enumerate(batch)]
                inputs.extend(("padding", "ACDEFGHI") for _ in range(128 - len(inputs)))
                _, _, tokens = self.converter(inputs)
                fixed = self.torch.full((128, 52), self.alphabet.padding_idx, dtype=self.torch.long)
                fixed[:, : tokens.shape[1]] = tokens
                hidden = self.model(fixed.to(self.device), repr_layers=[6])["representations"][6]
                for index, sequence in enumerate(batch):
                    values = hidden[index, 1 : len(sequence) + 1].float().cpu().numpy()
                    self.cache[sequence] = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        return np.asarray([self.cache[seq] for seq in sequences], dtype=np.float32).reshape(-1, 320)


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class TargetEnsemble:
    """Separate nonlinear models per target; absent measurements are not negatives."""

    def __init__(self, *, seed=42, trees=128):
        self.seed, self.trees = seed, trees
        self.targets = {}

    def fit(self, embeddings, targets, labels):
        enable_dependencies()
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier

        x, y, targets = np.asarray(embeddings), np.asarray(labels), np.asarray(targets)
        if (
            x.ndim != 2
            or len(x) != len(y)
            or len(y) != len(targets)
            or not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all()
        ):
            raise ValueError("invalid labeled embedding rows")
        self.targets = {}
        for target in sorted(set(targets)):
            take = targets == target
            xt, yt = x[take], y[take]
            prior = float((yt.sum() + 1) / (len(yt) + 2))
            models = []
            if len(np.unique(yt)) == 2:
                models = [
                    RandomForestClassifier(
                        n_estimators=self.trees,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        random_state=self.seed,
                        n_jobs=1,
                    ),
                    LGBMClassifier(
                        n_estimators=self.trees,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=10,
                        reg_lambda=1.0,
                        random_state=self.seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ]
                for model in models:
                    model.fit(xt, yt)
            self.targets[str(target)] = {
                "models": models,
                "x": normalized_rows(xt),
                "y": yt,
                "prior": prior,
            }
        return self

    def predict_components(self, embeddings, target):
        entry = self.targets[target]
        x = np.asarray(embeddings)
        values = [
            model.booster_.predict(x)
            if hasattr(model, "booster_")
            else model.predict_proba(x)[:, 1]
            for model in entry["models"]
        ]
        if not values:
            values = [np.full(len(x), entry["prior"])] * 2
        similarities = normalized_rows(x) @ entry["x"].T
        k = min(15, len(entry["y"]))
        neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        weights = np.maximum(np.take_along_axis(similarities, neighbors, axis=1), 0) + 1e-6
        neighbor_score = (np.sum(weights * entry["y"][neighbors], axis=1) + entry["prior"]) / (
            weights.sum(axis=1) + 1
        )
        return np.column_stack([*values, neighbor_score])

    def predict(self, embeddings):
        return np.column_stack(
            [self.predict_components(embeddings, target).mean(axis=1) for target in self.targets]
        )


def prediction_metrics(labels, probabilities):
    enable_dependencies()
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    return {
        "rows": len(labels),
        "positive_rate": float(labels.mean()),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, probabilities))
        if np.any(labels)
        else None,
    }


def train_oracle(examples, output, *, device="cuda"):
    enable_dependencies()
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from amp_challenge.generators.baseline import baseline_feature_matrix

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = Path(examples).read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    # Use the existing homology/study component folds, never random assay-row splits.
    component_folds = {}
    for row in rows:
        for identity in (row["sequence_id"], row["union_component_id"]):
            if component_folds.setdefault(identity, row["fold"]) != row["fold"]:
                raise ValueError("sequence or study/homology component crosses folds")
    sequences = [row["sequence"] for row in rows]
    embedder = PeptideEmbedder(device=device)
    x = embedder.encode(sequences)
    y = np.array([row["label"] for row in rows])
    targets = np.array([row["canonical_target"] for row in rows])
    folds = np.array([row["fold"] for row in rows])
    descriptors = baseline_feature_matrix(sequences)
    names = (
        "embedding_ensemble",
        "random_forest",
        "lightgbm",
        "cosine_neighbors",
        "linear_embedding",
        "descriptor_logistic",
    )
    predictions = {name: np.full(len(rows), np.nan) for name in names}
    for fold in sorted(set(folds)):
        fit, test = folds != fold, folds == fold
        model = TargetEnsemble(seed=42).fit(x[fit], targets[fit], y[fit])
        for target in sorted(set(targets[test])):
            train_target, test_target = fit & (targets == target), test & (targets == target)
            if target not in model.targets:
                raise ValueError(f"target {target} has no training support in fold {fold}")
            components = model.predict_components(x[test_target], target)
            for column, name in enumerate(names[1:4]):
                predictions[name][test_target] = components[:, column]
            predictions[names[0]][test_target] = components.mean(axis=1)
            for name, matrix in (("linear_embedding", x), ("descriptor_logistic", descriptors)):
                if len(set(y[train_target])) == 1:
                    p = np.full(test_target.sum(), model.targets[target]["prior"])
                else:
                    baseline = make_pipeline(
                        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42)
                    )
                    baseline.fit(matrix[train_target], y[train_target])
                    p = baseline.predict_proba(matrix[test_target])[:, 1]
                predictions[name][test_target] = p
        print(json.dumps({"oracle_fold_completed": int(fold)}), flush=True)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("incomplete out-of-fold predictions")
    report = {
        "evaluation": "existing_homology_study_grouped_five_fold_predictions_not_biological_validation",
        "rows": len(rows),
        "unique_sequences": len(set(sequences)),
        "embedding_checkpoint_sha256": embedder.sha256,
        "examples_sha256": hashlib.sha256(payload).hexdigest(),
        "ensemble_weights": {"random_forest": 1 / 3, "lightgbm": 1 / 3, "cosine_neighbors": 1 / 3},
        "metrics": {name: prediction_metrics(y, p) for name, p in predictions.items()},
        "by_target": {
            str(target): {
                name: prediction_metrics(y[targets == target], p[targets == target])
                for name, p in predictions.items()
            }
            for target in sorted(set(targets))
        },
        "uncertainty": "member_disagreement_is_not_calibrated_uncertainty",
        "unsupported_endpoints": ["continuous_MIC", "hemolysis", "MDR_strain_specific_activity"],
    }
    model = TargetEnsemble(seed=42).fit(x, targets, y)
    joblib.dump(model, output / "oracle.joblib")
    np.savez_compressed(
        output / "heldout_predictions.npz", labels=y, folds=folds, targets=targets, **predictions
    )
    np.savez_compressed(
        output / "training_embeddings.npz", sequences=np.asarray(sequences), embeddings=x
    )
    report["oracle_sha256"] = hashlib.sha256((output / "oracle.joblib").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


class CompetitionOracle:
    def __init__(self, root, *, embedder=None, device="cuda"):
        enable_dependencies()
        import joblib

        root = Path(root)
        self.report = json.loads((root / "report.json").read_text())
        self.model_sha256 = hashlib.sha256((root / "oracle.joblib").read_bytes()).hexdigest()
        if self.model_sha256 != self.report["oracle_sha256"]:
            raise ValueError("oracle file differs from training report")
        # Load only our own locally trained model, never an untrusted pickle.
        self.model = joblib.load(root / "oracle.joblib")
        bundled_checkpoint = root.parent / "embedding" / CHECKPOINT.name
        self.embedder = embedder or PeptideEmbedder(
            bundled_checkpoint if bundled_checkpoint.is_file() else CHECKPOINT, device=device
        )
        if self.embedder.sha256 != self.report["embedding_checkpoint_sha256"]:
            raise ValueError("oracle embedding checkpoint differs")

    def predict_targets(self, sequences):
        return self.model.predict(self.embedder.encode(sequences))

    def score(self, sequences):
        return self.predict_targets(sequences).mean(axis=1)

    def __call__(self, sequence):
        return float(self.score([sequence])[0])


def main():
    from amp_challenge.models.competition_oracle import train_oracle as fit_and_save
    from amp_challenge.workflows.peptide_proxy_initial import EXAMPLES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=EXAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    fit_and_save(args.examples, args.output, device=args.device)


if __name__ == "__main__":
    main()

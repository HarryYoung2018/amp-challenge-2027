"""Generate the deterministic AMP Challenge library and ranked top portfolio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Indel

from amp_challenge.acquisition import (
    CandidateBatch,
    MixedAcquisitionSelector,
    SelectionConfig,
    SelectionResult,
)
from amp_challenge.generators.baseline import (
    FEATURE_NAMES,
    OBJECTIVE_NAMES,
    GeneratedPool,
    baseline_embeddings,
    baseline_feature_matrix,
    baseline_proxy_ensemble,
    coarse_cluster_ids,
    generate_baseline_pool,
    novelty_proxy,
)
from amp_challenge.sequences import canonical_sequence_id

DEFAULT_REFERENCE = Path(__file__).resolve().parents[3] / "data" / "antibacterial.fasta"


@dataclass(frozen=True)
class GenerationSummary:
    library_path: Path
    top_path: Path
    metadata_path: Path
    manifest_path: Path
    library_size: int
    top_size: int
    seed: int
    library_sha256: str
    top_sha256: str


class ReferenceIndex:
    """Organizer-reference lookup using the validator's exact normalized-indel ratio."""

    def __init__(self, sequences: Sequence[str]) -> None:
        unique = tuple(sorted(set(sequences)))
        self.sequences = unique
        buckets: dict[int, list[str]] = {}
        for sequence in unique:
            buckets.setdefault(len(sequence), []).append(sequence)
        self.by_length = {length: tuple(values) for length, values in buckets.items()}

    def max_ratio(self, sequence: str, *, threshold: float = 0.8) -> tuple[float, str | None]:
        """Return max normalized indel similarity and its reference sequence."""

        best = 0.0
        best_reference: str | None = None
        candidate_length = len(sequence)
        for reference_length in sorted(self.by_length):
            # A length-only upper bound for Levenshtein.ratio.  At exactly the
            # threshold a candidate remains valid, so only strict exceedance can
            # be skipped/failed.
            upper_bound = (
                2.0
                * min(candidate_length, reference_length)
                / (candidate_length + reference_length)
            )
            if upper_bound <= best:
                continue
            for reference in self.by_length[reference_length]:
                ratio = float(Indel.normalized_similarity(sequence, reference, score_cutoff=best))
                if ratio > best:
                    best = ratio
                    best_reference = reference
                if best > threshold:
                    return best, best_reference
        return best, best_reference


def generate_submission(
    *,
    n_sequences: int = 50_000,
    top_k: int = 100,
    seed: int = 42,
    output_dir: Path | str = Path("generate"),
    reference_path: Path | str = DEFAULT_REFERENCE,
    max_reference_ratio: float = 0.8,
) -> GenerationSummary:
    """Run the baseline generator, ensemble, mixed acquisition, and export."""

    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if top_k <= 0 or top_k > n_sequences:
        raise ValueError("top_k must be positive and no larger than n_sequences")
    if not 0 <= max_reference_ratio <= 1:
        raise ValueError("max_reference_ratio must be in [0, 1]")

    output = Path(output_dir)
    reference = Path(reference_path)
    if not reference.is_file():
        raise FileNotFoundError(f"organizer reference not found: {reference}")
    reference_sequences = _read_fasta_sequences(reference)
    reference_index = ReferenceIndex(reference_sequences)

    pool = generate_baseline_pool(
        n_sequences,
        seed=seed,
        forbidden=set(reference_sequences),
    )
    features = baseline_feature_matrix(pool.sequences)
    _, ensemble_result = baseline_proxy_ensemble(features)
    eligible = np.ones(n_sequences, dtype=bool)
    clusters = coarse_cluster_ids(features, pool.families)
    specialist_quotas = _specialist_quotas(top_k)
    config = SelectionConfig(
        batch_size=top_k,
        specialist_quotas=specialist_quotas,
        max_per_cluster=max(2, int(np.ceil(top_k / 20))),
        strict_cluster_cap=True,
        seed=seed,
    )
    similarity_cache: dict[int, tuple[float, str | None]] = {}

    while True:
        candidates = CandidateBatch(
            sequences=pool.sequences,
            objective_mean=ensemble_result.utility_mean,
            objective_std=ensemble_result.total_std,
            novelty=novelty_proxy(features),
            embeddings=baseline_embeddings(features),
            cluster_ids=clusters,
            specialist_scores={
                name: ensemble_result.risk_adjusted_utility[:, index]
                for index, name in enumerate(OBJECTIVE_NAMES)
            },
            eligible=eligible,
        )
        selection = MixedAcquisitionSelector(config).select(candidates)
        invalid: list[int] = []
        for index in selection.indices:
            if index not in similarity_cache:
                similarity_cache[index] = reference_index.max_ratio(
                    pool.sequences[index], threshold=max_reference_ratio
                )
            ratio, _ = similarity_cache[index]
            if ratio > max_reference_ratio:
                invalid.append(index)
        if not invalid:
            break
        eligible[np.asarray(invalid)] = False
        if np.count_nonzero(eligible) < top_k:
            raise RuntimeError("reference-similarity filtering left too few candidates")

    ranked_indices, ranked_reasons, ranked_scores, ranked_acquisition_scores = _hedged_ranking(
        selection
    )
    output.mkdir(parents=True, exist_ok=True)
    library_path = output / "library.fasta"
    top_path = output / "top.fasta"
    metadata_path = output / "top_metadata.csv"
    manifest_path = output / "manifest.json"
    _write_library_fasta(library_path, pool, seed=seed)
    _write_top_fasta(top_path, pool, ranked_indices, ranked_reasons)
    _write_top_metadata(
        metadata_path,
        pool=pool,
        indices=ranked_indices,
        reasons=ranked_reasons,
        conservative_scores=ranked_scores,
        acquisition_scores=ranked_acquisition_scores,
        features=features,
        ensemble_mean=ensemble_result.mean,
        ensemble_std=ensemble_result.total_std,
        similarities=similarity_cache,
        clusters=clusters,
    )

    library_sha = _sha256_file(library_path)
    top_sha = _sha256_file(top_path)
    manifest = {
        "format_version": 1,
        "generator": "transparent_physicochemical_baseline_v0",
        "warning": "Reproducibility baseline; proxy scores are not efficacy or safety claims.",
        "seed": seed,
        "library_size": n_sequences,
        "top_size": top_k,
        "max_reference_ratio": max_reference_ratio,
        "reference_path": str(reference),
        "reference_records": len(reference_index.sequences),
        "reference_sha256": _sha256_file(reference),
        "library_sha256": library_sha,
        "top_sha256": top_sha,
        "strategy_counts": selection.strategy_counts,
        "cluster_counts": selection.cluster_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return GenerationSummary(
        library_path=library_path,
        top_path=top_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        library_size=n_sequences,
        top_size=top_k,
        seed=seed,
        library_sha256=library_sha,
        top_sha256=top_sha,
    )


def _specialist_quotas(top_k: int) -> dict[str, int]:
    specialist_total = top_k // 2
    base, remainder = divmod(specialist_total, len(OBJECTIVE_NAMES))
    return {name: base + int(index < remainder) for index, name in enumerate(OBJECTIVE_NAMES)}


def _hedged_ranking(
    selection: SelectionResult,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    """Keep award-profile coverage in both the first half and full top list."""

    size = len(selection.indices)
    first_half_size = size // 2
    positions_by_specialist: dict[str, list[int]] = {}
    for position, reason in enumerate(selection.reasons):
        if reason.startswith("specialist:"):
            positions_by_specialist.setdefault(reason, []).append(position)

    first_half: set[int] = set()
    for positions in positions_by_specialist.values():
        first_half.update(positions[: max(1, len(positions) // 2)])
    for position in range(size):
        if len(first_half) >= first_half_size:
            break
        first_half.add(position)

    first_positions = sorted(
        first_half,
        key=lambda position: (
            -selection.conservative_scores[position],
            selection.indices[position],
        ),
    )
    remaining_positions = sorted(
        (position for position in range(size) if position not in first_half),
        key=lambda position: (
            -selection.conservative_scores[position],
            selection.indices[position],
        ),
    )
    order = first_positions + remaining_positions
    return (
        tuple(selection.indices[position] for position in order),
        tuple(selection.reasons[position] for position in order),
        tuple(selection.conservative_scores[position] for position in order),
        tuple(selection.acquisition_scores[position] for position in order),
    )


def _read_fasta_sequences(path: Path) -> tuple[str, ...]:
    sequences: list[str] = []
    chunks: list[str] = []
    saw_header = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if saw_header:
                sequences.append("".join(chunks).upper())
            saw_header = True
            chunks = []
        elif saw_header:
            chunks.append(line)
        else:
            raise ValueError(f"sequence before FASTA header in {path}")
    if saw_header:
        sequences.append("".join(chunks).upper())
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError(f"empty or malformed FASTA: {path}")
    return tuple(sequences)


def _write_library_fasta(path: Path, pool: GeneratedPool, *, seed: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for index, (sequence, family) in enumerate(
            zip(pool.sequences, pool.families, strict=True), start=1
        ):
            handle.write(f">amp_{index:05d} family={family} seed={seed}\n{sequence}\n")


def _write_top_fasta(
    path: Path,
    pool: GeneratedPool,
    indices: Sequence[int],
    reasons: Sequence[str],
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for rank, (index, reason) in enumerate(zip(indices, reasons, strict=True), start=1):
            handle.write(
                f">rank_{rank:03d} id={canonical_sequence_id(pool.sequences[index])} acquisition={reason}\n"
                f"{pool.sequences[index]}\n"
            )


def _write_top_metadata(
    path: Path,
    *,
    pool: GeneratedPool,
    indices: Sequence[int],
    reasons: Sequence[str],
    conservative_scores: Sequence[float],
    acquisition_scores: Sequence[float],
    features: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    similarities: dict[int, tuple[float, str | None]],
    clusters: Sequence[str],
) -> None:
    fields = [
        "rank",
        "sequence_id",
        "sequence",
        "generator_family",
        "cluster_id",
        "acquisition_reason",
        "conservative_score",
        "acquisition_score",
        "max_reference_ratio",
        "nearest_reference",
        *FEATURE_NAMES,
        *(f"mean_{name}" for name in OBJECTIVE_NAMES),
        *(f"std_{name}" for name in OBJECTIVE_NAMES),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, (index, reason, score, acquisition_score) in enumerate(
            zip(
                indices,
                reasons,
                conservative_scores,
                acquisition_scores,
                strict=True,
            ),
            start=1,
        ):
            ratio, nearest = similarities[index]
            row: dict[str, object] = {
                "rank": rank,
                "sequence_id": canonical_sequence_id(pool.sequences[index]),
                "sequence": pool.sequences[index],
                "generator_family": pool.families[index],
                "cluster_id": clusters[index],
                "acquisition_reason": reason,
                "conservative_score": f"{score:.10g}",
                "acquisition_score": f"{acquisition_score:.10g}",
                "max_reference_ratio": f"{ratio:.10g}",
                "nearest_reference": nearest or "",
            }
            row.update(
                {
                    name: f"{features[index, column]:.10g}"
                    for column, name in enumerate(FEATURE_NAMES)
                }
            )
            row.update(
                {
                    f"mean_{name}": f"{ensemble_mean[index, column]:.10g}"
                    for column, name in enumerate(OBJECTIVE_NAMES)
                }
            )
            row.update(
                {
                    f"std_{name}": f"{ensemble_std[index, column]:.10g}"
                    for column, name in enumerate(OBJECTIVE_NAMES)
                }
            )
            writer.writerow(row)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("generate"))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-reference-ratio", type=float, default=0.8)
    parser.add_argument("--method", choices=("evolutionary", "baseline"), default="evolutionary")
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "checkpoints/competition/evolutionary",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "checkpoints/competition/oracle",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.method == "evolutionary":
        from amp_challenge.workflows.competition_generate import generate_evolutionary_submission

        if args.max_reference_ratio != 0.8:
            parser.error("evolutionary export uses the organizer's 0.8 similarity threshold")
        summary = generate_evolutionary_submission(
            release=args.release,
            oracle_root=args.oracle,
            n_sequences=args.n_sequences,
            top_k=args.top_k,
            seed=args.seed,
            output_dir=args.output_dir,
            reference_path=args.reference,
            device=args.device,
        )
    else:
        summary = generate_submission(
            n_sequences=args.n_sequences,
            top_k=args.top_k,
            seed=args.seed,
            output_dir=args.output_dir,
            reference_path=args.reference,
            max_reference_ratio=args.max_reference_ratio,
        )
    print(
        json.dumps(
            {
                "library": str(summary.library_path),
                "library_size": summary.library_size,
                "library_sha256": summary.library_sha256,
                "top": str(summary.top_path),
                "top_size": summary.top_size,
                "top_sha256": summary.top_sha256,
                "seed": summary.seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Fixed-default scalar genetic baseline on a disclosed activity proxy.

Reuses existing genetic edit and tournament-selection numerics, but intentionally
does not construct the legacy two-Gram-objective archive. This is an adapted,
untuned scalar baseline, not the full evolutionary diffusion method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from amp_challenge.benchmarks.frozen_peptide_proxy import (
    ACCEPTED_EXAMPLES_SHA256,
    FrozenPeptideProxy,
)
from amp_challenge.evaluation.peptide_proxy_campaign import (
    ProviderCapabilities,
    run_campaign,
    verify_campaign,
)
from amp_challenge.evaluation.peptide_proxy_protocol import (
    ProxyProtocol,
    fingerprint,
    load_protocol,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2 import _propose_edit
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import parameters
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

ATTEMPT_CAP = 65536


@dataclass(frozen=True)
class ScalarParent:
    sequence: str
    sequence_key: str
    fitness: float


def scalar_population(history: Sequence[Mapping[str, Any]]) -> tuple[ScalarParent, ...]:
    """Use only successful disclosed scalar observations, no fabricated objectives."""
    observed: dict[str, ScalarParent] = {}
    for row in history:
        if row.get("status", "success") not in {"success", "duplicate"}:
            continue
        sequence = row["sequence"]
        if sequence != canonicalize_sequence(sequence):
            raise ValueError("noncanonical successful observation")
        score = float(row["oracle_score"])
        if not math.isfinite(score):
            raise ValueError("nonfinite disclosed observation")
        parent = ScalarParent(sequence, canonical_sequence_id(sequence), score)
        if sequence in observed and observed[sequence] != parent:
            raise ValueError("conflicting frozen oracle observations")
        observed[sequence] = parent
    return tuple(sorted(observed.values(), key=lambda row: (-row.fitness, row.sequence_key))[:512])


class ScalarGeneticProvider:
    def __init__(
        self,
        protocol: ProxyProtocol,
        *,
        seed: int,
        excluded_sequences: frozenset[str] = frozenset(),
    ) -> None:
        if seed not in protocol.seeds:
            raise ValueError("seed is not declared")
        arm = next(arm for arm in protocol.arms if arm.name == "genetic_algorithm")
        if arm.constraint != "none" or arm.method != "genetic_algorithm":
            raise ValueError("scalar baseline must use the unconstrained genetic arm")
        self.seed = seed
        self.excluded_sequences = frozenset(excluded_sequences)
        self.exclusion_sha256 = fingerprint(sorted(self.excluded_sequences))
        self.config = parameters("ga_t3_default")
        self.batches: list[dict] = []
        source_root = Path(__file__).resolve().parents[1]
        self.source_hashes = {
            relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
            for relative in (
                "workflows/peptide_proxy_ga.py",
                "generators/search/peptide_ga.py",
                "generators/search/peptide_ga_tunable_v2.py",
                "generators/search/peptide_ga_tunable_v2_records.py",
            )
        }
        self.capabilities = ProviderCapabilities(
            protocol_sha256=protocol.protocol_sha256,
            arm_name=arm.name,
            initial_size=protocol.initial_size,
            additional_evaluations=protocol.additional_evaluations,
            constraint_scope=protocol.constraint_scope,
            ground_cost_id=protocol.ground_cost_id,
            clip_ratio_width=protocol.clip_ratio_width,
            per_transition_total_variation_limit=protocol.per_transition_total_variation_limit,
            metric_threshold=arm.threshold,
            metric_units=arm.units,
            admission_evidence="fixed-default scalar genetic adaptation; no neural policy update; "
            "clipping and transition guards not applicable; no tuning claim; "
            f"sequence-only exclusion sha256={self.exclusion_sha256}",
            implementation_sha256=fingerprint(self.source_hashes),
        )

    def propose(self, history: Sequence[Mapping[str, Any]], batch_size: int) -> list[str]:
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError("batch size must be between one and sixteen")
        population = scalar_population(history)
        if not population:
            raise ValueError("no successful observed parents")
        stream = fingerprint(
            {
                "seed": self.seed,
                "configuration": asdict(self.config),
                "history": [
                    (r["sequence"], r.get("status", "success"), r.get("oracle_score"))
                    for r in history
                ],
                "excluded_sequence_sha256": self.exclusion_sha256,
            }
        )
        seen = {row["sequence"] for row in history} | set(self.excluded_sequences)
        accepted, provenance = [], []
        rejected = {"out_of_support": 0, "previously_seen_or_excluded": 0}
        for attempt in range(ATTEMPT_CAP):
            edit = _propose_edit(
                population, self.config, seed=self.seed, stream_sha256=stream, attempt_index=attempt
            )
            child = edit.sequence
            if not 8 <= len(child) <= 50 or not set(child) <= set(self.config.alphabet):
                rejected["out_of_support"] += 1
                continue
            if child in seen:
                rejected["previously_seen_or_excluded"] += 1
                continue
            seen.add(child)
            accepted.append(child)
            provenance.append({"attempt": attempt, "edit": asdict(edit)})
            if len(accepted) == batch_size:
                self.batches.append(
                    {
                        "stream_sha256": stream,
                        "attempts": attempt + 1,
                        "parent_count": len(population),
                        "rejected": rejected,
                        "accepted": provenance,
                    }
                )
                return accepted
        raise RuntimeError("genetic proposal attempt cap exhausted without a complete batch")

    def finalize(self, history: Sequence[Mapping[str, Any]]) -> dict:
        best = scalar_population(history)[:100]
        return {
            "selection": [{"sequence": row.sequence, "oracle_score": row.fitness} for row in best],
            "selection_rule": "top_100_unique_observed_scalar_scores_no_predictor",
            "method": "adapted_fixed_default_scalar_genetic_baseline_not_tuned",
            "configuration": asdict(self.config),
            "source_hashes": self.source_hashes,
            "excluded_sequence_sha256": self.exclusion_sha256,
            "excluded_sequences_count": len(self.excluded_sequences),
            "batches": self.batches,
            "claim_scope": "activity_proxy_not_quickvina_or_biology",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument(
        "--excluded-sequences",
        type=Path,
        help="Optional sequence-only JSON array; defaults to pinned training identities",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    initial = json.loads(args.initial.read_text())
    if args.excluded_sequences is None:
        from amp_challenge.workflows.peptide_proxy_initial import EXAMPLES

        payload = EXAMPLES.read_bytes()
        if hashlib.sha256(payload).hexdigest() != ACCEPTED_EXAMPLES_SHA256:
            raise ValueError("oracle training inventory changed")
        # Parse trusted training inventory only at the orchestration boundary;
        # discard all labels before constructing the adaptive provider.
        excluded = sorted({json.loads(line)["sequence"] for line in payload.splitlines()})
        del payload
    else:
        excluded = json.loads(args.excluded_sequences.read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) for s in excluded):
        raise ValueError("exclusion input must contain sequence strings only")
    oracle = FrozenPeptideProxy(args.oracle_model, expected_sha256=initial["oracle_sha256"])
    provider = ScalarGeneticProvider(
        protocol, seed=initial["seed"], excluded_sequences=frozenset(excluded)
    )
    result = run_campaign(
        protocol=protocol,
        initial_manifest=initial,
        provider=provider,
        oracle=oracle,
        oracle_id=initial["oracle_id"],
        oracle_sha256=oracle.model_sha256,
        output_dir=args.output,
        max_seconds=args.max_seconds,
    )
    audit = verify_campaign(args.output, protocol)
    with (args.output / "independent_accounting.json").open("x") as stream:
        json.dump(audit, stream, sort_keys=True, indent=2)
    # Full proposal provenance stays in result.json; keep scheduler logs compact.
    summary = {key: value for key, value in result.items() if key != "final_selection"}
    print(json.dumps({"result": summary, "accounting_audit": audit}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

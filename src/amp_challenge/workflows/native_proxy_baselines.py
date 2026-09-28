"""Matched genetic and frozen native categorical baselines, fourteen plus two seats.

These providers see only the complete paid controller history. Reserved peptides
are shared sequence-only inputs, never additional free labels. The categorical
baseline uses genuine full-mask native diffusion, not genetic edits, and makes
no policy updates or uncharged predictions for final ranking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

import numpy as np

from amp_challenge.evaluation.peptide_proxy_campaign import ProviderCapabilities
from amp_challenge.evaluation.peptide_proxy_protocol import fingerprint
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.workflows.peptide_proxy_ga import ScalarGeneticProvider, scalar_population


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _valid(sequence):
    return (
        isinstance(sequence, str)
        and 8 <= len(sequence) <= 50
        and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
    )


class _MatchedSeats:
    def __init__(self, protocol, *, seed, reserves, excluded_sequences, deadline):
        if (protocol.initial_size, protocol.additional_evaluations, protocol.batch_size) != (
            512,
            1024,
            16,
        ):
            raise ValueError("matched providers require 512 initial and 64 batches of sixteen")
        if seed not in protocol.seeds:
            raise ValueError("undeclared seed")
        self.protocol, self.seed, self.deadline = protocol, seed, deadline
        self.reserves, self.excluded = tuple(reserves), frozenset(excluded_sequences)
        if (
            len(self.reserves) != 128
            or len(set(self.reserves)) != 128
            or not all(_valid(seq) for seq in self.reserves)
            or set(self.reserves) & self.excluded
        ):
            raise ValueError("invalid frozen reserve inventory")
        self.reserves_sha256 = fingerprint(self.reserves)
        self.exclusions_sha256 = fingerprint(sorted(self.excluded))
        self.initial_sha256 = None
        self.last_history = None
        self.last_proposal = None
        self.batches = []

    def _check_history(self, history, *, final=False):
        if monotonic() >= self.deadline:
            raise TimeoutError("original baseline deadline exhausted")
        expected = 512 + 16 * len(self.batches)
        if len(history) != expected or (final and expected != 1536):
            raise ValueError("complete paid history does not match provider round")
        sequences = [row["sequence"] for row in history]
        if (
            len(set(sequences)) != len(sequences)
            or not all(_valid(seq) for seq in sequences)
            or set(sequences) & self.excluded
        ):
            raise ValueError("charged history violates common eligibility or uniqueness")
        if set(sequences[:512]) & set(self.reserves):
            raise ValueError("initial dataset overlaps reserved peptides")
        initial_hash = fingerprint(list(history[:512]))
        if self.initial_sha256 is None:
            self.initial_sha256 = initial_hash
        if initial_hash != self.initial_sha256:
            raise ValueError("frozen initial dataset changed")
        if self.last_history is not None and (
            fingerprint(list(history[:-16])) != self.last_history
            or tuple(sequences[-16:]) != self.last_proposal
        ):
            raise ValueError("paid history no longer matches the previous proposal")

    def _propose(self, history, batch_size, method):
        if batch_size != 16 or type(batch_size) is not int or len(self.batches) >= 64:
            raise ValueError("matched proposal requires an available sixteen-seat batch")
        self._check_history(history)
        selected, evidence = method(history)
        blocked = (
            self.excluded | frozenset(self.reserves) | frozenset(row["sequence"] for row in history)
        )
        if (
            len(selected) != 14
            or len(set(selected)) != 14
            or any(not _valid(seq) or seq in blocked for seq in selected)
        ):
            raise ValueError("method selected an invalid, duplicate or reserved peptide")
        start = len(self.batches) * 2
        result = tuple(selected) + self.reserves[start : start + 2]
        self.last_history, self.last_proposal = fingerprint(list(history)), result
        self.batches.append(
            dict(
                round_index=len(self.batches) + 1,
                history_sha256=self.last_history,
                history_rows=len(history),
                method_sequences=list(selected),
                reserve_sequences=list(result[-2:]),
                evidence=evidence,
            )
        )
        return list(result)

    def _finalize(self, history, method):
        self._check_history(history, final=True)
        return {
            "method": method,
            "selection": [
                dict(sequence=row.sequence, oracle_score=row.fitness)
                for row in scalar_population(history)[:100]
            ],
            "selection_rule": "top_100_unique_successful_paid_scalar_scores_no_predictor",
            "history_sha256": fingerprint(list(history)),
            "history_rows": len(history),
            "initial_sha256": self.initial_sha256,
            "reserve_sequences_sha256": self.reserves_sha256,
            "common_exclusions_sha256": self.exclusions_sha256,
            "batches": self.batches,
            "claim_scope": "computational_activity_proxy_not_docking_or_biological_validation",
        }


class MatchedGeneticProvider(_MatchedSeats):
    """Existing fixed-default genetic kernel; no tuning claim or history truncation."""

    def __init__(self, protocol, *, seed, reserves, excluded_sequences, deadline):
        super().__init__(
            protocol,
            seed=seed,
            reserves=reserves,
            excluded_sequences=excluded_sequences,
            deadline=deadline,
        )
        self.genetic = ScalarGeneticProvider(
            protocol, seed=seed, excluded_sequences=self.excluded | frozenset(self.reserves)
        )
        self.capabilities = replace(
            self.genetic.capabilities,
            implementation_sha256=fingerprint(
                [self.genetic.capabilities.implementation_sha256, _source_hash()]
            ),
            admission_evidence=self.genetic.capabilities.admission_evidence
            + f"; matched14+2; reserve_sha256={self.reserves_sha256}; common_exclusions_sha256={self.exclusions_sha256}",
        )

    def propose(self, history, batch_size):
        def method(rows):
            selected = self.genetic.propose(rows, 14)
            return selected, self.genetic.batches[-1]

        return self._propose(history, batch_size, method)

    def finalize(self, history):
        result = self._finalize(history, "matched_fixed_default_scalar_genetic_not_tuned")
        result["configuration"] = asdict(self.genetic.config)
        return result


class NativeCategoricalProvider(_MatchedSeats):
    """Frozen ten-checkpoint full-mask diffusion with initial empirical lengths.

    Every parent residue is masked at the maximum level; initial sequences only
    provide a frozen length distribution. Adaptive scores do not change sampling.
    This is an adapted sampling baseline, not the legacy post-hoc predictor arm.
    """

    def __init__(self, protocol, *, seed, units, reserves, excluded_sequences, deadline):
        super().__init__(
            protocol,
            seed=seed,
            reserves=reserves,
            excluded_sequences=excluded_sequences,
            deadline=deadline,
        )
        self.units = tuple(units)
        if tuple(unit.triple for unit in self.units) != TRIPLES:
            raise ValueError("categorical baseline requires ten ordered native units")
        for unit in self.units:
            unit.check()
        self.identities = tuple(unit.policy_sha256 for unit in self.units)
        arm = next(arm for arm in protocol.arms if arm.name == "categorical")
        if arm.constraint != "none":
            raise ValueError("frozen categorical baseline cannot claim a trained constraint")
        self.capabilities = ProviderCapabilities(
            protocol.protocol_sha256,
            arm.name,
            512,
            1024,
            protocol.constraint_scope,
            protocol.ground_cost_id,
            protocol.clip_ratio_width,
            protocol.per_transition_total_variation_limit,
            arm.threshold,
            arm.units,
            "genuine frozen ten-native-checkpoint full-mask diffusion; no policy updates; "
            f"matched14+2; reserve_sha256={self.reserves_sha256}; common_exclusions_sha256={self.exclusions_sha256}",
            fingerprint([_source_hash(), self.identities]),
        )

    def propose(self, history, batch_size):
        return self._propose(history, batch_size, self._generate)

    def _generate(self, history):
        for unit, identity in zip(self.units, self.identities, strict=True):
            unit.check()
            if unit.policy_sha256 != identity:
                raise ValueError("frozen categorical checkpoint changed")
        round_index = len(self.batches) + 1
        seed = int(fingerprint(["frozen-native-categorical-v1", self.seed, round_index])[:15], 16)
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        lengths = tuple(len(row["sequence"]) for row in history[:512])
        blocked = (
            self.excluded | frozenset(self.reserves) | frozenset(row["sequence"] for row in history)
        )
        selected, records, attempts = [], [], 0
        while len(selected) < 14 and attempts < 480:
            # Rotate the first checkpoint so fourteen seats do not always give
            # the same four checkpoints twice the exposure of the others.
            offset = ((round_index - 1) * 14) % len(self.units)
            ordered_units = self.units[offset:] + self.units[:offset]
            for unit in ordered_units:
                if monotonic() >= self.deadline:
                    raise TimeoutError("categorical diffusion exceeded original deadline")
                if attempts >= 480 or len(selected) == 14:
                    break
                length = lengths[int(rng.integers(len(lengths)))]
                ordinal = (round_index - 1) * 480 + attempts
                (trace,) = sample_native_proposals(
                    unit.model,
                    ("A" * length,),
                    start_levels=(unit.model.config.levels,),
                    seed=seed,
                    ordinals=(ordinal,),
                )
                attempts += 1
                sequence = trace.endpoint
                accepted = _valid(sequence) and sequence not in blocked and sequence not in selected
                records.append(
                    dict(
                        triple=unit.triple,
                        policy_sha256=unit.policy_sha256,
                        seed=seed,
                        ordinal=ordinal,
                        parent_length=length,
                        start_level=unit.model.config.levels,
                        trace_sha256=fingerprint(asdict(trace)),
                        endpoint=sequence,
                        accepted=accepted,
                    )
                )
                if trace.model_sha256 != unit.policy_sha256:
                    raise ValueError("categorical trace checkpoint identity differs")
                if accepted:
                    selected.append(sequence)
        if len(selected) != 14:
            raise RuntimeError(
                "categorical attempt cap exhausted without fourteen unique eligible peptides"
            )
        return selected, dict(attempts=attempts, traces=records, policies=self.identities)

    def finalize(self, history):
        return self._finalize(
            history, "matched_frozen_native_categorical_sampling_not_legacy_posthoc_predictor"
        )


def make_baseline_provider(
    protocol, *, arm_name, seed, reserves, excluded_sequences, deadline, units=None
):
    """Factory for the controller's already audited common inputs and assets."""
    kwargs = dict(
        seed=seed, reserves=reserves, excluded_sequences=excluded_sequences, deadline=deadline
    )
    if arm_name == "genetic_algorithm":
        return MatchedGeneticProvider(protocol, **kwargs)
    if arm_name == "categorical":
        return NativeCategoricalProvider(protocol, units=units, **kwargs)
    raise ValueError("unknown matched baseline arm")


def build_baseline_provider(
    protocol,
    *,
    arm_name,
    seed,
    initial,
    reserves,
    excluded_sequences,
    checkpoints,
    audit,
    device,
    deadline,
    output_root,
    admission_evidence,
):
    """Provision audited native assets and save source/common-input provenance."""
    from amp_challenge.evaluation.peptide_proxy_protocol import verify_initial_manifest
    from amp_challenge.generators.diffusion.native_baseline_operators import _NativeUnit
    from amp_challenge.generators.search.native_static_campaign import _load_audited_units

    def check():
        if monotonic() >= deadline:
            raise TimeoutError("baseline provisioning exhausted original deadline")

    check()
    verify_initial_manifest(initial, protocol)
    if initial["seed"] != seed:
        raise ValueError("initial manifest seed differs from baseline seed")
    root, checkpoints = Path(output_root), Path(checkpoints)
    root.mkdir(exist_ok=False)
    training, manifests = set(), {}
    for ordinal, triple in enumerate(TRIPLES):
        check()
        directory = checkpoints / f"{ordinal:02d}"
        manifest_bytes = (directory / "manifest.json").read_bytes()
        projection = (directory / "training_projection.jsonl").read_bytes()
        manifest = json.loads(manifest_bytes)
        projection_hash = hashlib.sha256(projection).hexdigest()
        if projection_hash != manifest["artifacts"]["training_projection.jsonl"]["sha256"]:
            raise ValueError("baseline generator training projection changed")
        training.update(json.loads(line)["sequence"] for line in projection.splitlines() if line)
        manifests[triple] = dict(
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            training_projection_sha256=projection_hash,
        )
    if not training <= frozenset(excluded_sequences):
        raise ValueError("common exclusions omit actual generator training sequences")
    units = None
    if arm_name == "categorical":
        initializations, corpora = _load_audited_units(
            checkpoints=checkpoints,
            audit=audit,
            device=device,
            check=check,
        )
        units = tuple(
            _NativeUnit(initialization, corpus)
            for initialization, corpus in zip(initializations, corpora, strict=True)
        )
    provider = make_baseline_provider(
        protocol,
        arm_name=arm_name,
        seed=seed,
        reserves=reserves,
        excluded_sequences=excluded_sequences,
        deadline=deadline,
        units=units,
    )
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    evidence = {
        "arm_name": arm_name,
        "seed": seed,
        "admission_evidence": admission_evidence,
        "initial_manifest_sha256": initial["manifest_sha256"],
        "reserves_sha256": provider.reserves_sha256,
        "common_exclusions_sha256": provider.exclusions_sha256,
        "generator_sources": manifests,
        "checkpoint_audit_sha256": hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
        "source_hashes": source_hashes,
        "policy_identities": list(provider.identities) if units is not None else [],
        "adaptation": "frozen_native_fullmask_sampling"
        if units is not None
        else "fixed_default_scalar_genetic_not_tuned",
    }
    provider.capabilities = replace(
        provider.capabilities,
        implementation_sha256=fingerprint(evidence),
        admission_evidence=provider.capabilities.admission_evidence
        + "; provisioning="
        + fingerprint(evidence),
    )
    with (root / "provenance.json").open("x") as stream:
        json.dump(evidence, stream, sort_keys=True, indent=2, allow_nan=False)
    check()
    return provider

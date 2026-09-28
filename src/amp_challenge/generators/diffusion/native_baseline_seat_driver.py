"""Execute native pool generation and frozen-posterior method-seat selection.

Controller-owned history authentication, posterior fitting, private reserves and
oracle dispatch stay outside this component. Existing directories cannot be
reopened: retained artifacts are evidence, not process-restoration authority.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.generators.diffusion.native_baseline_context import NativeBaselineContextEnsemble
from amp_challenge.generators.diffusion.native_baseline_context_records import (
    ContextEnvelope,
    document_sha,
    plain,
)
from amp_challenge.generators.diffusion.native_baseline_operators import (
    MODES,
    NativeBaselineEnsemble,
    sequence_id,
    verify_native_pool,
)
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    read_native_posterior,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import canonical_json_bytes
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.representations.candidate_features import _exclusive_write

CONTRACT = "native_baseline_selection_v1_20260914"


@dataclass(frozen=True, slots=True)
class BaselineSeatSchedule:
    mode: str
    round_index: int
    pool_sha256: str
    posterior_binding_sha256: str
    permitted_ids_sha256: str
    waves: tuple[tuple[str, ...], ...]
    sha256: str


class NativeBaselineSeatDriver:
    """One live baseline owner, with exclusive persistent phase artifacts.

    For updating baselines pass NativeBaselineContextEnsemble, never the old
    unfiltered update path. Categorical alone accepts NativeBaselineEnsemble.
    `generate` exposes the visible pool; `select` accepts only a subset of its
    IDs from the private controller. Neither operation calls an oracle.
    """

    def __init__(
        self, ensemble, output_root, *, deadline_monotonic, clock_epoch_id, monotonic=time.monotonic
    ):
        if type(ensemble) is NativeBaselineContextEnsemble:
            inner = ensemble.inner
            if (deadline_monotonic, clock_epoch_id) != (
                ensemble.context.deadline_monotonic,
                ensemble.context.clock_epoch_id,
            ) or monotonic is not ensemble.monotonic:
                raise ValueError("context baseline must retain its original clock")
        elif type(ensemble) is NativeBaselineEnsemble and ensemble.mode == MODES[0]:
            inner = ensemble
        else:
            raise TypeError("updating baseline requires the context-eligible native operator")
        if inner._history is not None or inner._pool is not None:
            raise ValueError("seat driver requires an unused native baseline")
        if (
            type(deadline_monotonic) is not float
            or not math.isfinite(deadline_monotonic)
            or type(clock_epoch_id) is not str
            or not clock_epoch_id
        ):
            raise ValueError("original finite deadline and clock epoch required")
        self._ensemble, self._inner, self._clock = ensemble, inner, monotonic
        self._deadline = deadline_monotonic
        self._root = Path(output_root)
        if not self._root.is_absolute():
            raise ValueError("absolute persistent output directory required")
        self._terminal, self._history, self._pool, self._schedule = False, None, None, None
        self._generation_key, self._selection_key = None, None
        self._fixed_identity = self._identity()
        self._last_clock = -math.inf
        self._check_clock()
        if deadline_monotonic - self._last_clock > 7200:
            raise ValueError("baseline deadline exceeds the original 7200-second ceiling")
        self._root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._save(
            "started.json",
            {
                "contract": CONTRACT,
                "mode": inner.mode,
                "run_id": inner.run_id,
                "seed": inner.seed,
                "deadline_monotonic": deadline_monotonic,
                "clock_epoch_id": clock_epoch_id,
                "protocol_ten_checkpoint_mixture": inner.protocol_ten_checkpoint_mixture,
                "oracle_calls": 0,
                "scientific_evidence_accepted": False,
            },
        )

    def _identity(self):
        return document_sha(
            (
                self._inner.mode,
                self._inner.context,
                self._inner.run_id,
                self._inner.seed,
                self._inner.oracle_bundle_sha256,
            )
        )

    def _check_clock(self):
        now = self._clock()
        if (
            type(now) not in (float, int)
            or not math.isfinite(now)
            or now < self._last_clock
            or now >= self._deadline
        ):
            raise TimeoutError("native baseline original deadline/monotonic clock failed")
        self._last_clock = now
        if self._terminal:
            raise RuntimeError("native baseline driver is terminal after failed work")
        if self._identity() != self._fixed_identity:
            raise ValueError("native baseline fixed mode/context/source changed")

    def _save(self, name, value):
        _exclusive_write(self._root / name, canonical_json_bytes(plain(value)))

    def _fail(self, phase, error):
        if not self._terminal:
            self._terminal = True
            operator_failure = getattr(self._ensemble, "last_failure", None)
            if type(operator_failure) is ContextEnvelope:
                operator_failure = {
                    "sha256": operator_failure.sha256,
                    "document": operator_failure.document(),
                }
            self._save(
                "failure.json",
                {
                    "phase": phase,
                    "type": type(error).__name__,
                    "message": str(error),
                    "round_index": None if self._history is None else self._history.round_index,
                    "last_monotonic": self._last_clock,
                    "operator_failure": operator_failure,
                    "oracle_calls": 0,
                    "scientific_evidence_accepted": False,
                },
            )

    def generate(
        self, history, *, expected_previous_head_sha256, eligibility=None, expectations=None
    ):
        """Advance native policy and execute its complete, independently replayed pool."""
        try:
            self._check_clock()
            if type(history) is not VerifiedHistorySnapshot:
                raise TypeError("exact verified charged history required")
            history.__post_init__()
            if not history.complete or history.round_index == 29:
                raise ValueError("complete nonterminal history required")
            if history.previous_wave_head_sha256 != expected_previous_head_sha256:
                raise ValueError("controller history predecessor differs")
            key = document_sha((history, expected_previous_head_sha256, eligibility, expectations))
            if self._generation_key == key and self._pool is not None:
                return self._pool
            if self._history is not None:
                if self._inner.mode != MODES[2]:
                    raise ValueError("static schedule cannot consume adaptive responses")
                if self._schedule is None or history.round_index != self._history.round_index + 1:
                    raise ValueError("next round requires the prior selected method seats")
                prefix = self._history.observations
                if (
                    history.observations[: len(prefix)] != prefix
                    or tuple(
                        row.sequence for row in history.observations[len(prefix) : len(prefix) + 14]
                    )
                    != self._schedule.waves[0]
                ):
                    raise ValueError("charged prefix/prior 14 method seats differ")
            directory = self._root / f"round-{history.round_index:02d}"
            directory.mkdir()
            self._save(
                f"{directory.name}/input.json",
                {
                    "history": history,
                    "eligibility": eligibility,
                    "expectations": expectations,
                    "generation_key": key,
                },
            )
            self._pool, self._schedule, self._selection_key = None, None, None
            if type(self._ensemble) is NativeBaselineContextEnsemble:
                envelope = self._ensemble.advance(history, eligibility, expected=expectations)
                self._save(f"{directory.name}/update.json", envelope.document())
            else:
                if eligibility is not None or expectations is not None:
                    raise ValueError("categorical does not consume update eligibility")
                updates = self._ensemble.advance(
                    history, expected_previous_head_sha256=expected_previous_head_sha256
                )
                self._save(f"{directory.name}/update.json", updates)
            self._history = history
            self._check_clock()
            block = 256 if self._inner.mode == MODES[2] else 2048
            ordinal = 0
            while True:
                self._check_clock()
                if type(self._ensemble) is NativeBaselineContextEnsemble:
                    pool = self._ensemble.propose(
                        expected_update_envelope_sha256=self._ensemble.last_envelope.sha256,
                        max_new_attempts=block,
                    )
                else:
                    pool = self._ensemble.propose(max_new_attempts=block)
                    verify_native_pool(self._ensemble, pool)
                self._save(f"{directory.name}/pool-{ordinal:03d}.json", pool)
                self._check_clock()
                if pool.status != "in_progress":
                    break
                ordinal += 1
                block = 128
            if pool.status != "complete":
                raise ValueError("native generation attempt cap exhausted; no replacement pool")
            self._pool, self._generation_key = pool, key
            return pool
        except BaseException as error:
            self._fail("generate", error)
            raise

    def select(self, posterior, *, expected_binding, permitted_sequence_ids):
        """Score through the existing posterior seam and publish 392 or 14 seats.

        The caller must pin expected_binding independently of the supplied port.
        The actual NativeFeaturePosterior is supported without an adapter.
        """
        try:
            self._check_clock()
            if self._pool is None:
                raise ValueError("completed native pool required before scoring")
            pool = self._pool
            if type(expected_binding) is not FrozenNativePosteriorBinding:
                raise TypeError("exact frozen posterior binding required")
            expected_binding.__post_init__()
            binding_key = document_sha(expected_binding)
            policies = self._inner.policy_identities
            for unit in self._inner._units:
                unit.check()
            if (
                expected_binding.history_sha256 != self._history.sha256
                or expected_binding.objective_context_sha256 != self._inner.context.context_sha256
            ):
                raise ValueError("posterior must be fitted to this exact charged history")
            visible_ids = frozenset(map(sequence_id, pool.accepted_sequences))
            if (
                type(permitted_sequence_ids) is not frozenset
                or not permitted_sequence_ids <= visible_ids
            ):
                raise ValueError("controller permission must be a subset of visible pool IDs")
            permitted_key = document_sha(tuple(sorted(permitted_sequence_ids)))
            key = document_sha((pool.sha256, binding_key, permitted_key))
            if self._schedule is not None:
                if key != self._selection_key:
                    raise ValueError("frozen baseline selection cannot be reranked")
                return self._schedule
            directory = f"round-{pool.round_index:02d}"
            self._save(
                f"{directory}/selection-started.json",
                {
                    "posterior_binding": expected_binding,
                    "permitted_sequence_ids": tuple(sorted(permitted_sequence_ids)),
                    "selection_key": key,
                },
            )
            scores = []
            for offset in range(0, len(pool.accepted_sequences), 128):
                self._check_clock()
                sequences = pool.accepted_sequences[offset : offset + 128]
                batch = read_native_posterior(
                    posterior,
                    sequences,
                    expected_binding=expected_binding,
                    context=self._inner.context,
                )
                self._save(f"{directory}/scores-{offset // 128:03d}.json", batch)
                self._check_clock()
                for unit in self._inner._units:
                    unit.check()
                if (
                    document_sha(expected_binding) != binding_key
                    or self._inner.policy_identities != policies
                    or self._inner._history != self._history
                    or self._inner._pool != pool
                ):
                    raise ValueError("baseline history/pool/posterior changed during scoring")
                scores.extend(batch.scores)
            first = {
                attempt.trace.endpoint: attempt.attempt_index
                for attempt in pool.attempts
                if attempt.rejection_reason is None
            }
            scores_by_sequence = dict(zip(pool.accepted_sequences, scores, strict=True))
            ranking = sorted(
                (
                    sequence
                    for sequence, score in zip(pool.accepted_sequences, scores, strict=True)
                    if score.feasible and sequence_id(sequence) in permitted_sequence_ids
                ),
                key=lambda sequence: (
                    -self._inner.context.scalarize(scores_by_sequence[sequence].objectives),
                    sequence_id(sequence),
                    first[sequence],
                ),
            )
            needed = 14 if self._inner.mode == MODES[2] else 392
            self._save(f"{directory}/ranking.json", {"sequences": ranking, "required": needed})
            if len(ranking) < needed:
                raise ValueError(
                    f"baseline underfilled: {len(ranking)} eligible for {needed} seats"
                )
            waves = tuple(tuple(ranking[offset : offset + 14]) for offset in range(0, needed, 14))
            fields = (
                self._inner.mode,
                pool.round_index,
                pool.sha256,
                binding_key,
                permitted_key,
                waves,
            )
            schedule = BaselineSeatSchedule(*fields, document_sha((CONTRACT, fields)))
            self._check_clock()
            self._save(f"{directory}/schedule.json", asdict(schedule))
            self._check_clock()
            self._schedule, self._selection_key = schedule, key
            return schedule
        except BaseException as error:
            self._fail("select", error)
            raise

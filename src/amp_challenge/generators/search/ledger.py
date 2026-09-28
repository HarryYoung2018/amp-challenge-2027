"""Append-only proposal ledger with a sequence-keyed transposition DAG."""

from __future__ import annotations

from dataclasses import dataclass

from amp_challenge.generators.search.records import (
    BranchRecord,
    EdgeRecord,
    EvaluationRecord,
    ProposalRecord,
    RolloutRecord,
    _canonical_sequence_key,
)


@dataclass(frozen=True, slots=True)
class TranspositionEvent:
    """Admission result for one proposal without discarding failed events."""

    event_index: int
    proposal_id: str
    edge_id: str
    sequence_key: str
    first_proposal_id: str
    duplicate: bool
    hard_valid: bool
    selected_for_evaluation: bool
    has_any_cached_evaluation: bool
    dag_edge_admitted: bool
    graph_rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class SequenceEntry:
    """Immutable view of all proposals and evaluations for one sequence."""

    sequence_key: str
    proposal_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]

    @property
    def first_proposal_id(self) -> str:
        return self.proposal_ids[0]


class SearchLedger:
    """Retain proposal provenance while exposing a cycle-free lineage DAG."""

    def __init__(self) -> None:
        self._branches: dict[str, BranchRecord] = {}
        self._rollouts: dict[str, RolloutRecord] = {}
        self._proposals: dict[str, ProposalRecord] = {}
        self._edges: dict[str, EdgeRecord] = {}
        self._evaluations: dict[str, EvaluationRecord] = {}
        self._events: list[TranspositionEvent] = []
        self._selection_set_configs: dict[str, tuple[tuple[str, ...], str, int]] = {}
        self._proposal_ids_by_sequence: dict[str, list[str]] = {}
        self._evaluation_ids_by_sequence: dict[str, list[str]] = {}
        self._evaluation_ids_by_cache_key: dict[tuple[str, str, str], list[str]] = {}
        self._dag_nodes: set[str] = set()
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}

    @property
    def events(self) -> tuple[TranspositionEvent, ...]:
        return tuple(self._events)

    @property
    def invalid_events(self) -> tuple[TranspositionEvent, ...]:
        return tuple(event for event in self._events if not event.hard_valid)

    @property
    def duplicate_events(self) -> tuple[TranspositionEvent, ...]:
        return tuple(event for event in self._events if event.duplicate)

    @property
    def unevaluated_events(self) -> tuple[TranspositionEvent, ...]:
        evaluated = {evaluation.proposal_id for evaluation in self._evaluations.values()}
        return tuple(event for event in self._events if event.proposal_id not in evaluated)

    @property
    def branches(self) -> tuple[BranchRecord, ...]:
        return tuple(self._branches.values())

    @property
    def rollouts(self) -> tuple[RolloutRecord, ...]:
        return tuple(self._rollouts.values())

    @property
    def proposals(self) -> tuple[ProposalRecord, ...]:
        return tuple(self._proposals.values())

    @property
    def edges(self) -> tuple[EdgeRecord, ...]:
        return tuple(self._edges.values())

    @property
    def evaluations(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._evaluations.values())

    @property
    def dag_sequences(self) -> frozenset[str]:
        return frozenset(self._dag_nodes)

    def add_branch(self, branch: BranchRecord) -> None:
        if not isinstance(branch, BranchRecord):
            raise TypeError("branch must be a BranchRecord")
        if branch.branch_id in self._branches:
            raise ValueError(f"duplicate branch_id: {branch.branch_id}")
        if branch.parent_branch_id is not None and branch.parent_branch_id not in self._branches:
            raise ValueError(f"unknown parent branch: {branch.parent_branch_id}")
        self._branches[branch.branch_id] = branch

    def add_rollout(self, rollout: RolloutRecord) -> None:
        if not isinstance(rollout, RolloutRecord):
            raise TypeError("rollout must be a RolloutRecord")
        if rollout.rollout_id in self._rollouts:
            raise ValueError(f"duplicate rollout_id: {rollout.rollout_id}")
        try:
            branch = self._branches[rollout.branch_id]
        except KeyError as error:
            raise ValueError(f"unknown rollout branch: {rollout.branch_id}") from error
        if rollout.policy_version != branch.policy_version:
            raise ValueError("rollout and branch policy versions must match")
        self._rollouts[rollout.rollout_id] = rollout

    def rollout(self, rollout_id: str) -> RolloutRecord:
        """Return one immutable rollout record by ID."""

        return self._rollouts[rollout_id]

    def proposal(self, proposal_id: str) -> ProposalRecord:
        """Return one immutable proposal record by ID."""

        return self._proposals[proposal_id]

    def edge(self, edge_id: str) -> EdgeRecord:
        """Return one immutable edge record by ID."""

        return self._edges[edge_id]

    def record_proposal(self, proposal: ProposalRecord, edge: EdgeRecord) -> TranspositionEvent:
        """Append a proposal and attempt to add its edge to the lineage DAG."""

        if not isinstance(proposal, ProposalRecord):
            raise TypeError("proposal must be a ProposalRecord")
        if not isinstance(edge, EdgeRecord):
            raise TypeError("edge must be an EdgeRecord")
        if proposal.proposal_id in self._proposals:
            raise ValueError(f"duplicate proposal_id: {proposal.proposal_id}")
        if edge.edge_id in self._edges:
            raise ValueError(f"duplicate edge_id: {edge.edge_id}")
        if edge.proposal_id != proposal.proposal_id:
            raise ValueError("edge and proposal IDs must match")
        if edge.rollout_id != proposal.rollout_id:
            raise ValueError("edge and proposal rollout IDs must match")
        try:
            rollout = self._rollouts[proposal.rollout_id]
        except KeyError as error:
            raise ValueError(f"unknown rollout: {proposal.rollout_id}") from error
        if proposal.policy_version != rollout.policy_version:
            raise ValueError("proposal and rollout policy versions must match")
        self._validate_selection_set(proposal)

        sequence_key = proposal.sequence_key
        previous_ids = self._proposal_ids_by_sequence.get(sequence_key, [])
        duplicate = bool(previous_ids)
        first_proposal_id = previous_ids[0] if previous_ids else proposal.proposal_id
        has_any_cached_evaluation = bool(self._evaluation_ids_by_sequence.get(sequence_key))
        admitted, graph_rejection_reason = self._admit_dag_edge(
            sequence_key=sequence_key,
            parent_sequence_keys=edge.parent_sequence_keys,
            hard_valid=proposal.hard_valid,
        )

        self._proposals[proposal.proposal_id] = proposal
        self._edges[edge.edge_id] = edge
        self._proposal_ids_by_sequence.setdefault(sequence_key, []).append(proposal.proposal_id)
        event = TranspositionEvent(
            event_index=len(self._events),
            proposal_id=proposal.proposal_id,
            edge_id=edge.edge_id,
            sequence_key=sequence_key,
            first_proposal_id=first_proposal_id,
            duplicate=duplicate,
            hard_valid=proposal.hard_valid,
            selected_for_evaluation=proposal.selection.selected,
            has_any_cached_evaluation=has_any_cached_evaluation,
            dag_edge_admitted=admitted,
            graph_rejection_reason=graph_rejection_reason,
        )
        self._events.append(event)
        return event

    def validate_selection_sets(self) -> None:
        """Finalize selection pools after every proposal in a round is recorded."""

        for selection_set_id, (eligible_ids, _, _) in self._selection_set_configs.items():
            missing = tuple(
                proposal_id for proposal_id in eligible_ids if proposal_id not in self._proposals
            )
            if missing:
                raise ValueError(
                    f"selection set {selection_set_id!r} has unlogged members: {missing}"
                )
            mismatched = tuple(
                proposal_id
                for proposal_id in eligible_ids
                if self._proposals[proposal_id].selection.selection_set_id != selection_set_id
            )
            if mismatched:
                raise ValueError(
                    f"selection set {selection_set_id!r} has mismatched members: {mismatched}"
                )

    def record_evaluation(self, evaluation: EvaluationRecord) -> None:
        """Append an evaluation and expose it through every sequence transposition."""

        if not isinstance(evaluation, EvaluationRecord):
            raise TypeError("evaluation must be an EvaluationRecord")
        if evaluation.evaluation_id in self._evaluations:
            raise ValueError(f"duplicate evaluation_id: {evaluation.evaluation_id}")
        try:
            proposal = self._proposals[evaluation.proposal_id]
        except KeyError as error:
            raise ValueError(f"unknown evaluation proposal: {evaluation.proposal_id}") from error
        if not proposal.hard_valid:
            raise ValueError("an invalid proposal cannot be evaluated")
        if not proposal.selection.selected:
            raise ValueError("proposal was not selected for evaluation")
        self._evaluations[evaluation.evaluation_id] = evaluation
        self._evaluation_ids_by_sequence.setdefault(proposal.sequence_key, []).append(
            evaluation.evaluation_id
        )
        cache_key = (
            proposal.sequence_key,
            evaluation.fidelity,
            evaluation.evaluator_version,
        )
        self._evaluation_ids_by_cache_key.setdefault(cache_key, []).append(evaluation.evaluation_id)

    def sequence_entry(self, sequence: str) -> SequenceEntry:
        sequence_key = _canonical_sequence_key(sequence)
        try:
            proposal_ids = self._proposal_ids_by_sequence[sequence_key]
        except KeyError as error:
            raise KeyError(sequence_key) from error
        return SequenceEntry(
            sequence_key=sequence_key,
            proposal_ids=tuple(proposal_ids),
            evaluation_ids=tuple(self._evaluation_ids_by_sequence.get(sequence_key, ())),
        )

    def cached_evaluations(
        self, sequence: str, *, fidelity: str, evaluator_version: str
    ) -> tuple[EvaluationRecord, ...]:
        """Return only cache entries compatible with fidelity and evaluator version."""

        sequence_key = _canonical_sequence_key(sequence)
        if not isinstance(fidelity, str) or not fidelity.strip():
            raise ValueError("fidelity must be a non-empty string")
        if not isinstance(evaluator_version, str) or not evaluator_version.strip():
            raise ValueError("evaluator_version must be a non-empty string")
        cache_key = (sequence_key, fidelity.strip(), evaluator_version.strip())
        return tuple(
            self._evaluations[evaluation_id]
            for evaluation_id in self._evaluation_ids_by_cache_key.get(cache_key, ())
        )

    def all_cached_evaluations(self, sequence: str) -> tuple[EvaluationRecord, ...]:
        """Return every version for audit, without implying cache compatibility."""

        sequence_key = _canonical_sequence_key(sequence)
        return tuple(
            self._evaluations[evaluation_id]
            for evaluation_id in self._evaluation_ids_by_sequence.get(sequence_key, ())
        )

    def parents(self, sequence: str) -> frozenset[str]:
        sequence_key = _canonical_sequence_key(sequence)
        return frozenset(self._parents.get(sequence_key, ()))

    def children(self, sequence: str) -> frozenset[str]:
        sequence_key = _canonical_sequence_key(sequence)
        return frozenset(self._children.get(sequence_key, ()))

    def _admit_dag_edge(
        self,
        *,
        sequence_key: str,
        parent_sequence_keys: tuple[str, ...],
        hard_valid: bool,
    ) -> tuple[bool, str | None]:
        if not hard_valid:
            return False, "hard_invalid"
        missing_parents = tuple(
            parent for parent in parent_sequence_keys if parent not in self._dag_nodes
        )
        if missing_parents:
            return False, "unknown_parent"
        if sequence_key in parent_sequence_keys or any(
            self._is_reachable(sequence_key, parent) for parent in parent_sequence_keys
        ):
            return False, "cycle"

        self._dag_nodes.add(sequence_key)
        self._parents.setdefault(sequence_key, set()).update(parent_sequence_keys)
        self._children.setdefault(sequence_key, set())
        for parent in parent_sequence_keys:
            self._children.setdefault(parent, set()).add(sequence_key)
            self._parents.setdefault(parent, set())
        return True, None

    def _validate_selection_set(self, proposal: ProposalRecord) -> None:
        decision = proposal.selection
        config = (
            decision.eligible_proposal_ids,
            decision.policy_version,
            decision.seed,
        )
        previous = self._selection_set_configs.get(decision.selection_set_id)
        if previous is not None and previous != config:
            raise ValueError("selection set configuration is inconsistent")
        for proposal_id in decision.eligible_proposal_ids:
            previous_proposal = self._proposals.get(proposal_id)
            if (
                previous_proposal is not None
                and previous_proposal.selection.selection_set_id != decision.selection_set_id
            ):
                raise ValueError("selection set membership is inconsistent")
        for selection_set_id, (eligible_ids, _, _) in self._selection_set_configs.items():
            if (
                proposal.proposal_id in eligible_ids
                and selection_set_id != decision.selection_set_id
            ):
                raise ValueError("selection set membership is inconsistent")
        self._selection_set_configs.setdefault(decision.selection_set_id, config)

    def _is_reachable(self, start: str, target: str) -> bool:
        if start not in self._dag_nodes:
            return False
        pending = [start]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self._children.get(current, ()))
        return False

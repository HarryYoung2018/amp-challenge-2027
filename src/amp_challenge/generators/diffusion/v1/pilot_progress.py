"""Immutable operational progress records for the native-diffusion v1 evaluator.

This module is deliberately standard-library-only.  Progress records are not
scientific artifacts and never enter a trainer, evaluator, or pilot semantic
bundle.  Each record says that its process actor has *entered* one exact phase;
the following record is therefore the durable proof that the preceding phase
returned.  A failed evaluator leaves an immutable, self-consistent prefix.  A
complete trace is sealed read-only and becomes externally PID/digest anchored
only when the supervisor publishes its separate operational telemetry.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ARTIFACT = "native_categorical_diffusion_v1_r128_evaluator_progress"
CHECKPOINT_STEPS = (250, 500, 1000, 2000, 4000)
SUPERVISOR_PROGRESS_PHASES = (
    "score_projection_staging",
    "evaluator_child_launch",
)
EVALUATOR_PROCESS_PROGRESS_PHASES = (
    "release_authenticated",
    "score_input_authentication",
    "cuda_runtime_attestation",
    *(
        phase
        for step in CHECKPOINT_STEPS
        for phase in (
            f"archive_checkpoint_{step:06d}_load",
            f"archive_checkpoint_{step:06d}_inference",
        )
    ),
    "scoring_archive_construction",
    "fold_method_scoring",
    "bundle_serialization",
    "prepublication_input_revalidation",
    "bundle_publication",
    "published_bundle_verification",
    "reinference_setup",
    *(
        phase
        for step in CHECKPOINT_STEPS
        for phase in (
            f"reinference_checkpoint_{step:06d}_load",
            f"reinference_checkpoint_{step:06d}_inference",
        )
    ),
    "reinference_comparison",
    "final_input_revalidation",
    "final_bundle_verification",
    "evaluation_result_ready",
)
EVALUATOR_PROGRESS_PHASES = SUPERVISOR_PROGRESS_PHASES + EVALUATOR_PROCESS_PROGRESS_PHASES
_PHASE_ACTORS = (
    *(("supervisor",) * len(SUPERVISOR_PROGRESS_PHASES)),
    *(("evaluator",) * len(EVALUATOR_PROCESS_PROGRESS_PHASES)),
)

_RECORD_FIELDS = (
    "actor",
    "artifact",
    "child_contract_sha256",
    "elapsed_monotonic_ns",
    "git_commit",
    "monotonic_origin_ns",
    "outer_fold",
    "parent_contract_sha256",
    "phase",
    "previous_marker_sha256",
    "process_cpu_ns",
    "process_pid",
    "schema_version",
    "score_release_sha256",
    "sequence_number",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_MAX_MARKER_BYTES = 8_192
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


@dataclass(frozen=True, slots=True)
class EvaluatorProgressTrace:
    """Verified identity of one sealed trace or self-consistent durable prefix."""

    phases: tuple[str, ...]
    marker_sha256s: tuple[str, ...]
    monotonic_origin_ns: int | None
    elapsed_monotonic_ns: tuple[int, ...]
    process_cpu_ns: tuple[int, ...]
    supervisor_process_pid: int | None
    evaluator_process_pid: int | None
    terminal_sha256: str | None
    complete: bool

    def __post_init__(self) -> None:
        if self.phases != EVALUATOR_PROGRESS_PHASES[: len(self.phases)]:
            raise ValueError("progress trace phases are not an exact prefix")
        counts = {
            len(self.phases),
            len(self.marker_sha256s),
            len(self.elapsed_monotonic_ns),
            len(self.process_cpu_ns),
        }
        if len(counts) != 1:
            raise ValueError("progress trace field counts differ")
        for value in self.marker_sha256s:
            _sha256(value, label="progress marker SHA-256")
        if self.phases:
            _clock_value(
                self.monotonic_origin_ns,
                label="trace monotonic origin",
            )
        elif self.monotonic_origin_ns is not None:
            raise ValueError("empty progress trace cannot claim a monotonic origin")
        previous_elapsed = -1
        previous_process_cpu = {"supervisor": -1, "evaluator": -1}
        for index, (elapsed, process_cpu) in enumerate(
            zip(self.elapsed_monotonic_ns, self.process_cpu_ns, strict=True)
        ):
            elapsed = _clock_value(elapsed, label="trace elapsed monotonic clock")
            process_cpu = _clock_value(process_cpu, label="trace process CPU clock")
            actor = _PHASE_ACTORS[index]
            if elapsed < previous_elapsed or process_cpu < previous_process_cpu[actor]:
                raise ValueError("progress trace clock regressed")
            previous_elapsed = elapsed
            previous_process_cpu[actor] = process_cpu
        actors = _PHASE_ACTORS[: len(self.phases)]
        if "supervisor" in actors:
            _process_pid(self.supervisor_process_pid, label="supervisor process PID")
        elif self.supervisor_process_pid is not None:
            raise ValueError("empty progress trace cannot claim a supervisor process")
        if "evaluator" in actors:
            _process_pid(self.evaluator_process_pid, label="evaluator process PID")
        elif self.evaluator_process_pid is not None:
            raise ValueError("supervisor-only progress cannot claim an evaluator process")
        if (
            self.supervisor_process_pid is not None
            and self.supervisor_process_pid == self.evaluator_process_pid
        ):
            raise ValueError("supervisor and evaluator progress require distinct process PIDs")
        if self.phases:
            if self.terminal_sha256 != self.marker_sha256s[-1]:
                raise ValueError("progress terminal digest differs from the final marker")
        elif self.terminal_sha256 is not None:
            raise ValueError("empty progress trace cannot claim a terminal digest")
        if type(self.complete) is not bool:
            raise TypeError("progress completion must be an exact bool")
        if self.complete and self.phases != EVALUATOR_PROGRESS_PHASES:
            raise ValueError("progress completion requires the exact phase census")


class EvaluatorProgressWriter:
    """Publish one actor's exact portion of a no-overwrite progress sequence."""

    __slots__ = (
        "_actor",
        "_child_contract_sha256",
        "_directory",
        "_git_commit",
        "_monotonic_ns",
        "_monotonic_origin",
        "_outer_fold",
        "_parent_contract_sha256",
        "_previous_elapsed_ns",
        "_previous_process_cpu_ns",
        "_previous_sha256",
        "_process_cpu_origin",
        "_process_pid",
        "_process_time_ns",
        "_score_release_sha256",
        "_sealed",
        "_sequence_number",
        "_supervisor_process_pid",
    )

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        child_contract_sha256: str,
        parent_contract_sha256: str,
        git_commit: str,
        outer_fold: int,
        score_release_sha256: str,
        _supervisor_process_pid: int | None = None,
        _monotonic_ns: Callable[[], int] = time.monotonic_ns,
        _process_time_ns: Callable[[], int] = time.process_time_ns,
    ) -> None:
        """Start a fresh trace owned by the supervisor process."""

        self._directory = _absolute_directory(directory, required_mode=0o700)
        if tuple(self._directory.iterdir()):
            raise FileExistsError("evaluator progress directory must be fresh and empty")
        self._bind_identity(
            child_contract_sha256=child_contract_sha256,
            parent_contract_sha256=parent_contract_sha256,
            git_commit=git_commit,
            outer_fold=outer_fold,
            score_release_sha256=score_release_sha256,
        )
        self._bind_directory_fold()
        self._actor = "supervisor"
        self._process_pid = _process_pid(
            os.getpid() if _supervisor_process_pid is None else _supervisor_process_pid,
            label="supervisor process PID",
        )
        self._supervisor_process_pid = self._process_pid
        if not callable(_monotonic_ns) or not callable(_process_time_ns):
            raise TypeError("progress clocks must be callable")
        self._monotonic_ns = _monotonic_ns
        self._process_time_ns = _process_time_ns
        self._monotonic_origin = _clock_value(
            self._monotonic_ns(),
            label="monotonic origin",
        )
        self._process_cpu_origin = _clock_value(
            self._process_time_ns(),
            label="process CPU origin",
        )
        self._previous_elapsed_ns = -1
        self._previous_process_cpu_ns = -1
        self._previous_sha256 = self._score_release_sha256
        self._sequence_number = 0
        self._sealed = False

    @classmethod
    def resume_evaluator(
        cls,
        directory: str | os.PathLike[str],
        *,
        child_contract_sha256: str,
        parent_contract_sha256: str,
        git_commit: str,
        outer_fold: int,
        score_release_sha256: str,
        expected_supervisor_process_pid: int,
        trace_monotonic_origin_ns: int,
        _evaluator_process_pid: int | None = None,
        _monotonic_ns: Callable[[], int] = time.monotonic_ns,
        _process_time_ns: Callable[[], int] = time.process_time_ns,
    ) -> EvaluatorProgressWriter:
        """Resume only the exact two-marker supervisor-to-evaluator handoff."""

        writer = cls.__new__(cls)
        writer._directory = _absolute_directory(directory, required_mode=0o700)
        writer._bind_identity(
            child_contract_sha256=child_contract_sha256,
            parent_contract_sha256=parent_contract_sha256,
            git_commit=git_commit,
            outer_fold=outer_fold,
            score_release_sha256=score_release_sha256,
        )
        writer._bind_directory_fold()
        supervisor_pid = _process_pid(
            expected_supervisor_process_pid,
            label="expected supervisor process PID",
        )
        trace = verify_evaluator_progress(
            writer._directory,
            expected_child_contract_sha256=writer._child_contract_sha256,
            expected_parent_contract_sha256=writer._parent_contract_sha256,
            expected_git_commit=writer._git_commit,
            expected_outer_fold=writer._outer_fold,
            expected_score_release_sha256=writer._score_release_sha256,
            expected_supervisor_process_pid=supervisor_pid,
            require_complete=False,
        )
        if trace.phases != SUPERVISOR_PROGRESS_PHASES:
            raise ValueError("evaluator progress is not the exact supervisor handoff prefix")
        if trace.terminal_sha256 is None or not trace.elapsed_monotonic_ns:
            raise ValueError("evaluator progress handoff lacks its terminal identity")
        if not callable(_monotonic_ns) or not callable(_process_time_ns):
            raise TypeError("progress clocks must be callable")
        writer._actor = "evaluator"
        writer._process_pid = _process_pid(
            os.getpid() if _evaluator_process_pid is None else _evaluator_process_pid,
            label="evaluator process PID",
        )
        if writer._process_pid == supervisor_pid:
            raise ValueError("supervisor and evaluator progress require distinct process PIDs")
        writer._supervisor_process_pid = supervisor_pid
        writer._monotonic_ns = _monotonic_ns
        writer._process_time_ns = _process_time_ns
        monotonic_origin = _clock_value(
            trace_monotonic_origin_ns,
            label="trace monotonic origin",
        )
        if trace.monotonic_origin_ns != monotonic_origin:
            raise ValueError("trace monotonic origin differs from supervisor progress")
        writer._monotonic_origin = monotonic_origin
        writer._process_cpu_origin = _clock_value(
            writer._process_time_ns(),
            label="evaluator process CPU origin",
        )
        writer._previous_elapsed_ns = trace.elapsed_monotonic_ns[-1]
        writer._previous_process_cpu_ns = -1
        writer._previous_sha256 = trace.terminal_sha256
        writer._sequence_number = len(SUPERVISOR_PROGRESS_PHASES)
        writer._sealed = False
        return writer

    def _bind_identity(
        self,
        *,
        child_contract_sha256: str,
        parent_contract_sha256: str,
        git_commit: str,
        outer_fold: int,
        score_release_sha256: str,
    ) -> None:
        self._child_contract_sha256 = _sha256(
            child_contract_sha256,
            label="child contract SHA-256",
        )
        self._parent_contract_sha256 = _sha256(
            parent_contract_sha256,
            label="parent contract SHA-256",
        )
        self._git_commit = _git_commit(git_commit)
        self._outer_fold = _outer_fold(outer_fold)
        self._score_release_sha256 = _sha256(
            score_release_sha256,
            label="score-release SHA-256",
        )

    def _bind_directory_fold(self) -> None:
        if self._directory.name != str(self._outer_fold):
            raise ValueError("evaluator progress directory does not bind its outer fold")

    @property
    def actor(self) -> str:
        """Return the sole process actor this writer is allowed to advance."""

        return self._actor

    @property
    def process_pid(self) -> int:
        """Return the process PID bound to new records from this writer."""

        return self._process_pid

    @property
    def monotonic_origin_ns(self) -> int:
        """Return the origin that must cross the supervisor/evaluator handoff."""

        return self._monotonic_origin

    @property
    def next_phase(self) -> str | None:
        """Return the only phase that may be entered next."""

        if self._sequence_number == len(EVALUATOR_PROGRESS_PHASES):
            return None
        return EVALUATOR_PROGRESS_PHASES[self._sequence_number]

    def advance(self, phase: str) -> str:
        """Atomically publish the next phase and return its SHA-256 digest."""

        if self._sealed:
            raise RuntimeError("sealed evaluator progress cannot advance")
        expected = self.next_phase
        if expected is None:
            raise RuntimeError("complete evaluator progress cannot advance")
        if type(phase) is not str or phase != expected:
            raise ValueError(f"evaluator progress expected phase {expected!r}")
        expected_actor = _PHASE_ACTORS[self._sequence_number]
        if expected_actor != self._actor:
            raise RuntimeError(f"evaluator progress requires handoff to {expected_actor}")

        monotonic_now = _clock_value(
            self._monotonic_ns(),
            label="monotonic progress clock",
        )
        process_cpu_now = _clock_value(
            self._process_time_ns(),
            label="process CPU progress clock",
        )
        elapsed_ns = monotonic_now - self._monotonic_origin
        process_cpu_ns = process_cpu_now - self._process_cpu_origin
        if elapsed_ns < self._previous_elapsed_ns or process_cpu_ns < self._previous_process_cpu_ns:
            raise RuntimeError("evaluator progress clock regressed")
        if elapsed_ns < 0 or process_cpu_ns < 0:
            raise RuntimeError("evaluator progress clock precedes its origin")

        sequence_number = self._sequence_number
        document: dict[str, object] = {
            "schema_version": 1,
            "artifact": ARTIFACT,
            "actor": self._actor,
            "child_contract_sha256": self._child_contract_sha256,
            "parent_contract_sha256": self._parent_contract_sha256,
            "git_commit": self._git_commit,
            "monotonic_origin_ns": self._monotonic_origin,
            "outer_fold": self._outer_fold,
            "score_release_sha256": self._score_release_sha256,
            "process_pid": self._process_pid,
            "sequence_number": sequence_number,
            "phase": phase,
            "previous_marker_sha256": self._previous_sha256,
            "elapsed_monotonic_ns": elapsed_ns,
            "process_cpu_ns": process_cpu_ns,
        }
        payload = canonical_progress_json_bytes(document)
        digest = hashlib.sha256(payload).hexdigest()
        destination = self._directory / _marker_name(sequence_number, phase)
        _publish_no_replace(destination, payload)

        self._previous_sha256 = digest
        self._previous_elapsed_ns = elapsed_ns
        self._previous_process_cpu_ns = process_cpu_ns
        self._sequence_number += 1
        verify_evaluator_progress(
            self._directory,
            expected_child_contract_sha256=self._child_contract_sha256,
            expected_parent_contract_sha256=self._parent_contract_sha256,
            expected_git_commit=self._git_commit,
            expected_outer_fold=self._outer_fold,
            expected_score_release_sha256=self._score_release_sha256,
            expected_supervisor_process_pid=self._supervisor_process_pid,
            expected_evaluator_process_pid=(
                self._process_pid if self._actor == "evaluator" else None
            ),
            require_complete=False,
            _allow_unsealed_complete=(self._sequence_number == len(EVALUATOR_PROGRESS_PHASES)),
        )
        return digest

    def seal(self) -> EvaluatorProgressTrace:
        """Verify the complete chain and permanently seal its directory."""

        if self._sealed:
            raise RuntimeError("evaluator progress is already sealed")
        if self._actor != "evaluator":
            raise RuntimeError("only the evaluator process may seal progress")
        if self.next_phase is not None:
            raise RuntimeError("incomplete evaluator progress cannot be sealed")
        verify_evaluator_progress(
            self._directory,
            expected_child_contract_sha256=self._child_contract_sha256,
            expected_parent_contract_sha256=self._parent_contract_sha256,
            expected_git_commit=self._git_commit,
            expected_outer_fold=self._outer_fold,
            expected_score_release_sha256=self._score_release_sha256,
            expected_supervisor_process_pid=self._supervisor_process_pid,
            expected_evaluator_process_pid=self._process_pid,
            require_complete=True,
            _allow_unsealed_complete=True,
        )
        _seal_directory_read_only(self._directory)
        _fsync_directory(self._directory)
        _fsync_directory(self._directory.parent)
        self._sealed = True
        return verify_evaluator_progress(
            self._directory,
            expected_child_contract_sha256=self._child_contract_sha256,
            expected_parent_contract_sha256=self._parent_contract_sha256,
            expected_git_commit=self._git_commit,
            expected_outer_fold=self._outer_fold,
            expected_score_release_sha256=self._score_release_sha256,
            expected_supervisor_process_pid=self._supervisor_process_pid,
            expected_evaluator_process_pid=self._process_pid,
            require_complete=True,
        )


def canonical_progress_json_bytes(value: Mapping[str, object]) -> bytes:
    """Return exact compact finite canonical JSON with one trailing newline."""

    if not isinstance(value, Mapping):
        raise TypeError("progress JSON value must be a mapping")
    try:
        payload = (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("progress value is not finite canonical JSON") from error
    if not 0 < len(payload) <= _MAX_MARKER_BYTES:
        raise ValueError("progress marker exceeds its exact size bound")
    return payload


def verify_evaluator_progress(
    directory: str | os.PathLike[str],
    *,
    expected_child_contract_sha256: str,
    expected_parent_contract_sha256: str,
    expected_git_commit: str,
    expected_outer_fold: int,
    expected_score_release_sha256: str,
    expected_supervisor_process_pid: int | None = None,
    expected_evaluator_process_pid: int | None = None,
    expected_terminal_sha256: str | None = None,
    require_complete: bool,
    _allow_unsealed_complete: bool = False,
) -> EvaluatorProgressTrace:
    """Verify one self-consistent prefix or an optionally anchored sealed trace."""

    if type(require_complete) is not bool or type(_allow_unsealed_complete) is not bool:
        raise TypeError("progress verification flags must be exact bools")
    child_sha256 = _sha256(
        expected_child_contract_sha256,
        label="expected child contract SHA-256",
    )
    parent_sha256 = _sha256(
        expected_parent_contract_sha256,
        label="expected parent contract SHA-256",
    )
    commit = _git_commit(expected_git_commit)
    fold = _outer_fold(expected_outer_fold)
    release_sha256 = _sha256(
        expected_score_release_sha256,
        label="expected score-release SHA-256",
    )
    expected_supervisor_pid = (
        None
        if expected_supervisor_process_pid is None
        else _process_pid(
            expected_supervisor_process_pid,
            label="expected supervisor process PID",
        )
    )
    expected_evaluator_pid = (
        None
        if expected_evaluator_process_pid is None
        else _process_pid(
            expected_evaluator_process_pid,
            label="expected evaluator process PID",
        )
    )
    expected_terminal = (
        None
        if expected_terminal_sha256 is None
        else _sha256(expected_terminal_sha256, label="expected progress terminal SHA-256")
    )

    root = _absolute_directory(directory, required_mode=None)
    if root.name != str(fold):
        raise ValueError("evaluator progress directory does not bind its expected outer fold")
    before = os.lstat(root)
    mode = stat.S_IMODE(before.st_mode)
    if mode not in {0o700, 0o555}:
        raise ValueError("evaluator progress directory has an unsafe mode")
    entries = tuple(sorted(root.iterdir(), key=lambda value: value.name))
    if len(entries) > len(EVALUATOR_PROGRESS_PHASES):
        raise ValueError("evaluator progress has too many marker files")
    phases = EVALUATOR_PROGRESS_PHASES[: len(entries)]
    expected_names = tuple(_marker_name(index, phase) for index, phase in enumerate(phases))
    if tuple(entry.name for entry in entries) != expected_names:
        raise ValueError("evaluator progress inventory is not an exact phase prefix")
    full_census = len(entries) == len(EVALUATOR_PROGRESS_PHASES)
    if full_census:
        if mode == 0o555:
            complete = True
        elif mode == 0o700:
            complete = False
            if require_complete and not _allow_unsealed_complete:
                raise ValueError("complete evaluator progress directory is not sealed")
        else:  # pragma: no cover - the safe-mode census above is exhaustive
            raise ValueError("complete evaluator progress directory has an unsafe mode")
    elif mode != 0o700:
        raise ValueError("partial evaluator progress directory must remain private")
    else:
        complete = False
    if require_complete and not complete and not (full_census and _allow_unsealed_complete):
        raise ValueError("evaluator progress trace is incomplete")

    marker_sha256s: list[str] = []
    elapsed_monotonic_ns: list[int] = []
    process_cpu_ns: list[int] = []
    monotonic_origin_ns: int | None = None
    previous_sha256 = release_sha256
    previous_elapsed = -1
    previous_process_cpu = {"supervisor": -1, "evaluator": -1}
    observed_pid = {"supervisor": None, "evaluator": None}
    expected_pid = {
        "supervisor": expected_supervisor_pid,
        "evaluator": expected_evaluator_pid,
    }
    for sequence_number, (phase, path) in enumerate(zip(phases, entries, strict=True)):
        payload = _read_marker(path)
        document = _parse_canonical_object(payload)
        if tuple(sorted(document)) != _RECORD_FIELDS:
            raise ValueError("evaluator progress marker has a non-exact schema")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("evaluator progress schema version is invalid")
        if document["artifact"] != ARTIFACT:
            raise ValueError("evaluator progress artifact identity changed")
        actor = _PHASE_ACTORS[sequence_number]
        fixed: dict[str, object] = {
            "actor": actor,
            "child_contract_sha256": child_sha256,
            "parent_contract_sha256": parent_sha256,
            "git_commit": commit,
            "outer_fold": fold,
            "score_release_sha256": release_sha256,
            "sequence_number": sequence_number,
            "phase": phase,
            "previous_marker_sha256": previous_sha256,
        }
        if any(
            type(document[name]) is not type(value) or document[name] != value
            for name, value in fixed.items()
        ):
            raise ValueError("evaluator progress marker differs from its bound identity or order")
        marker_monotonic_origin_ns = _clock_value(
            document["monotonic_origin_ns"],
            label="progress marker monotonic origin",
        )
        if monotonic_origin_ns is None:
            monotonic_origin_ns = marker_monotonic_origin_ns
        elif marker_monotonic_origin_ns != monotonic_origin_ns:
            raise ValueError("evaluator progress monotonic origin changed")
        pid = _process_pid(document["process_pid"], label=f"{actor} process PID")
        if observed_pid[actor] is None:
            observed_pid[actor] = pid
        if pid != observed_pid[actor] or (
            expected_pid[actor] is not None and pid != expected_pid[actor]
        ):
            raise ValueError("evaluator progress process PID changed or is unexpected")
        elapsed = _clock_value(
            document["elapsed_monotonic_ns"],
            label="elapsed monotonic progress clock",
        )
        process_cpu = _clock_value(
            document["process_cpu_ns"],
            label="process CPU progress clock",
        )
        if elapsed < previous_elapsed or process_cpu < previous_process_cpu[actor]:
            raise ValueError("evaluator progress marker clock regressed")
        previous_elapsed = elapsed
        previous_process_cpu[actor] = process_cpu
        elapsed_monotonic_ns.append(elapsed)
        process_cpu_ns.append(process_cpu)
        digest = hashlib.sha256(payload).hexdigest()
        marker_sha256s.append(digest)
        previous_sha256 = digest

    after = os.lstat(root)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ValueError("evaluator progress directory changed during verification")
    terminal_sha256 = marker_sha256s[-1] if marker_sha256s else None
    for actor in ("supervisor", "evaluator"):
        if expected_pid[actor] is not None and observed_pid[actor] is None:
            raise ValueError(f"expected {actor} process PID actor is absent")
    if (
        observed_pid["supervisor"] is not None
        and observed_pid["supervisor"] == observed_pid["evaluator"]
    ):
        raise ValueError("supervisor and evaluator progress require distinct process PIDs")
    if expected_terminal is not None and terminal_sha256 != expected_terminal:
        raise ValueError("evaluator progress terminal SHA-256 is unexpected")
    return EvaluatorProgressTrace(
        phases=phases,
        marker_sha256s=tuple(marker_sha256s),
        monotonic_origin_ns=monotonic_origin_ns,
        elapsed_monotonic_ns=tuple(elapsed_monotonic_ns),
        process_cpu_ns=tuple(process_cpu_ns),
        supervisor_process_pid=observed_pid["supervisor"],
        evaluator_process_pid=observed_pid["evaluator"],
        terminal_sha256=terminal_sha256,
        complete=complete,
    )


def _marker_name(sequence_number: int, phase: str) -> str:
    if type(sequence_number) is not int or not 0 <= sequence_number < len(
        EVALUATOR_PROGRESS_PHASES
    ):
        raise ValueError("progress sequence number is outside the exact phase census")
    if type(phase) is not str or phase != EVALUATOR_PROGRESS_PHASES[sequence_number]:
        raise ValueError("progress phase does not match its exact sequence number")
    return f"{sequence_number:03d}-{phase}.json"


def _absolute_directory(
    value: str | os.PathLike[str],
    *,
    required_mode: int | None,
) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    _reject_symlink_chain(path)
    observed = os.lstat(path)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("evaluator progress path must be a real directory")
    if observed.st_uid != os.getuid():
        raise ValueError("evaluator progress directory has the wrong owner")
    if required_mode is not None and stat.S_IMODE(observed.st_mode) != required_mode:
        raise ValueError("evaluator progress directory has the wrong mode")
    return path


def _reject_symlink_chain(path: Path) -> None:
    candidates = (*tuple(reversed(path.parents)), path)
    for candidate in candidates:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError as error:
            raise ValueError("evaluator progress path does not exist") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("evaluator progress path traverses a symbolic link")


def _publish_no_replace(destination: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_MARKER_BYTES:
        raise ValueError("progress marker payload must be non-empty bounded bytes")
    directory = _absolute_directory(destination.parent, required_mode=0o700)
    staging_parent = _absolute_directory(directory.parent, required_mode=0o700)
    if os.path.lexists(destination):
        raise FileExistsError("evaluator progress publication is strictly no-overwrite")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{directory.name}-{destination.stem}-",
        dir=staging_parent,
    )
    temporary = Path(temporary_raw)
    published = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short evaluator progress marker write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        error_number = _renameat2_noreplace(temporary, destination)
        if error_number == 0:
            published = True
        elif error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError("evaluator progress publication is strictly no-overwrite")
        elif error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            os.link(temporary, destination, follow_symlinks=False)
            os.unlink(temporary)
            published = True
        else:
            raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
        _fsync_directory(directory)
        _fsync_directory(staging_parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
    if _read_marker(destination) != payload:
        raise RuntimeError("published evaluator progress marker changed bytes")


def _renameat2_noreplace(source: Path, destination: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        return errno.ENOSYS
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return 0
    return ctypes.get_errno()


def _read_marker(path: Path) -> bytes:
    _reject_symlink_chain(path)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_MARKER_BYTES
    ):
        raise ValueError("evaluator progress marker is not one immutable regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened_before = os.fstat(descriptor)
        while chunk := os.read(descriptor, _MAX_MARKER_BYTES + 1):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = os.lstat(path)
    fingerprints = {
        _stat_fingerprint(value) for value in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError("evaluator progress marker changed while being read")
    return payload


def _parse_canonical_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_MARKER_BYTES:
        raise ValueError("evaluator progress marker has invalid bounded bytes")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate evaluator progress JSON key")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        document = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("evaluator progress marker is not strict UTF-8 JSON") from error
    if type(document) is not dict:
        raise ValueError("evaluator progress marker must be one JSON object")
    typed = cast(dict[str, object], document)
    if canonical_progress_json_bytes(typed) != payload:
        raise ValueError("evaluator progress marker is not canonical JSON")
    return typed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_directory_read_only(path: Path) -> None:
    """Seal the opened directory inode rather than a raceable path lookup."""

    _reject_symlink_chain(path)
    named_before = os.lstat(path)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        if (
            _stat_fingerprint(named_before) != _stat_fingerprint(opened_before)
            or not stat.S_ISDIR(opened_before.st_mode)
            or stat.S_ISLNK(opened_before.st_mode)
            or stat.S_IMODE(opened_before.st_mode) != 0o700
            or opened_before.st_uid != os.getuid()
        ):
            raise ValueError("evaluator progress directory changed before sealing")
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        opened_after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            _stat_fingerprint(opened_after) != _stat_fingerprint(named_after)
            or stat.S_IMODE(opened_after.st_mode) != 0o555
        ):
            raise RuntimeError("evaluator progress directory changed while sealing")
    finally:
        os.close(descriptor)


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_commit(value: object) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        raise ValueError("Git commit must be a lowercase 40-character ID")
    return value


def _outer_fold(value: object) -> int:
    if type(value) is not int or value not in (0, 1, 2, 3):
        raise ValueError("outer fold must be an exact integer in 0..3")
    return value


def _process_pid(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 1:
        raise ValueError(f"{label} must be an exact positive non-init integer")
    return value


def _clock_value(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one self-consistent native-diffusion evaluator progress trace; "
            "optional PID and terminal arguments externally anchor it."
        )
    )
    parser.add_argument("--directory", required=True)
    parser.add_argument("--child-contract-sha256", required=True)
    parser.add_argument("--parent-contract-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--outer-fold", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--score-release-sha256", required=True)
    parser.add_argument("--supervisor-process-pid", type=int)
    parser.add_argument("--evaluator-process-pid", type=int)
    parser.add_argument("--terminal-sha256")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a trace for a launcher or an independent operational audit."""

    arguments = _parser().parse_args(argv)
    verify_evaluator_progress(
        arguments.directory,
        expected_child_contract_sha256=arguments.child_contract_sha256,
        expected_parent_contract_sha256=arguments.parent_contract_sha256,
        expected_git_commit=arguments.git_commit,
        expected_outer_fold=arguments.outer_fold,
        expected_score_release_sha256=arguments.score_release_sha256,
        expected_supervisor_process_pid=arguments.supervisor_process_pid,
        expected_evaluator_process_pid=arguments.evaluator_process_pid,
        expected_terminal_sha256=arguments.terminal_sha256,
        require_complete=arguments.require_complete,
    )
    return 0


__all__ = [
    "ARTIFACT",
    "CHECKPOINT_STEPS",
    "EVALUATOR_PROCESS_PROGRESS_PHASES",
    "EVALUATOR_PROGRESS_PHASES",
    "SUPERVISOR_PROGRESS_PHASES",
    "EvaluatorProgressTrace",
    "EvaluatorProgressWriter",
    "canonical_progress_json_bytes",
    "main",
    "verify_evaluator_progress",
]


if __name__ == "__main__":
    raise SystemExit(main())

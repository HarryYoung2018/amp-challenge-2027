"""Absolute allocation-clock watchdog for native-diffusion v1 pilot processes.

This file intentionally uses only the Python standard library.  The producer
launcher authenticates its committed bytes and active Slurm allocation before
invoking it, so an expired or malformed containment contract is rejected before
the UV environment, PyTorch, a GPU query, or a projection is opened.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Protocol

_JOB_RE = re.compile(r"[1-9][0-9]{0,14}")
_STEP_RE = re.compile(r"([1-9][0-9]{0,14})\.0")
_WATCHED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
_POLL_SECONDS = 0.25
_TERM_GRACE_SECONDS = 5.0
_KILL_WAIT_SECONDS = 5.0
_ROLE_RESERVE_SECONDS = {"controller": 60, "worker": 120}
_ROLE_EXECUTABLE = {
    "controller": "/usr/bin/bash",
    "worker": "/home/yonghan.yang/.local/bin/uv",
}
_CONTAINMENT_ENVIRONMENT = (
    "AMP_ALLOCATION_START_EPOCH_SECONDS",
    "AMP_ALLOCATION_END_EPOCH_SECONDS",
    "AMP_STEP_DEADLINE_EPOCH_SECONDS",
    "AMP_WORKER_DEADLINE_EPOCH_SECONDS",
    "AMP_CHILD_TERMINATION_RESERVE_SECONDS",
    "AMP_CONTROLLER_CLEANUP_RESERVE_SECONDS",
)


class WatchdogInterrupted(RuntimeError):
    """Raised when the allocation watchdog receives an external signal."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"allocation watchdog interrupted by signal {signum}")
        self.signum = signum


class _WaitableProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float) -> int: ...


@dataclass(frozen=True, slots=True)
class _DeadlinePair:
    epoch_seconds: int
    monotonic_seconds: float


class _PosixSpawnProcess:
    """Small bounded-wait adapter around an exact ``posix_spawn`` child PID."""

    def __init__(self, pid: int, command: tuple[str, ...]) -> None:
        if type(pid) is not int or pid <= 1:
            raise RuntimeError("posix_spawn returned an unsafe child PID")
        self.pid = pid
        self._command = command
        self._returncode: int | None = None

    def poll(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        if waited_pid == 0:
            return None
        if waited_pid != self.pid:
            raise RuntimeError("waitpid returned another process")
        self._returncode = os.waitstatus_to_exitcode(status)
        return self._returncode

    def wait(self, timeout: float) -> int:
        if type(timeout) is not float or timeout <= 0.0:
            raise ValueError("watchdog child wait must be a positive exact float")
        deadline = time.monotonic() + timeout
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(self._command, timeout)
            time.sleep(min(0.05, remaining))


def run_until_absolute_deadline(
    command: Sequence[str],
    *,
    role: str,
    deadline_epoch_seconds: int,
    cleanup_deadline_epoch_seconds: int,
    slurm_step_id: str,
    environment: Mapping[str, str] | None = None,
    _wall_clock: Callable[[], float] = time.time,
    _monotonic_clock: Callable[[], float] = time.monotonic,
    _spawn: Callable[[tuple[str, ...], Mapping[str, str]], _WaitableProcess] | None = None,
    _cancel_step: Callable[[str, str, float], None] | None = None,
    _wait_step: Callable[[str, float], bool] | None = None,
) -> int:
    """Run one exact pilot boundary, terminating it at the absolute cutoff."""

    values = tuple(command)
    environment_values = os.environ if environment is None else environment
    _validate_invocation(
        values,
        role=role,
        deadline_epoch_seconds=deadline_epoch_seconds,
        cleanup_deadline_epoch_seconds=cleanup_deadline_epoch_seconds,
        slurm_step_id=slurm_step_id,
        environment=environment_values,
    )
    if not callable(_wall_clock) or not callable(_monotonic_clock):
        raise TypeError("watchdog clock seams must be callable")
    spawn = _spawn_child if _spawn is None else _spawn
    cancel_step = _cancel_slurm_step if _cancel_step is None else _cancel_step
    wait_step = _wait_for_slurm_step_exit if _wait_step is None else _wait_step
    if not callable(spawn) or not callable(cancel_step) or not callable(wait_step):
        raise TypeError("watchdog process seams must be callable")
    monotonic_origin = _monotonic_clock()
    wall_origin = _wall_clock()
    if (
        type(monotonic_origin) is not float
        or monotonic_origin <= 0.0
        or type(wall_origin) is not float
        or not 1_000_000_000.0 <= wall_origin <= 9_999_999_999.0
    ):
        raise RuntimeError("watchdog clocks returned unsafe values")
    deadline = _DeadlinePair(
        epoch_seconds=deadline_epoch_seconds,
        monotonic_seconds=monotonic_origin + deadline_epoch_seconds - wall_origin,
    )
    cleanup_deadline = _DeadlinePair(
        epoch_seconds=cleanup_deadline_epoch_seconds,
        monotonic_seconds=monotonic_origin + cleanup_deadline_epoch_seconds - wall_origin,
    )
    if (
        _remaining(
            deadline,
            wall_clock=_wall_clock,
            monotonic_clock=_monotonic_clock,
        )
        <= 0.0
    ):
        raise TimeoutError("watchdog deadline expired before child launch")

    process: _WaitableProcess | None = None
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _WATCHED_SIGNALS)
        try:
            process = spawn(values, environment_values)
        finally:
            # ``_spawn_child`` clears the child's mask atomically in
            # posix_spawn; pending parent signals are released only after PID
            # ownership is established here.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        while True:
            remaining = _remaining(
                deadline,
                wall_clock=_wall_clock,
                monotonic_clock=_monotonic_clock,
            )
            if remaining <= 0.0:
                _terminate(
                    process,
                    role=role,
                    slurm_step_id=slurm_step_id,
                    cleanup_deadline=cleanup_deadline,
                    wall_clock=_wall_clock,
                    monotonic_clock=_monotonic_clock,
                    cancel_step=cancel_step,
                    wait_step=wait_step,
                )
                process = None
                return 124
            try:
                returncode = process.wait(timeout=min(_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue
            completed_process = process
            if role == "controller":
                _terminate(
                    completed_process,
                    role=role,
                    slurm_step_id=slurm_step_id,
                    cleanup_deadline=cleanup_deadline,
                    wall_clock=_wall_clock,
                    monotonic_clock=_monotonic_clock,
                    cancel_step=cancel_step,
                    wait_step=wait_step,
                )
            process = None
            return returncode
    except BaseException:
        if process is not None:
            _terminate(
                process,
                role=role,
                slurm_step_id=slurm_step_id,
                cleanup_deadline=cleanup_deadline,
                wall_clock=_wall_clock,
                monotonic_clock=_monotonic_clock,
                cancel_step=cancel_step,
                wait_step=wait_step,
            )
        raise


def _validate_invocation(
    command: tuple[str, ...],
    *,
    role: str,
    deadline_epoch_seconds: int,
    cleanup_deadline_epoch_seconds: int,
    slurm_step_id: str,
    environment: Mapping[str, str],
) -> None:
    if role not in _ROLE_RESERVE_SECONDS:
        raise ValueError("watchdog role must be controller or worker")
    if (
        not command
        or any(type(value) is not str or not value for value in command)
        or command[0] != _ROLE_EXECUTABLE[role]
    ):
        raise ValueError("watchdog command differs from its exact role executable")
    for label, value in (
        ("deadline", deadline_epoch_seconds),
        ("cleanup deadline", cleanup_deadline_epoch_seconds),
    ):
        if type(value) is not int or not 1_000_000_000 <= value <= 9_999_999_999:
            raise ValueError(f"watchdog {label} must be an exact ten-digit epoch second")
    if cleanup_deadline_epoch_seconds - deadline_epoch_seconds != _ROLE_RESERVE_SECONDS[role]:
        raise ValueError("watchdog deadlines do not preserve the exact role reserve")
    step_match = _STEP_RE.fullmatch(slurm_step_id) if type(slurm_step_id) is str else None
    if step_match is None:
        raise ValueError("watchdog requires exact Slurm step zero identity")
    if not isinstance(environment, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in environment.items()
    ):
        raise TypeError("watchdog environment must be a string mapping")
    job_id = environment.get("SLURM_JOB_ID")
    if type(job_id) is not str or _JOB_RE.fullmatch(job_id) is None:
        raise ValueError("watchdog requires one positive Slurm job ID")
    if step_match.group(1) != job_id:
        raise ValueError("watchdog step identity differs from its allocation")
    if role == "worker" and environment.get("SLURM_STEP_ID") != "0":
        raise ValueError("worker watchdog must execute inside exact Slurm step zero")
    parsed = {
        name: _canonical_epoch_or_reserve(environment, name) for name in _CONTAINMENT_ENVIRONMENT
    }
    start = parsed["AMP_ALLOCATION_START_EPOCH_SECONDS"]
    end = parsed["AMP_ALLOCATION_END_EPOCH_SECONDS"]
    step_deadline = parsed["AMP_STEP_DEADLINE_EPOCH_SECONDS"]
    worker_deadline = parsed["AMP_WORKER_DEADLINE_EPOCH_SECONDS"]
    if (
        end - start != 15 * 60
        or end - step_deadline != 60
        or step_deadline - worker_deadline != 120
        or parsed["AMP_CHILD_TERMINATION_RESERVE_SECONDS"] != 120
        or parsed["AMP_CONTROLLER_CLEANUP_RESERVE_SECONDS"] != 60
    ):
        raise ValueError("watchdog environment violates the exact containment hierarchy")
    expected_deadline = step_deadline if role == "controller" else worker_deadline
    expected_cleanup = end if role == "controller" else step_deadline
    if (
        deadline_epoch_seconds != expected_deadline
        or cleanup_deadline_epoch_seconds != expected_cleanup
    ):
        raise ValueError("watchdog CLI deadlines differ from the authenticated environment")


def _canonical_epoch_or_reserve(environment: Mapping[str, str], name: str) -> int:
    raw = environment.get(name)
    if (
        type(raw) is not str
        or not raw.isascii()
        or not raw.isdecimal()
        or (len(raw) > 1 and raw.startswith("0"))
    ):
        raise ValueError(f"watchdog environment {name} is not a canonical decimal")
    return int(raw)


def _canonical_cli_epoch(raw: str, *, label: str) -> int:
    if type(raw) is not str or len(raw) != 10 or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"watchdog {label} must be one canonical ten-digit epoch second")
    return int(raw)


def _spawn_child(command: tuple[str, ...], environment: Mapping[str, str]) -> _WaitableProcess:
    file_actions = ((os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0o666),)
    pid = os.posix_spawn(
        command[0],
        command,
        dict(environment),
        file_actions=file_actions,
        # POSIX process-group creation is sufficient for the bounded killpg
        # cleanup below and is available on the target Slurm nodes.  The
        # stronger setsid extension is not implemented by their libc.
        setpgroup=0,
        setsigdef=_WATCHED_SIGNALS,
        setsigmask=(),
    )
    return _PosixSpawnProcess(pid, command)


def _remaining(
    deadline: _DeadlinePair,
    *,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> float:
    wall_now = wall_clock()
    monotonic_now = monotonic_clock()
    if (
        type(wall_now) is not float
        or not 1_000_000_000.0 <= wall_now <= 9_999_999_999.0
        or type(monotonic_now) is not float
        or monotonic_now <= 0.0
    ):
        raise RuntimeError("watchdog clock changed to an unsafe value")
    return min(deadline.epoch_seconds - wall_now, deadline.monotonic_seconds - monotonic_now)


def _terminate(
    process: _WaitableProcess,
    *,
    role: str,
    slurm_step_id: str,
    cleanup_deadline: _DeadlinePair,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
    cancel_step: Callable[[str, str, float], None],
    wait_step: Callable[[str, float], bool],
) -> None:
    # Once cleanup owns the process/step identities, repeated controller or
    # operator signals must not re-enter and abort the exact-step proof.  They
    # remain pending until every bounded cleanup action has finished.  Cleanup
    # subprocesses inherit this mask intentionally and are themselves bounded
    # by the authenticated allocation-end clock and killed on timeout.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _WATCHED_SIGNALS)
    cleanup_error: BaseException | None = None
    try:
        _terminate_with_control_signals_blocked(
            process,
            role=role,
            slurm_step_id=slurm_step_id,
            cleanup_deadline=cleanup_deadline,
            wall_clock=wall_clock,
            monotonic_clock=monotonic_clock,
            cancel_step=cancel_step,
            wait_step=wait_step,
        )
    except BaseException as error:
        cleanup_error = error
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except BaseException as signal_error:
        if cleanup_error is not None:
            raise cleanup_error from signal_error
        raise
    if cleanup_error is not None:
        raise cleanup_error


def _terminate_with_control_signals_blocked(
    process: _WaitableProcess,
    *,
    role: str,
    slurm_step_id: str,
    cleanup_deadline: _DeadlinePair,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
    cancel_step: Callable[[str, str, float], None],
    wait_step: Callable[[str, float], bool],
) -> None:
    process_running = process.poll() is None
    if not process_running and role != "controller":
        return
    cleanup_error: BaseException | None = None
    if process_running:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    if role == "controller":
        try:
            cancel_step(
                slurm_step_id,
                "TERM",
                _bounded_cleanup_wait(
                    cleanup_deadline,
                    cap_seconds=5.0,
                    wall_clock=wall_clock,
                    monotonic_clock=monotonic_clock,
                ),
            )
        except BaseException as error:  # retain cleanup ownership before reporting
            cleanup_error = error
    if process_running:
        with suppress(subprocess.TimeoutExpired):
            process.wait(
                timeout=_bounded_cleanup_wait(
                    cleanup_deadline,
                    cap_seconds=_TERM_GRACE_SECONDS,
                    wall_clock=wall_clock,
                    monotonic_clock=monotonic_clock,
                )
            )

    remote_step_gone = True
    if role == "controller":
        try:
            remote_step_gone = wait_step(
                slurm_step_id,
                _bounded_cleanup_wait(
                    cleanup_deadline,
                    cap_seconds=_TERM_GRACE_SECONDS,
                    wall_clock=wall_clock,
                    monotonic_clock=monotonic_clock,
                ),
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
            remote_step_gone = False

    process_running = process.poll() is None
    if process_running:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if role == "controller" and not remote_step_gone:
        try:
            cancel_step(
                slurm_step_id,
                "KILL",
                _bounded_cleanup_wait(
                    cleanup_deadline,
                    cap_seconds=5.0,
                    wall_clock=wall_clock,
                    monotonic_clock=monotonic_clock,
                ),
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
    if process_running:
        process.wait(
            timeout=_bounded_cleanup_wait(
                cleanup_deadline,
                cap_seconds=_KILL_WAIT_SECONDS,
                wall_clock=wall_clock,
                monotonic_clock=monotonic_clock,
            )
        )
    if role == "controller" and not remote_step_gone:
        try:
            remote_step_gone = wait_step(
                slurm_step_id,
                _bounded_cleanup_wait(
                    cleanup_deadline,
                    cap_seconds=_KILL_WAIT_SECONDS,
                    wall_clock=wall_clock,
                    monotonic_clock=monotonic_clock,
                ),
            )
        except BaseException as error:
            cleanup_error = cleanup_error or error
            remote_step_gone = False
    if role == "controller" and not remote_step_gone:
        raise RuntimeError(
            "watchdog could not prove exact Slurm step disappearance"
        ) from cleanup_error


def _bounded_cleanup_wait(
    cleanup_deadline: _DeadlinePair,
    *,
    cap_seconds: float,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> float:
    remaining = _remaining(
        cleanup_deadline,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
    )
    if remaining <= 0.0:
        raise TimeoutError("watchdog exhausted its cleanup reserve")
    return min(cap_seconds, remaining)


def _cancel_slurm_step(slurm_step_id: str, signal_name: str, timeout_seconds: float) -> None:
    if (
        _STEP_RE.fullmatch(slurm_step_id) is None
        or signal_name not in {"TERM", "KILL"}
        or type(timeout_seconds) is not float
        or not 0.0 < timeout_seconds <= 5.0
    ):
        raise ValueError("unsafe Slurm step cancellation request")
    completed = subprocess.run(
        ("/usr/bin/scancel", "--quiet", f"--signal={signal_name}", slurm_step_id),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError("scancel rejected the exact watched Slurm step")


def _wait_for_slurm_step_exit(slurm_step_id: str, timeout_seconds: float) -> bool:
    if (
        _STEP_RE.fullmatch(slurm_step_id) is None
        or type(timeout_seconds) is not float
        or not 0.0 < timeout_seconds <= 5.0
    ):
        raise ValueError("unsafe Slurm step-disappearance request")
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        if not _slurm_step_is_active(slurm_step_id, min(1.0, remaining)):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(_POLL_SECONDS, remaining))


def _slurm_step_is_active(slurm_step_id: str, timeout_seconds: float) -> bool:
    if (
        _STEP_RE.fullmatch(slurm_step_id) is None
        or type(timeout_seconds) is not float
        or not 0.0 < timeout_seconds <= 1.0
    ):
        raise ValueError("unsafe active-step query")
    completed = subprocess.run(
        (
            "/usr/bin/squeue",
            "--local",
            f"--steps={slurm_step_id}",
            "--noheader",
            "--format=%i",
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("watchdog could not query the exact Slurm step")
    step_ids = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if step_ids not in ((), (slurm_step_id,)):
        raise RuntimeError("active-step query returned an unexpected identity")
    return bool(step_ids)


def _watchdog_signal(signum: int, _frame: object) -> None:
    raise WatchdogInterrupted(signum)


@contextmanager
def _signal_boundary():
    previous = {value: signal.getsignal(value) for value in _WATCHED_SIGNALS}
    interrupted = False

    def interrupt_once(signum: int, frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        _watchdog_signal(signum, frame)

    try:
        for value in _WATCHED_SIGNALS:
            signal.signal(value, interrupt_once)
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(_ROLE_RESERVE_SECONDS), required=True)
    parser.add_argument("--deadline-epoch-seconds", required=True)
    parser.add_argument("--cleanup-deadline-epoch-seconds", required=True)
    parser.add_argument("--slurm-step-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        with _signal_boundary():
            return run_until_absolute_deadline(
                command,
                role=args.role,
                deadline_epoch_seconds=_canonical_cli_epoch(
                    args.deadline_epoch_seconds,
                    label="deadline",
                ),
                cleanup_deadline_epoch_seconds=_canonical_cli_epoch(
                    args.cleanup_deadline_epoch_seconds,
                    label="cleanup deadline",
                ),
                slurm_step_id=args.slurm_step_id,
            )
    except WatchdogInterrupted as error:
        print(str(error), file=sys.stderr)
        return 128 + error.signum
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Commit and consume small producer records with hard-link no-replace semantics."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_RECORD_BYTES = 65_536
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class RecordError(RuntimeError):
    """Fail-closed record publication or consumption error."""


class RecordNotCommitted(RecordError):
    """A missing or prepared record that a bounded waiter may retry."""


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RecordError("record directory is not a safe absolute path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RecordError("short record write")
        remaining = remaining[written:]


def _read_held(
    descriptor: int,
    maximum_bytes: int,
    *,
    required_nlink: int,
) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != required_nlink
        or before.st_size > maximum_bytes
    ):
        raise RecordError("record mode, link count, type, or size is invalid")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)

    def fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
            value.st_nlink,
        )

    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise RecordError("record changed while read")
    return payload, fingerprint(after)


def publish_record(
    directory: Path,
    name: str,
    payload: bytes,
    *,
    phase_hook: Callable[[str], None] | None = None,
) -> str:
    """Publish a record; successful temporary unlink is the sole commit point.

    Before that unlink, a linked destination has link count two and consumers
    reject it. Failures are retained as prepared records; this function never
    removes an ambiguous destination or a competing owner's entry.
    """

    if not _NAME.fullmatch(name):
        raise RecordError("record name is invalid")
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_RECORD_BYTES:
        raise RecordError("record payload is empty or exceeds its byte cap")
    if not payload.endswith(b"\n"):
        raise RecordError("record payload must end with LF")
    directory_fd = _open_absolute_directory(directory)
    temporary = f".{name}.prepared-{os.getpid()}-{secrets.token_hex(12)}"
    descriptor = -1
    linked_descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        prepared_payload, prepared = _read_held(descriptor, len(payload), required_nlink=1)
        if prepared_payload != payload:
            raise RecordError("prepared record readback mismatch")
        if phase_hook is not None:
            phase_hook("prepared")
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise RecordError("record destination already exists") from error
        if phase_hook is not None:
            phase_hook("linked")
        linked_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        linked_payload, linked = _read_held(
            linked_descriptor,
            len(payload),
            required_nlink=2,
        )
        rebound_payload, rebound = _read_held(descriptor, len(payload), required_nlink=2)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            linked_payload != payload
            or rebound_payload != payload
            or (linked[0], linked[1]) != (prepared[0], prepared[1])
            or (rebound[0], rebound[1]) != (prepared[0], prepared[1])
            or (entry.st_dev, entry.st_ino) != (prepared[0], prepared[1])
        ):
            raise RecordError("linked record inode or content mismatch")
        os.fsync(directory_fd)
        digest = hashlib.sha256(payload).hexdigest()
        if phase_hook is not None:
            phase_hook("verified_before_commit")
        final_linked, final_linked_identity = _read_held(
            linked_descriptor,
            len(payload),
            required_nlink=2,
        )
        final_prepared, final_prepared_identity = _read_held(
            descriptor,
            len(payload),
            required_nlink=2,
        )
        final_entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            final_linked != payload
            or final_prepared != payload
            or (final_linked_identity[0], final_linked_identity[1]) != (prepared[0], prepared[1])
            or (final_prepared_identity[0], final_prepared_identity[1])
            != (prepared[0], prepared[1])
            or (final_entry.st_dev, final_entry.st_ino, final_entry.st_nlink)
            != (prepared[0], prepared[1], 2)
        ):
            raise RecordError("record destination changed at commit")
        # Formal commit point. Consumers now see exactly one link. No fallible
        # publication operation follows this unlink.
        os.unlink(temporary, dir_fd=directory_fd)
        return digest
    finally:
        if linked_descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(linked_descriptor)
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(directory_fd)


def read_record(directory: Path, name: str, maximum_bytes: int = _MAX_RECORD_BYTES) -> bytes:
    if not _NAME.fullmatch(name):
        raise RecordError("record name is invalid")
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or not 0 < maximum_bytes <= _MAX_RECORD_BYTES
    ):
        raise RecordError("record byte cap is invalid")
    directory_fd = _open_absolute_directory(directory)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except FileNotFoundError as error:
            raise RecordNotCommitted("record is absent") from error
        info = os.fstat(descriptor)
        if stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink > 1:
            raise RecordNotCommitted("record is prepared but not committed")
        payload, fingerprint = _read_held(descriptor, maximum_bytes, required_nlink=1)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink) != (
            fingerprint[0],
            fingerprint[1],
            fingerprint[5],
            fingerprint[6],
        ):
            raise RecordError("record entry was substituted")
        rebound, rebound_fingerprint = _read_held(
            descriptor,
            maximum_bytes,
            required_nlink=1,
        )
        if rebound != payload or rebound_fingerprint != fingerprint:
            raise RecordError("record changed during consumption")
        if not payload.endswith(b"\n"):
            raise RecordError("record does not end with LF")
        return payload
    finally:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(directory_fd)


def wait_for_record(directory: Path, name: str, timeout_seconds: int) -> bytes:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= 600
    ):
        raise RecordError("record wait timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return read_record(directory, name)
        except RecordNotCommitted:
            if time.monotonic() >= deadline:
                raise RecordError("record did not commit before the bounded deadline") from None
            time.sleep(0.2)


def parse_record(payload: bytes) -> dict[str, str]:
    if not payload.endswith(b"\n") or payload.count(b"\n") > 64:
        raise RecordError("record framing exceeds its cap")
    values: dict[str, str] = {}
    for encoded in payload[:-1].split(b"\n"):
        try:
            line = encoded.decode("ascii")
        except UnicodeDecodeError as error:
            raise RecordError("record is not ASCII") from error
        if line.count("=") != 1:
            raise RecordError("record field is malformed")
        key, value = line.split("=", 1)
        if (
            not _FIELD.fullmatch(key)
            or key in values
            or not value
            or len(value) > 4_096
            or any(ord(character) < 32 or ord(character) >= 127 for character in value)
        ):
            raise RecordError("record field is unsafe")
        values[key] = value
    if not values:
        raise RecordError("record has no fields")
    return values


def seal_record_directory(directory: Path, expected_names: set[str]) -> dict[str, str]:
    if (
        not expected_names
        or len(expected_names) > 32
        or any(not _NAME.fullmatch(name) for name in expected_names)
    ):
        raise RecordError("record inventory declaration is invalid")
    directory_fd = _open_absolute_directory(directory)
    descriptors: dict[str, int] = {}
    try:
        observed: set[str] = set()
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                if len(observed) >= 32 or entry.name in observed:
                    raise RecordError("record directory inventory exceeds its cap")
                observed.add(entry.name)
        if observed != expected_names:
            raise RecordError("record directory inventory is incomplete or ambiguous")
        digests: dict[str, str] = {}
        for name in sorted(expected_names):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            descriptors[name] = descriptor
            payload, fingerprint = _read_held(
                descriptor,
                _MAX_RECORD_BYTES,
                required_nlink=1,
            )
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink) != (
                fingerprint[0],
                fingerprint[1],
                fingerprint[5],
                fingerprint[6],
            ):
                raise RecordError("record entry changed before sealing")
            parse_record(payload)
            digests[name] = hashlib.sha256(payload).hexdigest()
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
        for name, descriptor in descriptors.items():
            payload, fingerprint = _read_held(
                descriptor,
                _MAX_RECORD_BYTES,
                required_nlink=1,
            )
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink) != (
                fingerprint[0],
                fingerprint[1],
                fingerprint[5],
                fingerprint[6],
            ) or hashlib.sha256(payload).hexdigest() != digests[name]:
                raise RecordError("record changed while directory was sealed")
        return digests
    finally:
        for descriptor in descriptors.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--directory", type=Path, required=True)
    publish.add_argument("--name", required=True)
    wait = subparsers.add_parser("wait")
    wait.add_argument("--directory", type=Path, required=True)
    wait.add_argument("--name", required=True)
    wait.add_argument("--timeout-seconds", type=int, default=300)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--directory", type=Path, required=True)
    inspect.add_argument("--name", required=True)
    inspect.add_argument("--timeout-seconds", type=int, default=300)
    inspect.add_argument("--field")
    seal = subparsers.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--expected-name", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "publish":
        payload = sys.stdin.buffer.read(_MAX_RECORD_BYTES + 1)
        digest = publish_record(args.directory, args.name, payload)
        print(json.dumps({"sha256": digest, "status": "committed"}, sort_keys=True))
        return 0
    if args.command == "wait":
        payload = wait_for_record(args.directory, args.name, args.timeout_seconds)
        sys.stdout.buffer.write(payload)
        return 0
    if args.command == "inspect":
        payload = wait_for_record(args.directory, args.name, args.timeout_seconds)
        values = parse_record(payload)
        if args.field is not None:
            if not _FIELD.fullmatch(args.field) or args.field not in values:
                raise RecordError("requested record field is absent or invalid")
            print(values[args.field])
        else:
            print(
                json.dumps(
                    {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "values": values,
                    },
                    sort_keys=True,
                )
            )
        return 0
    digests = seal_record_directory(args.directory, set(args.expected_name))
    print(json.dumps({"records": digests, "status": "sealed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

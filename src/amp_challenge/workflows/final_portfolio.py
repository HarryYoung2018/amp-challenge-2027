"""Build a reward-protected final portfolio from a ledger and posterior draws."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
import tempfile
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import numpy as np

from amp_challenge.selection import (
    PortfolioCandidates,
    RewardProtectedConfig,
    RewardProtectedPortfolioSelector,
    RewardProtectedResult,
)
from amp_challenge.sequences import canonicalize_sequence
from amp_challenge.workflows.select import CandidateLedger, read_candidate_ledger

_NPZ_KEYS = {
    "sequences",
    "objectives",
    "reward_samples",
    "physicochemical_features",
    "quality_probability",
    "out_of_distribution",
    "calibrated_lcb",
}
_AUDIT_COLUMNS = (
    "portfolio_rank",
    "portfolio_candidate_index",
    "portfolio_ledger_row",
    "portfolio_sequence_id",
    "portfolio_reason",
    "portfolio_expected_reward",
    "portfolio_baseline_member",
)


@dataclass(frozen=True, slots=True)
class FinalPortfolioRunConfig:
    """Named reward categories and their locked selector policy."""

    objectives: tuple[str, ...]
    portfolio: RewardProtectedConfig


@dataclass(frozen=True, slots=True)
class FinalPortfolioExecution:
    """Completed final-portfolio run and its deterministic artifacts."""

    config: FinalPortfolioRunConfig
    ledger: CandidateLedger
    result: RewardProtectedResult
    fasta_path: Path
    audit_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    """Filesystem identity used to detect an input changing around a hash."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _InputFingerprint:
    """Stable content hash plus the identity of the bytes that were hashed."""

    sha256: str
    size_bytes: int
    identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _PinnedInput:
    """One continuously open input whose descriptor owns the parsed bytes."""

    path: Path
    handle: BinaryIO
    fingerprint: _InputFingerprint

    @property
    def parse_path(self) -> Path:
        """Return a process-local path that reopens only the pinned inode."""

        return Path(f"/proc/self/fd/{self.handle.fileno()}")


def load_final_portfolio_config(path: str | Path) -> FinalPortfolioRunConfig:
    """Load the versioned, strict final-portfolio TOML schema."""

    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    unexpected = set(document) - {"schema_version", "objectives", "portfolio"}
    if unexpected:
        raise ValueError(f"unexpected final-portfolio config key(s): {sorted(unexpected)}")
    if document.get("schema_version") != 1:
        raise ValueError("final-portfolio schema_version must be 1")
    objectives_raw = document.get("objectives")
    if (
        not isinstance(objectives_raw, list)
        or not objectives_raw
        or not all(isinstance(name, str) and name.strip() for name in objectives_raw)
    ):
        raise ValueError("objectives must be a non-empty string array")
    objectives = tuple(name.strip() for name in objectives_raw)
    if len(objectives) != len(set(objectives)):
        raise ValueError("objectives must be unique")

    portfolio_raw = document.get("portfolio")
    if not isinstance(portfolio_raw, dict):
        raise ValueError("[portfolio] must be a TOML table")
    allowed = {field.name for field in fields(RewardProtectedConfig)}
    unexpected_portfolio = set(portfolio_raw) - allowed
    if unexpected_portfolio:
        raise ValueError(f"unexpected [portfolio] config key(s): {sorted(unexpected_portfolio)}")
    values: dict[str, Any] = dict(portfolio_raw)
    for name in ("objective_weights", "category_tolerances"):
        value = values.get(name)
        if isinstance(value, list):
            values[name] = tuple(value)
    try:
        config = RewardProtectedConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid [portfolio] configuration: {error}") from error
    if config.objective_weights is not None and len(config.objective_weights) != len(objectives):
        raise ValueError("objective_weights must have one entry per configured objective")
    if not isinstance(config.category_tolerances, int | float) and len(
        config.category_tolerances
    ) != len(objectives):
        raise ValueError("category_tolerances must have one entry per configured objective")
    return FinalPortfolioRunConfig(objectives=objectives, portfolio=config)


def _string_values(values: np.ndarray, *, name: str) -> tuple[str, ...]:
    if values.ndim != 1:
        raise ValueError(f"posterior {name!r} must be a one-dimensional string array")
    result: list[str] = []
    for raw in values:
        if isinstance(raw, bytes):
            value = raw.decode("utf-8")
        elif isinstance(raw, str | np.str_):
            value = str(raw)
        else:
            raise ValueError(f"posterior {name!r} must contain only strings")
        if not value:
            raise ValueError(f"posterior {name!r} cannot contain empty values")
        result.append(value)
    return tuple(result)


def _reorder_optional(
    archive: Mapping[str, np.ndarray],
    name: str,
    order: np.ndarray,
) -> np.ndarray | None:
    if name not in archive:
        return None
    values = np.asarray(archive[name])
    if values.ndim == 0 or values.shape[0] != len(order):
        raise ValueError(f"posterior {name!r} must have candidate as its first dimension")
    if np.array_equal(order, np.arange(len(order), dtype=np.int64)):
        return values
    return values[order]


def load_portfolio_candidates(
    posterior_path: str | Path,
    *,
    ledger: CandidateLedger,
    objectives: Sequence[str],
) -> PortfolioCandidates:
    """Load and sequence-align a non-pickle NPZ posterior artifact."""

    with np.load(posterior_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        unexpected = keys - _NPZ_KEYS
        if unexpected:
            raise ValueError(f"unexpected posterior array(s): {sorted(unexpected)}")
        missing = {
            "sequences",
            "objectives",
            "reward_samples",
            "quality_probability",
            "out_of_distribution",
        } - keys
        if missing:
            raise ValueError(f"posterior artifact is missing array(s): {sorted(missing)}")
        posterior_sequences = tuple(
            canonicalize_sequence(sequence)
            for sequence in _string_values(np.asarray(archive["sequences"]), name="sequences")
        )
        if len(set(posterior_sequences)) != len(posterior_sequences):
            raise ValueError("posterior sequences are not unique after canonicalization")
        posterior_objectives = _string_values(
            np.asarray(archive["objectives"]),
            name="objectives",
        )
        if posterior_objectives != tuple(objectives):
            raise ValueError("posterior objectives do not exactly match configured objective order")
        ledger_sequences = tuple(ledger.candidates.sequences)
        if set(posterior_sequences) != set(ledger_sequences):
            raise ValueError("posterior and ledger must contain exactly the same sequences")
        source_index = {sequence: index for index, sequence in enumerate(posterior_sequences)}
        order = np.asarray(
            [source_index[sequence] for sequence in ledger_sequences], dtype=np.int64
        )

        rewards = np.asarray(archive["reward_samples"])
        if rewards.ndim != 3 or rewards.shape[0] != len(order):
            raise ValueError("posterior reward_samples must have shape (candidate, draw, category)")
        if rewards.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("posterior reward_samples must use float32 or float64 storage")
        if not np.array_equal(order, np.arange(len(order), dtype=np.int64)):
            rewards = rewards[order]
        if rewards.shape[2] != len(objectives):
            raise ValueError("posterior reward_samples category dimension is incorrect")
        posterior_means = np.mean(rewards, axis=1, dtype=np.float64)
        if not np.allclose(
            posterior_means,
            ledger.candidates.objective_mean,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("posterior reward-sample means do not match ledger objective means")

        physicochemical = _reorder_optional(
            archive,
            "physicochemical_features",
            order,
        )
        quality = _reorder_optional(archive, "quality_probability", order)
        ood = _reorder_optional(archive, "out_of_distribution", order)
        lcb = _reorder_optional(archive, "calibrated_lcb", order)

    return PortfolioCandidates(
        sequences=ledger_sequences,
        reward_samples=rewards,
        novelty=ledger.candidates.novelty,
        embeddings=ledger.candidates.embeddings,
        physicochemical_features=physicochemical,
        cluster_ids=ledger.candidates.cluster_ids,
        quality_probability=quality,
        out_of_distribution=ood,
        calibrated_lcb=lcb,
        eligible=ledger.candidates.eligible,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(stat_result: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        mode=stat_result.st_mode,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
    )


def _descriptor_fingerprint(handle: BinaryIO) -> _InputFingerprint:
    """Hash a pinned regular-file descriptor without reopening its pathname."""

    descriptor = handle.fileno()
    before = _file_identity(os.fstat(descriptor))
    digest = hashlib.sha256()
    bytes_read = 0
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        bytes_read += len(chunk)
    after = _file_identity(os.fstat(descriptor))
    handle.seek(0)
    if before != after or bytes_read != before.size_bytes:
        raise RuntimeError("pinned input changed while fingerprinting")
    return _InputFingerprint(
        sha256=digest.hexdigest(),
        size_bytes=bytes_read,
        identity=before,
    )


def _absolute_without_symlinks(path: str | Path, *, label: str) -> Path:
    """Resolve dot components while rejecting every existing symbolic link."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} does not exist: {absolute}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} path contains a symbolic link: {current}")
    return absolute


def _path_identity(path: Path, *, label: str) -> _FileIdentity:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} path binding disappeared: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _file_identity(metadata)


@contextmanager
def _pinned_input(path: str | Path, *, label: str) -> Iterator[_PinnedInput]:
    """Continuously pin, fingerprint, and expose one no-follow input file."""

    absolute = _absolute_without_symlinks(path, label=label)
    before = _path_identity(absolute, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = _file_identity(os.fstat(descriptor))
        after_open = _path_identity(absolute, label=label)
        if before != opened or opened != after_open:
            raise RuntimeError(f"{label} path changed while it was pinned: {absolute}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            fingerprint = _descriptor_fingerprint(handle)
            if fingerprint.identity != _path_identity(absolute, label=label):
                raise RuntimeError(f"{label} path changed while it was fingerprinted: {absolute}")
            yield _PinnedInput(path=absolute, handle=handle, fingerprint=fingerprint)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _revalidate_pinned_input(value: _PinnedInput, *, label: str) -> None:
    """Rehash the parsed descriptor and reauthenticate its external name."""

    observed = _descriptor_fingerprint(value.handle)
    path_identity = _path_identity(value.path, label=label)
    if observed != value.fingerprint or path_identity != value.fingerprint.identity:
        raise RuntimeError(f"{label} input changed after pre-parse validation: {value.path}")


def _input_fingerprint_summary(fingerprint: _InputFingerprint) -> dict[str, object]:
    """Return path-independent evidence for the exact pre-parse bytes."""

    return {
        "sha256": fingerprint.sha256,
        "size_bytes": fingerprint.size_bytes,
    }


def _metrics_dict(metrics: object) -> dict[str, object]:
    return asdict(metrics)  # type: ignore[arg-type]


def _lexists(path: Path) -> bool:
    """Return true for files, directories, and broken symlinks."""

    return os.path.lexists(path)


def _new_staged_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fasta(
    path: Path,
    *,
    ledger: CandidateLedger,
    result: RewardProtectedResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(result.indices))))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, (index, reason, score) in enumerate(
            zip(result.indices, result.reasons, result.expected_scores, strict=True),
            start=1,
        ):
            metadata = ledger.metadata[index]
            handle.write(
                f">rank={rank:0{width}d} sequence_id={metadata.sequence_id} "
                f"reason={quote(reason, safe=':._-')} expected_reward={score:.12g}\n"
                f"{ledger.candidates.sequences[index]}\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_audit(
    path: Path,
    *,
    ledger: CandidateLedger,
    result: RewardProtectedResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = set(result.baseline_indices)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*_AUDIT_COLUMNS, *ledger.fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, (index, reason, score) in enumerate(
            zip(result.indices, result.reasons, result.expected_scores, strict=True),
            start=1,
        ):
            metadata = ledger.metadata[index]
            input_values = metadata.as_dict()
            input_values["sequence"] = ledger.candidates.sequences[index]
            writer.writerow(
                {
                    "portfolio_rank": rank,
                    "portfolio_candidate_index": index,
                    "portfolio_ledger_row": metadata.ledger_row,
                    "portfolio_sequence_id": metadata.sequence_id,
                    "portfolio_reason": reason,
                    "portfolio_expected_reward": format(score, ".17g"),
                    "portfolio_baseline_member": str(index in baseline).lower(),
                    **input_values,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_staged_bundle(staged: Sequence[tuple[Path, Path]]) -> None:
    """Publish write-once files, with the summary/commit marker ordered last."""

    published: list[tuple[Path, int, int]] = []
    try:
        for temporary, final in staged:
            staged_stat = temporary.lstat()
            os.link(temporary, final)
            published.append((final, staged_stat.st_dev, staged_stat.st_ino))
        for directory in sorted({final.parent for _, final in staged}, key=str):
            _fsync_directory(directory)
    except BaseException as publication_error:
        rollback_failures: list[str] = []
        for final, expected_device, expected_inode in reversed(published):
            try:
                final_stat = final.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                rollback_failures.append(f"could not inspect {final}: {error}")
                continue
            if (final_stat.st_dev, final_stat.st_ino) != (expected_device, expected_inode):
                rollback_failures.append(f"refusing to unlink concurrently replaced output {final}")
                continue
            try:
                final.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                rollback_failures.append(f"could not unlink {final}: {error}")
        if rollback_failures:
            details = "; ".join(rollback_failures)
            raise RuntimeError(
                f"portfolio publication failed and rollback was incomplete: {details}"
            ) from publication_error
        raise


def run_final_portfolio(
    ledger_path: str | Path,
    *,
    posterior_path: str | Path,
    config_path: str | Path,
    fasta_output: str | Path,
    audit_output: str | Path,
    summary_output: str | Path,
) -> FinalPortfolioExecution:
    """Build the portfolio and write deterministic selection evidence."""

    ledger_path = Path(os.path.abspath(os.fspath(ledger_path)))
    posterior_path = Path(os.path.abspath(os.fspath(posterior_path)))
    config_path = Path(os.path.abspath(os.fspath(config_path)))
    raw_outputs = (Path(fasta_output), Path(audit_output), Path(summary_output))
    fasta_path, audit_path, summary_path = (output.resolve() for output in raw_outputs)
    inputs = {ledger_path, posterior_path, config_path}
    outputs = {fasta_path, audit_path, summary_path}
    if len(outputs) != 3:
        raise ValueError("FASTA, audit, and summary outputs must be distinct")
    if inputs & outputs:
        raise ValueError("portfolio outputs cannot overwrite an input")
    for output in raw_outputs:
        if _lexists(output):
            raise FileExistsError(f"refusing to overwrite existing portfolio output: {output}")
    for output in sorted(outputs, key=str):
        if _lexists(output):
            raise FileExistsError(f"refusing to overwrite existing portfolio output: {output}")

    input_paths = {
        "ledger": ledger_path,
        "posterior": posterior_path,
        "config": config_path,
    }
    input_stack = ExitStack()
    try:
        pinned_inputs = {
            name: input_stack.enter_context(_pinned_input(path, label=name))
            for name, path in input_paths.items()
        }
        input_fingerprints = {name: pinned.fingerprint for name, pinned in pinned_inputs.items()}
        run_config = load_final_portfolio_config(pinned_inputs["config"].parse_path)
        ledger = read_candidate_ledger(
            pinned_inputs["ledger"].parse_path,
            objectives=run_config.objectives,
        )
        reserved = {name for name in ledger.fieldnames if name.startswith("portfolio_")}
        if reserved:
            raise ValueError(
                f"candidate ledger uses reserved portfolio column(s): {sorted(reserved)}"
            )
        candidates = load_portfolio_candidates(
            pinned_inputs["posterior"].parse_path,
            ledger=ledger,
            objectives=run_config.objectives,
        )
        result = RewardProtectedPortfolioSelector(run_config.portfolio).select(candidates)
    except BaseException:
        input_stack.close()
        raise
    staged_list: list[tuple[Path, Path]] = []
    try:
        for final in (fasta_path, audit_path, summary_path):
            staged_list.append((_new_staged_path(final), final))
    except BaseException:
        for temporary, _ in staged_list:
            temporary.unlink(missing_ok=True)
        input_stack.close()
        raise
    staged = tuple(staged_list)
    try:
        staged_fasta, staged_audit, staged_summary = (temporary for temporary, _ in staged)
        _write_fasta(staged_fasta, ledger=ledger, result=result)
        _write_audit(staged_audit, ledger=ledger, result=result)
        summary = {
            "schema_version": 2,
            "policy": "reward_protected_final_portfolio_v2",
            "objectives": list(run_config.objectives),
            "candidate_count": len(candidates.sequences),
            "posterior_draws": int(candidates.reward_samples.shape[1]),
            "posterior_storage_dtype": str(candidates.reward_samples.dtype),
            "input_fingerprints": {
                name: _input_fingerprint_summary(fingerprint)
                for name, fingerprint in input_fingerprints.items()
            },
            "ledger_sha256": input_fingerprints["ledger"].sha256,
            "posterior_sha256": input_fingerprints["posterior"].sha256,
            "config_sha256": input_fingerprints["config"].sha256,
            "fasta_sha256": _sha256(staged_fasta),
            "audit_sha256": _sha256(staged_audit),
            "portfolio_size": len(result.indices),
            "uniform_sample_size": run_config.portfolio.uniform_sample_size,
            "panel_draws": run_config.portfolio.panel_draws,
            "panel_seed": run_config.portfolio.panel_seed,
            "confirmation_panel_draws": run_config.portfolio.confirmation_panel_draws,
            "confirmation_panel_seed": run_config.portfolio.confirmation_panel_seed,
            "cvar_alpha": run_config.portfolio.cvar_alpha,
            "mean_only_cutoff": result.mean_only_cutoff,
            "fell_back_to_mean": result.fell_back_to_mean,
            "fallback_reason": result.fallback_reason,
            "baseline_solver": asdict(result.baseline_solver),
            "confirmation": asdict(result.confirmation),
            "baseline_metrics": _metrics_dict(result.baseline_metrics),
            "final_metrics": _metrics_dict(result.final_metrics),
            "baseline_sequence_ids": [
                ledger.metadata[index].sequence_id for index in result.baseline_indices
            ],
            "final_sequence_ids": [ledger.metadata[index].sequence_id for index in result.indices],
            "search_swaps": [asdict(swap) for swap in result.search_swaps],
            "swaps": [asdict(swap) for swap in result.swaps],
        }
        _write_summary(staged_summary, summary)
        for name, pinned in pinned_inputs.items():
            _revalidate_pinned_input(pinned, label=name)
        _publish_staged_bundle(staged)
    finally:
        for temporary, _ in staged:
            with suppress(FileNotFoundError):
                temporary.unlink()
        input_stack.close()
    return FinalPortfolioExecution(
        config=run_config,
        ledger=ledger,
        result=result,
        fasta_path=fasta_path,
        audit_path=audit_path,
        summary_path=summary_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a reward-protected final AMP portfolio."
    )
    parser.add_argument("ledger", type=Path, help="candidate CSV ledger")
    parser.add_argument(
        "--posterior",
        type=Path,
        required=True,
        help="aligned NPZ posterior reward draws",
    )
    parser.add_argument("--config", type=Path, required=True, help="portfolio TOML")
    parser.add_argument("--fasta-out", type=Path, required=True, help="ranked FASTA")
    parser.add_argument("--audit-out", type=Path, required=True, help="selection audit CSV")
    parser.add_argument("--summary-out", type=Path, required=True, help="reward-gate summary JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = run_final_portfolio(
            args.ledger,
            posterior_path=args.posterior,
            config_path=args.config,
            fasta_output=args.fasta_out,
            audit_output=args.audit_out,
            summary_output=args.summary_out,
        )
    except (
        OSError,
        RuntimeError,
        UnicodeError,
        csv.Error,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"AMP final-portfolio error: {error}", file=sys.stderr)
        return 2
    print(
        "AMP final portfolio complete: "
        f"selected={len(execution.result.indices):,} "
        f"swaps={len(execution.result.swaps):,} "
        f"fallback={str(execution.result.fell_back_to_mean).lower()} "
        f"fasta={execution.fasta_path} audit={execution.audit_path} "
        f"summary={execution.summary_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script path is preferred
    raise SystemExit(main())

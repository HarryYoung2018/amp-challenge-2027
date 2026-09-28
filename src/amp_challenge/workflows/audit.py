"""Audit generated FASTA files against the organizer's acceptance contract.

The parser and checks in this module intentionally mirror the organizer's
reference validator.  In particular, sequence lines are uppercased and joined
but embedded whitespace is *not* removed, and top-set novelty uses the same
normalized-indel score as ``Levenshtein.ratio`` rather than the project's
alignment-identity helper. RapidFuzz supplies that equivalent score under a
permissive license.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.distance.Indel import normalized_similarity as levenshtein_ratio

from amp_challenge.constants import (
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACID_SET,
)

DEFAULT_LIBRARY_SIZE = 50_000
DEFAULT_TOP_SIZE = 100
DEFAULT_REFERENCE_PATH = Path("data/antibacterial.fasta")
MAX_TOP_REFERENCE_RATIO = 0.8
DEFAULT_MAX_REPORTED_ISSUES = 50


@dataclass(frozen=True, slots=True)
class FastaContents:
    """Headers and sequences parsed with the organizer's FASTA semantics."""

    headers: tuple[str, ...]
    sequences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Machine-usable summary of a submission contract audit."""

    library_count: int
    top_count: int
    reference_count: int
    issue_count: int
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every audited organizer constraint passed."""

        return self.issue_count == 0

    @property
    def omitted_issue_count(self) -> int:
        """Return the number of issues not retained in the bounded report."""

        return self.issue_count - len(self.issues)


class _IssueCollector:
    """Count every issue while retaining only a bounded diagnostic sample."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_reported_issues must be at least 1")
        self.limit = limit
        self.total = 0
        self.items: list[str] = []

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.items) < self.limit:
            self.items.append(message)


def read_organizer_fasta(path: str | Path) -> FastaContents:
    """Read FASTA exactly as the organizer's current reference validator does.

    Blank lines are skipped.  A header begins only when the stripped line starts
    with ``>``.  Sequence lines are stripped at their ends, uppercased, and
    concatenated; internal whitespace remains and is later rejected as a
    non-standard character.  Text before the first header is ignored.
    """

    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    sequence_parts: list[str] = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(sequence_parts))
            header = line[1:]
            sequence_parts = []
        else:
            sequence_parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(sequence_parts))

    return FastaContents(headers=tuple(headers), sequences=tuple(sequences))


def _validate_library(
    contents: FastaContents,
    *,
    expected_size: int,
    issues: _IssueCollector,
) -> set[str]:
    sequences = contents.sequences
    if not sequences:
        issues.add("library FASTA contains no records")
    elif len(sequences) != expected_size:
        issues.add(f"library contains {len(sequences):,} records; expected {expected_size:,}")

    seen: set[str] = set()
    for record_number, (header, sequence) in enumerate(
        zip(contents.headers, sequences, strict=True), start=1
    ):
        if not header.strip():
            issues.add(f"library record {record_number} has an empty header")

        if not sequence:
            issues.add(f"library record {record_number} has an empty sequence")
        else:
            invalid = sorted(set(sequence) - STANDARD_AMINO_ACID_SET)
            if invalid:
                rendered = ", ".join(repr(character) for character in invalid)
                issues.add(
                    f"library record {record_number} contains non-standard character(s): {rendered}"
                )
            if len(sequence) < MIN_SEQUENCE_LENGTH:
                issues.add(
                    f"library record {record_number} has length {len(sequence)}; "
                    f"minimum is {MIN_SEQUENCE_LENGTH}"
                )
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                issues.add(
                    f"library record {record_number} has length {len(sequence)}; "
                    f"maximum is {MAX_SEQUENCE_LENGTH}"
                )

        if sequence in seen:
            issues.add(f"library record {record_number} duplicates sequence {sequence!r}")
        seen.add(sequence)

    return seen


def _validate_top(
    contents: FastaContents,
    *,
    expected_size: int,
    library_sequences: set[str],
    issues: _IssueCollector,
) -> tuple[str, ...]:
    """Check top size, subset membership, and uniqueness.

    The organizer does not inspect top FASTA headers or independently repeat
    alphabet/length checks: membership in the already validated library carries
    those constraints through.  This function preserves that behavior.
    """

    sequences = contents.sequences
    if len(sequences) != expected_size:
        issues.add(f"top FASTA contains {len(sequences):,} records; expected {expected_size:,}")

    seen: set[str] = set()
    comparable: list[str] = []
    for record_number, sequence in enumerate(sequences, start=1):
        if sequence not in library_sequences:
            issues.add(f"top record {record_number} is not present in the library")
        else:
            comparable.append(sequence)
        if sequence in seen:
            issues.add(f"top record {record_number} duplicates sequence {sequence!r}")
        seen.add(sequence)
    return tuple(comparable)


def _first_similarity_violation(
    top_sequences: Sequence[str],
    reference_sequences: Sequence[str],
) -> tuple[str, str, float] | None:
    """Return the first deterministic organizer-ratio violation, if one exists."""

    references = sorted(set(reference_sequences))
    for candidate in sorted(set(top_sequences)):
        for reference in references:
            score = levenshtein_ratio(candidate, reference)
            if score > MAX_TOP_REFERENCE_RATIO:
                return candidate, reference, score
    return None


def audit_submission(
    library_path: str | Path,
    *,
    top_path: str | Path,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    expected_library_size: int = DEFAULT_LIBRARY_SIZE,
    expected_top_size: int = DEFAULT_TOP_SIZE,
    max_reported_issues: int = DEFAULT_MAX_REPORTED_ISSUES,
) -> AuditReport:
    """Audit library and ranked-top FASTAs against the organizer contract.

    ``expected_library_size`` and ``expected_top_size`` exist for bounded smoke
    tests.  Their defaults remain the official 50,000 and 100 record counts.
    File-system and decoding errors are allowed to propagate so callers can
    distinguish unreadable inputs from contract violations.
    """

    if expected_library_size < 1:
        raise ValueError("expected_library_size must be at least 1")
    if expected_top_size < 1:
        raise ValueError("expected_top_size must be at least 1")

    library = read_organizer_fasta(library_path)
    top = read_organizer_fasta(top_path)
    reference = read_organizer_fasta(reference_path)
    issues = _IssueCollector(max_reported_issues)

    library_sequences = _validate_library(
        library,
        expected_size=expected_library_size,
        issues=issues,
    )

    exact_overlap = sorted(library_sequences & set(reference.sequences))
    if exact_overlap:
        examples = ", ".join(repr(sequence) for sequence in exact_overlap[:3])
        suffix = "" if len(exact_overlap) <= 3 else ", ..."
        issues.add(
            f"library has {len(exact_overlap):,} exact antibacterial-reference "
            f"overlap(s): {examples}{suffix}"
        )

    comparable_top = _validate_top(
        top,
        expected_size=expected_top_size,
        library_sequences=library_sequences,
        issues=issues,
    )
    violation = _first_similarity_violation(comparable_top, reference.sequences)
    if violation is not None:
        candidate, reference_sequence, score = violation
        issues.add(
            "top candidate exceeds the organizer Levenshtein-ratio limit: "
            f"{candidate!r} vs {reference_sequence!r} = {score:.6f} "
            f"> {MAX_TOP_REFERENCE_RATIO:.1f}"
        )

    return AuditReport(
        library_count=len(library.sequences),
        top_count=len(top.sequences),
        reference_count=len(reference.sequences),
        issue_count=issues.total,
        issues=tuple(issues.items),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the ``amp-audit`` command-line parser."""

    parser = argparse.ArgumentParser(
        description="Audit AMP Challenge library and top FASTAs before submission."
    )
    parser.add_argument("library", type=Path, help="path to the full library FASTA")
    parser.add_argument("--top", type=Path, required=True, help="path to the ranked top FASTA")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help=f"organizer antibacterial reference (default: {DEFAULT_REFERENCE_PATH})",
    )
    parser.add_argument(
        "--expected-library-size",
        type=_positive_integer,
        default=DEFAULT_LIBRARY_SIZE,
        help=f"required library record count (default: {DEFAULT_LIBRARY_SIZE})",
    )
    parser.add_argument(
        "--expected-top-size",
        type=_positive_integer,
        default=DEFAULT_TOP_SIZE,
        help=f"required top record count (default: {DEFAULT_TOP_SIZE})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI; return 0 on pass, 1 on violations, or 2 on I/O error."""

    args = build_parser().parse_args(argv)
    try:
        report = audit_submission(
            args.library,
            top_path=args.top,
            reference_path=args.reference,
            expected_library_size=args.expected_library_size,
            expected_top_size=args.expected_top_size,
        )
    except (OSError, UnicodeError) as error:
        print(f"AMP audit input error: {error}", file=sys.stderr)
        return 2

    counts = (
        f"library={report.library_count:,}, top={report.top_count:,}, "
        f"reference={report.reference_count:,}"
    )
    if report.passed:
        print(f"AMP contract audit PASSED ({counts})")
        return 0

    print(
        f"AMP contract audit FAILED with {report.issue_count:,} issue(s) ({counts})",
        file=sys.stderr,
    )
    for issue in report.issues:
        print(f"- {issue}", file=sys.stderr)
    if report.omitted_issue_count:
        print(
            f"- ... {report.omitted_issue_count:,} additional issue(s) omitted",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":  # pragma: no cover - console-script path is preferred
    raise SystemExit(main())

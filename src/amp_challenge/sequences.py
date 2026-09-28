"""Canonicalization, validation, and stable identifiers for peptide strings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .constants import (
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACID_SET,
)


class SequenceValidationError(ValueError):
    """Raised when a sequence violates the requested string-level contract."""


@dataclass(frozen=True, slots=True)
class SequenceValidation:
    """Result of validating a normalized sequence."""

    sequence: str
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def normalize_sequence(sequence: str) -> str:
    """Uppercase a sequence and remove whitespace, without validating it.

    Whitespace is removed so multiline FASTA entries and pasted strings receive
    the same ID.  Punctuation is deliberately retained and subsequently rejected
    rather than silently interpreted as a modification annotation.
    """

    if not isinstance(sequence, str):
        raise TypeError(f"sequence must be str, got {type(sequence).__name__}")
    return "".join(sequence.split()).upper()


def validate_sequence(
    sequence: str,
    *,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> SequenceValidation:
    """Validate the alphabet and length of a normalized peptide sequence.

    This cannot establish that a source peptide has free termini or lacks other
    chemical modifications; those properties must be checked in source metadata.
    """

    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    if max_length < min_length:
        raise ValueError("max_length must be greater than or equal to min_length")

    normalized = normalize_sequence(sequence)
    issues: list[str] = []
    if not normalized:
        issues.append("sequence is empty")
    if len(normalized) < min_length:
        issues.append(f"sequence length {len(normalized)} is below {min_length}")
    if len(normalized) > max_length:
        issues.append(f"sequence length {len(normalized)} exceeds {max_length}")

    invalid = sorted(set(normalized) - STANDARD_AMINO_ACID_SET)
    if invalid:
        issues.append("non-standard residue(s): " + ", ".join(invalid))
    return SequenceValidation(sequence=normalized, issues=tuple(issues))


def canonicalize_sequence(
    sequence: str,
    *,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> str:
    """Return the canonical sequence or raise :class:`SequenceValidationError`."""

    result = validate_sequence(
        sequence,
        min_length=min_length,
        max_length=max_length,
    )
    if result.issues:
        raise SequenceValidationError("; ".join(result.issues))
    return result.sequence


def canonical_sequence_id(sequence: str) -> str:
    """Return the lowercase hexadecimal SHA-256 of a canonical valid sequence."""

    canonical = canonicalize_sequence(sequence)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# Concise alias for table-building code.
sequence_id = canonical_sequence_id

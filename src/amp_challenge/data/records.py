"""Immutable normalized records with explicit source and censoring provenance."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..sequences import canonical_sequence_id, canonicalize_sequence

CensorRelation = Literal["eq", "approx", "lt", "le", "gt", "ge", "range"]

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RANGE_PATTERN = re.compile(
    rf"^\s*({_NUMBER})\s*(?:-|\u2013|\u2014|\bto\b)\s*({_NUMBER})\s*(.*?)\s*$",
    re.IGNORECASE,
)
_VALUE_PATTERN = re.compile(rf"^\s*(<=|>=|<|>|=|~)?\s*({_NUMBER})\s*(.*?)\s*$")
_MISSING_VALUES = frozenset({"", "-", "na", "n/a", "nan", "none", "nd", "not determined"})


def normalize_concentration_unit(unit: str | None) -> str | None:
    """Normalize common MIC unit spellings without guessing unknown units."""

    if unit is None:
        return None
    stripped = unit.strip().strip("()[]").replace("μ", "u").replace("µ", "u")
    compact = stripped.lower().replace(" ", "")
    aliases = {
        "ug/ml": "ug/mL",
        "mcg/ml": "ug/mL",
        "microgram/ml": "ug/mL",
        "micrograms/ml": "ug/mL",
        "ng/ml": "ng/mL",
        "pg/ml": "pg/mL",
        "mg/l": "mg/L",
        "ug/l": "ug/L",
        "mg/ml": "mg/mL",
        "g/l": "g/L",
        "um": "uM",
        "umol/l": "uM",
        "umolar": "uM",
        "micromolar": "uM",
        "nm": "nM",
        "nmol/l": "nM",
        "nanomolar": "nM",
        "pm": "pM",
        "pmol/l": "pM",
        # One pmol per mL is numerically one nmol per L.
        "pmol/ml": "nM",
        "mm": "mM",
        "mmol/l": "mM",
        "millimolar": "mM",
        "m": "M",
        "mol/l": "M",
    }
    return aliases.get(compact, stripped or None)


@dataclass(frozen=True, slots=True)
class CensoredValue:
    """An interval-valued measurement that preserves censoring semantics.

    ``lower`` or ``upper`` is ``None`` for an unbounded side.  Callers should
    train censoring-aware objectives where possible; ``representative_value`` is
    supplied only for descriptive summaries and simple baselines.
    """

    relation: CensorRelation
    lower: float | None
    upper: float | None
    lower_inclusive: bool
    upper_inclusive: bool
    unit: str | None = None
    raw: str | None = None
    source_unit: str | None = None

    def __post_init__(self) -> None:
        allowed_relations = {"eq", "approx", "lt", "le", "gt", "ge", "range"}
        if self.relation not in allowed_relations:
            raise ValueError(f"unsupported censor relation: {self.relation!r}")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lower is None and self.upper is None:
            raise ValueError("at least one measurement bound is required")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower bound cannot exceed upper bound")
        if self.relation in {"eq", "approx"} and not (
            self.lower is not None
            and self.upper is not None
            and self.lower == self.upper
            and self.lower_inclusive
            and self.upper_inclusive
        ):
            raise ValueError(f"{self.relation} requires one inclusive exact value")
        if self.relation in {"lt", "le"} and not (self.lower is None and self.upper is not None):
            raise ValueError(f"{self.relation} requires only an upper bound")
        if self.relation in {"gt", "ge"} and not (self.lower is not None and self.upper is None):
            raise ValueError(f"{self.relation} requires only a lower bound")
        if self.relation == "range" and not (self.lower is not None and self.upper is not None):
            raise ValueError("range requires lower and upper bounds")
        expected_lower_inclusive = self.relation in {"eq", "approx", "ge", "range"}
        expected_upper_inclusive = self.relation in {"eq", "approx", "le", "range"}
        if self.lower_inclusive != expected_lower_inclusive:
            raise ValueError(f"inconsistent lower-bound inclusivity for {self.relation}")
        if self.upper_inclusive != expected_upper_inclusive:
            raise ValueError(f"inconsistent upper-bound inclusivity for {self.relation}")
        object.__setattr__(self, "unit", normalize_concentration_unit(self.unit))
        object.__setattr__(
            self,
            "source_unit",
            normalize_concentration_unit(self.source_unit),
        )

    @property
    def is_censored(self) -> bool:
        return self.relation in {"lt", "le", "gt", "ge"}

    @property
    def representative_value(self) -> float:
        """Return a boundary or interval midpoint for summaries, not labels."""

        if self.lower is None:
            assert self.upper is not None
            return self.upper
        if self.upper is None:
            return self.lower
        if self.lower > 0 and self.upper > 0 and self.lower != self.upper:
            return math.sqrt(self.lower * self.upper)
        return (self.lower + self.upper) / 2.0

    def scaled(self, factor: float, *, unit: str) -> CensoredValue:
        """Scale both bounds while retaining inclusivity and raw provenance."""

        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("scale factor must be finite and positive")
        return CensoredValue(
            relation=self.relation,
            lower=None if self.lower is None else self.lower * factor,
            upper=None if self.upper is None else self.upper * factor,
            lower_inclusive=self.lower_inclusive,
            upper_inclusive=self.upper_inclusive,
            unit=unit,
            raw=self.raw,
            source_unit=self.source_unit or self.unit,
        )


def parse_censored_value(
    value: str | int | float | None,
    *,
    unit: str | None = None,
) -> CensoredValue | None:
    """Parse exact, approximate, one-sided, or ranged numeric measurements.

    Examples include ``"4"``, ``"<= 8 ug/mL"``, ``">256"``, and ``"4-8"``.
    Missing-value tokens return ``None``.  More complex prose is rejected rather
    than silently coerced.
    """

    explicit_unit = normalize_concentration_unit(unit)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric measurement")
    if isinstance(value, int | float):
        numeric = float(value)
        return CensoredValue(
            relation="eq",
            lower=numeric,
            upper=numeric,
            lower_inclusive=True,
            upper_inclusive=True,
            unit=explicit_unit,
            raw=str(value),
            source_unit=explicit_unit,
        )

    raw = str(value).strip()
    if raw.lower() in _MISSING_VALUES:
        return None
    normalized_symbols = raw.replace("≤", "<=").replace("≥", ">=")

    range_match = _RANGE_PATTERN.fullmatch(normalized_symbols)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        embedded_unit = normalize_concentration_unit(range_match.group(3))
        parsed_unit = _resolve_unit(explicit_unit, embedded_unit)
        return CensoredValue(
            relation="range",
            lower=lower,
            upper=upper,
            lower_inclusive=True,
            upper_inclusive=True,
            unit=parsed_unit,
            raw=raw,
            source_unit=parsed_unit,
        )

    value_match = _VALUE_PATTERN.fullmatch(normalized_symbols)
    if not value_match:
        raise ValueError(f"cannot parse censored numeric value: {raw!r}")
    operator = value_match.group(1) or "="
    numeric = float(value_match.group(2))
    embedded_unit = normalize_concentration_unit(value_match.group(3))
    parsed_unit = _resolve_unit(explicit_unit, embedded_unit)
    relation_by_operator: dict[str, CensorRelation] = {
        "=": "eq",
        "~": "approx",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
    }
    relation = relation_by_operator[operator]
    lower = numeric if relation in {"eq", "approx", "gt", "ge"} else None
    upper = numeric if relation in {"eq", "approx", "lt", "le"} else None
    return CensoredValue(
        relation=relation,
        lower=lower,
        upper=upper,
        lower_inclusive=relation in {"eq", "approx", "ge"},
        upper_inclusive=relation in {"eq", "approx", "le"},
        unit=parsed_unit,
        raw=raw,
        source_unit=parsed_unit,
    )


def _resolve_unit(explicit: str | None, embedded: str | None) -> str | None:
    if explicit is not None and embedded is not None and explicit != embedded:
        raise ValueError(f"conflicting concentration units: {explicit!r} and {embedded!r}")
    return explicit or embedded


def convert_concentration_to_um(
    measurement: CensoredValue,
    *,
    molecular_weight_da: float | None = None,
) -> CensoredValue:
    """Convert supported concentration units to micromolar.

    Mass-concentration conversion requires molecular weight and assumes the
    measured material is the unmodified free-termini sequence.  Unknown or
    missing units raise instead of being guessed.
    """

    unit = normalize_concentration_unit(measurement.unit)
    molar_factors = {"uM": 1.0, "nM": 1e-3, "pM": 1e-6, "mM": 1e3, "M": 1e6}
    if unit in molar_factors:
        return measurement.scaled(molar_factors[unit], unit="uM")
    mass_numerators = {
        "ug/mL": 1e3,
        "mg/L": 1e3,
        "ug/L": 1.0,
        "ng/mL": 1.0,
        "pg/mL": 1e-3,
        "mg/mL": 1e6,
        "g/L": 1e6,
    }
    if unit in mass_numerators:
        if molecular_weight_da is None:
            raise ValueError(f"molecular weight is required to convert {unit}")
        if not math.isfinite(molecular_weight_da) or molecular_weight_da <= 0:
            raise ValueError("molecular_weight_da must be finite and positive")
        return measurement.scaled(
            mass_numerators[unit] / molecular_weight_da,
            unit="uM",
        )
    raise ValueError(f"unsupported or missing concentration unit: {unit!r}")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Location of one source observation or FASTA entry."""

    source: str
    record_id: str | None = None
    path: str | None = None
    row_number: int | None = None
    extra: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("provenance source cannot be empty")
        if self.row_number is not None and self.row_number < 1:
            raise ValueError("row_number must be positive")
        normalized_extra = tuple(sorted((str(key), str(value)) for key, value in self.extra))
        object.__setattr__(self, "extra", normalized_extra)

    @classmethod
    def from_mapping(
        cls,
        *,
        source: str,
        record_id: str | None = None,
        path: str | None = None,
        row_number: int | None = None,
        extra: Mapping[str, object] | None = None,
        include_empty: bool = False,
    ) -> Provenance:
        pairs = ()
        if extra:
            pairs = tuple(
                (str(key), str(value))
                for key, value in extra.items()
                if value is not None and (include_empty or str(value).strip())
            )
        return cls(
            source=source,
            record_id=record_id,
            path=path,
            row_number=row_number,
            extra=pairs,
        )


@dataclass(frozen=True, slots=True)
class AssayObservation:
    """One normalized endpoint observation linked to its source row.

    ``exposure_concentration`` keeps an effect such as percent hemolysis
    separate from the dose at which it was observed. ``source_text`` retains
    the complete source statement when a numeric token alone is insufficient
    to reconstruct endpoint semantics.
    """

    endpoint: str
    value: CensoredValue
    provenance: Provenance
    organism: str | None = None
    strain: str | None = None
    gram: Literal["positive", "negative", "unknown"] = "unknown"
    assay: str | None = None
    exposure_concentration: CensoredValue | None = None
    source_text: str | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("assay endpoint cannot be empty")
        if self.gram not in {"positive", "negative", "unknown"}:
            raise ValueError(f"invalid gram category: {self.gram!r}")
        if self.source_text is not None and not self.source_text.strip():
            raise ValueError("assay source_text cannot be blank")


@dataclass(frozen=True, slots=True)
class PeptideRecord:
    """A canonical sequence with all retained provenance and observations."""

    sequence_id: str
    sequence: str
    provenance: tuple[Provenance, ...]
    assays: tuple[AssayObservation, ...] = ()

    def __post_init__(self) -> None:
        canonical = canonicalize_sequence(self.sequence)
        expected_id = canonical_sequence_id(canonical)
        if self.sequence != canonical:
            raise ValueError("PeptideRecord.sequence must already be canonical")
        if self.sequence_id != expected_id:
            raise ValueError("sequence_id does not match canonical SHA-256")
        if not self.provenance:
            raise ValueError("at least one provenance entry is required")

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        *,
        provenance: Provenance,
        assays: tuple[AssayObservation, ...] = (),
    ) -> PeptideRecord:
        canonical = canonicalize_sequence(sequence)
        return cls(
            sequence_id=canonical_sequence_id(canonical),
            sequence=canonical,
            provenance=(provenance,),
            assays=assays,
        )

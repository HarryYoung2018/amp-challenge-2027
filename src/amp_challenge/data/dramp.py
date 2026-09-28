"""Conservative normalization of the immutable DRAMP general-peptide deposit."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from amp_challenge.data.fetch import DataArtifact, load_data_manifest, verify_artifact
from amp_challenge.data.prepare import (
    IngestionResult,
    RejectedEndpoint,
    RejectedRecord,
    deduplicate_records,
    endpoint_rejection_id,
)
from amp_challenge.data.records import (
    AssayObservation,
    CensoredValue,
    PeptideRecord,
    Provenance,
    convert_concentration_to_um,
    parse_censored_value,
)
from amp_challenge.data.starter import write_normalized_dataset
from amp_challenge.descriptors import molecular_weight
from amp_challenge.sequences import SequenceValidationError

GramClass = Literal["positive", "negative", "unknown"]
_EXPECTED_SOURCE_ROWS = 5_084

DRAMP_WORKBOOK_COLUMNS = (
    "DRAMP_ID",
    "Sequence",
    "Sequence_Length",
    "Name",
    "Swiss_Prot_Entry",
    "Family",
    "Gene",
    "Source",
    "Activity",
    "Protein_existence",
    "Structure",
    "Structure_Description",
    "PDB_ID",
    "Comments",
    "Target_Organism",
    "Hemolytic_activity",
    "Binding_Target",
    "Pubmed_ID",
    "Reference",
    "Author",
    "Title",
)

_REQUIRED_COLUMNS = {
    "DRAMP_ID",
    "Sequence",
    "Sequence_Length",
    "Name",
    "Family",
    "Source",
    "Activity",
    "Structure",
    "Comments",
    "Target_Organism",
    "Hemolytic_activity",
    "Pubmed_ID",
    "Reference",
}

_NUMBER = r"\d+(?:\.\d+)?"
_RANGE = rf"{_NUMBER}(?:\s*(?:-|\u2013|\u2014|to)\s*{_NUMBER})?"
_UNIT = (
    r"(?:pmol\s*/\s*m[lL]|[munp\u03bc\u00b5]?g\s*/\s*m[lL]|"
    r"[munp\u03bc\u00b5]?mol\s*/\s*[lL]|[munp\u03bc\u00b5]?[mM])"
)
_MIC_PATTERN = re.compile(
    rf"\bMIC\s*(?P<relation><=|>=|=|<|>|\u2264|\u2265)?\s*"
    rf"(?P<value>{_RANGE})\s*(?P<unit>{_UNIT})(?![A-Za-z])",
    re.IGNORECASE,
)
_MIC_MARKER = re.compile(r"\bMIC\b", re.IGNORECASE)

_RELATION = r"(?:<=|>=|=|<|>|\u2264|\u2265)"
_HC50_PATTERN = re.compile(
    rf"\bHC\s*[_-]?\s*50\b\s*(?P<relation>{_RELATION})?\s*"
    rf"(?P<value>{_RANGE})\s*(?P<unit>{_UNIT})(?![A-Za-z])",
    re.IGNORECASE,
)
_HC50_MARKER = re.compile(r"\bHC\s*[_-]?\s*50\b", re.IGNORECASE)
_UNSUPPORTED_HALF_MAX_MARKER = re.compile(
    r"\b(?:HD|LC|EC)\s*[_-]?\s*50\b",
    re.IGNORECASE,
)
_UNSUPPORTED_HEMOLYSIS_ENDPOINT_ALIAS_MARKER = re.compile(
    r"\b(?:"
    r"MHC|RCH|"
    r"(?:LD|IC|ED|EC)\s*[_-]?\s*\d{1,3}|"
    r"lethal\s+concentration|"
    r"HU\s*/\s*mg"
    r")\b",
    re.IGNORECASE,
)
_HEMOLYSIS_TERM = r"(?:h(?:ae|e)molytic\s+activity|h(?:ae|e)molysis)"
_PERCENT_HEMOLYSIS_PATTERN = re.compile(
    rf"(?P<effect_relation>{_RELATION})?\s*(?P<effect>{_RANGE})\s*%\s*"
    rf"{_HEMOLYSIS_TERM}\s+at\s+"
    rf"(?P<exposure_relation>{_RELATION})?\s*(?P<exposure>{_RANGE})\s*"
    rf"(?P<unit>{_UNIT})(?![A-Za-z])",
    re.IGNORECASE,
)
_NUMERIC_HEMOLYSIS_MARKER = re.compile(
    rf"(?:{_NUMBER}\s*%\s*[^.;#]{{0,48}}{_HEMOLYSIS_TERM}|"
    rf"{_HEMOLYSIS_TERM}\s*[^.;#]{{0,48}}{_NUMBER}\s*%)",
    re.IGNORECASE,
)
_PLUS_MINUS_UNCERTAINTY_MARKER = re.compile(
    r"(?:[\u00b1\u2213]|\+\s*/\s*[-\u2212]|\bplus\s+(?:or\s+)?minus\b)",
    re.IGNORECASE,
)
_BLOOD_TARGET_MARKER = re.compile(
    r"\b(?:erythrocytes?|red\s+blood\s+cells?|RBCs?|blood)\b",
    re.IGNORECASE,
)
_BLOOD_SPECIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhuman\b", re.IGNORECASE), "human"),
    (re.compile(r"\brabbits?\b", re.IGNORECASE), "rabbit"),
    (re.compile(r"\b(?:mice|mouse)\b", re.IGNORECASE), "mouse"),
    (re.compile(r"\brats?\b", re.IGNORECASE), "rat"),
    (re.compile(r"\b(?:porcine|pigs?)\b", re.IGNORECASE), "pig"),
    (re.compile(r"\b(?:calf|calves|bovine|cattle)\b", re.IGNORECASE), "cattle"),
    (re.compile(r"\b(?:ovine|sheep)\b", re.IGNORECASE), "sheep"),
    (re.compile(r"\b(?:caprine|goats?)\b", re.IGNORECASE), "goat"),
    (re.compile(r"\b(?:equine|horses?)\b", re.IGNORECASE), "horse"),
    (re.compile(r"\b(?:chicken|chickens)\b", re.IGNORECASE), "chicken"),
)

_CHEMISTRY_MARKERS: tuple[tuple[str, str], ...] = (
    ("ptm:", "explicit post-translational modification"),
    ("amidat", "amidated terminus"),
    ("n-terminal acetyl", "acetylated terminus"),
    ("n-terminally acetyl", "acetylated terminus"),
    ("head-to-tail cycl", "head-to-tail cyclization"),
    ("cyclic peptide", "cyclic peptide"),
    ("cyclic tachyplesin", "cyclic peptide"),
    ("cyclotide", "cyclotide"),
    ("disulfide", "disulfide-bonded peptide"),
    ("disulphide", "disulfide-bonded peptide"),
    ("cystine knot", "disulfide-bonded peptide"),
    ("cysteine knot", "disulfide-bonded peptide"),
    ("(oxidized)", "oxidation-state-specific peptide"),
    ("(oxidised)", "oxidation-state-specific peptide"),
    ("(reduced)", "reduction-state-specific peptide"),
)


class HemolysisEndpointQuarantine(ValueError):
    """Expected source ambiguity that invalidates only one hemolysis field."""

    def __init__(
        self,
        reason_code: str,
        reason_detail: str,
        *,
        discarded_candidate_observations: int = 0,
    ) -> None:
        super().__init__(f"hemolysis observation quarantine: {reason_detail}")
        self.reason_code = reason_code
        self.reason_detail = reason_detail
        self.discarded_candidate_observations = discarded_candidate_observations

    def with_discarded_candidates(self, count: int) -> HemolysisEndpointQuarantine:
        return HemolysisEndpointQuarantine(
            self.reason_code,
            self.reason_detail,
            discarded_candidate_observations=count,
        )


def _cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def _normalized_unit(raw: str) -> str:
    compact = re.sub(r"\s+", "", raw).replace("\u03bc", "u").replace("\u00b5", "u")
    lower = compact.lower()
    aliases = {
        "ug/ml": "ug/mL",
        "ng/ml": "ng/mL",
        "pg/ml": "pg/mL",
        "mg/ml": "mg/mL",
        "pmol/ml": "pmol/mL",
        "umol/l": "uM",
        "nmol/l": "nM",
        "mmol/l": "mM",
        "pmol/l": "pM",
        "um": "uM",
        "nm": "nM",
        "mm": "mM",
        "pm": "pM",
        "m": "M",
    }
    if lower not in aliases:
        raise ValueError(f"unsupported DRAMP concentration unit: {raw!r}")
    return aliases[lower]


def _gram_at(text: str, position: int) -> GramClass:
    prefix = text[:position].lower()
    section_start = prefix.rfind("##")
    if section_start >= 0:
        prefix = prefix[section_start + 2 :]
    positive = max(prefix.rfind("gram-positive"), prefix.rfind("gram positive"))
    negative = max(prefix.rfind("gram-negative"), prefix.rfind("gram negative"))
    if positive < 0 and negative < 0:
        return "unknown"
    non_bacterial = max(
        prefix.rfind("fungi:"),
        prefix.rfind("fungus:"),
        prefix.rfind("yeast:"),
        prefix.rfind("viruses:"),
        prefix.rfind("virus:"),
        prefix.rfind("parasites:"),
        prefix.rfind("parasite:"),
    )
    if non_bacterial > max(positive, negative):
        return "unknown"
    return "positive" if positive > negative else "negative"


_LEADING_TARGET_NOISE = re.compile(
    r"^(?:\[(?:Ref\.[^\]]+|Swiss_Prot Entry [^\]]+)\]\s*)*"
    r"(?:(?:and|or)\s+)?"
    r"(?:(?:active|inactive|effective|ineffective)(?:\s+activity)?\s+"
    r"(?:against|towards?)\s+)?"
    r"(?:(?:(?:gram[- ]?(?:positive|negative))\s+"
    r"(?:bacteria|bacterium|strains?)?|fungi|fungus|yeasts?|viruses?|parasites?)\s*:\s*)?",
    re.IGNORECASE,
)

_CONDITION_WITH_COLON = re.compile(
    r"^(?P<condition>(?:in|at|under)\b[^:]{1,160})\s*:\s*(?P<target>.+)$",
    re.IGNORECASE,
)
_CONDITION_WITHOUT_COLON = re.compile(
    r"^(?P<condition>in\s+\d+(?:\.\d+)?%\s+(?:TSB|SAB|MH)\s+in\s+"
    r"(?:SPB|PIL)(?:\s+medium)?)\s+(?P<target>.+)$",
    re.IGNORECASE,
)
_BRACKETED_CONDITION = re.compile(
    r"^\[(?P<condition>(?:No\s+NaCl|\d+(?:\.\d+)?\s*mM\s+NaCl))\]\s*:?[ ]*"
    r"(?P<target>.+)$",
    re.IGNORECASE,
)
_BRACKETED_CONDITION_START = re.compile(
    r"\[(?:No\s+NaCl|\d+(?:\.\d+)?\s*mM\s+NaCl)\]\s*:?[ ]*",
    re.IGNORECASE,
)
_TARGET_SEPARATOR = re.compile(r"\)\s*(?:[,;.]|\b(?:and|or)\b)\s*", re.IGNORECASE)
_INLINE_CONDITION = re.compile(
    r"^\s+(?P<condition>(?:in|at|under)\b[^;,#)]{1,160}?)"
    r"(?=\s*(?:[;,#)]|$))",
    re.IGNORECASE,
)

_CONTEXT_EDGE_CHARACTERS = " \t\r\n#;,.:"


def _clean_target_context(fragment: str) -> str:
    # Keep square brackets for the first substitution so a leading DRAMP
    # ``[Ref.<id>]`` token remains recognizable.
    # Parentheses and brackets within a target carry strain/isolate information
    # and must not be stripped as if they were separators.
    context = fragment.strip(_CONTEXT_EDGE_CHARACTERS)
    previous = None
    while context != previous:
        previous = context
        context = _LEADING_TARGET_NOISE.sub("", context).strip(_CONTEXT_EDGE_CHARACTERS)
    # A few deposited rows have a duplicated closing parenthesis after each
    # MIC. Remove only an unmatched edge token; balanced strain parentheses
    # remain source-faithful.
    while context.startswith(")") and context.count(")") > context.count("("):
        context = context[1:].strip(_CONTEXT_EDGE_CHARACTERS)
    return context


def _split_condition_context(fragment: str) -> tuple[str, str | None]:
    """Separate a literal assay condition prefix from its source target.

    Colon-delimited ``In/At/Under ...:`` headings are unambiguous in DRAMP.
    The deposit also has a small family of medium headings without a colon;
    only their explicit TSB/SAB/MH and SPB/PIL grammar is recognized. This
    intentionally avoids guessing where an arbitrary biological name begins.
    """

    context = _clean_target_context(fragment)
    for pattern in (
        _CONDITION_WITH_COLON,
        _CONDITION_WITHOUT_COLON,
        _BRACKETED_CONDITION,
    ):
        match = pattern.fullmatch(context)
        if match is None:
            continue
        condition = " ".join(match.group("condition").split())
        if pattern is _BRACKETED_CONDITION:
            condition = re.sub(r"(?<=\d)(?=mM\b)", " ", condition, flags=re.IGNORECASE)
        target = _clean_target_context(match.group("target"))
        if target:
            return target, condition
    return context, None


def _unclosed_parenthesis(text: str, position: int) -> int | None:
    depth = 0
    for index in range(position - 1, -1, -1):
        character = text[index]
        if character == ")":
            depth += 1
        elif character == "(":
            if depth == 0:
                return index
            depth -= 1
    return None


def _matching_parenthesis(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _clause_start(text: str, position: int) -> int:
    """Find a source-authored section/sentence boundary before one target."""

    section = text.rfind("##", 0, position)
    sentence = text.rfind(").", 0, position)
    bracketed_conditions = tuple(_BRACKETED_CONDITION_START.finditer(text, 0, position))
    bracketed = bracketed_conditions[-1].start() if bracketed_conditions else -1
    if bracketed > max(section, sentence):
        return bracketed
    if section > sentence:
        return section + 2
    if sentence >= 0:
        return sentence + 2
    return 0


def _target_fragment(text: str, position: int) -> tuple[str, str]:
    """Return the target fragment and its enclosing source clause.

    A completed earlier MIC can either delimit a new target or be a second
    measurement of the same target. Walking earlier MIC parentheses lets us
    distinguish ``A (MIC=x) and B (MIC=y)`` from ``A (MIC=x) (MIC=y)`` without
    relying on a taxonomy dictionary.
    """

    measurement_open = _unclosed_parenthesis(text, position)
    initial_end = position if measurement_open is None else measurement_open
    clause_start = _clause_start(text, initial_end)
    target_start = clause_start
    target_end = initial_end
    separators = tuple(_TARGET_SEPARATOR.finditer(text, clause_start, initial_end))
    if separators:
        candidate_start = separators[-1].end()
        if _clean_target_context(text[candidate_start:initial_end]):
            target_start = candidate_start
    # Unsupported measurements (for example ``MIC=5.4±1.0``) are deliberately
    # absent from the label parser, but their MIC marker still delimits the next
    # source target.
    prior_matches = tuple(_MIC_MARKER.finditer(text, clause_start, initial_end))
    for prior in reversed(prior_matches):
        prior_open = _unclosed_parenthesis(text, prior.start())
        if prior_open is None or prior_open < clause_start:
            # Parenthesis-free MIC prose is rare. Retain the conservative
            # punctuation fallback rather than inferring a biological name.
            separators = [
                text.rfind(",", prior.end(), target_end),
                text.rfind(";", prior.end(), target_end),
            ]
            separator = max(separators)
            if separator >= 0 and _clean_target_context(text[separator + 1 : target_end]):
                target_start = separator + 1
                break
            continue
        if prior_open == target_end:
            # Both MIC tokens are inside one assay parenthesis.
            continue
        prior_close = _matching_parenthesis(text, prior_open)
        if prior_close is None or prior_close >= target_end:
            target_end = min(target_end, prior_open)
            continue
        between = _clean_target_context(text[prior_close + 1 : target_end])
        if between:
            target_start = max(target_start, prior_close + 1)
            break
        # Adjacent MIC parentheses contain repeated measurements for the same
        # target. Move the target endpoint before the earlier measurement.
        target_end = prior_open

    if target_start >= target_end:
        target_start = clause_start
    return text[target_start:target_end], text[clause_start:initial_end]


def _inline_condition(text: str, position: int) -> str | None:
    match = _INLINE_CONDITION.match(text[position:])
    if match is None:
        return None
    return " ".join(match.group("condition").strip().split())


def _target_context(text: str, position: int) -> tuple[str, str | None]:
    """Return an organism-centered target and literal condition for one MIC.

    DRAMP often encodes a list as ``organism (MIC=x), organism (MIC=y)``.  The
    most reliable anchor is the opening parenthesis containing the measurement.
    Conditions are separated only when the source uses an explicit supported
    heading, so phrases such as ``in low-salt`` are not fabricated as strains.
    """

    fragment, clause = _target_fragment(text, position)
    target, direct_condition = _split_condition_context(fragment)
    _, clause_condition = _split_condition_context(clause)
    return target or "target context unavailable", direct_condition or clause_condition


def _chemistry_exclusion(row: dict[str, str]) -> str | None:
    searchable = "\n".join(
        (
            row.get("Name", ""),
            row.get("Family", ""),
            row.get("Structure", ""),
            row.get("Structure_Description", ""),
            row.get("Comments", ""),
            row.get("Title", ""),
            row.get("Target_Organism", ""),
            row.get("Hemolytic_activity", ""),
        )
    ).lower()
    for marker, reason in _CHEMISTRY_MARKERS:
        if marker in searchable:
            return reason
    return None


def _numeric_measurement(
    value: str,
    *,
    relation: str | None,
    unit: str,
) -> CensoredValue:
    normalized_relation = (relation or "=").replace("\u2264", "<=").replace("\u2265", ">=")
    is_range = re.search(r"(?:-|\u2013|\u2014|\bto\b)", value) is not None
    if is_range and normalized_relation != "=":
        raise ValueError("censor relation applied to a numeric range")
    if is_range:
        bounds = [float(part) for part in re.split(r"\s*(?:-|\u2013|\u2014|\bto\b)\s*", value)]
        value = f"{min(bounds):.12g}-{max(bounds):.12g}"
    measurement = parse_censored_value(
        value if is_range else f"{normalized_relation}{value}",
        unit=unit,
    )
    if measurement is None:  # pragma: no cover - guarded by the numeric grammar
        raise ValueError("missing numeric measurement")
    return measurement


def _blood_target_context(text: str, position: int) -> tuple[str, str | None]:
    """Return a source-faithful blood/cell target following one measurement.

    A target is admitted only when the same punctuation-delimited clause says
    against and names blood, erythrocytes, red blood cells, or RBCs. This
    deliberately excludes an unqualified HC50 because it could refer to a
    different cytotoxicity endpoint.
    """

    boundaries = tuple(
        index for token in (";", ".", "##") if (index := text.find(token, position)) >= 0
    )
    end = min(boundaries, default=len(text))
    clause = text[position:end]
    match = re.search(r"\bagainst\s+(?P<target>.+)$", clause, re.IGNORECASE)
    if match is None:
        raise HemolysisEndpointQuarantine(
            "missing_blood_target",
            "numeric hemolysis evidence lacks an explicit following blood target",
        )
    target_source = match.group("target")
    later_measurements = tuple(_HC50_PATTERN.finditer(target_source)) + tuple(
        _PERCENT_HEMOLYSIS_PATTERN.finditer(target_source)
    )
    if later_measurements:
        target_source = target_source[: min(item.start() for item in later_measurements)]
        target_source = re.sub(
            r"(?:,|\b(?:and|or))\s*$",
            "",
            target_source,
            flags=re.IGNORECASE,
        )
    target = " ".join(target_source.strip(_CONTEXT_EDGE_CHARACTERS).split())
    while target.endswith(")") and target.count(")") > target.count("("):
        target = target[:-1].rstrip(_CONTEXT_EDGE_CHARACTERS)
    while target.startswith("(") and target.count("(") > target.count(")"):
        target = target[1:].lstrip(_CONTEXT_EDGE_CHARACTERS)
    if not target or _BLOOD_TARGET_MARKER.search(target) is None:
        raise HemolysisEndpointQuarantine(
            "ambiguous_or_nonblood_target",
            "numeric hemolysis evidence has a non-blood or ambiguous target",
        )
    if re.search(r"\bblood\s+agar\b", target, re.IGNORECASE):
        raise HemolysisEndpointQuarantine(
            "blood_agar_target",
            "blood agar is not a blood-cell toxicity target",
        )
    organism = next(
        (canonical for pattern, canonical in _BLOOD_SPECIES if pattern.search(target)),
        None,
    )
    return target, organism


def _unparsed_numeric_hemolysis(
    text: str,
    spans: Sequence[tuple[int, int]],
) -> tuple[str, str] | None:
    remaining = list(text)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    residue = "".join(remaining)
    if _HC50_MARKER.search(residue):
        return "unsupported_or_ambiguous_hc50", "unsupported or ambiguous HC50 expression"
    unsupported = _UNSUPPORTED_HALF_MAX_MARKER.search(residue)
    if unsupported is not None:
        return "unsupported_half_max_alias", (
            f"unsupported hemolysis endpoint alias {unsupported.group(0)!r}; "
            "HD50/LC50/EC50 are not assumed equivalent to HC50"
        )
    if _NUMERIC_HEMOLYSIS_MARKER.search(residue):
        return (
            "unsupported_or_ambiguous_concentration_specific_hemolysis",
            "unsupported or ambiguous concentration-specific hemolysis expression",
        )
    unsupported = _UNSUPPORTED_HEMOLYSIS_ENDPOINT_ALIAS_MARKER.search(residue)
    if unsupported is not None:
        return "unsupported_hemolysis_endpoint_alias", (
            f"unsupported hemolysis endpoint alias {unsupported.group(0)!r}; "
            "source-specific toxicity endpoints are not assumed equivalent to "
            "HC50 or percent hemolysis at an exposure concentration"
        )
    return None


def _hemolysis_observations(
    hemolysis_text: str,
    *,
    sequence: str,
    provenance: Provenance,
) -> tuple[AssayObservation, ...]:
    """Parse only explicit HC50 or percent-at-concentration blood-cell evidence.

    Qualitative text, including non-hemolytic, yields no numeric observation.
    Numeric-looking text that invokes a supported endpoint but cannot satisfy
    the grammar and target requirements raises so the source row is quarantined
    instead of partially interpreted.
    """

    source_text = hemolysis_text.strip()
    if not source_text or source_text.lower() in {"unknown", "-", "n/a", "na"}:
        return ()
    if _PLUS_MINUS_UNCERTAINTY_MARKER.search(source_text):
        raise HemolysisEndpointQuarantine(
            "unsupported_uncertainty_notation",
            "mean-plus/minus uncertainty is not represented by the normalized endpoint schema",
        )

    observations: list[AssayObservation] = []
    parsed_spans: list[tuple[int, int]] = []
    weight = molecular_weight(sequence)

    for match in _HC50_PATTERN.finditer(source_text):
        try:
            target, organism = _blood_target_context(source_text, match.end())
        except HemolysisEndpointQuarantine as error:
            raise error.with_discarded_candidates(len(observations)) from error
        try:
            measurement = _numeric_measurement(
                match.group("value"),
                relation=match.group("relation"),
                unit=_normalized_unit(match.group("unit")),
            )
            normalized = convert_concentration_to_um(
                measurement,
                molecular_weight_da=weight,
            )
        except ValueError as error:
            raise HemolysisEndpointQuarantine(
                "unsupported_hc50_measurement",
                str(error),
                discarded_candidate_observations=len(observations),
            ) from error
        if normalized.representative_value <= 0:
            raise HemolysisEndpointQuarantine(
                "nonpositive_hc50",
                "HC50 must be positive",
                discarded_candidate_observations=len(observations),
            )
        observations.append(
            AssayObservation(
                endpoint="hc50",
                value=normalized,
                provenance=provenance,
                organism=organism,
                strain=target,
                assay="DRAMP-curated HC50; primary assay method not normalized",
                source_text=source_text,
            )
        )
        parsed_spans.append(match.span())

    for match in _PERCENT_HEMOLYSIS_PATTERN.finditer(source_text):
        try:
            target, organism = _blood_target_context(source_text, match.end())
        except HemolysisEndpointQuarantine as error:
            raise error.with_discarded_candidates(len(observations)) from error
        try:
            effect = _numeric_measurement(
                match.group("effect"),
                relation=match.group("effect_relation"),
                unit="%",
            )
        except ValueError as error:
            raise HemolysisEndpointQuarantine(
                "unsupported_hemolysis_effect_measurement",
                str(error),
                discarded_candidate_observations=len(observations),
            ) from error
        effect_bounds = tuple(value for value in (effect.lower, effect.upper) if value is not None)
        if not effect_bounds or min(effect_bounds) < 0 or max(effect_bounds) > 100:
            raise HemolysisEndpointQuarantine(
                "hemolysis_percent_out_of_range",
                "hemolysis percentage must be within [0, 100]",
                discarded_candidate_observations=len(observations),
            )
        try:
            exposure = _numeric_measurement(
                match.group("exposure"),
                relation=match.group("exposure_relation"),
                unit=_normalized_unit(match.group("unit")),
            )
            normalized_exposure = convert_concentration_to_um(
                exposure,
                molecular_weight_da=weight,
            )
        except ValueError as error:
            raise HemolysisEndpointQuarantine(
                "unsupported_hemolysis_exposure_measurement",
                str(error),
                discarded_candidate_observations=len(observations),
            ) from error
        observations.append(
            AssayObservation(
                endpoint="hemolysis_percent",
                value=effect,
                provenance=provenance,
                organism=organism,
                strain=target,
                assay=(
                    "DRAMP-curated percent hemolysis at explicit concentration; "
                    "primary assay method not normalized"
                ),
                exposure_concentration=normalized_exposure,
                source_text=source_text,
            )
        )
        parsed_spans.append(match.span())

    unparsed = _unparsed_numeric_hemolysis(source_text, parsed_spans)
    if unparsed is not None:
        reason_code, reason_detail = unparsed
        if observations:
            reason_code = "mixed_supported_and_unsupported_numeric_hemolysis"
        raise HemolysisEndpointQuarantine(
            reason_code,
            reason_detail,
            discarded_candidate_observations=len(observations),
        )
    return tuple(observations)


def _mic_observations(
    target_text: str,
    *,
    sequence: str,
    provenance: Provenance,
) -> tuple[AssayObservation, ...]:
    observations: list[AssayObservation] = []
    weight = molecular_weight(sequence)
    for match in _MIC_PATTERN.finditer(target_text):
        unit = _normalized_unit(match.group("unit"))
        try:
            measurement = _numeric_measurement(
                match.group("value"),
                relation=match.group("relation"),
                unit=unit,
            )
        except ValueError as error:
            # An inequality applied to a range has no unambiguous interval
            # contract in the source text, so this observation is skipped.
            if str(error) == "censor relation applied to a numeric range":
                continue
            raise
        normalized = convert_concentration_to_um(
            measurement,
            molecular_weight_da=weight,
        )
        strain, outer_condition = _target_context(target_text, match.start())
        conditions = tuple(
            dict.fromkeys(
                condition
                for condition in (outer_condition, _inline_condition(target_text, match.end()))
                if condition is not None
            )
        )
        assay = "DRAMP-curated MIC; primary assay method not normalized"
        if conditions:
            assay += f"; source condition: {' | '.join(conditions)}"
        observations.append(
            AssayObservation(
                endpoint="mic",
                value=normalized,
                provenance=provenance,
                strain=strain,
                gram=_gram_at(target_text, match.start()),
                assay=assay,
            )
        )
    return tuple(observations)


def ingest_dramp_general(
    path: str | Path,
    *,
    artifact: DataArtifact,
    strict: bool = False,
) -> IngestionResult:
    """Read challenge-compatible DRAMP MIC and numeric hemolysis rows.

    The source workbook stores activity as prose. Only explicit MIC, HC50, or
    percent-hemolysis-at-concentration values with supported units are accepted.
    Qualitative hemolysis claims do not become numeric labels. Ambiguous numeric
    hemolysis prose quarantines that complete source field while independent MIC
    observations remain eligible. Declared PTMs, terminal modifications,
    cyclization, oxidation/reduction state, or disulfide bonding still quarantine
    the complete source row because transfer to a free-terminus linear sequence
    would be chemically ambiguous.
    """

    try:
        from openpyxl import load_workbook
    except ImportError as error:  # pragma: no cover - environment integration guard
        raise RuntimeError("DRAMP XLSX ingestion requires the project 'data' extra") from error

    input_path = Path(path)
    workbook = load_workbook(input_path, read_only=True, data_only=True)
    if len(workbook.sheetnames) != 1:
        raise ValueError(f"expected one DRAMP worksheet, found {workbook.sheetnames}")
    worksheet = workbook[workbook.sheetnames[0]]
    rows: Iterable[tuple[object, ...]] = worksheet.iter_rows(values_only=True)
    try:
        header = tuple(_cell(value) for value in next(iter(rows)))
    except StopIteration as error:
        raise ValueError("DRAMP workbook is empty") from error
    if len(header) != len(set(header)):
        raise ValueError("DRAMP workbook has duplicate column names")
    missing = _REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"DRAMP workbook missing column(s): {sorted(missing)}")

    records: list[PeptideRecord] = []
    rejected: list[RejectedRecord] = []
    rejected_endpoints: list[RejectedEndpoint] = []
    for row_number, values in enumerate(rows, start=2):
        row = {name: _cell(value) for name, value in zip(header, values, strict=True)}
        sequence = row.get("Sequence", "")
        provenance_extra = {
            **row,
            "source_repository": artifact.source_repository,
            "source_version": artifact.source_commit,
            "source_sha256": artifact.sha256,
            "source_license": artifact.license,
            "training_status": artifact.training_status,
        }
        provenance = Provenance.from_mapping(
            source=artifact.name,
            record_id=row.get("DRAMP_ID") or None,
            path=str(input_path),
            row_number=row_number,
            extra=provenance_extra,
        )
        try:
            base = PeptideRecord.from_sequence(sequence, provenance=provenance)
            exclusion = _chemistry_exclusion(row)
            if exclusion is not None:
                raise ValueError(f"chemistry quarantine: {exclusion}")
            mic_observations = _mic_observations(
                row.get("Target_Organism", ""),
                sequence=base.sequence,
                provenance=provenance,
            )
            try:
                hemolysis_observations = _hemolysis_observations(
                    row.get("Hemolytic_activity", ""),
                    sequence=base.sequence,
                    provenance=provenance,
                )
            except HemolysisEndpointQuarantine as error:
                if strict:
                    raise
                source_text = row.get("Hemolytic_activity", "").strip()
                endpoint_provenance = Provenance.from_mapping(
                    source=artifact.name,
                    record_id=row.get("DRAMP_ID") or None,
                    path=str(input_path),
                    row_number=row_number,
                    extra=provenance_extra,
                    # This separate ledger provenance is an exact source-row
                    # audit trail. Accepted assay bytes remain compatible with
                    # v5 while blank workbook cells stay explicit here.
                    include_empty=True,
                )
                rejected_endpoints.append(
                    RejectedEndpoint(
                        rejection_id=endpoint_rejection_id(
                            provenance=endpoint_provenance,
                            endpoint_family="hemolysis",
                            source_field="Hemolytic_activity",
                            sequence_id=base.sequence_id,
                        ),
                        scope="endpoint_field",
                        endpoint_family="hemolysis",
                        source_field="Hemolytic_activity",
                        sequence_id=base.sequence_id,
                        sequence=base.sequence,
                        raw_sequence=sequence or None,
                        source_text=source_text,
                        reason_code=error.reason_code,
                        reason_detail=error.reason_detail,
                        discarded_candidate_observations=(error.discarded_candidate_observations),
                        provenance=endpoint_provenance,
                    )
                )
                hemolysis_observations = ()
            observations = (
                *mic_observations,
                *hemolysis_observations,
            )
            if not observations:
                raise ValueError(
                    "no explicit MIC, HC50, or concentration-specific hemolysis "
                    "with supported units"
                )
            records.append(
                PeptideRecord(
                    sequence_id=base.sequence_id,
                    sequence=base.sequence,
                    provenance=base.provenance,
                    assays=observations,
                )
            )
        except (TypeError, ValueError, SequenceValidationError) as error:
            if strict:
                raise ValueError(f"{input_path}:{row_number}: {error}") from error
            rejected.append(
                RejectedRecord(
                    source=artifact.name,
                    path=str(input_path),
                    row_number=row_number,
                    raw_sequence=sequence or None,
                    reason=str(error),
                )
            )
    workbook.close()
    return IngestionResult(
        records=deduplicate_records(records),
        rejected=tuple(rejected),
        rejected_endpoints=tuple(sorted(rejected_endpoints, key=lambda item: item.rejection_id)),
    )


def prepare_dramp(
    *,
    manifest_path: str | Path,
    snapshot_root: str | Path,
    output_dir: str | Path,
    strict: bool = False,
) -> dict[str, object]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_path}")
    manifest = load_data_manifest(manifest_path)
    artifacts = [artifact for artifact in manifest.artifacts if artifact.name == "dramp_general_v2"]
    if len(artifacts) != 1:
        raise ValueError("manifest must contain exactly one dramp_general_v2 artifact")
    artifact = artifacts[0]
    if artifact.training_status != "approved":
        raise ValueError("dramp_general_v2 is not approved in the selected manifest")
    source_path = Path(snapshot_root) / artifact.relative_path
    verify_artifact(source_path, artifact)
    result = ingest_dramp_general(source_path, artifact=artifact, strict=strict)
    summary = write_normalized_dataset(
        output_path,
        result=result,
        artifacts=(artifact,),
        schema_version=2,
        expected_source_rows=_EXPECTED_SOURCE_ROWS,
    )
    summary["parser_id"] = "amp_challenge.data.dramp:v7"
    summary["chemistry_policy"] = (
        "challenge-length canonical sequences; quarantine explicit PTM, terminal "
        "modification, cyclization, cyclotide, disulfide, and assay-specific "
        "oxidation/reduction markers"
    )
    summary["assay_policy"] = (
        "explicit MIC or HC50 with supported concentration units, plus explicit "
        "percent hemolysis at a supported exposure concentration; normalize "
        "concentrations to uM using free-termini sequence molecular weight; retain "
        "source units, censoring, ranges, blood-cell context, and raw source text; "
        "quarantine mean-plus/minus uncertainty until the normalized schema can "
        "represent its semantics; quarantine source-specific MHC, RCH, LD, IC, "
        "ED, EC, lethal-concentration, and hemolytic-unit endpoint aliases rather "
        "than coercing them into HC50 or percent hemolysis; "
        "quarantine an ambiguous hemolysis field without discarding an independent "
        "valid MIC; never map qualitative non-hemolytic prose to a numeric label"
    )
    endpoint_counts: dict[str, int] = {}
    for record in result.records:
        for observation in record.assays:
            endpoint_counts[observation.endpoint] = endpoint_counts.get(observation.endpoint, 0) + 1
    summary["endpoint_counts"] = dict(sorted(endpoint_counts.items()))
    reasons: dict[str, int] = {}
    for rejected in result.rejected:
        reasons[rejected.reason] = reasons.get(rejected.reason, 0) + 1
    summary["rejection_reasons"] = dict(sorted(reasons.items()))
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data/public_snapshots.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_dramp(
        manifest_path=args.manifest,
        snapshot_root=args.snapshot_root,
        output_dir=args.output_dir,
        strict=args.strict,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Shared constants for competition-compatible linear peptide sequences.

The values here intentionally encode only constraints that can be checked from a
plain sequence string.  Terminal chemistry and other modifications must be
validated from source metadata during ingestion.
"""

from __future__ import annotations

STANDARD_AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AMINO_ACID_SET: frozenset[str] = frozenset(STANDARD_AMINO_ACIDS)

MIN_SEQUENCE_LENGTH = 8
MAX_SEQUENCE_LENGTH = 50
SHA256_HEX_LENGTH = 64

# Approximate average residue masses in a peptide chain, in Da.  Add one water
# molecule for a linear peptide with free N/C termini.
RESIDUE_MASSES_DA: dict[str, float] = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 131.1926,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
WATER_MASS_DA = 18.01528

# Eisenberg consensus hydrophobicity scale.  It is useful for comparisons but
# is not a substitute for an experimentally calibrated solubility model.
EISENBERG_HYDROPHOBICITY: dict[str, float] = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

HYDROPHOBIC_RESIDUES: frozenset[str] = frozenset("ACFILMVWY")
AROMATIC_RESIDUES: frozenset[str] = frozenset("FWY")
BASIC_RESIDUES: frozenset[str] = frozenset("HKR")
ACIDIC_RESIDUES: frozenset[str] = frozenset("DE")

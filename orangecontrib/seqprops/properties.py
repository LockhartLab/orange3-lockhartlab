"""Physicochemical properties of protein and DNA/RNA sequences (via Biopython)."""

import numpy as np
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils.ProtParam import ProteinAnalysis

PROTEIN, NUCLEOTIDE = "protein", "nucleotide"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUCLEOTIDE_LETTERS = set("ACGTUN")
SEQUENCE_LETTERS = set(AMINO_ACIDS) | set("BUXZN")

# Feature groups: PROPERTIES exists for both kinds, SKEW only for DNA/RNA.
PROPERTIES, SKEW = "properties", "skew"

FEATURES = {
    PROTEIN: {
        PROPERTIES: [
            "length", "molecular weight", "isoelectric point", "charge at pH 7",
            "GRAVY (hydropathy)", "aromaticity", "instability index",
            "helix fraction", "turn fraction", "sheet fraction",
            "extinction coeff (reduced)",
        ],
    },
    NUCLEOTIDE: {
        PROPERTIES: ["length", "GC %", "molecular weight", "melting temp (C)"],
        SKEW: ["GC skew", "AT skew", "CpG obs/exp"],
    },
}


def clean(seq):
    """Upper-case, drop whitespace and stop symbols; missing values become ''."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    return "".join(str(seq).upper().split()).replace("*", "")


def looks_like_sequence(seq):
    return bool(seq) and set(seq) <= SEQUENCE_LETTERS


def detect_type(sequences):
    letters = set("".join(sequences))
    if letters and letters <= NUCLEOTIDE_LETTERS:
        return NUCLEOTIDE
    return PROTEIN


def feature_names(kind, groups=(PROPERTIES, SKEW)):
    """Names of the requested groups, in a fixed order; groups that do not
    apply to `kind` are skipped."""
    return [name for group, names in FEATURES[kind].items() if group in groups
            for name in names]


# --- protein -----------------------------------------------------------------

def _protein_properties(seq, raw):
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    return [len(seq), pa.molecular_weight(), pa.isoelectric_point(),
            pa.charge_at_pH(7.0), pa.gravy(), pa.aromaticity(),
            pa.instability_index(), helix, turn, sheet,
            pa.molar_extinction_coefficient()[0]]


# --- DNA / RNA ---------------------------------------------------------------

def _nucleotide_properties(seq, raw):
    is_rna = "U" in seq and "T" not in seq
    try:
        tm = mt.Tm_NN(seq.replace("U", "T"))
    except Exception:  # sequences too short or unusual for the NN tables
        tm = np.nan
    return [len(seq), 100 * gc_fraction(seq),
            molecular_weight(seq, seq_type="RNA" if is_rna else "DNA"), tm]


def _ratio(a, b):
    return a / b if b else np.nan


def _skew(seq, raw):
    seq = seq.replace("U", "T")
    a, c, g, t = (seq.count(base) for base in "ACGT")
    # Observed/expected CpG (Gardiner-Garden & Frommer 1987); ~1 when CG
    # occurs as often as the C and G content predicts, low in CpG-depleted DNA.
    cpg = _ratio(seq.count("CG") * len(seq), c * g)
    return [_ratio(g - c, g + c), _ratio(a - t, a + t), cpg]


VALID_LETTERS = {PROTEIN: AMINO_ACIDS, NUCLEOTIDE: "ACGTU"}
FUNCTIONS = {
    PROTEIN: {PROPERTIES: _protein_properties},
    NUCLEOTIDE: {PROPERTIES: _nucleotide_properties,
                 SKEW: _skew},
}


def compute(sequences, kind, groups=(PROPERTIES, SKEW)):
    """Return (names, X, n_invalid) for already cleaned sequences.

    Letters that are not standard residues/bases (X, B, Z, N, ...) are
    ignored; sequences with no valid letters get missing values.
    """
    groups = [g for g in FEATURES[kind] if g in groups]
    names = feature_names(kind, groups)
    valid = VALID_LETTERS[kind]

    rows, n_invalid = [], 0
    for raw in sequences:
        seq = "".join(c for c in raw if c in valid)
        if not seq:
            n_invalid += 1
            rows.append([np.nan] * len(names))
            continue
        rows.append([value for g in groups for value in FUNCTIONS[kind][g](seq, raw)])
    X = np.array(rows, dtype=float).reshape(len(sequences), len(names))
    return names, X, n_invalid

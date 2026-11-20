"""K-mer bag of words: one column per k-mer, with its count or frequency.

Any letters are counted, so the same code works for protein, DNA and RNA.
The k-mer columns know how to compute themselves (Orange's compute_value),
so data that arrives later, such as a test set sent to Predictions, gets
the same columns, computed from its own sequence column.
"""

from collections import Counter
from itertools import product

import numpy as np

from Orange.data import ContinuousVariable, Domain
from Orange.data.util import SharedComputeValue, get_unique_names

from orangecontrib.lockhartlab import properties as sp


def sequence_values(data, var):
    """Cleaned sequences from column `var` of `data`."""
    column = data.get_column(var)
    if var.is_discrete:
        column = [var.values[int(v)] if not np.isnan(v) else "" for v in column]
    return [sp.clean(s) for s in column]


def count_kmers(seq, k):
    """Counts of the overlapping k-mers of `seq`."""
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def kmer_matrix(sequences, kmers, frequencies):
    """Counts of `kmers` in each sequence. With `frequencies`, counts are
    divided by the number of k-mers of the same length in the sequence."""
    index = {kmer: j for j, kmer in enumerate(kmers)}
    ks = sorted({len(kmer) for kmer in kmers})
    X = np.zeros((len(sequences), len(kmers)))
    for i, seq in enumerate(sequences):
        for k in ks:
            for kmer, n in count_kmers(seq, k).items():
                if kmer in index:
                    X[i, index[kmer]] = n
    if frequencies:
        for k in ks:
            cols = [j for j, kmer in enumerate(kmers) if len(kmer) == k]
            totals = np.array([max(len(s) - k + 1, 0) for s in sequences], dtype=float)
            totals[totals == 0] = np.nan
            X[:, cols] /= totals[:, None]
    return X


class SequenceTable:
    """Computes many columns at once from one sequence column.

    Orange shares the result among the columns within one transformation,
    but learners that wrap columns again (SVM normalizes them) make Orange
    call this once per column, each time with a new intermediate table.
    So the last result is remembered and reused when the sequences match.
    """

    def __init__(self, sequence_var):
        self.sequence_var = sequence_var
        self._last = None
        self._hash = None

    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self is other or (type(self) is type(other)
                                 and self._key() == other._key())

    def __hash__(self):
        # Orange hashes variables (and so this) very often; with thousands
        # of k-mers in the key, hashing it each time dominates run time.
        if getattr(self, "_hash", None) is None:
            self._hash = hash(self._key())
        return self._hash

    def input_var(self, data):
        """The column of `data` to read the sequences from."""
        return self.sequence_var

    def compute(self, sequences):
        raise NotImplementedError

    def __call__(self, data):
        var = self.input_var(data)
        column = np.asarray(data.get_column(var))
        vocabulary = tuple(var.values) if var.is_discrete else None
        last = getattr(self, "_last", None)
        if (last is not None and last[0] == vocabulary
                and _same_column(last[1], column)):
            return last[2]
        result = self.compute(sequence_values(data, var))
        self._last = (vocabulary, column.copy(), result)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_last", None)  # caches; never save them with models
        state.pop("_hash", None)  # string hashes differ between sessions
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._last = None
        self._hash = None


def _same_column(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return bool(np.all(a == b))


class KmerTable(SequenceTable):
    """Computes all k-mer columns at once (shared by the k-mer variables)."""

    InheritEq = True  # equality and hashing come from SequenceTable

    def __init__(self, sequence_var, kmers, frequencies):
        super().__init__(sequence_var)
        self.kmers = tuple(kmers)
        self.frequencies = frequencies

    def compute(self, sequences):
        return kmer_matrix(sequences, self.kmers, self.frequencies)

    def _key(self):
        return self.sequence_var, self.kmers, self.frequencies


class KmerColumn(SharedComputeValue):
    def __init__(self, table, index):
        super().__init__(table)
        self.index = index

    def compute(self, data, shared_data):
        return shared_data[:, self.index]

    def __eq__(self, other):
        return super().__eq__(other) and self.index == other.index

    def __hash__(self):
        return hash((super().__hash__(), self.index))


def observed_kmers(sequences, k_from, k_to):
    """The k-mers that occur in `sequences`, shortest first, then A-Z."""
    return [kmer for k in range(k_from, k_to + 1)
            for kmer in sorted(set().union(*(count_kmers(s, k) for s in sequences)))]


# Listing every possible k-mer above this many columns is refused.
MAX_ALL_KMERS = 100_000


def alphabet(sequences):
    """The standard letters of the sequences: the 20 amino acids for
    proteins, ACGT for DNA, or ACGU for RNA (U and no T)."""
    if sp.detect_type(sequences) == sp.PROTEIN:
        return sp.AMINO_ACIDS
    letters = set("".join(sequences))
    return "ACGU" if "U" in letters and "T" not in letters else "ACGT"


def n_possible_kmers(letters, k_from, k_to):
    return sum(len(letters) ** k for k in range(k_from, k_to + 1))


def kmer_vocabulary(sequences, k_from, k_to, include_absent=False):
    """The k-mer columns, shortest first, then A-Z. Normally the k-mers that
    occur in `sequences`; with `include_absent`, every k-mer of the standard
    alphabet as well, so the columns do not depend on which sequences are
    in the data. K-mers with other letters (such as X) that occur are kept."""
    observed = observed_kmers(sequences, k_from, k_to)
    if not include_absent:
        return observed
    letters = alphabet(sequences)
    n = n_possible_kmers(letters, k_from, k_to)
    if n > MAX_ALL_KMERS:
        raise ValueError(f"all k-mers would make {n:,} columns; "
                         f"use a smaller k (at most {MAX_ALL_KMERS:,}).")
    kmers = set(observed)
    for k in range(k_from, k_to + 1):
        kmers.update("".join(p) for p in product(letters, repeat=k))
    return sorted(kmers, key=lambda kmer: (len(kmer), kmer))


def kmer_variables(sequence_var, kmers, frequencies, domain):
    """One variable per k-mer; names that clash with `domain` get a suffix."""
    table = KmerTable(sequence_var, kmers, frequencies)
    names = get_unique_names(domain, list(kmers))
    return [ContinuousVariable(name, compute_value=KmerColumn(table, j))
            for j, name in enumerate(names)]


def add_kmer_columns(data, sequence_var, k_from, k_to, frequencies,
                     include_absent=False):
    """`data` with one column added per k-mer (see `kmer_vocabulary`)."""
    kmers = kmer_vocabulary(sequence_values(data, sequence_var), k_from, k_to,
                            include_absent)
    domain = data.domain
    new_vars = kmer_variables(sequence_var, kmers, frequencies, domain)
    out = data.transform(Domain(domain.attributes + tuple(new_vars),
                                domain.class_vars, domain.metas))
    out.name = data.name
    return out

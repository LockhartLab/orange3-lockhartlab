"""K-mer bag of words: one column per k-mer, with its count or frequency.

Any letters are counted, so the same code works for protein, DNA and RNA.
The k-mer columns know how to compute themselves (Orange's compute_value),
so data that arrives later, such as a test set sent to Predictions, gets
the same columns, computed from its own sequence column.
"""

from collections import Counter

import numpy as np

from Orange.data import ContinuousVariable
from Orange.data.util import SharedComputeValue, get_unique_names

from orangecontrib.seqprops import properties as sp


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


class KmerTable:
    """Computes all k-mer columns at once (shared by the k-mer variables)."""

    def __init__(self, sequence_var, kmers, frequencies):
        self.sequence_var = sequence_var
        self.kmers = tuple(kmers)
        self.frequencies = frequencies

    def __call__(self, data):
        return kmer_matrix(sequence_values(data, self.sequence_var),
                           self.kmers, self.frequencies)

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

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


def kmer_variables(sequence_var, kmers, frequencies, domain):
    """One variable per k-mer; names that clash with `domain` get a suffix."""
    table = KmerTable(sequence_var, kmers, frequencies)
    names = get_unique_names(domain, list(kmers))
    return [ContinuousVariable(name, compute_value=KmerColumn(table, j))
            for j, name in enumerate(names)]

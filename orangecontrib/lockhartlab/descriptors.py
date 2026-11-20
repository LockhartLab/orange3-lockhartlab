"""Sequence descriptors that can recompute themselves for prediction data."""

from Orange.data import ContinuousVariable
from Orange.data.util import SharedComputeValue, get_unique_names

from orangecontrib.lockhartlab import properties as sp
from orangecontrib.lockhartlab.kmers import SequenceTable


class DescriptorTable(SequenceTable):
    """Compute all descriptors together, once per transformed table."""

    InheritEq = True  # equality and hashing come from SequenceTable

    def __init__(self, sequence_var, kind):
        super().__init__(sequence_var)
        self.kind = kind

    def input_var(self, data):
        # Use the incoming vocabulary before unseen categorical sequences
        # are mapped to missing values in the training variable's vocabulary.
        try:
            return data.domain[self.sequence_var.name]
        except KeyError:
            return self.sequence_var

    def compute(self, sequences):
        return sp.compute(sequences, self.kind)[1]

    def _key(self):
        return self.sequence_var, self.kind


class DescriptorColumn(SharedComputeValue):
    def __init__(self, table, index):
        super().__init__(table)
        self.index = index

    def compute(self, data, shared_data):
        return shared_data[:, self.index]

    def __eq__(self, other):
        return super().__eq__(other) and self.index == other.index

    def __hash__(self):
        return hash((super().__hash__(), self.index))


def descriptor_variables(sequence_var, kind, domain):
    """Descriptor columns with reusable calculations and unique names."""
    table = DescriptorTable(sequence_var, kind)
    names = get_unique_names(domain, sp.feature_names(kind))
    return [ContinuousVariable(name, compute_value=DescriptorColumn(table, j))
            for j, name in enumerate(names)]

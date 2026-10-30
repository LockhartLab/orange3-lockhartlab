import numpy as np

from Orange.data import Domain, DiscreteVariable, StringVariable, Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.seqprops.widgets.owsequenceproperties import (
    OWSequenceProperties, PROTEIN)


def make_table(seqs, name="sequence", as_meta=True):
    if as_meta:
        domain = Domain([], DiscreteVariable("class", values=("a", "b")),
                        [StringVariable(name)])
        rows = [[["a", "b"][i % 2], s] for i, s in enumerate(seqs)]
    else:
        domain = Domain([DiscreteVariable(name, values=tuple(sorted(set(seqs))))])
        rows = [[s] for s in seqs]
    return Table.from_list(domain, rows)


class TestOWSequenceProperties(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSequenceProperties)

    def test_protein(self):
        data = make_table(["MKTAYIAKQRQISFVKSHFSRQ", "gigkflhsakkfgkafvgeimns*"])
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.data)
        names = [v.name for v in out.domain.attributes]
        self.assertEqual(len(names), 11)
        self.assertEqual(out.get_column("length").tolist(), [22, 23])
        self.assertEqual(out.domain.class_var, data.domain.class_var)
        self.assertEqual(out.domain.metas, data.domain.metas)

    def test_nucleotide_and_options(self):
        data = make_table(["ATGCGCGATAT", "AUGGCCAUUGUAAUG"])
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual([v.name for v in out.domain.attributes][:2], ["length", "GC %"])
        self.assertEqual(len(out.domain.attributes), 7)
        self.assertTrue(self.widget.dna_box.isEnabled())

        self.widget.include_skew = False
        self.widget.commit.now()
        self.assertEqual(len(self.get_output(self.widget.Outputs.data).domain.attributes), 4)

        # DNA-only groups are ignored (and their box disabled) for proteins
        self.widget.include_skew = True
        self.widget.sequence_type = PROTEIN
        self.widget.commit.now()
        self.assertEqual(len(self.get_output(self.widget.Outputs.data).domain.attributes), 11)
        self.assertFalse(self.widget.dna_box.isEnabled())

    def test_invalid_and_categorical_column(self):
        data = make_table(["ACDK", "XXXX", "MKV"], name="peptide", as_meta=False)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.invalid.is_shown())
        out = self.get_output(self.widget.Outputs.data)
        self.assertTrue(np.isnan(out.get_column("length")[1]))

    def test_no_sequence_column(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.assertTrue(self.widget.Error.no_sequence_column.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        # a categorical class with word-like values is not a sequence column
        self.send_signal(self.widget.Inputs.data,
                         make_table(["toxin", "amp"], name="kind", as_meta=False))
        self.assertTrue(self.widget.Error.no_sequence_column.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_sequence_column.is_shown())

    def test_name_clash(self):
        data = make_table(["MKV", "ACD"])
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.data)
        self.send_signal(self.widget.Inputs.data, out)
        names = [v.name for v in self.get_output(self.widget.Outputs.data).domain.attributes]
        self.assertEqual(len(names), len(set(names)))

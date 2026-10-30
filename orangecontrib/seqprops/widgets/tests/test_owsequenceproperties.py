from unittest.mock import patch

import numpy as np

from Orange.data import Domain, DiscreteVariable, StringVariable, Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.seqprops.widgets.owsequenceproperties import (
    OWSequenceProperties, NOT_CHOSEN, PROTEIN, NUCLEOTIDE)

# Protein with threonine (T) and selenocysteine (U): read as DNA, it used to
# crash Biopython ("'U' is not a valid unambiguous letter for DNA").
SELENOPROTEIN = "MKTAYIAKQRQISFVKSHFSRQUGCRVLTA"


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

    def choose(self, sequence_type, widget=None):
        """Click a sequence-type button, as a student would."""
        widget = widget or self.widget
        widget.type_buttons.button(sequence_type).click()

    def output(self):
        return self.get_output(self.widget.Outputs.data)

    def test_nothing_selected_at_first(self):
        self.assertEqual(self.widget.sequence_type, NOT_CHOSEN)
        self.assertFalse(any(b.isChecked() for b in self.widget.type_buttons.buttons()))
        self.send_signal(self.widget.Inputs.data, make_table(["MKV", "ACD"]))
        self.assertIsNone(self.output())
        self.assertTrue(self.widget.Information.choose_type.is_shown())

        self.choose(PROTEIN)
        self.assertFalse(self.widget.Information.choose_type.is_shown())
        self.assertIsNotNone(self.output())

    def test_choice_saved_in_workflow_but_not_default_for_new_widgets(self):
        self.choose(NUCLEOTIDE)
        # saved in the workflow file: reopening the workflow keeps it
        stored = self.widget.settingsHandler.pack_data(self.widget)
        reopened = self.create_widget(OWSequenceProperties, stored_settings=stored)
        self.assertEqual(reopened.sequence_type, NUCLEOTIDE)
        self.assertTrue(reopened.type_buttons.button(NUCLEOTIDE).isChecked())
        # ...but a widget newly dragged onto the canvas still asks
        self.widget.settingsHandler.update_defaults(self.widget)
        fresh = self.create_widget(OWSequenceProperties)
        self.assertEqual(fresh.sequence_type, NOT_CHOSEN)

    def test_migrate_old_workflows(self):
        for old, new in ((0, NOT_CHOSEN), (1, PROTEIN), (2, NUCLEOTIDE)):
            w = self.create_widget(
                OWSequenceProperties,
                stored_settings={"sequence_type": old, "__version__": 1})
            self.assertEqual(w.sequence_type, new, f"old value {old}")

    def test_protein(self):
        data = make_table(["MKTAYIAKQRQISFVKSHFSRQ", "gigkflhsakkfgkafvgeimns*"])
        self.send_signal(self.widget.Inputs.data, data)
        self.choose(PROTEIN)
        out = self.output()
        self.assertEqual(len(out.domain.attributes), 11)
        self.assertEqual(out.get_column("length").tolist(), [22, 23])
        self.assertEqual(out.domain.class_var, data.domain.class_var)
        self.assertEqual(out.domain.metas, data.domain.metas)
        self.assertFalse(self.widget.dna_box.isEnabled())

    def test_nucleotide_and_options(self):
        data = make_table(["ATGCGCGATAT", "AUGGCCAUUGUAAUG"])
        self.send_signal(self.widget.Inputs.data, data)
        self.choose(NUCLEOTIDE)
        out = self.output()
        self.assertEqual([v.name for v in out.domain.attributes][:2], ["length", "GC %"])
        self.assertEqual(len(out.domain.attributes), 7)
        self.assertTrue(self.widget.dna_box.isEnabled())

        self.widget.controls.include_skew.click()
        self.widget.commit.now()
        self.assertEqual(len(self.output().domain.attributes), 4)

    def test_protein_read_as_dna_is_an_error_not_a_crash(self):
        """The bug report: protein data with DNA / RNA selected."""
        self.send_signal(self.widget.Inputs.data,
                         make_table([SELENOPROTEIN, "MKVLAAGIVGLLLA"]))
        self.choose(NUCLEOTIDE)
        self.assertTrue(self.widget.Error.protein_as_dna.is_shown())
        self.assertFalse(self.widget.Error.unexpected.is_shown())
        self.assertIsNone(self.output())
        # switching to Protein fixes it
        self.choose(PROTEIN)
        self.assertFalse(self.widget.Error.protein_as_dna.is_shown())
        self.assertEqual(len(self.output()), 2)

    def test_dna_read_as_protein_warns(self):
        self.send_signal(self.widget.Inputs.data, make_table(["ATGCGCGATAT", "GGCCAATT"]))
        self.choose(PROTEIN)
        self.assertTrue(self.widget.Warning.dna_as_protein.is_shown())
        self.assertIsNotNone(self.output())

    def test_one_bad_sequence_does_not_stop_the_rest(self):
        # row 2 mixes T and U, which is neither DNA nor RNA
        self.send_signal(self.widget.Inputs.data,
                         make_table(["ATGCGCGATAT", "ACGTUACGU", "GGCCAATT"]))
        self.choose(NUCLEOTIDE)
        out = self.output()
        self.assertTrue(self.widget.Warning.failed.is_shown())
        self.assertIn("row 2", str(self.widget.Warning.failed))
        self.assertIn("both T and U", str(self.widget.Warning.failed))
        lengths = out.get_column("length")
        self.assertTrue(np.isnan(lengths[1]))
        self.assertEqual(lengths[[0, 2]].tolist(), [11, 8])

    def test_unexpected_errors_are_shown_not_raised(self):
        self.send_signal(self.widget.Inputs.data, make_table(["MKV", "ACD"]))
        with patch("orangecontrib.seqprops.properties.compute",
                   side_effect=RuntimeError("boom")):
            self.choose(PROTEIN)
        self.assertTrue(self.widget.Error.unexpected.is_shown())
        self.assertIn("boom", str(self.widget.Error.unexpected))
        self.assertIsNone(self.output())

    def test_empty_sequences_and_categorical_column(self):
        data = make_table(["ACDK", "XXXX", "MKV"], name="peptide", as_meta=False)
        self.send_signal(self.widget.Inputs.data, data)
        self.choose(PROTEIN)
        self.assertTrue(self.widget.Warning.empty.is_shown())
        self.assertIn("row 2", str(self.widget.Warning.empty))
        self.assertTrue(np.isnan(self.output().get_column("length")[1]))

    def test_no_sequence_column(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.assertTrue(self.widget.Error.no_sequence_column.is_shown())
        self.assertIsNone(self.output())
        # a categorical class with word-like values is not a sequence column
        self.send_signal(self.widget.Inputs.data,
                         make_table(["toxin", "amp"], name="kind", as_meta=False))
        self.assertTrue(self.widget.Error.no_sequence_column.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_sequence_column.is_shown())

    def test_name_clash(self):
        self.choose(PROTEIN)
        self.send_signal(self.widget.Inputs.data, make_table(["MKV", "ACD"]))
        self.send_signal(self.widget.Inputs.data, self.output())
        names = [v.name for v in self.output().domain.attributes]
        self.assertEqual(len(names), len(set(names)))

import unittest

import numpy as np

from Orange.data import Domain, DiscreteVariable, StringVariable, Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.seqprops import kmers as km
from orangecontrib.seqprops.widgets.owkmers import OWKmers


def make_table(seqs, name="sequence"):
    domain = Domain([], DiscreteVariable("class", values=("a", "b")),
                    [StringVariable(name)])
    return Table.from_list(domain, [[["a", "b"][i % 2], s] for i, s in enumerate(seqs)])


class TestKmers(unittest.TestCase):
    def test_count_kmers(self):
        self.assertEqual(km.count_kmers("ACGTAC", 2),
                         {"AC": 2, "CG": 1, "GT": 1, "TA": 1})
        self.assertEqual(km.count_kmers("AC", 3), {})

    def test_observed_kmers_sorted_by_length_then_name(self):
        self.assertEqual(km.observed_kmers(["ACA", "CC"], 1, 2),
                         ["A", "C", "AC", "CA", "CC"])

    def test_frequencies_are_per_k(self):
        X = km.kmer_matrix(["AACG"], ["A", "C", "AA", "CG"], frequencies=True)
        np.testing.assert_allclose(X, [[2 / 4, 1 / 4, 1 / 3, 1 / 3]])

    def test_short_sequence_frequencies_are_missing(self):
        X = km.kmer_matrix(["A"], ["A", "AC"], frequencies=True)
        self.assertEqual(X[0, 0], 1)
        self.assertTrue(np.isnan(X[0, 1]))


class TestOWKmers(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWKmers)

    def output(self):
        return self.get_output(self.widget.Outputs.data)

    def test_default_k_1_to_2_frequencies(self):
        data = make_table(["ACGT", "aagg"])
        self.send_signal(self.widget.Inputs.data, data)
        out = self.output()
        names = [v.name for v in out.domain.attributes]
        self.assertEqual(names[:4], ["A", "C", "G", "T"])
        self.assertIn("AG", names)
        self.assertAlmostEqual(out[1, "A"], 0.5)
        self.assertEqual(out.domain.class_var, data.domain.class_var)
        self.assertEqual(out.domain.metas, data.domain.metas)

    def test_counts_and_k_range(self):
        self.send_signal(self.widget.Inputs.data, make_table(["MKVLAAMKV", "GIGK"]))
        self.widget.controls.frequencies.buttons[0].click()
        self.widget.k_from, self.widget.k_to = 3, 3
        self.widget.commit.now()
        out = self.output()
        self.assertTrue(all(len(v.name) == 3 for v in out.domain.attributes))
        self.assertEqual(out[0, "MKV"], 2)
        self.assertEqual(out[1, "MKV"], 0)

    def test_k_range_stays_ordered(self):
        self.widget.controls.k_from.setValue(4)
        self.assertEqual(self.widget.k_to, 4)
        self.widget.controls.k_to.setValue(2)
        self.assertEqual(self.widget.k_from, 2)

    def test_new_data_gets_the_same_columns(self):
        """A test set (e.g. for Predictions) is transformed to the same k-mers."""
        self.send_signal(self.widget.Inputs.data, make_table(["ACGT", "AAGG"]))
        train = self.output()
        test = make_table(["GTTA", "CCXX"]).transform(train.domain)
        self.assertEqual(test.domain, train.domain)
        self.assertAlmostEqual(test[0, "T"], 0.5)
        self.assertAlmostEqual(test[0, "GT"], 1 / 3)
        self.assertEqual(test[1, "A"], 0)   # unseen letters just don't count

    def test_no_sequence_column(self):
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.assertTrue(self.widget.Error.no_sequence_column.is_shown())
        self.assertIsNone(self.output())

    def test_many_columns_warning(self):
        rng = np.random.default_rng(0)
        seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 300)) for _ in range(60)]
        self.send_signal(self.widget.Inputs.data, make_table(seqs))
        self.widget.k_from, self.widget.k_to = 4, 4
        self.widget.commit.now()
        self.assertTrue(self.widget.Warning.many_columns.is_shown())
        self.assertIsNotNone(self.output())

    def test_name_clash(self):
        data = make_table(["ACGT", "AAGG"])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.data, self.output())
        names = [v.name for v in self.output().domain.attributes]
        self.assertEqual(len(names), len(set(names)))

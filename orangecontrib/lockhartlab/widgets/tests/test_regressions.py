"""Integration regressions for derived sequence features and evaluation.

Run with QT_QPA_PLATFORM=offscreen python -m unittest \
    orangecontrib.lockhartlab.widgets.tests.test_regressions
"""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from Orange.classification import LogisticRegressionLearner, SVMLearner
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
from Orange.evaluation import Recall
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.lockhartlab import featuresets as fs, kmers as km, properties as sp
from orangecontrib.lockhartlab.descriptors import DescriptorTable, descriptor_variables
from orangecontrib.lockhartlab.widgets.owcomparefeaturesets import OWCompareFeatureSets
from orangecontrib.lockhartlab.widgets.owsequenceproperties import (
    OWSequenceProperties, PROTEIN, NUCLEOTIDE)


def sequence_table(sequences):
    domain = Domain([], DiscreteVariable("class", values=("a", "b")),
                    [StringVariable("sequence")])
    return Table.from_list(domain, [
        ["a" if i % 2 == 0 else "b", sequence]
        for i, sequence in enumerate(sequences)
    ])


def feature_tables(y):
    domain = Domain([ContinuousVariable("signal"), ContinuousVariable("index")],
                    DiscreteVariable("target", values=("a", "b")))
    source = Table.from_numpy(
        domain, np.column_stack([np.arange(len(y)) % 2, np.arange(len(y))]), y)
    return [source[:, ["signal", "target"]], source[:, ["index", "target"]]]


class TestSequencePropertiesRegressions(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSequenceProperties)
        self.widget.type_buttons.button(PROTEIN).click()

    def output_for(self, data):
        self.send_signal(self.widget.Inputs.data, data)
        return self.get_output(self.widget.Outputs.data)

    def test_new_sequences_recompute_descriptors_in_training_domain(self):
        """Raw prediction data must get descriptors from its own sequences."""
        training = self.output_for(sequence_table(["MKV", "ACDK"]))
        self.assertIsNotNone(training)

        raw_test = sequence_table(["MKTAYIAKQRQISFVKSHFSRQ",
                                   "GIGKFLHSAKKFGKAFVGEIMNS"])
        transformed = raw_test.transform(training.domain)

        np.testing.assert_array_equal(transformed.get_column("length"), [22, 23])
        self.assertTrue(np.isfinite(transformed.X).all(),
                        "Valid new sequences must not become missing descriptors")

    def test_nucleotide_sequences_recompute_in_training_domain(self):
        self.widget.type_buttons.button(NUCLEOTIDE).click()
        training = self.output_for(sequence_table(["ATGCGCGATAT", "AUGGCCAUUGUAAUG"]))

        transformed = sequence_table(["CCGG", "AAUU"]).transform(training.domain)

        np.testing.assert_array_equal(transformed.get_column("length"), [4, 4])
        np.testing.assert_array_equal(transformed.get_column("GC %"), [100, 0])

    def test_prediction_transform_survives_serialization(self):
        training = self.output_for(sequence_table(["MKV", "ACDK"]))
        restored_domain = pickle.loads(pickle.dumps(training.domain))

        transformed = sequence_table(["GIGK", "MKVLA"]).transform(restored_domain)

        np.testing.assert_array_equal(transformed.get_column("length"), [4, 5])
        self.assertTrue(np.isfinite(transformed.X).all())

    def test_new_categorical_sequences_recompute_descriptors(self):
        train_var = DiscreteVariable("peptide", values=("MKV", "ACDK"))
        training = self.output_for(Table.from_list(Domain([train_var]),
                                                  [["MKV"], ["ACDK"]]))
        test_var = DiscreteVariable("peptide", values=("GIGK", "MKVLA"))
        raw_test = Table.from_list(Domain([test_var]), [["GIGK"], ["MKVLA"]])

        transformed = raw_test.transform(training.domain)

        np.testing.assert_array_equal(transformed.get_column("length"), [4, 5])

    def test_preserves_row_ids_for_feature_set_comparison(self):
        """Adding descriptors must preserve the identity of existing rows."""
        source = sequence_table(["MKV", "ACDK", "MKVLA", "GIGK"])
        descriptors = self.output_for(source)
        self.assertIsNotNone(descriptors)

        # These feature sets contain the same observations and target values.
        self.assertIsNone(fs.rows_problem([source, descriptors]),
                          "Compare Feature Sets must accept a descriptor branch "
                          "of the same source data")
        np.testing.assert_array_equal(descriptors.ids, source.ids)

    def test_accepts_sparse_existing_features(self):
        """Descriptor extraction must work when incoming attributes are sparse."""
        source = sequence_table(["MKV", "ACDK"])
        domain = Domain([ContinuousVariable("existing feature")],
                        source.domain.class_vars, source.domain.metas)
        sparse_source = Table.from_numpy(
            domain, csr_matrix([[1.0], [0.0]]), source.Y, source.metas)

        out = self.output_for(sparse_source)

        self.assertIsNotNone(out, str(self.widget.Error.unexpected))
        self.assertFalse(self.widget.Error.unexpected.is_shown())
        np.testing.assert_array_equal(out.get_column("existing feature"), [1, 0])
        np.testing.assert_array_equal(out.get_column("length"), [3, 4])


class TestFeatureSetRegressions(unittest.TestCase):
    def test_unknown_targets_are_excluded_consistently_before_cross_validation(self):
        """All sets must evaluate the same labeled rows and produce Recall."""
        y = np.array([0, 1] * 6, dtype=float)
        y[0] = np.nan
        sets = feature_tables(y)

        results = fs.evaluate(sets, [LogisticRegressionLearner()], k=3)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(len(result.actual), 11,
                             "The row with an unknown target must not be evaluated")
            self.assertTrue(np.isfinite(result.actual).all())
            np.testing.assert_array_equal(
                np.bincount(result.actual.astype(int)), [5, 6])
            self.assertTrue(np.isfinite(Recall(result)).all())
            np.testing.assert_array_equal(result.row_indices, results[0].row_indices)
            np.testing.assert_array_equal(result.actual, results[0].actual)


class TestCompareFeatureSetsRegressions(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCompareFeatureSets)

    def send_all(self, y):
        for i, table in enumerate(feature_tables(y)):
            self.send_signal(self.widget.Inputs.data, table, i)
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner(), 0)

    def test_unknown_targets_are_filtered_and_reported(self):
        y = np.array([0, 1] * 6, dtype=float)
        y[0] = np.nan
        self.send_all(y)
        self.wait_until_finished(timeout=30000)

        result = self.get_output(self.widget.Outputs.evaluations_results)

        self.assertIsNotNone(result)
        self.assertTrue(self.widget.Warning.missing_targets.is_shown())
        self.assertEqual(len(result.actual), 11)
        self.assertTrue(np.isfinite(Recall(result)).all())

    def test_fold_count_uses_only_labeled_rows(self):
        for n_known in (4, 0):
            with self.subTest(n_known=n_known):
                y = np.array([0, 1] * 6, dtype=float)
                y[n_known:] = np.nan
                self.send_all(y)

                self.assertTrue(self.widget.Error.too_many_folds.is_shown())
                self.assertTrue(self.widget.Warning.missing_targets.is_shown())
                self.assertIsNone(self.get_output(self.widget.Outputs.evaluations_results))


class TestSharedSequenceComputation(unittest.TestCase):
    """SVM wraps every feature column again (normalization), which made
    Orange recompute all k-mers or properties once per column."""

    def setUp(self):
        letters = "ACDEFGHIKLMNPQRSTVWY"
        self.data = sequence_table(
            [letters[i % 7:] + letters[:i % 5] for i in range(30)])
        self.sequence_var = self.data.domain["sequence"]

    def features(self, variables):
        domain = self.data.domain
        return self.data.transform(
            Domain(tuple(variables), domain.class_vars, domain.metas))

    def assert_computed_once_by_svm(self, table_class, variables):
        training = self.features(variables)
        model = SVMLearner()(training)
        calls = []
        original = table_class.compute

        def counted(table, sequences):
            calls.append(len(sequences))
            return original(table, sequences)

        with patch.object(table_class, "compute", counted):
            predicted = model(self.data[:10])
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(predicted, model(training[:10]))

    def test_svm_predictions_count_kmers_once(self):
        sequences = km.sequence_values(self.data, self.sequence_var)
        kmers = km.observed_kmers(sequences, 1, 2)
        self.assert_computed_once_by_svm(
            km.KmerTable,
            km.kmer_variables(self.sequence_var, kmers, True, self.data.domain))

    def test_svm_predictions_compute_properties_once(self):
        self.assert_computed_once_by_svm(
            DescriptorTable,
            descriptor_variables(self.sequence_var, sp.PROTEIN, self.data.domain))

    def test_changed_sequences_are_recomputed(self):
        sequences = km.sequence_values(self.data, self.sequence_var)
        variables = km.kmer_variables(
            self.sequence_var, km.observed_kmers(sequences, 1, 1), False,
            self.data.domain)
        table = variables[0].compute_value.compute_shared
        first = table(self.data[:2])
        second = table(sequence_table(["AAAA", "CC"]))
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(
            second[:, variables.index(next(v for v in variables if v.name == "A"))],
            [4, 0])

    def test_saved_models_do_not_keep_the_cache(self):
        sequences = km.sequence_values(self.data, self.sequence_var)
        variables = km.kmer_variables(
            self.sequence_var, km.observed_kmers(sequences, 1, 1), True,
            self.data.domain)
        table = variables[0].compute_value.compute_shared
        table(self.data)
        restored = pickle.loads(pickle.dumps(table))
        self.assertIsNone(restored._last)
        self.assertEqual(restored, table)
        self.assertEqual(hash(restored), hash(table))


if __name__ == "__main__":
    unittest.main()

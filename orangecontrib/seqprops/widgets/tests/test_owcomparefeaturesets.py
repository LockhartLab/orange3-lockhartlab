import unittest

import numpy as np

from Orange.classification import KNNLearner, LogisticRegressionLearner, TreeLearner
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, Table
from Orange.evaluation import AUC, CrossValidation, Recall
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.seqprops import featuresets as fs
from orangecontrib.seqprops.balance import Balance, UP
from orangecontrib.seqprops.widgets.owcomparefeaturesets import OWCompareFeatureSets


def data(seed=0):
    """Class 'a' is told apart by x1 only; x2 is noise; 30 a / 90 b."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * 30 + [1] * 90, dtype=float)
    X = np.column_stack([rng.normal(0, 1, 120) + 1.5 * (y == 0), rng.normal(0, 1, 120)])
    domain = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")],
                    DiscreteVariable("y", values=("a", "b")))
    return Table.from_numpy(domain, X, y)


def view(table, columns, name):
    out = table[:, columns + ["y"]]
    out.name = name
    return out


def named(learner, name):
    learner.name = name
    return learner


class TestFeatureSets(unittest.TestCase):
    def test_same_scores_as_cross_validation(self):
        """Each set scores exactly as Test & Score's 5-fold CV would."""
        d = data()
        sets = [view(d, ["x1"], "x1"), view(d, ["x2"], "x2")]
        results = fs.evaluate(sets, [LogisticRegressionLearner()])
        for table, res in zip(sets, results):
            alone = CrossValidation(k=5, stratified=True, random_state=42)(
                table, [LogisticRegressionLearner()])
            np.testing.assert_array_equal(res.predicted, alone.predicted)

    def test_combine(self):
        d = data()
        sets = [view(d, ["x1"], "x1"), view(d, ["x2"], "x2")]
        results = fs.evaluate(sets, [LogisticRegressionLearner(), KNNLearner()])
        out = fs.combine(results, ["x1", "x2"], ["LR", "kNN"])
        self.assertEqual(out.learner_names, ["LR [x1]", "kNN [x1]", "LR [x2]", "kNN [x2]"])
        self.assertEqual(out.predicted.shape, (4, 120))
        self.assertEqual(out.probabilities.shape, (4, 120, 2))
        auc = AUC(out)
        self.assertGreater(auc[0], auc[2] + 0.2)   # x1 is informative, x2 is not

    def test_rows_problem(self):
        d = data()
        self.assertIsNone(fs.rows_problem([view(d, ["x1"], "a"), view(d, ["x2"], "b")]))
        self.assertIsNotNone(fs.rows_problem([d, d[:100]]))
        self.assertIsNotNone(fs.rows_problem([d, d[::-1]]))
        self.assertIsNotNone(fs.rows_problem([Table.from_table(
            Domain(d.domain.attributes), d)]))


class TestOWCompareFeatureSets(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCompareFeatureSets)
        d = data()
        self.sets = [view(d, ["x1"], "x1 only"), view(d, ["x2"], "x2 only"),
                     view(d, ["x1", "x2"], "both")]

    def run_widget(self):
        self.wait_until_finished(timeout=30000)
        return self.get_output(self.widget.Outputs.evaluations_results)

    def send_all(self, learners=None):
        for i, table in enumerate(self.sets):
            self.send_signal(self.widget.Inputs.data, table, i)
        for i, learner in enumerate(learners or [named(LogisticRegressionLearner(), "LR"),
                                                 named(TreeLearner(), "Tree")]):
            self.send_signal(self.widget.Inputs.learner, learner, i)

    def rows(self):
        model = self.widget.score_table.model
        return [model.item(r, 0).text() for r in range(model.rowCount())]

    def test_every_model_on_every_set(self):
        self.send_all()
        res = self.run_widget()
        self.assertEqual(res.learner_names, [
            "LR [x1 only]", "Tree [x1 only]", "LR [x2 only]", "Tree [x2 only]",
            "LR [both]", "Tree [both]"])
        self.assertEqual(self.rows(), res.learner_names)
        self.assertEqual(len(res.actual), 120)
        auc = AUC(res)
        self.assertGreater(auc[0], auc[2] + 0.2)

    def test_scores_in_table_for_target(self):
        self.send_all()
        self.run_widget()
        self.widget.class_selection = "a"
        self.widget._update_table()
        model = self.widget.score_table.model
        header = [model.horizontalHeaderItem(c).text() for c in range(model.columnCount())]
        recall = model.item(0, header.index("Recall")).data(0)
        alone = CrossValidation(k=5, stratified=True, random_state=42)(
            self.sets[0], [LogisticRegressionLearner()])
        self.assertAlmostEqual(recall, Recall(alone, target=0)[0])

    def test_preprocessor_balances_training(self):
        self.send_all([named(LogisticRegressionLearner(), "LR")])
        plain = Recall(self.run_widget(), target=0)
        self.send_signal(self.widget.Inputs.preprocessor, Balance(UP))
        balanced = Recall(self.run_widget(), target=0)
        self.assertGreater(balanced[0], plain[0] + 0.2)

    def test_different_rows_is_an_error(self):
        self.send_all()
        self.send_signal(self.widget.Inputs.data, self.sets[0][:100], 0)
        self.assertTrue(self.widget.Error.rows.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.evaluations_results))
        self.send_signal(self.widget.Inputs.data, self.sets[0], 0)
        self.assertFalse(self.widget.Error.rows.is_shown())
        self.assertIsNotNone(self.run_widget())

    def test_removing_inputs(self):
        self.send_all()
        self.run_widget()
        self.send_signal(self.widget.Inputs.data, None, 1)
        self.send_signal(self.widget.Inputs.learner, None, 0)
        self.assertEqual(self.run_widget().learner_names, ["Tree [x1 only]", "Tree [both]"])

    def test_needs_inputs(self):
        self.assertTrue(self.widget.Information.need_inputs.is_shown())
        self.send_signal(self.widget.Inputs.data, self.sets[0], 0)
        self.assertTrue(self.widget.Information.need_inputs.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.evaluations_results))

    def test_fallback_names(self):
        for table in self.sets:
            table.name = "same"
        self.send_all()
        self.assertEqual(self.widget.set_names(), ["Set 1", "Set 2", "Set 3"])

    def test_learner_that_fails(self):
        d = data()
        numeric = Table.from_numpy(Domain(d.domain.attributes, ContinuousVariable("t")),
                                   d.X, d.Y)
        self.sets = [numeric]
        self.send_all([named(LogisticRegressionLearner(), "LR")])
        self.run_widget()
        self.assertTrue(self.widget.Warning.learner_failed.is_shown())
        self.assertIn("(error)", self.rows()[0])


if __name__ == "__main__":
    unittest.main()

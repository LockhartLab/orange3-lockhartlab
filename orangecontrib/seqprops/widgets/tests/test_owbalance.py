import unittest

import numpy as np

from Orange.classification import KNNLearner, SVMLearner, TreeLearner
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, Table
from Orange.evaluation import CrossValidation, Recall
from Orange.widgets.tests.base import WidgetTest

from Orange.widgets.evaluate.owtestandscore import OWTestAndScore

from orangecontrib.seqprops.balance import Balance, BalancedLearner, balanced_rows, DOWN, UP
from orangecontrib.seqprops.widgets.owbalance import OWBalance


def imbalanced(n_small=30, n_large=90, seed=0):
    """Two overlapping classes, three times more of class 'b'."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.8, 1, (n_small, 2)), rng.normal(0, 1, (n_large, 2))])
    y = np.array([0] * n_small + [1] * n_large, dtype=float)
    domain = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")],
                    DiscreteVariable("y", values=("a", "b")))
    return Table.from_numpy(domain, X, y)


def named(learner, name):
    """Name a learner the way Orange's model widgets do."""
    learner.name = name
    return learner


class RecordingLearner(KNNLearner):
    """kNN that remembers the class counts it was trained on."""
    seen = []

    def fit(self, X, Y, W=None):
        RecordingLearner.seen.append(np.bincount(Y.astype(int).ravel(), minlength=2))
        return super().fit(X, Y, W)


class TestBalancedRows(unittest.TestCase):
    def test_down_and_up(self):
        y = np.array([0, 1, 1, 1, 0, 1, 1, 2, 1, np.nan])
        rng = np.random.default_rng(0)
        down = balanced_rows(y, DOWN, rng)
        self.assertEqual(np.bincount(y[down].astype(int)).tolist(), [1, 1, 1])
        up = balanced_rows(y, UP, rng)
        self.assertEqual(np.bincount(y[up].astype(int)).tolist(), [6, 6, 6])
        # downsampling never repeats a row; nothing with a missing class
        self.assertEqual(len(set(down)), len(down))
        self.assertFalse(np.isnan(y[up]).any())

    def test_single_class_is_left_alone(self):
        y = np.array([1.0, 1, 1])
        self.assertEqual(balanced_rows(y, DOWN, np.random.default_rng(0)).tolist(), [0, 1, 2])


class TestBalancedLearner(unittest.TestCase):
    def test_only_training_folds_are_resampled(self):
        data = imbalanced()
        RecordingLearner.seen = []
        res = CrossValidation(k=5, stratified=True, random_state=0)(
            data, [BalancedLearner(RecordingLearner(), DOWN)])
        # every fold trained on equal classes...
        self.assertEqual(len(RecordingLearner.seen), 5)
        for counts in RecordingLearner.seen:
            self.assertEqual(counts[0], counts[1])
        # ...but every original row was tested once, at the real 30 / 90 ratio
        self.assertEqual(np.bincount(res.actual.astype(int)).tolist(), [30, 90])

        RecordingLearner.seen = []
        CrossValidation(k=5, stratified=True, random_state=0)(
            data, [BalancedLearner(RecordingLearner(), UP)])
        self.assertEqual(len(RecordingLearner.seen), 5)
        for counts in RecordingLearner.seen:
            self.assertEqual(counts.tolist(), [72, 72])

    def test_helps_minority_recall(self):
        data = imbalanced()
        res = CrossValidation(k=5, stratified=True, random_state=0)(
            data, [SVMLearner(), BalancedLearner(SVMLearner(), DOWN)])
        plain, balanced = Recall(res, target=0)
        self.assertGreater(balanced, plain + 0.2)

    def test_works_with_models_that_cannot_use_weights(self):
        data = imbalanced()
        for base in (KNNLearner(), TreeLearner()):
            model = BalancedLearner(base, UP)(data)
            self.assertEqual(len(model(data)), len(data))

    def test_rejects_numeric_target(self):
        data = Table("housing")
        learner = BalancedLearner(KNNLearner())
        self.assertIsNotNone(learner.incompatibility_reason(data.domain))
        self.assertIsNone(learner.incompatibility_reason(imbalanced().domain))

    def test_name(self):
        svm = named(SVMLearner(), "SVM")
        self.assertEqual(BalancedLearner(svm, DOWN).name, "SVM (downsampled)")
        self.assertEqual(BalancedLearner(svm, UP).name, "SVM (upsampled)")


class TestBalancePreprocessor(unittest.TestCase):
    def test_balances_table(self):
        data = imbalanced()
        self.assertEqual(np.bincount(Balance(DOWN)(data).Y.astype(int)).tolist(), [30, 30])
        self.assertEqual(np.bincount(Balance(UP)(data).Y.astype(int)).tolist(), [90, 90])

    def test_rejects_numeric_target(self):
        with self.assertRaises(ValueError):
            Balance()(Table("housing"))

    def test_cross_validation_balances_only_training_folds(self):
        """As Test & Score's Preprocessor: every model trains on balanced
        folds, every original row is tested once at the real ratio."""
        data = imbalanced()
        RecordingLearner.seen = []
        res = CrossValidation(k=5, stratified=True, random_state=0)(
            data, [RecordingLearner(), RecordingLearner()], preprocessor=Balance(UP))
        self.assertEqual(len(RecordingLearner.seen), 10)
        for counts in RecordingLearner.seen:
            self.assertEqual(counts.tolist(), [72, 72])
        self.assertEqual(np.bincount(res.actual.astype(int)).tolist(), [30, 90])


class TestOWBalance(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWBalance)

    def test_no_learner_no_output(self):
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))

    def test_wraps_learner_and_names_it(self):
        self.send_signal(self.widget.Inputs.base_learner, named(SVMLearner(), "SVM"))
        learner = self.get_output(self.widget.Outputs.learner)
        self.assertIsInstance(learner, BalancedLearner)
        self.assertEqual(learner.method, DOWN)
        self.assertEqual(learner.name, "SVM (downsampled)")

        self.widget.controls.method.buttons[UP].click()
        learner = self.get_output(self.widget.Outputs.learner)
        self.assertEqual(learner.method, UP)
        self.assertEqual(learner.name, "SVM (upsampled)")

    def test_model_output_with_data(self):
        self.send_signal(self.widget.Inputs.base_learner, named(KNNLearner(), "kNN"))
        self.send_signal(self.widget.Inputs.data, imbalanced())
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "kNN (downsampled)")

    def test_numeric_target_shows_error(self):
        self.send_signal(self.widget.Inputs.base_learner, KNNLearner())
        self.send_signal(self.widget.Inputs.data, Table("housing"))
        self.assertTrue(self.widget.Error.data_error.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

    def test_preprocessor_output_without_learner(self):
        self.widget.apply()
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsInstance(pp, Balance)
        self.assertEqual(pp.method, DOWN)
        self.widget.controls.method.buttons[UP].click()
        self.assertEqual(self.get_output(self.widget.Outputs.preprocessor).method, UP)

    def test_one_balance_for_all_models_in_test_and_score(self):
        self.widget.apply()
        data = imbalanced()
        ts = self.create_widget(OWTestAndScore)
        self.send_signal(ts.Inputs.train_data, data, widget=ts)
        self.send_signal(ts.Inputs.learner, named(SVMLearner(), "SVM"), 0, widget=ts)
        self.send_signal(ts.Inputs.learner, named(KNNLearner(), "kNN"), 1, widget=ts)
        self.send_signal(ts.Inputs.preprocessor,
                         self.get_output(self.widget.Outputs.preprocessor), widget=ts)
        self.process_events(lambda: self.get_output(ts.Outputs.evaluations_results, widget=ts) is not None)
        res = self.get_output(ts.Outputs.evaluations_results, widget=ts)
        self.assertEqual(np.bincount(res.actual.astype(int)).tolist(), [30, 90])
        plain = CrossValidation(k=10, stratified=True, random_state=0)(
            data, [SVMLearner()])
        self.assertGreater(Recall(res, target=0)[0], Recall(plain, target=0)[0] + 0.2)

    def test_removing_learner(self):
        self.send_signal(self.widget.Inputs.base_learner, SVMLearner())
        self.send_signal(self.widget.Inputs.base_learner, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.learner))

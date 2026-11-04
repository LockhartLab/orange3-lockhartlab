"""Cross-validate several learners on several feature sets of the same data.

Each feature set is the same rows with different columns (for example from
several Select Columns widgets). Every learner is cross-validated on every
set with the same folds, so all the scores can be compared in one table.
"""

import numpy as np

from Orange.evaluation import CrossValidation, Results

# Test & Score's seed, so a set here scores the same as it does there.
RANDOM_STATE = 42


def rows_problem(tables):
    """Why `tables` cannot be compared, or None if they can."""
    first = tables[0]
    if first.domain.class_var is None:
        return "The data has no target variable."
    for table in tables[1:]:
        if table.domain.class_var != first.domain.class_var:
            return "The feature sets have different target variables."
        if len(table) != len(first) or not np.array_equal(table.Y, first.Y, equal_nan=True) \
                or not np.array_equal(table.ids, first.ids):
            return ("The feature sets must contain the same rows. "
                    "Select columns, not rows, after splitting the data.")
    return None


def can_stratify(data, k):
    """Whether every class has at least `k` rows (needed to stratify)."""
    if not data.domain.has_discrete_class:
        return False
    counts = np.bincount(data.Y[~np.isnan(data.Y)].astype(int))
    return min(filter(None, counts), default=0) >= k


def evaluate(tables, learners, preprocessor=None, k=5, stratified=True,
             callback=None):
    """Cross-validate every learner on every table; one Results per table."""
    results = []
    for i, table in enumerate(tables):
        def progress(p, i=i):
            if callback is not None:
                callback((i + p) / len(tables))
        cv = CrossValidation(k=k, stratified=stratified, random_state=RANDOM_STATE,
                             store_data=True)
        results.append(cv(table, learners, preprocessor, callback=progress))
    return results


def combined_name(learner_name, set_name):
    return f"{learner_name} [{set_name}]"


def combine(results, set_names, learner_names):
    """One Results with a model per (feature set, learner), set by set."""
    first = results[0]
    for res in results[1:]:
        assert np.array_equal(res.row_indices, first.row_indices)
    out = Results()
    out.data = first.data
    out.domain = first.domain
    out.row_indices = first.row_indices
    out.folds = first.folds
    out.actual = first.actual
    out.predicted = np.vstack([res.predicted for res in results])
    if all(res.probabilities is not None for res in results):
        out.probabilities = np.vstack([res.probabilities for res in results])
    out.learners = [learner for res in results for learner in res.learners]
    out.failed = [failed for res in results for failed in res.failed]
    out.models = None
    out.learner_names = [combined_name(learner, set_name)
                         for set_name in set_names for learner in learner_names]
    return out

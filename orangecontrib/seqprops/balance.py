"""A learner that trains another learner on class-balanced data.

Only the data the learner is trained on is resampled. Inside Test & Score's
cross-validation that is the training part of each fold, so the test rows
keep their real class ratio and no copied row can end up in a test fold.
"""

import numpy as np

from Orange.base import Learner, Model

DOWN, UP = 0, 1
METHOD_NAMES = {DOWN: "downsampled", UP: "upsampled"}


def balanced_rows(y, method, rng):
    """Row indices with every class equally often: the larger classes are
    sampled down to the smallest (DOWN), or the smaller classes are sampled
    up, with replacement, to the largest (UP). Rows with a missing class are
    dropped."""
    known = ~np.isnan(y)
    classes, counts = np.unique(y[known], return_counts=True)
    if len(classes) < 2:
        return np.flatnonzero(known)
    n = counts.min() if method == DOWN else counts.max()
    rows = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=n > k)
                           for c, k in zip(classes, counts)])
    return np.sort(rows)


class BalancedLearner(Learner):
    """Wraps `base_learner`; each time it is trained, it first resamples
    the training data so that all classes are equally frequent."""
    __returns__ = Model  # whatever `base_learner` returns

    def __init__(self, base_learner, method=DOWN, random_state=0,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.base_learner = base_learner
        self.method = method
        self.random_state = random_state
        self.name = f"{base_learner.name} ({METHOD_NAMES[method]})"

    def incompatibility_reason(self, domain):
        if len(domain.class_vars) != 1 or not domain.class_var.is_discrete:
            return "Balancing needs a single categorical target."
        return self.base_learner.incompatibility_reason(domain)

    def fit_storage(self, data):
        rng = np.random.default_rng(self.random_state)
        return self.base_learner(data[balanced_rows(data.Y, self.method, rng)])

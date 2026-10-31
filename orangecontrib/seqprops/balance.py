"""Class balancing that touches only the data a model is trained on.

`BalancedLearner` wraps one learner. `Balance` is a preprocessor: given to
Test & Score's Preprocessor input, it balances the training part of each
fold for every model there. Either way the test rows keep their real class
ratio and no copied row can end up in a test fold.
"""

import numpy as np

from Orange.base import Learner, Model
from Orange.preprocess.preprocess import Preprocess

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


def _check_target(domain):
    if len(domain.class_vars) != 1 or not domain.class_var.is_discrete:
        return "Balancing needs a single categorical target."
    return None


class Balance(Preprocess):
    """Resample a table so that all classes are equally frequent."""

    def __init__(self, method=DOWN, random_state=0):
        self.method = method
        self.random_state = random_state

    def __call__(self, data):
        reason = _check_target(data.domain)
        if reason is not None:
            raise ValueError(reason)
        rng = np.random.default_rng(self.random_state)
        return data[balanced_rows(data.Y, self.method, rng)]


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
        return _check_target(domain) or self.base_learner.incompatibility_reason(domain)

    def fit_storage(self, data):
        return self.base_learner(Balance(self.method, self.random_state)(data))

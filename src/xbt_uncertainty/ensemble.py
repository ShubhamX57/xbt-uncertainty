"""An ensemble of decision trees, each trained on a bootstrap sample of cruises."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone


def cruise_bootstrap(groups, rng):
    """Row indices for one bootstrap sample drawn over cruises, not casts.

    Cruises are drawn with replacement and every cast of a drawn cruise is
    included, once per draw. Resampling casts instead would put casts from the
    same cruise on both sides of each member's in-bag and out-of-bag split.
    """
    groups = pd.Series(np.asarray(groups))
    rows_by_group = groups.groupby(groups).indices
    names = list(rows_by_group)
    drawn = rng.choice(len(names), size=len(names), replace=True)
    return np.concatenate([rows_by_group[names[i]] for i in drawn])


def _fit_member(model, X, y, rows):
    return clone(model).fit(X.iloc[rows], y.iloc[rows])


class CruiseBaggedTrees:
    """Votes of n_members models turned into class probabilities.

    The probability of a class is the fraction of members that predict it.
    """

    def __init__(self, model, n_members=50, seed=0, n_jobs=-1):
        self.model = model
        self.n_members = n_members
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y, groups):
        y = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.array(sorted(y.unique()))
        rng = np.random.default_rng(self.seed)
        samples = [cruise_bootstrap(groups, rng) for _ in range(self.n_members)]
        self.members_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_member)(self.model, X, y, rows) for rows in samples
        )
        return self

    def votes(self, X):
        """Array of shape (n_casts, n_classes) with the number of votes per class."""
        index = {label: i for i, label in enumerate(self.classes_)}
        counts = np.zeros((len(X), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(X))
        for member in self.members_:
            predicted = member.predict(X)
            counts[rows, [index[label] for label in predicted]] += 1
        return counts

    def predict_proba(self, X):
        return self.votes(X) / self.n_members

    def average_proba(self, X):
        """Mean of the members' own leaf probabilities, as a random forest does.

        Votes only record which class each member picks, so a cast whose
        features are shared by two models can still get every vote. Averaging
        keeps the class mix in each member's leaf.
        """
        out = np.zeros((len(X), len(self.classes_)))
        for member in self.members_:
            columns = np.searchsorted(self.classes_, member.classes_)
            out[:, columns] += member.predict_proba(X)
        return out / self.n_members

    def predict(self, X):
        return self.classes_[self.votes(X).argmax(axis=1)]

"""Feature selection by permutation importance.

The importance of a feature is the mean increase in squared error when that feature is shuffled in the evaluation data
(Breiman, 2001). Where it is measured matters: on the fitting data it rewards features the model memorized, and on the
data later used to report performance it leaks that data into the selection (see oof_permutation_importance).
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold


def permutation_importance_mean(estimator, X_fit, y_fit, X_eval, y_eval, n_repeats=5, seed=42, n_jobs=1):
    """Fit a clone of `estimator` on (X_fit, y_fit) and return the mean permutation importance of every feature on
    (X_eval, y_eval), with scoring negative MSE (so a positive importance means the feature reduces the error).

    Args:
        estimator: unfitted regressor (it is cloned, never modified).
        X_fit: feature matrix the clone is fitted on.
        y_fit: target values of X_fit.
        X_eval: feature matrix whose columns are permuted (same columns as X_fit).
        y_eval: target values of X_eval.
        n_repeats: number of permutations of each feature; their score drops are averaged.
        seed: random_state of the permutations.
        n_jobs: processes over features; with single-threaded estimators the result does not depend on it.
    """
    fitted = clone(estimator).fit(X_fit, y_fit)
    result = permutation_importance(fitted, X_eval, y_eval, scoring="neg_mean_squared_error", n_repeats=n_repeats,
                                    random_state=seed, n_jobs=n_jobs)
    return result.importances_mean


def oof_permutation_importance(estimator, X, y, n_folds=5, n_repeats=5, seed=42, n_jobs=1):
    """Out-of-fold permutation importance: in each of n_folds folds (KFold, shuffled with `seed`) a clone is fitted on the
    other folds and the features are permuted on the held-out fold; the n_folds importance vectors are averaged.

    Every importance is measured on molecules the fitted model has not seen, and only the rows passed in are used, so
    selecting features with it inside a training set keeps the test set out of the selection.

    Args:
        estimator: unfitted regressor (it is cloned for every fold, never modified).
        X: feature matrix of the rows used for the selection (e.g. the training part only).
        y: target values of X.
        n_folds: number of folds of the KFold.
        n_repeats: number of permutations of each feature per fold.
        seed: random_state of both the KFold shuffle and the permutations.
        n_jobs: processes over features; with single-threaded estimators the result does not depend on it.
    """
    X, y = np.asarray(X), np.asarray(y)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold = [permutation_importance_mean(estimator, X[tr], y[tr], X[va], y[va], n_repeats, seed, n_jobs)
                for tr, va in folds.split(X)]
    return np.mean(per_fold, axis=0)


def select_by_importance(importances, tol=1e-9):
    """Boolean mask of the features whose importance exceeds `tol`.

    A feature the model never uses has an importance that is zero up to floating-point noise (about 1e-15), so a plain
    "> 0" keeps a random subset of them; the default tolerance sits well above that noise and below genuine importances.

    Args:
        importances: one importance per feature (e.g. from oof_permutation_importance).
        tol: features with an importance above this value are kept.
    """
    return np.asarray(importances) > tol


class PermutationImportanceSelector(SelectorMixin, BaseEstimator):
    """Transformer that keeps the features with an out-of-fold permutation importance above `tol`
    (oof_permutation_importance followed by select_by_importance), computed on the data it is fitted on. Placed first in
    a Pipeline, the selection is redone inside every training set the pipeline is fitted on, which is what a test of
    the whole model-building procedure (cross-validation, y-randomization) requires.

    If no feature passes `tol`, which can happen with a randomized response, all features are kept (no selection) and
    `fallback_` is True, so that the procedure still yields a model and the fitted selector records the event.

    Args:
        estimator: unfitted regressor whose permutation importance ranks the features (cloned, never modified).
        n_folds: folds of the out-of-fold importance.
        n_repeats: permutations of each feature per fold.
        tol: features with an importance above this value are kept.
        seed: random_state of the folds and of the permutations.
        n_jobs: processes over features.

    Attributes:
        importances_: importance of every feature, from the last fit.
        fallback_: True if no feature passed `tol` and all were kept.
    """

    def __init__(self, estimator, n_folds=5, n_repeats=5, tol=1e-9, seed=42, n_jobs=1):
        self.estimator = estimator
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.tol = tol
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.importances_ = oof_permutation_importance(self.estimator, X, y, self.n_folds, self.n_repeats, self.seed,
                                                       self.n_jobs)
        support = select_by_importance(self.importances_, self.tol)
        self.fallback_ = not support.any()
        self.support_ = np.ones_like(support) if self.fallback_ else support
        self.n_features_in_ = len(support)
        return self

    def _get_support_mask(self):
        return self.support_

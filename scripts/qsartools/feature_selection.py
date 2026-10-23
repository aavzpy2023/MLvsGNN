"""Feature selection by permutation importance.

The importance of a feature is the mean increase in squared error when that feature is shuffled in the evaluation data
(Breiman, 2001). Where it is measured matters: on the fitting data it rewards features the model memorized, and on the
data later used to report performance it leaks that data into the selection (see oof_permutation_importance).
"""
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold


def permutation_importance_mean(estimator, X_fit, y_fit, X_eval, y_eval, n_repeats=5, seed=42, n_jobs=1):
    """Fit a clone of `estimator` on (X_fit, y_fit) and return the mean permutation importance of every feature on
    (X_eval, y_eval), with scoring negative MSE (so a positive importance means the feature reduces the error).

    n_jobs parallelizes over features; with single-threaded estimators the result does not depend on it.
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
    """
    return np.asarray(importances) > tol

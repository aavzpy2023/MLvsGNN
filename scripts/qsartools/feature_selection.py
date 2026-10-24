"""Feature selection by permutation importance.

The importance of a feature is the mean increase in squared error when that feature is shuffled in data the model was
not fitted on: the idea of Breiman (2001, Machine Learning 45:5-32, section 10) for random forests, which permutes the
out-of-bag samples of each tree, in the model-agnostic form of Fisher, Rudin & Dominici (2019, J. Mach. Learn. Res.
20(177)) that permutes held-out data of any fitted model. Where it is measured matters: on the fitting data it rewards
features the model memorized, and on the data later used to report performance it leaks that data into the selection
(see oof_permutation_importance).

Two selection rules are offered. "threshold" (select_by_importance) keeps every feature whose importance exceeds a
numerical-noise tolerance; it is a choice of this package, not a published rule. "pimp" keeps the features whose
importance is significant against a null distribution obtained by permuting the response (PIMP: Altmann, Tolosi,
Sander & Lengauer, 2010, Bioinformatics 26:1340-1347).
"""
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
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


PIMP_DISTRIBUTIONS = ("gaussian", "lognormal", "gamma", "nonparametric")


def _null_importance(estimator, X, y, n_folds, n_repeats, seed, permutation_seed):
    y_perm = np.random.default_rng(permutation_seed).permutation(y)
    return oof_permutation_importance(estimator, X, y_perm, n_folds, n_repeats, seed)


def pimp_p_values(estimator, X, y, n_null=100, distribution="gaussian", n_folds=5, n_repeats=5, seed=42, n_jobs=1):
    """PIMP p-value of every feature (Altmann et al., 2010): the response is permuted n_null times, the importance of
    every feature is recomputed each time (the null importances), a distribution is fitted to the null importances of
    each feature by maximum likelihood, and the p-value is the probability of an importance at least as large as the
    one measured with the real response.

    The importance is oof_permutation_importance, computed with the same folds and permutation seed for the real and
    the permuted responses. Following Altmann et al., for the Gaussian fit every null variance smaller than the mean null
    variance of all features is raised to that mean, so that a nearly constant null distribution does not produce an
    artificially small p-value. Their automatic choice of distribution by Kolmogorov-Smirnov tests is not implemented.

    Args:
        estimator: unfitted regressor (cloned for every fit, never modified).
        X: (n, d) feature matrix.
        y: n target values.
        n_null: number of permutations of the response (s in Altmann et al., who used 10 to 100).
        distribution: "gaussian", "lognormal" or "gamma" (fitted by maximum likelihood; lognormal and gamma need
            strictly positive null importances), or "nonparametric" (the fraction of null importances >= the real one).
        n_folds: folds of the out-of-fold importance.
        n_repeats: permutations of each feature per fold.
        seed: random_state of the folds and of the feature permutations; permutation i of the response uses
            numpy.random.default_rng(seed + i).
        n_jobs: processes over the n_null permutations.

    Returns:
        (importances, p_values, null_importances): the importances with the real response, one p-value per feature,
        and the (n_null, d) array of null importances.
    """
    if distribution not in PIMP_DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {PIMP_DISTRIBUTIONS}, not {distribution!r}")
    X, y = np.asarray(X), np.asarray(y)
    importances = oof_permutation_importance(estimator, X, y, n_folds, n_repeats, seed)
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_null_importance)(estimator, X, y, n_folds, n_repeats, seed, seed + i) for i in range(n_null)))
    if distribution == "nonparametric":
        return importances, (null >= importances).mean(axis=0), null
    if distribution == "gaussian":
        mean, var = null.mean(axis=0), null.var(axis=0, ddof=1)
        var = np.maximum(var, var.mean())
        return importances, stats.norm.sf(importances, loc=mean, scale=np.sqrt(var)), null
    if (null <= 0).any():
        raise ValueError(f"the {distribution} distribution needs strictly positive null importances")
    family = stats.lognorm if distribution == "lognormal" else stats.gamma
    p = np.array([family.sf(v, *family.fit(null[:, j], floc=0)) for j, v in enumerate(importances)])
    return importances, p, null


class PermutationImportanceSelector(SelectorMixin, BaseEstimator):
    """Transformer that selects features by out-of-fold permutation importance, computed on the data it is fitted on.
    Placed first in a Pipeline, the selection is redone inside every training set the pipeline is fitted on, which is
    what a test of the whole model-building procedure (cross-validation, y-randomization) requires.

    method="threshold" keeps the features with an importance above `tol` (select_by_importance); method="pimp" keeps
    those with a PIMP p-value below `alpha` (pimp_p_values; Altmann et al. used 0.05). If no feature is kept, which can
    happen with a randomized response, all features are kept (no selection) and `fallback_` is True, so that the
    procedure still yields a model and the fitted selector records the event.

    Args:
        estimator: unfitted regressor whose permutation importance ranks the features (cloned, never modified).
        method: "threshold" or "pimp".
        n_folds: folds of the out-of-fold importance.
        n_repeats: permutations of each feature per fold.
        tol: importance threshold of method="threshold".
        alpha: significance level of method="pimp".
        n_null: permutations of the response of method="pimp".
        distribution: null distribution of method="pimp", as for pimp_p_values.
        seed: random_state of the folds and of the permutations.
        n_jobs: processes (over features for "threshold", over response permutations for "pimp").

    Attributes:
        importances_: importance of every feature with the real response, from the last fit.
        p_values_: PIMP p-value of every feature (method="pimp" only).
        fallback_: True if no feature was selected and all were kept.
    """

    def __init__(self, estimator, method="threshold", n_folds=5, n_repeats=5, tol=1e-9, alpha=0.05, n_null=100,
                 distribution="gaussian", seed=42, n_jobs=1):
        self.estimator = estimator
        self.method = method
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.tol = tol
        self.alpha = alpha
        self.n_null = n_null
        self.distribution = distribution
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.method == "threshold":
            self.importances_ = oof_permutation_importance(self.estimator, X, y, self.n_folds, self.n_repeats,
                                                           self.seed, self.n_jobs)
            support = select_by_importance(self.importances_, self.tol)
        elif self.method == "pimp":
            self.importances_, self.p_values_, _ = pimp_p_values(self.estimator, X, y, self.n_null, self.distribution,
                                                                 self.n_folds, self.n_repeats, self.seed, self.n_jobs)
            support = self.p_values_ < self.alpha
        else:
            raise ValueError(f"method must be 'threshold' or 'pimp', not {self.method!r}")
        self.fallback_ = not support.any()
        self.support_ = np.ones_like(support) if self.fallback_ else support
        self.n_features_in_ = len(support)
        return self

    def _get_support_mask(self):
        return self.support_

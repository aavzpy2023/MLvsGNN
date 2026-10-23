"""Y-randomization (response permutation) test against chance correlation (Rucker et al., 2007, J. Chem. Inf. Model.
47:2345-2357): the target is shuffled between samples, which destroys any real structure-activity relationship, and the
same model is cross-validated again. A model that learned real structure scores far above the shuffled models."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score


def _cv_r2(estimator, X, y, n_folds, seed, n_jobs=1):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clone(estimator), X, y, cv=folds, scoring="r2", n_jobs=n_jobs)))


def _permuted_r2(estimator, X, y, n_folds, seed):
    y_shuffled = np.random.default_rng(seed).permutation(y)
    return _cv_r2(estimator, X, y_shuffled, n_folds, seed)


def y_randomization(estimator, X, y, n_permutations=100, n_folds=5, seed=42, n_jobs=1):
    """Mean K-fold cross-validated R2 of `estimator` with the true target and with n_permutations shuffled targets.

    The true-target score uses KFold(n_folds, shuffle=True, random_state=seed). Permutation i (i = 0..n-1) shuffles the
    target with numpy.random.default_rng(seed + i) and is cross-validated with the folds of seed + i, so every result is
    reproducible and independent of n_jobs. The estimator is cloned for every fit and never modified.

    Returns a dict: true_r2; permuted_r2 (array of n_permutations scores); permuted_mean; permuted_sd (population
    standard deviation); z = (true_r2 - permuted_mean) / permuted_sd; p_empirical = (1 + number of permuted scores
    >= true_r2) / (n_permutations + 1), the usual permutation p-value.
    """
    y = np.asarray(y)
    true_r2 = _cv_r2(estimator, X, y, n_folds, seed, n_jobs=n_jobs)
    permuted = np.array(Parallel(n_jobs=n_jobs)(delayed(_permuted_r2)(estimator, X, y, n_folds, seed + i)
                                                for i in range(n_permutations)))
    mean, sd = float(permuted.mean()), float(permuted.std())
    return {"true_r2": true_r2, "permuted_r2": permuted, "permuted_mean": mean, "permuted_sd": sd,
            "z": (true_r2 - mean) / sd if sd > 0 else 0.0,
            "p_empirical": float((1 + np.sum(permuted >= true_r2)) / (n_permutations + 1))}

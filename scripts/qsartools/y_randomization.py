"""Y-randomization and its pseudodescriptor variant: permutation tests against chance correlation (Rucker, Rucker &
Meringer, 2007, J. Chem. Inf. Model. 47:2345-2357).

Y-randomization permutes the response, "leaving all x data untouched, and performing the whole model building
procedure as it would be done for real y data" (Rucker et al., p. 2346). The risk it addresses comes from building the
model, above all from selecting a few descriptors out of many, so `estimator` must contain the whole procedure (e.g. a
Pipeline whose first step selects the descriptors): a test of the final model alone does not see that risk. Rucker et
al. also compare the real model with models built on random pseudodescriptors for the real response, because real
descriptors are intercorrelated and y-randomization alone gives lower chance scores; they propose reporting both.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

VARIANTS = ("y_permutation", "pseudodescriptors")
STATISTICS = ("fit_r2", "cv_r2")


def _score(estimator, X, y, statistic, n_folds, seed, n_jobs=1):
    if statistic == "fit_r2":
        return float(r2_score(y, clone(estimator).fit(X, y).predict(X)))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clone(estimator), X, y, cv=folds, scoring="r2", n_jobs=n_jobs)))


def _shuffle_columns(X, rng):
    values = np.asarray(X)
    shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(values.shape[1])])
    return pd.DataFrame(shuffled, columns=X.columns, index=X.index) if isinstance(X, pd.DataFrame) else shuffled


def _randomized_score(estimator, X, y, variant, statistic, n_folds, seed):
    rng = np.random.default_rng(seed)
    if variant == "y_permutation":
        return _score(estimator, X, rng.permutation(y), statistic, n_folds, seed)
    return _score(estimator, _shuffle_columns(X, rng), y, statistic, n_folds, seed)


def y_randomization(estimator, X, y, n_permutations=100, n_folds=5, seed=42, n_jobs=1, variant="y_permutation",
                    statistic="fit_r2"):
    """Score of `estimator` on the real data and on n_permutations randomized versions of it.

    Randomization i (i = 0..n-1) uses numpy.random.default_rng(seed + i) and, with statistic="cv_r2", the folds
    KFold(n_folds, shuffle=True, random_state=seed + i); the real-data score uses the folds of `seed`. Every result is
    reproducible and independent of n_jobs; the estimator is cloned for every fit and never modified.

    Args:
        estimator: unfitted regressor containing the whole model-building procedure (descriptor selection included).
        X: feature matrix (array or DataFrame).
        y: target values, in the same order as the rows of X.
        n_permutations: number of randomizations; the smallest attainable p-value is 1 / (n_permutations + 1).
        n_folds: number of folds of each cross-validation (statistic="cv_r2" only).
        seed: base seed; the real-data folds use it and randomization i uses seed + i.
        n_jobs: processes; the real-data cross-validation is parallel over folds, the randomizations over
            randomizations.
        variant: "y_permutation" (the response is permuted, X untouched: Rucker et al.'s y-randomization) or
            "pseudodescriptors" (the real response with descriptors made random by shuffling every column
            independently, which keeps each descriptor's distribution but destroys its relation to y and to the other
            descriptors; Rucker et al. drew random numbers).
        statistic: "fit_r2" (R2 of the model fitted to all the data, as in Rucker et al.'s multilinear regression) or
            "cv_r2" (mean K-fold cross-validated R2). Use "cv_r2" for flexible models such as tree ensembles, which fit
            even a random response almost perfectly and would make "fit_r2" uninformative.

    Returns:
        Dict: true_r2; permuted_r2 (array of n_permutations scores of the randomized data); permuted_mean; permuted_sd
        (population standard deviation); z = (true_r2 - permuted_mean) / permuted_sd; p_empirical = (1 + number of
        randomized scores >= true_r2) / (n_permutations + 1), the permutation-test p-value.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, not {statistic!r}")
    y = np.asarray(y)
    true_r2 = _score(estimator, X, y, statistic, n_folds, seed, n_jobs=n_jobs)
    permuted = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_randomized_score)(estimator, X, y, variant, statistic, n_folds, seed + i)
        for i in range(n_permutations)))
    mean, sd = float(permuted.mean()), float(permuted.std())
    return {"true_r2": true_r2, "permuted_r2": permuted, "permuted_mean": mean, "permuted_sd": sd,
            "z": (true_r2 - mean) / sd if sd > 0 else 0.0,
            "p_empirical": float((1 + np.sum(permuted >= true_r2)) / (n_permutations + 1))}

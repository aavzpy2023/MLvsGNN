import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from qsartools.feature_selection import oof_permutation_importance, permutation_importance_mean, select_by_importance

RNG = np.random.default_rng(0)
X = RNG.normal(size=(150, 6))
Y = 3 * X[:, 0] + 1.5 * X[:, 1] + RNG.normal(0, 0.1, 150)   # only features 0 and 1 matter


def test_matches_sklearn_and_ranks_informative_features():
    imp = permutation_importance_mean(LinearRegression(), X[:100], Y[:100], X[100:], Y[100:], n_repeats=5, seed=1)
    ref = permutation_importance(LinearRegression().fit(X[:100], Y[:100]), X[100:], Y[100:],
                                 scoring="neg_mean_squared_error", n_repeats=5, random_state=1).importances_mean
    np.testing.assert_array_equal(imp, ref)
    assert imp[0] > imp[1] > max(imp[2:]) and imp[0] > 1


def test_oof_is_the_mean_of_per_fold_importances():
    imp = oof_permutation_importance(LinearRegression(), X, Y, n_folds=3, n_repeats=2, seed=4)
    per_fold = []
    for tr, va in KFold(3, shuffle=True, random_state=4).split(X):
        model = LinearRegression().fit(X[tr], Y[tr])
        per_fold.append(permutation_importance(model, X[va], Y[va], scoring="neg_mean_squared_error", n_repeats=2,
                                               random_state=4).importances_mean)
    np.testing.assert_allclose(imp, np.mean(per_fold, axis=0))


def test_unused_feature_has_zero_importance_and_is_not_selected():
    # A constant column is never used by a tree, so its importance is exactly zero
    Xc = np.hstack([X, np.ones((150, 1))])
    imp = oof_permutation_importance(RandomForestRegressor(n_estimators=20, random_state=0), Xc, Y, n_folds=3,
                                     n_repeats=2)
    assert abs(imp[-1]) < 1e-12
    mask = select_by_importance(imp)
    assert mask[0] and mask[1] and not mask[-1]


def test_threshold():
    np.testing.assert_array_equal(select_by_importance([1e-3, 1e-15, -0.2, 2e-9], tol=1e-9), [True, False, False, True])


def test_parallel_equals_serial():
    model = RandomForestRegressor(n_estimators=15, random_state=0, n_jobs=1)
    a = oof_permutation_importance(model, X, Y, n_folds=3, n_repeats=2, n_jobs=1)
    b = oof_permutation_importance(model, X, Y, n_folds=3, n_repeats=2, n_jobs=3)
    np.testing.assert_array_equal(a, b)

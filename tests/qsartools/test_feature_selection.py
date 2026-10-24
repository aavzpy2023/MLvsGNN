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


def test_selector_reproduces_oof_selection_and_is_refitted_in_a_pipeline():
    from sklearn.pipeline import Pipeline
    from qsartools.feature_selection import PermutationImportanceSelector
    rng = np.random.default_rng(3)
    Xs = rng.normal(size=(100, 6))
    ys = 2 * Xs[:, 0] - Xs[:, 1] + rng.normal(0, 0.1, 100)
    selector = PermutationImportanceSelector(LinearRegression(), n_folds=4, n_repeats=3, seed=1).fit(Xs, ys)
    expected = select_by_importance(oof_permutation_importance(LinearRegression(), Xs, ys, 4, 3, 1))
    np.testing.assert_array_equal(selector.get_support(), expected)
    assert selector.get_support()[:2].all() and not selector.fallback_
    pipe = Pipeline([("select", PermutationImportanceSelector(LinearRegression(), n_folds=4, n_repeats=3, seed=1)),
                     ("model", LinearRegression())]).fit(Xs[:60], ys[:60])
    assert pipe.named_steps["select"].importances_.shape == (6,) and pipe.predict(Xs[60:]).shape == (40,)


def test_selector_keeps_every_feature_when_none_passes():
    from qsartools.feature_selection import PermutationImportanceSelector
    rng = np.random.default_rng(4)
    Xs, ys = rng.normal(size=(60, 5)), rng.normal(size=60)
    selector = PermutationImportanceSelector(LinearRegression(), n_folds=3, n_repeats=2, tol=1e9).fit(Xs, ys)
    assert selector.fallback_ and selector.get_support().all() and selector.transform(Xs).shape == (60, 5)


def _informative_data(seed=5, n=80):
    rng = np.random.default_rng(seed)
    Xs = rng.normal(size=(n, 5))
    return Xs, 2 * Xs[:, 0] - 1.5 * Xs[:, 1] + rng.normal(0, 0.3, n)


def test_pimp_separates_informative_from_noise_features():
    from qsartools.feature_selection import pimp_p_values
    Xs, ys = _informative_data()
    imp, p, null = pimp_p_values(LinearRegression(), Xs, ys, n_null=20, n_folds=4, n_repeats=2, seed=1)
    assert null.shape == (20, 5) and p.shape == (5,)
    assert (p[:2] < 0.01).all() and (p[2:] > 0.05).all()


def test_pimp_nonparametric_is_the_fraction_of_null_importances_above_the_real_one():
    from qsartools.feature_selection import pimp_p_values
    Xs, ys = _informative_data()
    imp, p, null = pimp_p_values(LinearRegression(), Xs, ys, n_null=10, distribution="nonparametric", n_folds=3,
                                 n_repeats=2, seed=2)
    np.testing.assert_array_equal(p, (null >= imp).mean(axis=0))


def test_pimp_gaussian_raises_small_null_variances_to_the_mean():
    from scipy import stats
    from qsartools.feature_selection import pimp_p_values
    Xs, ys = _informative_data()
    imp, p, null = pimp_p_values(LinearRegression(), Xs, ys, n_null=10, n_folds=3, n_repeats=2, seed=3)
    var = null.var(axis=0, ddof=1)
    expected = stats.norm.sf(imp, loc=null.mean(axis=0), scale=np.sqrt(np.maximum(var, var.mean())))
    np.testing.assert_allclose(p, expected)


def test_pimp_options_are_checked():
    import pytest
    from qsartools.feature_selection import pimp_p_values
    Xs, ys = _informative_data()
    with pytest.raises(ValueError, match="distribution"):
        pimp_p_values(LinearRegression(), Xs, ys, n_null=2, distribution="weibull")
    with pytest.raises(ValueError, match="strictly positive"):
        pimp_p_values(LinearRegression(), Xs, ys, n_null=5, distribution="gamma", n_folds=3, n_repeats=1)


def test_selector_with_pimp_keeps_the_significant_features():
    from qsartools.feature_selection import PermutationImportanceSelector
    Xs, ys = _informative_data()
    selector = PermutationImportanceSelector(LinearRegression(), method="pimp", n_null=20, n_folds=4, n_repeats=2,
                                             seed=1).fit(Xs, ys)
    assert list(selector.get_support()) == [True, True, False, False, False] and selector.p_values_.shape == (5,)

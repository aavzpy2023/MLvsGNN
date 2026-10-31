import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from qsartools.y_randomization import y_randomization

RNG = np.random.default_rng(0)
X = RNG.normal(size=(120, 8))
Y_SIGNAL = X @ RNG.normal(size=8) + RNG.normal(0, 0.3, 120)
Y_NOISE = RNG.normal(size=120)


def test_real_signal_is_far_above_permutations():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=30, seed=1, statistic="cv_r2")
    assert res["true_r2"] > 0.9 and abs(res["permuted_mean"]) < 0.2
    assert res["z"] > 10 and res["p_empirical"] == 1 / 31


def test_pure_noise_is_not_distinguishable():
    res = y_randomization(Ridge(), X, Y_NOISE, n_permutations=30, seed=1, statistic="cv_r2")
    assert res["p_empirical"] > 0.05 and abs(res["z"]) < 3


def test_true_score_matches_sklearn_cross_validation():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, n_folds=4, seed=3, statistic="cv_r2")
    ref = cross_val_score(Ridge(), X, Y_SIGNAL, cv=KFold(4, shuffle=True, random_state=3), scoring="r2").mean()
    assert res["true_r2"] == ref


def test_permutation_definition():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=3, seed=5, statistic="cv_r2")
    for i, score in enumerate(res["permuted_r2"]):
        y_perm = np.random.default_rng(5 + i).permutation(Y_SIGNAL)
        ref = cross_val_score(Ridge(), X, y_perm, cv=KFold(5, shuffle=True, random_state=5 + i), scoring="r2").mean()
        assert score == ref


def test_reproducible_and_independent_of_n_jobs():
    a = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, n_jobs=1)
    b = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, n_jobs=3)
    np.testing.assert_array_equal(a["permuted_r2"], b["permuted_r2"])
    assert a["true_r2"] == b["true_r2"]


def test_resumed_run_equals_an_uninterrupted_one():
    full = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, statistic="cv_r2")
    saved = {}
    y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, statistic="cv_r2", n_jobs=2,
                    on_result=saved.__setitem__)
    assert sorted(saved) == [-1, 0, 1, 2, 3, 4, 5] and saved[-1] == full["true_r2"]
    rerun = []
    resumed = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, statistic="cv_r2", true_r2=saved[-1],
                              done={i: saved[i] for i in (0, 3, 4)}, on_result=lambda i, s: rerun.append(i))
    assert sorted(rerun) == [1, 2, 5]
    np.testing.assert_array_equal(resumed["permuted_r2"], full["permuted_r2"])
    assert resumed["p_empirical"] == full["p_empirical"]


def test_estimator_is_not_modified():
    model = Ridge()
    y_randomization(model, X, Y_SIGNAL, n_permutations=2)
    assert not hasattr(model, "coef_")


def test_fit_r2_is_the_r2_of_the_model_fitted_to_all_the_data():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, seed=4)
    assert res["true_r2"] == r2_score(Y_SIGNAL, Ridge().fit(X, Y_SIGNAL).predict(X))
    y_perm = np.random.default_rng(4).permutation(Y_SIGNAL)
    assert res["permuted_r2"][0] == r2_score(y_perm, Ridge().fit(X, y_perm).predict(X))


def test_pseudodescriptors_shuffle_every_column_and_keep_the_response():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, seed=7, variant="pseudodescriptors")
    rng = np.random.default_rng(7)
    X_rand = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    assert res["permuted_r2"][0] == r2_score(Y_SIGNAL, Ridge().fit(X_rand, Y_SIGNAL).predict(X_rand))
    np.testing.assert_array_equal(np.sort(X_rand, axis=0), np.sort(X, axis=0))


def test_pseudodescriptors_keep_dataframe_columns():
    frame = pd.DataFrame(X, columns=[f"d{i}" for i in range(X.shape[1])])
    a = y_randomization(Ridge(), frame, Y_SIGNAL, n_permutations=3, seed=2, variant="pseudodescriptors")
    b = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=3, seed=2, variant="pseudodescriptors")
    np.testing.assert_array_equal(a["permuted_r2"], b["permuted_r2"])


def test_fit_r2_cannot_tell_a_forest_from_chance_but_cv_r2_can():
    forest = RandomForestRegressor(n_estimators=30, random_state=0)
    fit = y_randomization(forest, X, Y_NOISE, n_permutations=5, seed=1)
    cv = y_randomization(forest, X, Y_NOISE, n_permutations=5, seed=1, statistic="cv_r2")
    assert fit["permuted_mean"] > 0.6 and cv["permuted_mean"] < 0.1


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="variant"):
        y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, variant="shuffle_x")
    with pytest.raises(ValueError, match="statistic"):
        y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, statistic="q2")

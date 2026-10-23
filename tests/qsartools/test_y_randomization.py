import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from qsartools.y_randomization import y_randomization

RNG = np.random.default_rng(0)
X = RNG.normal(size=(120, 8))
Y_SIGNAL = X @ RNG.normal(size=8) + RNG.normal(0, 0.3, 120)
Y_NOISE = RNG.normal(size=120)


def test_real_signal_is_far_above_permutations():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=30, seed=1)
    assert res["true_r2"] > 0.9 and abs(res["permuted_mean"]) < 0.2
    assert res["z"] > 10 and res["p_empirical"] == 1 / 31


def test_pure_noise_is_not_distinguishable():
    res = y_randomization(Ridge(), X, Y_NOISE, n_permutations=30, seed=1)
    assert res["p_empirical"] > 0.05 and abs(res["z"]) < 3


def test_true_score_matches_sklearn_cross_validation():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=2, n_folds=4, seed=3)
    ref = cross_val_score(Ridge(), X, Y_SIGNAL, cv=KFold(4, shuffle=True, random_state=3), scoring="r2").mean()
    assert res["true_r2"] == ref


def test_permutation_definition():
    res = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=3, seed=5)
    for i, score in enumerate(res["permuted_r2"]):
        y_perm = np.random.default_rng(5 + i).permutation(Y_SIGNAL)
        ref = cross_val_score(Ridge(), X, y_perm, cv=KFold(5, shuffle=True, random_state=5 + i), scoring="r2").mean()
        assert score == ref


def test_reproducible_and_independent_of_n_jobs():
    a = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, n_jobs=1)
    b = y_randomization(Ridge(), X, Y_SIGNAL, n_permutations=6, seed=2, n_jobs=3)
    np.testing.assert_array_equal(a["permuted_r2"], b["permuted_r2"])
    assert a["true_r2"] == b["true_r2"]


def test_estimator_is_not_modified():
    model = Ridge()
    y_randomization(model, X, Y_SIGNAL, n_permutations=2)
    assert not hasattr(model, "coef_")

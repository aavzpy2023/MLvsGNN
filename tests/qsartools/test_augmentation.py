import numpy as np
from sklearn.model_selection import GroupKFold

from qsartools.augmentation import gaussian_noise_augmentation

RNG = np.random.default_rng(0)
X = RNG.integers(0, 2, (40, 30)).astype(float)
Y = RNG.normal(6, 1, 40)


def test_shapes_order_and_groups():
    Xa, ya, g = gaussian_noise_augmentation(X, Y, noise_levels=(0.01, 0.001))
    assert Xa.shape == (120, 30) and ya.shape == (120,) and g.shape == (120,)
    np.testing.assert_array_equal(Xa[:40], X)
    np.testing.assert_array_equal(ya, np.tile(Y, 3))
    np.testing.assert_array_equal(g, np.tile(np.arange(40), 3))


def test_copies_only():
    Xa, ya, g = gaussian_noise_augmentation(X, Y, include_original=False)
    assert Xa.shape == (80, 30) and not np.array_equal(Xa[:40], X)
    np.testing.assert_array_equal(g, np.tile(np.arange(40), 2))


def test_noise_scale():
    Xa, _, _ = gaussian_noise_augmentation(np.zeros((2000, 5)), np.zeros(2000), noise_levels=(0.5, 0.01), seed=3)
    assert abs(np.std(Xa[2000:4000]) - 0.5) < 0.01 and abs(np.std(Xa[4000:]) - 0.01) < 0.0002
    assert abs(np.mean(Xa[2000:])) < 0.01


def test_reproducible_and_seed_dependent():
    a = gaussian_noise_augmentation(X, Y, seed=1)[0]
    np.testing.assert_array_equal(a, gaussian_noise_augmentation(X, Y, seed=1)[0])
    assert not np.array_equal(a, gaussian_noise_augmentation(X, Y, seed=2)[0])


def test_groups_keep_copies_together_in_group_kfold():
    Xa, ya, g = gaussian_noise_augmentation(X, Y)
    for tr, te in GroupKFold(n_splits=5).split(Xa, ya, g):
        assert not set(g[tr]) & set(g[te])


def test_input_is_not_modified():
    Xc = X.copy()
    gaussian_noise_augmentation(Xc, Y)
    np.testing.assert_array_equal(Xc, X)

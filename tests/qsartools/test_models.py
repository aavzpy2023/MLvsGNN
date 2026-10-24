import numpy as np
import pytest
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GroupKFold, KFold

from qsartools.models import (BASE_MODEL_NAMES, base_regressors, clip_bounds, clip_predictions, combination_id,
                              descriptor_pipeline, model_combinations, stacking_regressor)

RNG = np.random.default_rng(0)
X_BIN = RNG.integers(0, 2, (120, 64)).astype(float)
Y = X_BIN[:, :5].sum(1) + RNG.normal(0, 0.3, 120)
X_DESC = np.hstack([RNG.normal(size=(120, 10)), np.full((120, 1), 3.0)])
X_DESC[RNG.integers(0, 120, 15), RNG.integers(0, 10, 15)] = np.nan


def test_base_regressors_and_combinations():
    base = base_regressors(n_estimators=10)
    assert tuple(base) == BASE_MODEL_NAMES
    combos = model_combinations(base, max_size=3)
    assert len(combos) == 6 + 15 + 20
    assert combination_id(combos[0]) == "RF" and combination_id(combos[6]) == "RF+ET" and len(combos[-1]) == 3


def test_stacking_regressor_returns_fresh_estimators():
    base = base_regressors(n_estimators=10)
    single = stacking_regressor({"SVM": base["SVM"]})
    assert single is not base["SVM"] and not isinstance(single, StackingRegressor)
    stack = stacking_regressor({"RF": base["RF"], "SVM": base["SVM"]})
    assert isinstance(stack, StackingRegressor) and [n for n, _ in stack.estimators] == ["RF", "SVM"]
    assert stack.estimators[0][1] is not base["RF"]


def test_stacking_fits_and_is_reproducible():
    base = base_regressors(n_estimators=20, seed=3)
    combo = {k: base[k] for k in ("RF", "XGB", "SVM")}
    p1 = stacking_regressor(combo).fit(X_BIN[:100], Y[:100]).predict(X_BIN[100:])
    p2 = stacking_regressor(combo).fit(X_BIN[:100], Y[:100]).predict(X_BIN[100:])
    np.testing.assert_array_equal(p1, p2)
    assert np.corrcoef(p1, Y[100:])[0, 1] > 0.5


def test_stacking_accepts_explicit_grouped_splits():
    groups = np.repeat(np.arange(40), 3)
    splits = list(GroupKFold(n_splits=5).split(X_BIN, Y, groups))
    base = base_regressors(n_estimators=10)
    stack = stacking_regressor({"RF": base["RF"], "SVM": base["SVM"]}, cv=splits).fit(X_BIN, Y)
    assert stack.predict(X_BIN[:3]).shape == (3,)


def test_descriptor_pipeline():
    base = base_regressors(n_estimators=10)
    assert [s for s, _ in descriptor_pipeline(base["SVM"], binary_features=True).steps] == ["model"]
    pipe = descriptor_pipeline(base["SVM"], binary_features=False, selector_trees=20).fit(X_DESC, Y)
    assert [s for s, _ in pipe.steps] == ["impute", "scale", "select", "model"]
    assert not pipe.named_steps["select"].get_support()[-1], "a constant feature has zero importance and is dropped"
    assert np.isfinite(pipe.predict(X_DESC)).all()


def test_clipping():
    lo, hi = clip_bounds([2.0, 5.0, 9.0], margin=3.0)
    assert (lo, hi) == (-1.0, 12.0)
    clipped, n = clip_predictions([-5.0, 0.0, 7.0, 50.0], lo, hi)
    np.testing.assert_array_equal(clipped, [-1.0, 0.0, 7.0, 12.0])
    assert n == 2


def test_stacking_shuffle_only_applies_to_a_number_of_folds():
    base = base_regressors(n_estimators=10)
    combo = {"RF": base["RF"], "SVM": base["SVM"]}
    assert stacking_regressor(combo, shuffle=False).cv == 5
    assert isinstance(stacking_regressor(combo).cv, KFold) and stacking_regressor(combo).cv.shuffle
    shuffled = stacking_regressor(combo, seed=7).cv
    assert isinstance(shuffled, KFold) and shuffled.shuffle and shuffled.random_state == 7 and shuffled.n_splits == 5
    splits = list(GroupKFold(n_splits=3).split(X_BIN, Y, np.arange(120) % 30))
    assert stacking_regressor(combo, cv=splits, shuffle=True).cv is splits

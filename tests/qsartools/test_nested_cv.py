import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from qsartools.nested_cv import inner_cv_scores, nested_cv, repeat_scores

RNG = np.random.default_rng(0)
X = RNG.normal(size=(120, 6))
Y = X @ np.array([1.5, -2.0, 0.5, 0.0, 0.0, 1.0]) + RNG.normal(0, 0.5, 120)
OUTER = list(KFold(4, shuffle=True, random_state=1).split(X))


class RecordingRegressor(BaseEstimator, RegressorMixin):
    """Predicts the training mean and records, in a class-level list, the row ids (column 0) of every fit."""
    seen = []

    def fit(self, X, y):
        RecordingRegressor.seen.extend(np.asarray(X)[:, 0].astype(int).tolist())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def test_inner_scores_one_row_per_candidate_in_input_order():
    scores = inner_cv_scores({"mean": DummyRegressor(), "linear": LinearRegression()}, X, Y, n_folds=5, n_repeats=2)
    assert list(scores["Model"]) == ["mean", "linear"]
    assert list(scores.columns) == ["Model", "R2_inner_mean", "R2_inner_std", "MAE_inner_mean"]
    assert scores.loc[1, "R2_inner_mean"] > 0.9 > scores.loc[0, "R2_inner_mean"]


def test_pooled_score_is_the_r2_of_all_out_of_fold_predictions_of_a_repeat():
    scores = inner_cv_scores({"linear": LinearRegression()}, X, Y, n_folds=5, n_repeats=1, seed=3)
    oof = cross_val_predict(LinearRegression(), X, Y, cv=KFold(5, shuffle=True, random_state=3))
    assert scores.loc[0, "R2_inner_mean"] == pytest.approx(r2_score(Y, oof), abs=1e-12)
    assert scores.loc[0, "R2_inner_std"] == 0.0


def test_fold_mean_with_one_repeat_is_the_mean_of_kfold_scores():
    scores = inner_cv_scores({"linear": LinearRegression()}, X, Y, n_folds=5, n_repeats=1, seed=3,
                             inner_score="fold_mean")
    per_fold = [r2_score(Y[te], LinearRegression().fit(X[tr], Y[tr]).predict(X[te]))
                for tr, te in KFold(5, shuffle=True, random_state=3).split(X)]
    assert scores.loc[0, "R2_inner_mean"] == pytest.approx(np.mean(per_fold), abs=1e-12)


def test_repeats_change_the_partitions():
    one = inner_cv_scores({"rf": RandomForestRegressor(n_estimators=5, random_state=0)}, X, Y, n_repeats=1)
    three = inner_cv_scores({"rf": RandomForestRegressor(n_estimators=5, random_state=0)}, X, Y, n_repeats=3)
    assert three.loc[0, "R2_inner_std"] > 0 and three.loc[0, "R2_inner_mean"] != one.loc[0, "R2_inner_mean"]


def test_selects_the_better_candidate_in_every_outer_fold():
    records, predictions = nested_cv({"mean": DummyRegressor(), "linear": LinearRegression()}, X, Y, OUTER,
                                     inner_repeats=2)
    assert [r["Fold"] for r in records] == [0, 1, 2, 3]
    assert all(r["Selected_Model"] == "linear" and r["R2_outer"] > 0.9 for r in records)
    assert [r["N_test"] for r in records] == [len(te) for _, te in OUTER]
    assert sorted(predictions["row"]) == list(range(len(Y)))


def test_select_by_mae():
    records, _ = nested_cv({"mean": DummyRegressor(), "linear": LinearRegression()}, X, Y, OUTER[:1],
                           inner_repeats=1, select_by="mae")
    assert records[0]["Selected_Model"] == "linear"


def test_ties_go_to_the_first_candidate():
    for order in (("simple", "complex"), ("complex", "simple")):
        candidates = {name: LinearRegression() for name in order}
        records, _ = nested_cv(candidates, X, Y, OUTER[:1], inner_repeats=2)
        assert records[0]["Selected_Model"] == order[0]


def test_outer_test_rows_never_reach_any_fit_of_their_fold():
    ids = np.column_stack([np.arange(len(Y)), X])
    RecordingRegressor.seen.clear()
    leaked = []

    def check(record, ranking, model, predictions):
        test_rows = set(OUTER[record["Fold"]][1].tolist())
        leaked.append(len(test_rows & set(RecordingRegressor.seen)))
        RecordingRegressor.seen.clear()

    nested_cv({"spy": RecordingRegressor()}, ids, Y, OUTER, inner_repeats=2, n_jobs=1, on_fold=check)
    assert leaked == [0, 0, 0, 0]


def test_results_do_not_depend_on_n_jobs():
    candidates = {"ridge": Ridge(), "rf": RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=1)}

    def run(n_jobs):
        records, predictions = nested_cv(candidates, X, Y, OUTER, inner_repeats=2, n_jobs=n_jobs)
        return [{k: v for k, v in r.items() if k != "Time_s"} for r in records], predictions

    (r1, p1), (r2, p2) = run(1), run(2)
    assert r1 == r2 and p1.equals(p2)


def test_skip_folds_and_callback():
    calls = []
    records, predictions = nested_cv({"linear": LinearRegression()}, X, Y, OUTER, inner_repeats=1, skip_folds={0, 2},
                                     on_fold=lambda record, ranking, model, p: calls.append((record["Fold"], len(p))))
    assert [r["Fold"] for r in records] == [1, 3]
    assert calls == [(1, len(OUTER[1][1])), (3, len(OUTER[3][1]))]
    assert set(predictions["Fold"]) == {1, 3}


def test_make_estimator_is_called_for_every_fit():
    built = []

    def make(candidate):
        built.append(candidate)
        return LinearRegression()

    nested_cv({"a": "spec-a", "b": "spec-b"}, X, Y, OUTER[:1], inner_folds=3, inner_repeats=2, make_estimator=make)
    assert len(built) == 2 * 3 * 2 + 1


def test_clipping_uses_the_training_range():
    far = ConstantRegressor(value=1e6)
    unclipped = nested_cv({"far": far}, X, Y, OUTER[:1], inner_repeats=1)[0][0]
    clipped = nested_cv({"far": far}, X, Y, OUTER[:1], inner_repeats=1, clip_margin=1.0)[0][0]
    assert unclipped["N_predictions_clipped"] == 0
    assert clipped["N_predictions_clipped"] == len(OUTER[0][1])
    assert clipped["MAE_outer"] < unclipped["MAE_outer"]


def test_repeat_scores_pool_the_folds_of_each_repeat():
    outer = [s for seed in (1, 2) for s in KFold(4, shuffle=True, random_state=seed).split(X)]
    _, predictions = nested_cv({"linear": LinearRegression()}, X, Y, outer, inner_repeats=1)
    per_repeat, summary = repeat_scores(predictions, folds_per_repeat=4)
    assert list(per_repeat["N"]) == [len(Y), len(Y)]
    first = predictions[predictions["Fold"] < 4]
    assert per_repeat.loc[0, "R2"] == pytest.approx(r2_score(first["y_true"], first["y_pred"]))
    assert summary["R2_min"] <= summary["R2_mean"] <= summary["R2_max"]


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="inner_score"):
        nested_cv({"linear": LinearRegression()}, X, Y, OUTER, inner_score="median")
    with pytest.raises(ValueError, match="select_by"):
        nested_cv({"linear": LinearRegression()}, X, Y, OUTER, select_by="rmse")

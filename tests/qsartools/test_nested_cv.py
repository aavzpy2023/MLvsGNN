import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from qsartools.nested_cv import inner_cv_scores, nested_cv

RNG = np.random.default_rng(0)
X = RNG.normal(size=(120, 6))
Y = X @ np.array([1.5, -2.0, 0.5, 0.0, 0.0, 1.0]) + RNG.normal(0, 0.1, 120)
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
    scores = inner_cv_scores({"mean": DummyRegressor(), "linear": LinearRegression()}, X, Y, n_folds=5)
    assert list(scores["Model"]) == ["mean", "linear"]
    assert list(scores.columns) == ["Model", "R2_inner_mean", "R2_inner_std", "MAE_inner_mean"]
    assert scores.loc[1, "R2_inner_mean"] > 0.95 > scores.loc[0, "R2_inner_mean"]


def test_selects_the_better_candidate_in_every_outer_fold():
    records = nested_cv({"mean": DummyRegressor(), "linear": LinearRegression()}, X, Y, OUTER)
    assert [r["Fold"] for r in records] == [0, 1, 2, 3]
    assert all(r["Selected_Model"] == "linear" and r["R2_outer"] > 0.95 for r in records)
    assert [r["N_test"] for r in records] == [len(te) for _, te in OUTER]


def test_outer_test_rows_never_reach_any_fit_of_their_fold():
    ids = np.column_stack([np.arange(len(Y)), X])
    RecordingRegressor.seen.clear()
    leaked = []

    def check(record, ranking, model):
        test_rows = set(OUTER[record["Fold"]][1].tolist())
        leaked.append(len(test_rows & set(RecordingRegressor.seen)))
        RecordingRegressor.seen.clear()

    nested_cv({"spy": RecordingRegressor()}, ids, Y, OUTER, n_jobs=1, on_fold=check)
    assert leaked == [0, 0, 0, 0]


def test_results_do_not_depend_on_n_jobs():
    candidates = {"rf": RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=1), "ridge": Ridge()}

    def strip(records):
        return [{k: v for k, v in r.items() if k != "Time_s"} for r in records]

    assert strip(nested_cv(candidates, X, Y, OUTER, n_jobs=1)) == strip(nested_cv(candidates, X, Y, OUTER, n_jobs=2))


def test_skip_folds_and_callback():
    calls = []
    records = nested_cv({"linear": LinearRegression()}, X, Y, OUTER, skip_folds={0, 2},
                        on_fold=lambda record, ranking, model: calls.append((record["Fold"], len(ranking))))
    assert [r["Fold"] for r in records] == [1, 3] and calls == [(1, 1), (3, 1)]


def test_make_estimator_is_called_for_every_fit():
    built = []

    def make(candidate):
        built.append(candidate)
        return LinearRegression()

    nested_cv({"a": "spec-a", "b": "spec-b"}, X, Y, OUTER[:1], inner_folds=3, make_estimator=make)
    assert sorted(built).count("spec-a") >= 3 and len(built) == 2 * 3 + 1


def test_clipping_uses_the_training_range():
    far = ConstantRegressor(value=1e6)
    unclipped = nested_cv({"far": far}, X, Y, OUTER[:1])[0]
    clipped = nested_cv({"far": far}, X, Y, OUTER[:1], clip_margin=1.0)[0]
    assert unclipped["N_predictions_clipped"] == 0
    assert clipped["N_predictions_clipped"] == len(OUTER[0][1])
    assert clipped["MAE_outer"] < unclipped["MAE_outer"]

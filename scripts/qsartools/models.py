"""Regression models for tabular molecular features: base learners, stacking ensembles of them, the descriptor
preprocessing pipeline and a prediction-clipping safeguard."""
import itertools

import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

BASE_MODEL_NAMES = ("RF", "ET", "LGBM", "XGB", "SVM", "kNN")


def base_regressors(n_estimators=200, seed=42, n_jobs=1):
    """Six untuned base regressors with fixed hyperparameters, keyed by BASE_MODEL_NAMES:
    Random Forest, Extra Trees, LightGBM and XGBoost (n_estimators trees each), SVR (RBF kernel, C=10, epsilon=0.1) and
    k-nearest neighbours (k=5, cosine distance).

    Args:
        n_estimators: number of trees of RF, ET, LightGBM and XGBoost.
        seed: random_state of the four tree ensembles (SVR and kNN are deterministic).
        n_jobs: threads per model. 1 keeps tree predictions bit-reproducible: a multi-threaded forest sums its trees in
            a thread-dependent order.
    """
    return {
        "RF": RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs),
        "ET": ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs),
        "LGBM": lgb.LGBMRegressor(n_estimators=n_estimators, random_state=seed, verbosity=-1, n_jobs=n_jobs),
        "XGB": XGBRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, verbosity=0),
        "SVM": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
        "kNN": KNeighborsRegressor(n_neighbors=5, metric="cosine", n_jobs=n_jobs),
    }


def model_combinations(base_models, max_size=3):
    """Every combination of 1 to max_size base models, as dicts name -> estimator, in itertools order
    (all singles, then all pairs, then all triples). With the six base regressors and max_size=3 there are 41.

    Args:
        base_models: dict name -> unfitted estimator (e.g. the output of base_regressors).
        max_size: largest number of members in a combination.
    """
    return [dict(combo) for k in range(1, max_size + 1) for combo in itertools.combinations(base_models.items(), k)]


def combination_id(combo):
    """'RF+XGB+SVM' for a combination dict.

    Args:
        combo: dict name -> estimator; only its keys (in order) are used.
    """
    return "+".join(combo)


def stacking_regressor(combo, cv=5, n_jobs=1, shuffle=False, seed=42):
    """A fresh (cloned) estimator for a combination: the model itself if it has one member, otherwise a stacking ensemble
    whose RidgeCV meta-learner is fitted on out-of-fold predictions of the members (Wolpert, 1992; Breiman, 1996).

    With an integer `cv`, scikit-learn splits the rows into contiguous blocks in their input order. If the rows are
    ordered by anything related to the target (potency, data source, publication date), each block then covers a
    different part of the target range, the out-of-fold predictions are biased block by block, and the meta-learner
    is fitted on those biased predictions. shuffle=True removes that dependence on row order.

    Args:
        combo: dict name -> unfitted estimator; the names become the StackingRegressor step names (so a member's
            parameter is set with set_params(name__param=...)).
        cv: inner cross-validation of StackingRegressor that produces the out-of-fold predictions: a number of folds,
            or an explicit list of (train, test) index pairs (e.g. grouped splits for augmented data).
        n_jobs: number of members fitted in parallel by StackingRegressor.
        shuffle: if True and `cv` is a number of folds, the folds are drawn after shuffling the rows
            (KFold(cv, shuffle=True, random_state=seed)); ignored when `cv` is an explicit list of splits.
        seed: random_state of that shuffle.
    """
    items = list(combo.items())
    if len(items) == 1:
        return clone(items[0][1])
    if shuffle and isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
    return StackingRegressor(estimators=[(n, clone(m)) for n, m in items], final_estimator=RidgeCV(), cv=cv, n_jobs=n_jobs)


def descriptor_pipeline(estimator, binary_features, selector_trees=200, seed=42):
    """Wrap an estimator in the preprocessing appropriate to its input, fitted inside whatever data it is fitted on.

    Binary fingerprints are passed unchanged. Continuous descriptors go through median imputation, standardization and
    feature selection that keeps every feature with a non-zero importance in a Random Forest (threshold 1e-9).

    Args:
        estimator: final regressor of the pipeline (step name "model").
        binary_features: True for fingerprints (no preprocessing), False for continuous descriptors.
        selector_trees: number of trees of the Random Forest used by the feature selector.
        seed: random_state of that Random Forest.
    """
    if binary_features:
        return Pipeline([("model", estimator)])
    selector = SelectFromModel(RandomForestRegressor(n_estimators=selector_trees, random_state=seed, n_jobs=1), threshold=1e-9)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("select", selector),
                     ("model", estimator)])


def clip_bounds(y_train, margin=3.0):
    """(lower, upper) = training range of the target extended by `margin` units on both sides.

    Args:
        y_train: target values of the training set.
        margin: extension beyond the training minimum and maximum, in target units.
    """
    return float(np.min(y_train) - margin), float(np.max(y_train) + margin)


def clip_predictions(y_pred, lower, upper):
    """Predictions clipped to [lower, upper] and the number of predictions that had to be clipped. A safeguard against
    models that extrapolate without bound (e.g. after a non-linear feature transformation of an outlying molecule).

    Args:
        y_pred: predictions.
        lower: lowest allowed value (e.g. from clip_bounds).
        upper: highest allowed value.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    return np.clip(y_pred, lower, upper), int(((y_pred < lower) | (y_pred > upper)).sum())

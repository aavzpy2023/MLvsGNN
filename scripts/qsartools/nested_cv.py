"""Nested cross-validation: model selection by an inner cross-validation inside the training part of every outer fold,
so that the outer test parts, which give the reported performance, never influence which model is chosen (Varma &
Simon, 2006, BMC Bioinformatics 7:91; Cawley & Talbot, 2010, J. Mach. Learn. Res. 11:2079-2107; repeated nested
cross-validation for QSAR: Krstajic et al., 2014, J. Cheminform. 6:10).

The score obtained is that of the whole procedure (select by inner CV, refit on the outer training part, predict the
outer test part), not of one fixed model: each outer fold may select a different candidate.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from qsartools.models import clip_bounds, clip_predictions


def _build(candidate, make_estimator):
    return make_estimator(candidate) if make_estimator is not None else clone(candidate)


def _fit_predict(estimator, X_fit, y_fit, X_eval, clip_margin):
    estimator.fit(X_fit, y_fit)
    pred = np.asarray(estimator.predict(X_eval), dtype=float).ravel()
    if clip_margin is None:
        return pred, 0
    return clip_predictions(pred, *clip_bounds(y_fit, clip_margin))


def _inner_task(candidate, make_estimator, X, y, train_idx, val_idx, clip_margin):
    pred, _ = _fit_predict(_build(candidate, make_estimator), X[train_idx], y[train_idx], X[val_idx], clip_margin)
    return r2_score(y[val_idx], pred), mean_absolute_error(y[val_idx], pred)


def inner_cv_scores(candidates, X, y, n_folds=5, seed=42, make_estimator=None, clip_margin=None, n_jobs=1,
                    task_cost=None):
    """K-fold cross-validated R2 and MAE of every candidate on (X, y): the selection step of nested_cv.

    The work is parallelized over (candidate, fold) pairs, so that no process waits behind a few expensive candidates;
    every fit is independent, so the scores do not depend on n_jobs.

    Args:
        candidates: dict name -> candidate. A candidate is an unfitted estimator, or anything `make_estimator` turns
            into one (e.g. a dict of base models for qsartools.models.stacking_regressor).
        X: (n, d) feature matrix (the training part of an outer fold, when called by nested_cv).
        y: n target values.
        n_folds: number of folds of the KFold.
        seed: random_state of the KFold shuffle.
        make_estimator: callable candidate -> fresh unfitted estimator, called for every fit (e.g. to wrap the model in
            a preprocessing pipeline that is then fitted inside each fold). None clones the candidate.
        clip_margin: if given, predictions are clipped to the training range of the target extended by this many
            units (qsartools.models.clip_bounds) before scoring; None leaves them unclipped.
        n_jobs: number of joblib processes.
        task_cost: optional callable candidate -> relative cost; the most expensive tasks are dispatched first.

    Returns:
        DataFrame with one row per candidate, in the input order: Model (the name), R2_inner_mean, R2_inner_std
        (population standard deviation over folds) and MAE_inner_mean.
    """
    X, y = np.asarray(X), np.asarray(y)
    names = list(candidates)
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    tasks = [(ci, fi) for ci in range(len(names)) for fi in range(len(splits))]
    if task_cost is not None:
        tasks.sort(key=lambda t: -task_cost(candidates[names[t[0]]]))
    out = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_inner_task)(candidates[names[ci]], make_estimator, X, y, *splits[fi], clip_margin) for ci, fi in tasks)
    scores = dict(zip(tasks, out))
    rows = []
    for ci, name in enumerate(names):
        r2 = np.array([scores[(ci, fi)][0] for fi in range(len(splits))])
        mae = np.array([scores[(ci, fi)][1] for fi in range(len(splits))])
        rows.append({"Model": name, "R2_inner_mean": r2.mean(), "R2_inner_std": r2.std(), "MAE_inner_mean": mae.mean()})
    return pd.DataFrame(rows)


def nested_cv(candidates, X, y, outer_splits, inner_folds=5, seed=42, make_estimator=None, clip_margin=None, n_jobs=1,
              task_cost=None, skip_folds=(), on_fold=None):
    """Nested cross-validation over the given outer folds.

    In every outer fold: all candidates are scored by inner_cv_scores on the outer training part only; the one with
    the highest mean inner R2 is refitted on the whole outer training part and predicts the outer test part, which
    gives that fold's R2 and MAE.

    Args:
        candidates: dict name -> candidate, as for inner_cv_scores.
        X: (n, d) feature matrix of all samples.
        y: n target values.
        outer_splits: iterable of (train_idx, test_idx) pairs, e.g. qsartools.splits.outer_splits; pass the same
            folds to every model that is compared, so that the comparison is paired fold by fold.
        inner_folds: number of folds of the inner KFold.
        seed: random_state of the inner KFold shuffle (the same in every outer fold).
        make_estimator: callable candidate -> fresh unfitted estimator, as for inner_cv_scores.
        clip_margin: prediction clipping margin, as for inner_cv_scores; the bounds always come from the targets the
            model is fitted on.
        n_jobs: number of joblib processes of the inner cross-validation.
        task_cost: optional callable candidate -> relative cost, as for inner_cv_scores.
        skip_folds: indices of outer folds not to run (e.g. those already in a checkpoint).
        on_fold: optional callable on_fold(record, ranking, model) called after each outer fold, with the record
            returned for that fold, the inner scores sorted from best to worst and the refitted selected model, so
            that the caller can checkpoint it or extract more information from the model.

    Returns:
        List with one dict per outer fold run: Fold, Selected_Model, R2_outer, MAE_outer, N_train, N_test,
        N_predictions_clipped and Time_s (selection plus refit, in seconds).
    """
    X, y = np.asarray(X), np.asarray(y)
    records = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        if fold in skip_folds:
            continue
        start = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        ranking = inner_cv_scores(candidates, X_tr, y_tr, inner_folds, seed, make_estimator, clip_margin, n_jobs,
                                  task_cost).sort_values(by="R2_inner_mean", ascending=False)
        best = ranking.iloc[0]["Model"]
        model = _build(candidates[best], make_estimator)
        pred, n_clipped = _fit_predict(model, X_tr, y_tr, X[test_idx], clip_margin)
        record = {"Fold": fold, "Selected_Model": best, "R2_outer": r2_score(y[test_idx], pred),
                  "MAE_outer": mean_absolute_error(y[test_idx], pred), "N_train": len(train_idx),
                  "N_test": len(test_idx), "N_predictions_clipped": n_clipped, "Time_s": time.time() - start}
        records.append(record)
        if on_fold is not None:
            on_fold(record, ranking, model)
    return records

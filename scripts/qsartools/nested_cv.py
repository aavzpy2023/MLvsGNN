"""Repeated nested cross-validation: model selection by a (repeated) inner cross-validation inside the training part of
every outer fold, so that the outer test parts, which give the reported performance, never influence which model is
chosen.

The procedure follows Krstajic et al. (2014, J. Cheminform. 6:10): Algorithm 1 (selection by repeated grid-search
cross-validation) runs inside every outer fold of Algorithm 2 (repeated nested cross-validation), the "cross-validatory
assessment of the cross-validatory choice" of Stone (1974) whose bias without nesting was shown by Varma & Simon (2006,
BMC Bioinformatics 7:91). The variance of the selection criterion, reduced by repeating the inner cross-validation, is
what lets model selection over-fit (Cawley & Talbot, 2010, J. Mach. Learn. Res. 11:2079-2107).

The score obtained is that of the whole protocol (select, refit on the outer training part, predict the outer test
part), not of one fixed model: each outer fold may select a different candidate.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

from qsartools.models import clip_bounds, clip_predictions

INNER_SCORES = ("pooled", "fold_mean")
SELECT_BY = ("r2", "mae")


def _build(candidate, make_estimator):
    return make_estimator(candidate) if make_estimator is not None else clone(candidate)


def _fit_predict(estimator, X_fit, y_fit, X_eval, clip_margin):
    estimator.fit(X_fit, y_fit)
    pred = np.asarray(estimator.predict(X_eval), dtype=float).ravel()
    if clip_margin is None:
        return pred, 0
    return clip_predictions(pred, *clip_bounds(y_fit, clip_margin))


def _inner_task(candidate, make_estimator, X, y, train_idx, val_idx, clip_margin):
    return _fit_predict(_build(candidate, make_estimator), X[train_idx], y[train_idx], X[val_idx], clip_margin)[0]


def _check_option(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, not {value!r}")


def inner_cv_scores(candidates, X, y, n_folds=5, n_repeats=5, seed=42, inner_score="pooled", make_estimator=None,
                    clip_margin=None, n_jobs=1, task_cost=None):
    """Repeated K-fold cross-validated R2 and MAE of every candidate on (X, y): the selection step of nested_cv
    (Krstajic et al., 2014, Algorithm 1).

    With inner_score="pooled" (Algorithm 1, step 1c), the out-of-fold predictions of one repeat, one per sample, are
    scored together; the scores of the n_repeats repeats are then averaged (step 2). With "fold_mean", every fold is
    scored on its own and the n_folds x n_repeats fold scores are averaged (each fold then uses its own target mean as
    the R2 reference). The work is parallelized over (candidate, fold) pairs; every fit is independent, so the scores do
    not depend on n_jobs.

    Args:
        candidates: dict name -> candidate, ordered from the simplest to the most complex (the order breaks ties in
            nested_cv). A candidate is an unfitted estimator, or anything `make_estimator` turns into one (e.g. a dict
            of base models for qsartools.models.stacking_regressor).
        X: (n, d) feature matrix (the training part of an outer fold, when called by nested_cv).
        y: n target values.
        n_folds: number of folds per repeat (V1 in Krstajic et al.).
        n_repeats: number of repeats with different random partitions (Nexp1 in Krstajic et al.).
        seed: random_state of the RepeatedKFold; with n_repeats=1 its folds equal KFold(n_folds, shuffle=True,
            random_state=seed).
        inner_score: "pooled" (Krstajic et al.) or "fold_mean" (mean of per-fold scores).
        make_estimator: callable candidate -> fresh unfitted estimator, called for every fit (e.g. to wrap the model in
            a preprocessing pipeline that is then fitted inside each fold). None clones the candidate.
        clip_margin: if given, predictions are clipped to the range of the targets the model is fitted on, extended by
            this many units (qsartools.models.clip_bounds), before scoring; None leaves them unclipped.
        n_jobs: number of joblib processes.
        task_cost: optional callable candidate -> relative cost; the most expensive tasks are dispatched first.

    Returns:
        DataFrame with one row per candidate, in the input order: Model (the name), R2_inner_mean, R2_inner_std and
        MAE_inner_mean. With "pooled" the mean and the population standard deviation are over repeats; with
        "fold_mean" they are over all folds of all repeats.
    """
    _check_option("inner_score", inner_score, INNER_SCORES)
    X, y = np.asarray(X), np.asarray(y)
    names = list(candidates)
    splits = list(RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed).split(X))
    tasks = [(ci, si) for ci in range(len(names)) for si in range(len(splits))]
    if task_cost is not None:
        tasks.sort(key=lambda t: -task_cost(candidates[names[t[0]]]))
    out = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_inner_task)(candidates[names[ci]], make_estimator, X, y, *splits[si], clip_margin) for ci, si in tasks)
    preds = dict(zip(tasks, out))
    rows = []
    for ci, name in enumerate(names):
        if inner_score == "pooled":
            r2, mae = [], []
            for r in range(n_repeats):
                oof = np.empty(len(y))
                for si in range(r * n_folds, (r + 1) * n_folds):
                    oof[splits[si][1]] = preds[(ci, si)]
                r2.append(r2_score(y, oof))
                mae.append(mean_absolute_error(y, oof))
        else:
            r2 = [r2_score(y[splits[si][1]], preds[(ci, si)]) for si in range(len(splits))]
            mae = [mean_absolute_error(y[splits[si][1]], preds[(ci, si)]) for si in range(len(splits))]
        r2, mae = np.array(r2), np.array(mae)
        rows.append({"Model": name, "R2_inner_mean": r2.mean(), "R2_inner_std": r2.std(), "MAE_inner_mean": mae.mean()})
    return pd.DataFrame(rows)


def nested_cv(candidates, X, y, outer_splits, inner_folds=5, inner_repeats=5, seed=42, inner_score="pooled",
              select_by="r2", make_estimator=None, clip_margin=None, n_jobs=1, task_cost=None, skip_folds=(),
              on_fold=None):
    """Repeated nested cross-validation over the given outer folds (Krstajic et al., 2014, Algorithm 2).

    In every outer fold: all candidates are scored by inner_cv_scores on the outer training part only; the best one
    (highest mean inner R2, or lowest mean inner MAE; on a tie, the first in the order of `candidates`, i.e. the
    simplest) is refitted on the whole outer training part and predicts the outer test part.

    Args:
        candidates: dict name -> candidate, as for inner_cv_scores, ordered from the simplest to the most complex.
        X: (n, d) feature matrix of all samples.
        y: n target values.
        outer_splits: iterable of (train_idx, test_idx) pairs, e.g. qsartools.splits.outer_splits (V2 folds repeated
            Nexp2 times in Krstajic et al.). Pass the same folds to every model that is compared, so that the
            comparison is paired fold by fold.
        inner_folds: number of folds per inner repeat (V1).
        inner_repeats: number of inner repeats (Nexp1); Krstajic et al. used V1 = V2 = 10 and Nexp1 = Nexp2 = 50.
        seed: random_state of the inner RepeatedKFold (the same in every outer fold).
        inner_score: "pooled" or "fold_mean", as for inner_cv_scores.
        select_by: "r2" (highest mean inner R2) or "mae" (lowest mean inner MAE).
        make_estimator: callable candidate -> fresh unfitted estimator, as for inner_cv_scores.
        clip_margin: prediction clipping margin, as for inner_cv_scores; the bounds always come from the targets the
            model is fitted on.
        n_jobs: number of joblib processes of the inner cross-validation.
        task_cost: optional callable candidate -> relative cost, as for inner_cv_scores.
        skip_folds: indices of outer folds not to run (e.g. those already in a checkpoint).
        on_fold: optional callable on_fold(record, ranking, model, predictions) called after each outer fold, with the
            record of that fold, the inner scores sorted from best to worst, the refitted selected model and a
            DataFrame of its outer test predictions (Fold, row, y_true, y_pred), so that the caller can checkpoint
            them or extract more information from the model.

    Returns:
        (records, predictions): records is a list with one dict per outer fold run (Fold, Selected_Model, R2_outer,
        MAE_outer, N_train, N_test, N_predictions_clipped, Time_s = selection plus refit in seconds); predictions is
        a DataFrame of every outer test prediction (Fold, row, y_true, y_pred), the input of repeat_scores.
    """
    _check_option("inner_score", inner_score, INNER_SCORES)
    _check_option("select_by", select_by, SELECT_BY)
    X, y = np.asarray(X), np.asarray(y)
    sort_col, ascending = ("R2_inner_mean", False) if select_by == "r2" else ("MAE_inner_mean", True)
    records, all_preds = [], []
    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        if fold in skip_folds:
            continue
        start = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        # mergesort is stable: among equal scores the earlier (simpler) candidate stays first
        ranking = inner_cv_scores(candidates, X_tr, y_tr, inner_folds, inner_repeats, seed, inner_score, make_estimator,
                                  clip_margin, n_jobs, task_cost).sort_values(by=sort_col, ascending=ascending,
                                                                              kind="mergesort")
        best = ranking.iloc[0]["Model"]
        model = _build(candidates[best], make_estimator)
        pred, n_clipped = _fit_predict(model, X_tr, y_tr, X[test_idx], clip_margin)
        record = {"Fold": fold, "Selected_Model": best, "R2_outer": r2_score(y[test_idx], pred),
                  "MAE_outer": mean_absolute_error(y[test_idx], pred), "N_train": len(train_idx),
                  "N_test": len(test_idx), "N_predictions_clipped": n_clipped, "Time_s": time.time() - start}
        fold_preds = pd.DataFrame({"Fold": fold, "row": test_idx, "y_true": y[test_idx], "y_pred": pred})
        records.append(record)
        all_preds.append(fold_preds)
        if on_fold is not None:
            on_fold(record, ranking, model, fold_preds)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(
        columns=["Fold", "row", "y_true", "y_pred"])
    return records, predictions


def repeat_scores(predictions, folds_per_repeat):
    """Nested cross-validation error of every outer repeat, scored on all its test predictions together, and their
    summary (Krstajic et al., 2014, Algorithm 2, steps 2d and 3).

    Args:
        predictions: DataFrame with columns Fold, y_true and y_pred, e.g. from nested_cv; folds must be numbered
            repeat by repeat (0..V2-1 are the first repeat), as qsartools.splits.outer_splits yields them.
        folds_per_repeat: number of outer folds per repeat (V2).

    Returns:
        (per_repeat, summary): per_repeat is a DataFrame with one row per repeat (Repeat, R2, MAE, N); summary is a dict
        with the mean, minimum and maximum of R2 and of MAE over repeats (the "P-estimate" and "P-estimated interval"
        of Krstajic et al.).
    """
    repeat = np.asarray(predictions["Fold"]) // folds_per_repeat
    rows = []
    for r in np.unique(repeat):
        p = predictions[repeat == r]
        rows.append({"Repeat": int(r), "R2": r2_score(p["y_true"], p["y_pred"]),
                     "MAE": mean_absolute_error(p["y_true"], p["y_pred"]), "N": len(p)})
    per_repeat = pd.DataFrame(rows)
    summary = {f"{m}_{s}": float(getattr(per_repeat[m], s)()) for m in ("R2", "MAE") for s in ("mean", "min", "max")}
    return per_repeat, summary

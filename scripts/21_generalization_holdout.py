"""Score the generalization cohort on the partition its authors published, and separately on the activity cliffs.

The nested cross-validation of step 01 answers how the models behave on average over resamplings of one data set. This
step answers a different question: on the single 80/20 split that MoleculeACE distributes, which is the partition every
paper using these sets reports on, how do the classical ensemble and the D-MPNN compare, and does the comparison change
on the compounds that belong to an activity cliff?

Both questions are worth reporting because they can disagree: a model can win on average over random partitions and
lose on one particular partition, and the cliff compounds are exactly where a fingerprint representation is expected to
struggle, since two molecules that differ in one substituent share almost all their bits while differing in potency.

Model selection stays inside the training partition: the combination of base learners is chosen by cross-validation on
the training compounds alone, so the published test set is never used to build the model it scores.
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths_config import *  # noqa: F401,F403
from logger_utils import setup_logger, explain
from qsartools.featurization import morgan_fingerprints
from qsartools.models import (base_regressors, clip_bounds, clip_predictions, combination_id, descriptor_pipeline,
                              model_combinations, stacking_regressor)
from qsartools.nested_cv import inner_cv_scores
from qsartools.parallel import SAFE_CPU_ENV, run_tasks

import gnn_utils

MOLECULEACE_DIR = os.path.join(DATA_DIR, "moleculeace")
INDEX_CSV = os.path.join(MOLECULEACE_DIR, "index.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "21_generalization_holdout.csv")
PRED_CSV = os.path.join(RESULTS_DIR, "21_generalization_holdout_predictions.csv")
# Same margin as the nested cross-validation of step 01, so that the two protocols clip predictions identically.
CLIP_MARGIN = 3.0


def make_estimator(combo):
    """A stacking ensemble of `combo` behind the preprocessing for binary fingerprint input, as in step 01."""
    return descriptor_pipeline(stacking_regressor(combo, cv=5), binary_features=True,
                               selector_trees=CFG['N_ESTIMATORS_TREES'], seed=RANDOM_STATE)


def score(y_true, y_pred):
    """R2 and MAE of a subset. R2 is undefined for fewer than two compounds, and MAE for none."""
    if len(y_true) == 0:
        return float("nan"), float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    return (r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")), mae


def rows_for(dataset, model, y_test, y_pred, cliff):
    """One row per scored subset: all test compounds, the cliff compounds, and the rest."""
    out = []
    for subset, mask in (("all", np.ones(len(y_test), dtype=bool)), ("cliff", cliff), ("noncliff", ~cliff)):
        r2, mae = score(y_test[mask], y_pred[mask])
        out.append(dict(dataset=dataset, model=model, subset=subset, n=int(mask.sum()), r2=r2, mae=mae))
    return out


def run_target(dataset, n_jobs):
    """Fit both models on the published training partition and score them on the published test partition."""
    df = pd.read_csv(os.path.join(MOLECULEACE_DIR, f"{dataset}.csv"))
    split = pd.read_csv(os.path.join(MOLECULEACE_DIR, f"{dataset}_split.csv"))
    # The two files are written in the same order by step 20; merging on the SMILES would collapse the
    # (already deduplicated) rows if that ever stopped holding, so the alignment is checked instead.
    if len(df) != len(split) or not (df["Smiles"].values == split["Smiles"].values).all():
        raise ValueError(f"{dataset}: data and split files are not aligned")

    y = df["pIC50 Value"].to_numpy(dtype=float)
    X = morgan_fingerprints(df["Smiles"].tolist())
    is_train = (split["split"] == "train").to_numpy()
    cliff_test = split.loc[~is_train, "cliff_mol"].to_numpy().astype(bool)
    X_train, y_train, X_test, y_test = X[is_train], y[is_train], X[~is_train], y[~is_train]

    base_models = base_regressors(n_estimators=CFG['N_ESTIMATORS_TREES'], seed=RANDOM_STATE, n_jobs=1)
    candidates = {combination_id(c): c for c in model_combinations(base_models, CFG['MAX_COMBO_SIZE'])}
    inner = inner_cv_scores(candidates, X_train, y_train, n_folds=CFG['INNER_N_SPLITS'],
                            n_repeats=INNER_N_REPEATS, seed=RANDOM_STATE, inner_score=INNER_SCORE,
                            make_estimator=make_estimator, clip_margin=CLIP_MARGIN, n_jobs=n_jobs)
    # The candidates are ordered from the simplest to the most complex, and idxmax keeps the first of a tie, so the
    # simplest combination wins one, exactly as in the nested selection of step 01.
    combo = candidates[inner.loc[inner["R2_inner_mean"].idxmax(), "Model"]]

    lower, upper = clip_bounds(y_train, margin=CLIP_MARGIN)
    estimator = make_estimator(combo)
    estimator.fit(X_train, y_train)
    pred_classical, _ = clip_predictions(estimator.predict(X_test), lower, upper)

    smiles = df["Smiles"].to_numpy()
    pred_gnn = gnn_utils.train_predict(smiles[is_train].tolist(), y_train, smiles[~is_train].tolist(),
                                       seed=RANDOM_STATE, cfg=gnn_utils.DEFAULT_CONFIG, threads=n_jobs)
    pred_gnn, _ = clip_predictions(np.asarray(pred_gnn, dtype=float).ravel(), lower, upper)

    rows = (rows_for(dataset, f"Classical ({combination_id(combo)})", y_test, pred_classical, cliff_test)
            + rows_for(dataset, "D-MPNN", y_test, pred_gnn, cliff_test))
    predictions = pd.DataFrame(dict(dataset=dataset, Smiles=smiles[~is_train], y_true=y_test,
                                    y_classical=pred_classical, y_gnn=pred_gnn, cliff_mol=cliff_test))
    return rows, predictions, combination_id(combo)


def run_target_task(task):
    """One target in its own process, for qsartools.parallel.run_tasks.

    The native libraries under chemprop occasionally die with an illegal instruction in the middle of a training, which
    takes the whole process down without raising. Run in a child process, such a death costs one target instead of the
    rest of the cohort, and the task is retried.

    Args:
        task: (dataset, n_jobs) as passed to run_target.
    """
    dataset, n_jobs = task
    started = time.time()
    rows, predictions, combo_id = run_target(dataset, n_jobs)
    return rows, predictions, combo_id, time.time() - started


def main():
    setup_logger(os.path.join(LOGS_DIR, "21_generalization_holdout.log"))
    print("=" * 105)
    print("21_generalization_holdout.py — published 80/20 split of the generalization cohort")
    print("=" * 105)
    explain("What this step does", """
        Fits a classical stacking ensemble on Morgan fingerprints and a default D-MPNN on the training partition that
        MoleculeACE publishes for each target, and scores both on its test partition. The base-learner combination is
        chosen by cross-validation inside the training partition only.

        Each model is scored three times: on the whole test partition, on the compounds its authors flagged as belonging
        to an activity cliff, and on the rest. The cliff subset is where a fingerprint representation is expected to be
        weakest, so reporting it separately shows whether the overall comparison hides a difference there.
    """)
    index = pd.read_csv(INDEX_CSV)
    n_jobs = CFG['N_JOBS']

    # Each target costs about half an hour, so the two files are rewritten after every one of them and a target already
    # present in them is skipped: the cohort can then be run in several sittings, or on more than one machine, without
    # repeating work. Delete the two files to force a clean run.
    rows, predictions, done = [], [], set()
    if os.path.exists(OUT_CSV):
        previous = pd.read_csv(OUT_CSV)
        rows = previous.to_dict("records")
        done = set(previous["dataset"])
        if os.path.exists(PRED_CSV):
            predictions = [pd.read_csv(PRED_CSV)]
        print(f"  [RESUME] {len(done)} target(s) already scored: {', '.join(sorted(done))}")

    tasks = []
    for _, entry in index.iterrows():
        if entry["dataset"] in done:
            print(f"  {entry['dataset']:<18} {entry['abbreviation']:<8} already scored, skipped", flush=True)
            continue
        tasks.append((entry["dataset"], n_jobs))
    described = {entry["dataset"]: entry for _, entry in index.iterrows()}

    def save(task, result):
        """Checkpoint one finished target, in the parent process, before the next one starts."""
        entry = described[task[0]]
        target_rows, target_predictions, combo_id, elapsed = result
        rows.extend(target_rows)
        predictions.append(target_predictions)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        pd.concat(predictions, ignore_index=True).to_csv(PRED_CSV, index=False)
        overall = {r["model"]: r["r2"] for r in target_rows if r["subset"] == "all"}
        classical_r2 = next(v for k, v in overall.items() if k.startswith("Classical"))
        print(f"  {entry['dataset']:<18} {entry['abbreviation']:<8} n={entry['n']:<5} {combo_id:<14} "
              f"classical R2={classical_r2:.4f}  D-MPNN R2={overall['D-MPNN']:.4f} "
              f"({elapsed:.0f}s)", flush=True)

    if tasks:
        # One target at a time: each one already uses every worker it is given
        os.environ.update(SAFE_CPU_ENV)
        run_tasks(run_target_task, tasks, n_workers=1, on_result=save)

    print(f"\n[EXPORT] {OUT_CSV}")
    print(f"[EXPORT] {PRED_CSV}")


if __name__ == "__main__":
    main()

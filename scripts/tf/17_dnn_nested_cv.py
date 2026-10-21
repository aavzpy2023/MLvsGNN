"""
Step 17 (TensorFlow environment): the network of Karimova & Senanayake (2025) on the 25 shared outer folds.

Two leak-free variants are scored on exactly the outer folds of the classical ensembles and the D-MPNN (exported by step 14,
matched by compound ID to the original PaDEL table):
  dnn_baseline  all descriptors, original preprocessing and network, early stopping on an internal validation part
  dnn_full      the complete original pipeline (importance groups G1-G3 and Gaussian augmentation), with the importance
                computed out-of-fold inside the outer training part, i.e. the R4 protocol of step 16
Everything is fitted on the outer training part only. Results are checkpointed per fold, so an interrupted run resumes.
Only CSV files are written; the statistical comparison is done by step 18 in the main environment.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import ROOT, RESULTS_DIR, LOGS_DIR, SPLIT_KIND, RANDOM_STATE
from logger_utils import setup_logger, explain

# The fold table is always read from the main results folder (step 14 exports every split kind there)
FOLDS_CSV = os.path.join(ROOT, "results", "14_outer_folds.csv")
CHECKPOINT_CSV = os.path.join(RESULTS_DIR, "17_dnn_nested_cv_checkpoint.csv")
OOF_CSV = os.path.join(RESULTS_DIR, "17_dnn_oof_predictions.csv")
SCORES_CSV = {m: os.path.join(RESULTS_DIR, f"17_{m}_fold_scores.csv") for m in ("dnn_baseline", "dnn_full")}

THREADS_PER_WORKER = int(os.environ.get("MLGNN_TF_THREADS", "4"))
CLIP_MARGIN = 3.0  # pIC50 units; identical to CLIP_MARGIN of step 01
MODELS = tuple(m for m in os.environ.get("MLGNN_K17_MODELS", "dnn_baseline,dnn_full").split(",") if m)

_TABLE = {}


def _table():
    if not _TABLE:
        import karimova_dnn as kd
        df, features = kd.load_table()
        _TABLE.update(df=df, features=features, pos={c: i for i, c in enumerate(df["cid"])})
    return _TABLE["df"], _TABLE["features"], _TABLE["pos"]


def run_task(task):
    import karimova_dnn as kd
    fold, model, train_cids, test_cids = task
    df, features, pos = _table()
    tr = np.array([pos[c] for c in train_cids])
    te = np.array([pos[c] for c in test_cids])
    y = df["pIC50 Value"].to_numpy()
    seed = RANDOM_STATE + fold
    t0 = time.time()

    if model == "dnn_full":
        imp, names = kd.oof_importance(df.iloc[tr][features], y[tr], seed)
        cols = kd.select_groups(imp, names, ("G1", "G2", "G3"))
    else:
        cols = features
    pre = kd.Preprocessor().fit(df.iloc[tr][cols])
    Xtr, Xte = pre.transform(df.iloc[tr][cols]), pre.transform(df.iloc[te][cols])
    pred, n_clipped = clip_to_training_range(kd.ensemble_predict_internal(Xtr, y[tr], Xte, seed,
                                                                          augment=(model == "dnn_full")), y[tr])
    r2, mae = kd.scores(y[te], pred)
    print(f"  fold {fold:>2} {model:<12} R2={r2:.4f} MAE={mae:.4f} features={Xtr.shape[1]} clipped={n_clipped} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return dict(Fold=fold, model=model, R2=r2, MAE=mae, N_train=len(tr), N_test=len(te), n_features=Xtr.shape[1],
                n_clipped=n_clipped, seconds=time.time() - t0), pd.DataFrame(dict(Fold=fold, model=model, cid=test_cids,
                                                                                  y_true=y[te], y_pred=pred))


def clip_to_training_range(pred, y_train, margin=CLIP_MARGIN):
    """Same safeguard as the classical models of step 01: predictions are clipped to the training range extended by
    `margin` pIC50 units. The original preprocessing (winsorizing and Yeo-Johnson fitted on the training part) extrapolates
    without bound for a few test compounds whose descriptors lie far outside the training range, and the network output
    then diverges. Returns (clipped prediction, number of clipped values)."""
    lo, hi = float(np.min(y_train)) - margin, float(np.max(y_train)) + margin
    return np.clip(pred, lo, hi), int(((pred < lo) | (pred > hi)).sum())


def _init_worker(threads):
    os.environ["MLGNN_TF_THREADS"] = str(threads)


def main():
    setup_logger(os.path.join(LOGS_DIR, "17_dnn_nested_cv.log"))
    print("=" * 105)
    print(f"17_dnn_nested_cv.py — Original deep network on the shared outer folds (split: {SPLIT_KIND}, TensorFlow environment)")
    print("=" * 105)
    explain("What this step does", """
        The original study scored its network once, on one random 15% test set. Here the same network, with the same
        descriptors, preprocessing and training settings, is scored on the 25 outer folds used for every other model of this
        study, so that its accuracy can be compared with theirs fold by fold. Two versions are run: the baseline network on
        all descriptors, and the complete original pipeline (descriptor selection by importance plus noisy copies of the
        training molecules). In both, nothing about a test fold is used before that fold is scored.
    """)
    import karimova_dnn as kd
    print(f"  Original repository: {kd.KARIMOVA_DIR} (commit {kd.check_clone()})")
    if not os.path.exists(FOLDS_CSV):
        print(f"[ERROR] Missing {FOLDS_CSV}. Run step 14 first.")
        sys.exit(1)
    folds = pd.read_csv(FOLDS_CSV).query("split_kind == @SPLIT_KIND")
    df, _, pos = _table()
    missing = set(folds["cid"]) - set(pos)
    if missing:
        print(f"[ERROR] {len(missing)} compounds of the folds are not in the original descriptor table.")
        sys.exit(1)

    done = pd.read_csv(CHECKPOINT_CSV) if os.path.exists(CHECKPOINT_CSV) else pd.DataFrame(columns=["Fold", "model"])
    done_keys = set(zip(done["Fold"], done["model"]))
    tasks = []
    # dnn_full first: it is the longest, so starting it early shortens the total time
    for model in sorted(MODELS, key=lambda m: m != "dnn_full"):
        for fold, g in folds.groupby("fold"):
            if (fold, model) not in done_keys:
                tasks.append((int(fold), model, g.query("role == 'train'")["cid"].to_numpy(),
                              g.query("role == 'test'")["cid"].to_numpy()))
    print(f"  {len(tasks)} fold x model tasks to run ({len(done_keys)} already in the checkpoint)")

    t0 = time.time()
    if tasks:
        n_workers = max(1, min(len(tasks), (os.cpu_count() or 1) // THREADS_PER_WORKER))
        print(f"  {n_workers} workers x {THREADS_PER_WORKER} threads")
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
            for row, pred in pool.imap_unordered(run_task, tasks):
                pd.DataFrame([row]).to_csv(CHECKPOINT_CSV, mode="a", header=not os.path.exists(CHECKPOINT_CSV), index=False)
                pred.to_csv(OOF_CSV, mode="a", header=not os.path.exists(OOF_CSV), index=False)

    scores = pd.read_csv(CHECKPOINT_CSV)
    for model, path in SCORES_CSV.items():
        s = scores[scores["model"] == model].sort_values("Fold")
        if len(s):
            s[["Fold", "R2", "MAE", "N_train", "N_test", "n_clipped"]].to_csv(path, index=False)
            print(f"  {model:<12} {len(s)} folds | R2 = {s['R2'].mean():.4f} ± {s['R2'].std(ddof=1):.4f} | "
                  f"MAE = {s['MAE'].mean():.4f} | clipped predictions: {int(s['n_clipped'].sum())} -> {path}")
    print(f"  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""
Step 16 (TensorFlow environment): leakage ladder for the final model of Karimova & Senanayake (2025).

Starting from the published protocol of the augmented network (R0, reported R2 = 0.85), one source of information leak is
removed at a time, on the same original split and with the same seeds:
  R0  published: importance and early stopping on the test set, nb8 re-split of preprocessed train+test, copies-only training
  R1  no re-split: augmentation of the original training rows only, scored on the original test set
  R2  early stopping and learning-rate schedule driven by an internal validation part of the training rows
  R3  importance computed out-of-fold inside the training rows instead of on the test set
  R4  the 14 duplicate structures removed (859 compounds, the curated set of this study)
R4 is the leak-free version of the final model. Leak-free versions of the earlier stages (baseline, importance-selected)
are fitted under the R4 protocol for reference. Only CSV files are written.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import RESULTS_DIR, LOGS_DIR, MAIN_FILE
from logger_utils import setup_logger, explain

LADDER_CSV = os.path.join(RESULTS_DIR, "16_leakage_ladder.csv")
PRED_CSV = os.path.join(RESULTS_DIR, "16_leakage_ladder_predictions.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "16_leakage_ladder_summary.csv")
R0_OVERLAP_CSV = os.path.join(RESULTS_DIR, "16_r0_test_overlap.csv")

N_SEEDS = int(os.environ.get("MLGNN_K16_SEEDS", "10"))
THREADS_PER_WORKER = int(os.environ.get("MLGNN_TF_THREADS", "4"))
GROUPS_FINAL = ("G1", "G2", "G3")
GROUPS_ALL = ("G1", "G2", "G3", "G4")


def run_seed(seed):
    import karimova_dnn as kd
    df, features = kd.load_table()
    y = df["pIC50 Value"].to_numpy()
    cids = df["cid"].to_numpy()
    tr, te = kd.split_indices(len(df))
    curated = set(pd.read_csv(MAIN_FILE)["cid"])
    # Common evaluation set: original test molecules that survive duplicate removal (scored by every rung from R1 on)
    common = np.isin(cids[te], list(curated))
    rows, preds = [], []

    def record(rung, eval_set, test_cids, y_true, y_pred, n_features):
        r2, mae = kd.scores(y_true, y_pred)
        rows.append(dict(seed=seed, rung=rung, eval_set=eval_set, n_test=len(y_true), n_features=n_features, r2=r2, mae=mae))
        preds.append(pd.DataFrame(dict(seed=seed, rung=rung, eval_set=eval_set, cid=test_cids, y_true=y_true, y_pred=y_pred)))

    def record_original_test(rung, y_pred, n_features):
        record(rung, "original_test", cids[te], y[te], y_pred, n_features)
        record(rung, "common_test", cids[te][common], y[te][common], y_pred[common], n_features)

    def prepared(rows_train, rows_eval, columns):
        pre = kd.Preprocessor().fit(df.iloc[rows_train][columns])
        return pre.transform(df.iloc[rows_train][columns]), pre.transform(df.iloc[rows_eval][columns])

    # Importance as in nb5: one network early-stopped on the test set, features permuted on the test set
    Xtr_all, Xte_all = prepared(tr, te, features)
    pi_model = kd.train_dnn(Xtr_all, y[tr], Xte_all, y[te], seed * 100 + 50)
    cols_test_pi = kd.select_groups(kd.permutation_importance_mean(pi_model, Xte_all, y[te]), Xte_all.columns, GROUPS_FINAL)

    # R0 — published protocol
    Xtr, Xte = prepared(tr, te, cols_test_pi)
    y_val, pred, val_pos = kd.nb8_protocol(Xtr, y[tr], Xte, y[te], seed)
    order = np.concatenate([tr, te])
    record("R0_published", "own_resplit_test", cids[order[val_pos]], y_val, pred, Xtr.shape[1])

    # R1 — augmentation of the original training rows only; callbacks still monitor the test set
    X_aug, y_aug, _ = kd.augment(Xtr, y[tr], np.random.default_rng(seed))
    pred = kd.ensemble_predict(X_aug, y_aug, Xte, y[te], Xte, seed, kd.DROPOUT_AUG, kd.PATIENCE_AUG)
    record_original_test("R1_no_resplit", pred, Xtr.shape[1])

    # R2 — internal validation for every network, including the one used for the importance (still permuted on test)
    pi_model = kd.train_internal(Xtr_all, y[tr], seed * 100 + 50)
    cols = kd.select_groups(kd.permutation_importance_mean(pi_model, Xte_all, y[te]), Xte_all.columns, GROUPS_FINAL)
    Xtr, Xte = prepared(tr, te, cols)
    record_original_test("R2_internal_early_stopping",
                         kd.ensemble_predict_internal(Xtr, y[tr], Xte, seed, augment=True), Xtr.shape[1])

    # R3 — importance out-of-fold inside the training rows
    imp, names = kd.oof_importance(df.iloc[tr][features], y[tr], seed)
    cols = kd.select_groups(imp, names, GROUPS_FINAL)
    Xtr, Xte = prepared(tr, te, cols)
    record_original_test("R3_oof_importance", kd.ensemble_predict_internal(Xtr, y[tr], Xte, seed, augment=True), Xtr.shape[1])

    # R4 — duplicate structures removed from both sides of the split
    tr4 = tr[np.isin(cids[tr], list(curated))]
    te4 = te[common]
    imp, names = kd.oof_importance(df.iloc[tr4][features], y[tr4], seed)
    for rung, groups, augment in (("R4_no_duplicates", GROUPS_FINAL, True),
                                  ("clean_sel_G1234", GROUPS_ALL, False),
                                  ("clean_sel_G123", GROUPS_FINAL, False)):
        cols = kd.select_groups(imp, names, groups)
        Xtr, Xte = prepared(tr4, te4, cols)
        record(rung, "common_test", cids[te4], y[te4],
               kd.ensemble_predict_internal(Xtr, y[tr4], Xte, seed, augment=augment), Xtr.shape[1])
    Xtr, Xte = prepared(tr4, te4, features)
    record("clean_baseline", "common_test", cids[te4], y[te4],
           kd.ensemble_predict_internal(Xtr, y[tr4], Xte, seed), Xtr.shape[1])

    print(f"  seed {seed} done: " + " | ".join(f"{r['rung']}/{r['eval_set'][:6]} R2={r['r2']:.3f}" for r in rows), flush=True)
    return rows, pd.concat(preds, ignore_index=True)


def r0_overlap():
    """Composition of the set on which R0 is scored: rows that come from the original test set (on which the importance was
    computed) and rows from the original training set (on which the preprocessing was fitted). Deterministic."""
    import karimova_dnn as kd
    df, _ = kd.load_table()
    tr, te = kd.split_indices(len(df))
    _, val_pos = kd.split_indices(len(tr) + len(te))
    from_test = int((val_pos >= len(tr)).sum())
    return pd.DataFrame([dict(n_scored=len(val_pos), from_original_test=from_test,
                              from_original_train=len(val_pos) - from_test,
                              frac_preprocessing_fitted_on=(len(val_pos) - from_test) / len(val_pos))])


def _init_worker(threads):
    os.environ["MLGNN_TF_THREADS"] = str(threads)


def main():
    setup_logger(os.path.join(LOGS_DIR, "16_leakage_ladder.log"))
    print("=" * 105)
    print("16_leakage_ladder.py — How much of the published R2 = 0.85 comes from information leaks (TensorFlow environment)")
    print("=" * 105)
    explain("What this step does", """
        The final model of the original study reached R2 = 0.85 on its test molecules. In its code, those test molecules also
        decided which descriptors were kept and when training stopped, and part of them had been used to fit the data
        preprocessing. Each of these lets the model see information about the molecules it is scored on, which makes the
        score optimistic. This step removes the leaks one at a time (R0 = as published ... R4 = no leak left, duplicates
        removed) and re-scores the model after each change, with the same split and the same random seeds. The drop between two
        rungs is the part of the score that the removed leak was responsible for.
    """)
    import karimova_dnn as kd
    print(f"  Original repository: {kd.KARIMOVA_DIR} (commit {kd.check_clone()})")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    overlap = r0_overlap()
    overlap.to_csv(R0_OVERLAP_CSV, index=False)
    o = overlap.iloc[0]
    print(f"  R0 scoring set: {o.n_scored} molecules, {o.from_original_test} from the original test set and "
          f"{o.from_original_train} from the original training set (preprocessing fitted on them)")

    t0 = time.time()
    n_workers = max(1, min(N_SEEDS, (os.cpu_count() or 1) // THREADS_PER_WORKER))
    print(f"  {N_SEEDS} seeds on {n_workers} workers x {THREADS_PER_WORKER} threads")
    with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
        results = pool.map(run_seed, range(1, N_SEEDS + 1))
    ladder = pd.DataFrame([row for rows, _ in results for row in rows])
    ladder.to_csv(LADDER_CSV, index=False)
    pd.concat([p for _, p in results], ignore_index=True).to_csv(PRED_CSV, index=False)

    summary = (ladder.groupby(["rung", "eval_set"], sort=False)
               .agg(n_seeds=("r2", "size"), n_test=("n_test", "first"), n_features=("n_features", "median"),
                    r2_mean=("r2", "mean"), r2_sd=("r2", "std"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"))
               .reset_index())
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"  {time.time() - t0:.0f}s -> {LADDER_CSV}, {PRED_CSV}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

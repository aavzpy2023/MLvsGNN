import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import *
from logger_utils import setup_logger, explain
from qsartools.latex import newcommand
from nb_stats import compare_to_classical, EQUIVALENCE_MARGIN_R2, EQUIVALENCE_MARGIN_MAE
from qsartools.stats import corrected_ttest

LATEX_AUDIT = os.path.join(LATEX_DIR, "karimova_audit_variables.tex")
LADDER_DELTAS_CSV = os.path.join(RESULTS_DIR, "18_leakage_ladder_deltas.csv")
DNN_TESTS_CSV = os.path.join(RESULTS_DIR, "18_dnn_vs_other_models_tests.csv")
INPUTS = {name: os.path.join(RESULTS_DIR, f) for name, f in (
    ("verbatim", "15_karimova_verbatim.csv"), ("repro", "15_karimova_reproduction.csv"),
    ("ladder", "16_leakage_ladder.csv"), ("ladder_pred", "16_leakage_ladder_predictions.csv"),
    ("r0_overlap", "16_r0_test_overlap.csv"),
    ("dnn_baseline", "17_dnn_baseline_fold_scores.csv"), ("dnn_full", "17_dnn_full_fold_scores.csv"))}

STAGE_LABEL = {"baseline": "Baseline", "sel_G1234": "SelAll", "sel_G123": "SelFinal", "aug_G123": "Aug"}
SOURCE_LABEL = {"-": "", "published": "Pub", "fresh": "Fresh"}
RUNG_LABEL = {"R0_published": "RungZero", "R1_no_resplit": "RungOne", "R2_internal_early_stopping": "RungTwo",
              "R3_oof_importance": "RungThree", "R4_no_duplicates": "RungFour", "clean_baseline": "CleanBaseline",
              "clean_sel_G1234": "CleanSelAll", "clean_sel_G123": "CleanSelFinal"}
LADDER_STEPS = [("R1_no_resplit", "R0_published"), ("R2_internal_early_stopping", "R1_no_resplit"),
                ("R3_oof_importance", "R2_internal_early_stopping"), ("R4_no_duplicates", "R3_oof_importance"),
                ("R4_no_duplicates", "R0_published"), ("R4_no_duplicates", "clean_baseline")]
DNN_LABEL = {"dnn_baseline": "DnnBase", "dnn_full": "DnnFull"}
N_BOOT = 2000


def seed_mean_predictions(pred, rung):
    """Seed-averaged prediction of a rung on the common test set, one row per molecule (cid)."""
    p = pred.query("rung == @rung and eval_set == 'common_test'").groupby("cid").agg(y_true=("y_true", "first"),
                                                                                   y_pred=("y_pred", "mean"))
    return p.sort_index()


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ladder_deltas(ladder, pred):
    """Per-seed paired differences between rungs, and a paired bootstrap over test molecules of the seed-averaged predictions.
    Rungs scored on the same molecules are compared on the common test set; R0 is scored on its own re-split set, so any
    difference involving R0 is a difference of means between two different test sets, not a paired difference."""
    rows = []
    rng = np.random.RandomState(RANDOM_STATE)
    for new, old in LADDER_STEPS:
        paired = old != "R0_published"
        old_set = "common_test" if paired else "own_resplit_test"
        s_new = ladder.query("rung == @new and eval_set == 'common_test'").set_index("seed")
        s_old = ladder.query("rung == @old and eval_set == @old_set").set_index("seed")
        d = (s_new["r2"] - s_old["r2"]).dropna()
        row = dict(new=new, old=old, paired=paired, n_seeds=len(d), delta_r2_mean=d.mean(), delta_r2_sd=d.std(ddof=1),
                   delta_mae_mean=(s_new["mae"] - s_old["mae"]).mean(), frac_seeds_negative=float((d < 0).mean()),
                   boot_ci_low=np.nan, boot_ci_high=np.nan)
        if paired:
            a, b = seed_mean_predictions(pred, new), seed_mean_predictions(pred, old)
            a, b = a.loc[a.index.intersection(b.index)], b.loc[a.index.intersection(b.index)]
            y, pa, pb = a["y_true"].to_numpy(), a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
            boots = []
            for _ in range(N_BOOT):
                i = rng.randint(0, len(y), len(y))
                boots.append(r2(y[i], pa[i]) - r2(y[i], pb[i]))
            row["boot_ci_low"], row["boot_ci_high"] = np.percentile(boots, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def dnn_tests(dnn):
    """The two leak-free DNN variants against each other and against the default and tuned D-MPNN on the shared folds."""
    others = {"GnnDefault": pd.read_csv(GNN_FOLD_SCORES_CSV),
              "GnnTuned": pd.read_csv(GNN_TUNED_CHECKPOINT)[["Fold", "R2", "MAE", "N_train", "N_test"]]}
    others["DnnBase"] = dnn["dnn_baseline"]
    rows = []
    for model in ("dnn_full", "dnn_baseline"):
        a = dnn[model].sort_values("Fold").reset_index(drop=True)
        for label, b in others.items():
            if label == DNN_LABEL[model]:
                continue
            b = b.sort_values("Fold").reset_index(drop=True)
            if len(a) != len(b) or not (a["N_test"].values == b["N_test"].values).all():
                print(f"  [WARNING] Folds of {model} and {label} do not match; comparison skipped.")
                continue
            for metric, col, margin in (("RTwo", "R2", EQUIVALENCE_MARGIN_R2), ("Mae", "MAE", EQUIVALENCE_MARGIN_MAE)):
                res = corrected_ttest(a[col].values - b[col].values, a["N_train"].mean(), a["N_test"].mean(), margin=margin)
                rows.append(dict(model=DNN_LABEL[model], other=label, Metric=metric, mean_diff=res["mean_diff"],
                                 p_raw=res["p"], ci90_low=res["ci90_low"], ci90_high=res["ci90_high"], p_equiv=res["p_equiv"]))
    return pd.DataFrame(rows)


RUNG_DESCRIPTION = [
    ("R0_published", "None: the protocol as published"),
    ("R1_no_resplit", "Synthetic copies of test molecules reaching the training set"),
    ("R2_internal_early_stopping", "Early stopping and model choice guided by the test set"),
    ("R3_oof_importance", "Feature importance computed on the test set"),
    ("R4_no_duplicates", "The same structure on both sides of the split"),
    ("clean_baseline", "Reference: no feature selection at all"),
]


def write_ladder_table(rung_summary, deltas):
    """One row per rung of the leakage ladder, with the drop each removed leak accounts for. R0 is scored on the test
    partition its own protocol produces; every later rung on the molecules common to all of them, so the R0 difference
    is not paired and carries no bootstrap interval."""
    by_pair = {(d["new"], d["old"]): d for d in deltas.to_dict("records")}
    lines = ["% Auto-generated by 18_karimova_audit.py",
             "\\begin{tabular}{llrcc}", "\\toprule",
             "\\textbf{Rung} & \\textbf{Leak removed} & \\textbf{Features} & $R^2$ & $\\Delta R^2$ \\\\",
             "\\midrule"]
    for i, (rung, description) in enumerate(RUNG_DESCRIPTION):
        key = ("common_test" if rung != "R0_published" else "own_resplit_test")
        r = rung_summary.loc[(rung, key)]
        delta = ""
        d = by_pair.get((rung, RUNG_DESCRIPTION[i - 1][0]) if i else None)
        if d is not None:
            delta = f"${d['delta_r2_mean']:+.3f}$"
            if d["paired"]:
                delta += f" [${d['boot_ci_low']:+.3f}$, ${d['boot_ci_high']:+.3f}$]"
        label = rung.split("_")[0].replace("clean", "Clean")
        lines.append(f"{label} & {description} & {int(r['n_feat'])} & "
                     f"${r['m']:.3f} \\pm {r['s']:.3f}$ & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(LATEX_LADDER_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[EXPORT] Leakage ladder table saved to {LATEX_LADDER_TABLE}", flush=True)


def main():
    setup_logger(os.path.join(LOGS_DIR, "18_karimova_audit.log"))
    print("=" * 105)
    print("18_karimova_audit.py — Audit of the original deep network: reproduction, leakage ladder and shared-fold comparison")
    print("=" * 105)
    explain("What this step does", """
        Collects the results of steps 15-17, which ran in the TensorFlow environment, and turns them into the numbers used by
        the manuscript: (1) how well the original notebooks and their re-implementation reproduce the published R2 values,
        (2) how much R2 is lost each time one information leak is removed from the final model, with paired differences
        over seeds and a bootstrap over test molecules, and (3) how the leak-free original network compares, fold by fold,
        with the classical ensembles and the D-MPNN under the same corrected tests used everywhere else in the study.
    """)
    missing = [p for p in INPUTS.values() if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing inputs (run steps 15-17 first):\n  " + "\n  ".join(missing))
        sys.exit(1)
    data = {k: pd.read_csv(p) for k, p in INPUTS.items()}
    values = {}

    for _, r in data["verbatim"].iterrows():
        key = f"KarVerb{STAGE_LABEL[r['stage']]}"
        values[f"{key}PrintedRTwo"] = f"{r['r2_printed']:.3f}"
        values[f"{key}RerunRTwo"] = f"{r['r2_rerun']:.3f}"

    repro = data["repro"].groupby(["stage", "pi_source"]).agg(m=("r2", "mean"), s=("r2", "std"), lo=("r2", "min"),
                                                              hi=("r2", "max"), n=("r2", "size"))
    print("\nReproduction over seeds (step 15b):\n" + repro.round(4).to_string())
    for (stage, source), r in repro.iterrows():
        key = f"KarRepro{STAGE_LABEL[stage]}{SOURCE_LABEL[source]}"
        values[f"{key}RTwoMean"], values[f"{key}RTwoSd"] = f"{r['m']:.3f}", f"{r['s']:.3f}"
        values[f"{key}RTwoMin"], values[f"{key}RTwoMax"] = f"{r['lo']:.3f}", f"{r['hi']:.3f}"
    values["KarReproNSeeds"] = str(int(repro["n"].max()))

    ladder = data["ladder"]
    rung_summary = ladder.groupby(["rung", "eval_set"]).agg(m=("r2", "mean"), s=("r2", "std"), mae=("mae", "mean"),
                                                            n_test=("n_test", "first"), n_feat=("n_features", "median"))
    print("\nLeakage ladder (step 16):\n" + rung_summary.round(4).to_string())
    for (rung, eval_set), r in rung_summary.iterrows():
        if eval_set == "original_test":
            continue
        key = f"KarLadder{RUNG_LABEL[rung]}"
        values[f"{key}RTwoMean"], values[f"{key}RTwoSd"] = f"{r['m']:.3f}", f"{r['s']:.3f}"
        values[f"{key}Mae"], values[f"{key}NFeatures"] = f"{r['mae']:.3f}", str(int(r["n_feat"]))
    values["KarLadderNCommonTest"] = str(int(rung_summary.xs("common_test", level="eval_set")["n_test"].iloc[0]))
    o = data["r0_overlap"].iloc[0]
    values["KarRZeroNScored"] = str(int(o["n_scored"]))
    values["KarRZeroFromOriginalTest"] = str(int(o["from_original_test"]))
    values["KarRZeroFromOriginalTrain"] = str(int(o["from_original_train"]))

    deltas = ladder_deltas(ladder, data["ladder_pred"])
    deltas.to_csv(LADDER_DELTAS_CSV, index=False)
    write_ladder_table(rung_summary, deltas)
    print("\nLadder differences:\n" + deltas.round(4).to_string(index=False))
    for _, d in deltas.iterrows():
        key = f"KarDelta{RUNG_LABEL[d['new']]}Vs{RUNG_LABEL[d['old']]}"
        values[f"{key}RTwo"], values[f"{key}RTwoSd"] = f"{d['delta_r2_mean']:+.3f}", f"{d['delta_r2_sd']:.3f}"
        if d["paired"]:
            values[f"{key}CILow"], values[f"{key}CIHigh"] = f"{d['boot_ci_low']:+.3f}", f"{d['boot_ci_high']:+.3f}"

    dnn = {m: data[m] for m in ("dnn_baseline", "dnn_full")}
    checkpoint = pd.read_csv(CHECKPOINT_FILE)
    tables = []
    for model, s in dnn.items():
        label = DNN_LABEL[model]
        values[f"{label}RTwoMean"], values[f"{label}RTwoStd"] = f"{s['R2'].mean():.4f}", f"{s['R2'].std(ddof=1):.4f}"
        values[f"{label}Mae"] = f"{s['MAE'].mean():.4f}"
        values[f"{label}NClipped"] = str(int(s["n_clipped"].sum()))
        table, macros = compare_to_classical(s.sort_values("Fold").reset_index(drop=True), checkpoint, f"{label}Vs")
        values.update(macros)
        tables.append(table.assign(model=label))
    others = dnn_tests(dnn)
    for _, r in others.iterrows():
        key = f"{r['model']}Vs{r['other']}"
        values[f"{key}Diff{r['Metric']}"] = f"{r['mean_diff']:+.4f}"
        values[f"{key}P{r['Metric']}"] = f"{r['p_raw']:.3f}"
        values[f"{key}EquivP{r['Metric']}"] = f"{r['p_equiv']:.3f}"
    pd.concat(tables + [others.rename(columns={"other": "Classical", "mean_diff": "mean_diff_model_minus_classical"})],
              ignore_index=True).to_csv(DNN_TESTS_CSV, index=False)
    print("\nLeak-free original network on the shared folds:")
    for model, s in dnn.items():
        print(f"  {model:<12} R2 = {s['R2'].mean():.4f} ± {s['R2'].std(ddof=1):.4f} | MAE = {s['MAE'].mean():.4f}")
    print(others.round(4).to_string(index=False))

    os.makedirs(os.path.dirname(LATEX_AUDIT), exist_ok=True)
    with open(LATEX_AUDIT, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 18_karimova_audit.py\n")
        for name, value in values.items():
            f.write(newcommand(name, value))
    print(f"\n[EXPORT] {len(values)} LaTeX variables saved to {LATEX_AUDIT}")


if __name__ == "__main__":
    main()

"""
22_generalization_summary.py — The twelve MoleculeACE targets as one result: cross-validated cohort, published split,
and the tuned D-MPNN on the targets where the default one loses most.

Reads only what other runs wrote, in seconds:
  generalization/<target>/results/   steps 01, 09 (and 12 where it was run) under the 'generalization' profile
  generalization_holdout/results/    step 21, the published 80/20 partition with its activity-cliff labels
Writes latex/generalization_variables.tex, latex/generalization_table.tex and results/22_generalization_summary.csv.

Every difference is exported as the advantage of the classical model (classical minus D-MPNN), so that a positive
number always reads the same way; the per-target tests of steps 09 and 12 store the opposite sign.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import *
from logger_utils import setup_logger, explain
from qsartools.latex import newcommand
from qsartools import latex

GEN_DIR = os.environ.get("MLGNN_GEN_DIR") or os.path.join(ROOT, "generalization")
HOLDOUT_CSV = os.path.join(os.environ.get("MLGNN_GEN_HOLDOUT_DIR") or os.path.join(ROOT, "generalization_holdout"),
                           "results", "21_generalization_holdout.csv")
INDEX_CSV = os.path.join(DATA_DIR, "moleculeace", "index.csv")
LATEX_GEN = os.path.join(LATEX_DIR, "generalization_variables.tex")
LATEX_GEN_TABLE = os.path.join(LATEX_DIR, "generalization_table.tex")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "22_generalization_summary.csv")

ALPHA = 0.05
GEN_TABLE_COLS = "lrrcccrcc"
DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
               "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_token(abbreviation):
    """A target abbreviation as letters only, for a macro name: 'PIK3CA' -> 'PikThreeCa', 'AR' -> 'Ar'.

    Digits are spelled out because a LaTeX macro name cannot contain them, and each run of letters is capitalized
    once, so that two abbreviations that differ only in a digit ('JAK1', 'JAK2') stay distinct.
    """
    parts, run = [], ""
    for ch in abbreviation:
        if ch.isdigit():
            if run:
                parts.append(run.capitalize())
                run = ""
            parts.append(DIGIT_WORDS[ch])
        elif ch.isalpha():
            run += ch
    if run:
        parts.append(run.capitalize())
    if not parts:
        raise ValueError(f"Abbreviation {abbreviation!r} has no letters or digits")
    return "".join(parts)


def cohort_test(advantage):
    """Across-target summary of per-target classical advantages: the targets are the units, not the folds.

    Returns the count and mean, a 95% t interval, the one-sample t-test and the exact Wilcoxon signed-rank test
    against zero (both two-sided), and how many targets favour each family.
    """
    a = np.asarray(advantage, dtype=float)
    n = len(a)
    if n < 2:
        raise ValueError("An across-target test needs at least two targets")
    mean, sem = a.mean(), a.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * sem
    return dict(n=n, mean=mean, ci_low=mean - half, ci_high=mean + half,
                p_t=stats.ttest_1samp(a, 0.0).pvalue, p_wilcoxon=stats.wilcoxon(a).pvalue,
                n_classical=int((a > 0).sum()), n_neural=int((a < 0).sum()))


def fmt_p(value):
    """A p below the fourth decimal prints as a bound, not as 0.0000."""
    return "< 0.0001" if value < 1e-4 else f"{value:.4f}"


def target_cv_row(dataset, results_dir):
    """Steps 01 and 09 (and 12 when present) of one target, as one row; Morgan is the classical reference."""
    final = pd.read_csv(os.path.join(results_dir, "01_nested_cv_final_results.csv")).set_index("Mode")
    gnn = pd.read_csv(os.path.join(results_dir, "09_gnn_fold_scores.csv"))
    tests = pd.read_csv(os.path.join(results_dir, "09_gnn_vs_classical_tests.csv"))
    tests = tests[tests["Classical"] == "morgan"].set_index("Metric")
    row = dict(dataset=dataset, morgan_r2=final.loc["morgan", "mean"], rdkit_r2=final.loc["rdkit2d", "mean"],
               gnn_r2=gnn["R2"].mean(),
               adv_r2=-tests.loc["RTwo", "mean_diff_model_minus_classical"],
               p_holm_r2=tests.loc["RTwo", "p_holm"], p_equiv_r2=tests.loc["RTwo", "p_equiv"],
               # A larger MAE is worse, so the model-minus-classical MAE difference already is the classical advantage
               adv_mae=tests.loc["Mae", "mean_diff_model_minus_classical"],
               p_holm_mae=tests.loc["Mae", "p_holm"], p_equiv_mae=tests.loc["Mae", "p_equiv"])
    tuned_path = os.path.join(results_dir, "12_gnn_tuned_vs_classical_tests.csv")
    if os.path.exists(tuned_path):
        tuned = pd.read_csv(tuned_path)
        tuned = tuned[tuned["Label"] == "Morgan"].set_index("Metric")
        scores = pd.read_csv(os.path.join(results_dir, "12_gnn_tuned_checkpoint.csv"))
        row.update(tuned_r2=scores["R2"].mean(),
                   tuned_adv_r2=-tuned.loc["RTwo", "mean_diff_model_minus_classical"],
                   tuned_p_holm_r2=tuned.loc["RTwo", "p_holm"], tuned_p_equiv_r2=tuned.loc["RTwo", "p_equiv"])
    return row


def holdout_advantages(holdout):
    """Classical minus D-MPNN R2 on the published test partition, one column per subset (all, cliff, noncliff)."""
    family = np.where(holdout["model"].str.startswith("Classical"), "classical", "gnn")
    wide = holdout.assign(family=family).pivot_table(index=["dataset", "subset"], columns="family", values="r2")
    adv = (wide["classical"] - wide["gnn"]).unstack("subset")
    sizes = holdout[family == "classical"].pivot(index="dataset", columns="subset", values="n")
    return adv, sizes


def write_table(summary):
    """One row per target: the cross-validated comparison, then the published split overall and on its cliffs."""
    lines = ["% Auto-generated by 22_generalization_summary.py",
             latex.tabular_open(GEN_TABLE_COLS), "\\toprule",
             "& & & \\multicolumn{4}{c}{Cross-validation ($R^2$)} & \\multicolumn{2}{c}{Published split} \\\\",
             "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
             "\\textbf{Target} & \\textbf{$n$} & \\textbf{Cliffs (\\%)} & Morgan & D-MPNN & Advantage & "
             "$p_{\\mathrm{Holm}}$ & Advantage & Cliffs \\\\", "\\midrule"]
    for _, r in summary.iterrows():
        lines.append(f"{r['abbreviation']} ({r['endpoint']}) & {int(r['n'])} & {r['cliff_pct']:.0f} & "
                     f"{r['morgan_r2']:.3f} & {r['gnn_r2']:.3f} & ${r['adv_r2']:+.3f}$ & {r['p_holm_r2']:.3f} & "
                     f"${r['holdout_adv_all']:+.3f}$ & ${r['holdout_adv_cliff']:+.3f}$ \\\\")
    lines += ["\\bottomrule", latex.tabular_close(GEN_TABLE_COLS)]
    with open(LATEX_GEN_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[EXPORT] Generalization table saved to {LATEX_GEN_TABLE}", flush=True)


def cohort_macros(prefix, result, digits=3):
    return {f"{prefix}N": result["n"], f"{prefix}NClassical": result["n_classical"],
            f"{prefix}NNeural": result["n_neural"],
            f"{prefix}Mean": f"{result['mean']:+.{digits}f}",
            f"{prefix}CILow": f"{result['ci_low']:+.{digits}f}", f"{prefix}CIHigh": f"{result['ci_high']:+.{digits}f}",
            f"{prefix}PT": fmt_p(result["p_t"]), f"{prefix}PWilcoxon": fmt_p(result["p_wilcoxon"])}


def main():
    setup_logger(os.path.join(LOGS_DIR, "22_generalization_summary.log"))
    print("=" * 105)
    print("22_generalization_summary.py — Classical ensemble against the default D-MPNN on twelve external targets")
    print("=" * 105)
    explain("What this step does", """
        The MoleculeACE targets were run one at a time under the 'generalization' profile (steps 01 and 09, 5x1 folds),
        and once more on the published 80/20 partition (step 21). Each target gives a paired comparison of its own; this
        step treats the targets as the units of a second-level test: does the classical advantage hold across targets,
        on the published split, and on its activity cliffs? Where step 12 (the D-MPNN with a nested hyperparameter
        search) was run, it also reports whether tuning closes the gap.
    """)

    index = pd.read_csv(INDEX_CSV)
    missing = [d for d in index["dataset"]
               if not os.path.exists(os.path.join(GEN_DIR, d, "results", "09_gnn_vs_classical_tests.csv"))]
    if missing or not os.path.exists(HOLDOUT_CSV):
        print(f"[ERROR] Missing inputs: targets {missing}, holdout {os.path.exists(HOLDOUT_CSV)}. "
              "Run logs/run_generalization.sh and step 21 first.")
        sys.exit(1)

    cv = pd.DataFrame([target_cv_row(d, os.path.join(GEN_DIR, d, "results")) for d in index["dataset"]])
    adv, sizes = holdout_advantages(pd.read_csv(HOLDOUT_CSV))
    summary = (index.merge(cv, on="dataset")
               .merge(adv.add_prefix("holdout_adv_"), left_on="dataset", right_index=True)
               .merge(sizes.add_prefix("holdout_n_"), left_on="dataset", right_index=True))
    summary.to_csv(SUMMARY_CSV, index=False)
    write_table(summary)
    print(summary[["abbreviation", "n", "morgan_r2", "gnn_r2", "adv_r2", "p_holm_r2", "p_equiv_r2",
                   "holdout_adv_all", "holdout_adv_cliff"]].round(3).to_string(index=False))

    values = {"GenNTargets": len(summary), "GenNMin": int(summary["n"].min()), "GenNMax": int(summary["n"].max()),
              "GenCliffPctMin": f"{summary['cliff_pct'].min():.0f}",
              "GenCliffPctMax": f"{summary['cliff_pct'].max():.0f}",
              "GenNKi": int((summary["endpoint"] == "Ki").sum()), "GenNEC": int((summary["endpoint"] == "EC50").sum())}

    cv_test = cohort_test(summary["adv_r2"])
    values.update(cohort_macros("GenCvAdv", cv_test))
    values.update(cohort_macros("GenCvMaeAdv", cohort_test(summary["adv_mae"])))
    values.update({"GenCvNHolm": int((summary["p_holm_r2"] < ALPHA).sum()),
                   "GenCvNHolmMae": int((summary["p_holm_mae"] < ALPHA).sum()),
                   "GenCvNEquiv": int((summary["p_equiv_r2"] < ALPHA).sum()),
                   "GenCvNEquivMae": int((summary["p_equiv_mae"] < ALPHA).sum()),
                   "GenCvPEquivMin": f"{summary['p_equiv_r2'].min():.3f}",
                   "GenCvAdvMin": f"{summary['adv_r2'].min():+.3f}", "GenCvAdvMax": f"{summary['adv_r2'].max():+.3f}",
                   "GenNMorganOverRdkit": int((summary["morgan_r2"] > summary["rdkit_r2"]).sum())})

    for subset, prefix in (("all", "GenHoldoutAdv"), ("cliff", "GenCliffAdv"), ("noncliff", "GenNonCliffAdv")):
        values.update(cohort_macros(prefix, cohort_test(summary[f"holdout_adv_{subset}"])))
    values.update({"GenHoldoutNTestMin": int(summary["holdout_n_all"].min()),
                   "GenHoldoutNTestMax": int(summary["holdout_n_all"].max()),
                   "GenCliffNTestMin": int(summary["holdout_n_cliff"].min()),
                   "GenCliffNTestMax": int(summary["holdout_n_cliff"].max())})

    for _, r in summary.iterrows():
        key = f"Gen{macro_token(r['abbreviation'])}"
        values.update({f"{key}N": int(r["n"]), f"{key}MorganRTwo": f"{r['morgan_r2']:.3f}",
                       f"{key}GnnRTwo": f"{r['gnn_r2']:.3f}", f"{key}AdvRTwo": f"{r['adv_r2']:+.3f}",
                       f"{key}PHolmRTwo": f"{r['p_holm_r2']:.3f}", f"{key}PEquivRTwo": f"{r['p_equiv_r2']:.3f}",
                       f"{key}HoldoutAdv": f"{r['holdout_adv_all']:+.3f}",
                       f"{key}CliffAdv": f"{r['holdout_adv_cliff']:+.3f}"})
        if pd.notna(r.get("tuned_adv_r2", np.nan)):
            values.update({f"{key}TunedRTwo": f"{r['tuned_r2']:.3f}", f"{key}TunedAdvRTwo": f"{r['tuned_adv_r2']:+.3f}",
                           f"{key}TunedPHolmRTwo": f"{r['tuned_p_holm_r2']:.3f}",
                           f"{key}TunedPEquivRTwo": f"{r['tuned_p_equiv_r2']:.3f}",
                           f"{key}TuningGain": f"{r['adv_r2'] - r['tuned_adv_r2']:.3f}"})
    tuned = summary.dropna(subset=["tuned_adv_r2"]) if "tuned_adv_r2" in summary else summary.iloc[:0]
    values["GenNTuned"] = len(tuned)
    if len(tuned):
        values.update({"GenTunedNHolm": int((tuned["tuned_p_holm_r2"] < ALPHA).sum()),
                       "GenTunedNEquiv": int((tuned["tuned_p_equiv_r2"] < ALPHA).sum())})

    print(f"\nCross-validation: classical ahead in {cv_test['n_classical']}/{cv_test['n']}, "
          f"mean {cv_test['mean']:+.3f} [{cv_test['ci_low']:+.3f}, {cv_test['ci_high']:+.3f}], "
          f"t p={cv_test['p_t']:.2g}, Wilcoxon p={cv_test['p_wilcoxon']:.2g}")
    os.makedirs(os.path.dirname(LATEX_GEN), exist_ok=True)
    with open(LATEX_GEN, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 22_generalization_summary.py\n")
        for name, value in values.items():
            f.write(newcommand(name, value))
    print(f"[EXPORT] {len(values)} LaTeX variables saved to {LATEX_GEN}")
    print("[DONE] 22_generalization_summary.py complete.")


if __name__ == "__main__":
    main()

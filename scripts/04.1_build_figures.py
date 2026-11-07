import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import RESULTS_DIR, FIGURES_DIR, CSV_04_OUTPUT_FILE, CSV_04_DELTA_FILE, LATEX_ABLATION_TABLE
from logger_utils import explain

CSV_FILE = os.path.join(RESULTS_DIR, CSV_04_OUTPUT_FILE)
FIG_BASE = os.path.join(FIGURES_DIR, "04_augment_r2_comparison")

# Reference values reported in the original paper (DNN pipeline)
PAPER_R2 = {
    'Paper baseline': 0.75,
    'Paper + feat. sel.': 0.82,
    'Paper final (DNN)': 0.85,
}
REF_STYLE = [(':', '#7F7F7F'), ('--', '#7F7F7F'), ('-', '#333333')]

# Colour-blind-safe palette (Okabe-Ito), one colour per experiment
COLORS = ['#0072B2', '#009E73', '#E69F00']

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


def generate_figure():
    if not os.path.exists(CSV_FILE):
        print(f"Error: file not found: {CSV_FILE}. Run 04_augmentation_training.py first.")
        raise SystemExit(1)
    explain("What this figure shows", """
        R2 on the held-out test molecules for the experiments of step 04 (A baseline, B + feature selection, C +
        augmentation), each with its 95% bootstrap confidence interval, next to the values reported in the original paper
        (dashed lines). Intervals that overlap are not statistically distinguishable: with a small test set, apparent
        gains from feature selection or augmentation are often within noise. The two hollow points on the right are
        diagnostics: the same B and C, but with the bits selected using the test set (as in the original study). They are
        optimistically biased, and the gap to B and C shows how much that leak can inflate the score.
    """)
    df = pd.read_csv(CSV_FILE).set_index('Experiment')
    keys = ['Exp_A', 'Exp_B', 'Exp_C', 'Exp_Bleaky', 'Exp_Cleaky']
    n = len(keys)
    r2 = df.loc[keys, 'R2'].values
    lo = df.loc[keys, 'CI_Lower'].values
    hi = df.loc[keys, 'CI_Upper'].values
    mae = df.loc[keys, 'MAE'].values
    nf = df.loc[keys, 'N_Features'].astype(int).values
    ntr = df.loc[keys, 'Aug_Size'].astype(int).values

    names = ['A: Baseline', 'B: + Feature\nselection', 'C: + Gaussian\naugmentation',
             'B$_{leaky}$: + Feature\nselection', 'C$_{leaky}$: + Gaussian\naugmentation']
    xt = [f"{n}\n{f:,} feat. | n$_{{train}}$={t:,}" for n, f, t in zip(names, nf, ntr)]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(n)

    # Reference lines (paper) with direct labels instead of a legend
    for (lab, val), (ls, c) in zip(PAPER_R2.items(), REF_STYLE):
        ax.axhline(val, ls=ls, lw=1.2, color=c, zorder=1)
        ax.text(n - 0.44, val, f"{lab} ({val:.2f})", va='center', ha='left',
                fontsize=8, color=c)

    # Point estimate + 95% CI (no truncated bars)
    for i in range(n):
        leaky = i >= 3
        color = COLORS[i - 2] if leaky else COLORS[i]
        ax.errorbar(x[i], r2[i], yerr=[[r2[i] - lo[i]], [hi[i] - r2[i]]],
                    fmt='o', ms=9, capsize=6, lw=2, color=color,
                    mfc='white' if leaky else color, mec=color if leaky else 'white',
                    mew=1.8 if leaky else 1, zorder=3)
        ax.text(x[i] + 0.09, r2[i], f"R² = {r2[i]:.3f}\nMAE = {mae[i]:.3f}",
                va='center', ha='left', fontsize=8.5, color=color, zorder=4)
        ax.text(x[i], lo[i] - 0.012, f"[{lo[i]:.3f}, {hi[i]:.3f}]",
                va='top', ha='center', fontsize=7.5, color='#555555')

    ax.axvline(2.5, ls=':', lw=1.2, color='#555555', zorder=1)
    ax.text(3.5, 0.915, "Diagnostic: selection on test set (leaky)", va='top', ha='center',
            fontsize=8.5, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.57, 0.92)
    ax.set_ylabel("R² on held-out test set (95% bootstrap CI)")
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # widen right margin for reference labels
    fig.subplots_adjust(left=0.11, right=0.85, bottom=0.2, top=0.96)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f"{FIG_BASE}.{ext}", dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {FIG_BASE}.png / .pdf")


EXPERIMENT_DISPLAY = {
    "Exp_A": "A: no selection, no augmentation",
    "Exp_B": "B: selection on the development partition",
    "Exp_C": "C: B plus Gaussian-noise augmentation",
    "Exp_Bleaky": "B leaky: selection on the complete data set",
    "Exp_Cleaky": "C leaky: B leaky plus augmentation",
}


def write_ablation_table():
    """The five variants of the hold-out replication side by side, so that the figures the manuscript quotes for the
    leaky and the clean selection can be compared directly instead of being read one at a time from the text."""
    summary = pd.read_csv(CSV_FILE).set_index("Experiment")
    deltas = pd.read_csv(os.path.join(RESULTS_DIR, CSV_04_DELTA_FILE)).set_index("Comparison")
    # Each variant is compared with the one it modifies: the clean variants with the baseline, the leaky ones with
    # their own clean counterpart, which is what isolates the effect of the leak.
    against = {"Exp_B": "Exp_A", "Exp_C": "Exp_A", "Exp_Bleaky": "Exp_B", "Exp_Cleaky": "Exp_C"}
    lines = ["% Auto-generated by 04.1_build_figures.py",
             "\\begin{tabular}{lrrcc}", "\\toprule",
             "\\textbf{Variant} & \\textbf{Bits} & \\textbf{Train rows} & $R^2$ (95\\% CI) & "
             "$\\Delta R^2$ vs reference \\\\", "\\midrule"]
    for key, display in EXPERIMENT_DISPLAY.items():
        r = summary.loc[key]
        delta = ""
        ref = against.get(key)
        if ref is not None:
            d = deltas.loc[f"{key} - {ref}"]
            delta = (f"${d['Delta_R2_Point']:+.4f}$ [${d['CI_Lower']:+.3f}$, ${d['CI_Upper']:+.3f}$]"
                     f" \\textit{{vs {ref.replace('Exp_', '')}}}")
        lines.append(f"{display} & {int(r['N_Features'])} & {int(r['Aug_Size'])} & "
                     f"${r['R2']:.4f}$ [${r['CI_Lower']:.3f}$, ${r['CI_Upper']:.3f}$] & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(LATEX_ABLATION_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[EXPORT] Ablation table saved to {LATEX_ABLATION_TABLE}")


if __name__ == "__main__":
    generate_figure()
    write_ablation_table()

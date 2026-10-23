"""Project-specific comparisons against the classical models of step 01: equivalence margins, representation labels and
compare_to_classical. The corrected t-test and the Holm adjustment come from qsartools.stats."""
from qsartools.stats import corrected_ttest, holm

EQUIVALENCE_MARGIN_R2 = 0.05
EQUIVALENCE_MARGIN_MAE = 0.05


CLASSICAL_MODE_LABEL = {'morgan': 'Morgan', 'rdkit2d': 'RdkitTwoD', 'rdkit2d_fp': 'RdkitTwoDFp',
                        'rdkit2d3d_fp': 'RdkitTwoDThreeDFp'}


def compare_to_classical(scores, checkpoint, prefix):
    """Compare a model evaluated fold by fold (`scores`: Fold, R2, MAE, N_train, N_test) against every classical
    representation of the step 01 `checkpoint` on the same folds: corrected t-test, Holm adjustment within each metric,
    and equivalence test. Folds are paired through their order and checked through the test-fold sizes.
    Returns (DataFrame of comparisons, dict of LaTeX macros named `{prefix}{Representation}{Quantity}{Metric}`)."""
    import pandas as pd
    rows = []
    for metric, s_col, c_col, margin in (("RTwo", "R2", "R2_outer", EQUIVALENCE_MARGIN_R2),
                                         ("Mae", "MAE", "MAE_outer", EQUIVALENCE_MARGIN_MAE)):
        block = []
        for mode, label in CLASSICAL_MODE_LABEL.items():
            c = checkpoint[checkpoint['Mode'] == mode].sort_values('Fold').reset_index(drop=True)
            if len(c) != len(scores) or not (c['N_test'].values == scores['N_test'].values).all():
                print(f"  [WARNING] Folds of '{mode}' do not match the evaluated model's folds; comparison skipped.", flush=True)
                continue
            res = corrected_ttest(scores[s_col].values - c[c_col].values, scores['N_train'].mean(), scores['N_test'].mean(),
                                  margin=margin)
            block.append({"Metric": metric, "Classical": mode, "Label": label, "mean_diff_model_minus_classical": res["mean_diff"],
                          "t": res["t"], "p_raw": res["p"], "ci90_low": res["ci90_low"], "ci90_high": res["ci90_high"],
                          "equiv_margin": margin, "p_equiv": res["p_equiv"]})
        for row, p_adj in zip(block, holm([r["p_raw"] for r in block])):
            row["p_holm"] = p_adj
        rows += block
    macros = {}
    for r in rows:
        key = f"{prefix}{r['Label']}"
        macros[f"{key}Diff{r['Metric']}"] = f"{r['mean_diff_model_minus_classical']:+.4f}"
        macros[f"{key}P{r['Metric']}"] = f"{r['p_holm']:.3f}"
        macros[f"{key}EquivP{r['Metric']}"] = f"{r['p_equiv']:.3f}"
        macros[f"{key}CILow{r['Metric']}"] = f"{r['ci90_low']:+.3f}"
        macros[f"{key}CIHigh{r['Metric']}"] = f"{r['ci90_high']:+.3f}"
    for metric in ("RTwo", "Mae"):
        ps = [r['p_holm'] for r in rows if r['Metric'] == metric]
        if ps:
            macros[f"{prefix}MinP{metric}"] = f"{min(ps):.3f}"
    return pd.DataFrame(rows), macros

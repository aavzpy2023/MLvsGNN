import os
import sys

# Prevent C/C++ sublibraries from generating extra threads inside joblib worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
import math
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain

from rdkit import RDLogger
from qsartools.featurization import (RDKIT_2D_NAMES, RDKIT_3D_NAMES, morgan_fingerprints, rdkit_2d_descriptors,
                                     rdkit_3d_shape_descriptors)
from qsartools.latex import newcommand as macro
from qsartools.models import base_regressors, combination_id, descriptor_pipeline, model_combinations, stacking_regressor
from qsartools.nested_cv import nested_cv
from qsartools.splits import outer_splits
from qsartools.stats import corrected_ttest


RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# =========================================================
# PREDICTION CLIPPING
# =========================================================
CLIP_MARGIN = 3.0  # pIC50 units outside the [min, max] range of the training set
# n_train/n_test of a 5-fold split, used by the corrected t-tests of the summary (fixed ratio, not the exact fold sizes)
NB_TRAIN_TEST_RATIO = 4.0


NON_FEATURE_COLS = {
    'molecule chembl id', 'smiles', 'pic50 value', 'pic50',
    'standard value', 'ic50', 'cid', 'name', 'id',
}

# The reference PAPER_R2 is imported from paths_config.py

MODE_LATEX_LABEL = {
    'morgan': 'Morgan',
    'rdkit2d': 'RdkitTwoD',
    'rdkit2d_fp': 'RdkitTwoDFp',
    'rdkit2d3d_fp': 'RdkitTwoDThreeDFp',
    'csv_descriptors': 'CsvDescriptors',
}

MODE_DESCRIPTION = {
    'morgan': 'Morgan fingerprints (ECFP4, 2048 bits)',
    'rdkit2d': 'RDKit 2D physicochemical descriptors',
    'rdkit2d_fp': 'RDKit 2D descriptors + Morgan FP',
    'rdkit2d3d_fp': '2D + 3D (ETKDGv3 + MMFF94s) descriptors + Morgan FP',
    'csv_descriptors': 'Numeric columns in training CSV',
}

# =========================================================
# DESCRIPTOR GENERATION
# =========================================================
def detect_descriptor_columns(df):
    cols = []
    for c in df.columns:
        if c.strip().lower() in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_feature_matrix(df, mode, n_jobs):
    smiles = df['Smiles'].tolist()

    if mode == 'morgan':
        X = morgan_fingerprints(smiles)
        return X, [f'morgan_{i}' for i in range(X.shape[1])], True

    if mode == 'rdkit2d':
        return rdkit_2d_descriptors(smiles, n_jobs=n_jobs), list(RDKIT_2D_NAMES), False

    if mode == 'rdkit2d_fp':
        X2d, Xfp = rdkit_2d_descriptors(smiles, n_jobs=n_jobs), morgan_fingerprints(smiles)
        return np.hstack([X2d, Xfp]), list(RDKIT_2D_NAMES) + [f'morgan_{i}' for i in range(Xfp.shape[1])], False

    if mode == 'rdkit2d3d_fp':
        print("    [INFO] Generating 3D conformers (ETKDGv3 + MMFF)...", flush=True)
        X2d = rdkit_2d_descriptors(smiles, n_jobs=n_jobs)
        X3d = rdkit_3d_shape_descriptors(smiles, seed=RANDOM_STATE, n_jobs=n_jobs)
        Xfp = morgan_fingerprints(smiles)
        names = list(RDKIT_2D_NAMES) + list(RDKIT_3D_NAMES) + [f'morgan_{i}' for i in range(Xfp.shape[1])]
        return np.hstack([X2d, X3d, Xfp]), names, False

    if mode == 'csv_descriptors':
        cols = detect_descriptor_columns(df)
        X = df[cols].apply(pd.to_numeric, errors='coerce').values
        return X, cols, False

    raise ValueError(f"Unknown representation mode: {mode}")


# =========================================================
# MODELS & PIPELINES
# =========================================================
def build_pipeline(combo_dict, is_binary):
    return descriptor_pipeline(stacking_regressor(combo_dict, cv=5), is_binary,
                               selector_trees=CFG['N_ESTIMATORS_TREES'], seed=RANDOM_STATE)


# Rough relative fit cost per base model; only used to dispatch the heaviest tasks first.
_MODEL_COST = {'RF': 3.0, 'ET': 2.0, 'LGBM': 1.0, 'XGB': 1.5, 'SVM': 0.3, 'kNN': 0.1}


def _combo_cost(combo):
    cost = 3.0 + sum(_MODEL_COST[n] for n in combo)  # 3.0 = SelectFromModel RF
    return cost * 6 if len(combo) > 1 else cost  # StackingRegressor: cv=5 + final refit


# =========================================================
# CHECKPOINTING & SELECTION REGISTRY
# =========================================================
def load_done_folds(mode):
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    df = pd.read_csv(CHECKPOINT_FILE)
    return set(df[df['Mode'] == mode]['Fold'].unique().tolist())


def append_checkpoint(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(CHECKPOINT_FILE)
    df_row.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)


def append_selection_log(df_inner, mode, fold_idx):
    df_inner = df_inner.assign(Mode=mode, Fold=fold_idx)
    header = not os.path.exists(SELECTION_LOG_FILE)
    df_inner.to_csv(SELECTION_LOG_FILE, mode='a', header=header, index=False)


# =========================================================
# NESTED CROSS-VALIDATION LOOP
# =========================================================
def nested_cv_for_mode(mode, df_clean, cfg):
    total_folds = cfg['OUTER_N_SPLITS'] * cfg['OUTER_N_REPEATS']
    print("\n" + "#" * 100, flush=True)
    print(f"# NESTED CV ({cfg['OUTER_N_SPLITS']}x{cfg['OUTER_N_REPEATS']}={total_folds} Folds) — REPRESENTATION: {mode}", flush=True)
    print("#" * 100, flush=True)

    t_feat = time.time()
    X, feat_names, is_binary = build_feature_matrix(df_clean, mode, cfg['N_JOBS'])
    record_feature_qc(mode, X, feat_names, time.time() - t_feat)
    y = df_clean['pIC50 Value'].values
    n_features_total = X.shape[1]

    base_models = base_regressors(n_estimators=cfg['N_ESTIMATORS_TREES'], seed=RANDOM_STATE, n_jobs=1)
    all_combos = model_combinations(base_models, cfg['MAX_COMBO_SIZE'])

    outer_cv = list(outer_splits(df_clean['Smiles'].tolist(), cfg['OUTER_N_SPLITS'], cfg['OUTER_N_REPEATS'], RANDOM_STATE, SPLIT_KIND))
    done_folds = load_done_folds(mode)

    if done_folds:
        print(f"[CHECKPOINT] {len(done_folds)}/{total_folds} folds already completed for '{mode}', resuming...", flush=True)

    def save_fold(record, ranking, model):
        append_selection_log(ranking, mode, record['Fold'])
        n_selected = n_features_total
        if not is_binary:
            try:
                n_selected = int(model.named_steps['select'].get_support().sum())
            except Exception:
                pass
        append_checkpoint({
            'Mode': mode, 'Fold': record['Fold'], 'Selected_Model': record['Selected_Model'],
            'R2_outer': record['R2_outer'], 'MAE_outer': record['MAE_outer'],
            'N_train': record['N_train'], 'N_test': record['N_test'],
            'N_features_total': n_features_total, 'N_features_selected': n_selected,
            'N_predictions_clipped': record['N_predictions_clipped'],
            'Time_s': record['Time_s'],
        })
        print(f"[{mode}] Fold {record['Fold'] + 1}/{total_folds} -> Best: {record['Selected_Model']:<15} | "
              f"R2_outer: {record['R2_outer']:.4f} | MAE: {record['MAE_outer']:.4f} ({record['Time_s']:.1f}s)", flush=True)

    nested_cv({combination_id(c): c for c in all_combos}, X, y, outer_cv, inner_folds=cfg['INNER_N_SPLITS'],
              seed=RANDOM_STATE, make_estimator=partial(build_pipeline, is_binary=is_binary), clip_margin=CLIP_MARGIN,
              n_jobs=cfg['N_JOBS'], task_cost=_combo_cost, skip_folds=done_folds, on_fold=save_fold)


# =========================================================
# EXPORT TO LATEX AND FIGURES
# =========================================================
def latex_safe_combo(combo_id):
    return "".join(ch for ch in combo_id if ch.isalpha())


def newcommand(f, name, value):
    f.write(macro(name, value))


FEATURE_QC_FILE = os.path.join(RESULTS_DIR, "01_feature_qc.csv")
LEARNERS = ['RF', 'ET', 'LGBM', 'XGB', 'SVM', 'kNN']


def record_feature_qc(mode, X, feat_names, featurization_s):
    """Counts missing/failed values of a feature matrix before imputation (3D failures = rows with NaN in the 3D block)."""
    Xf = np.asarray(X, dtype=float)
    cols_3d = [i for i, n in enumerate(feat_names) if n in RDKIT_3D_NAMES]
    row = {'Mode': mode, 'N_rows': Xf.shape[0], 'N_features': Xf.shape[1],
           'N_cells_missing': int(np.isnan(Xf).sum()), 'N_rows_missing': int(np.isnan(Xf).any(axis=1).sum()),
           'N_rows_3D_failed': int(np.isnan(Xf[:, cols_3d]).any(axis=1).sum()) if cols_3d else 0,
           'Featurization_s': round(featurization_s, 2)}
    qc = pd.read_csv(FEATURE_QC_FILE) if os.path.exists(FEATURE_QC_FILE) else pd.DataFrame()
    if not qc.empty:
        qc = qc[qc['Mode'] != mode]
    pd.concat([qc, pd.DataFrame([row])], ignore_index=True).to_csv(FEATURE_QC_FILE, index=False)


def generate_latex_file(df_all, summary, df_ttest, top2_modes, paired_stats):
    with open(LATEX_PAPER, 'w', encoding='utf-8') as f:
        f.write("% =====================================================\n")
        f.write("% Nested CV Benchmark Variables (Auto-generated)\n")
        f.write("% =====================================================\n\n")

        newcommand(f, "NCompounds", len(pd.read_csv(TRAIN_FILE, on_bad_lines='skip').dropna(subset=['Smiles', 'pIC50 Value'])))
        newcommand(f, "OuterKFolds", CFG['OUTER_N_SPLITS'])
        newcommand(f, "OuterRepeats", CFG['OUTER_N_REPEATS'])
        newcommand(f, "TotalOuterFolds", CFG['OUTER_N_SPLITS'] * CFG['OUTER_N_REPEATS'])
        newcommand(f, "InnerKFolds", CFG['INNER_N_SPLITS'])
        newcommand(f, "MaxComboSize", CFG['MAX_COMBO_SIZE'])
        f.write("\n")

        for label, (_, val) in PAPER_R2.items():
            newcommand(f, f"{label}RTwo", f"{val:.2f}")
        f.write("\n")

        for _, row in summary.iterrows():
            mode = row['Mode']
            label = MODE_LATEX_LABEL.get(mode, mode.title())
            newcommand(f, f"{label}RTwoMean", f"{row['mean']:.4f}")
            newcommand(f, f"{label}RTwoStd", f"{row['std']:.4f}")
            newcommand(f, f"{label}NFolds", int(row['count']))

            df_mode = df_all[df_all['Mode'] == mode]
            counts = df_mode['Selected_Model'].value_counts()
            top_combo = counts.index[0]
            stability = counts.iloc[0] / len(df_mode) * 100
            newcommand(f, f"{label}BestArchitecture", top_combo)
            newcommand(f, f"{label}BestArchitectureSafe", latex_safe_combo(top_combo))
            newcommand(f, f"{label}Stability", f"{stability:.1f}")
            newcommand(f, f"{label}MaeMean", f"{df_mode['MAE_outer'].mean():.4f}")
            newcommand(f, f"{label}NFeaturesTotal", int(df_mode['N_features_total'].mean()))
            newcommand(f, f"{label}NFeaturesSelectedMean", f"{df_mode['N_features_selected'].mean():.1f}")

            members = df_mode['Selected_Model'].str.split('+')
            for learner in LEARNERS:
                newcommand(f, f"Sel{label}{learner}Pct", f"{members.map(lambda m: learner in m).mean() * 100:.0f}")
            sizes = members.map(len)
            for n_models, word in ((1, 'One'), (2, 'Two'), (3, 'Three')):
                newcommand(f, f"Sel{label}Size{word}Pct", f"{(sizes == n_models).mean() * 100:.0f}")
            newcommand(f, f"Sel{label}NArchitectures", len(counts))
            newcommand(f, f"Sel{label}TopCount", int(counts.iloc[0]))
            if len(counts) > 1:
                newcommand(f, f"Sel{label}Second", counts.index[1])
                newcommand(f, f"Sel{label}SecondCount", int(counts.iloc[1]))
            newcommand(f, f"Sel{label}OthersMaxCount", int(counts.iloc[2]) if len(counts) > 2 else 0)
            newcommand(f, f"Sel{label}NOthers", max(len(counts) - 2, 0))
        f.write("\n")

        newcommand(f, "NClippedPredictions", int(df_all['N_predictions_clipped'].sum()))
        newcommand(f, "NArchitectureCombos", sum(math.comb(len(LEARNERS), k) for k in range(1, CFG['MAX_COMBO_SIZE'] + 1)))
        newcommand(f, "NInnerFitsPerOuterFold",
                   sum(math.comb(len(LEARNERS), k) for k in range(1, CFG['MAX_COMBO_SIZE'] + 1)) * CFG['INNER_N_SPLITS'])
        if os.path.exists(FEATURE_QC_FILE):
            for _, qc in pd.read_csv(FEATURE_QC_FILE).iterrows():
                qlabel = MODE_LATEX_LABEL.get(qc['Mode'], qc['Mode'].title())
                newcommand(f, f"{qlabel}NCellsMissing", int(qc['N_cells_missing']))
                newcommand(f, f"{qlabel}NThreeDFailed", int(qc['N_rows_3D_failed']))
                newcommand(f, f"{qlabel}FeaturizationSec", f"{qc['Featurization_s']:.1f}")
        f.write("\n")

        for _, row in df_ttest.iterrows():
            mode = row['Mode']
            mlabel = MODE_LATEX_LABEL.get(mode, mode.title())
            plabel = row['Paper_reference_label']
            newcommand(f, f"PValue{mlabel}Vs{plabel}", f"{row['p_value']:.4f}")
            newcommand(f, f"TStat{mlabel}Vs{plabel}", f"{row['t_stat']:.3f}")
            sig = "true" if row['p_value'] < 0.05 else "false"
            newcommand(f, f"SigDiff{mlabel}Vs{plabel}", sig)
        f.write("\n")

        if paired_stats is not None:
            l1 = MODE_LATEX_LABEL.get(top2_modes[0], top2_modes[0].title())
            l2 = MODE_LATEX_LABEL.get(top2_modes[1], top2_modes[1].title())
            newcommand(f, "TopModeOne", l1)
            newcommand(f, "TopModeTwo", l2)
            newcommand(f, "PairedTTestStat", f"{paired_stats['t_stat']:.3f}")
            newcommand(f, "PairedTTestP", f"{paired_stats['t_p']:.4f}")
        f.write("\n")

        best_row = summary.iloc[0]
        blabel = MODE_LATEX_LABEL.get(best_row['Mode'], best_row['Mode'].title())
        newcommand(f, "WinnerMode", blabel)
        newcommand(f, "WinnerRTwoMean", f"{best_row['mean']:.4f}")
        newcommand(f, "WinnerRTwoStd", f"{best_row['std']:.4f}")

    print(f"[LATEX] Variables successfully exported to: {LATEX_PAPER}", flush=True)


# =========================================================
# FINAL ANALYSIS
# =========================================================
def summarize_and_test():
    df = pd.read_csv(CHECKPOINT_FILE)

    print("\n" + "=" * 100, flush=True)
    print("RESULTS SUMMARY (Nested CV 5x5)", flush=True)
    print("=" * 100, flush=True)

    summary = df.groupby('Mode')['R2_outer'].agg(['mean', 'std', 'count']).reset_index()
    summary = summary.sort_values(by='mean', ascending=False)
    explain("How to read the summary tables", """
        Table 1: mean R2 over all outer folds for each molecular description, with its standard deviation (spread between
        folds) and the number of folds. The top row is the best-performing description. Differences smaller than about one
        standard deviation are usually not meaningful.

        Table 2: tests whether our R2 differs from the R2 values reported in the original publication (PAPER_R2). The
        Nadeau-Bengio correction is applied because cross-validation folds overlap and share training data, which makes a
        naive t-test over-confident. 'Significant_p<0.05' = True means our result is really different from the reference
        (higher or lower: compare Our_R2_mean with Paper_R2); False means we reproduce it within statistical noise.

        The paired comparison at the end tests whether the two best descriptions differ from each other on the same folds.
    """)
    print("\nTable 1. R² by molecular representation:", flush=True)
    print(summary.to_string(index=False), flush=True)

    # 1. t-tests with Nadeau-Bengio correction vs Original Publication
    ttest_rows = []
    for mode in df['Mode'].unique():
        vals = df[df['Mode'] == mode]['R2_outer'].values
        for label, (desc, paper_val) in PAPER_R2.items():
            res = corrected_ttest(vals, n_train=NB_TRAIN_TEST_RATIO, n_test=1.0, popmean=paper_val)
            t_stat, p_val = res['t'], res['p']
            ttest_rows.append({
                'Mode': mode, 'Paper_reference_label': label, 'Paper_reference_desc': desc,
                'Paper_R2': paper_val, 'Our_R2_mean': vals.mean(), 'Our_R2_std': vals.std(),
                't_stat': t_stat, 'p_value': p_val, 'Significant_p<0.05': p_val < 0.05,
            })
    df_ttest = pd.DataFrame(ttest_rows)
    print("\nTable 2. t-tests (Nadeau-Bengio Corrected) vs Original Publication:", flush=True)
    print(df_ttest.drop(columns=['Paper_reference_desc']).to_string(index=False), flush=True)

    # 2. Paired comparison between the 2 best representations
    top2 = summary.head(2)['Mode'].tolist()
    paired_stats = None
    if len(top2) == 2:
        a = df[df['Mode'] == top2[0]].sort_values('Fold')['R2_outer'].values
        b = df[df['Mode'] == top2[1]].sort_values('Fold')['R2_outer'].values
        if len(a) == len(b) and len(a) > 1:
            res = corrected_ttest(a - b, n_train=NB_TRAIN_TEST_RATIO, n_test=1.0)
            t_stat, p_val = res['t'], res['p']
            paired_stats = {'t_stat': t_stat, 't_p': p_val}
            print(f"\nPaired comparison (Nadeau-Bengio) between {top2[0]} and {top2[1]}:", flush=True)
            print(f"  t = {t_stat:.3f}, p = {p_val:.4f}", flush=True)

    # Export main results
    summary.to_csv(FINAL_RESULTS_FILE, index=False)
    df_ttest.to_csv(FINAL_RESULTS_FILE.replace('.csv', '_ttests.csv'), index=False)

    # 3. Export to Supplementary Table S1 (Architecture Selection Frequency)
    if os.path.exists(SELECTION_LOG_FILE):
        log_df = pd.read_csv(SELECTION_LOG_FILE)
        if 'R2_inner_mean' in log_df.columns:
            winners_per_fold = (
                log_df.sort_values('R2_inner_mean', ascending=False)
                .groupby(['Mode', 'Fold'])
                .first()
                .reset_index()
            )

            freq_df = (
                winners_per_fold.groupby(['Mode', 'Model'])
                .size()
                .reset_index(name='times_selected')
            )

            total_folds = winners_per_fold.groupby('Mode')['Fold'].nunique()
            freq_df['total_folds'] = freq_df['Mode'].map(total_folds)
            freq_df['selection_percentage'] = (freq_df['times_selected'] / freq_df['total_folds']) * 100
            freq_df = freq_df.sort_values(by=['Mode', 'times_selected'], ascending=[True, False])

            out_path = os.path.join(RESULTS_DIR, '01_architecture_selection_frequency.csv')
            freq_df.to_csv(out_path, index=False)
            print(f"\n[SUPPLEMENTARY CSV] Selection frequency exported to: {out_path}", flush=True)
            explain("Architecture selection frequency", """
                For each description, how often each algorithm combination won the inner selection across folds. If one
                combination wins most folds, it is a stable choice; if many different ones win, the data cannot clearly
                distinguish them (many are about equally good). Step 03 uses the most frequent winner. SVM appearing very often
                suggests smooth, similarity-based models fit this dataset well.
            """)

    # Export to LaTeX and generate figure
    generate_latex_file(df, summary, df_ttest, top2, paired_stats)


def run_benchmark():
    # Initialize unified logger
    log_file_path = os.path.join(LOGS_DIR, "01_nested_cv_stacking.log")
    setup_logger(log_file_path)

    total_folds = CFG['OUTER_N_SPLITS'] * CFG['OUTER_N_REPEATS']
    print(f"[INFO] Active profile: {PROFILE} | N_JOBS={CFG['N_JOBS']} | Outer folds={CFG['OUTER_N_SPLITS']}x{CFG['OUTER_N_REPEATS']}={total_folds}", flush=True)
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    df_clean = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    y_all = df_clean['pIC50 Value']
    print(f"[DATA] {TRAIN_FILE}: {len(df)} rows read, {len(df_clean)} usable (SMILES and pIC50 present). "
          f"pIC50 range {y_all.min():.2f} - {y_all.max():.2f}, mean {y_all.mean():.2f}, SD {y_all.std():.2f}.", flush=True)
    explain("What this step does", f"""
        Builds and honestly evaluates machine-learning models that predict how potently a molecule inhibits TgDHFR (pIC50, higher
        = more potent) from its structure. The dataset has {len(df_clean)} molecules with measured potency.

        Molecules must be converted to numbers before a model can use them. Several descriptions are compared:
        {'; '.join(MODE_DESCRIPTION.get(m, m) for m in CFG['FEATURE_MODES'])}.
        A fingerprint is a long 0/1 list saying which small substructures are present; a descriptor is a number such as
        molecular weight, polarity or number of rotatable bonds; '3D' descriptors also depend on the molecule's shape.

        For each description, the models tried are single algorithms (Random Forest, Extra Trees, LightGBM, XGBoost, SVM,
        kNN) and stacks of up to {CFG['MAX_COMBO_SIZE']} of them combined by a final Ridge regression. NESTED cross-validation
        is used so the reported score is not optimistic: the data is split into {CFG['OUTER_N_SPLITS']} parts and the whole
        split is repeated {CFG['OUTER_N_REPEATS']} times (= {total_folds} outer folds). Inside every outer fold the training part is
        split again ({CFG['INNER_N_SPLITS']} inner folds) to choose the best algorithm combination; only then is that choice
        scored on the outer test part, which it has never seen. Selecting and scoring on the same data would inflate R2.

        This is by far the slowest step of the pipeline (each outer fold fits hundreds of models). Progress is saved after every
        fold (results/01_nested_cv_checkpoint.csv), so an interrupted run resumes where it stopped. IMPORTANT: if the training
        data changes, delete that checkpoint first or old-data folds would be silently reused.

        Per-fold line: 'Best' = the algorithm combination that won the inner selection; R2_outer = fraction of the variance in
        potency explained on the unseen part (1 = perfect, 0 = no better than predicting the mean); MAE = average error in
        pIC50 units (0.5 = about a 3-fold error in IC50); the time is for that fold.
    """)

    for mode in CFG['FEATURE_MODES']:
        try:
            nested_cv_for_mode(mode, df_clean, CFG)
        except ValueError as e:
            print(f"[WARNING] Skipping mode '{mode}': {e}", flush=True)

    if os.path.exists(CHECKPOINT_FILE):
        summarize_and_test()


if __name__ == "__main__":
    run_benchmark()

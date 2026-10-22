import os
import sys
import time
import warnings
import logging
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import RDLogger

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.base import clone
from xgboost import XGBRegressor

from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain
from tools.featurization import morgan_fingerprints

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION & PARAMETERS FOR DATA AUGMENTATION
# =========================================================
# Dynamically extract hardware/profile configurations
N_JOBS = ACTIVE_CFG.get('N_JOBS', 1)
N_ESTIMATORS = ACTIVE_CFG.get('N_ESTIMATORS_TREES', 200)
logger = logging.getLogger(__name__)

# =========================================================
# MODEL BASE ESTIMATORS (MULTITHREADING ENABLED)
# =========================================================
# Base estimators of the deployed Stacking Regressor. This architecture is fixed in advance; it is not the one most often
# selected by the nested CV in step 01 (RF+SVM for Morgan).
# Every model of this step is fitted and evaluated single-threaded: a multi-threaded RandomForest.predict() sums the trees
# in a thread-dependent order, which changes the last bits of the predictions between runs. Parallelism is applied
# across independent tasks instead (features in the permutation importance, runs in the ensemble), which cannot change results.
_BASE_ESTIMATORS = {
    'RF': ('rf', RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)),
    'XGB': ('xgb', XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1, verbosity=0)),
    'SVM': ('svm', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)),
}
STACK_MEMBERS = [_BASE_ESTIMATORS[name] for name in DEPLOYED_ARCHITECTURE]
# Members whose seed changes between the runs of the ensemble (SVR is deterministic)
SEEDED_MEMBERS = [n for n, _ in STACK_MEMBERS if n in ('rf', 'xgb')]

# =========================================================
# MOLECULAR REPRESENTATIONS
# =========================================================
def build_X(smiles_list):
    return morgan_fingerprints(smiles_list).astype(float)


# =========================================================
# MODEL BUILDER
# =========================================================
def build_model(groups=None, X=None, y=None):
    """Returns a fresh instance of the fixed stacking architecture (STACK_MEMBERS + RidgeCV meta-learner).
    With augmented data, `groups` (the original molecule of each row) makes the stacking's internal 5-fold split keep all
    noisy copies of a molecule on the same side, so that the meta-learner is not fitted on out-of-fold predictions of
    molecules whose copies the base models have seen."""
    estimators = [(n, clone(m)) for n, m in STACK_MEMBERS]
    if len(estimators) == 1:
        return clone(estimators[0][1])

    # If N_JOBS is set here, it will multiply threads (N_JOBS * N_JOBS) and freeze the server.
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5 if groups is None else list(GroupKFold(n_splits=5).split(X, y, groups)),
        n_jobs=1,
    )


# =========================================================
# PERMUTATION IMPORTANCE FEATURE SELECTION
# =========================================================
# Features that the model never uses have an importance that is zero up to floating-point noise (about 1e-15), so a plain
# "> 0" test keeps a random subset of them. The smallest genuine importances are about 1e-7.
IMPORTANCE_TOL = 1e-9


def _fit_and_permute(model, X_fit, y_fit, X_eval, y_eval, n_repeats):
    """Fits a clone of `model` on (X_fit, y_fit) and returns the mean permutation importance measured on (X_eval, y_eval)."""
    fitted = clone(model)
    fitted.fit(X_fit, y_fit)
    result = permutation_importance(
        fitted, X_eval, y_eval,
        scoring='neg_mean_squared_error',
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    return result.importances_mean


def _importances_to_mask(importances, label):
    """Keeps features whose importance exceeds IMPORTANCE_TOL and logs how many were kept."""
    mask = importances > IMPORTANCE_TOL
    n_zero = int((np.abs(importances) <= IMPORTANCE_TOL).sum())
    logger.info(f"[{label}] Features selected (importance > {IMPORTANCE_TOL:g}): {int(mask.sum())}/{len(mask)} "
                f"({n_zero} features have a numerically zero importance)")
    return mask


def select_features_by_permutation_importance(X, y, model, n_repeats=5):
    """
    Out-of-fold permutation feature importance on the development set.
    For each of PI_CV_FOLDS folds a clone of the model is fitted on the training part and the importance is measured on
    the held-out part; the fold means are averaged. Retains features whose averaged importance is positive beyond
    floating-point noise (> IMPORTANCE_TOL).
    Fitting and permuting on the same data would measure how much the trees memorised each bit, not how much it helps
    predict unseen molecules, so the importance is always computed on data the fitted model has not seen.
    Returns (mask, importances).
    """
    logger.info(f"Computing out-of-fold Permutation Importance (folds={PI_CV_FOLDS}, n_repeats={n_repeats}, n_jobs={N_JOBS})...")
    kf = KFold(n_splits=PI_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = [
        _fit_and_permute(model, X[tr], y[tr], X[va], y[va], n_repeats)
        for tr, va in kf.split(X)
    ]
    importances = np.mean(fold_importances, axis=0)
    return _importances_to_mask(importances, "out-of-fold"), importances


def select_features_leaky(X_dev, y_dev, X_test, y_test, model, n_repeats=5):
    """
    DIAGNOSTIC ONLY. Fits on the development set and permutes on the TEST set, which reproduces the selection protocol of
    the original study. The test labels therefore influence which bits are kept, so any score obtained with this mask is
    optimistically biased. It exists only to quantify that bias; it must never feed a persisted model.
    Returns (mask, importances).
    """
    logger.info("Computing LEAKY Permutation Importance (fit on dev, permute on test) — diagnostic only...")
    importances = _fit_and_permute(model, X_dev, y_dev, X_test, y_test, n_repeats)
    return _importances_to_mask(importances, "leaky"), importances


# =========================================================
# GAUSSIAN AUGMENTATION
# =========================================================
def augment_data(X_train, y_train, noise_levels=None, seed=RANDOM_STATE):
    """
    Applies Gaussian noise to training feature vectors (Section 3.5).
    Original sample + 1 copy per noise level -> 3x dataset expansion.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    rng = np.random.RandomState(seed)
    X_parts = [X_train]
    y_parts = [y_train]
    for sigma in noise_levels:
        noise = rng.normal(0, sigma, X_train.shape)
        X_parts.append(X_train + noise)
        y_parts.append(y_train)
    X_aug = np.vstack(X_parts)
    y_aug = np.concatenate(y_parts)
    groups = np.tile(np.arange(len(y_train)), len(X_parts))
    logger.info(f"Dataset expansion: {len(y_train)} -> {len(y_aug)} (noise levels: {noise_levels})")
    return X_aug, y_aug, groups


# =========================================================
# BOOTSTRAP STATISTICS & EMPIRICAL P-VALUES
# =========================================================
def bootstrap_r2_distribution(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Generates non-parametric bootstrap sampling distribution of R2 scores."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(r2_score(y_true[idx], y_pred[idx]))
    return np.array(boots)


def compute_bootstrap_metrics(boot_array, ci=0.95):
    """Calculates bootstrap mean and percentile-based confidence intervals."""
    mean_val = float(np.mean(boot_array))
    lo, hi = np.percentile(boot_array, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return mean_val, float(lo), float(hi)


def calculate_empirical_bootstrap_pvalue(boot_array, reference_val):
    """
    Calculates a two-tailed empirical bootstrap p-value against a point reference value.
    Avoids parametric t-test pseudoreplication artifacts on bootstrap draws.
    """
    prop_below = np.mean(boot_array <= reference_val)
    prop_above = np.mean(boot_array >= reference_val)
    p_emp = 2.0 * min(prop_below, prop_above)
    return float(min(1.0, p_emp))


# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def _fit_predict_run(X_train, y_train, X_test, run, groups=None):
    m = build_model(groups, X_train, y_train)
    m.set_params(**{f'{n}__random_state': RANDOM_STATE + run for n in SEEDED_MEMBERS})
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return pred.flatten() if pred.ndim > 1 else pred


def ensemble_predict(X_train, y_train, X_test, n_runs=N_ENSEMBLE_RUNS, groups=None):
    """Trains n_runs stochastic models (different seeds for the tree ensembles) and returns their averaged predictions.
    Each run is single-threaded and independent, so running them in parallel does not change the result."""
    all_preds = Parallel(n_jobs=max(1, min(n_runs, N_JOBS)))(
        delayed(_fit_predict_run)(X_train, y_train, X_test, run, groups) for run in range(n_runs))
    return np.mean(all_preds, axis=0)


def _score_experiment(label, y_test, y_pred, n_features_total, n_features_selected, aug_size, t0):
    """Computes R2, MAE and bootstrap CI for one experiment, logs them and returns the results entry."""
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    boot_r2s = bootstrap_r2_distribution(y_test, y_pred)
    _, ci_lo, ci_hi = compute_bootstrap_metrics(boot_r2s)
    logger.info(f"Results Exp {label} -> R2={r2:.4f} | MAE={mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] | "
                f"Time: {time.time() - t0:.1f}s")
    return dict(r2=r2, mae=mae, ci_lo=ci_lo, ci_hi=ci_hi, boot_r2s=boot_r2s,
                n_features_total=n_features_total, n_features_selected=n_features_selected, aug_size=aug_size)


# =========================================================
# LATEX EXPORT
# =========================================================
def newcommand(f, name, value):
    f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def export_latex(results_dict, delta_rows=None):
    os.makedirs(os.path.dirname(LATEX_FILE), exist_ok=True)
    with open(LATEX_FILE, 'w', encoding='utf-8') as f:
        f.write("% =====================================================\n")
        f.write("% Variables auto-generated by 04_augmentation_training.py\n")
        f.write("% Includes rigorous bootstrap CIs and non-parametric p-values\n")
        f.write("% =====================================================\n\n")

        newcommand(f, "ExpNDev", str(results_dict['split']['n_dev']))
        newcommand(f, "ExpNTest", str(results_dict['split']['n_test']))
        f.write("\n")

        f.write("% --- Experiment A: Baseline Stacking (No Feat Sel, No Aug) ---\n")
        a = results_dict['A']
        newcommand(f, "ExpARTwoMean", f"{a['r2']:.4f}")
        newcommand(f, "ExpAMae", f"{a['mae']:.4f}")
        newcommand(f, "ExpARTwoCILow", f"{a['ci_lo']:.4f}")
        newcommand(f, "ExpARTwoCIHigh", f"{a['ci_hi']:.4f}")
        f.write("\n")

        f.write("% --- Experiment B: Permutation Importance Selection ---\n")
        b = results_dict['B']
        newcommand(f, "ExpBRTwoMean", f"{b['r2']:.4f}")
        newcommand(f, "ExpBMae", f"{b['mae']:.4f}")
        newcommand(f, "ExpBRTwoCILow", f"{b['ci_lo']:.4f}")
        newcommand(f, "ExpBRTwoCIHigh", f"{b['ci_hi']:.4f}")
        newcommand(f, "ExpBNFeaturesTotal", str(b['n_features_total']))
        newcommand(f, "ExpBNFeaturesSelected", str(b['n_features_selected']))
        f.write("\n")

        f.write("% --- Experiment C: Feature Selection + Gaussian Augmentation ---\n")
        c = results_dict['C']
        newcommand(f, "ExpCRTwoMean", f"{c['r2']:.4f}")
        newcommand(f, "ExpCMae", f"{c['mae']:.4f}")
        newcommand(f, "ExpCRTwoCILow", f"{c['ci_lo']:.4f}")
        newcommand(f, "ExpCRTwoCIHigh", f"{c['ci_hi']:.4f}")
        newcommand(f, "ExpCAugFactor", str(1 + len(NOISE_LEVELS)))
        newcommand(f, "ExpCAugSizeTrain", str(c['aug_size']))
        newcommand(f, "ExpCNoiseLevels", str(NOISE_LEVELS).replace('[', '').replace(']', ''))
        f.write("\n")

        f.write("% --- Diagnostic: Experiments B/C with selection on the test set (leaky, original-study protocol) ---\n")
        for key in ('Bleaky', 'Cleaky'):
            r = results_dict[key]
            newcommand(f, f"Exp{key}RTwoMean", f"{r['r2']:.4f}")
            newcommand(f, f"Exp{key}Mae", f"{r['mae']:.4f}")
            newcommand(f, f"Exp{key}RTwoCILow", f"{r['ci_lo']:.4f}")
            newcommand(f, f"Exp{key}RTwoCIHigh", f"{r['ci_hi']:.4f}")
            newcommand(f, f"Exp{key}NFeaturesSelected", str(r['n_features_selected']))
        newcommand(f, "ExpLeakyMaskJaccard", f"{results_dict['leaky_mask_jaccard']:.4f}")
        f.write("\n")

        f.write("% --- Reference Benchmarks ---\n")
        for label, (_, val) in PAPER_R2.items():
            f.write(f"\\providecommand{{\\{label}RTwo}}{{{val:.2f}}}\n")
        f.write("\n")

        f.write("% --- Rigorous Statistical Evaluations vs Benchmarks ---\n")
        for exp_key in ('A', 'B', 'C', 'Bleaky', 'Cleaky'):
            exp = results_dict[exp_key]
            for paper_key, (_, paper_val) in PAPER_R2.items():
                is_within_ci = exp['ci_lo'] <= paper_val <= exp['ci_hi']
                p_boot = calculate_empirical_bootstrap_pvalue(exp['boot_r2s'], paper_val)
                sig = "false" if is_within_ci else "true"

                newcommand(f, f"Exp{exp_key}Vs{paper_key}InCI", "true" if is_within_ci else "false")
                newcommand(f, f"Exp{exp_key}Vs{paper_key}PBoot", f"{p_boot:.4f}")
                newcommand(f, f"Exp{exp_key}Vs{paper_key}Sig", sig)

        # --- Paired bootstrap differences between experiments ---
        if delta_rows:
            f.write("\n% --- Paired bootstrap R2 differences (same resampled test indices) ---\n")
            for d in delta_rows:
                tag = d['tag']  # e.g. CvsA
                newcommand(f, f"Exp{tag}DeltaRTwo", f"{d['delta_point']:+.4f}")
                newcommand(f, f"Exp{tag}DeltaCILow", f"{d['ci_lo']:+.4f}")
                newcommand(f, f"Exp{tag}DeltaCIHigh", f"{d['ci_hi']:+.4f}")
                newcommand(f, f"Exp{tag}DeltaPBoot", f"{d['p_boot']:.4f}")

    logger.info(f"Variables successfully exported to {LATEX_FILE}")


# =========================================================
# MAIN EXECUTION PIPELINE
# =========================================================
def run():
    # Initialize unified logger
    log_file_path = os.path.join(LOGS_DIR, "04_augmentation_training.log")
    setup_logger(log_file_path)

    logger.info("=" * 80)
    logger.info("04_augmentation_training.py — Feature Selection & Gaussian Augmentation Benchmark")
    logger.info("Evaluates classical ML pipeline parity against reported DNN benchmarks")
    logger.info("=" * 80)
    explain("What this step does", """
        Trains the final predictive model and tests variants on compounds the model never saw (15% of the data is set
        aside before anything else happens, so the test is honest):

        Exp A - baseline: a stacking ensemble (several different algorithms whose predictions are combined by a final model).
        Exp B - A plus feature selection: keeps only the fingerprint bits that actually help prediction (permutation
        importance: a bit is useful if randomly scrambling it makes predictions worse). The importance is computed out-of-fold
        inside the development set: each model is scored on molecules it was not fitted on, so a bit is kept only if it helps
        predict unseen molecules, not if the trees merely memorised it.
        Exp C - B plus data augmentation: adds slightly noisy copies of the training molecules' potencies (Gaussian noise) to
        see whether this makes the model more robust.
        Exp Bleaky / Exp Cleaky - diagnostics only: the same as B and C, but the bits are selected by permuting on the TEST
        set, as the original study did. The test molecules then influence the model, so these scores are optimistically
        biased; comparing them with B and C measures how much of a reported gain can come from that leak. They never feed
        the saved mask or models.

        Metrics: R2 (1 = perfect, 0 = no better than predicting the mean potency) and MAE (average error in pIC50 units;
        0.5 means a typical error of ~3-fold in IC50). The 95% CI comes from bootstrap resampling of the test set: it shows how
        much the score would move if a different sample of test molecules had been drawn. With so few test molecules the CI is
        wide, so small R2 differences between experiments are often not meaningful.

        Outputs used downstream: results/04_best_model.joblib (the un-augmented model used for the virtual screening in step 05)
        and results/04_selected_features_mask.npy (which fingerprint bits are kept).
    """)

    # 1. Load Data
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    logger.info(f"Data Loaded: {len(df)} compounds successfully loaded.")

    X_all = build_X(df['Smiles'].tolist())
    y_all = df['pIC50 Value'].values

    # 2. Train / Test Split (15% test held-out BEFORE selection/augmentation)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Data Split: Dev={len(y_dev)} | Test held-out={len(y_test)} (isolated from tuning/augmentation)")

    results = {'split': dict(n_dev=len(y_dev), n_test=len(y_test))}

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT A: Baseline Stacking Ensemble (No Feature Selection, No Augmentation)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("-" * 80)
    logger.info("EXPERIMENT A: ML Stacking Baseline (all fingerprint bits, original data only)")
    logger.info("-" * 80)
    t0 = time.time()
    y_pred_a = ensemble_predict(X_dev, y_dev, X_test)
    results['A'] = _score_experiment('A', y_test, y_pred_a, X_all.shape[1], X_all.shape[1], len(y_dev), t0)

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT B: Permutation Importance Feature Selection
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("-" * 80)
    logger.info("EXPERIMENT B: + Permutation Importance Feature Selection")
    logger.info("-" * 80)
    t0 = time.time()
    pi_model = build_model()
    mask_b, _ = select_features_by_permutation_importance(X_dev, y_dev, pi_model)

    X_dev_b = X_dev[:, mask_b]
    X_test_b = X_test[:, mask_b]

    y_pred_b = ensemble_predict(X_dev_b, y_dev, X_test_b)
    results['B'] = _score_experiment('B', y_test, y_pred_b, X_all.shape[1], int(mask_b.sum()), len(y_dev), t0)

    os.makedirs(os.path.dirname(MASK_FILE), exist_ok=True)
    np.save(MASK_FILE, mask_b)
    logger.info(f"Feature mask saved to: {MASK_FILE}")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT C: Feature Selection + Gaussian Augmentation
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("-" * 80)
    logger.info("EXPERIMENT C: + Gaussian Data Augmentation")
    logger.info("-" * 80)
    t0 = time.time()

    X_aug, y_aug, groups_aug = augment_data(X_dev_b, y_dev)

    y_pred_c = ensemble_predict(X_aug, y_aug, X_test_b, groups=groups_aug)
    results['C'] = _score_experiment('C', y_test, y_pred_c, X_all.shape[1], int(mask_b.sum()), len(y_aug), t0)

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC EXPERIMENTS Bleaky / Cleaky: selection on the TEST set (original-study protocol)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("-" * 80)
    logger.info("DIAGNOSTIC EXPERIMENTS Bleaky / Cleaky: feature selection on the test set (LEAKY, not for use)")
    logger.info("-" * 80)
    t0 = time.time()
    mask_leaky, _ = select_features_leaky(X_dev, y_dev, X_test, y_test, build_model())
    union = int((mask_leaky | mask_b).sum())
    jaccard = float((mask_leaky & mask_b).sum() / union) if union else 1.0
    results['leaky_mask_jaccard'] = jaccard
    logger.info(f"Jaccard index between leaky and out-of-fold masks: {jaccard:.4f}")
    logger.info("The leaky mask is NOT saved to MASK_FILE and does NOT feed any persisted model: "
                "MASK_FILE, MODEL_FILE and the augmented model are built from the out-of-fold mask (Exp B).")

    X_dev_bl = X_dev[:, mask_leaky]
    X_test_bl = X_test[:, mask_leaky]
    n_sel_leaky = int(mask_leaky.sum())

    y_pred_bleaky = ensemble_predict(X_dev_bl, y_dev, X_test_bl)
    results['Bleaky'] = _score_experiment('Bleaky', y_test, y_pred_bleaky, X_all.shape[1], n_sel_leaky,
                                          len(y_dev), t0)

    t0 = time.time()
    X_aug_l, y_aug_l, groups_aug_l = augment_data(X_dev_bl, y_dev)
    y_pred_cleaky = ensemble_predict(X_aug_l, y_aug_l, X_test_bl, groups=groups_aug_l)
    results['Cleaky'] = _score_experiment('Cleaky', y_test, y_pred_cleaky, X_all.shape[1], n_sel_leaky,
                                          len(y_aug_l), t0)

    # ──────────────────────────────────────────────────────────────────────────
    # EXECUTIVE SUMMARY & RIGOROUS HYPOTHESIS TESTING
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("EXECUTIVE SUMMARY — RIGOROUS STATISTICAL AUDIT")
    logger.info("=" * 80)
    explain("How to read the summary", """
        Each experiment (including the leaky diagnostics Bleaky / Cleaky) is compared with R2 values reported in the
        original paper (PAPER_R2). 'NOT significantly different'
        means the published value lies inside our 95% CI, i.e. we reproduce it within sampling uncertainty. p_boot is the
        fraction of bootstrap resamples that fall on the other side of the reference value (small = real difference).

        The paired delta R2 lines then ask whether adding feature selection (B vs A) or augmentation (C vs B / C vs A) truly
        improved the model. If the CI of a delta includes 0, the improvement cannot be distinguished from noise.
        The leaky deltas (Bleaky vs A / B, Cleaky vs C) show how much selecting bits on the test set inflates the score.
    """)

    for exp_key, exp_label in [('A', 'Exp A (baseline)'), ('B', 'Exp B (+feat.sel.)'), ('C', 'Exp C (+augment.)'),
                               ('Bleaky', 'Exp Bleaky (leaky, diagnostic)'), ('Cleaky', 'Exp Cleaky (leaky, diagnostic)')]:
        exp = results[exp_key]
        logger.info(
            f"{exp_label}: R2={exp['r2']:.4f} | MAE={exp['mae']:.4f} | 95% CI [{exp['ci_lo']:.4f}, {exp['ci_hi']:.4f}]")
        for paper_key, (desc, paper_val) in PAPER_R2.items():
            in_ci = exp['ci_lo'] <= paper_val <= exp['ci_hi']
            p_boot = calculate_empirical_bootstrap_pvalue(exp['boot_r2s'], paper_val)
            sig_text = "NOT significantly different (within 95% CI)" if in_ci else "SIGNIFICANTLY different"
            direction = "above" if exp['r2'] > paper_val else "below"
            logger.info(
                f"  vs {desc} (R2={paper_val}): {direction} by {abs(exp['r2'] - paper_val):.4f} | p_boot={p_boot:.4f} -> {sig_text}")

    # Paired Bootstrap Difference Tests.
    # bootstrap_r2_distribution() uses the same seed for every experiment, so draw i uses
    # the SAME resampled test indices in every experiment -> differences are properly paired.
    logger.info("-" * 80)
    logger.info("PAIRED BOOTSTRAP DIFFERENCES (delta R2):")
    delta_rows = []
    for tag, k1, k0 in [('CvsA', 'C', 'A'), ('BvsA', 'B', 'A'), ('CvsB', 'C', 'B'),
                          ('BleakyvsA', 'Bleaky', 'A'), ('BleakyvsB', 'Bleaky', 'B'), ('CleakyvsC', 'Cleaky', 'C')]:
        diff = results[k1]['boot_r2s'] - results[k0]['boot_r2s']
        d_mean, d_lo, d_hi = compute_bootstrap_metrics(diff)
        d_p = calculate_empirical_bootstrap_pvalue(diff, 0.0)
        d_point = results[k1]['r2'] - results[k0]['r2']
        delta_rows.append(dict(tag=tag, Comparison=f"Exp_{k1} - Exp_{k0}", delta_point=d_point,
                               delta_boot_mean=d_mean, ci_lo=d_lo, ci_hi=d_hi, p_boot=d_p))
        logger.info(f"Delta R2 (Exp {k1} - Exp {k0}): point={d_point:+.4f} | boot mean={d_mean:+.4f} "
                    f"(95% CI [{d_lo:+.4f}, {d_hi:+.4f}], p_boot={d_p:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL PERSISTENCE FOR VIRTUAL SCREENING (05_virtual_screening.py)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("-" * 80)
    logger.info("PERSISTENCE AUDIT — Matching Section 2.5 Protocol")
    logger.info("-" * 80)

    # Clean model (unaugmented) as required by paper Section 2.5 for Virtual Screening
    logger.info("Fitting clean baseline model (NO augmentation) on dev set for Virtual Screening...")
    final_model_clean = build_model()
    final_model_clean.fit(X_dev_b, y_dev)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(final_model_clean, MODEL_FILE)
    logger.info(f"Saved unaugmented clean baseline model to: {MODEL_FILE}")

    # Optional: Save augmented model under separate filename if required for downstream auditing
    aug_model_path = MODEL_FILE.replace('.joblib', '_augmented.joblib')
    final_model_aug = build_model(groups_aug, X_aug, y_aug)
    final_model_aug.fit(X_aug, y_aug)
    joblib.dump(final_model_aug, aug_model_path)
    logger.info(f"Saved augmented model separately to: {aug_model_path}")

    csv_data = []
    for exp_key in ['A', 'B', 'C', 'Bleaky', 'Cleaky']:
        csv_data.append({
            'Experiment': f'Exp_{exp_key}',
            'R2': results[exp_key]['r2'],
            'MAE': results[exp_key]['mae'],
            'CI_Lower': results[exp_key]['ci_lo'],
            'CI_Upper': results[exp_key]['ci_hi'],
            'N_Features': results[exp_key]['n_features_selected'],
            'Aug_Size': results[exp_key]['aug_size']
        })

    df_results = pd.DataFrame(csv_data)
    csv_path = os.path.join(RESULTS_DIR, CSV_04_OUTPUT_FILE)
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results summary saved to {csv_path}")

    # Paired differences, full bootstrap distributions and test predictions (for figures/audits)
    df_delta = pd.DataFrame([{
        'Comparison': d['Comparison'], 'Delta_R2_Point': d['delta_point'],
        'Delta_R2_BootMean': d['delta_boot_mean'], 'CI_Lower': d['ci_lo'],
        'CI_Upper': d['ci_hi'], 'P_Boot': d['p_boot']} for d in delta_rows])
    df_delta.to_csv(os.path.join(RESULTS_DIR, CSV_04_DELTA_FILE), index=False)

    pd.DataFrame({f'Exp_{k}': results[k]['boot_r2s'] for k in ('A', 'B', 'C', 'Bleaky', 'Cleaky')}).to_csv(
        os.path.join(RESULTS_DIR, CSV_04_BOOT_FILE), index=False)

    pd.DataFrame({'y_true': y_test, 'pred_Exp_A': y_pred_a, 'pred_Exp_B': y_pred_b,
                  'pred_Exp_C': y_pred_c, 'pred_Exp_Bleaky': y_pred_bleaky,
                  'pred_Exp_Cleaky': y_pred_cleaky}).to_csv(os.path.join(RESULTS_DIR, CSV_04_PRED_FILE), index=False)
    logger.info(f"Delta R2, bootstrap distributions and test predictions saved to {RESULTS_DIR}")

    # Export artifacts
    export_latex(results, delta_rows)
    logger.info("04_augmentation_training.py execution complete with Q1 statistical rigor.")


if __name__ == "__main__":
    run()

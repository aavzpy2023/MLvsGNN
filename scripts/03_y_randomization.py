import os
import sys
import time
import warnings
import logging
import numpy as np
import pandas as pd

# Suppress scikit-learn feature name mismatch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain

from sklearn.pipeline import Pipeline

from qsartools.feature_selection import PermutationImportanceSelector
from qsartools.featurization import morgan_fingerprints
from qsartools.latex import format_macros
from qsartools.models import base_regressors, stacking_regressor
from qsartools.y_randomization import y_randomization

logger = logging.getLogger(__name__)


# =========================================================
# 1. DYNAMIC DISCOVERY OF THE BEST MODEL
# =========================================================
# Architecture of the deployed screening model (DEPLOYED_ARCHITECTURE in paths_config, shared with step 04)
DEPLOYED_MODE = 'morgan'
# Rucker et al. (2007): y-randomization must repeat the whole model-building procedure, and its result is compared with
# models built on random pseudodescriptors. "architecture" is the stacking on all 2048 bits; "procedure" adds the
# out-of-fold permutation-importance bit selection of step 04 in front of it, redone inside every training set.
TESTS = [("YRandom", "architecture", "y_permutation"), ("YRandomPseudo", "architecture", "pseudodescriptors"),
         ("YRandomProc", "procedure", "y_permutation"), ("YRandomProcPseudo", "procedure", "pseudodescriptors")]
# Verdict by the permutation p-value; with 100 randomizations the smallest attainable value is 1/101
ALPHA = 0.01


def main():
    # Initialize unified logger
    log_file_path = os.path.join(LOGS_DIR, "03_y_randomization.log")
    setup_logger(log_file_path)

    logger.info("=" * 80)
    logger.info("03_y_randomization.py — Model Robustness Test (Y-Randomization)")
    logger.info("=" * 80)
    explain("What this step does", """
        Sanity check that the model really learned chemistry and did not just memorise noise or benefit from a data leak.

        Procedure (Y-randomization): the measured potencies (pIC50) are randomly shuffled between molecules, which destroys any
        true structure-activity relationship, and the same model is trained and cross-validated again. This is repeated many
        times. A healthy model scores well on the real data and about R2 = 0 (or negative) on the shuffled data.

        R2 is the fraction of the variation in potency that the model explains: 1.0 is perfect, 0 is no better than always
        predicting the average potency, negative is worse than that.
    """)

    best_mode, best_architecture = DEPLOYED_MODE, '+'.join(DEPLOYED_ARCHITECTURE)
    logger.info(f"Deployed model -> representation: '{best_mode}', architecture: '{best_architecture}'")

    # 1. Load Data
    logger.info(f"Loading training data from: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    df = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    y_true = df['pIC50 Value'].values
    logger.info(f"Successfully loaded {len(df)} valid SMILES and target values.")

    # 2. Feature Matrix Generation
    logger.info(f"Generating feature matrix X for representation mode '{best_mode}'...")
    X_raw = morgan_fingerprints(df['Smiles'].tolist()).astype(np.int8)

    X = pd.DataFrame(X_raw, columns=[f"fp_{i}" for i in range(X_raw.shape[1])])
    logger.info(f"Feature matrix dimensions: {X.shape}")

    base_models = base_regressors(seed=RANDOM_STATE)
    model = stacking_regressor({k: base_models[k] for k in DEPLOYED_ARCHITECTURE}, cv=CV_FOLDS)
    procedures = {
        "architecture": model,
        "procedure": Pipeline([("select", PermutationImportanceSelector(model, n_folds=PI_CV_FOLDS, seed=RANDOM_STATE)),
                               ("model", model)]),
    }

    explain("The four tests", f"""
        Rucker et al. (2007) define y-randomization as permuting the potencies and repeating the WHOLE model-building
        procedure, because the risk of chance correlation comes mainly from selecting a few descriptors out of many. They
        also propose a second comparison, with descriptors replaced by random ones (pseudodescriptors), because real
        descriptors are intercorrelated and shuffled potencies alone give lower chance scores.

        Each test is run on the stacking alone ("architecture", all 2048 bits) and on the procedure that produces the
        screening model ("procedure": the out-of-fold permutation-importance bit selection of step 04, redone in every
        training set, then the stacking). Scores are {CV_FOLDS}-fold cross-validated R2: tree ensembles fit even random
        potencies almost perfectly, so the R2 of the fit would not discriminate. A test passes when the permutation
        p-value is <= {ALPHA}.
    """)
    # Every score is saved as soon as it is computed (Randomization -1 is the real data) and reused by a later run, so an
    # interruption costs only the randomizations in progress. Delete the checkpoint when the procedure changes.
    saved = {}
    if os.path.exists(YRANDOM_CHECKPOINT_FILE):
        for row in pd.read_csv(YRANDOM_CHECKPOINT_FILE, float_precision="round_trip").itertuples():
            saved.setdefault(row.Test, {})[int(row.Randomization)] = float(row.R2)
        logger.info(f"Resuming from {YRANDOM_CHECKPOINT_FILE}: "
                    + ", ".join(f"{k} {len(v)} scores" for k, v in saved.items()))
    else:
        os.makedirs(os.path.dirname(YRANDOM_CHECKPOINT_FILE), exist_ok=True)
        with open(YRANDOM_CHECKPOINT_FILE, "w") as f:
            f.write("Test,Randomization,R2\n")

    def checkpoint(prefix):
        def save(i, score):
            with open(YRANDOM_CHECKPOINT_FILE, "a") as f:
                f.write(f"{prefix},{i},{score!r}\n")
        return save

    rows, values = [], {}
    start_time = time.time()
    for prefix, procedure, variant in TESTS:
        done = saved.get(prefix, {})
        logger.info(f"[{prefix}] {procedure} / {variant}: real data and {N_PERMUTATIONS} randomizations "
                    f"({CV_FOLDS}-fold CV, {N_JOBS} worker processes; {len(done)} scores from the checkpoint)...")
        t0 = time.time()
        res = y_randomization(procedures[procedure], X, y_true, n_permutations=N_PERMUTATIONS, n_folds=CV_FOLDS,
                              seed=RANDOM_STATE, n_jobs=N_JOBS, variant=variant, statistic="cv_r2",
                              true_r2=done.get(-1), done={i: s for i, s in done.items() if i >= 0},
                              on_result=checkpoint(prefix))
        passed = res["p_empirical"] <= ALPHA
        logger.info(f"[{prefix}] real R2 = {res['true_r2']:.4f} | randomized R2 = {res['permuted_mean']:.4f} "
                    f"± {res['permuted_sd']:.4f} | Z = {res['z']:.1f} | p = {res['p_empirical']:.4f} -> "
                    f"{'PASSED' if passed else 'FAILED'} ({time.time() - t0:.0f}s)")
        rows += [{"Test": prefix, "Procedure": procedure, "Variant": variant, "Randomization": i, "R2": r2}
                 for i, r2 in enumerate(res["permuted_r2"])]
        values.update({f"{prefix}TrueRTwo": f"{res['true_r2']:.4f}", f"{prefix}MeanRTwo": f"{res['permuted_mean']:.4f}",
                       f"{prefix}StdRTwo": f"{res['permuted_sd']:.4f}", f"{prefix}ZScore": f"{res['z']:.1f}",
                       f"{prefix}PValue": f"{res['p_empirical']:.4f}", f"{prefix}Passed": "true" if passed else "false"})
    elapsed_time = time.time() - start_time

    csv_path = os.path.join(RESULTS_DIR, "03_y_randomization_scores.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Randomized scores of the four tests saved to {csv_path}")

    values.update({"YRandomArchitecture": best_architecture, "YRandomNPermutations": N_PERMUTATIONS,
                   "YRandomCVFolds": CV_FOLDS, "YRandomAlpha": ALPHA,
                   "YRandomMinPValue": f"{1 / (N_PERMUTATIONS + 1):.4f}"})

    # 5. Export Variables to LaTeX
    latex_file = os.path.join(LATEX_DIR, "yrandom_variables.tex")
    logger.info(f"Exporting Y-Randomization LaTeX variables to: {latex_file}")
    try:
        os.makedirs(os.path.dirname(latex_file), exist_ok=True)
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(format_macros(values, header="Auto-generated by 03_y_randomization.py"))
        logger.info("LaTeX variable export completed successfully.")
    except Exception as e:
        logger.error(f"Failed to export LaTeX variables: {e}")

    logger.info(f"Total execution time: {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()

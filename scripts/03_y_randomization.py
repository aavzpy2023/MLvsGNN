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

from qsartools.featurization import morgan_fingerprints
from qsartools.latex import format_macros
from qsartools.models import base_regressors, stacking_regressor
from qsartools.y_randomization import y_randomization

logger = logging.getLogger(__name__)


# =========================================================
# 1. DYNAMIC DISCOVERY OF THE BEST MODEL
# =========================================================
# Architecture of the deployed screening model (DEPLOYED_ARCHITECTURE in paths_config, shared with step 04), evaluated on
# all 2048 Morgan bits: this validates the architecture, not the feature-selected model deployed in step 05.
DEPLOYED_MODE = 'morgan'


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

    logger.info(f"Evaluating the unpermuted model and {N_PERMUTATIONS} target permutations "
                f"({CV_FOLDS}-fold CV, {N_JOBS} worker processes)...")
    start_time = time.time()
    res = y_randomization(model, X, y_true, n_permutations=N_PERMUTATIONS, n_folds=CV_FOLDS, seed=RANDOM_STATE,
                          n_jobs=N_JOBS)
    true_r2, random_scores = res["true_r2"], res["permuted_r2"]
    logger.info(f"Baseline Unpermuted R2: {true_r2:.4f}")

    elapsed_time = time.time() - start_time
    df_random = pd.DataFrame({'Permuted_R2': random_scores})
    csv_path = os.path.join(RESULTS_DIR, "03_y_randomization_scores.csv")
    df_random.to_csv(csv_path, index=False)
    logger.info(f"Y-randomization distribution saved to {csv_path}")

    mean_random, std_random, z_score = res["permuted_mean"], res["permuted_sd"], res["z"]

    explain("How to read the report below", f"""
        'Baseline R2' is the real model ({true_r2:.2f}). 'Random R2' is what the same model achieves when potencies are shuffled
        ({mean_random:.2f} on average over {N_PERMUTATIONS} shuffles). The Z-score counts how many standard deviations the real
        result sits above the shuffled results; the pass criterion used here is real R2 > mean + 2.33 SD (~99% confidence).

        If the verdict is PASSED, the predictive signal is not a chance correlation. Note that this test uses Morgan
        fingerprints and the stacking architecture of the deployed screening model (RF+XGB+SVM, as in step 04).
    """)
    logger.info("-" * 80)
    logger.info("Y-RANDOMIZATION INTEGRITY REPORT")
    logger.info("-" * 80)
    logger.info(f"  Baseline R2 (True Target) : {true_r2:.4f}")
    logger.info(f"  Random R2 (Mean)          : {mean_random:.4f}")
    logger.info(f"  Random R2 (Std)           : {std_random:.4f}")
    logger.info(f"  Z-Score                   : {z_score:.2f}")

    # 99% confidence interval threshold (~2.33 std deviations)
    if true_r2 > (mean_random + 2.33 * std_random):
        logger.info("ACADEMIC VERDICT: PASSED. Model performance on true target is significantly superior to random chance (no memory leakage/overfitting).")
    else:
        logger.warning("ACADEMIC VERDICT: FAILED. High risk of chance correlation or data leakage.")

    # 5. Export Variables to LaTeX
    latex_file = os.path.join(LATEX_DIR, "yrandom_variables.tex")
    logger.info(f"Exporting Y-Randomization LaTeX variables to: {latex_file}")
    try:
        os.makedirs(os.path.dirname(latex_file), exist_ok=True)
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(format_macros({
                "YRandomTrueRTwo": f"{true_r2:.4f}",
                "YRandomMeanRTwo": f"{mean_random:.4f}",
                "YRandomStdRTwo": f"{std_random:.4f}",
                "YRandomZScore": f"{z_score:.1f}",
                "YRandomArchitecture": best_architecture,
                "YRandomNPermutations": N_PERMUTATIONS,
                "YRandomCVFolds": CV_FOLDS,
            }, header="Auto-generated by 03_y_randomization.py"))
        logger.info("LaTeX variable export completed successfully.")
    except Exception as e:
        logger.error(f"Failed to export LaTeX variables: {e}")

    logger.info(f"Total execution time: {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()

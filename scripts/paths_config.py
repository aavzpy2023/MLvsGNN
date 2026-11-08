# =========================================================
# paths_config.py — Central path and experimental configuration
# Import this in every script: from paths_config import *
# =========================================================
import os

# Project root = parent of the scripts/ folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
DATA_DIR      = os.path.join(ROOT, "data")
RECEPTOR_DIR  = os.path.join(ROOT, "receptor")
RESULTS_DIR   = os.path.join(ROOT, "results")
LATEX_DIR     = os.path.join(ROOT, "latex")
FIGURES_DIR   = os.path.join(ROOT, "figures")
LOGS_DIR      = os.path.join(ROOT, "logs")

# Sensitivity analysis: run_pipeline.py sets MLGNN_OUT_DIR so that a second run never overwrites the main outputs
SENS_DIR = os.environ.get("MLGNN_SENS_DIR") or os.path.join(ROOT, "sensitivity")
if os.environ.get("MLGNN_OUT_DIR"):
    _out = os.environ["MLGNN_OUT_DIR"]
    RESULTS_DIR, LATEX_DIR, FIGURES_DIR, LOGS_DIR = (os.path.join(_out, d) for d in ("results", "latex", "figures", "logs"))
    for _d in (RESULTS_DIR, LATEX_DIR, FIGURES_DIR, LOGS_DIR):
        os.makedirs(_d, exist_ok=True)

# Data files
RAW_FILE      = os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique_full.csv")
MAIN_FILE     = os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique_dedup.csv")
SENS_FILE     = os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique_dedup_with_outliers.csv")
TRAIN_FILE    = os.environ.get("MLGNN_TRAIN_FILE") or MAIN_FILE
FDA_FILE      = os.path.join(DATA_DIR, "PubChem_FDA-approved_NoInorganics.csv")

# Receptor files
RECEPTOR_PDB  = os.path.join(RECEPTOR_DIR, "receptor.pdb")
RECEPTOR_PDBQT= os.path.join(RECEPTOR_DIR, "receptor.pdbqt")

# Result files
CHECKPOINT_FILE      = os.path.join(RESULTS_DIR, "01_nested_cv_checkpoint.csv")
SELECTION_LOG_FILE   = os.path.join(RESULTS_DIR, "01_nested_cv_selection_log.csv")
OOF_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "01_nested_cv_oof_predictions.csv")
YRANDOM_CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "03_y_randomization_checkpoint.csv")
FINAL_RESULTS_FILE   = os.path.join(RESULTS_DIR, "01_nested_cv_final_results.csv")
MODEL_FILE           = os.path.join(RESULTS_DIR, "04_best_model.joblib")
MASK_FILE            = os.path.join(RESULTS_DIR, "04_selected_features_mask.npy")

# ADMET & Docking Candidates
FDA_RAW_CANDIDATES_CSV   = os.path.join(RESULTS_DIR, "05_FDA_Candidates_For_Docking.csv")
FDA_ADMET_CANDIDATES_CSV = os.path.join(RESULTS_DIR, "08_ADMET_CANDIDATES_For_Docking.csv")
FDA_CANDIDATES_CSV       = os.path.join(RESULTS_DIR, "05_FDA_Candidates_For_Docking.csv")
DOCKING_RESULTS_CSV       = os.path.join(RESULTS_DIR, "07_Final_Validation_Hybrid.csv")
AD_SENSITIVITY_CSV       = os.path.join(RESULTS_DIR, "05_ad_sensitivity.csv")

# LaTeX files
LATEX_PAPER     = os.path.join(LATEX_DIR, "paper_variables.tex")
LATEX_AUGMENT   = os.path.join(LATEX_DIR, "augment_variables.tex")
LATEX_FDA       = os.path.join(LATEX_DIR, "fda_variables.tex")
LATEX_DOCKING   = os.path.join(LATEX_DIR, "docking_variables.tex")
LATEX_REDOCKING = os.path.join(LATEX_DIR, "redocking_variables.tex")
LATEX_SCREENING  = os.path.join(LATEX_DIR, "screening_variables.tex")
LATEX_DATA       = os.path.join(LATEX_DIR, "data_variables.tex")
LATEX_ADMET      = os.path.join(LATEX_DIR, "admet_variables.tex")
LATEX_TIMING     = os.path.join(LATEX_DIR, "timing_variables.tex")
LATEX_COST_TABLE = os.path.join(LATEX_DIR, "cost_table.tex")
LATEX_FDA_TABLE  = os.path.join(LATEX_DIR, "fda_candidates_table.tex")
LATEX_AD_SENSITIVITY_TABLE = os.path.join(LATEX_DIR, "ad_sensitivity_table.tex")
LATEX_ADMET_TABLE = os.path.join(LATEX_DIR, "admet_table.tex")
LATEX_LADDER_TABLE = os.path.join(LATEX_DIR, "leakage_ladder_table.tex")
LATEX_LADDER_SEEDS_TABLE = os.path.join(LATEX_DIR, "leakage_ladder_seeds_table.tex")
LATEX_GNN_PAIRWISE_TABLE = os.path.join(LATEX_DIR, "gnn_pairwise_table.tex")
LATEX_GNN_TUNED_CONFIG_TABLE = os.path.join(LATEX_DIR, "gnn_tuned_config_table.tex")
LATEX_ABLATION_TABLE = os.path.join(LATEX_DIR, "ablation_table.tex")
LATEX_STRONG_TABLE = os.path.join(LATEX_DIR, "strong_baselines_table.tex")
LATEX_SCAFFOLD_TABLE = os.path.join(LATEX_DIR, "scaffold_table.tex")
LATEX_PAIRWISE_TABLE = os.path.join(LATEX_DIR, "pairwise_nb_table.tex")
LATEX_SCAFFOLD_PAIRWISE_TABLE = os.path.join(LATEX_DIR, "scaffold_pairwise_table.tex")
LATEX_STRONG_PAIRWISE_TABLE = os.path.join(LATEX_DIR, "strong_pairwise_table.tex")
LATEX_SELECTION_TABLE = os.path.join(LATEX_DIR, "selection_table.tex")
LATEX_YRANDOM_TABLE = os.path.join(LATEX_DIR, "yrandom_table.tex")

# Figure files
FIGURE_NESTED_CV = os.path.join(FIGURES_DIR, "r2_by_representation_boxplot.png")
FIGURE_AUGMENT   = os.path.join(FIGURES_DIR, "augment_r2_comparison.png")

RANDOM_STATE = 42

# Outer-fold strategy: 'random' (repeated K-fold) or 'scaffold' (Bemis-Murcko scaffold-grouped K-fold); see qsartools/splits.py
SPLIT_KIND = os.environ.get('MLGNN_SPLIT', 'random')

# =========================================================
# PROFILES TO RUN (5x5 REPEATED NESTED CV) (1)
# =========================================================
# Select the profile without editing the file:  export MLGNN_PROFILE=laptop
PROFILE = os.environ.get('MLGNN_PROFILE', 'workstation')

PROFILES = {
    # One representation only: enough to develop or debug a single step, but the figures and the LaTeX variables of the
    # other three representations are then never written, so this profile cannot produce a complete manuscript.
    'laptop': dict(
        N_JOBS=2,
        FEATURE_MODES=['morgan'],
        MAX_COMBO_SIZE=2,
        OUTER_N_SPLITS=3,
        OUTER_N_REPEATS=1,
        INNER_N_SPLITS=3,
        N_ESTIMATORS_TREES=50,
    ),
    # Smoke run of the whole pipeline (run_pipeline.py --quick). It keeps every representation and every combination
    # size, so each figure, table and LaTeX variable of the manuscript is still produced, and cuts only what costs time:
    # one outer repeat instead of five, a smaller inner loop, fewer trees, fewer permutations and fewer ensemble runs.
    # The numbers it produces are therefore NOT the published ones; --quick writes them to a separate output directory.
    'quick': dict(
        N_JOBS=46,
        FEATURE_MODES=['morgan', 'rdkit2d', 'rdkit2d_fp', 'rdkit2d3d_fp'],
        MAX_COMBO_SIZE=3,
        OUTER_N_SPLITS=5,
        OUTER_N_REPEATS=1,
        INNER_N_SPLITS=3,
        N_ESTIMATORS_TREES=50,
    ),
    # Generalization benchmark over the MoleculeACE targets (run_pipeline.py --only G..). Identical to 'workstation'
    # except for a single outer repeat: 12 targets at five repeats each would cost weeks, and the comparison between
    # models is paired within every target, so it does not need the 25 folds of the main benchmark. The main TgDHFR
    # analysis keeps its 5x5 protocol; it is re-run under this profile only to place it among the other targets.
    'generalization': dict(
        N_JOBS=46,
        FEATURE_MODES=['morgan', 'rdkit2d'],
        MAX_COMBO_SIZE=3,
        OUTER_N_SPLITS=5,
        OUTER_N_REPEATS=1,
        INNER_N_SPLITS=5,
        N_ESTIMATORS_TREES=200,
    ),
    'workstation': dict(
        N_JOBS=46,
        FEATURE_MODES=['morgan', 'rdkit2d', 'rdkit2d_fp', 'rdkit2d3d_fp'],
        MAX_COMBO_SIZE=3,
        OUTER_N_SPLITS=5,
        OUTER_N_REPEATS=5,
        INNER_N_SPLITS=5,
        N_ESTIMATORS_TREES=200,
    ),
}
if PROFILE not in PROFILES:
    raise ValueError(f"Unknown MLGNN_PROFILE '{PROFILE}'. Valid options: {list(PROFILES)}")

# Never request more workers than the machine has (override: export MLGNN_N_JOBS=8)
_CPU = os.cpu_count() or 1
_n_jobs_env = os.environ.get('MLGNN_N_JOBS')
PROFILES[PROFILE]['N_JOBS'] = (
    max(1, int(_n_jobs_env)) if _n_jobs_env else min(PROFILES[PROFILE]['N_JOBS'], _CPU)
)
CFG = PROFILES[PROFILE]
if os.environ.get('MLGNN_MODES'):
    CFG['FEATURE_MODES'] = [m for m in os.environ['MLGNN_MODES'].split(',') if m]

# FIGURES

mode_labels = {
    'morgan': 'Morgan FP',
    'rdkit2d': 'RDKit 2D',
    'rdkit2d_fp': 'RDKit 2D FP',
    'rdkit2d3d_fp': 'RDKit 2D+3D FP'
}


# =========================================================
# y-RANDOMIZATION EXPERIMENT CONFIGURATION (3)
# =========================================================
# 100 in a real run: the smallest attainable p is 1/(N+1), so alpha = 0.01 needs at least 99. The 'quick' profile uses
# 5, which cannot reach that alpha and is only meant to prove that the step runs.
N_PERMUTATIONS = 5 if PROFILE == 'quick' else 100

# Inner cross-validation of the nested CV (Krstajic et al., 2014, Algorithm 1): number of repeats of the inner K-fold (Nexp1)
# and how a candidate is scored ('pooled': all out-of-fold predictions of a repeat together; 'fold_mean': mean of per-fold
# scores). MLGNN_INNER_REPEATS=1 MLGNN_INNER_SCORE=fold_mean reproduces the protocol of the results before 2026-10-24.
INNER_N_REPEATS = int(os.environ.get('MLGNN_INNER_REPEATS', '5'))
INNER_SCORE = os.environ.get('MLGNN_INNER_SCORE', 'pooled')

# Dynamically extract CV_FOLDS and N_JOBS from the active profile
ACTIVE_CFG = PROFILES[PROFILE]
CV_FOLDS = ACTIVE_CFG.get('OUTER_N_SPLITS', 5)
N_JOBS = ACTIVE_CFG.get('N_JOBS', 1)



# =========================================================
# CONFIGURATION & PARAMETERS FOR DATA AUGMENTATION (4)
# =========================================================
TEST_SIZE = 0.15  # 15% holdout test set (identical to paper)
N_ENSEMBLE_RUNS = 2 if PROFILE == 'quick' else 5  # stochastic ensemble runs (averaged)
# Fixed architecture of the deployed screening model (base-estimator names of step 01), shared by steps 03 and 04.
DEPLOYED_ARCHITECTURE = ('RF', 'XGB', 'SVM')
PI_CV_FOLDS = 5  # folds of the out-of-fold permutation importance inside the development partition
N_BOOTSTRAP = 2000  # Resampling iterations for non-parametric CIs
LATEX_FILE = LATEX_AUGMENT
FIGURE_FILE = FIGURE_AUGMENT

# Gaussian noise perturbation levels
NOISE_LEVELS = [0.01, 0.001]
CSV_04_OUTPUT_FILE = "04_augmentation_summary.csv"
CSV_04_DELTA_FILE = "04_augmentation_delta_r2.csv"          # paired bootstrap differences
CSV_04_BOOT_FILE = "04_augmentation_bootstrap_r2.csv"       # full bootstrap R2 distributions
CSV_04_PRED_FILE = "04_augmentation_test_predictions.csv"   # test-set predictions per experiment



# =========================================================
# CONFIGURATION FOR VIRTUAL SCREENING (5)
# =========================================================
# We directly use the variables exposed in paths_config.py
# Applicability domain (qsartools.applicability_domain): similarity to the training set (Sheridan et al., 2004), the mean
# Tanimoto similarity to the AD_K nearest training molecules. Inside the domain = at least mean + AD_Z standard deviations
# of the training distances (Sahigara et al., 2012, default z = 0.5). The value of z is arbitrary: the literature gives no
# rule for it, so the screening reports the similarity of every candidate and does not exclude on it.
AD_K = 5
AD_Z = 0.5
MAX_MW = 1000.0
# Miscoverage of the split-conformal interval reported with every predicted potency, calibrated on the hold-out of
# step 04 (qsartools.stats.conformal_halfwidth). Its coverage is marginal over that partition, not conditional on a
# given molecule, so it does not widen outside the applicability domain.
CONFORMAL_ALPHA = 0.10
# Applicability-domain thresholds compared in the sensitivity table: (label, strategy, z, percentile)
AD_SENSITIVITY_RULES = (("mean", "mean_std", 0.0, None), ("mean + 0.5 SD", "mean_std", 0.5, None),
                        ("mean + 1 SD", "mean_std", 1.0, None), ("95th percentile", "p95", None, 95),
                        ("twice the mean", "twice_mean", None, None))

# Allowed atoms (identical to the original paper)
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

# References for internal precision validation
KNOWN_VALUES = {
    'Pyrimethamine': 6.56,
    'Trimethoprim': 5.57,
}

# Candidates from the original paper (Table 2)
PAPER_TOP = ['Bisacodyl', 'Etodolac', 'Triamterene', 'Finerenone',
             'Methotrexate', 'Pyrimethamine', 'Trimethoprim']


# =========================================================
# REDOCKING CONFIGURATION (6)
# =========================================================
PDB_FILE = RECEPTOR_PDB
RECEPTOR_FILE = RECEPTOR_PDBQT
LIGAND_CODE = "CP6"
# Pyrimethamine; the crystal PDB has no bond orders, so they are assigned from this template (redocking, step 06)
LIGAND_TEMPLATE_SMILES = "CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1"
CHAIN = "B"  # DHFR active site chain
EXHAUSTIVENESS = 32
RMSD_THRESHOLD = 2.0
BOX_SIZE = 20.0


# =========================================================
# DOCKING & VINA EXPERIMENTAL PARAMETERS
# =========================================================
# Active site grid box coordinates (detected from crystal ligand)
CENTER_X, CENTER_Y, CENTER_Z = 3.689, 39.992, -62.818
BOX_SIZE       = 20.0   # Angstroms
EXHAUSTIVENESS = 32     # Search depth (32 = publication quality)
N_POSES        = 3      # Top poses per ligand

# Decision Thresholds (ML + Physics classification)
ML_ACTIVE_THRESH   = 6.5   # pIC50 > 6.5 = predicted active
DOCK_ACTIVE_THRESH = -8.0  # kcal/mol < -8.0 = binding confirmed
ML_INACTIVE_THRESH   = 6.0   # pIC50 < 6.0 = predicted inactive
DOCK_INACTIVE_THRESH = -7.5  # kcal/mol > -7.5 = binding not supported

# Compound Selection Settings
TOP_CANDIDATES_COUNT = 15
REF_CONTROL_NAMES    = [
    'Pyrimethamine', 'Trimethoprim', 'Bisacodyl',
    'Etodolac', 'Triamterene', 'Methotrexate', 'Chlorambucil'
]

# Control compounds exported to LaTeX: tuple(search_substring, latex_prefix)
LATEX_REF_COMPOUNDS = [
    ('Chlorambucil', 'Chlorambucil'),
    ('Bisacodyl', 'Bisacodyl'),
    ('Pyrimethamine', 'Pyrimethamine'),
    ('Folic Acid', 'FolicAcid'),
    ('Methotrexate', 'Methotrexate'),
    ('Triamterene', 'Triamterene'),
    ('Pralatrexate', 'Pralatrexate'),
    ('Trimetrexate', 'Trimetrexate'),
    ('Trimethoprim', 'Trimethoprim')
]

# =========================================================
# HARDWARE & ADMET EXPERIMENTAL PARAMETERS
# =========================================================
# Hardware
CORES_ADMET = min(48, os.cpu_count() or 1)

# ADMET Thresholds
HERG_THRESH  = 0.5         # Probability < 0.5 = Low cardiotoxicity risk
CACO2_THRESH = -5.15       # Permeability > -5.15 log(cm/s) = Moderate/High Oral Permeability

# Lipinski Rule of Five Parameters
LIPINSKI_MAX_HDONORS    = 5
LIPINSKI_MAX_HACCEPTORS = 10
LIPINSKI_MAX_LOGP       = 5.0


# =========================================================
# GNN BASELINE CONFIGURATION
# =========================================================
# Result files
GNN_RESULTS_CSV        = os.path.join(RESULTS_DIR, "09_gnn_results.csv")
GNN_OOF_PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "09_gnn_oof_predictions.csv")
GNN_FOLD_SCORES_CSV    = os.path.join(RESULTS_DIR, "09_gnn_fold_scores.csv")
GNN_TUNED_CHECKPOINT   = os.path.join(RESULTS_DIR, "12_gnn_tuned_checkpoint.csv")
GNN_TUNED_VS_CLASSICAL_CSV = os.path.join(RESULTS_DIR, "12_gnn_tuned_vs_classical_tests.csv")
GNN_VS_CLASSICAL_CSV   = os.path.join(RESULTS_DIR, "09_gnn_vs_classical_tests.csv")
LATEX_GNN       = os.path.join(LATEX_DIR, "gnn_variables.tex")
LATEX_GNN_TUNED = os.path.join(LATEX_DIR, "gnn_tuned_variables.tex")
FIGURE_GNN      = os.path.join(FIGURES_DIR, "gnn_comparison.png")

# Experimental Parameters
N_WORKERS_GNN    = 0
N_BOOTSTRAP_GNN  = 2000
TEST_SIZE_GNN    = 0.15

# Reference R2 values for plotting/comparison
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection', 0.75),
    'PaperSelected': ('After Permutation Importance selection', 0.82),
    'PaperFinal': ('Data augmentation + DNN ensemble', 0.85),
}


def get_classical_r2(results_file):
    """
    Dynamically loads the best classical ML R2 scores (mean AND std) from script 01 results.
    Prevents hardcoding values and adapts to new runs.
    Returns a tuple (means_dict, stds_dict) with matching keys.
    """
    # Fallback values in case script 01 hasn't been run yet
    default_r2 = {
        'Morgan FP (RF+XGB+SVM)': 0.7407,
        'RDKit 2D+FP (best)': 0.7322,
    }
    default_std = {
        'Morgan FP (RF+XGB+SVM)': 0.0,
        'RDKit 2D+FP (best)': 0.0,
    }

    if not os.path.exists(results_file):
        return default_r2, default_std

    try:
        import csv
        means, stds = {}, {}
        with open(results_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                means[row['Mode']] = float(row['mean'])
                stds[row['Mode']] = float(row['std'])

        # Extract dynamic values
        morgan_mean = means.get('morgan', default_r2['Morgan FP (RF+XGB+SVM)'])
        morgan_std = stds.get('morgan', default_std['Morgan FP (RF+XGB+SVM)'])

        # Get the best RDKit score from all modes that contain 'rdkit' (same mode used for its std)
        rdkit_modes = [k for k in means if 'rdkit' in k]
        if rdkit_modes:
            best_rdkit_mode = max(rdkit_modes, key=lambda k: means[k])
            rdkit_mean = means[best_rdkit_mode]
            rdkit_std = stds[best_rdkit_mode]
        else:
            rdkit_mean = default_r2['RDKit 2D+FP (best)']
            rdkit_std = default_std['RDKit 2D+FP (best)']

        rdkit_label = {'rdkit2d': 'RDKit 2D', 'rdkit2d_fp': 'RDKit 2D+FP',
                       'rdkit2d3d_fp': 'RDKit 2D+3D+FP'}.get(best_rdkit_mode, 'RDKit') if rdkit_modes else 'RDKit'
        r2 = {
            'Morgan FP (Best Ensemble)': round(morgan_mean, 4),
            f'{rdkit_label} (Best Ensemble)': round(rdkit_mean, 4),
        }
        std = {
            'Morgan FP (Best Ensemble)': round(morgan_std, 4),
            f'{rdkit_label} (Best Ensemble)': round(rdkit_std, 4),
        }
        return r2, std
    except Exception:
        return default_r2, default_std


# Generate the dictionaries dynamically reading the 01_nested_cv_final_results.csv
CLASSICAL_R2, CLASSICAL_R2_STD = get_classical_r2(FINAL_RESULTS_FILE)

# Ensure all output directories exist
for d in [RESULTS_DIR, LATEX_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

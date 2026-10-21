"""
Faithful re-implementation of the deep-network protocol of Karimova & Senanayake (2025), shared by steps 15-17.

Everything below mirrors the original notebooks (https://github.com/karimova/pIC50_paper, commit KARIMOVA_COMMIT):
preprocessing chain and its order (nb3/nb5/nb7/nb8), network, optimizer and callbacks, the importance-group binning
(nb7/nb8) and the Gaussian augmentation (nb8). What differs from the original is stated where it happens: every run is
seeded and TensorFlow ops are deterministic, so that results are reproducible, which the original was not.
This module runs in the separate TensorFlow environment (MLGNN_TF_PYTHON).
"""
import os
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KARIMOVA_DIR = os.environ.get("MLGNN_KARIMOVA_DIR") or os.path.join(os.path.dirname(ROOT), "karimova_pIC50_paper")
KARIMOVA_COMMIT = "70e944619d0eb964bc8b4ef831bb27f12160d33c"
KARIMOVA_TABLE = os.path.join(KARIMOVA_DIR, "3-DNN_Baseline", "V2-merged_df_noOutliers.csv")
KARIMOVA_IMPORTANCE = os.path.join(KARIMOVA_DIR, "5-Feature_Importance", "V2-feature_importance_model_PermImp.csv")
NON_FEATURE_COLUMNS = ['Standard Value', 'pIC50 Value', 'Smiles', 'QT-pIC50 Value', 'cid']

# Original hyperparameters
TEST_SIZE = 0.15
SPLIT_SEED = 42
N_RUNS = 5                              # runs averaged into one ensemble prediction
DROPOUT_BASE = (0.4, 0.3, 0.2)          # nb3, nb5, nb7
DROPOUT_AUG = (0.3, 0.2, 0.1)           # nb8
PATIENCE_BASE = 30
PATIENCE_AUG = 60
LEARNING_RATE = 0.0007
EPOCHS = 200
BATCH_SIZE = 32
NOISE_LEVELS = (0.01, 0.001)
PI_REPEATS = 5
# nb7/nb8 bins with right=False and reversed labels: Group_1 = [1e-3, 1e-2), ..., Group_4 = [1e-6, 1e-5).
# Importances >= 1e-2 fall outside every bin and are therefore dropped by the original code; kept as is.
GROUP_BINS = {'G1': (1e-3, 1e-2), 'G2': (1e-4, 1e-3), 'G3': (1e-5, 1e-4), 'G4': (1e-6, 1e-5)}

N_THREADS = int(os.environ.get("MLGNN_TF_THREADS", "4"))
tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()


def karimova_commit():
    """Commit of the local clone of the original repository, or None if it cannot be read."""
    try:
        out = subprocess.run(["git", "-C", KARIMOVA_DIR, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_clone():
    if not os.path.exists(KARIMOVA_TABLE):
        raise SystemExit(f"[ERROR] Original descriptor table not found: {KARIMOVA_TABLE}. Clone "
                         f"https://github.com/karimova/pIC50_paper (commit {KARIMOVA_COMMIT}) and set MLGNN_KARIMOVA_DIR.")
    commit = karimova_commit()
    if commit != KARIMOVA_COMMIT:
        print(f"[WARNING] Clone at {KARIMOVA_DIR} is at commit {commit}, expected {KARIMOVA_COMMIT}.")
    return commit


def load_table():
    """The 873-compound table of the original study (PaDEL 2D + 3D descriptors and PubChem fingerprint), in its row order."""
    df = pd.read_csv(KARIMOVA_TABLE).reset_index(drop=True)
    features = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, features


class Preprocessor:
    """nb3 chain, each step fitted on the training rows only: median imputation, constant (tol 0.999) and duplicate
    feature removal, Gaussian winsorizing of both tails, Yeo-Johnson, standardization."""

    def fit(self, X):
        self.steps = [MeanMedianImputer(imputation_method='median'), DropConstantFeatures(tol=0.999),
                      DropDuplicateFeatures(), Winsorizer(capping_method='gaussian', tail='both'), YeoJohnsonTransformer()]
        for step in self.steps:
            X = step.fit_transform(X)
        self.columns = list(X.columns)
        self.scaler = StandardScaler().fit(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_dnn(n_features, dropouts):
    """128-64-32-16 ReLU network with batch normalization after every hidden layer and the given dropout after the first three."""
    layers = [Input(shape=(n_features,))]
    for units, rate in zip((128, 64, 32), dropouts):
        layers += [Dense(units, activation='relu'), BatchNormalization(), Dropout(rate)]
    layers += [Dense(16, activation='relu'), BatchNormalization(), Dense(1, activation='linear')]
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error', metrics=['mae'])
    return model


def train_dnn(X_train, y_train, X_monitor, y_monitor, seed, dropouts=DROPOUT_BASE, patience=PATIENCE_BASE):
    """One training run. (X_monitor, y_monitor) drives ReduceLROnPlateau and EarlyStopping with best-weight restoration;
    in the original protocol this is the test set itself."""
    tf.keras.utils.set_random_seed(int(seed))
    model = build_dnn(X_train.shape[1], dropouts)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.000001),
                 EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32),
              validation_data=(np.asarray(X_monitor, dtype=np.float32), np.asarray(y_monitor, dtype=np.float32)),
              batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=0, callbacks=callbacks)
    return model


def predict(model, X):
    return model(np.asarray(X, dtype=np.float32), training=False).numpy().ravel()


def ensemble_predict(X_train, y_train, X_monitor, y_monitor, X_eval, seed, dropouts=DROPOUT_BASE, patience=PATIENCE_BASE):
    """Average of N_RUNS runs (the original 'Average' row); run r of ensemble `seed` uses seed * 100 + r."""
    preds = [predict(train_dnn(X_train, y_train, X_monitor, y_monitor, seed * 100 + r, dropouts, patience), X_eval)
             for r in range(N_RUNS)]
    return np.mean(preds, axis=0)


class _KerasRegressor:
    """Minimal estimator interface so that sklearn's permutation_importance can score an already-trained Keras model."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return predict(self.model, X)


def permutation_importance_mean(model, X_eval, y_eval, seed=SPLIT_SEED, n_repeats=PI_REPEATS):
    """Mean increase in MSE when each feature of X_eval is shuffled (nb5: negative MSE scoring, 5 repeats)."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(_KerasRegressor(model), np.asarray(X_eval, dtype=np.float32), np.asarray(y_eval),
                                    scoring='neg_mean_squared_error', n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return result.importances_mean


def select_groups(importances, feature_names, groups=('G1', 'G2', 'G3')):
    """Feature names whose importance falls in the bins of the requested groups (original binning, see GROUP_BINS)."""
    imp = pd.Series(np.asarray(importances, dtype=float), index=list(feature_names))
    keep = np.zeros(len(imp), dtype=bool)
    for g in groups:
        lo, hi = GROUP_BINS[g]
        keep |= (imp.values >= lo) & (imp.values < hi)
    return list(imp.index[keep])


def published_importance():
    """The importance table published with the original study (output of its nb5)."""
    df = pd.read_csv(KARIMOVA_IMPORTANCE)
    return df['Importance'].to_numpy(), df['Feature'].tolist()


def augment(X, y, rng, noise_levels=NOISE_LEVELS, include_original=False, groups=None):
    """One noisy copy of every row per noise level. The original (nb8) trains on the copies only (include_original=False).
    Returns (X_aug, y_aug, groups_aug), where groups_aug is the source-row index (or `groups`) of every output row."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    src = np.arange(len(y)) if groups is None else np.asarray(groups)
    X_parts, y_parts, g_parts = ([X], [y], [src]) if include_original else ([], [], [])
    for sigma in noise_levels:
        X_parts.append(X + rng.normal(0, sigma, X.shape))
        y_parts.append(y)
        g_parts.append(src)
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(g_parts)


def split_indices(n, seed=SPLIT_SEED):
    """The original 15% split (train_test_split with seed 42), as positional indices."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)


def nb8_protocol(Xtr, ytr, Xte, yte, seed):
    """Augmentation exactly as in nb8. The already preprocessed train and test rows are concatenated in that order and
    split again with the same seed (so the new test part mixes rows of both, and the preprocessing was fitted on some of
    them); only the noisy copies are used for training; the new test part is both the monitoring and the scoring set.
    Returns (y_val, prediction, positions of the new test rows in the concatenated order)."""
    X_f = np.vstack([np.asarray(Xtr), np.asarray(Xte)])
    y_f = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    dev_pos, val_pos = split_indices(len(y_f))
    X_aug, y_aug, _ = augment(X_f[dev_pos], y_f[dev_pos], np.random.default_rng(seed))
    pred = ensemble_predict(X_aug, y_aug, X_f[val_pos], y_f[val_pos], X_f[val_pos], seed, DROPOUT_AUG, PATIENCE_AUG)
    return y_f[val_pos], pred, val_pos


class _LearningRateLog(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.rates = []

    def on_epoch_begin(self, epoch, logs=None):
        self.rates.append(float(np.asarray(self.model.optimizer.learning_rate)))


def train_internal(X, y, seed, dropouts=DROPOUT_BASE, patience=PATIENCE_BASE, augment_rng=None):
    """One run that never looks at the test set, in two stages.
    1) The network is trained on 85% of the training rows while the original callbacks (ReduceLROnPlateau,
       EarlyStopping) monitor the other 15%; this fixes the number of epochs (the best validation epoch) and records the
       learning-rate schedule. With augment_rng, the noisy copies are made from the 85% only, so no copy of a
       validation molecule is trained on.
    2) The network is trained again from the same initialization on all training rows (augmented the same way) for that
       number of epochs, replaying the recorded learning rates. The final model therefore uses every training molecule,
       as the original protocol did, so that a difference with the original is due to the monitoring set and not to
       training on fewer molecules."""
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    fit_pos, val_pos = split_indices(len(y))
    X_fit, y_fit = X[fit_pos], y[fit_pos]
    X_all, y_all = X, y
    if augment_rng is not None:
        X_fit, y_fit, _ = augment(X_fit, y_fit, augment_rng)
        X_all, y_all, _ = augment(X, y, augment_rng)

    tf.keras.utils.set_random_seed(int(seed))
    model = build_dnn(X.shape[1], dropouts)
    lr_log = _LearningRateLog()
    history = model.fit(X_fit, y_fit, validation_data=(X[val_pos], y[val_pos]), batch_size=BATCH_SIZE, epochs=EPOCHS,
                        shuffle=True, verbose=0,
                        callbacks=[lr_log, ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.000001),
                                   EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])
    n_epochs = int(np.argmin(history.history['val_loss'])) + 1
    rates = lr_log.rates[:n_epochs]

    tf.keras.utils.set_random_seed(int(seed))
    final = build_dnn(X.shape[1], dropouts)
    final.fit(X_all, y_all, batch_size=BATCH_SIZE, epochs=n_epochs, shuffle=True, verbose=0,
              callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: rates[epoch])])
    return final


def ensemble_predict_internal(X, y, X_eval, seed, augment=False):
    """N_RUNS runs with internal validation (see train_internal), averaged; augmented runs use the nb8 network settings."""
    dropouts, patience = (DROPOUT_AUG, PATIENCE_AUG) if augment else (DROPOUT_BASE, PATIENCE_BASE)
    preds = []
    for r in range(N_RUNS):
        rng = np.random.default_rng(seed * 100 + r) if augment else None
        preds.append(predict(train_internal(X, y, seed * 100 + r, dropouts, patience, rng), X_eval))
    return np.mean(preds, axis=0)


def oof_importance(X_raw, y, seed, n_folds=5):
    """Out-of-fold permutation importance inside the given rows: in each fold the preprocessing is fitted and a network is
    trained (internal validation) on the other folds, and the features are permuted on the held-out fold. Importances are
    averaged per raw feature over the folds; a feature removed by the preprocessing of a fold counts as 0 there.
    Returns (importances, feature names) aligned with X_raw.columns."""
    from sklearn.model_selection import KFold
    y = np.asarray(y)
    total = pd.Series(0.0, index=list(X_raw.columns))
    for k, (tr, va) in enumerate(KFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED).split(X_raw)):
        pre = Preprocessor().fit(X_raw.iloc[tr])
        X_tr, X_va = pre.transform(X_raw.iloc[tr]), pre.transform(X_raw.iloc[va])
        model = train_internal(X_tr, y[tr], seed * 100 + 60 + k)
        imp = pd.Series(permutation_importance_mean(model, X_va, y[va]), index=pre.columns)
        total = total.add(imp.reindex(total.index, fill_value=0.0))
    return (total / n_folds).to_numpy(), list(total.index)


def scores(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    return float(r2_score(y_true, y_pred)), float(mean_absolute_error(y_true, y_pred))

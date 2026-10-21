"""
Step 19: stronger neural baselines on the 25 shared outer folds.

The D-MPNN of steps 09/12 is a single network trained from scratch. Four variants test whether a stronger neural model
closes the gap to the classical ensembles:
  dmpnn_ens5   the default D-MPNN averaged over 5 seeds (removes the seed-to-seed variance of one network)
  dmpnn_rdkit  the default D-MPNN with the 217 RDKit 2D descriptors appended to the learned molecular vector
  chemeleon    message passing initialized from the CheMeleon foundation model (pretrained on PubChem descriptors), then
               fine-tuned with the same 100-epoch schedule
  mlp_ecfp     a multilayer perceptron on the 2048-bit Morgan fingerprints (a neural model without learned representation)
Every variant is fitted on the outer training part only, with no tuning, and results are checkpointed per fold.
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import *
from logger_utils import setup_logger, explain
from splits import outer_splits

CHECKPOINT_19 = os.path.join(RESULTS_DIR, "19_gnn_strong_checkpoint.csv")
OOF_19 = os.path.join(RESULTS_DIR, "19_gnn_strong_oof_predictions.csv")
TESTS_19 = os.path.join(RESULTS_DIR, "19_gnn_strong_vs_classical_tests.csv")
LATEX_GNN_STRONG = os.path.join(LATEX_DIR, "gnn_strong_variables.tex")
CHEMELEON_FILE = os.path.join(os.path.expanduser("~"), ".chemprop", "chemeleon_mp.pt")

VARIANTS = [v for v in os.environ.get("MLGNN_K19_VARIANTS", "mlp_ecfp,dmpnn_ens5,dmpnn_rdkit,chemeleon").split(",") if v]
VARIANT_LABEL = {"dmpnn_ens5": "DmpnnEns", "dmpnn_rdkit": "DmpnnRdkit", "chemeleon": "Chemeleon", "mlp_ecfp": "MlpEcfp"}
# Relative cost of one fold, used to start the longest tasks first
VARIANT_COST = {"chemeleon": 10, "dmpnn_ens5": 5, "dmpnn_rdkit": 1, "mlp_ecfp": 0.1}
N_ENSEMBLE = 5
EPOCHS = int(os.environ.get("MLGNN_GNN_EPOCHS", "100"))
THREADS_PER_WORKER = int(os.environ.get("MLGNN_K19_THREADS", "4"))


def morgan(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32) for s in smiles])


def rdkit_2d(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        vals = []
        for _, fn in Descriptors._descList:
            try:
                vals.append(fn(m))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    X = np.array(rows, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


def train_predict_mpnn(smiles_tr, y_tr, smiles_te, seed, threads, x_d_tr=None, x_d_te=None, foundation=None):
    """One Chemprop MPNN with the default settings of step 09 (hidden size 300, depth 3, mean aggregation, batch norm,
    100 epochs, standardized targets). x_d_*: extra molecule descriptors, already imputed and scaled with the training
    statistics, concatenated to the learned vector before the feed-forward network. foundation: path of a CheMeleon
    checkpoint whose message passing initializes the network (it then requires the v2 atom featurizer)."""
    import torch
    from lightning import pytorch as pl
    from chemprop import data as cpdata, featurizers, models, nn as cpnn
    torch.set_num_threads(threads)
    pl.seed_everything(seed, workers=True)
    if foundation:
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
        ckpt = torch.load(foundation, weights_only=True)
        mp_block = cpnn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp_block.load_state_dict(ckpt["state_dict"])
    else:
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        mp_block = cpnn.BondMessagePassing()

    def dataset(smiles, ys, x_d):
        points = [cpdata.MoleculeDatapoint.from_smi(s, [t], x_d=None if x_d is None else x_d[i])
                  for i, (s, t) in enumerate(zip(smiles, ys))]
        return cpdata.MoleculeDataset(points, featurizer)

    train_dset = dataset(smiles_tr, y_tr, x_d_tr)
    test_dset = dataset(smiles_te, np.zeros(len(smiles_te)), x_d_te)
    scaler = train_dset.normalize_targets()
    test_dset.normalize_targets(scaler)
    n_extra = 0 if x_d_tr is None else x_d_tr.shape[1]
    ffn = cpnn.RegressionFFN(input_dim=mp_block.output_dim + n_extra)
    mpnn = models.MPNN(mp_block, cpnn.MeanAggregation(), ffn, batch_norm=True, metrics=[cpnn.metrics.RMSE()])
    trainer = pl.Trainer(max_epochs=EPOCHS, enable_progress_bar=False, enable_model_summary=False,
                         enable_checkpointing=False, logger=False, accelerator="cpu")
    trainer.fit(mpnn, cpdata.build_dataloader(train_dset, shuffle=True, num_workers=0))
    preds = trainer.predict(mpnn, cpdata.build_dataloader(test_dset, shuffle=False, num_workers=0))
    return scaler.inverse_transform(torch.cat(preds).numpy().reshape(-1, 1)).flatten()


def scaled_descriptors(X_tr, X_te):
    """Median imputation and standardization fitted on the training part; constant columns are dropped."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median").fit(X_tr)
    A, B = imp.transform(X_tr), imp.transform(X_te)
    keep = A.std(axis=0) > 0
    sc = StandardScaler().fit(A[:, keep])
    return sc.transform(A[:, keep]).astype(np.float32), np.clip(sc.transform(B[:, keep]), -10, 10).astype(np.float32)


def run_task(task):
    from sklearn.metrics import mean_absolute_error, r2_score
    variant, fold, tr, te, smiles, y, threads = task
    t0 = time.time()
    seed = RANDOM_STATE + fold
    s_tr, s_te = [smiles[i] for i in tr], [smiles[i] for i in te]
    if variant == "dmpnn_ens5":
        pred = np.mean([train_predict_mpnn(s_tr, y[tr], s_te, seed + 1000 * k, threads) for k in range(N_ENSEMBLE)], axis=0)
    elif variant == "dmpnn_rdkit":
        X = rdkit_2d(smiles)
        x_tr, x_te = scaled_descriptors(X[tr], X[te])
        pred = train_predict_mpnn(s_tr, y[tr], s_te, seed, threads, x_tr, x_te)
    elif variant == "chemeleon":
        pred = train_predict_mpnn(s_tr, y[tr], s_te, seed, threads, foundation=CHEMELEON_FILE)
    elif variant == "mlp_ecfp":
        from sklearn.neural_network import MLPRegressor
        X = morgan(smiles)
        mlp = MLPRegressor(hidden_layer_sizes=(512, 128), alpha=1e-3, early_stopping=True, validation_fraction=0.1,
                           max_iter=500, random_state=seed)
        pred = mlp.fit(X[tr], y[tr]).predict(X[te])
    else:
        raise ValueError(f"Unknown variant '{variant}'")
    r2, mae = float(r2_score(y[te], pred)), float(mean_absolute_error(y[te], pred))
    print(f"  {variant:<12} fold {fold:>2} R2={r2:.4f} MAE={mae:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return (dict(variant=variant, Fold=fold, R2=r2, MAE=mae, N_train=len(tr), N_test=len(te), Time_s=time.time() - t0),
            pd.DataFrame(dict(variant=variant, Fold=fold, row=te, y_true=y[te], y_pred=pred)))


def main():
    setup_logger(os.path.join(LOGS_DIR, "19_gnn_strong_baselines.log"))
    print("=" * 105)
    print("19_gnn_strong_baselines.py — Stronger neural baselines on the shared outer folds")
    print("=" * 105)
    explain("What this step does", """
        The graph network of steps 09 and 12 was trained from scratch, one network per fold. A fair objection is that a
        stronger neural model might have matched the classical ensembles. This step tries four: an average of five graph
        networks, a graph network that also receives the RDKit descriptors, a graph network that starts from CheMeleon (a
        model pretrained on about a million PubChem molecules), and a plain neural network on the Morgan fingerprints. All
        are scored on the same 25 folds as every other model and compared with the same corrected tests.
    """)
    if "chemeleon" in VARIANTS and not os.path.exists(CHEMELEON_FILE):
        print(f"[ERROR] CheMeleon weights not found at {CHEMELEON_FILE} "
              "(download https://zenodo.org/records/15460715/files/chemeleon_mp.pt).")
        sys.exit(1)

    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    smiles, y = df['Smiles'].tolist(), df['pIC50 Value'].to_numpy()
    folds = list(outer_splits(smiles, CFG['OUTER_N_SPLITS'], CFG['OUTER_N_REPEATS'], RANDOM_STATE, SPLIT_KIND))

    done = pd.read_csv(CHECKPOINT_19) if os.path.exists(CHECKPOINT_19) else pd.DataFrame(columns=["variant", "Fold"])
    done_keys = set(zip(done["variant"], done["Fold"]))
    tasks = [(v, f, tr, te, smiles, y, THREADS_PER_WORKER)
             for v in sorted(VARIANTS, key=lambda v: -VARIANT_COST.get(v, 1))
             for f, (tr, te) in enumerate(folds) if (v, f) not in done_keys]
    print(f"  {len(tasks)} variant x fold tasks to run ({len(done_keys)} already in the checkpoint)")

    if tasks:
        n_workers = max(1, min(len(tasks), (os.cpu_count() or 1) // THREADS_PER_WORKER))
        print(f"  {n_workers} workers x {THREADS_PER_WORKER} threads")
        # ProcessPoolExecutor, not multiprocessing.Pool: if a worker dies, Pool replaces it and waits forever for the lost
        # task, whereas the executor raises BrokenProcessPool. Finished folds are already in the checkpoint, so a re-run
        # only repeats the missing ones.
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as pool:
            for fut in as_completed([pool.submit(run_task, t) for t in tasks]):
                row, pred = fut.result()
                pd.DataFrame([row]).to_csv(CHECKPOINT_19, mode="a", header=not os.path.exists(CHECKPOINT_19), index=False)
                pred.to_csv(OOF_19, mode="a", header=not os.path.exists(OOF_19), index=False)

    from nb_stats import compare_to_classical, nb_paired, EQUIVALENCE_MARGIN_R2, EQUIVALENCE_MARGIN_MAE
    scores = pd.read_csv(CHECKPOINT_19)
    checkpoint = pd.read_csv(CHECKPOINT_FILE)
    gnn_default = pd.read_csv(GNN_FOLD_SCORES_CSV).sort_values("Fold").reset_index(drop=True)
    values, tables = {}, []
    for variant in VARIANTS:
        s = scores[scores["variant"] == variant].sort_values("Fold").reset_index(drop=True)
        if len(s) != len(folds):
            print(f"  [WARNING] {variant}: {len(s)}/{len(folds)} folds available; not summarized.")
            continue
        label = VARIANT_LABEL[variant]
        s[["Fold", "R2", "MAE", "N_train", "N_test"]].to_csv(os.path.join(RESULTS_DIR, f"19_{variant}_fold_scores.csv"),
                                                             index=False)
        values[f"{label}RTwoMean"], values[f"{label}RTwoStd"] = f"{s['R2'].mean():.4f}", f"{s['R2'].std(ddof=1):.4f}"
        values[f"{label}Mae"], values[f"{label}FoldMinutes"] = f"{s['MAE'].mean():.4f}", f"{s['Time_s'].mean() / 60:.1f}"
        table, macros = compare_to_classical(s, checkpoint, f"{label}Vs")
        values.update(macros)
        tables.append(table.assign(model=label))
        for metric, col, margin in (("RTwo", "R2", EQUIVALENCE_MARGIN_R2), ("Mae", "MAE", EQUIVALENCE_MARGIN_MAE)):
            res = nb_paired(s[col].values - gnn_default[col].values, s["N_train"].mean(), s["N_test"].mean(), margin)
            values[f"{label}VsGnnDefaultDiff{metric}"] = f"{res['mean_diff']:+.4f}"
            values[f"{label}VsGnnDefaultP{metric}"] = f"{res['p']:.3f}"
        print(f"  {variant:<12} R2 = {s['R2'].mean():.4f} ± {s['R2'].std(ddof=1):.4f} | MAE = {s['MAE'].mean():.4f} | "
              f"vs default D-MPNN dR2 = {values[f'{label}VsGnnDefaultDiffRTwo']} (p = {values[f'{label}VsGnnDefaultPRTwo']})")
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(TESTS_19, index=False)
    os.makedirs(os.path.dirname(LATEX_GNN_STRONG), exist_ok=True)
    with open(LATEX_GNN_STRONG, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 19_gnn_strong_baselines.py\n")
        for name, value in values.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    print(f"[EXPORT] {len(values)} LaTeX variables saved to {LATEX_GNN_STRONG}")


if __name__ == "__main__":
    main()

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import *
from logger_utils import setup_logger, explain
from qsartools.splits import holdout_split, outer_splits

FOLDS_CSV = os.path.join(RESULTS_DIR, "14_outer_folds.csv")


def fold_rows(smiles, cids, kind):
    """Long-format membership of every compound in every outer fold of the given split kind."""
    n_splits, n_repeats = CFG['OUTER_N_SPLITS'], CFG['OUTER_N_REPEATS']
    rows = []
    folds = outer_splits(smiles, n_splits, n_repeats, RANDOM_STATE, kind)
    for fold, (train_idx, test_idx) in enumerate(folds):
        for role, idx in (("train", train_idx), ("test", test_idx)):
            rows.append(pd.DataFrame({"split_kind": kind, "fold": fold, "repeat": fold // n_splits,
                                      "role": role, "cid": cids[idx]}))
    return rows


def main():
    setup_logger(os.path.join(LOGS_DIR, "14_export_folds.log"))
    print("=" * 105)
    print("14_export_folds.py — Export of the shared outer folds for the TensorFlow steps")
    print("=" * 105)
    explain("What this step does", """
        The audit of the original deep network (steps 15-17) runs in a separate Python environment with TensorFlow. So that
        the network is scored on exactly the same molecules as the classical ensembles and the D-MPNN, this step writes the
        25 outer folds (random and scaffold-grouped) and the 15% hold-out split of step 04 to a table keyed by compound ID.
        Nothing is trained here.
    """)

    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip').dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    if df['cid'].duplicated().any():
        print("[ERROR] Duplicate compound IDs in the training set; fold membership by cid would be ambiguous.")
        sys.exit(1)
    smiles = df['Smiles'].tolist()
    cids = df['cid'].to_numpy()

    parts = fold_rows(smiles, cids, "random") + fold_rows(smiles, cids, "scaffold")

    # Same call as step 04, which splits arrays of this length with the same seed
    dev_idx, test_idx = holdout_split(len(df), test_size=TEST_SIZE, seed=RANDOM_STATE)
    for role, idx in (("train", dev_idx), ("test", test_idx)):
        parts.append(pd.DataFrame({"split_kind": "holdout", "fold": 0, "repeat": 0, "role": role, "cid": cids[idx]}))

    folds = pd.concat(parts, ignore_index=True)

    # Every compound must be tested exactly once per repeat of each outer scheme
    for kind, g in folds.query("role == 'test' and split_kind != 'holdout'").groupby("split_kind"):
        per_repeat = g.groupby(["repeat", "cid"]).size()
        n_rep = g["repeat"].nunique()
        if per_repeat.max() != 1 or len(per_repeat) != len(df) * n_rep:
            print(f"[ERROR] Split '{kind}' does not test every compound exactly once per repeat.")
            sys.exit(1)
        sizes = g.groupby("fold").size()
        print(f"  {kind:<9} folds={sizes.size:>2} | test size min={sizes.min()} max={sizes.max()} | repeats={n_rep}")
    if len(set(dev_idx) | set(test_idx)) != len(df) or set(dev_idx) & set(test_idx):
        print("[ERROR] The hold-out split is not a partition of the training set.")
        sys.exit(1)
    print(f"  holdout   dev={len(dev_idx)} | test={len(test_idx)}")

    folds.to_csv(FOLDS_CSV, index=False)
    print(f"\n[+] {len(folds)} rows ({len(df)} compounds) written to: {FOLDS_CSV}")


if __name__ == "__main__":
    main()

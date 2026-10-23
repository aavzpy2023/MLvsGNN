"""Train/test partitions for molecular data: repeated random K-fold, repeated scaffold-grouped K-fold and a hold-out split.

All functions return positional indices into the input list, so the same partitions can be reused by every model that is
compared (paired evaluation).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import RepeatedKFold, train_test_split


def murcko_scaffolds(smiles):
    """Bemis-Murcko scaffold SMILES (without stereochemistry) of each molecule, as an array of strings.

    Acyclic molecules have the empty scaffold "" and therefore all share one group; a SMILES that cannot be parsed also
    gets "".
    """
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else "")
    return np.array(out)


def scaffold_repeated_kfold(smiles, n_splits=5, n_repeats=5, seed=42):
    """Repeated scaffold-grouped K-fold: every molecule sharing a scaffold falls in the same fold, so no scaffold is in both
    the training and the test part of a split. In each repeat the scaffolds are visited in a random order and each is
    assigned to the currently smallest fold. Yields (train_idx, test_idx), both sorted, repeat by repeat."""
    groups = murcko_scaffolds(smiles)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.RandomState(seed)
    for _ in range(n_repeats):
        folds = [[] for _ in range(n_splits)]
        for g in rng.permutation(unique):
            min(folds, key=len).extend(members[g])
        for k in range(n_splits):
            test = np.sort(np.array(folds[k]))
            train = np.sort(np.concatenate([np.array(folds[j]) for j in range(n_splits) if j != k]))
            yield train, test


def random_repeated_kfold(n_samples, n_splits=5, n_repeats=5, seed=42):
    """Repeated random K-fold (scikit-learn RepeatedKFold). Yields (train_idx, test_idx), repeat by repeat."""
    yield from RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.zeros(n_samples))


def outer_splits(smiles, n_splits=5, n_repeats=5, seed=42, kind="random"):
    """Repeated K-fold over the molecules `smiles`: kind="random" or kind="scaffold" (scaffold-grouped)."""
    if kind == "scaffold":
        yield from scaffold_repeated_kfold(smiles, n_splits, n_repeats, seed)
    elif kind == "random":
        yield from random_repeated_kfold(len(smiles), n_splits, n_repeats, seed)
    else:
        raise ValueError(f"Unknown split kind '{kind}' (expected 'random' or 'scaffold')")


def holdout_split(n_samples, test_size=0.15, seed=42):
    """One random hold-out partition (scikit-learn train_test_split on positions). Returns (train_idx, test_idx)."""
    return train_test_split(np.arange(n_samples), test_size=test_size, random_state=seed)

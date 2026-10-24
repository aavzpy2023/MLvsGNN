"""Train/test partitions for molecular data: repeated random K-fold, repeated scaffold-grouped K-fold and a hold-out split.

A molecule's scaffold is its Bemis-Murcko framework (Bemis & Murcko, 1996, J. Med. Chem. 39:2887-2893): its ring systems
and the linkers that join them, with the side chains removed. They analysed frameworks at two levels, as molecular
graphs only (without regard to atom type, hybridization and bond order) and with those properties considered; both are
offered. Bemis & Murcko describe frameworks, not splits: grouping molecules by framework for cross-validation, and the
way the groups are dealt out to the folds below, are choices of this package.

All functions return positional indices into the input list, so the same partitions can be reused by every model that is
compared (paired evaluation).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import RepeatedKFold, train_test_split


def murcko_scaffolds(smiles, generic=False):
    """Bemis-Murcko framework SMILES (without stereochemistry) of each molecule, as an array of strings.

    Acyclic molecules have the empty scaffold "" and therefore all share one group; a SMILES that cannot be parsed also
    gets "".

    Args:
        smiles: sequence of SMILES strings.
        generic: False keeps atom types, hybridization and bond orders (the second level of Bemis & Murcko, 2506
            frameworks in their 5120 drugs); True reduces the framework to its molecular graph, every atom becoming a
            carbon and every bond single (their first level, 1179 frameworks), which groups more molecules together.
    """
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append("")
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        out.append(Chem.MolToSmiles(scaffold, isomericSmiles=False))
    return np.array(out)


def scaffold_repeated_kfold(smiles, n_splits=5, n_repeats=5, seed=42):
    """Repeated scaffold-grouped K-fold: every molecule sharing a scaffold falls in the same fold, so no scaffold is in both
    the training and the test part of a split. In each repeat the scaffolds are visited in a random order and each is
    assigned to the currently smallest fold. Yields (train_idx, test_idx), both sorted, repeat by repeat.

    Args:
        smiles: sequence of SMILES strings; the indices refer to positions in it.
        n_splits: number of folds per repeat (K).
        n_repeats: number of repeats, each with a different random order of the scaffolds.
        seed: seed of the numpy RandomState that orders the scaffolds.
    """
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
    """Repeated random K-fold (scikit-learn RepeatedKFold). Yields (train_idx, test_idx), repeat by repeat.

    Args:
        n_samples: number of samples; the indices run from 0 to n_samples - 1.
        n_splits: number of folds per repeat (K).
        n_repeats: number of repeats, each with a different shuffle.
        seed: random_state of RepeatedKFold.
    """
    yield from RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.zeros(n_samples))


def outer_splits(smiles, n_splits=5, n_repeats=5, seed=42, kind="random"):
    """Repeated K-fold over the molecules `smiles`. Yields (train_idx, test_idx), repeat by repeat.

    Args:
        smiles: sequence of SMILES strings; the indices refer to positions in it.
        n_splits: number of folds per repeat (K).
        n_repeats: number of repeats.
        seed: random seed of the partition.
        kind: "random" (random_repeated_kfold) or "scaffold" (scaffold_repeated_kfold).
    """
    if kind == "scaffold":
        yield from scaffold_repeated_kfold(smiles, n_splits, n_repeats, seed)
    elif kind == "random":
        yield from random_repeated_kfold(len(smiles), n_splits, n_repeats, seed)
    else:
        raise ValueError(f"Unknown split kind '{kind}' (expected 'random' or 'scaffold')")


def holdout_split(n_samples, test_size=0.15, seed=42):
    """One random hold-out partition (scikit-learn train_test_split on positions). Returns (train_idx, test_idx).

    Args:
        n_samples: number of samples; the indices run from 0 to n_samples - 1.
        test_size: fraction of the samples (float in (0, 1)) or number of samples (int) placed in the test part.
        seed: random_state of train_test_split.
    """
    return train_test_split(np.arange(n_samples), test_size=test_size, random_state=seed)

"""Applicability domain of a model trained on a set of molecules: elemental composition filter and an Isolation Forest
fitted on the training representation (Liu, Ting & Zhou, 2008)."""
import numpy as np
from rdkit import Chem
from sklearn.ensemble import IsolationForest

# H, C, N, O, F, P, S, Cl, Br, I
ORGANIC_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


def contains_only_elements(molecule, allowed_atomic_numbers=ORGANIC_ELEMENTS):
    """True if every atom of the molecule (SMILES or RDKit molecule) has an atomic number in the allowed set; False for a
    molecule that cannot be parsed."""
    mol = molecule if isinstance(molecule, Chem.Mol) else Chem.MolFromSmiles(str(molecule))
    if mol is None:
        return False
    return all(atom.GetAtomicNum() in allowed_atomic_numbers for atom in mol.GetAtoms())


def fit_isolation_forest(X_train, contamination=0.05, seed=42, n_jobs=1):
    """Isolation Forest fitted on the training representation. `contamination` is the fraction of training samples placed
    outside the domain by construction, which sets the decision threshold."""
    return IsolationForest(contamination=contamination, random_state=seed, n_jobs=n_jobs).fit(X_train)


def in_domain(forest, X):
    """Boolean array: True for the samples of X that the fitted forest labels as inliers."""
    return forest.predict(X) == 1

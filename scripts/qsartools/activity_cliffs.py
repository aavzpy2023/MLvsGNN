"""Activity cliffs: pairs of highly similar molecules with a large difference in potency.

The fingerprint criterion of MoleculeACE (van Tilborg, Alenicheva & Grisoni, 2022, J. Chem. Inf. Model. 62:5938-5951) is
used: Tanimoto similarity of binary fingerprints (ECFP4 in MoleculeACE) of at least 0.9 and a potency difference of at
least one log unit (tenfold). MoleculeACE also accepts pairs that pass a scaffold or SMILES similarity instead; those
criteria are not implemented here.
"""
import numpy as np


def tanimoto_matrix(fingerprints):
    """Pairwise Tanimoto similarity |A and B| / |A or B| of the rows of a binary (0/1) matrix, as an (n, n) array.
    Two empty fingerprints have similarity 0."""
    X = np.asarray(fingerprints, dtype=np.float64)
    intersection = X @ X.T
    counts = X.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def activity_cliff_pairs(fingerprints, potency, similarity_threshold=0.9, potency_difference=1.0):
    """Every pair (i, j), i < j, with Tanimoto similarity >= similarity_threshold and |potency_i - potency_j| >=
    potency_difference. Returns an (m, 4) array of rows (i, j, similarity, absolute potency difference)."""
    potency = np.asarray(potency, dtype=float)
    sim = tanimoto_matrix(fingerprints)
    delta = np.abs(potency[:, None] - potency[None, :])
    i, j = np.where(np.triu((sim >= similarity_threshold) & (delta >= potency_difference), k=1))
    return np.column_stack([i, j, sim[i, j], delta[i, j]]) if len(i) else np.empty((0, 4))


def cliff_compounds(n_compounds, pairs):
    """Boolean mask of the compounds that take part in at least one cliff pair."""
    mask = np.zeros(n_compounds, dtype=bool)
    if len(pairs):
        mask[np.asarray(pairs)[:, :2].astype(int).ravel()] = True
    return mask

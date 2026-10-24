"""Activity cliffs: pairs of highly similar molecules with a large difference in potency.

The definition is that of MoleculeACE (van Tilborg, Alenicheva & Grisoni, 2022, J. Chem. Inf. Model. 62:5938-5951): a
pair is an activity cliff if its similarity is at least 0.9 according to at least one of three metrics (Tanimoto
similarity of extended-connectivity fingerprints, Tanimoto similarity of the fingerprints of the molecular scaffolds, and
the scaled Levenshtein distance between SMILES strings) and the activities differ by more than tenfold.

Only the Tanimoto similarity of binary fingerprints is computed here; the similarity matrices of the other two metrics
can be passed in `extra_similarities`. With the fingerprint metric alone, pairs that MoleculeACE would find through the
scaffold or the SMILES metric are missed, so the count of cliffs is a lower bound.
"""
import numpy as np


def tanimoto_matrix(fingerprints):
    """Pairwise Tanimoto similarity |A and B| / |A or B| of the rows of a binary (0/1) matrix, as an (n, n) array.
    Two empty fingerprints have similarity 0.

    Args:
        fingerprints: (n, n_bits) array-like of 0/1 values, one fingerprint per row.
    """
    X = np.asarray(fingerprints, dtype=np.float64)
    intersection = X @ X.T
    counts = X.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def activity_cliff_pairs(fingerprints, potency, similarity_threshold=0.9, potency_difference=1.0, extra_similarities=()):
    """Every pair (i, j), i < j, whose similarity is >= similarity_threshold by the Tanimoto metric of the fingerprints or
    by any of the extra metrics, and whose potencies differ by MORE than potency_difference (MoleculeACE: "more than
    tenfold"). Returns an (m, 4) array of rows (i, j, similarity, absolute potency difference), the similarity being the
    largest over the metrics.

    Args:
        fingerprints: (n, n_bits) array-like of 0/1 values, one fingerprint per row.
        potency: n potencies on a log scale (e.g. pIC50), in the same order as the fingerprints.
        similarity_threshold: minimum similarity of a pair, in each metric.
        potency_difference: potency difference in log units that must be exceeded (1.0 = tenfold); a difference equal to
            it, up to rounding (1e-9), does not count.
        extra_similarities: further (n, n) similarity matrices in [0, 1] (e.g. of the scaffolds, or of the SMILES strings
            as 1 - normalized Levenshtein distance); a pair passes if any of them, or the fingerprints, reach the
            threshold.
    """
    potency = np.asarray(potency, dtype=float)
    sim = tanimoto_matrix(fingerprints)
    for extra in extra_similarities:
        sim = np.maximum(sim, np.asarray(extra, dtype=float))
    delta = np.abs(potency[:, None] - potency[None, :])
    i, j = np.where(np.triu((sim >= similarity_threshold) & (delta > potency_difference + 1e-9), k=1))
    return np.column_stack([i, j, sim[i, j], delta[i, j]]) if len(i) else np.empty((0, 4))


def cliff_compounds(n_compounds, pairs):
    """Boolean mask of the compounds that take part in at least one cliff pair.

    Args:
        n_compounds: total number of compounds (length of the mask).
        pairs: output of activity_cliff_pairs; only its first two columns (i, j) are used.
    """
    mask = np.zeros(n_compounds, dtype=bool)
    if len(pairs):
        mask[np.asarray(pairs)[:, :2].astype(int).ravel()] = True
    return mask

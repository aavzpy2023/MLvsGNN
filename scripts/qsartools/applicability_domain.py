"""Applicability domain of a model trained on a set of molecules: similarity to the training set, an elemental
composition filter and an Isolation Forest fitted on the training representation.

Similarity to the training set (Sheridan, Feuston, Maiorov & Kearsley, 2004, J. Chem. Inf. Comput. Sci. 44:1912-1928):
the prediction error of a QSAR model grows as a molecule becomes less similar to the training set, measured by the mean
similarity to its k most similar training molecules (SIMILARITYNEAREST k, with k = 1, 3, 5) or by the number of training
neighbours above a similarity cutoff (NEIGHBORCOUNT). Sheridan et al. use Dice similarity of atom pairs and give no
in/out threshold, the cutoff being the user's; here similarity is the Tanimoto coefficient of binary fingerprints, and
the threshold is derived from the training set as described by Sahigara, Mansouri, Ballabio, Mauri, Consonni &
Todeschini (2012, Molecules 17:4791-4810), who state that for distance-based approaches "no strict rules were evident
from the literature about defining thresholds" and compare several strategies (the distance mean, twice the mean, the
95th percentile and mean + z standard deviations, z = 0.5 by default), computed on the distances of the training
compounds to their 5 nearest neighbours.

The Isolation Forest is that of Liu, Ting & Zhou (2008, Proc. IEEE ICDM, 413-422): t = 100 trees, each grown on a
sub-sample of psi = min(256, n) instances, and the anomaly score s(x, n) = 2^(-E(h(x)) / c(n)) of their Eq. 2, where
s near 1 marks an anomaly, s well below 0.5 a normal instance, and 0.5 the boundary. Using it as the applicability
domain of a QSAR model, and the elemental filter, are choices of this package, not part of that paper.
"""
import numpy as np
from rdkit import Chem
from sklearn.ensemble import IsolationForest

# H, C, N, O, F, P, S, Cl, Br, I
ORGANIC_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


def contains_only_elements(molecule, allowed_atomic_numbers=ORGANIC_ELEMENTS):
    """True if every atom of the molecule (SMILES or RDKit molecule) has an atomic number in the allowed set; False for a
    molecule that cannot be parsed.

    Args:
        molecule: SMILES string or RDKit molecule.
        allowed_atomic_numbers: set of allowed atomic numbers (H, C, N, O, F, P, S, Cl, Br, I by default).
    """
    mol = molecule if isinstance(molecule, Chem.Mol) else Chem.MolFromSmiles(str(molecule))
    if mol is None:
        return False
    return all(atom.GetAtomicNum() in allowed_atomic_numbers for atom in mol.GetAtoms())


def fit_isolation_forest(X_train, contamination="auto", seed=42, n_jobs=1):
    """Isolation Forest (Liu et al., 2008: 100 trees, sub-samples of min(256, n)) fitted on the training representation.

    Args:
        X_train: (n, d) training representation, the same one the model uses (e.g. the selected fingerprint bits).
        contamination: "auto" places outside the domain every sample with anomaly score s > 0.5, the boundary of Liu et
            al.; a number in (0, 0.5] instead sets the threshold so that this fraction of the training samples falls
            outside by construction (scikit-learn's rule, not Liu et al.'s).
        seed: random_state of the forest.
        n_jobs: threads used to fit and score the forest.
    """
    return IsolationForest(contamination=contamination, random_state=seed, n_jobs=n_jobs).fit(X_train)


def in_domain(forest, X):
    """Boolean array: True for the samples of X that the fitted forest labels as inliers.

    Args:
        forest: Isolation Forest returned by fit_isolation_forest.
        X: (m, d) representation of the molecules to check, with the same columns as the training representation.
    """
    return forest.predict(X) == 1


def anomaly_scores(forest, X):
    """Anomaly score s(x, n) of Liu et al. (2008, Eq. 2) of every sample: near 1 for anomalies, about 0.5 or less for
    normal samples.

    Args:
        forest: Isolation Forest returned by fit_isolation_forest.
        X: (m, d) representation of the molecules to score, with the same columns as the training representation.
    """
    return -forest.score_samples(X)


STRATEGIES = ("mean_std", "p95", "twice_mean")


def tanimoto_to_training(query_fp, train_fp):
    """Tanimoto similarity |A and B| / |A or B| between every query and every training fingerprint, as a
    (n_query, n_train) array; two empty fingerprints have similarity 0.

    Args:
        query_fp: (m, d) array-like of 0/1 values, one fingerprint per row.
        train_fp: (n, d) array-like of 0/1 values.
    """
    Q, T = np.asarray(query_fp, dtype=np.float64), np.asarray(train_fp, dtype=np.float64)
    inter = Q @ T.T
    union = Q.sum(axis=1)[:, None] + T.sum(axis=1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def nearest_similarity(query_fp, train_fp, k=5, exclude_self=False):
    """SIMILARITYNEAREST k of Sheridan et al. (2004): the mean Tanimoto similarity of every query to its k most similar
    training molecules.

    Args:
        query_fp: (m, d) array-like of 0/1 values.
        train_fp: (n, d) array-like of 0/1 values.
        k: number of nearest training molecules averaged (Sheridan et al. used 1, 3 and 5).
        exclude_self: if True, query_fp is train_fp itself and each molecule is not counted as its own neighbour.
    """
    sim = tanimoto_to_training(query_fp, train_fp)
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[1] - int(exclude_self))
    return np.sort(sim, axis=1)[:, -k:].mean(axis=1)


def neighbor_count(query_fp, train_fp, cutoff=0.7, exclude_self=False):
    """NEIGHBORCOUNT of Sheridan et al. (2004): the number of training molecules with similarity >= cutoff to each query.
    (Their plots use this count + 1, to allow a logarithmic axis.)

    Args:
        query_fp: (m, d) array-like of 0/1 values.
        train_fp: (n, d) array-like of 0/1 values.
        cutoff: similarity at or above which a training molecule is a neighbour (user-chosen; 0.7 is Sheridan et al.'s
            value for Dice similarity of atom pairs, not for Tanimoto of Morgan bits).
        exclude_self: as for nearest_similarity.
    """
    sim = tanimoto_to_training(query_fp, train_fp)
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    return (sim >= cutoff).sum(axis=1)


def fit_similarity_domain(train_fp, k=5, strategy="mean_std", z=0.5, percentile=95):
    """Similarity domain of a training set: the minimum mean similarity to the k nearest training molecules that a
    query needs to be inside, derived from the training molecules themselves (each one against the others).

    The training distances are 1 - nearest_similarity(train, train, k, exclude_self=True). With strategy "mean_std" the
    distance threshold is their mean + z standard deviations, with "p95" their `percentile`-th percentile, with
    "twice_mean" twice their mean (Sahigara et al., 2012; their default z is 0.5). The returned minimum similarity is
    1 - that distance.

    Args:
        train_fp: (n, d) array-like of 0/1 values.
        k: number of nearest neighbours (Sahigara et al. use 5).
        strategy: "mean_std", "p95" or "twice_mean".
        z: multiplier of the standard deviation of "mean_std".
        percentile: percentile of "p95".

    Returns:
        Dict with train_fp (the fingerprints, as a float array), k, strategy, min_similarity and the mean and standard
        deviation of the training distances.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    train = np.asarray(train_fp, dtype=np.float64)
    dist = 1.0 - nearest_similarity(train, train, k, exclude_self=True)
    mean, sd = float(dist.mean()), float(dist.std(ddof=1))
    limit = {"mean_std": mean + z * sd, "p95": float(np.percentile(dist, percentile)), "twice_mean": 2 * mean}[strategy]
    return {"train_fp": train, "k": k, "strategy": strategy, "min_similarity": 1.0 - limit,
            "distance_mean": mean, "distance_sd": sd}


def similarity_to_domain(domain, query_fp):
    """Mean similarity of every query to its k nearest training molecules (nearest_similarity) for a domain returned by
    fit_similarity_domain.

    Args:
        domain: dict returned by fit_similarity_domain.
        query_fp: (m, d) array-like of 0/1 values.
    """
    return nearest_similarity(query_fp, domain["train_fp"], domain["k"])


def in_similarity_domain(domain, query_fp):
    """Boolean array: True for the queries whose mean similarity to their k nearest training molecules is at least the
    domain's minimum similarity.

    Args:
        domain: dict returned by fit_similarity_domain.
        query_fp: (m, d) array-like of 0/1 values.
    """
    return similarity_to_domain(domain, query_fp) >= domain["min_similarity"]

import numpy as np
import pytest
from rdkit import Chem

from qsartools.applicability_domain import ORGANIC_ELEMENTS, contains_only_elements, fit_isolation_forest, in_domain


@pytest.mark.parametrize("smiles,expected", [("CCO", True), ("c1ccccc1Cl", True), ("CC(=O)O[Na]", False),
                                             ("C[Si](C)(C)C", False), ("[Pt](Cl)Cl", False), ("ICBr", True),
                                             ("not_a_smiles", False)])
def test_contains_only_elements(smiles, expected):
    assert contains_only_elements(smiles) is expected


def test_accepts_molecules_and_custom_sets():
    mol = Chem.MolFromSmiles("CC(=O)O[Na]")
    assert contains_only_elements(mol, ORGANIC_ELEMENTS | {11})
    assert not contains_only_elements("CCO", {6, 1})


def test_isolation_forest_flags_outliers():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (500, 4))
    forest = fit_isolation_forest(train, contamination=0.05, seed=1)
    assert in_domain(forest, train).mean() == pytest.approx(0.95, abs=0.01)
    far = rng.normal(8, 1, (50, 4))
    assert not in_domain(forest, far).any()
    assert in_domain(forest, np.zeros((1, 4))).all()


def test_reproducible():
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(300, 3)), rng.normal(0, 2, (200, 3))
    a = in_domain(fit_isolation_forest(train, seed=3), test)
    np.testing.assert_array_equal(a, in_domain(fit_isolation_forest(train, seed=3, n_jobs=2), test))


def test_auto_contamination_is_liu_boundary_of_score_half():
    from qsartools.applicability_domain import anomaly_scores
    rng = np.random.default_rng(1)
    X_tr = np.vstack([rng.normal(size=(300, 4)), rng.normal(8, 1, size=(6, 4))])
    forest = fit_isolation_forest(X_tr, seed=0)
    s = anomaly_scores(forest, X_tr)
    assert ((s > 0.5) == ~in_domain(forest, X_tr)).all()
    assert (s[-6:] > 0.6).all() and np.median(s[:300]) < 0.5


FP = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=float)


def test_tanimoto_to_training_matches_definition():
    from qsartools.applicability_domain import tanimoto_to_training
    sim = tanimoto_to_training(FP[:2], FP)
    assert sim[0, 0] == 1.0 and sim[0, 1] == pytest.approx(2 / 3) and sim[0, 2] == 0.0
    assert sim[0, 3] == 0.0 and tanimoto_to_training(FP[3:], FP[3:])[0, 0] == 0.0


def test_nearest_similarity_is_the_mean_of_the_k_largest():
    from qsartools.applicability_domain import nearest_similarity, tanimoto_to_training
    sim = tanimoto_to_training(FP[:1], FP)[0]
    assert nearest_similarity(FP[:1], FP, k=1)[0] == 1.0
    assert nearest_similarity(FP[:1], FP, k=2)[0] == pytest.approx(np.sort(sim)[-2:].mean())
    own = nearest_similarity(FP, FP, k=1, exclude_self=True)
    assert own[0] == pytest.approx(2 / 3) and own[1] == pytest.approx(2 / 3)


def test_neighbor_count_uses_the_cutoff_and_can_exclude_self():
    from qsartools.applicability_domain import neighbor_count
    assert list(neighbor_count(FP, FP, cutoff=0.6)) == [2, 2, 1, 0]  # an empty fingerprint is not similar to itself
    assert list(neighbor_count(FP, FP, cutoff=0.6, exclude_self=True)) == [1, 1, 0, 0]


def test_similarity_domain_strategies_and_membership():
    from qsartools.applicability_domain import fit_similarity_domain, in_similarity_domain, nearest_similarity
    rng = np.random.default_rng(0)
    train = (rng.random((200, 64)) < 0.3).astype(float)
    dist = 1 - nearest_similarity(train, train, 5, exclude_self=True)
    d = fit_similarity_domain(train, k=5, strategy="mean_std", z=0.5)
    assert d["min_similarity"] == pytest.approx(1 - (dist.mean() + 0.5 * dist.std(ddof=1)))
    assert fit_similarity_domain(train, strategy="p95")["min_similarity"] == pytest.approx(1 - np.percentile(dist, 95))
    assert fit_similarity_domain(train, strategy="twice_mean")["min_similarity"] == pytest.approx(1 - 2 * dist.mean())
    far = np.zeros((1, 64))  # no bit in common with anything: similarity 0
    assert in_similarity_domain(d, train[:5]).sum() >= 1 and not in_similarity_domain(d, far)[0]
    with pytest.raises(ValueError, match="strategy"):
        fit_similarity_domain(train, strategy="median")

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from qsartools.activity_cliffs import activity_cliff_pairs, cliff_compounds, tanimoto_matrix
from qsartools.featurization import morgan_fingerprints

SMILES = ["CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl", "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Br",
          "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N", "CCO", "c1ccccc1O", "c1ccccc1N"]


def test_tanimoto_matches_rdkit():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bitvects = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    sim = tanimoto_matrix(morgan_fingerprints(SMILES))
    for i, fp in enumerate(bitvects):
        np.testing.assert_allclose(sim[i], DataStructs.BulkTanimotoSimilarity(fp, bitvects))


def test_tanimoto_properties():
    sim = tanimoto_matrix(morgan_fingerprints(SMILES))
    np.testing.assert_allclose(np.diag(sim), 1.0)
    np.testing.assert_allclose(sim, sim.T)
    assert ((sim >= 0) & (sim <= 1)).all()
    assert tanimoto_matrix(np.zeros((2, 8)))[0, 1] == 0.0


def test_cliff_pairs_definition():
    X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    potency = np.array([8.0, 6.5, 7.6, 4.0])
    pairs = activity_cliff_pairs(X, potency, similarity_threshold=0.9, potency_difference=1.0)
    # (0, 1): similarity 0.9, difference 1.5 -> cliff; (0, 2): difference 0.4 -> not; (1, 2): 1.1 -> cliff; 3 dissimilar
    np.testing.assert_array_equal(pairs[:, :2], [[0, 1], [1, 2]])
    np.testing.assert_allclose(pairs[:, 2], [0.9, 1.0])
    np.testing.assert_allclose(pairs[:, 3], [1.5, 1.1])
    np.testing.assert_array_equal(cliff_compounds(4, pairs), [True, True, True, False])


def test_no_pairs():
    X = np.eye(5, dtype=np.uint8)
    pairs = activity_cliff_pairs(X, np.arange(5.0))
    assert pairs.shape == (0, 4) and not cliff_compounds(5, pairs).any()


def test_halogen_swap_is_a_cliff_only_with_a_large_potency_gap():
    X = morgan_fingerprints(SMILES[:2])
    sim = tanimoto_matrix(X)[0, 1]
    assert len(activity_cliff_pairs(X, [8.0, 6.0], similarity_threshold=sim)) == 1
    assert len(activity_cliff_pairs(X, [8.0, 7.5], similarity_threshold=sim)) == 0

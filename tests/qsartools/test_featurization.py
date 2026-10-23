import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

from qsartools.featurization import (RDKIT_2D_NAMES, RDKIT_3D_NAMES, morgan_fingerprints, rdkit_2d_descriptors,
                                 rdkit_3d_shape_descriptors)

SMILES = ["CCO", "c1ccccc1O", "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl", "CC(=O)Nc1ccc(O)cc1"]


def reference_morgan(smiles, radius, n_bits):
    """Same bits through an independent RDKit API (bit vector converted with DataStructs)."""
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class TestMorgan:
    def test_matches_independent_rdkit_api(self):
        X = morgan_fingerprints(SMILES)
        assert X.shape == (len(SMILES), 2048) and X.dtype == np.uint8
        for row, s in zip(X, SMILES):
            np.testing.assert_array_equal(row, reference_morgan(s, 2, 2048))

    @pytest.mark.parametrize("radius,n_bits", [(1, 1024), (3, 4096)])
    def test_radius_and_length(self, radius, n_bits):
        X = morgan_fingerprints(SMILES, radius=radius, n_bits=n_bits)
        assert X.shape == (len(SMILES), n_bits)
        np.testing.assert_array_equal(X[2], reference_morgan(SMILES[2], radius, n_bits))

    def test_accepts_molecules(self):
        mols = [Chem.MolFromSmiles(s) for s in SMILES]
        np.testing.assert_array_equal(morgan_fingerprints(mols), morgan_fingerprints(SMILES))

    def test_binary(self):
        assert set(np.unique(morgan_fingerprints(SMILES))) <= {0, 1}

    def test_invalid_raises_by_default(self):
        with pytest.raises(ValueError, match="Invalid molecule"):
            morgan_fingerprints(["CCO", "not_a_smiles"])

    def test_invalid_fill_gives_zero_row(self):
        X = morgan_fingerprints(["CCO", "not_a_smiles"], on_invalid="fill")
        assert X[1].sum() == 0 and X[0].sum() > 0

    def test_bad_on_invalid_value(self):
        with pytest.raises(ValueError, match="on_invalid"):
            morgan_fingerprints(SMILES, on_invalid="zeros")


class TestRdkit2D:
    def test_shape_and_names(self):
        X = rdkit_2d_descriptors(SMILES)
        assert X.shape == (len(SMILES), len(RDKIT_2D_NAMES))
        assert RDKIT_2D_NAMES == [name for name, _ in Descriptors._descList]

    @pytest.mark.parametrize("name,function", [("MolWt", Descriptors.MolWt), ("TPSA", Descriptors.TPSA),
                                               ("MolLogP", Descriptors.MolLogP), ("HeavyAtomCount", Descriptors.HeavyAtomCount)])
    def test_values_match_rdkit(self, name, function):
        X = rdkit_2d_descriptors(SMILES)
        j = RDKIT_2D_NAMES.index(name)
        np.testing.assert_allclose(X[:, j], [function(Chem.MolFromSmiles(s)) for s in SMILES])

    def test_parallel_equals_serial(self):
        np.testing.assert_array_equal(rdkit_2d_descriptors(SMILES, n_jobs=2), rdkit_2d_descriptors(SMILES, n_jobs=1))

    def test_no_infinite_values(self):
        assert not np.isinf(rdkit_2d_descriptors(SMILES)).any()

    def test_invalid(self):
        with pytest.raises(ValueError):
            rdkit_2d_descriptors(["CCO", "not_a_smiles"])
        X = rdkit_2d_descriptors(["CCO", "not_a_smiles"], on_invalid="fill")
        assert np.isnan(X[1]).all() and not np.isnan(X[0]).all()


class TestRdkit3D:
    def test_shape_and_determinism(self):
        X = rdkit_3d_shape_descriptors(SMILES)
        assert X.shape == (len(SMILES), len(RDKIT_3D_NAMES))
        np.testing.assert_array_equal(X, rdkit_3d_shape_descriptors(SMILES))

    def test_physical_constraints(self):
        X = rdkit_3d_shape_descriptors(SMILES[1:])
        col = {n: X[:, i] for i, n in enumerate(RDKIT_3D_NAMES)}
        assert (col["PMI1"] <= col["PMI2"]).all() and (col["PMI2"] <= col["PMI3"]).all()
        assert (col["NPR1"] <= col["NPR2"] + 1e-12).all() and (col["NPR2"] <= 1 + 1e-12).all()
        # Normalized principal moment ratios are defined as PMI1/PMI3 and PMI2/PMI3
        np.testing.assert_allclose(col["NPR1"], col["PMI1"] / col["PMI3"])
        np.testing.assert_allclose(col["NPR2"], col["PMI2"] / col["PMI3"])

    def test_benzene_is_planar(self):
        # A planar rigid body has PMI3 = PMI1 + PMI2
        X = rdkit_3d_shape_descriptors(["c1ccccc1"])[0]
        pmi = dict(zip(RDKIT_3D_NAMES, X))
        assert pmi["PMI3"] == pytest.approx(pmi["PMI1"] + pmi["PMI2"], rel=1e-3)

    def test_invalid_fill(self):
        X = rdkit_3d_shape_descriptors(["CCO", "not_a_smiles"], on_invalid="fill")
        assert np.isnan(X[1]).all() and not np.isnan(X[0]).any()

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

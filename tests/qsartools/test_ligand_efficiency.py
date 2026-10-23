import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from qsartools.ligand_efficiency import METRICS, ligand_efficiency_metrics

PYRIMETHAMINE = "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl"


def test_definitions_with_independent_rdkit_functions():
    mol = Chem.MolFromSmiles(PYRIMETHAMINE)
    m = ligand_efficiency_metrics(6.56, mol)
    assert m["LE"] == pytest.approx(1.37 * 6.56 / mol.GetNumHeavyAtoms())
    average_mass = sum(atom.GetMass() for atom in Chem.AddHs(mol).GetAtoms())
    assert m["BEI"] == pytest.approx(6.56 / average_mass * 1000)
    assert m["LLE"] == pytest.approx(6.56 - Crippen.MolLogP(mol))
    assert m["SEI"] == pytest.approx(6.56 / rdMolDescriptors.CalcTPSA(mol) * 100)


def test_pyrimethamine_heavy_atom_count_and_le():
    # C12H13ClN4: 17 heavy atoms
    assert ligand_efficiency_metrics(6.56, PYRIMETHAMINE)["LE"] == pytest.approx(1.37 * 6.56 / 17)


def test_rounding_and_order():
    m = ligand_efficiency_metrics(7.0, "CCO", decimals=2)
    assert tuple(m) == METRICS and all(round(v, 2) == v for v in m.values())


def test_zero_denominators_and_invalid():
    # Methane: one heavy atom and no polar surface
    m = ligand_efficiency_metrics(5.0, "C")
    assert m["LE"] == pytest.approx(1.37 * 5) and np.isnan(m["SEI"])
    assert all(np.isnan(v) for v in ligand_efficiency_metrics(5.0, "not_a_smiles").values())

"""Ligand efficiency metrics: potency normalized by size, mass, lipophilicity or polar surface area.

LE  = 1.37 * pIC50 / HAC         (Hopkins et al., 2004; kcal/mol per heavy atom at 300 K)
BEI = 1000 * pIC50 / MW          (Abad-Zapatero & Metz, 2005; MW in Da, i.e. pIC50 per kDa)
LLE = pIC50 - cLogP              (Leeson & Springthorpe, 2007; Crippen logP)
SEI = 100 * pIC50 / TPSA         (Abad-Zapatero & Metz, 2005; topological polar surface area in A^2)
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

METRICS = ("LE", "BEI", "LLE", "SEI")


def ligand_efficiency_metrics(pic50, molecule, decimals=None):
    """The four metrics of one molecule (SMILES or RDKit molecule) for a potency pIC50, as a dict LE, BEI, LLE, SEI.

    A metric whose denominator is zero (no heavy atoms, no polar surface) is NaN, and so are all four for a molecule that
    cannot be parsed.

    Args:
        pic50: potency as pIC50 (-log10 of the IC50 in mol/L), measured or predicted.
        molecule: SMILES string or RDKit molecule.
        decimals: if given, every metric is rounded to this number of decimals; None leaves them unrounded.
    """
    mol = molecule if isinstance(molecule, Chem.Mol) else Chem.MolFromSmiles(str(molecule))
    if mol is None:
        return dict.fromkeys(METRICS, np.nan)
    hac = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    mw, logp, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
    values = {"LE": 1.37 * pic50 / hac if hac > 0 else np.nan,
              "BEI": pic50 / mw * 1000 if mw > 0 else np.nan,
              "LLE": pic50 - logp,
              "SEI": pic50 / tpsa * 100 if tpsa > 0 else np.nan}
    if decimals is not None:
        values = {k: round(v, decimals) for k, v in values.items()}
    return values

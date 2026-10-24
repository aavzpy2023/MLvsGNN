"""Ligand efficiency metrics: potency normalized by size, mass, lipophilicity or polar surface area.

LE  = 1.37 * pIC50 / HAC         (Hopkins et al., 2004, Drug Discov. Today 9:430-431: Delta g = Delta G / N non-hydrogen
                                  atoms, Delta G = -RT ln K at 300 K, so -Delta G = RT ln(10) pK = 1.37 pK kcal/mol;
                                  an IC50 is taken as if it were the K. Leeson & Springthorpe, 2007, use pIC50 / HAC)
BEI = 1000 * pIC50 / MW          (Abad-Zapatero & Metz, 2005, Drug Discov. Today 10:464-469; MW in Da, i.e. pIC50 per kDa)
LLE = pIC50 - logP               (Leeson & Springthorpe, 2007, Nat. Rev. Drug Discov. 6:881-890, who use cLogP or LogD;
                                  here RDKit's Crippen logP)
SEI = 100 * pIC50 / TPSA         (Abad-Zapatero & Metz, 2005; polar surface area in A^2 per 100 A^2, calculated as in
                                  Ertl et al., 2000, which is RDKit's TPSA)

The metrics describe the ligand, so by default only the largest fragment of a salt or mixture is used.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from qsartools.featurization import largest_fragment as _largest_fragment

METRICS = ("LE", "BEI", "LLE", "SEI")


def ligand_efficiency_metrics(pic50, molecule, decimals=None, largest_fragment=True, le_factor=1.37):
    """The four metrics of one molecule (SMILES or RDKit molecule) for a potency pIC50, as a dict LE, BEI, LLE, SEI.

    A metric whose denominator is zero (no heavy atoms, no polar surface) is NaN, and so are all four for a molecule that
    cannot be parsed.

    Args:
        pic50: potency as pIC50 (-log10 of the IC50 in mol/L), measured or predicted.
        molecule: SMILES string or RDKit molecule.
        decimals: if given, every metric is rounded to this number of decimals; None leaves them unrounded.
        largest_fragment: if True, a salt or mixture is reduced to its largest fragment (RDKit's LargestFragmentChooser,
            by number of heavy atoms) before the metrics are computed, so that counter-ions do not count as ligand
            atoms or mass; False uses every atom.
        le_factor: multiplier of pIC50 / HAC in LE: 1.37 is RT ln(10) in kcal/mol at 300 K (Hopkins et al.), 1.0 gives
            Leeson & Springthorpe's pIC50 / HAC.
    """
    mol = molecule if isinstance(molecule, Chem.Mol) else Chem.MolFromSmiles(str(molecule))
    if mol is None:
        return dict.fromkeys(METRICS, np.nan)
    if largest_fragment:
        mol = _largest_fragment(mol)
    hac = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    mw, logp, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
    values = {"LE": le_factor * pic50 / hac if hac > 0 else np.nan,
              "BEI": pic50 / mw * 1000 if mw > 0 else np.nan,
              "LLE": pic50 - logp,
              "SEI": pic50 / tpsa * 100 if tpsa > 0 else np.nan}
    if decimals is not None:
        values = {k: round(v, decimals) for k, v in values.items()}
    return values

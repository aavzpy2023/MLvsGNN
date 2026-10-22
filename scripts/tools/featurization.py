"""Molecular featurization: Morgan fingerprints, RDKit 2D descriptors and RDKit 3D shape descriptors.

Inputs are SMILES strings or RDKit molecules; outputs are NumPy arrays with one row per input, in input order.
"""
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdFingerprintGenerator

RDKIT_2D_NAMES = [name for name, _ in Descriptors._descList]
RDKIT_3D_FUNCTIONS = {
    'Asphericity': Descriptors3D.Asphericity, 'Eccentricity': Descriptors3D.Eccentricity,
    'InertialShapeFactor': Descriptors3D.InertialShapeFactor, 'NPR1': Descriptors3D.NPR1, 'NPR2': Descriptors3D.NPR2,
    'PMI1': Descriptors3D.PMI1, 'PMI2': Descriptors3D.PMI2, 'PMI3': Descriptors3D.PMI3,
    'RadiusOfGyration': Descriptors3D.RadiusOfGyration, 'SpherocityIndex': Descriptors3D.SpherocityIndex,
}
RDKIT_3D_NAMES = list(RDKIT_3D_FUNCTIONS)
ON_INVALID = ("raise", "fill")


def to_mol(molecule):
    """RDKit molecule from a SMILES string (None if it cannot be parsed); an RDKit molecule is returned unchanged."""
    if isinstance(molecule, Chem.Mol):
        return molecule
    return Chem.MolFromSmiles(str(molecule))


def _check_on_invalid(on_invalid):
    if on_invalid not in ON_INVALID:
        raise ValueError(f"on_invalid must be one of {ON_INVALID}, not {on_invalid!r}")


def _parse(molecule, on_invalid):
    mol = to_mol(molecule)
    if mol is None and on_invalid == "raise":
        raise ValueError(f"Invalid molecule: {molecule!r}")
    return mol


def morgan_fingerprints(molecules, radius=2, n_bits=2048, on_invalid="raise"):
    """Morgan bit vectors (ECFP4 for radius 2), without chirality, as an (n, n_bits) uint8 array of 0/1.

    on_invalid: "raise" (default) raises ValueError for a molecule that cannot be parsed; "fill" gives it an all-zero row.
    """
    _check_on_invalid(on_invalid)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(molecules), n_bits), dtype=np.uint8)
    for i, molecule in enumerate(molecules):
        mol = _parse(molecule, on_invalid)
        if mol is not None:
            out[i] = generator.GetFingerprintAsNumPy(mol)
    return out


def _rdkit_2d_row(molecule, on_invalid):
    mol = _parse(molecule, on_invalid)
    if mol is None:
        return np.full(len(RDKIT_2D_NAMES), np.nan)
    values = []
    for _, function in Descriptors._descList:
        try:
            values.append(function(mol))
        except Exception:
            values.append(np.nan)
    return np.array(values, dtype=float)


def rdkit_2d_descriptors(molecules, n_jobs=1, on_invalid="raise"):
    """The full RDKit 2D descriptor list (RDKIT_2D_NAMES) as an (n, len(RDKIT_2D_NAMES)) float array.

    A descriptor that fails, or is infinite, is NaN; imputation is left to the model pipeline, so that it can be fitted on
    training data only. on_invalid: "raise" (default) or "fill" (a row of NaN).

    Reproducibility note: RDKit's Ipc and AvgIpc are numerically unstable for very large molecules (Ipc of order 1e98)
    and can differ in their last digits depending on which molecules the same process computed before. Results are
    identical for the same inputs, order and n_jobs.
    """
    _check_on_invalid(on_invalid)
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_rdkit_2d_row)(m, on_invalid) for m in molecules)
    X = np.array(rows, dtype=float).reshape(len(molecules), len(RDKIT_2D_NAMES))
    return np.where(np.isinf(X), np.nan, X)


def _rdkit_3d_row(molecule, seed, on_invalid):
    mol = _parse(molecule, on_invalid)
    missing = np.full(len(RDKIT_3D_NAMES), np.nan)
    if mol is None:
        return missing
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) == -1:
            if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed) == -1:
                return missing
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        return np.array([function(mol) for function in RDKIT_3D_FUNCTIONS.values()], dtype=float)
    except Exception:
        return missing


def rdkit_3d_shape_descriptors(molecules, seed=42, n_jobs=1, on_invalid="raise"):
    """Ten RDKit 3D shape descriptors (RDKIT_3D_NAMES) of one conformer per molecule, as an (n, 10) float array.

    Hydrogens are added and one conformer is embedded with ETKDGv3 (random seed `seed`; random initial coordinates as a
    fallback) and minimized with MMFF94. A molecule that cannot be embedded gets a row of NaN; infinite values are NaN.
    on_invalid: "raise" (default) or "fill" (a row of NaN).
    """
    _check_on_invalid(on_invalid)
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_rdkit_3d_row)(m, seed, on_invalid) for m in molecules)
    X = np.array(rows, dtype=float).reshape(len(molecules), len(RDKIT_3D_NAMES))
    return np.where(np.isinf(X), np.nan, X)

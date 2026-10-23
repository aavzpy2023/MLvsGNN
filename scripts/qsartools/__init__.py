"""qsartools: reusable building blocks for small-data QSAR modelling and its statistical evaluation.

Every module follows the same contract, checked by tests/qsartools/test_agnostic.py:
- it depends only on general scientific libraries (NumPy, SciPy, pandas, scikit-learn, LightGBM, XGBoost, RDKit, joblib),
  never on the code of a particular project;
- it reads and writes no files: data and parameters go in, results come out;
- it assumes no column names, targets or thresholds of a particular data set: everything is a parameter with a
  documented default;
- it keeps no global state that changes results: randomness is controlled by explicit seed parameters.
- every public function documents the meaning of each parameter in an 'Args:' section of its docstring.
"""

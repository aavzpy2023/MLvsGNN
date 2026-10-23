import numpy as np
import pytest
from sklearn.model_selection import RepeatedKFold, train_test_split

from qsartools.splits import holdout_split, murcko_scaffolds, outer_splits, random_repeated_kfold, scaffold_repeated_kfold

# Three benzene derivatives, two pyridines, two naphthalenes, an acyclic molecule and singletons
SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccncc1O", "c1ccncc1C", "c1ccc2ccccc2c1O", "c1ccc2ccccc2c1N",
          "CCO", "C1CCCCC1O", "C1CCNCC1", "c1ccoc1", "c1ccsc1C", "O=C1CCCN1", "c1cnc2ccccc2c1"]


def check_partition(splits, n, n_splits):
    for r in range(len(splits) // n_splits):
        tests = [te for _, te in splits[r * n_splits:(r + 1) * n_splits]]
        assert sorted(np.concatenate(tests)) == list(range(n)), "every sample tested once per repeat"
        for tr, te in splits[r * n_splits:(r + 1) * n_splits]:
            assert not set(tr) & set(te) and len(tr) + len(te) == n


def test_murcko_scaffolds():
    sc = murcko_scaffolds(["c1ccccc1O", "c1ccccc1N", "CCO", "not_a_smiles", "C[C@H](N)c1ccccc1"])
    assert sc[0] == sc[1] == "c1ccccc1" and sc[2] == "" and sc[3] == "" and sc[4] == "c1ccccc1"


def test_random_equals_sklearn_and_is_a_partition():
    splits = list(random_repeated_kfold(len(SMILES), n_splits=3, n_repeats=2, seed=1))
    ref = list(RepeatedKFold(n_splits=3, n_repeats=2, random_state=1).split(np.zeros(len(SMILES))))
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(splits, ref))
    check_partition(splits, len(SMILES), 3)


def test_scaffold_split_keeps_scaffolds_together():
    groups = murcko_scaffolds(SMILES)
    splits = list(scaffold_repeated_kfold(SMILES, n_splits=3, n_repeats=3, seed=0))
    check_partition(splits, len(SMILES), 3)
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])


def test_scaffold_split_is_reproducible_and_seed_dependent():
    a = list(scaffold_repeated_kfold(SMILES, 3, 2, seed=0))
    b = list(scaffold_repeated_kfold(SMILES, 3, 2, seed=0))
    c = list(scaffold_repeated_kfold(SMILES, 3, 2, seed=5))
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c))


def test_outer_splits_dispatch():
    assert len(list(outer_splits(SMILES, 3, 2, 0, "random"))) == 6
    assert len(list(outer_splits(SMILES, 3, 2, 0, "scaffold"))) == 6
    with pytest.raises(ValueError, match="Unknown split kind"):
        list(outer_splits(SMILES, kind="cluster"))


def test_holdout_equals_train_test_split():
    tr, te = holdout_split(100, test_size=0.15, seed=42)
    ref_tr, ref_te = train_test_split(np.arange(100), test_size=0.15, random_state=42)
    assert np.array_equal(tr, ref_tr) and np.array_equal(te, ref_te) and len(te) == 15

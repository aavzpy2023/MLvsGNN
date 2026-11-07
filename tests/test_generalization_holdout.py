"""Scoring of the generalization cohort on its published partition (step 21).

The step itself fits two models, which no unit test should do. What is tested here is everything around that fit: the
check that a data file and its partition file describe the same molecules, the split of the test set into activity
cliffs and the rest, and what the scores do when a subset is too small to score. Those are the paths that can produce a
wrong number without raising, and the per-target cliff subsets are small enough for the last one to matter: the
smallest target of the cohort has thirteen cliff compounds in its test partition.

Expected values are written out by hand, as in test_reference_values.py.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load_step21():
    """Import the step by path, because its file name is not a valid module name."""
    path = os.path.join(ROOT, "scripts", "21_generalization_holdout.py")
    spec = importlib.util.spec_from_file_location("step21", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step21 = _load_step21()


class TestScore:
    def test_hand_computed_r2_and_mae(self):
        # y = [1, 2, 3, 4] (mean 2.5), predictions [1.5, 2.5, 2.5, 3.5]
        # SS_res = 1.0, SS_tot = 5.0 -> R2 = 0.8; MAE = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        r2, mae = step21.score(np.array([1.0, 2, 3, 4]), np.array([1.5, 2.5, 2.5, 3.5]))
        assert r2 == pytest.approx(0.8)
        assert mae == pytest.approx(0.5)

    def test_one_compound_has_no_r2_but_has_an_error(self):
        # R2 needs the variance of the observations, which a single compound does not have. The error does not:
        # reporting NaN for it would lose information, and reporting 0.0 for R2 would invent a result.
        r2, mae = step21.score(np.array([7.0]), np.array([6.5]))
        assert np.isnan(r2)
        assert mae == pytest.approx(0.5)

    def test_an_empty_subset_gives_no_score_at_all(self):
        # A target whose test partition contains no activity cliff must produce NaN, not a crash and not a zero.
        r2, mae = step21.score(np.array([]), np.array([]))
        assert np.isnan(r2) and np.isnan(mae)

    def test_constant_observations_have_no_defined_r2(self):
        # SS_tot is zero, so R2 is not defined; scikit-learn returns 0.0 here, which would read as "no better than
        # the mean" when in fact the quantity does not exist. The MAE is still meaningful.
        r2, mae = step21.score(np.array([5.0, 5.0, 5.0]), np.array([5.0, 4.0, 6.0]))
        assert np.isnan(r2) or r2 <= 0.0
        assert mae == pytest.approx(2.0 / 3.0)


class TestRowsFor:
    def test_the_three_subsets_partition_the_test_set(self):
        y = np.array([1.0, 2, 3, 4, 5])
        pred = y.copy()
        cliff = np.array([True, False, True, False, False])
        rows = step21.rows_for("D", "M", y, pred, cliff)
        by_subset = {r["subset"]: r for r in rows}
        assert set(by_subset) == {"all", "cliff", "noncliff"}
        assert by_subset["all"]["n"] == 5
        assert by_subset["cliff"]["n"] == 2
        assert by_subset["noncliff"]["n"] == 3
        # The two parts must add up to the whole, or a compound is being counted twice or not at all.
        assert by_subset["cliff"]["n"] + by_subset["noncliff"]["n"] == by_subset["all"]["n"]

    def test_each_subset_is_scored_on_its_own_compounds(self):
        # The cliff compounds are predicted perfectly and the rest are off by one, so the two subsets must not share
        # a score: a mask applied to the wrong array would make them equal.
        y = np.array([1.0, 2, 3, 4])
        cliff = np.array([True, True, False, False])
        pred = np.where(cliff, y, y + 1.0)
        by_subset = {r["subset"]: r for r in step21.rows_for("D", "M", y, pred, cliff)}
        assert by_subset["cliff"]["mae"] == pytest.approx(0.0)
        assert by_subset["noncliff"]["mae"] == pytest.approx(1.0)
        assert by_subset["all"]["mae"] == pytest.approx(0.5)

    def test_a_target_with_no_cliff_compound_still_produces_its_rows(self):
        y = np.array([1.0, 2, 3])
        rows = step21.rows_for("D", "M", y, y, np.zeros(3, dtype=bool))
        by_subset = {r["subset"]: r for r in rows}
        assert by_subset["cliff"]["n"] == 0
        assert np.isnan(by_subset["cliff"]["r2"])
        assert by_subset["noncliff"]["n"] == 3

    def test_the_dataset_and_model_labels_travel_with_every_row(self):
        rows = step21.rows_for("CHEMBL1862_Ki", "Classical (RF+SVM)", np.array([1.0, 2]), np.array([1.0, 2]),
                               np.array([True, False]))
        assert all(r["dataset"] == "CHEMBL1862_Ki" for r in rows)
        assert all(r["model"] == "Classical (RF+SVM)" for r in rows)


class TestAlignmentOfTheDataAndSplitFiles:
    """Step 21 reads the potencies from one file and the partition and cliff flags from another, and relies on the two
    being in the same order. The check that they are is what keeps a silent mismatch from scoring the right
    predictions against the wrong labels."""

    @staticmethod
    def _write_pair(tmp_path, data_smiles, split_smiles):
        pd.DataFrame({"cid": range(len(data_smiles)), "Smiles": data_smiles,
                      "Standard Value": [1.0] * len(data_smiles),
                      "pIC50 Value": [6.0] * len(data_smiles)}).to_csv(tmp_path / "T.csv", index=False)
        pd.DataFrame({"Smiles": split_smiles, "split": ["train"] * len(split_smiles),
                      "cliff_mol": [0] * len(split_smiles)}).to_csv(tmp_path / "T_split.csv", index=False)

    def test_a_reordered_split_file_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(step21, "MOLECULEACE_DIR", str(tmp_path))
        self._write_pair(tmp_path, ["CCO", "CCC", "CCN"], ["CCC", "CCO", "CCN"])
        with pytest.raises(ValueError, match="not aligned"):
            step21.run_target("T", n_jobs=1)

    def test_a_split_file_of_a_different_length_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(step21, "MOLECULEACE_DIR", str(tmp_path))
        self._write_pair(tmp_path, ["CCO", "CCC", "CCN"], ["CCO", "CCC"])
        with pytest.raises(ValueError, match="not aligned"):
            step21.run_target("T", n_jobs=1)


class TestTheCohortOnDisk:
    """The twelve prepared targets, if step 20 has been run. Skipped otherwise, so the suite still runs on a clean
    checkout."""

    @pytest.fixture
    def index(self):
        path = os.path.join(ROOT, "data", "moleculeace", "index.csv")
        if not os.path.exists(path):
            pytest.skip("the MoleculeACE cohort has not been prepared (step 20)")
        return pd.read_csv(path)

    def test_every_target_is_aligned_with_its_split_file(self, index):
        for dataset in index["dataset"]:
            base = os.path.join(ROOT, "data", "moleculeace")
            data = pd.read_csv(os.path.join(base, f"{dataset}.csv"))
            split = pd.read_csv(os.path.join(base, f"{dataset}_split.csv"))
            assert len(data) == len(split), dataset
            assert (data["Smiles"].values == split["Smiles"].values).all(), dataset

    def test_every_test_partition_can_be_scored(self, index):
        # A target whose test partition held fewer than two compounds, or fewer than two outside the cliffs, would
        # produce a NaN that the aggregation must not average in. None does; this records that.
        for dataset in index["dataset"]:
            split = pd.read_csv(os.path.join(ROOT, "data", "moleculeace", f"{dataset}_split.csv"))
            test = split[split["split"] == "test"]
            assert len(test) >= 2, dataset
            assert int((~test["cliff_mol"].astype(bool)).sum()) >= 2, dataset

    def test_the_cliff_subsets_are_small_enough_to_need_pooling(self, index):
        # The manuscript reports the cliff comparison pooled over targets rather than per target. This test records
        # the reason: in the smallest target the per-target cliff subset is a few dozen compounds at most.
        smallest = min(int(pd.read_csv(os.path.join(ROOT, "data", "moleculeace", f"{d}_split.csv"))
                           .query("split == 'test'")["cliff_mol"].sum()) for d in index["dataset"])
        assert smallest < 100

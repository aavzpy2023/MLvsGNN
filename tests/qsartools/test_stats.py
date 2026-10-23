import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import r2_score

from qsartools.stats import (bootstrap_distribution, bootstrap_p_value, corrected_ttest, corrected_welch_test, holm,
                             percentile_interval, r2)

RNG = np.random.default_rng(0)
DIFF = RNG.normal(0.02, 0.03, 25)


class TestCorrectedTtest:
    def test_without_correction_equals_scipy_one_sample_ttest(self):
        # n_test = 0 removes the correction term, leaving the ordinary one-sample t-test
        res = corrected_ttest(DIFF, n_train=1, n_test=0, popmean=0.01)
        ref = stats.ttest_1samp(DIFF, 0.01)
        assert res["t"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
        assert res["df"] == len(DIFF) - 1

    def test_correction_inflates_the_standard_error(self):
        naive = corrected_ttest(DIFF, n_train=1, n_test=0)["se"]
        corrected = corrected_ttest(DIFF, n_train=4, n_test=1)["se"]
        # (1/J + n_test/n_train) / (1/J) = 1 + J/4 for J = 25 folds
        assert corrected / naive == pytest.approx(np.sqrt(1 + len(DIFF) / 4))

    def test_tost_equals_two_one_sided_scipy_tests(self):
        margin = 0.05
        res = corrected_ttest(DIFF, n_train=1, n_test=0, margin=margin)
        p_lower = stats.ttest_1samp(DIFF, -margin, alternative="greater").pvalue
        p_upper = stats.ttest_1samp(DIFF, margin, alternative="less").pvalue
        assert res["p_equiv"] == pytest.approx(max(p_lower, p_upper))

    def test_ci90_matches_scipy_interval(self):
        res = corrected_ttest(DIFF, n_train=1, n_test=0, margin=0.05)
        lo, hi = stats.t.interval(0.90, df=len(DIFF) - 1, loc=DIFF.mean(), scale=stats.sem(DIFF))
        assert (res["ci90_low"], res["ci90_high"]) == pytest.approx((lo, hi))

    def test_equivalence_iff_interval_inside_margin(self):
        for margin in (0.01, 0.03, 0.05, 0.1):
            res = corrected_ttest(DIFF, n_train=4, n_test=1, margin=margin)
            inside = -margin < res["ci90_low"] and res["ci90_high"] < margin
            assert (res["p_equiv"] < 0.05) == inside

    def test_zero_variance(self):
        # Exactly representable constants, so that the sample variance is exactly zero
        res = corrected_ttest(np.full(10, 0.25), n_train=4, n_test=1, margin=0.5)
        assert res["p"] == 1.0 and res["p_equiv"] == 0.0
        assert corrected_ttest(np.full(10, 0.5), n_train=4, n_test=1, margin=0.25)["p_equiv"] == 1.0


class TestWelch:
    def test_formula_and_antisymmetry(self):
        a, b = RNG.normal(0.75, 0.04, 25), RNG.normal(0.73, 0.05, 25)
        res = corrected_welch_test(a, b, n_train=687, n_test=172)
        inflation = 1 / 25 + 172 / 687
        se = np.sqrt(inflation * a.var(ddof=1) + inflation * b.var(ddof=1))
        assert res["mean_diff"] == pytest.approx(b.mean() - a.mean()) and res["se"] == pytest.approx(se)
        assert res["p"] == pytest.approx(2 * stats.t.sf(abs((b.mean() - a.mean()) / se), 48))
        swapped = corrected_welch_test(b, a, n_train=687, n_test=172)
        assert swapped["t"] == pytest.approx(-res["t"]) and swapped["p"] == pytest.approx(res["p"])


class TestHolm:
    def test_hand_computed_example(self):
        # sorted 0.005, 0.01, 0.03, 0.04 -> 4*0.005, 3*0.01, 2*0.03, max(1*0.04, 0.06)
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

    def test_properties(self):
        p = RNG.uniform(0, 0.2, 12)
        adj = holm(p)
        assert (adj >= p).all() and (adj <= 1).all()
        order = np.argsort(p)
        assert (np.diff(adj[order]) >= 0).all()
        assert holm([0.7]) == pytest.approx([0.7]) and holm([0.6, 0.7]).max() == 1.0


class TestBootstrap:
    y = RNG.normal(6, 1, 129)
    pred = y + RNG.normal(0, 0.5, 129)

    def test_r2_equals_sklearn(self):
        assert r2(self.y, self.pred) == pytest.approx(r2_score(self.y, self.pred), abs=1e-12)

    def test_reproducible_and_paired(self):
        a = bootstrap_distribution(self.y, self.pred, n_boot=300, seed=7)
        np.testing.assert_array_equal(a, bootstrap_distribution(self.y, self.pred, n_boot=300, seed=7))
        # Same seed -> same resamples: a model compared with itself has a zero paired difference
        assert np.all(a - bootstrap_distribution(self.y, self.pred, n_boot=300, seed=7) == 0)

    def test_custom_metric(self):
        mae = bootstrap_distribution(self.y, self.pred, metric=lambda t, p: np.mean(np.abs(t - p)), n_boot=200)
        assert mae.shape == (200,) and (mae > 0).all()

    def test_interval_and_p_value(self):
        dist = bootstrap_distribution(self.y, self.pred, n_boot=1000, seed=1)
        mean, lo, hi = percentile_interval(dist)
        assert lo < r2(self.y, self.pred) < hi and lo < mean < hi
        assert bootstrap_p_value(dist, mean) > 0.5
        assert bootstrap_p_value(dist, 2.0) == 0.0
        assert bootstrap_p_value(np.zeros(10), 0.0) == 1.0

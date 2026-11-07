"""Fixed reference values for the quantities the manuscript reports.

The rest of the suite checks our functions against scikit-learn, SciPy and RDKit. That shows we call those libraries
correctly, but it cannot show that the result is right: a reader who distrusts an implementation is not reassured by a
test that compares it with another implementation. The expectations in this file are therefore written out as literal
numbers, derived by hand from the published definition of each quantity, with the arithmetic shown in the docstring so
that it can be checked on paper. They do not change when a dependency is upgraded, and a failure here means either our
code or the library beneath it has changed behaviour.

Every reference is a small case chosen so that the arithmetic closes exactly or to a few decimals.
"""
import numpy as np
import pytest

from qsartools.applicability_domain import nearest_similarity, tanimoto_to_training
from qsartools.splits import murcko_scaffolds
from qsartools.stats import corrected_ttest, holm, r2


class TestCoefficientOfDetermination:
    """R2 = 1 - SS_res / SS_tot, with SS_tot about the mean of the observations."""

    def test_hand_computed(self):
        # y = [1, 2, 3, 4], mean 2.5; predictions [1.5, 2.5, 2.5, 3.5]
        # SS_res = 0.25 + 0.25 + 0.25 + 0.25 = 1.0
        # SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        # R2 = 1 - 1.0 / 5.0 = 0.8
        assert r2([1, 2, 3, 4], [1.5, 2.5, 2.5, 3.5]) == pytest.approx(0.8)

    def test_predicting_the_mean_gives_zero(self):
        # SS_res = SS_tot by construction, so R2 = 0 exactly.
        assert r2([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]) == pytest.approx(0.0)

    def test_is_negative_when_worse_than_the_mean(self):
        # y = [1, 2, 3, 4], predictions [4, 3, 2, 1]: SS_res = 9 + 1 + 1 + 9 = 20, SS_tot = 5, R2 = 1 - 4 = -3.
        assert r2([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-3.0)


class TestNadeauBengioCorrection:
    """Nadeau and Bengio (2003, Mach. Learn. 52:239-281), the correction every comparison in the manuscript uses:
    the variance of the mean of J per-fold differences is inflated from s^2 / J to s^2 (1/J + n_test / n_train)."""

    def test_corrected_standard_error_is_hand_computable(self):
        # diff = [0.1, 0.2, 0.3, 0.4]: mean 0.25, sample variance (ddof=1) = 0.05 / 3 = 0.0166667.
        # With n_train = 800 and n_test = 200 the factor is 1/4 + 200/800 = 0.25 + 0.25 = 0.5.
        # se = sqrt(0.0166667 * 0.5) = sqrt(0.00833333) = 0.0912871
        out = corrected_ttest([0.1, 0.2, 0.3, 0.4], n_train=800, n_test=200)
        assert out["mean_diff"] == pytest.approx(0.25)
        assert out["se"] == pytest.approx(0.0912871, abs=1e-6)
        assert out["t"] == pytest.approx(0.25 / 0.0912871, abs=1e-6)
        assert out["df"] == 3

    def test_the_five_fold_ratio_of_the_manuscript(self):
        # A 5-fold outer loop trains on 4/5 and tests on 1/5, so n_test / n_train = 1/4, the value Methods quotes.
        # With 25 differences of sample variance s^2, the factor is 1/25 + 1/4 = 0.29.
        diff = np.array([0.02] * 13 + [-0.02] * 12)
        n = len(diff)
        s2 = diff.var(ddof=1)
        expected = float(np.sqrt(s2 * (1.0 / n + 0.25)))
        out = corrected_ttest(diff, n_train=4, n_test=1)
        assert out["se"] == pytest.approx(expected)
        # The correction must make the error bar wider, never narrower, than the uncorrected one.
        assert out["se"] > float(np.sqrt(s2 / n))

    def test_uncorrected_case_is_the_textbook_standard_error(self):
        # n_test = 0 removes the correction term and leaves se = s / sqrt(J).
        diff = [0.1, 0.2, 0.3, 0.4]
        out = corrected_ttest(diff, n_train=1, n_test=0)
        assert out["se"] == pytest.approx(np.std(diff, ddof=1) / np.sqrt(4))

    def test_equivalence_interval_is_the_mean_plus_minus_t95_times_se(self):
        # df = 3, so t_0.95 = 2.353363 (Student table). mean 0.25, se 0.0912871 from the case above.
        # ci = 0.25 +- 2.353363 * 0.0912871 = [0.0352, 0.4648]
        out = corrected_ttest([0.1, 0.2, 0.3, 0.4], n_train=800, n_test=200, margin=0.05)
        assert out["ci90_low"] == pytest.approx(0.0352, abs=5e-4)
        assert out["ci90_high"] == pytest.approx(0.4648, abs=5e-4)

    def test_equivalence_is_declared_exactly_when_the_interval_fits_in_the_margin(self):
        # The manuscript states this correspondence, so it is checked as a property and not as a p-value.
        for margin in (0.01, 0.05, 0.2, 1.0, 5.0):
            out = corrected_ttest([0.1, 0.2, 0.3, 0.4], n_train=800, n_test=200, margin=margin)
            inside = -margin < out["ci90_low"] and out["ci90_high"] < margin
            assert inside == (out["p_equiv"] < 0.05)


class TestHolm:
    """Holm (1979, Scand. J. Stat. 6:65-70), step-down: the i-th smallest p is multiplied by (n - i + 1) and the
    sequence is made non-decreasing."""

    def test_hand_computed_sequence(self):
        # p = [0.01, 0.04, 0.03] sorted -> 0.01, 0.03, 0.04 with multipliers 3, 2, 1
        #   0.01 * 3 = 0.03
        #   0.03 * 2 = 0.06       (already above 0.03, so no clamping)
        #   0.04 * 1 = 0.04 -> raised to 0.06 to keep the sequence non-decreasing
        # back in input order: [0.03, 0.06, 0.06]
        assert list(np.round(holm([0.01, 0.04, 0.03]), 10)) == [0.03, 0.06, 0.06]

    def test_single_p_value_is_unchanged(self):
        assert holm([0.023])[0] == pytest.approx(0.023)

    def test_adjustment_never_lowers_a_p_value_and_never_exceeds_one(self):
        raw = [0.001, 0.2, 0.049, 0.6, 0.03]
        adjusted = holm(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))
        assert all(a <= 1.0 for a in adjusted)


class TestTanimoto:
    """Tanimoto coefficient of two binary vectors: |A and B| / |A or B|."""

    def test_hand_computed_on_bit_vectors(self):
        # a = 1 1 1 0 0, b = 0 1 1 1 0 -> intersection 2, union 4, T = 0.5
        a = np.array([[1, 1, 1, 0, 0]])
        b = np.array([[0, 1, 1, 1, 0]])
        assert tanimoto_to_training(a, b)[0, 0] == pytest.approx(0.5)

    def test_identical_vectors_give_one_and_disjoint_give_zero(self):
        a = np.array([[1, 0, 1, 0]])
        assert tanimoto_to_training(a, a)[0, 0] == pytest.approx(1.0)
        assert tanimoto_to_training(a, np.array([[0, 1, 0, 1]]))[0, 0] == pytest.approx(0.0)

    def test_a_single_shared_bit_out_of_three(self):
        # a = 1 1 0, b = 1 0 1 -> intersection 1, union 3, T = 1/3
        assert tanimoto_to_training(np.array([[1, 1, 0]]),
                                    np.array([[1, 0, 1]]))[0, 0] == pytest.approx(1.0 / 3.0)


class TestNearestSimilarity:
    """SIMILARITYNEAREST k of Sheridan et al. (2004, J. Chem. Inf. Comput. Sci. 44:1912-1928): the mean Tanimoto
    similarity of a query to its k most similar training molecules."""

    def test_mean_of_the_two_nearest_is_hand_computable(self):
        query = np.array([[1, 1, 1, 0]])
        train = np.array([[1, 1, 1, 0],    # identical            -> T = 1
                          [1, 1, 0, 0],    # 2 shared of 3 union  -> T = 2/3
                          [0, 0, 0, 1]])   # disjoint             -> T = 0
        # The two nearest are 1 and 2/3, so the mean is 5/6.
        assert nearest_similarity(query, train, k=2)[0] == pytest.approx(5.0 / 6.0)

    def test_k_one_is_the_single_best_match(self):
        query = np.array([[1, 1, 1, 0]])
        train = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
        assert nearest_similarity(query, train, k=1)[0] == pytest.approx(1.0)

    def test_excluding_self_skips_the_perfect_match(self):
        # Each molecule is its own nearest neighbour at T = 1; with exclude_self the first one must fall back to the
        # other, at 2 shared bits over 3 in the union.
        train = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        assert nearest_similarity(train, train, k=1, exclude_self=True)[0] == pytest.approx(2.0 / 3.0)


class TestMurckoScaffold:
    """Bemis and Murcko (1996, J. Med. Chem. 39:2887-2893): the framework left after removing the side chains."""

    def test_known_scaffolds(self):
        # Benzene is its own scaffold; toluene and ethylbenzene reduce to it once the side chain is stripped.
        assert list(murcko_scaffolds(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"])) == ["c1ccccc1"] * 3

    def test_an_acyclic_molecule_has_an_empty_scaffold(self):
        assert list(murcko_scaffolds(["CCCC"])) == [""]

    def test_two_rings_joined_by_a_linker_keep_the_linker(self):
        # The linker between two rings is part of the framework, unlike a terminal side chain.
        assert list(murcko_scaffolds(["c1ccccc1CCc1ccccc1"])) == ["c1ccc(CCc2ccccc2)cc1"]

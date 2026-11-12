"""Summary of the generalization cohort across its twelve targets (step 22).

Two things can go wrong here without raising. A macro name built from a target abbreviation can collide with another
one (JAK1 and JAK2 differ only in a digit, which a LaTeX name cannot hold), and the second-level test can be computed
on the wrong sign or the wrong units. Expected values are written out by hand.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load_step22():
    """Import the step by path, because its file name is not a valid module name."""
    path = os.path.join(ROOT, "scripts", "22_generalization_summary.py")
    spec = importlib.util.spec_from_file_location("step22", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step22 = _load_step22()


class TestMacroToken:
    @pytest.mark.parametrize("abbreviation, token", [
        ("AR", "Ar"), ("GHSR", "Ghsr"), ("JAK1", "JakOne"), ("JAK2", "JakTwo"),
        ("PIK3CA", "PikThreeCa"), ("CLK4", "ClkFour"), ("HRH1", "HrhOne"),
    ])
    def test_spells_digits_and_capitalizes_each_run(self, abbreviation, token):
        assert step22.macro_token(abbreviation) == token

    def test_tokens_of_the_cohort_are_distinct_and_letters_only(self):
        cohort = ["JAK1", "FXR", "AR", "GHSR", "CLK4", "GR", "ABL1", "GSK3", "KOR", "PIK3CA", "HRH1", "JAK2"]
        tokens = [step22.macro_token(a) for a in cohort]
        assert len(set(tokens)) == len(tokens)
        assert all(t.isalpha() for t in tokens)

    def test_rejects_an_abbreviation_with_nothing_to_name(self):
        with pytest.raises(ValueError):
            step22.macro_token("-")


class TestCohortTest:
    def test_counts_and_mean_follow_the_sign_of_the_advantage(self):
        result = step22.cohort_test([0.1, 0.2, -0.1, 0.4])
        assert result["n"] == 4
        assert result["n_classical"] == 3 and result["n_neural"] == 1
        assert result["mean"] == pytest.approx(0.15)

    def test_interval_is_the_t_interval_over_targets(self):
        # mean 0.15, sd 0.2081666, sem 0.1040833, t(0.975, 3) = 3.182446
        result = step22.cohort_test([0.1, 0.2, -0.1, 0.4])
        assert result["ci_low"] == pytest.approx(0.15 - 0.331239, abs=1e-5)
        assert result["ci_high"] == pytest.approx(0.15 + 0.331239, abs=1e-5)

    def test_all_positive_twelve_gives_the_smallest_exact_wilcoxon_p(self):
        # With twelve targets all on one side, the exact two-sided signed-rank p is 2 / 2**12.
        result = step22.cohort_test([0.01 * (i + 1) for i in range(12)])
        assert result["p_wilcoxon"] == pytest.approx(2 / 2 ** 12)

    def test_needs_two_targets(self):
        with pytest.raises(ValueError):
            step22.cohort_test([0.1])


class TestFormatP:
    def test_bound_below_the_fourth_decimal(self):
        assert step22.fmt_p(3e-5) == "< 0.0001"
        assert step22.fmt_p(0.00049) == "0.0005"

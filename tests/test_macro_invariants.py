"""Relations that must hold among the exported values, whatever the data say.

The macro files are written by different steps at different times, and the pipeline is routinely run in parts: a
`--only 19` here, a `--only figures` there, a branch into its own output directory. Nothing checked that the set left
on disk came from one coherent execution, which is the most likely way for the manuscript to end up quoting two runs
at once.

These are the identities and bounds that a coherent set satisfies: a partition that adds up, a ratio that matches the
two values it divides, a count that cannot exceed the set it is drawn from, a mean that lies inside its own interval.
None of them encodes a result. A test that asserted which model wins would turn a finding into an axiom, and the
first run that disagreed would be "fixed" by editing the test; so the rule is arithmetic and logic only, never the
sign of an effect.

Several relations cross files on purpose, because those are the ones that catch a stale macro: the calibration set of
step 05 is the hold-out of step 04, the Morgan fold cost in minutes exported by step 19 is the one in seconds exported
by step 09, and the benchmark score of step 09 must be one of the executions step 09.1 recorded.
"""
import glob
import math
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def v():
    """Every exported macro as a float where possible, keyed by name."""
    raw = {}
    for path in glob.glob(os.path.join(ROOT, "latex", "*_variables.tex")):
        with open(path, encoding="utf-8") as fh:
            raw.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fh.read()))
    out = {}
    for name, value in raw.items():
        try:
            out[name] = float(value.replace("$", "").replace("+", "").strip())
        except ValueError:
            continue
    return out


def stale(*names):
    return (f"{', '.join(names)} disagree; the macro files probably come from different runs. "
            f"Re-run the steps that write them (see MACROS.md) and refresh the catalogue.")


class TestPartitionsAddUp:
    def test_the_holdout_and_the_development_set_are_the_whole_data_set(self, v):
        assert v["ExpNDev"] + v["ExpNTest"] == v["NCompounds"], stale("ExpNDev", "ExpNTest", "NCompounds")

    def test_the_conformal_calibration_set_is_the_holdout(self, v):
        # Step 05 calibrates on the partition step 04 held out; two different numbers mean one of them is stale
        assert v["ScreenConformalNCal"] == v["ExpNTest"], stale("ScreenConformalNCal", "ExpNTest")

    def test_the_curated_set_is_the_one_the_benchmark_used(self, v):
        assert v["DataNMain"] == v["NCompounds"] == v["OutlierNMain"], stale("DataNMain", "NCompounds", "OutlierNMain")
        assert v["DataNRaw"] >= v["DataNMain"], "the curated set cannot be larger than the released table"
        assert v["OutlierNSens"] > v["OutlierNMain"], "the sensitivity benchmark keeps records the main one drops"

    def test_the_outer_folds_are_the_repeats_times_the_splits(self, v):
        assert v["TotalOuterFolds"] == v["OuterKFolds"] * v["OuterRepeats"], \
            stale("TotalOuterFolds", "OuterKFolds", "OuterRepeats")

    def test_the_candidates_are_the_top_ranked_plus_the_references(self, v):
        assert v["ScreenNCandidates"] == v["ScreenNTopML"] + v["ScreenNReference"], \
            stale("ScreenNCandidates", "ScreenNTopML", "ScreenNReference")


class TestCountsStayInsideTheirSet:
    def test_the_screening_funnel_only_narrows(self, v):
        assert v["ScreenNDomain"] <= v["ScreenNStructural"] <= v["ScreenNLibrary"], \
            stale("ScreenNDomain", "ScreenNStructural", "ScreenNLibrary")

    def test_the_subsets_of_the_candidates(self, v):
        for name in ("ScreenNCandidatesInDomain", "ScreenNCandidatesInTrain"):
            assert v[name] <= v["ScreenNCandidates"], stale(name, "ScreenNCandidates")
        assert v["ScreenNCandidatesInDomainNovel"] <= v["ScreenNCandidatesInDomain"], \
            stale("ScreenNCandidatesInDomainNovel", "ScreenNCandidatesInDomain")

    def test_a_percentage_is_a_percentage(self, v):
        for name in ("MorganStability", "ScreenConformalCoverage"):
            assert 0 <= v[name] <= 100, f"{name} = {v[name]} is not a percentage"

    def test_the_scaffold_folds_have_a_range(self, v):
        assert v["ScaffoldFoldSizeMin"] <= v["ScaffoldFoldSizeMax"], \
            stale("ScaffoldFoldSizeMin", "ScaffoldFoldSizeMax")


class TestDerivedQuantitiesMatchTheirParts:
    def test_the_morgan_fold_cost_is_the_same_in_both_units(self, v):
        # Step 19 exports the minutes, step 09 the seconds; the two are the same measurement
        assert v["ClassicalMorganFoldMinutes"] == pytest.approx(v["ClassicalMorganFoldSec"] / 60, abs=0.05), \
            stale("ClassicalMorganFoldMinutes", "ClassicalMorganFoldSec")

    def test_the_morgan_fold_lies_between_the_cheapest_and_the_dearest(self, v):
        assert v["ClassicalFoldSecMin"] <= v["ClassicalMorganFoldSec"] <= v["ClassicalFoldSecMax"], \
            stale("ClassicalFoldSecMin", "ClassicalMorganFoldSec", "ClassicalFoldSecMax")

    def test_the_totals_are_the_per_fold_cost_times_the_folds(self, v):
        assert v["ClassicalMorganTotalHours"] == pytest.approx(
            v["ClassicalMorganFoldSec"] * v["TotalOuterFolds"] / 3600, abs=0.1), \
            stale("ClassicalMorganTotalHours", "ClassicalMorganFoldSec", "TotalOuterFolds")
        assert v["GnnTotalMin"] == pytest.approx(v["GnnFoldSecMean"] * v["TotalOuterFolds"] / 60, abs=1), \
            stale("GnnTotalMin", "GnnFoldSecMean", "TotalOuterFolds")

    @pytest.mark.parametrize("factor, minutes", [("ChemeleonCostFactor", "ChemeleonFoldMinutes"),
                                                 ("DmpnnEnsCostFactor", "DmpnnEnsFoldMinutes"),
                                                 ("DmpnnRdkitCostFactor", "DmpnnRdkitFoldMinutes"),
                                                 ("MlpEcfpCostFactor", "MlpEcfpFoldMinutes")])
    def test_every_cost_factor_is_its_own_ratio(self, v, factor, minutes):
        # The tolerance covers the rounding of the three values, each exported to one or two decimals
        assert v[factor] == pytest.approx(v[minutes] / v["ClassicalMorganFoldMinutes"], abs=0.06), \
            stale(factor, minutes, "ClassicalMorganFoldMinutes")

    def test_the_search_size_is_the_combinations_times_the_inner_folds(self, v):
        assert v["GnnSearchFits"] == v["NArchitectureCombos"] * v["InnerKFolds"], \
            stale("GnnSearchFits", "NArchitectureCombos", "InnerKFolds")
        assert v["NInnerFitsPerOuterFold"] == v["GnnSearchFits"], \
            stale("NInnerFitsPerOuterFold", "GnnSearchFits")

    def test_the_number_of_combinations_is_the_sum_of_the_binomials(self, v):
        expected = sum(math.comb(6, k) for k in range(1, int(v["MaxComboSize"]) + 1))
        assert v["NArchitectureCombos"] == expected, stale("NArchitectureCombos", "MaxComboSize")


class TestIntervalsContainWhatTheyDescribe:
    def test_the_benchmark_mean_lies_inside_its_own_interval(self, v):
        assert v["ChemPropDMPNNFoldCILow"] < v["ChemPropDMPNNFoldRTwoMean"] < v["ChemPropDMPNNFoldCIHigh"], \
            stale("ChemPropDMPNNFoldCILow", "ChemPropDMPNNFoldRTwoMean", "ChemPropDMPNNFoldCIHigh")

    def test_the_reproducibility_range_is_its_own_extremes(self, v):
        assert v["GnnRepeatRTwoRange"] == pytest.approx(v["GnnRepeatRTwoMax"] - v["GnnRepeatRTwoMin"], abs=0.001), \
            stale("GnnRepeatRTwoRange", "GnnRepeatRTwoMax", "GnnRepeatRTwoMin")

    def test_the_current_benchmark_is_one_of_the_recorded_executions(self, v):
        # Step 09 appends its score to the log that step 09.1 summarizes, so the current mean cannot fall outside it
        assert v["GnnRepeatRTwoMin"] <= v["ChemPropDMPNNFoldRTwoMean"] <= v["GnnRepeatRTwoMax"], \
            stale("ChemPropDMPNNFoldRTwoMean", "GnnRepeatRTwoMin", "GnnRepeatRTwoMax")

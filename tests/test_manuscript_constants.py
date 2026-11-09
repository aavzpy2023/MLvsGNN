"""The configuration values the manuscript states in words, pinned to the code that actually uses them.

Most numbers in the text come from `latex/*.tex`, which the pipeline writes, and step 11 fails if one of those macros
is missing. It cannot see a number written by hand: for the checker such a number does not exist, so a literal keeps
reading correctly long after the code behind it changed. These are the literals that remain, and they are all
configuration rather than measurement: a tree count, a schedule, a grid, a noise level. Re-running the pipeline does
not move them, but editing `paths_config.py`, `gnn_utils.py` or a step would silently falsify the sentence that
describes them.

Each test builds the expected phrase from the live value and looks for it in the section, so a changed constant fails
here with the file to edit. Rewording one of these sentences also fails, which is the intended cost: the phrase is the
anchor. A value that gained a macro belongs in the pipeline's export instead, and the last test keeps those two from
disagreeing.
"""
import os
import re

import pytest

from nb_stats import EQUIVALENCE_MARGIN_MAE, EQUIVALENCE_MARGIN_R2
from paths_config import CFG, NOISE_LEVELS, TEST_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(ROOT, "sections")
LATEX_DIR = os.path.join(ROOT, "latex")


def section(name):
    with open(os.path.join(SECTIONS, name), encoding="utf-8") as fh:
        return fh.read()


def macros():
    """Every macro the pipeline exported, as {name: value}."""
    out = {}
    for file_name in os.listdir(LATEX_DIR):
        if file_name.endswith("_variables.tex"):
            with open(os.path.join(LATEX_DIR, file_name), encoding="utf-8") as fh:
                out.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fh.read()))
    return out


def gnn_config():
    """gnn_utils imports chemprop, which is slow and optional, so it is loaded only by the tests that need it."""
    return pytest.importorskip("gnn_utils")


class TestClassicalProtocol:
    def test_the_number_of_trees_is_the_configured_one(self):
        assert f"the tree ensembles used {CFG['N_ESTIMATORS_TREES']} trees" in section("02materials.tex")

    def test_the_holdout_fraction_is_the_configured_one(self):
        phrase = f"{TEST_SIZE * 100:.0f}\\% hold-out"
        assert phrase in section("02materials.tex") and phrase in section("03Results.tex")

    def test_the_outlier_fence_factor_is_the_one_used_to_build_the_data(self):
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("step00", os.path.join(ROOT, "scripts", "00_prepare_dataset.py"))
        step00 = module_from_spec(spec)
        spec.loader.exec_module(step00)
        text = section("02materials.tex")
        assert f"Q1 $-$ {step00.IQR_FACTOR}\\,IQR" in text and f"Q3 $+$ {step00.IQR_FACTOR}\\,IQR" in text

    def test_the_augmentation_noise_levels_are_the_published_ones(self):
        text = section("02materials.tex")
        for sigma in NOISE_LEVELS:
            assert f"$\\sigma = {sigma}$" in text, f"the manuscript does not state the noise level {sigma}"

    def test_the_combination_size_limit_matches_the_search(self):
        # The text spells out the three binomial terms of the search; a fourth would change the count it reports
        assert CFG["MAX_COMBO_SIZE"] == 3
        assert r"\binom{6}{1} + \binom{6}{2} + \binom{6}{3}" in section("02materials.tex")


class TestNeuralProtocol:
    def test_the_training_schedule_is_the_one_the_code_runs(self):
        epochs = gnn_config().EPOCHS
        text = section("02materials.tex")
        assert f"a fixed number of {epochs} epochs" in text
        assert text.count(f"{epochs}-epoch schedule") == 2, "both the tuned variant and CheMeleon state the schedule"

    def test_the_default_architecture_is_the_one_the_code_builds(self):
        default = gnn_config().DEFAULT_CONFIG
        dropout = "no dropout" if default["dropout"] == 0 else f"dropout {default['dropout']}"
        phrase = f"(hidden size {default['d_h']}, depth {default['depth']}, {dropout})"
        assert section("02materials.tex").count(phrase) == 2, f"the manuscript no longer states {phrase} twice"

    def test_the_tuned_grid_is_the_one_the_code_searches(self):
        utils = gnn_config()
        sizes, depths, dropouts = utils.HIDDEN_SIZES, utils.DEPTHS, utils.DROPOUTS
        phrase = (f"hidden size {sizes[0]} or {sizes[1]}, message-passing depth {depths[0]} or {depths[1]}, "
                  f"dropout {dropouts[0]:.0f} or {dropouts[1]}")
        assert phrase in section("02materials.tex")

    def test_the_inner_validation_fraction_is_the_one_the_code_holds_out(self):
        fraction = gnn_config().INNER_VALIDATION_FRACTION
        text = section("02materials.tex")
        assert f"{fraction * 100:.0f}\\% of the compounds were held out at random" in text
        assert f"trained on the remaining {100 - fraction * 100:.0f}\\%" in text


class TestExportedValuesAgreeWithTheCode:
    def test_the_equivalence_margins_are_exported_as_the_statistics_module_defines_them(self):
        exported = macros()
        assert exported["EquivMarginRTwo"] == f"{EQUIVALENCE_MARGIN_R2}"
        assert exported["EquivMarginMae"] == f"{EQUIVALENCE_MARGIN_MAE}"

    @pytest.mark.parametrize("literal, macro", [("pm 0.05", "EquivMarginRTwo"), ("2048-bit", "MorganNFeaturesTotal")])
    def test_a_value_with_a_macro_is_not_also_written_by_hand(self, literal, macro):
        offenders = [f for f in sorted(os.listdir(SECTIONS)) if f.endswith(".tex") and literal in section(f)]
        assert not offenders, f"{offenders} write {literal!r} by hand; use \\{macro} so the text follows the pipeline"

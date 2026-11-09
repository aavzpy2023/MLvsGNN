"""Every number the manuscript states comes from a macro, and every macro agrees with the code behind it.

Two things have to hold for a sentence about a setting to stay true. The macro that carries the value must match the
code that uses it, which is what the first group checks: it compares `latex/config_variables.tex` against
`paths_config`, `gnn_utils` and the steps that own a constant of their own. And the sentence must not restate the
value as a literal beside the macro, which is what the second group forbids.

Neither group reads the prose to validate a number, by design. An earlier version of these tests looked for phrases
built from the live values, which pinned the numbers but also broke whenever a sentence was reworded, and rewording is
something the authors must stay free to do. The anchoring now runs the other way: the text may change at will as long
as the number in it is a macro, and a value that has no macro is caught by the literal check rather than by a phrase.
"""
import ast
import os
import re

import pytest

from nb_stats import EQUIVALENCE_MARGIN_MAE, EQUIVALENCE_MARGIN_R2
from paths_config import CFG, NOISE_LEVELS, RANDOM_STATE, TEST_SIZE
from qsartools.featurization import morgan_fingerprints

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(ROOT, "sections")
LATEX_DIR = os.path.join(ROOT, "latex")
SCRIPTS = os.path.join(ROOT, "scripts")


def macros(file_name=None):
    """Exported macros as {name: value}: of one file, or of every macro file when none is given."""
    files = [file_name] if file_name else [f for f in os.listdir(LATEX_DIR) if f.endswith("_variables.tex")]
    out = {}
    for name in files:
        with open(os.path.join(LATEX_DIR, name), encoding="utf-8") as fh:
            out.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fh.read()))
    return out


def assigned_literal(script, name):
    """The value assigned to `name` at the top level of a step, read without importing it."""
    with open(os.path.join(SCRIPTS, script), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=script)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not assigned at the top level of {script}")


@pytest.fixture(scope="module")
def config():
    exported = macros("config_variables.tex")
    assert exported, "latex/config_variables.tex is empty; run `python run_pipeline.py --only 00.1`"
    return exported


class TestExportedConfigurationMatchesTheCode:
    def test_the_seed_and_the_tree_count(self, config):
        assert config["RandomSeed"] == str(RANDOM_STATE)
        assert config["NTrees"] == str(CFG["N_ESTIMATORS_TREES"])

    def test_the_holdout_fraction(self, config):
        assert config["HoldoutPct"] == f"{TEST_SIZE * 100:.0f}"
        assert config["GnnInnerTrainPct"] == str(100 - int(config["GnnInnerValPct"]))

    def test_the_constants_that_live_inside_a_step(self, config):
        assert config["IqrFactor"] == str(assigned_literal("00_prepare_dataset.py", "IQR_FACTOR"))
        assert config["ClipMargin"] == f"{assigned_literal('01_nested_cv_stacking.py', 'CLIP_MARGIN'):.0f}"
        assert config["DmpnnEnsembleSeeds"] == str(assigned_literal("19_gnn_strong_baselines.py", "N_ENSEMBLE"))

    def test_the_fingerprint_radius_is_the_featurizer_default(self, config):
        import inspect
        assert config["MorganRadius"] == str(inspect.signature(morgan_fingerprints).parameters["radius"].default)

    def test_the_augmentation_noise_levels(self, config):
        assert config["NoiseLevelHigh"] == str(max(NOISE_LEVELS))
        assert config["NoiseLevelLow"] == str(min(NOISE_LEVELS))

    def test_the_neural_schedule_and_architecture(self, config):
        utils = pytest.importorskip("gnn_utils")
        assert config["GnnEpochs"] == str(utils.EPOCHS)
        assert config["GnnHiddenSize"] == str(utils.DEFAULT_CONFIG["d_h"])
        assert config["GnnDepth"] == str(utils.DEFAULT_CONFIG["depth"])
        assert config["GnnDropoutWord"] == ("no dropout" if utils.DEFAULT_CONFIG["dropout"] == 0
                                            else f"dropout {utils.DEFAULT_CONFIG['dropout']}")

    def test_the_tuned_grid(self, config):
        utils = pytest.importorskip("gnn_utils")
        assert config["GnnGridHiddenSizes"] == " or ".join(str(s) for s in utils.HIDDEN_SIZES)
        assert config["GnnGridDepths"] == " or ".join(str(d) for d in utils.DEPTHS)
        assert config["GnnGridDropouts"] == " or ".join(f"{p:g}" for p in utils.DROPOUTS)
        # The count the manuscript quotes is the size of the grid, not of the three axes separately
        assert len(utils.GRID) == len(utils.HIDDEN_SIZES) * len(utils.DEPTHS) * len(utils.DROPOUTS)

    def test_the_equivalence_margins(self):
        exported = macros()
        assert exported["EquivMarginRTwo"] == f"{EQUIVALENCE_MARGIN_R2}"
        assert exported["EquivMarginMae"] == f"{EQUIVALENCE_MARGIN_MAE}"


class TestNoValueIsAlsoWrittenByHand:
    LITERALS = [("pm 0.05", "EquivMarginRTwo"), ("2048-bit", "MorganNFeaturesTotal"),
                ("200 trees", "NTrees"), ("100 epochs", "GnnEpochs"), ("100-epoch", "GnnEpochs"),
                ("hidden size 300", "GnnHiddenSize"), ("seed 42", "RandomSeed"), ("1.5\\,IQR", "IqrFactor"),
                ("15\\% hold-out", "HoldoutPct")]

    @pytest.mark.parametrize("literal, macro", LITERALS)
    def test_a_value_with_a_macro_is_not_restated_as_a_literal(self, literal, macro):
        offenders = []
        for name in sorted(f for f in os.listdir(SECTIONS) if f.endswith(".tex")):
            with open(os.path.join(SECTIONS, name), encoding="utf-8") as fh:
                if literal in fh.read():
                    offenders.append(name)
        assert not offenders, f"{offenders} write {literal!r} by hand; use \\{macro} so the text follows the code"

    def test_every_forbidden_literal_names_an_exported_macro(self):
        exported = macros()
        missing = [m for _, m in self.LITERALS if m not in exported]
        assert not missing, f"{missing} are not exported, so forbidding their literal would leave no way to state it"

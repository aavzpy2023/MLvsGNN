"""Every generated table fills the text width, in every layout.

The manuscript is typeset by three classes of different text width (the reading copy, Elsevier's cas-sc and Springer's
sn-jnl at 372 pt, the narrowest) and the supplement by a fourth. A table therefore cannot be given a width by the
template: it has to carry one. Two constructions do that, tabularx where a prose column can absorb the slack by
wrapping and tabular* with a stretchable intercolumn skip where every column is short, and both measure exactly
\\textwidth wherever they are placed.

What is excluded matters as much: no box may be put around a tabular. sn-jnl wraps a table in threeparttable, and a
\\resizebox there fails with "Missing \\endgroup inserted"; \\resizebox also rescales the type, so a table set that way
no longer matches the body text. These tests pin the construction in the generated files, check that the generator
scripts still agree with what is on disk, and keep the boxes from coming back into the templates.
"""
import ast
import os
import re

import pytest

from qsartools import latex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_DIR = os.path.join(ROOT, "latex")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ROOTS = ["paper_reading.tex", "paper_ML_GNN.tex", "paper_springer.tex", "supplementary_ML_GNN.tex"]

OPEN_RE = re.compile(r"\\begin\{(tabularx|tabular\*)\}\{\\textwidth\}\{(?:@\{\\extracolsep\{\\fill\}\})?(.*)\}")

# The constant each generator script defines for a table, against the file that table is written to. A script whose
# constant drifts from the file on disk means the next pipeline run would silently undo the conversion.
SCRIPT_CONSTANTS = {
    "02_statistical_tests.py": {"PAIRWISE_COLS": "pairwise_nb_table.tex"},
    "03_y_randomization.py": {"YRANDOM_COLS": "yrandom_table.tex"},
    "04.1_build_figures.py": {"ABLATION_COLS": "ablation_table.tex"},
    "05.1_build_figures.py": {"FDA_COLS": "fda_candidates_table.tex"},
    "08.1_build_figures.py": {"ADMET_COLS": "admet_table.tex"},
    "09.1_build_figures.py": {"COST_COLS": "cost_table.tex"},
    "10_outlier_sensitivity.py": {"OUTLIER_COLS": "outlier_table.tex"},
    "13_scaffold_split.py": {"SCAFFOLD_PAIRWISE_COLS": "scaffold_pairwise_table.tex",
                             "SCAFFOLD_COLS": "scaffold_table.tex"},
    "18_karimova_audit.py": {"LADDER_COLS": "leakage_ladder_table.tex"},
    "19_gnn_strong_baselines.py": {"GNN_PAIRWISE_COLS": "gnn_pairwise_table.tex",
                                   "STRONG_PAIRWISE_COLS": "strong_pairwise_table.tex",
                                   "TUNED_CONFIG_COLS": "gnn_tuned_config_table.tex",
                                   "STRONG_COLS": "strong_baselines_table.tex"},
}


def generated_tables():
    if not os.path.isdir(LATEX_DIR):
        return []
    return sorted(f for f in os.listdir(LATEX_DIR) if f.endswith("_table.tex"))


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def colspec_of(filename):
    """The column specification the generated file on disk actually carries."""
    match = OPEN_RE.search(read(os.path.join(LATEX_DIR, filename)))
    return match.group(2) if match else None


@pytest.fixture(scope="module")
def tables():
    found = generated_tables()
    if not found:
        pytest.skip("no generated tables on disk (the pipeline has not been run)")
    return found


class TestTheGeneratedFiles:
    def test_every_table_declares_the_text_width(self, tables):
        for name in tables:
            match = OPEN_RE.search(read(os.path.join(LATEX_DIR, name)))
            assert match is not None, f"{name} is not a full-width tabularx or tabular*"

    def test_every_table_closes_the_environment_it_opened(self, tables):
        for name in tables:
            text = read(os.path.join(LATEX_DIR, name))
            env = OPEN_RE.search(text).group(1)
            assert text.rstrip().endswith("\\end{" + env + "}"), name
            # A plain \end{tabular} left behind would close the wrong environment.
            assert "\\end{tabular}" not in text, name

    def test_tabularx_is_used_exactly_when_a_column_can_wrap(self, tables):
        for name in tables:
            spec = colspec_of(name)
            env = OPEN_RE.search(read(os.path.join(LATEX_DIR, name))).group(1)
            assert env == ("tabularx" if "X" in spec else "tabular*"), name

    def test_no_table_is_wrapped_in_a_box(self, tables):
        # \resizebox is what sn-jnl rejects and what rescales the type away from the body size.
        for name in tables:
            text = read(os.path.join(LATEX_DIR, name))
            for box in ("resizebox", "scalebox", "adjustbox"):
                assert box not in text, f"{name} uses \\{box}"

    def test_the_column_count_survived_the_conversion(self, tables):
        # A spec that lost or gained a column would misalign every row; count the alignment tabs of a data row.
        for name in tables:
            text = read(os.path.join(LATEX_DIR, name))
            spec = colspec_of(name)
            declared = len(re.sub(r">\{[^}]*\}", "", spec))
            rows = [ln for ln in text.splitlines() if ln.rstrip().endswith(r"\\") and "multicolumn" not in ln]
            assert rows, name
            assert max(ln.count("&") for ln in rows) + 1 == declared, name


class TestTheGeneratorsStillAgree:
    """The files above are build products. These tests read the constant each script would use next time it runs."""

    @staticmethod
    def _constants(script):
        """Evaluate the module-level <NAME>_COLS assignments of a script without importing it."""
        tree = ast.parse(read(os.path.join(SCRIPTS_DIR, script)))
        env = {"latex": latex}
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.endswith("_COLS"):
                    out[target.id] = eval(compile(ast.Expression(node.value), "<spec>", "eval"), env)
        return out

    @pytest.mark.parametrize("script", sorted(SCRIPT_CONSTANTS))
    def test_each_script_defines_the_constants_its_tables_need(self, script):
        assert set(self._constants(script)) >= set(SCRIPT_CONSTANTS[script])

    @pytest.mark.parametrize("script", sorted(SCRIPT_CONSTANTS))
    def test_each_constant_matches_the_file_the_script_writes(self, script):
        constants = self._constants(script)
        for const, filename in SCRIPT_CONSTANTS[script].items():
            if not os.path.exists(os.path.join(LATEX_DIR, filename)):
                pytest.skip(f"{filename} has not been generated")
            assert colspec_of(filename) == constants[const], f"{script}:{const} drifted from {filename}"

    def test_no_script_writes_a_bare_tabular_any_more(self):
        for name in sorted(os.listdir(SCRIPTS_DIR)):
            if name.endswith(".py"):
                assert r"\begin{tabular}" not in read(os.path.join(SCRIPTS_DIR, name)), name


class TestTheTemplates:
    """The width now belongs to the table, so no template may add one back."""

    @pytest.mark.parametrize("root", ROOTS)
    def test_the_class_can_typeset_a_tabularx(self, root):
        text = read(os.path.join(ROOT, root))
        assert "\\usepackage{tabularx}" in text, root
        # >{\raggedright\arraybackslash} is an array package construction.
        assert "\\usepackage{array}" in text, root

    @pytest.mark.parametrize("root", ROOTS)
    def test_no_generated_table_is_boxed_at_its_call_site(self, root):
        text = read(os.path.join(ROOT, root))
        for match in re.finditer(r"\\(resizebox|scalebox|adjustbox)[^\n]*", text):
            assert "latex/" not in match.group(0), f"{root}: {match.group(0)[:80]}"

    def test_projtable_passes_the_table_through_untouched(self):
        # The three manuscript layouts share the sections, so \projtable is where a layout could still box a table.
        for root in ROOTS[:3]:
            text = read(os.path.join(ROOT, root))
            match = re.search(r"\\newcommand\{\\projtable\}\[1\]\{(.*)\}", text)
            assert match is not None, root
            assert match.group(1) == r"\input{#1}", f"{root}: {match.group(1)}"

    def test_the_sections_never_name_a_boxing_command_for_a_table(self):
        sections = os.path.join(ROOT, "sections")
        for name in sorted(os.listdir(sections)):
            if not name.endswith(".tex"):
                continue
            text = read(os.path.join(sections, name))
            for match in re.finditer(r"\\(resizebox|scalebox|adjustbox)[^\n]*", text):
                assert "latex/" not in match.group(0), f"{name}: {match.group(0)[:80]}"

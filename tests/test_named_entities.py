"""The names the prose uses must be the ones the current results carry.

Macros keep the numbers honest, but not the words around them. When the screening was re-run with a new feature mask,
the ranking changed and the text went on naming compounds that had dropped out of it; nothing failed, because no
number was wrong. The same can happen to the architecture the text calls deployed, or to a compound described as
belonging to the training set.

So these check names against the files that define them, and nothing else. They do not assert which compound should
win or how well a model should do: that is a result, and a test that fixed it would invite editing the test instead of
reading the run. What they assert is that every compound the prose names is still in the table the prose is describing.
"""
import os
import re

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(ROOT, "sections")
RESULTS = os.path.join(ROOT, "results")
CANDIDATES = os.path.join(RESULTS, "05_FDA_Candidates_For_Docking.csv")
SELECTION = os.path.join(RESULTS, "01_architecture_selection_frequency.csv")

# Compounds the prose may name, as they appear in the text. Each must still be in the candidates table.
NAMED_IN_PROSE = ("trimetrexate", "methotrexate", "pralatrexate", "triamterene", "pyrimethamine", "trimethoprim",
                  "bisacodyl", "folic acid")


def prose(*names):
    out = []
    for name in names:
        with open(os.path.join(SECTIONS, name), encoding="utf-8") as fh:
            out.append(re.sub(r"(?<!\\)%.*$", "", fh.read(), flags=re.M))
    return "\n".join(out)


@pytest.fixture(scope="module")
def candidates():
    if not os.path.isfile(CANDIDATES):
        pytest.skip("the screening has not been run in this working copy")
    return pd.read_csv(CANDIDATES)


def test_every_compound_the_prose_names_is_still_a_candidate(candidates):
    text = prose("03Results.tex", "04Discussion.tex", "05Conclusions.tex", "00abstract.tex").lower()
    listed = " | ".join(candidates["Name"].astype(str)).lower()
    named = [c for c in NAMED_IN_PROSE if re.search(rf"\b{re.escape(c)}\b", text)]
    assert named, "no compound of NAMED_IN_PROSE appears in the text; the list is out of date"
    missing = [c for c in named if c not in listed]
    assert not missing, (f"{missing} are named in the text but are not in the current candidates table. The screening "
                         f"was re-run and the ranking changed: rewrite the sentences that name them.")


def test_the_compounds_named_as_leading_are_the_top_ranked(candidates):
    """The text says the three antifolates lead by absolute potency; that is a claim about the table's order."""
    top = candidates.nlargest(5, "pIC50_pred")["Name"].astype(str).str.lower().tolist()
    for compound in ("trimetrexate", "methotrexate", "pralatrexate"):
        assert any(compound in name for name in top), \
            (f"the text presents {compound} as leading the ranking, but it is no longer in the five highest "
             f"predictions ({top}); the paragraph needs rewriting.")


def test_the_architecture_called_deployed_is_the_one_most_often_selected():
    """\\MorganBestArchitecture is quoted as the deployed combination; the selection log must still agree."""
    if not os.path.isfile(SELECTION):
        pytest.skip("the nested cross-validation has not been run in this working copy")
    morgan = pd.read_csv(SELECTION).query("Mode == 'morgan'").sort_values("times_selected", ascending=False)
    most_frequent = morgan.iloc[0]["Model"]
    with open(os.path.join(ROOT, "latex", "paper_variables.tex"), encoding="utf-8") as fh:
        exported = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", fh.read()))
    assert exported["MorganBestArchitecture"] == most_frequent, \
        (f"the selection log now prefers {most_frequent} while the macro says "
         f"{exported['MorganBestArchitecture']}; the deployed model and the text disagree.")

    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from paths_config import DEPLOYED_ARCHITECTURE
    assert "+".join(DEPLOYED_ARCHITECTURE) == most_frequent, \
        (f"DEPLOYED_ARCHITECTURE is {'+'.join(DEPLOYED_ARCHITECTURE)} but the inner loop now prefers "
         f"{most_frequent}; the screening would deploy a model the benchmark no longer selects.")


def test_a_compound_called_a_training_compound_really_is_one(candidates):
    """The text says pyrimethamine, trimethoprim and methotrexate belong to the training partition."""
    if "In_train" not in candidates.columns:
        pytest.skip("the candidates table predates the training-membership column")
    membership = {str(n).lower(): bool(t) for n, t in zip(candidates["Name"], candidates["In_train"])}
    for compound in ("pyrimethamine", "trimethoprim", "methotrexate"):
        rows = [v for name, v in membership.items() if compound in name]
        assert rows and all(rows), \
            f"the text calls {compound} a training compound, which the candidates table no longer says"

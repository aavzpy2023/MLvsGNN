"""No number may enter the prose by hand unless it is listed here with a reason.

Step 11 fails when a macro is missing, but a number typed into a sentence is invisible to it: for the checker it does
not exist, so it keeps reading correctly long after the value behind it changed. That is how every drift this project
has had got in. This closes the door: the six prose sections are scanned for numeric literals, and anything not in
ALLOWED fails with the sentence around it.

The test does not care how a sentence is written, only that its numbers come from macros, so rewording is free. To add
a literal on purpose, add it to ALLOWED with the reason it cannot be a macro. Four entries are marked KNOWN GAP: they
are measurements that should be macros and are not yet, and they are listed so that the debt is visible rather than
silently tolerated.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(ROOT, "sections")
PROSE = ("00abstract.tex", "01Introduction.tex", "02materials.tex", "03Results.tex", "04Discussion.tex",
         "05Conclusions.tex")

# A literal, and why it is not a macro. Anything absent from this mapping fails the test.
ALLOWED = {
    # Confidence and significance levels: conventions of the test, not measurements of this data set
    "90": "coverage of the confidence intervals of the paired comparisons",
    "95": "coverage of the bootstrap intervals of the ladder and the benchmark",
    "0.05": "the significance level, which is not the equivalence margin (that one is a macro)",
    "2000": "bootstrap resamples of the hold-out ablation",
    # Criteria and hyperparameters inherited from a source rather than measured here
    "1.5": "log units of disagreement above which the original study discarded an assay",
    "0.30": "the ligand efficiency commonly taken as a lower bound, from Hopkins et al.",
    "5": "repeats of the permutation importance, and the k of the k-nearest-neighbours regressor",
    "10": "the C of the support vector regressor",
    "8": "an illustrative potency in prose ('a compound predicted at 8'), not a measured value",
    # Structure of the protocol, fixed by the design of the cross-validation
    "1": "the unit inside the formulas: 1/J of the corrected variance, the 1/4 ratio, 1 minus a similarity",
    "4": "the n_test/n_train ratio of five-fold cross-validation, written as 1/4",
    "6": "the six regressor families of the architecture search, in the sum of the binomials",
    "15": "a term of that sum",
    "20": "a term of that sum",
    "2": "warm-up epochs of Chemprop's default learning-rate schedule",
    "25": "the outer folds, in two figure captions that name them literally",
    # The workstation, described once in the environment subsection
    "26.04": "the Ubuntu release of the workstation",
    "2.4": "the clock of the workstation in GHz",
    "24": "physical cores of the workstation",
    "48": "threads of the workstation",
    "247": "RAM of the workstation in GB",
    # KNOWN GAP: measured durations still written by hand. They move to macros derived from the master logs when the
    # generalization section is written, which is also when the total changes.
    "19": "KNOWN GAP: measured hours of the y-randomization test",
    "46": "KNOWN GAP: measured hours of the cheapest nested cross-validation run",
    "57": "KNOWN GAP: measured hours of the dearest nested cross-validation run",
    "0.2": "KNOWN GAP: approximate width of the hold-out intervals in R^2 units",
}

NUMBER = re.compile(r"(?<![\w.\\\-])\d+(?:\.\d+)?(?![\w.])")
COMMENT = re.compile(r"(?<!\\)%.*$", re.M)
# Digits that belong to the typesetting rather than to the content: exponents and subscripts ($R^2$, $pIC_{50}$,
# $10^{-9}$), the arguments of \binom, and LaTeX lengths (p{0.35\textwidth}, width=0.8\textwidth)
STRUCTURAL = (re.compile(r"[\^_]\{?$"), re.compile(r"^\s*\}?\s*\\(?:text|line)width"),
              re.compile(r"\\binom\{?$"), re.compile(r"(?:width|height)\s*=\s*$"),
              re.compile(r"\\(?:text|line)width\}?\{?$"))


def literals_in(name):
    """[(literal, context)] of every hand-written number in a prose section, typesetting digits excluded."""
    with open(os.path.join(SECTIONS, name), encoding="utf-8") as fh:
        text = COMMENT.sub("", fh.read())
    text = re.sub(r"\\(?:texttt|url|path)\{[^}]*\}", " ", text)
    text = re.sub(r"\\binom\{[^}]*\}\{[^}]*\}", " ", text)
    out = []
    for match in NUMBER.finditer(text):
        before, after = text[max(0, match.start() - 24):match.start()], text[match.end():match.end() + 24]
        if any(p.search(before) for p in STRUCTURAL[:1] + STRUCTURAL[2:]) or STRUCTURAL[1].match(after):
            continue
        context = " ".join(text[max(0, match.start() - 60):match.end() + 40].split())
        out.append((match.group(0), context))
    return out


@pytest.mark.parametrize("name", PROSE)
def test_no_literal_outside_the_allowed_list(name):
    unexpected = [(lit, ctx) for lit, ctx in literals_in(name) if lit not in ALLOWED]
    assert not unexpected, "\n".join(
        [f"{name} writes numbers by hand that no macro carries:"]
        + [f"  {lit!r} in: …{ctx}…" for lit, ctx in unexpected]
        + ["Export the value from the step that measures it and cite the macro, or, if it genuinely cannot be one, "
           "add it to ALLOWED in this test with the reason."])


def test_the_allowed_list_has_no_dead_entries():
    """A literal that no longer appears has been fixed; remove it so the list keeps meaning something."""
    present = {lit for name in PROSE for lit, _ in literals_in(name)}
    dead = sorted(set(ALLOWED) - present)
    assert not dead, f"{dead} are allowed but no longer appear in the prose; drop them from ALLOWED"


def test_every_known_gap_is_still_a_gap():
    """The four measurements written by hand are tracked here, so closing one is noticed and the entry removed."""
    gaps = sorted(lit for lit, reason in ALLOWED.items() if reason.startswith("KNOWN GAP"))
    assert gaps == ["0.2", "19", "46", "57"], \
        f"the tracked gaps changed to {gaps}; update this test and the reason in ALLOWED"

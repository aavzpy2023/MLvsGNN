"""Where the manuscript's source lives, and what each directory is allowed to hold.

The manuscript is one body of text typeset by four documents: a reading copy, two journal layouts and the supplement.
That only works if the content exists once. Two files carry it, sections/manuscript_body.tex listing the sections in
order and sections/manuscript_variables.tex listing the macro files the pipeline writes, and every document reads
those instead of repeating the lists. A document that repeats them drifts silently: it keeps compiling while a
section or a new step's macros are missing from it alone.

Both files are written by hand, so neither may sit in latex/, which holds nothing but build products. These tests pin
that split, and check that the two lists still match what is on disk.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_DIR = os.path.join(ROOT, "latex")
SECTIONS_DIR = os.path.join(ROOT, "sections")
BODY = os.path.join(SECTIONS_DIR, "manuscript_body.tex")
VARIABLES = os.path.join(SECTIONS_DIR, "manuscript_variables.tex")
MANUSCRIPTS = ["paper_reading.tex", "paper_ML_GNN.tex", "paper_springer.tex"]
DOCUMENTS = MANUSCRIPTS + ["supplementary_ML_GNN.tex"]

INPUT_RE = re.compile(r"\\input\{([^}]*)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.M)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def inputs_of(path):
    """Every path a file \\input's: comments out (several generated files document their own use in one), and the
    #1 of a \\newcommand definition out, since that is a parameter and not a file."""
    return [p for p in INPUT_RE.findall(COMMENT_RE.sub("", read(path))) if not p.startswith("#")]


class TestTheGeneratedDirectory:
    def test_nothing_in_latex_pulls_in_another_file(self):
        # A generated macro file or table never includes anything; an \input there means a hand-written aggregator
        # has been dropped into a directory of build products, where a regeneration would not recreate it.
        for name in sorted(os.listdir(LATEX_DIR)):
            if name.endswith(".tex"):
                assert not inputs_of(os.path.join(LATEX_DIR, name)), f"latex/{name} includes another file"

    @pytest.mark.parametrize("name", ["manuscript_body.tex", "manuscript_variables.tex"])
    def test_the_hand_written_lists_are_not_build_products(self, name):
        assert not os.path.exists(os.path.join(LATEX_DIR, name))
        assert os.path.exists(os.path.join(SECTIONS_DIR, name))


class TestEveryDocumentReadsTheSameContent:
    @pytest.mark.parametrize("doc", MANUSCRIPTS)
    def test_a_layout_takes_the_body_from_the_one_list(self, doc):
        assert "sections/manuscript_body.tex" in inputs_of(os.path.join(ROOT, doc))

    @pytest.mark.parametrize("doc", DOCUMENTS)
    def test_a_document_takes_the_macros_from_the_one_list(self, doc):
        assert "sections/manuscript_variables.tex" in inputs_of(os.path.join(ROOT, doc))

    @pytest.mark.parametrize("doc", DOCUMENTS)
    def test_no_document_repeats_the_lists(self, doc):
        # Reaching into latex/ for a macro file, or naming a numbered section directly, is how a layout starts to
        # diverge from the others.
        for included in inputs_of(os.path.join(ROOT, doc)):
            assert not included.endswith("_variables.tex") or included == "sections/manuscript_variables.tex", (
                f"{doc} includes {included} instead of the aggregator")
            assert not re.match(r"sections/0[1-6][A-Za-z_]", included), f"{doc} includes {included} directly"


class TestTheListsMatchWhatIsOnDisk:
    def test_every_section_the_body_names_exists(self):
        for included in inputs_of(BODY):
            assert os.path.exists(os.path.join(ROOT, included)), included

    def test_every_macro_file_the_pipeline_writes_is_listed(self):
        listed = {os.path.basename(p) for p in inputs_of(VARIABLES)}
        on_disk = {f for f in os.listdir(LATEX_DIR) if f.endswith("_variables.tex")}
        if not on_disk:
            pytest.skip("the pipeline has not been run")
        missing = on_disk - listed
        assert not missing, f"macros generated but read by no document: {sorted(missing)}"

    def test_every_listed_macro_file_is_one_the_pipeline_writes(self):
        for included in inputs_of(VARIABLES):
            assert included.startswith("latex/"), included
            if not os.path.exists(os.path.join(ROOT, included)):
                pytest.skip(f"{included} has not been generated")

    def test_the_body_names_the_sections_in_reading_order(self):
        # The order of this list is the order of the manuscript; a sorted comparison would not catch a swap.
        numbered = [os.path.basename(p) for p in inputs_of(BODY) if re.match(r"0\d", os.path.basename(p))]
        assert numbered == sorted(numbered), numbered


DECLARATIONS = os.path.join(SECTIONS_DIR, "manuscript_declarations.tex")
DECLARATION_RE = re.compile(r"\\projdeclaration\{([^}]*)\}\{([^}]+)\}")


def declarations():
    return DECLARATION_RE.findall(COMMENT_RE.sub("", read(DECLARATIONS)))


class TestTheDeclarations:
    """Funding, competing interests and the rest. Journals want them in different places and under different
    headings, so the statement is a file and the heading is the layout's."""

    def test_every_declaration_names_a_file_that_exists(self):
        for heading, path in declarations():
            assert os.path.exists(os.path.join(ROOT, path)), f"{heading} -> {path}"

    def test_no_declaration_file_carries_its_own_heading(self):
        # A \section inside one of these would come out twice in the layouts that add their own, and in the wrong
        # shape in Springer, which puts them all under a single Declarations heading.
        for heading, path in declarations():
            text = COMMENT_RE.sub("", read(os.path.join(ROOT, path)))
            assert "\\section" not in text, f"{path} sets its own heading"

    def test_no_declaration_file_is_left_out_of_the_list(self):
        listed = {os.path.basename(p) for _, p in declarations()}
        on_disk = {f for f in os.listdir(SECTIONS_DIR) if re.match(r"06_\d+_", f)}
        assert on_disk - listed == set(), f"declarations written but never printed: {sorted(on_disk - listed)}"

    @pytest.mark.parametrize("doc", MANUSCRIPTS)
    def test_every_layout_prints_the_declarations_and_says_how(self, doc):
        text = read(os.path.join(ROOT, doc))
        assert "sections/manuscript_declarations.tex" in inputs_of(os.path.join(ROOT, doc)), doc
        assert re.search(r"\\newcommand\{\\projdeclaration\}\[2\]", text), f"{doc} does not define \\projdeclaration"

    def test_the_statements_are_not_also_buried_in_the_body(self):
        # These used to be the last sections of the Conclusions; printing them twice is the failure this guards.
        body_sections = [p for p in inputs_of(BODY)]
        for path in body_sections:
            text = COMMENT_RE.sub("", read(os.path.join(ROOT, path)))
            for stale in ("Data and Code Availability", "Generative AI", "Competing interests"):
                assert stale not in text, f"{path} still contains the {stale!r} declaration"


# Publishers, classes and the journals this manuscript has been aimed at. A shared file naming one of these has
# stopped being shared: the next layout inherits a statement, or a comment, written for someone else's rules.
# Only names that cannot also be ordinary prose: "nature" and "frontiers" are words this manuscript uses.
PUBLISHER_NAMES = re.compile(
    r"\b(BMC|BioMed Central|Springer|Elsevier|Wiley|Taylor\s*&\s*Francis|MDPI|PLOS|IEEE"
    r"|sn-jnl|cas-sc|cas-dc|elsarticle|Journal of Cheminformatics|JCIM)\b", re.I)


class TestTheSharedSourceNamesNoJournal:
    """sections/ is the manuscript; the journal exists only in the layout that typesets it. The statements and their
    order differ between publishers, so a file here that names one is a file that will be wrong for the next."""

    @pytest.mark.parametrize("name", sorted(f for f in os.listdir(SECTIONS_DIR) if f.endswith(".tex")))
    def test_no_section_names_a_publisher_or_a_class(self, name):
        for number, line in enumerate(read(os.path.join(SECTIONS_DIR, name)).splitlines(), start=1):
            found = PUBLISHER_NAMES.search(line)
            # A citation key or a bibliography entry may legitimately carry a publisher's name.
            if found and "\\cite" not in line:
                pytest.fail(f"sections/{name}:{number} names {found.group(0)!r}: {line.strip()[:90]}")

    def test_a_layout_is_still_allowed_to_name_its_own_journal(self):
        # The counterpart of the rule above: the knowledge has to live somewhere, and that somewhere is the template.
        assert PUBLISHER_NAMES.search(read(os.path.join(ROOT, "paper_springer.tex")))


METADATA = os.path.join(SECTIONS_DIR, "manuscript_metadata.tex")
AUTHOR_RE = re.compile(r"\\projauthor(?:corr)?\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}")
# Commands that belong to one journal class and would tie the shared metadata to it.
CLASS_MARKUP = re.compile(r"\\(author|affil|affiliation|address|fnm|sur|orgdiv|orgname|orgaddress|street"
                          r"|ead|email|credit|cormark|cortext|shortauthors|shorttitle|title)\b")


def authors():
    """The author list as the metadata states it: (initials, given, family, email, orcid) per entry."""
    return AUTHOR_RE.findall(COMMENT_RE.sub("", read(METADATA)))


class TestTheFrontMatterIsDataNotMarkup:
    """The title, the authors and the affiliation are facts about the paper, not about a journal. They live once,
    in sections/manuscript_metadata.tex, and each layout renders them in the notation its class wants. Before this
    split the four documents each spelled the block out, and they had drifted: two of them named a different
    corresponding author, and one had the city and the region swapped."""

    def test_the_metadata_is_hand_written_source(self):
        assert os.path.exists(METADATA)
        assert not os.path.exists(os.path.join(LATEX_DIR, "manuscript_metadata.tex"))

    @pytest.mark.parametrize("doc", DOCUMENTS)
    def test_every_document_reads_the_metadata(self, doc):
        assert "sections/manuscript_metadata.tex" in inputs_of(os.path.join(ROOT, doc))

    def test_the_metadata_names_no_class_specific_command(self):
        # The point of the file: a new journal is a new set of renderers in its own template, and nothing here.
        for number, line in enumerate(COMMENT_RE.sub("", read(METADATA)).splitlines(), start=1):
            found = CLASS_MARKUP.search(line)
            assert not found, f"manuscript_metadata.tex:{number} uses {found.group(0)!r}, which belongs to a class"

    @pytest.mark.parametrize("doc", MANUSCRIPTS)
    def test_every_layout_renders_both_kinds_of_author(self, doc):
        # A layout that defines only \projauthor drops the corresponding author silently.
        text = read(os.path.join(ROOT, doc))
        for macro in (r"\\projauthor", r"\\projauthorcorr"):
            assert re.search(r"newcommand\{" + macro + r"\}\[6\]", text), f"{doc} does not define {macro}"
        assert r"\projauthorlist" in text, doc

    def test_exactly_one_author_corresponds(self):
        corresponding = COMMENT_RE.sub("", read(METADATA)).count(r"\projauthorcorr{")
        assert corresponding == 1, f"{corresponding} corresponding authors"

    def test_the_contributions_come_from_the_same_list(self):
        # Stating the roles twice is how the author block and the declaration start to disagree.
        text = read(os.path.join(SECTIONS_DIR, "06_4_Authors_Contributions.tex"))
        assert r"\projauthorlist" in text
        for initials, _given, _family, _email, _orcid in authors():
            assert initials not in COMMENT_RE.sub("", text), f"{initials} is written out instead of rendered"

    @pytest.mark.parametrize("name", sorted(
        [f for f in os.listdir(SECTIONS_DIR) if f.endswith(".tex") and f != "manuscript_metadata.tex"]
        + [d for d in DOCUMENTS]))
    def test_no_other_file_repeats_an_author_detail(self, name):
        path = os.path.join(ROOT, name) if name in DOCUMENTS else os.path.join(SECTIONS_DIR, name)
        text = COMMENT_RE.sub("", read(path))
        for _initials, _given, family, email, orcid in authors():
            for detail in (family, email, orcid):
                if detail:
                    assert detail not in text, f"{name} repeats {detail!r} instead of reading the metadata"

"""LaTeX variables for results that are inserted into a manuscript. Only text is produced; writing it is left to the
caller."""
import re

_MACRO_NAME = re.compile(r"[A-Za-z]+")


def newcommand(name, value):
    r"""One line '\newcommand{\name}{value}'. LaTeX macro names may contain letters only, so a name with digits,
    underscores or other characters raises ValueError instead of producing a file that fails to compile.

    Args:
        name: macro name without the backslash (letters only).
        value: replacement text; any object, written with str() (format numbers before passing them).
    """
    if not _MACRO_NAME.fullmatch(name):
        raise ValueError(f"Invalid LaTeX macro name {name!r}: only letters are allowed")
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def format_macros(values, header=None):
    """All macros of a dict name -> value, in insertion order, preceded by an optional comment header line.

    Args:
        values: dict macro name -> value, as for newcommand.
        header: optional text of a first comment line ('% header'); None writes no header.
    """
    lines = [f"% {header}\n"] if header else []
    return "".join(lines + [newcommand(name, value) for name, value in values.items()])


# A table is typeset at the full text width of whatever class renders it, so that the manuscript, the two journal
# layouts and the supplement all show it the same way. Two constructions do that without a box: tabularx, when one
# column holds prose and can absorb the slack by wrapping, and tabular* with a stretchable intercolumn skip, when
# every column is short. A box (\resizebox, \adjustbox) is not an option: sn-jnl wraps a table in threeparttable and
# fails with "Missing \endgroup inserted" on any box around the tabular.
RAGGED_X = r">{\raggedright\arraybackslash}X"
CENTERED_X = r">{\centering\arraybackslash}X"


def tabular_open(colspec):
    """Opening line of a full-width table. An 'X' anywhere in colspec selects tabularx, otherwise tabular*.

    Args:
        colspec: column specification, as for tabular; an 'X' column (use RAGGED_X) absorbs the slack by wrapping.
    """
    if "X" in colspec:
        return r"\begin{tabularx}{\textwidth}{" + colspec + "}"
    return r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + colspec + "}"


def tabular_close(colspec):
    """Closing line matching tabular_open for the same colspec.

    Args:
        colspec: the same specification passed to tabular_open; a different one closes the wrong environment.
    """
    return r"\end{tabularx}" if "X" in colspec else r"\end{tabular*}"


# A table taller than one page cannot be a float: floats do not break. longtable breaks but cannot carry an X column,
# so a full-width table that must wrap a prose column AND run over a page break is built as xltabular, which is
# longtable with tabularx's measurement. It is typeset where it stands rather than floated, and its caption travels
# inside the environment, so the caption text is passed in instead of being written around the \input.
def breaking_table(colspec, caption, label, header, rows):
    r"""All lines of a full-width table that repeats its header on every page it breaks onto.

    Args:
        colspec: column specification, as for tabular_open (an 'X' column absorbs the slack by wrapping).
        caption: caption text, or a macro that expands to it, so the prose can stay in the hand-written document.
        label: key for \label, without the command.
        header: the header row, without its trailing row separator.
        rows: body lines, each already ending in a row separator.
    """
    head = ["\\toprule", f"{header} \\\\", "\\midrule"]
    return ["\\begin{xltabular}{\\textwidth}{" + colspec + "}",
            f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            *head, "\\endfirsthead",
            # The continuation repeats the number and the header, never the caption prose: a caption of a few lines
            # reprinted on every page costs more rows than it explains. The empty short caption keeps the
            # continuation out of the list of tables.
            "\\caption[]{(\\textit{continued})}\\\\",
            *head, "\\endhead",
            "\\bottomrule", "\\endlastfoot",
            *rows,
            "\\end{xltabular}"]

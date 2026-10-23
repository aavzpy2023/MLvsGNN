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

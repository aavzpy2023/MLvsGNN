"""
11_check_manuscript.py — Checks that the manuscript only uses artifacts that the pipeline produced.

Follows every \\input, \\projtable and \\projdeclaration from each root .tex file in the project (the manuscript and the supplementary material, each
with its own \\documentclass) and verifies, per document, that
  * each \\includegraphics file and each \\input file exists;
  * every project variable (a CamelCase macro such as \\MorganRTwoMean) used in the text is defined by one of the generated
    latex/*.tex files or by the preamble.
Exits with a non-zero status when something is missing, so a stale or absent figure/number fails the pipeline.
"""
import glob
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import ROOT, LOGS_DIR
from logger_utils import setup_logger, explain

# Kernel and class commands that happen to be camel case, which the heuristic below would otherwise read as
# project variables that nothing defines.
STANDARD_CAMELCASE = {"LaTeX", "TeX", "BibTeX", "AA", "WriteBookmarks", "AtEndDocument", "MakeLowercase",
                      "MakeUppercase"}
DEFINE = re.compile(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*\{?\\([A-Za-z]+)|\\def\\([A-Za-z]+)")
USE = re.compile(r"(?<!\\)\\([A-Za-z]+)")  # a doubled backslash is a line break followed by text

# \input is not the only way a file enters a document. The layouts pull in the generated tables and the shared
# declarations through the project's own macros, and a file reached that way has to be checked like any other: each
# pattern below captures the path argument.
INCLUDES = (
    re.compile(r"\\input\{([^}]+)\}"),
    re.compile(r"\\projtable\{([^}]+)\}"),
    re.compile(r"\\projdeclaration\{[^}]*\}\{([^}]+)\}"),
)


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def resolve(path_str, base_dir):
    candidates = [path_str, path_str + ".tex"] if not os.path.splitext(path_str)[1] else [path_str]
    for c in candidates:
        full = os.path.join(base_dir, c)
        if os.path.isfile(full):
            return full
    return None


def main():
    setup_logger(os.path.join(LOGS_DIR, "11_check_manuscript.log"))
    explain("What this step does", """
        Final consistency check of the paper against the pipeline outputs. It lists every figure and every included file the
        manuscript refers to and confirms they exist, and it confirms that each result variable used in the text was generated.
        A failure here means the text refers to something the pipeline did not produce (or no longer produces).
    """)
    # A vendor template dropped in the project should never halt the pipeline, so a file that is not UTF-8 is read
    # leniently rather than raising; it is still checked if it declares a document class.
    mains = sorted(f for f in glob.glob(os.path.join(ROOT, "*.tex"))
                   if r"\documentclass" in strip_comments(
                       open(f, encoding="utf-8", errors="replace").read()))
    if not mains:
        sys.exit(f"No root .tex file (one with \\documentclass) found in {ROOT}")

    errors = []
    for main_tex in mains:
        errors += check_document(main_tex)

    if errors:
        print("\n[FAIL] The manuscript refers to artifacts that do not exist:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        sys.exit(1)
    print(f"[OK] Every figure, included file, and variable used by the {len(mains)} document(s) is produced by the "
          "pipeline.", flush=True)


def check_document(main_tex):
    """Errors found in the document rooted at `main_tex`; its variables must be defined by itself or by what it inputs."""
    print(f"\n[DOC] {os.path.relpath(main_tex, ROOT)}", flush=True)
    texts, errors, graphics, queue, seen = {}, [], set(), [main_tex], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = strip_comments(open(path, encoding="utf-8").read())
        texts[path] = text
        for rel in [m for pattern in INCLUDES for m in pattern.findall(text)]:
            if rel.startswith("#"):
                continue  # \input{#1} inside a \newcommand is a parameter, not a file

            target = resolve(rel, ROOT)
            if target is None:
                errors.append(f"missing \\input file '{rel}' (referenced in {os.path.relpath(path, ROOT)})")
            else:
                queue.append(target)
        for rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            graphics.add(rel)

    for rel in sorted(graphics):
        found = any(os.path.isfile(os.path.join(ROOT, rel + ext)) for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"))
        print(f"[{'OK' if found else 'MISSING'}] figure {rel}", flush=True)
        if not found:
            errors.append(f"missing figure '{rel}'")

    all_text = "\n".join(texts.values())
    for name in sorted(set(re.findall(r"\\(?:newcommand|providecommand)\*?\s*\{\\([A-Za-z]+[0-9][A-Za-z0-9]*)\}", all_text))):
        errors.append(f"variable name \\{name} contains a digit, which LaTeX cannot use in a command name")
    defined = {a or b for a, b in DEFINE.findall(all_text)}
    used = {m for m in USE.findall(all_text) if sum(ch.isupper() for ch in m) >= 2}
    undefined = sorted(used - defined - STANDARD_CAMELCASE)
    for name in undefined:
        errors.append(f"variable \\{name} is used but never defined")

    print(f"[CHECK] {len(texts)} files followed, {len(graphics)} figures, {len(used)} project variables used, "
          f"{len(undefined)} undefined", flush=True)
    return errors


if __name__ == "__main__":
    main()

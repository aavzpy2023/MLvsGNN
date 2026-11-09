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
from paths_config import ROOT, LATEX_DIR as LATEX_DIR_LOCAL, LOGS_DIR
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

    write_catalogue()

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



# ──────────────────────────────────────────────────────────────────────────────
# Macro catalogue
# ──────────────────────────────────────────────────────────────────────────────
# The pipeline exports far more values than the manuscript cites, on purpose: a reviewer's objection is answered by
# looking a number up instead of recomputing it. That only works if the values are discoverable, which they are not
# while they sit in seventeen generated files. This writes one searchable index of all of them, with the current value,
# whether the text already cites it and where, and the command that regenerates its family. Notes are kept in a
# hand-written CSV and merged in, so that a suggestion survives the next regeneration.
MACROS_MD = os.path.join(ROOT, "MACROS.md")
MACRO_NOTES_CSV = os.path.join(ROOT, "macro_notes.csv")
MACRO_FAMILIES_CSV = os.path.join(ROOT, "macro_families.csv")
NEWCOMMAND = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
PRODUCER = re.compile(r"auto-generated by (\S+)", re.I)
# Which step reads a file and which writes it, so that "regenerate this family, then refresh those" is derived from
# the code and cannot go stale. Only direct uses of a path constant are seen: a path passed through a variable or
# built at call time is missed, which can only understate a dependency, never invent one.
WRITES = (r"to_csv\(\s*([A-Z][A-Z_0-9]{3,})", r"np\.save\(\s*([A-Z][A-Z_0-9]{3,})",
          r"joblib\.dump\([^,]+,\s*([A-Z][A-Z_0-9]{3,})", r"open\(\s*([A-Z][A-Z_0-9]{3,})\s*,\s*['\"][wa]")
READS = (r"read_csv\(\s*([A-Z][A-Z_0-9]{3,})", r"np\.load\(\s*([A-Z][A-Z_0-9]{3,})",
         r"joblib\.load\(\s*([A-Z][A-Z_0-9]{3,})", r"open\(\s*([A-Z][A-Z_0-9]{3,})\s*\)")


def read_csv_rows(path):
    """Rows of a hand-written CSV as dicts, or an empty list when the file does not exist."""
    if not os.path.isfile(path):
        return []
    import csv
    with open(path, encoding="utf-8", newline="") as fh:
        return [row for row in csv.DictReader(fh) if any((v or "").strip() for v in row.values())]


def macro_notes():
    """{macro: note} from the hand-written CSV, empty when it does not exist."""
    return {row["macro"].strip(): (row.get("note") or "").strip()
            for row in read_csv_rows(MACRO_NOTES_CSV) if row.get("macro")}


def macro_families():
    """{file name: {title, prefixes, description}} from the hand-written CSV; the one editorial part of the index."""
    return {row["file"].strip(): {k: (row.get(k) or "").strip()
                                  for k in ("title", "step", "prefixes", "description")}
            for row in read_csv_rows(MACRO_FAMILIES_CSV) if row.get("file")}


def step_of(script):
    """The pipeline step id a script file name carries, or None for a helper module."""
    match = re.match(r"(\d+(?:\.\d+)?)_", os.path.basename(script))
    return match.group(1) if match else None


def script_of_step(step):
    """File name of the script a step id names, for the two macro files whose header does not declare its producer."""
    if not step:
        return "?"
    found = sorted(glob.glob(os.path.join(ROOT, "scripts", f"{step}_*.py")))
    return os.path.basename(found[0]) if found else "?"


def output_dependencies():
    """(readers, writers) of every path constant, as {constant: {step id}}, over the pipeline scripts."""
    readers, writers = {}, {}
    scripts_dir = os.path.join(ROOT, "scripts")
    for path in sorted(glob.glob(os.path.join(scripts_dir, "*.py")) + glob.glob(os.path.join(scripts_dir, "tf", "*.py"))):
        step = step_of(path)
        if step is None:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        for patterns, into in ((WRITES, writers), (READS, readers)):
            for pattern in patterns:
                for constant in re.findall(pattern, text):
                    into.setdefault(constant, set()).add(step)
    # A step that writes a file is not a consumer of it
    for constant, steps in writers.items():
        readers.get(constant, set()).difference_update(steps)
    return readers, writers


def refresh_after(step, readers, writers):
    """Steps that read something this step writes, and steps whose output it reads: downstream, then upstream."""
    if step is None:
        return [], []
    downstream, upstream = set(), set()
    for constant, steps in writers.items():
        if step in steps:
            downstream |= readers.get(constant, set())
    for constant, steps in readers.items():
        if step in steps:
            upstream |= writers.get(constant, set())
    return sorted(downstream - {step}), sorted(upstream - {step})


def macro_usage():
    """{macro: [section files that use it]}, over the hand-written sources only (never the generated files)."""
    usage = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "sections", "*.tex")) + glob.glob(os.path.join(ROOT, "*.tex"))):
        text = strip_comments(open(path, encoding="utf-8", errors="replace").read())
        for name in set(USE.findall(text)):
            usage.setdefault(name, []).append(os.path.relpath(path, ROOT))
    return usage


def catalogue_markdown():
    """The whole catalogue as Markdown, so that a test can compare it with the file on disk."""
    usage, notes, families = macro_usage(), macro_notes(), macro_families()
    readers, writers = output_dependencies()
    groups, total, cited = [], 0, 0
    for path in sorted(glob.glob(os.path.join(LATEX_DIR_LOCAL, "*_variables.tex"))):
        file_name = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        producer = (PRODUCER.search(text) or [None, None])[1]
        producer = producer.rstrip(";") if producer else script_of_step(families.get(file_name, {}).get("step"))
        rows = []
        for name, value in sorted(NEWCOMMAND.findall(text)):
            where = ", ".join(sorted(os.path.basename(f) for f in usage.get(name, [])))
            rows.append((name, value, where, notes.get(name, "")))
            total += 1
            cited += bool(where)
        if rows:
            family = families.get(file_name, {})
            groups.append((file_name, family, producer, family.get("step") or step_of(producer), rows))

    out = ["<!-- Auto-generated by 11_check_manuscript.py. Do not edit: run `python run_pipeline.py --only 11`.",
           "     Titles and prefixes come from macro_families.csv, notes from macro_notes.csv; edit those. -->",
           "", "# Macro catalogue", "",
           f"{total} values exported by the pipeline, of which {cited} are cited by the manuscript and "
           f"{total - cited} are available and not cited. A value that is not cited is not a gap: it is there so that "
           "an objection can be answered by looking it up instead of recomputing it.", "",
           "Every number in the manuscript is one of these macros, never a literal, so re-running a step updates the "
           "text and rewording a sentence cannot change a value. To propose one for the text, add a row to "
           "`macro_notes.csv`; the note then appears in the Note column.", "",
           "| Family | Prefixes | Values | Cited | Step |", "| --- | --- | --- | --- | --- |"]
    for file_name, family, producer, step, rows in groups:
        anchor = (family.get("title") or file_name).lower().replace(" ", "-").replace(",", "").replace("'", "")
        n_cited = sum(1 for _, _, where, _ in rows if where)
        out.append(f"| [{family.get('title') or file_name}](#{anchor}) | {family.get('prefixes') or '—'} | "
                   f"{len(rows)} | {n_cited} | {step or '—'} |")
    out.append("")

    for file_name, family, producer, step, rows in groups:
        downstream, upstream = refresh_after(step, readers, writers)
        out += [f"## {family.get('title') or file_name}", ""]
        if family.get("description"):
            out += [family["description"], ""]
        lines = [f"`{file_name}`, written by `{producer}`."]
        lines.append(f"Regenerate with `python run_pipeline.py --only {step}`." if step
                     else "Regenerate by running the script named above.")
        if upstream:
            lines.append(f"It reads the output of step(s) {', '.join(upstream)}, so regenerating one of those means "
                         f"regenerating this family too.")
        if downstream:
            lines.append(f"Step(s) {', '.join(downstream)} read what it writes, so refresh those after it.")
        out += [" ".join(lines), "",
                "| Macro | Current value | Cited in | Note |", "| --- | --- | --- | --- |"]
        for name, value, where, note in rows:
            out.append(f"| `\\{name}` | {value} | {where or '—'} | {note} |")
        out.append("")
    return "\n".join(out) + "\n"


def write_catalogue():
    with open(MACROS_MD, "w", encoding="utf-8") as fh:
        fh.write(catalogue_markdown())
    print(f"[EXPORT] Macro catalogue saved to {os.path.relpath(MACROS_MD, ROOT)}", flush=True)

if __name__ == "__main__":
    main()


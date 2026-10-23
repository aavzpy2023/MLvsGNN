"""The package contract of qsartools (see its __init__): generic dependencies only, no file I/O, no global state."""
import ast
import os
import sys

import pytest

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts",
                           "qsartools")
MODULES = sorted(f for f in os.listdir(PACKAGE_DIR) if f.endswith(".py"))
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "pandas", "sklearn", "rdkit", "joblib", "statsmodels"}
# Calls and attributes that would read or write files or change global state
FORBIDDEN_CALLS = {"open", "savefig", "to_csv", "to_excel", "to_parquet", "to_pickle", "save", "savez", "savetxt", "dump",
                   "seed", "set_random_seed", "makedirs", "mkdir", "remove", "unlink", "chdir"}


def parsed(module):
    with open(os.path.join(PACKAGE_DIR, module), encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=module)


@pytest.mark.parametrize("module", MODULES)
def test_only_generic_dependencies(module):
    imported = set()
    for node in ast.walk(parsed(module)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    foreign = imported - set(sys.stdlib_module_names) - ALLOWED_THIRD_PARTY - {"qsartools"}
    assert not foreign, f"{module} imports {sorted(foreign)}; qsartools may only depend on {sorted(ALLOWED_THIRD_PARTY)}"


@pytest.mark.parametrize("module", MODULES)
def test_no_file_io_or_global_state(module):
    offending = []
    for node in ast.walk(parsed(module)):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in FORBIDDEN_CALLS:
                offending.append(f"line {node.lineno}: {name}()")
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            offending.append(f"line {node.lineno}: os.environ")
    assert not offending, f"{module}: {offending}"

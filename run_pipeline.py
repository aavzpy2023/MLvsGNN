"""
run_pipeline.py — Runs the full computational pipeline (scripts 01 -> 09.1) and logs every step.

Usage (from any directory):
    python run_pipeline.py                    # run everything
    python run_pipeline.py --list             # show the steps
    python run_pipeline.py --check            # verify Python dependencies, run nothing
    python run_pipeline.py --from 05          # resume from step 05 onwards
    python run_pipeline.py --only 04 04.1     # run selected steps only
    python run_pipeline.py --only figures     # rebuild every figure and table from existing results (seconds)
    python run_pipeline.py --no-sensitivity   # skip the with-extreme-values re-run (steps S01, S02, 10)
    python run_pipeline.py --no-scaffold      # skip the scaffold-split re-run (steps SC01, SC09, 13)
    python run_pipeline.py --strict           # error (instead of silent skip) if ChemProp cannot load
Environment variables (optional): MLGNN_PROFILE (laptop|workstation), MLGNN_N_JOBS,
    MLGNN_TF_PYTHON (python of the separate TensorFlow environment, needed only by the scripts/tf/ steps).
"""


import argparse
import importlib.util
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata

ROOT = os.path.dirname(os.path.abspath(__file__))
# MLGNN_OUT_DIR redirects every output, as it does in paths_config, so that a reduced run (--quick) cannot overwrite
# the logs of the real one. A normal run never has it set: it is passed to the sensitivity and scaffold steps alone.
LOGS_DIR = os.path.join(os.environ.get("MLGNN_OUT_DIR") or ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

PIPELINE = [
    "scripts/00_prepare_dataset.py",
    "scripts/01_nested_cv_stacking.py",
    "scripts/02_statistical_tests.py",
    "scripts/03_y_randomization.py",
    "scripts/04_augmentation_training.py",
    "scripts/05_virtual_screening.py",
    "scripts/06_redocking_validation.py",
    "scripts/07_molecular_docking.py",
    "scripts/08_admet_profiling.py",
    "scripts/09_gnn_baseline.py",
    "scripts/12_gnn_tuned.py",
    "scripts/14_export_folds.py",
    "scripts/19_gnn_strong_baselines.py",
]

# Audit of the original TensorFlow/Keras network: these steps run with the interpreter of a separate environment
# (MLGNN_TF_PYTHON, see requirements-tf-lock.txt), because TensorFlow is not part of the main one. They live in scripts/tf/,
# which the import scan of --check does not enter, and they only read inputs and write CSV files to results/.
TF_PYTHON = os.environ.get("MLGNN_TF_PYTHON")
TF_PIPELINE = [
    "scripts/tf/15_karimova_reproduction.py",
    "scripts/tf/16_leakage_ladder.py",
    "scripts/tf/17_dnn_nested_cv.py",
]
TF_REQUIRED_MODULES = ["tensorflow", "keras", "feature_engine", "sklearn", "pandas", "numpy"]
# Main-environment summary of the TensorFlow steps (statistics and LaTeX variables); runs after them
AUDIT = ["scripts/18_karimova_audit.py"]

# Figure and table stage: runs automatically after every computation step has finished (so no figure can read a
# result that does not exist yet) and can be launched alone with `--only figures`, without recomputing anything.
# The last entry checks that the manuscript only references figures, inputs and variables that exist.
FIGURES = [
    # First: the settings this run used, which every later check reads and no step else exports
    "scripts/00.1_export_config.py",
    "scripts/01.1_build_figures.py",
    "scripts/04.1_build_figures.py",
    "scripts/05.1_build_figures.py",
    "scripts/07.1_build_figures.py",
    "scripts/08.1_build_figures.py",
    "scripts/09.1_build_figures.py",
    "scripts/11_check_manuscript.py",
]

# Outlier sensitivity analysis: steps 01 and 02 re-run on the set that keeps the extreme pIC50 values, writing to
# sensitivity/ (paths_config honours MLGNN_OUT_DIR); step 10 then compares both runs. Entries are (step id, script).
SENS_DIR = os.environ.get("MLGNN_SENS_DIR") or os.path.join(ROOT, "sensitivity")
SENS_ENV = {"MLGNN_OUT_DIR": SENS_DIR,
            "MLGNN_TRAIN_FILE": os.path.join(ROOT, "data", "V2-combined_bd_chmbl_unique_dedup_with_outliers.csv")}
SENSITIVITY = [
    ("S01", "scripts/01_nested_cv_stacking.py", SENS_ENV),
    ("S02", "scripts/02_statistical_tests.py", SENS_ENV),
    ("10", "scripts/10_outlier_sensitivity.py", {}),
]

# Scaffold-split robustness: the classical ensembles of the four representations (01) and the default D-MPNN (09) are re-run on scaffold-grouped
# outer folds, writing to scaffold/; step 13 summarizes the comparison with the random folds.
SCAFFOLD_DIR = os.environ.get("MLGNN_SCAFFOLD_DIR") or os.path.join(ROOT, "scaffold")
SCAFFOLD_ENV = {"MLGNN_OUT_DIR": SCAFFOLD_DIR, "MLGNN_SPLIT": "scaffold",
                "MLGNN_MODES": "morgan,rdkit2d,rdkit2d_fp,rdkit2d3d_fp"}
SCAFFOLD = [
    ("SC01", "scripts/01_nested_cv_stacking.py", SCAFFOLD_ENV),
    ("SC09", "scripts/09_gnn_baseline.py", SCAFFOLD_ENV),
    ("SC17", "scripts/tf/17_dnn_nested_cv.py", SCAFFOLD_ENV),
    ("SC19", "scripts/19_gnn_strong_baselines.py", {**SCAFFOLD_ENV, "MLGNN_K19_VARIANTS": "dmpnn_ens5,chemeleon"}),
    ("13", "scripts/13_scaffold_split.py", {}),
]

# Import names always checked by --check, in addition to EVERY third-party module that
# scripts/*.py actually import (found automatically with `ast`, nothing is executed).
# meeko + vina (Python bindings) are used by steps 06/07; torch/lightning/chemprop by step 09.
# No external binary is needed by the pipeline: `obabel` is only used by scripts/prep_receptor.py,
# which is NOT part of the pipeline (receptor/receptor.pdbqt is versioned in the repository).
REQUIRED_MODULES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "rdkit", "xgboost",
                    "lightgbm", "meeko", "vina", "torch", "lightning", "chemprop"]
# PyPI distribution names printed in the environment snapshot
SNAPSHOT_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "rdkit",
                     "meeko", "vina", "torch", "lightning", "chemprop", "PyTDC", "seaborn",
                     "adjustText", "matplotlib", "joblib", "setuptools"]


def step_id(script):
    """'scripts/04.1_build_figures.py' -> '04.1'"""
    return os.path.basename(script).split("_", 1)[0]


class MasterStreamWriter:
    """Writes in real-time to both console and log files (individual and master)."""
    def __init__(self, log_filepaths):
        self.terminal = sys.stdout
        self.files = [open(fp, "a", encoding="utf-8") for fp in log_filepaths]

    def write(self, message):
        self.terminal.write(message)
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        self.terminal.flush()
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            f.close()


def environment_snapshot():
    lines = [f"Python {platform.python_version()} | {platform.system()} {platform.release()} | "
             f"{os.cpu_count()} CPU threads | profile={os.environ.get('MLGNN_PROFILE', 'workstation')}"]
    for pkg in SNAPSHOT_PACKAGES:
        try:
            lines.append(f"  {pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"  {pkg}: not installed")
    return "\n".join(lines)


def try_import(module):
    """Really imports `module` in a subprocess. Returns None if OK, else the last error line.
    (find_spec is not enough: e.g. dgl is 'installed' but fails to import with an unsupported torch.)"""
    r = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    if r.returncode == 0:
        return None
    lines = (r.stderr or "").strip().splitlines()
    return lines[-1][:160] if lines else "import failed"


def scan_imports():
    """Third-party top-level modules imported anywhere in scripts/*.py -> {module: {script, ...}}.
    Parsed with `ast` (nothing is executed); standard library and project modules are excluded."""
    import ast
    scripts_dir = os.path.join(ROOT, "scripts")
    if not os.path.isdir(scripts_dir):
        return {}
    stdlib = set(sys.stdlib_module_names)
    # Project modules: the step scripts, the shared modules next to them and the qsartools package
    local = {os.path.splitext(f)[0] for f in os.listdir(scripts_dir) if f.endswith(".py")} | {"qsartools"}
    tools_dir = os.path.join(scripts_dir, "qsartools")
    files = [os.path.join(scripts_dir, f) for f in sorted(os.listdir(scripts_dir)) if f.endswith(".py")]
    if os.path.isdir(tools_dir):
        files += [os.path.join(tools_dir, f) for f in sorted(os.listdir(tools_dir)) if f.endswith(".py")]
    found = {}
    for path in files:
        fname = os.path.relpath(path, scripts_dir)
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=fname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m not in stdlib and m not in local:
                    found.setdefault(m, set()).add(fname)
    return found


def check_dependencies():
    print(environment_snapshot())
    used = scan_imports()
    to_check = sorted(set(REQUIRED_MODULES) | set(used))
    print(f"\nThird-party modules imported by scripts/*.py ({len(used)}): {', '.join(sorted(used))}")
    print(f"Checking that {len(to_check)} modules really import (this can take a minute)...")
    bad = {m: e for m in to_check if (e := try_import(m))}
    print("\nModules that FAIL to import:", "none" if not bad else "")
    for m, e in bad.items():
        users = ", ".join(sorted(used.get(m, []))) or "listed in REQUIRED_MODULES"
        print(f"  - {m}: {e}\n      used by: {users}")
    print("\nNote: cross-check the list above against requirements.txt (module names differ from PyPI names,"
          " e.g. sklearn -> scikit-learn, tdc -> PyTDC).")
    return not bad and check_tf_environment()


def check_tf_environment():
    """Checks the separate TensorFlow interpreter of the scripts/tf/ steps; not having one is only a warning."""
    if not TF_PYTHON:
        print("\nMLGNN_TF_PYTHON is not set: the TensorFlow steps (scripts/tf/) cannot run, every other step is unaffected.")
        return True
    code = "import " + ", ".join(TF_REQUIRED_MODULES) + "; print(tensorflow.__version__, keras.__version__)"
    proc = subprocess.run([TF_PYTHON, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))
    if proc.returncode != 0:
        print(f"\nTensorFlow environment ({TF_PYTHON}) FAILS to import {TF_REQUIRED_MODULES}:\n{proc.stderr.strip()[-800:]}")
        return False
    tf_ver, keras_ver = proc.stdout.split()[-2:]
    print(f"\nTensorFlow environment OK ({TF_PYTHON}): tensorflow {tf_ver}, keras {keras_ver}")
    return True


def all_steps(include_sensitivity=True, include_scaffold=True):
    """Ordered (step id, script, extra environment) triples: the main pipeline, then the sensitivity analysis."""
    figures = [(step_id(s), s, {}) for s in FIGURES]
    steps = [(step_id(s), s, {}) for s in PIPELINE + TF_PIPELINE + AUDIT]
    return (steps + (SENSITIVITY if include_sensitivity else []) + (SCAFFOLD if include_scaffold else []) + figures)


def is_tf_step(script_path):
    return script_path.replace("\\", "/").startswith("scripts/tf/")


def execute_and_log(script_path, master_log_path, extra_env=None):
    script_filename = os.path.basename(script_path)
    logs_dir = os.path.join(extra_env["MLGNN_OUT_DIR"], "logs") if extra_env and "MLGNN_OUT_DIR" in extra_env else LOGS_DIR
    os.makedirs(logs_dir, exist_ok=True)
    step_log_path = os.path.join(logs_dir, script_filename.replace(".py", ".log"))

    with open(step_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== START LOG: {script_filename} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")

    writer = MasterStreamWriter([step_log_path, master_log_path])

    # MLGNN_LOG_MANAGED: tells logger_utils not to open the step log itself (this runner owns it)
    # MPLBACKEND=Agg: figure scripts must not need a display (headless/SSH runs crash with the Tk backend)
    env = dict(os.environ, MLGNN_LOG_MANAGED="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg",
               PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.validation",
               **(extra_env or {}))

    python = TF_PYTHON if is_tf_step(script_path) else sys.executable
    process = subprocess.Popen(
        [python, os.path.join(ROOT, script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=ROOT, env=env,
    )
    try:
        for line in iter(process.stdout.readline, ''):
            writer.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, script_path)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        writer.write(f"\n=== END LOG: {script_filename} [EXIT CODE: {process.returncode}] ===\n\n")
        writer.close()


def select_steps(args):
    steps = all_steps(include_sensitivity=not args.no_sensitivity, include_scaffold=not args.no_scaffold)
    ids = [sid for sid, _, _ in steps]
    if args.only:
        wanted = [i for w in args.only for i in ([step_id(s) for s in FIGURES] if w == "figures" else [w])]
        args.only = wanted
        unknown = set(args.only) - set(ids)
        if unknown:
            sys.exit(f"Unknown step(s): {sorted(unknown)}. Use --list.")
        return [st for st in steps if st[0] in set(args.only)]
    if args.start:
        if args.start not in ids:
            sys.exit(f"Unknown step '{args.start}'. Use --list.")
        return steps[ids.index(args.start):]
    return steps


def run_pipeline(steps=None):
    steps = steps if steps is not None else all_steps()
    # Fail before running anything, rather than hours into the pipeline
    tf_steps = [sid for sid, script, _ in steps if is_tf_step(script)]
    if tf_steps and not (TF_PYTHON and os.access(TF_PYTHON, os.X_OK)):
        sys.exit(f"Steps {tf_steps} need the TensorFlow interpreter: set MLGNN_TF_PYTHON to its python executable "
                 f"(current value: {TF_PYTHON!r}), or leave these steps out with --only.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(LOGS_DIR, f"master_pipeline_{timestamp}.log")

    print("=" * 80)
    print("INITIATING Q1-STANDARDIZED COMPUTATIONAL PIPELINE")
    print(f"Master execution log: {master_log_path}")
    print("=" * 80)
    snapshot = environment_snapshot()
    print(snapshot)
    with open(master_log_path, "w", encoding="utf-8") as f:
        f.write(snapshot + "\n\n")

    timings = []
    t_start = time.time()
    for sid, script, extra_env in steps:
        print(f"\n[>>>] Executing {sid}: {script}...")
        t0 = time.time()
        try:
            execute_and_log(script, master_log_path, extra_env)
        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] {script} failed with exit code {e.returncode}.")
            print("Halting pipeline to prevent cascading data corruption.")
            print(f"Fix the problem and resume with:  python run_pipeline.py --from {sid}")
            sys.exit(1)
        timings.append((f"{sid} {os.path.basename(script)}", time.time() - t0))
        print(f"[OK] {script} completed successfully ({timings[-1][1]:.0f}s).")

    print("\nStep timings:")
    for label, secs in timings:
        print(f"  {label:<40} {secs / 60:7.1f} min")
    print(f"  {'TOTAL':<40} {(time.time() - t_start) / 60:7.1f} min")
    print("\n[SUCCESS] Entire pipeline executed flawlessly.")


def step_log_duration(step_id_, script_path, extra_env):
    """Wall-clock duration of the last run of this step, from its log, or None if it has not run here.

    The log's first line carries the start time and the file's modification time is the end, which is what the step
    costs on this machine; it is a measurement, not a promise.
    """
    logs_dir = os.path.join(extra_env["MLGNN_OUT_DIR"], "logs") if extra_env and "MLGNN_OUT_DIR" in extra_env else LOGS_DIR
    path = os.path.join(logs_dir, os.path.basename(script_path).replace(".py", ".log"))
    try:
        with open(path, encoding="utf-8") as f:
            head = f.readline()
        start = datetime.strptime(re.search(r"\[([\d-]+ [\d:]+)\]", head).group(1), "%Y-%m-%d %H:%M:%S")
    except (OSError, AttributeError, ValueError):
        return None
    seconds = os.path.getmtime(path) - start.timestamp()
    return seconds if seconds >= 0 else None


def human_duration(seconds):
    if seconds is None:
        return "not run here"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def list_steps():
    """Every step in the order a full run executes them, with what its last run cost on this machine."""
    steps = all_steps()
    total = 0.0
    print(f"{'STEP':>5}  {'SCRIPT':<44} LAST RUN HERE")
    for sid, script, extra_env in steps:
        seconds = step_log_duration(sid, script, extra_env)
        total += seconds or 0.0
        print(f"{sid:>5}  {script:<44} {human_duration(seconds)}")
    print(f"\n{len(steps)} steps; the runs recorded in the logs add up to {total / 3600:.1f} h on this machine.")
    print("These are the durations of the LAST run of each step, so a step that resumed from a checkpoint, or that only\n"
          "re-summarized existing results, shows seconds instead of the hours it costs from scratch. A full run from\n"
          "scratch takes days: 01, S01 and SC01 are the nested cross-validations, 03 the y-randomization and 19/SC19\n"
          "the pretrained graph networks. Steps 01 and 03 checkpoint their progress and resume if interrupted. Use\n"
          "--quick for a reduced smoke run of the whole pipeline (its numbers are not the published ones).")


def main():
    ap = argparse.ArgumentParser(description="Run the ML vs GNN computational pipeline.")
    ap.add_argument("--list", action="store_true", help="list pipeline steps and exit")
    ap.add_argument("--check", action="store_true", help="check Python dependencies and exit")
    ap.add_argument("--from", dest="start", metavar="STEP", help="resume from this step (e.g. 05 or 04.1)")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps (e.g. 04 04.1); 'figures' selects the whole figure stage")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the outlier sensitivity analysis (steps S01, S02, 10) in a full run")
    ap.add_argument("--no-scaffold", action="store_true",
                    help="skip the scaffold-split robustness analysis (steps SC01, SC09, 13) in a full run")
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of silently skipping a model whose libraries cannot be imported (step 09)")
    ap.add_argument("--quick", action="store_true",
                    help="reduced smoke run of the whole pipeline, written to quick_run/ so that it cannot overwrite "
                         "the published results; its numbers are NOT the published ones")
    args = ap.parse_args()

    if args.quick and os.environ.get("MLGNN_PROFILE") != "quick":
        # Restart with the reduced profile in place: paths_config reads it at import time, here and in every step, and
        # the output directories must be redirected before anything writes. An MLGNN_OUT_DIR already set is respected.
        out_dir = os.environ.get("MLGNN_OUT_DIR") or os.path.join(ROOT, "quick_run")
        env = dict(os.environ, MLGNN_PROFILE="quick", MLGNN_OUT_DIR=out_dir,
                   MLGNN_SENS_DIR=os.environ.get("MLGNN_SENS_DIR") or os.path.join(out_dir, "sensitivity"),
                   MLGNN_SCAFFOLD_DIR=os.environ.get("MLGNN_SCAFFOLD_DIR") or os.path.join(out_dir, "scaffold"))
        os.makedirs(out_dir, exist_ok=True)
        print(f"[QUICK] reduced profile; every output goes to {out_dir}", flush=True)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    if args.list:
        list_steps()
        return
    if args.check:
        sys.exit(0 if check_dependencies() else 1)
    if args.strict:
        os.environ["MLGNN_STRICT"] = "1"
    run_pipeline(select_steps(args))


if __name__ == "__main__":
    main()

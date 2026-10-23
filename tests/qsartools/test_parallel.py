import os
import signal
import time

import pytest

from qsartools.parallel import run_tasks


def square(task):
    return task * task


def crash_once(task):
    """Kills its own worker the first time it sees the task flagged to crash (a marker file records that it did)."""
    value, marker = task
    if marker and not os.path.exists(marker):
        open(marker, "w").close()
        os.kill(os.getpid(), signal.SIGKILL)
    return value * 10


def always_crash(task):
    os.kill(os.getpid(), signal.SIGKILL)


def raise_error(task):
    raise ValueError(f"bad task {task}")


def test_results_and_callback():
    seen = []
    results = run_tasks(square, [1, 2, 3, 4], n_workers=2, on_result=lambda t, r: seen.append((t, r)), log=lambda m: None)
    assert sorted(results) == [1, 4, 9, 16]
    assert sorted(seen) == [(1, 1), (2, 4), (3, 9), (4, 16)]


def test_crashed_worker_is_retried(tmp_path):
    tasks = [(1, None), (2, str(tmp_path / "crash")), (3, None)]
    messages = []
    results = run_tasks(crash_once, tasks, n_workers=3, log=messages.append)
    assert sorted(results) == [10, 20, 30]
    assert any("worker died" in m for m in messages)


def record_run(task):
    """Leaves one file per execution in a directory; task 'crash' kills its worker, the others take a moment."""
    name, directory = task
    open(os.path.join(directory, f"{name}-{os.getpid()}"), "w").close()
    if name == "crash":
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(1.0)
    return name


def test_crash_does_not_rerun_siblings(tmp_path):
    tasks = [("a", str(tmp_path)), ("crash", str(tmp_path)), ("b", str(tmp_path))]
    with pytest.raises(RuntimeError, match="1 task"):
        run_tasks(record_run, tasks, n_workers=3, max_attempts=2, log=lambda m: None)
    runs = sorted(p.name.split("-")[0] for p in tmp_path.iterdir())
    assert runs == ["a", "b", "crash", "crash"]


def test_results_before_giving_up_are_kept():
    seen = []
    with pytest.raises(RuntimeError):
        run_tasks(crash_on_zero, [0, 1, 2], n_workers=1, max_attempts=2, on_result=lambda t, r: seen.append(r),
                  log=lambda m: None)
    assert sorted(seen) == [1, 2]


def crash_on_zero(task):
    if task == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    return task


def test_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError, match="unfinished after 2 attempts"):
        run_tasks(always_crash, [1], n_workers=1, max_attempts=2, log=lambda m: None)


def test_ordinary_exceptions_propagate():
    with pytest.raises(ValueError, match="bad task"):
        run_tasks(raise_error, [1], n_workers=1, log=lambda m: None)

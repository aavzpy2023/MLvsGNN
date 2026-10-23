"""Process-parallel execution of independent tasks that survives worker crashes."""
import multiprocessing as mp
import traceback
from collections import deque
from multiprocessing.connection import wait

# Native libraries (MKL, oneDNN) may pick an instruction set the CPU lacks and kill a worker with SIGILL; capping them
# at AVX is a no-op on CPUs that support it and prevents the crash on CPUs without AVX2.
SAFE_CPU_ENV = {"MKL_ENABLE_INSTRUCTIONS": "AVX", "DNNL_MAX_CPU_ISA": "AVX", "ONEDNN_MAX_CPU_ISA": "AVX"}


def _run_one(function, task, conn):
    try:
        outcome = (True, function(task))
    except BaseException as exc:
        outcome = (False, exc)
    try:
        conn.send(outcome)
    except Exception:
        # Unpicklable result or exception: report it as text rather than dying silently (which would look like a crash)
        conn.send((False, RuntimeError(traceback.format_exc())))
    conn.close()


def run_tasks(function, tasks, n_workers, on_result=None, max_attempts=3, log=print):
    """Run function(task) for every task in spawned processes, at most `n_workers` at a time, and return the results in
    completion order.

    Every task runs in its own process, so a process that dies abruptly (segmentation fault, illegal instruction, killed)
    costs only its own task, which is run again in a fresh process, up to `max_attempts` runs in total. Tasks running
    beside it are never interrupted. (A shared process pool would break and take them down too; workers that ignore
    SIGTERM, as PyTorch Lightning does, would then keep computing results that can no longer be returned.)
    A task whose function raises an ordinary exception is not retried: the remaining processes are killed and the
    exception propagates.

    Args:
        function: callable taking one task and returning its result. It must be importable by name (a module-level
            function, not a lambda or closure), because the "spawn" start method pickles it by reference.
        tasks: list of task arguments; each one, and each result, must be picklable.
        n_workers: maximum number of processes running at the same time.
        on_result: optional callable on_result(task, result), called in the parent as soon as each result arrives, so
            that the caller can checkpoint it before the remaining tasks finish.
        max_attempts: total number of runs allowed for a task whose process dies abruptly.
        log: callable taking one message string, used to report crashes (print by default).

    Returns:
        List of results in completion order (not task order).

    Raises:
        RuntimeError: at the end, if any task crashed `max_attempts` times; the results of the other tasks have already
            been passed to `on_result`.
    """
    ctx = mp.get_context("spawn")
    queue = deque(range(len(tasks)))
    attempts = [0] * len(tasks)
    running = {}
    results, failed = [], []
    try:
        while queue or running:
            while queue and len(running) < n_workers:
                i = queue.popleft()
                attempts[i] += 1
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_run_one, args=(function, tasks[i], send))
                proc.start()
                # Only the child must hold the sending end, so that its death closes the pipe and wakes wait()
                send.close()
                running[recv] = (i, proc)
            for conn in wait(list(running)):
                i, proc = running.pop(conn)
                try:
                    ok, payload = conn.recv()
                except EOFError:
                    proc.join()
                    if attempts[i] < max_attempts:
                        log(f"  [run_tasks] a worker died (exit code {proc.exitcode}); its task will be run again "
                            f"(attempt {attempts[i] + 1} of {max_attempts})")
                        queue.append(i)
                    else:
                        log(f"  [run_tasks] a worker died (exit code {proc.exitcode}); giving up on its task after "
                            f"{max_attempts} attempts")
                        failed.append(i)
                    continue
                finally:
                    conn.close()
                proc.join()
                if not ok:
                    raise payload
                results.append(payload)
                if on_result is not None:
                    on_result(tasks[i], payload)
    finally:
        # SIGKILL, not SIGTERM: frameworks that trap SIGTERM would otherwise keep training after the parent has left
        for _, proc in running.values():
            proc.kill()
            proc.join()
    if failed:
        raise RuntimeError(f"{len(failed)} task(s) still unfinished after {max_attempts} attempts (worker crashes)")
    return results

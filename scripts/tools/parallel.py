"""Process-parallel execution of independent tasks that survives worker crashes."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# Native libraries (MKL, oneDNN) may pick an instruction set the CPU lacks and kill a worker with SIGILL; capping them
# at AVX is a no-op on CPUs that support it and prevents the crash on CPUs without AVX2.
SAFE_CPU_ENV = {"MKL_ENABLE_INSTRUCTIONS": "AVX", "DNNL_MAX_CPU_ISA": "AVX", "ONEDNN_MAX_CPU_ISA": "AVX"}


def run_tasks(function, tasks, n_workers, on_result=None, max_rounds=3, log=print):
    """Run function(task) for every task in spawned worker processes and return the results in completion order.

    If a worker dies abruptly (segmentation fault, illegal instruction, killed), the pool breaks; the tasks that had not
    returned are run again in a fresh pool, up to `max_rounds` pools in total. A task whose function raises an ordinary
    exception is not retried: the exception propagates. `on_result(task, result)` is called as soon as each result arrives,
    so that the caller can checkpoint it. Raises RuntimeError if tasks are still unfinished after `max_rounds`.
    """
    pending = list(range(len(tasks)))
    results = []
    for round_ in range(1, max_rounds + 1):
        if not pending:
            break
        with ProcessPoolExecutor(min(n_workers, len(pending)), mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(function, tasks[i]): i for i in pending}
            done = set()
            # A broken pool fails only the futures that had not finished; finished ones keep their result
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    result = fut.result()
                except BrokenProcessPool:
                    continue
                done.add(i)
                results.append(result)
                if on_result is not None:
                    on_result(tasks[i], result)
        if len(done) < len(pending):
            log(f"  [run_tasks] a worker died; {len(pending) - len(done)} task(s) will be run again "
                f"(round {round_} of {max_rounds})")
        pending = [i for i in pending if i not in done]
    if pending:
        raise RuntimeError(f"{len(pending)} task(s) still unfinished after {max_rounds} rounds of worker crashes")
    return results

import multiprocessing as mp
from functools import wraps
import math
import statistics
from time import perf_counter

import numpy as np


def print_indented(text: str, indent: int = 0, indent_char: str = "  "):
    indentation = indent_char * indent
    for line in text.splitlines():
        print(f"{indentation}{line}")


def in_sub_process(func):
    """Decorator to run a function in a separate process and return its output."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        def target_wrapper(q, *args, **kwargs):
            # Execute the function and put the result in the queue
            result = func(*args, **kwargs)
            q.put(result)

        queue = mp.Queue()
        process = mp.Process(target=target_wrapper, args=(queue, *args), kwargs=kwargs)

        process.start()
        result = queue.get()
        process.join()

        return result

    return wrapper


def quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def time_statistics(
    times: list[float], quantiles: None | float = None, return_mode: str = "mean"
) -> float | list[float]:
    if quantiles is not None:
        ret = quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


class TestBench:
    variants = {}

    @classmethod
    @in_sub_process
    def test(cls, **kwargs) -> dict:
        """
        Run the test in a separate process and return the results.
        """
        try:
            fn = cls.get_fn(**kwargs)
        except Exception as e:
            return e

        kwargs_list = cls.preprocess_test_data(**kwargs)
        results = [fn(**kwargs) for kwargs in kwargs_list]
        results = cls.postprocess_test_data(results)
        return results

    @classmethod
    @in_sub_process
    def bench(
        cls,
        run_ms: int = 100,
        warmup_ms: int = 25,
        quantiles=None,
        return_mode: str = "mean",
        **kwargs,
    ) -> float:
        """
        Run the benchmark in a separate process and return the timing results.
        """
        bench_data = cls.get_bench_data(**kwargs)
        fn = cls.get_fn(**kwargs)
        iters = 5
        start_time = perf_counter()
        for _ in range(iters):
            cls.clear_cache()
            result = fn(**bench_data)
        cls.sync(result)
        end_time = perf_counter()
        elapsed_ms = (end_time - start_time) * 1e3
        iter_ms = elapsed_ms / iters
        warmup = max(int(warmup_ms / iter_ms), 1)
        iters = max(int(run_ms / iter_ms), 1)
        # Warmup
        for _ in range(warmup):
            result = fn(**bench_data)
        cls.sync(result)
        # Benchmark
        times = []
        for _ in range(iters):
            cls.clear_cache()
            start_time = perf_counter()
            result = fn(**bench_data)
            cls.sync(result)
            end_time = perf_counter()
            times.append((end_time - start_time) * 1e3)
        return time_statistics(times, quantiles, return_mode)

    def get_fn():
        """
        Return the function to be tested or benchmarked.
        """
        raise NotImplementedError

    def preprocess_test_data(**kwargs) -> list:
        """
        Preprocessing of inputs before testing.
        Takes reference inputs as numpy arrays as well as other keyword arguments and a list of arguments for each test run.
        """
        raise NotImplementedError

    def postprocess_test_data(results: list) -> dict:
        """
        Postprocessing of results after testing.
        Takes a list of results from each test run and returns a dictionary of processed results as numpy arrays.
        """
        raise NotImplementedError

    def get_bench_data(**kwargs) -> dict:
        """
        Generate benchmark data based on the provided arguments.
        """
        raise NotImplementedError

    @in_sub_process
    def get_ref_data(**kwargs) -> dict:
        """
        Generate reference data based on the provided arguments.
        Should use @in_sub_process decorator to run in a separate process if heavy device runtimes are loaded (torch, jax, etc.).
        """
        raise NotImplementedError

    def sync(value=None) -> None:
        """
        Synchronize the computation device.
        """
        raise NotImplementedError

    def clear_cache(size_mb: int = 128):
        """
        Flush the computation device cache by allocating and deallocating a buffer of the specified size in megabytes.
        """
        raise NotImplementedError


class NumpyTestBench(TestBench):
    def sync(value=None):
        return

    def clear_cache(size_mb: int = 128):
        n_elements = (size_mb * 1024 * 1024) // 8
        _ = np.zeros(n_elements, dtype=np.float64)
        _ = _ + 1.0
        return


class TorchTestBench(TestBench):
    def sync(value=None):
        import torch

        torch.cuda.synchronize()

    def clear_cache(size_mb: int = 128):
        import torch

        n_elements = (size_mb * 1024 * 1024) // 4
        _ = torch.zeros(n_elements, device="cuda", dtype=torch.float32)
        _ = _ + 1.0
        torch.cuda.synchronize()
        return


class JaxTestBench(TestBench):
    def sync(value=None):
        if value is None:
            import jax

            jax.effects_barrier()
        elif type(value) is list or type(value) is tuple:
            for v in value:
                v.block_until_ready()
        else:
            value.block_until_ready()

    @staticmethod
    def clear_cache(size_mb: int = 128):
        import jax
        import jax.numpy as jnp

        n_elements = (size_mb * 1024 * 1024) // 4
        _ = jax.device_put(jnp.zeros(n_elements, dtype=jnp.float32))
        _ = _ + 1.0
        _.block_until_ready()
        return

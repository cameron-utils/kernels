import os
import statistics
import draccus
from dataclasses import dataclass, field
import numpy as np
from typing import Callable
from datetime import datetime
import pandas as pd

from utils import (
    in_sub_process,
    print_indented,
    NumpyTestBench,
    TorchTestBench,
    JaxTestBench,
)

# Benchmark and test classes for different GEMM implementations


class NumpyGemm(NumpyTestBench):
    variants = {
        "default": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, **kwargs) -> Callable:
        return lambda a, b: np.matmul(a, b, dtype=acc_dtype)

    def preprocess_test_data(a, b, comp_dtype: str, **kwargs) -> list:
        return [
            {"a": _a, "b": _b}
            for _a, _b in zip(a.astype(comp_dtype), b.astype(comp_dtype))
        ]

    def postprocess_test_data(results: list) -> dict:
        c = np.stack(results, axis=0).astype(np.float32)
        return {"c": c}

    def get_bench_data(M: int, N: int, K: int, comp_dtype: str, **kwargs) -> dict:
        a = np.random.randn(M, K).astype(comp_dtype)
        b = np.random.randn(K, N).astype(comp_dtype)
        return {"a": a, "b": b}

    @classmethod
    @in_sub_process
    def get_ref_data(
        cls,
        M: int,
        N: int,
        K: int,
        comp_dtype: str,
        acc_dtype: str,
        iters: int,
        variant: str = "default",
        variance: float = 1.0,
        **kwargs,
    ) -> dict:
        a = np.random.randn(iters, M, K) * variance
        b = np.random.randn(iters, K, N) * variance
        fn = cls.get_fn(comp_dtype=comp_dtype, acc_dtype=acc_dtype, variant=variant)
        c = fn(a.astype(comp_dtype), b.astype(comp_dtype)).astype(np.float32)
        return {"a": a, "b": b, "c": c}


class TorchGemm(TorchTestBench):
    variants = {
        "default": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "bfloat16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, **kwargs) -> Callable:
        import torch

        # Internally torch matmul supposedly uses float32 accumulation for float16 inputs and then casts back to float16
        # Seems to still be the case for fp8 and fp4 on Blackwell Tensor cores but have not verified
        def fn(a, b):
            return torch.matmul(a, b).to(getattr(torch, acc_dtype))

        return fn

    def preprocess_test_data(a, b, comp_dtype: str, **kwargs) -> list:
        import torch

        return [
            {
                "a": torch.tensor(_a, device="cuda", dtype=getattr(torch, comp_dtype)),
                "b": torch.tensor(_b, device="cuda", dtype=getattr(torch, comp_dtype)),
            }
            for _a, _b in zip(a, b)
        ]

    def postprocess_test_data(results: list) -> dict:
        import torch

        c = torch.stack(results, axis=0).to("cpu", torch.float32).numpy()
        return {"c": c}

    def get_bench_data(M: int, N: int, K: int, comp_dtype: str, **kwargs) -> dict:
        import torch

        a = torch.randn(M, K, device="cuda", dtype=getattr(torch, comp_dtype))
        b = torch.randn(K, N, device="cuda", dtype=getattr(torch, comp_dtype))
        return {"a": a, "b": b}

    @classmethod
    @in_sub_process
    def get_ref_data(
        cls,
        M: int,
        N: int,
        K: int,
        comp_dtype: str,
        acc_dtype: str,
        iters: int,
        variant: str = "default",
        variance: float = 1.0,
        **kwargs,
    ) -> dict:
        import torch

        a = torch.randn(iters, M, K) * variance
        b = torch.randn(iters, K, N) * variance
        fn = cls.get_fn(comp_dtype=comp_dtype, acc_dtype=acc_dtype, variant=variant)
        c = fn(
            a.to("cuda", getattr(torch, comp_dtype)),
            b.to("cuda", getattr(torch, comp_dtype)),
        ).to("cpu", torch.float32)
        a = a.numpy()
        b = b.numpy()
        c = c.numpy()
        return {"a": a, "b": b, "c": c}


class JaxGemm(JaxTestBench):
    variants = {
        "default": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "bfloat16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, **kwargs) -> Callable:
        import jax
        import jax.numpy as jnp

        def fn(a, b):
            return jnp.matmul(a, b, preferred_element_type=getattr(jnp, acc_dtype))

        return jax.jit(fn)

    def preprocess_test_data(a, b, comp_dtype: str, **kwargs) -> list:
        import jax
        import jax.numpy as jnp

        return [
            {
                "a": jax.device_put(jnp.array(_a, dtype=comp_dtype)),
                "b": jax.device_put(jnp.array(_b, dtype=comp_dtype)),
            }
            for _a, _b in zip(a, b)
        ]

    def postprocess_test_data(results: list) -> dict:
        import jax.numpy as jnp
        import numpy as np

        c = jnp.stack(results, axis=0).astype(jnp.float32)
        return {"c": np.array(c)}

    def get_bench_data(M: int, N: int, K: int, comp_dtype: str, **kwargs) -> dict:
        import jax
        import jax.numpy as jnp

        a = jax.random.normal(jax.random.PRNGKey(0), (M, K)).astype(comp_dtype)
        b = jax.random.normal(jax.random.PRNGKey(1), (K, N)).astype(comp_dtype)
        return {"a": a, "b": b}

    @classmethod
    @in_sub_process
    def get_ref_data(
        cls,
        M: int,
        N: int,
        K: int,
        comp_dtype: str,
        acc_dtype: str,
        iters: int,
        variant: str = "default",
        variance: float = 1.0,
        **kwargs,
    ) -> dict:
        import jax
        import jax.numpy as jnp
        import numpy as np

        a = jax.random.normal(jax.random.PRNGKey(0), (iters, M, K)) * variance
        b = jax.random.normal(jax.random.PRNGKey(1), (iters, K, N)) * variance
        fn = cls.get_fn(comp_dtype=comp_dtype, acc_dtype=acc_dtype, variant=variant)
        c = fn(a.astype(comp_dtype), b.astype(comp_dtype)).astype(jnp.float32)
        return {"a": np.array(a), "b": np.array(b), "c": np.array(c)}


class CudaGemm(TorchGemm):
    variants = {
        "1_fp32_fp32": [
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            }
        ],
    }

    def get_fn(comp_dtype, **kwargs):
        import torch
        from kernels.gemm import cuda_gemm1_fp32_fp32 as gemm

        def fn(a, b):
            c = torch.empty(
                (a.shape[0], b.shape[1]), device="cuda", dtype=torch.float32
            )
            gemm(a, b, c)
            return c

        return fn


class TritonGemm(TorchGemm):
    variants = {
        "gemm1": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
        "gemm2": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, variant: str, **kwargs):
        if variant == "gemm1":
            from kernels.gemm import triton_gemm1 as gemm
        elif variant == "gemm2":
            from kernels.gemm import triton_gemm2 as gemm
        else:
            raise NotImplementedError(
                f"Unknown variant: {variant}, available: {list(TritonGemm.variants.keys())}"
            )

        return lambda a, b: gemm(a, b, acc_dtype=acc_dtype)


class TileLangGemm(TorchGemm):
    variants = {
        "default": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "bfloat16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, **kwargs):
        from kernels.gemm import tilelang_gemm as gemm

        return lambda a, b: gemm(a, b, acc_dtype=acc_dtype)


class TilusGemm(TorchGemm):
    variants = {
        "default": [
            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "bfloat16",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(acc_dtype: str, **kwargs):
        from kernels.gemm import tilus_gemm as gemm

        return lambda a, b: gemm(a, b, acc_dtype=acc_dtype)


class HelionGemm(TorchGemm):
    variants = {
        "default": [

            {
                "comp_dtype": "float16",
                "acc_dtype": "float16",
            },
            {
                "comp_dtype": "float16",
                "acc_dtype": "float32",
            },

            {
                "comp_dtype": "bfloat16",
                "acc_dtype": "float32",
            },
            {
                "comp_dtype": "float32",
                "acc_dtype": "float32",
            },
        ],
    }

    def get_fn(**kwargs):
        from kernels.gemm import helion_gemm as gemm

        return gemm


TARGETS = {
    "numpy": NumpyGemm,
    "torch": TorchGemm,
    "jax": JaxGemm,
    "cuda": CudaGemm,
    "triton": TritonGemm,
    "tilelang": TileLangGemm,
    "tilus": TilusGemm,
    "helion": HelionGemm,
}


# Configuration dataclasses


@dataclass
class TestConfig:
    run: bool = True  # Whether to run tests
    reuse_data: bool = True  # Whether to reuse existing test data
    save_data: bool = True  # Whether to save generated test data
    iters: int = 10  # Number of iterations for test data
    reference: str = "numpy"  # Reference target for correctness
    ref_variant: str | None = None  # Dont change
    ref_comp_dtype: str = "float32"  # Comp dtype for reference
    ref_acc_dtype: str = "float32"  # Acc dtype for reference
    atol: float = 1e-4  # Absolute tolerance for comparison
    rtol: float = 1e-4  # Relative tolerance for comparison
    ref_variance: float = 1.0  # Variance of the random data for reference


@dataclass
class BenchmarkConfig:
    run: bool = True  # Whether to run benchmarks
    run_ms: int = 100  # Duration to run each benchmark in milliseconds
    warmup_ms: int = 25  # Duration for warmup in milliseconds
    quantiles: list[float] | None = None  # Timing quantiles to report
    return_mode: str = "mean"  # How to aggregate timing results


@dataclass
class Config:
    M: list[int] | str = field(default_factory=lambda: [128])
    N: list[int] | str = field(default_factory=lambda: [128])
    K: list[int] | str = field(default_factory=lambda: [128])
    comp_dtype: list[str] | str = field(
        default_factory=lambda: ["float16", "bfloat16", "float32"]
    )  # Type used for computation
    acc_dtype: list[str] | str = field(
        default_factory=lambda: ["float16", "float32"]
    )  # Type used for accumulation

    targets: list[str] | str = "all"  # List of target names or "all"
    variants: dict[str, list[str] | str] = None  # Dont change
    # defined from targets using the following notation "target/variant1,target/variant2,..."

    save_results: bool = True  # Whether to save benchmark results to disk
    cpu_info: str | None = None
    ram_info: str | None = None
    gpu_info: str | None = None

    test: TestConfig = field(default_factory=TestConfig)
    bench: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self):
        if self.targets == "all":
            self.targets = list(TARGETS.keys())
        targets = []
        variants = {}
        for target_name in self.targets:
            targets.append(target_name)
            if "/" in target_name:
                target, variant = target_name.split("/", 1)
                if target in self.targets:
                    continue
                variants.setdefault(target, []).append(variant)
            else:
                variants[target_name] = TARGETS[target_name].variants
        self.targets = targets
        self.variants = variants
        assert all(
            [target in TARGETS for target in self.targets]
        ), f"Unknown targets: {set(self.targets) - set(TARGETS.keys())}, available targets: {list(TARGETS.keys())}"

        reference, ref_variant = (
            self.test.reference.split("/", 1)
            if "/" in self.test.reference
            else (self.test.reference, None)
        )
        if ref_variant is None:
            ref_variant = list(TARGETS[reference].variants)[0]
        assert (
            reference in TARGETS
        ), f"Unknown reference target: {reference}, available targets: {list(TARGETS.keys())}"
        assert (
            ref_variant in TARGETS[reference].variants
        ), f"Unknown variant: {ref_variant} for target {reference}, available variants: {TARGETS[reference].variants}"
        assert {
            "comp_dtype": self.test.ref_comp_dtype,
            "acc_dtype": self.test.ref_acc_dtype,
        } in TARGETS[reference].variants[ref_variant], (
            f"Unsupported dtype combination for reference target {reference}, variant {ref_variant}: "
            f"comp_dtype={self.test.ref_comp_dtype}, acc_dtype={self.test.ref_acc_dtype}, "
            f"supported: {TARGETS[reference].variants[ref_variant]}"
        )
        self.test.reference = reference
        self.test.ref_variant = ref_variant

        if isinstance(self.comp_dtype, str):
            self.comp_dtype = [self.comp_dtype]
        if isinstance(self.acc_dtype, str):
            self.acc_dtype = [self.acc_dtype]
        assert all(
            dtype in (supported_dtypes := ["float16", "bfloat16", "float32"])
            for dtype in self.comp_dtype
        ), f"Unsupported comp_dtype in {self.comp_dtype}, supported: {supported_dtypes}"
        assert all(
            dtype in (supported_dtypes := ["float16", "float32"])
            for dtype in self.acc_dtype
        ), f"Unsupported acc_dtype in {self.acc_dtype}, supported: {supported_dtypes}"

        assert self.bench.quantiles is None, f"Quantiles is not supported yet"
        assert self.bench.return_mode in (
            "mean",
            "min",
            "max",
            "median",
            "all",
        ), f"Unsupported return_mode: {self.bench.return_mode}"


@draccus.wrap()
def main(cfg: Config):
    print("GEMM Benchmark and Test Configuration:")
    print(cfg)

    results = {} if cfg.test.run and cfg.bench.run and cfg.save_results else None

    if cfg.test.run:
        print(
            f"Running GEMM tests with reference target: {cfg.test.reference}, variant: {cfg.test.ref_variant}, ref_comp_dtype: {cfg.test.ref_comp_dtype}, ref_acc_dtype: {cfg.test.ref_acc_dtype}"
        )

        for M, N, K in [(M, N, K) for M in cfg.M for N in cfg.N for K in cfg.K]:
            print_indented(
                f"Testing GEMM with M={M}, N={N}, K={K}",
                1,
            )

            def get_ref_data(
                M,
                N,
                K,
                comp_dtype,
                accum_dtype,
            ):
                reference = TARGETS[cfg.test.reference]
                reuse_data = cfg.test.reuse_data
                save_data = cfg.test.save_data
                iters = cfg.test.iters
                variant = cfg.test.ref_variant
                variance = cfg.test.ref_variance
                file_path = f"test/data/gemm_ref={cfg.test.reference}_variant={variant}_M={M}_N={N}_K={K}_comp_dtype={comp_dtype}_accum_dtype={accum_dtype}_iters={iters}_variance={variance}.npz"
                if reuse_data:
                    try:
                        ref_data = np.load(file_path)
                        print_indented("Loaded reference data from disk", 2)
                        return ref_data
                    except FileNotFoundError:
                        print_indented("Generating reference data", 2)
                ref_data = reference.get_ref_data(
                    M=M,
                    N=N,
                    K=K,
                    comp_dtype=comp_dtype,
                    acc_dtype=accum_dtype,
                    iters=iters,
                    variant=variant,
                    variance=variance,
                )
                if save_data:
                    if not os.path.exists("test/data"):
                        os.makedirs("test/data")
                    np.savez(
                        file_path,
                        **ref_data,
                    )
                    print_indented("Saved reference data to disk", 2)
                return ref_data

            ref_data = get_ref_data(
                M=M,
                N=N,
                K=K,
                comp_dtype=cfg.test.ref_comp_dtype,
                accum_dtype=cfg.test.ref_acc_dtype,
            )

            for target_name in cfg.targets:
                target = TARGETS[target_name]
                variants = cfg.variants[target_name]
                for variant in variants:
                    print_indented(f"Target: {target_name}, Variant: {variant}", 2)
                    for acc_dtype in cfg.acc_dtype:
                        for comp_dtype in cfg.comp_dtype:
                            print_indented(
                                f"Testing with comp_dtype={comp_dtype}, acc_dtype={acc_dtype}",
                                3,
                            )
                            if {
                                "comp_dtype": comp_dtype,
                                "acc_dtype": acc_dtype,
                            } not in target.variants[variant]:
                                print_indented(
                                    f"Skipping unsupported dtype combination: comp_dtype={comp_dtype}, acc_dtype={acc_dtype}",
                                    3,
                                )
                                continue
                            test_results = target.test(
                                **ref_data,
                                comp_dtype=comp_dtype,
                                acc_dtype=acc_dtype,
                                variant=variant,
                            )
                            cs = test_results["c"]
                            c_refs = ref_data["c"]
                            allclose = np.allclose(
                                cs, c_refs, atol=cfg.test.atol, rtol=cfg.test.rtol
                            )
                            if allclose:
                                print_indented("Test passed", 3)
                            else:
                                print_indented("Test failed", 3)
                                diff = np.abs(cs - c_refs)
                                max_diff = diff.max()
                                mean_diff = diff.mean()
                                print_indented(f"Max difference: {max_diff}", 4)
                                print_indented(f"Mean difference: {mean_diff}", 4)
                            if results is not None:
                                diff = np.abs(cs - c_refs)
                                max_diff = diff.max()
                                mean_diff = diff.mean()
                                results[
                                    f"{target_name}/{variant}_M={M}_N={N}_K={K}_comp_dtype={comp_dtype}_acc_dtype={acc_dtype}"
                                ] = {
                                    "datetime": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "target": target_name,
                                    "variant": variant,
                                    "M": M,
                                    "N": N,
                                    "K": K,
                                    "comp_dtype": comp_dtype,
                                    "acc_dtype": acc_dtype,
                                    "test_passed": allclose,
                                    "max_difference": float(max_diff),
                                    "mean_difference": float(mean_diff),
                                }

    if cfg.bench.run:
        print("Running GEMM benchmarks:")
        print(f"Target dtypes: comp_dtype={cfg.comp_dtype}, acc_dtype={cfg.acc_dtype}")
        for M, N, K in [(M, N, K) for M in cfg.M for N in cfg.N for K in cfg.K]:
            print_indented(
                f"Benchmarking GEMM with M={M}, N={N}, K={K}",
                1,
            )
            for target_name in cfg.targets:
                target = TARGETS[target_name]
                variants = cfg.variants[target_name]
                for variant in variants:
                    print_indented(f"Target: {target_name}, Variant: {variant}", 2)
                    for acc_dtype in cfg.acc_dtype:
                        for comp_dtype in cfg.comp_dtype:
                            print_indented(
                                f"Benchmarking with comp_dtype={comp_dtype}, acc_dtype={acc_dtype}",
                                3,
                            )
                            if {
                                "comp_dtype": comp_dtype,
                                "acc_dtype": acc_dtype,
                            } not in target.variants[variant]:
                                print_indented(
                                    f"Skipping unsupported dtype combination: comp_dtype={comp_dtype}, acc_dtype={acc_dtype}",
                                    3,
                                )
                                continue
                            times = target.bench(
                                M=M,
                                N=N,
                                K=K,
                                comp_dtype=comp_dtype,
                                acc_dtype=acc_dtype,
                                variant=variant,
                                run_ms=cfg.bench.run_ms,
                                warmup_ms=cfg.bench.warmup_ms,
                                quantiles=cfg.bench.quantiles,
                                return_mode=cfg.bench.return_mode,
                            )
                            if cfg.bench.quantiles is not None:
                                print_indented(
                                    f"Times (quantiles={cfg.bench.quantiles}): {times} ms",
                                    3,
                                )
                            else:
                                print_indented(f"Time: {times:.6f} ms", 3)
                            if results is not None:
                                if cfg.bench.return_mode == "all":
                                    results[
                                        f"{target_name}/{variant}_M={M}_N={N}_K={K}_comp_dtype={comp_dtype}_acc_dtype={acc_dtype}"
                                    ].update(
                                        {
                                            "time_min_ms": float(min(times)),
                                            "time_max_ms": float(max(times)),
                                            "time_mean_ms": float(
                                                statistics.mean(times)
                                            ),
                                            "time_median_ms": float(
                                                statistics.median(times)
                                            ),
                                        }
                                    )
                                else:
                                    results[
                                        f"{target_name}/{variant}_M={M}_N={N}_K={K}_comp_dtype={comp_dtype}_acc_dtype={acc_dtype}"
                                    ].update(
                                        {
                                            f"time_{cfg.bench.return_mode}_ms": float(
                                                times
                                            )
                                        }
                                    )

    if results is not None:
        extra_info = {
            "operation": "GEMM",
            "cpu_info": cfg.cpu_info,
            "ram_info": cfg.ram_info,
            "gpu_info": cfg.gpu_info,
            "reference": cfg.test.reference,
            "ref_variant": cfg.test.ref_variant,
            "ref_comp_dtype": cfg.test.ref_comp_dtype,
            "ref_acc_dtype": cfg.test.ref_acc_dtype,
            "ref_variance": cfg.test.ref_variance,
            "test_iters": cfg.test.iters,
        }
        if not os.path.exists("test/results"):
            os.makedirs("test/results")
        file_path = f"test/results/gemm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(list(results.values()))
        df = df.assign(**extra_info)
        print(f"Saving results to {file_path}")
        df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()

# Languages and Frameworks

## NumPy

Type: Python array library,
Devices: CPU,
[Homepage](https://numpy.org/),
[Repo](https://github.com/numpy/numpy)

NumPy is a Python array library with fast C based implementations of array operations.

## PyTorch

Type: Python deep learning library,
Devices: CPU, NVIDIA GPU, AMD GPU, Apple GPU, Intel GPU, Google TPU (via Torch XLA),
[Homepage](https://pytorch.org/),
[Repo](https://github.com/pytorch/pytorch)

PyTorch (Torch) is a popular open-source deep learning framework developed by Meta.
Provides an accelerated array library with automatic differentiation and built in utilities for building and training neural networks.
Also provides a compiler that lowers code into optimized Triton and other prebuilt kernels.

## JAX

Type: Python accelerated array library with automatic differentiation,
Devices: CPU, Google TPU, NVIDIA GPU, AMD GPU (via ROCm plugin), Intel GPU (experimental via oneAPI plugin), Apple GPU (experimental via Metal plugin),
[Homepage](https://jax.readthedocs.io/en/latest/),
[Repo](https://github.com/google/jax)

JAX is an accelerated array library developed by Google that provides automatic differentiation and just-in-time compilation via XLA.

## CuPy

Type: Python accelerated array library,
Devices: NVIDIA GPU, AMD GPU (experimental),
[Homepage](https://cupy.dev/),
[Repo](https://github.com/cupy/cupy)

CuPy is a GPU-accelerated array library for Python that provides a NumPy-like interface for NVIDIA GPUs and experimental support for AMD GPUs.

## CUDA C++

Type: CUDA C++,
Devices: NVIDIA GPU

For our purposes we reference to CUDA C++ as vanilla CUDA programming using the CUDA runtime API and CUDA C++ extensions.
C++ libraries and template frameworks such as CUTLASS are listed separately.

## Triton

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU, AMD GPU, Intel GPU (with XPU backend),
[Homepage](https://triton-lang.org/),
[Repo](https://github.com/triton-lang/triton)

Triton is an open-source GPU kernel compiler and python embedded language initially developed by OpenAI.
It operated at a higher level than CUDA, allowing users to write custom GPU kernels with less code and complexity but at the cost of detailed control of memory and execution.

## Triton Gluon

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU, AMD GPU,
[Repo](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py)

Gluon is a lower-level extension to Triton that provides more control over kernel execution and memory management while still leveraging Triton's abstractions.

## Triton Low-level Language Extensions (TLX)

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU,
[Repo](https://github.com/facebookexperimental/triton/tree/tlx)

TLX is another lower-level extension to Triton that exposes more of the underlying hardware capabilities and allows for more fine-grained optimizations.

## TileLang

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU,
[Homepage](https://tilelang.com/),
[Repo](https://github.com/tile-ai/tilelang)

A concise DSL for writing high-performance GPU kernels with a focus on tiling.
Build on top of the Apache TVM (Tensor Virtual Machine) stack.
Provides multiple abstraction levels including Tile programs, Tile libraries, and Thread Primitives.

Built kernels are cached and reused automatically based on kernel parameters.

## Tilus

Type: Python embedded kernel DSL,
[Homepage](https://nvidia.github.io/tilus/),
[Repo](https://github.com/NVIDIA/tilus)

Tilus kernel DSL with tiling abstractions, shared memory control and low precision support.

## Helion

Type: Python embedded kernel DSL,
Devices: (Same as Triton) NVIDIA GPU, AMD GPU, Intel GPU (with XPU backend),
[Homepage](https://helion-lang.org/),
[Repo](https://github.com/pytorch/helion)

Helion is a high-level DSL for writing Triton kernels with a PyTorch-like programming model.
It focuses on expressing computations in terms of tiles and relies on autotuning to optimize performance.
You can also implement various variants of algorithms and have Helion select the best one via autotuning.

## Pallas

Type: Python embedded DSL,
Devices: Google TPU, NVIDIA GPU, AMD GPU (via ROCm plugin),
[Homepage](https://docs.jax.dev/en/latest/pallas/index.html)

Pallas is a domain-specific language (DSL) for writing high-performance GPU and TPU kernels developed by Google and integrated into the JAX ecosystem.
Has 3 dialects:

- (Mosaic) TPU
- Mosaic GPU
- Triton GPU (Legacy)

## TileFusion

Type: C++ macro library,
Devices: NVIDIA GPU,
[Homepage](https://tiledtensor.github.io/tilefusion-docs/),
[Repo](https://github.com/microsoft/TileFusion)

## Warp

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU,
[Homepage](https://nvidia.github.io/warp/),
[Repo](https://github.com/NVIDIA/warp)

## ThunderKittens

Type: C++ library,
Devices: NVIDIA GPU,
[Repo](https://github.com/HazyResearch/ThunderKittens)

## CUTLASS C++

Type: C++ template library,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/cutlass/latest/overview.html),
[Repo](https://github.com/NVIDIA/cutlass)

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is a high-performance C++ template library for matrix-multiplication and related computations on NVIDIA GPUs.

## CuTe DSL

Type: Python embedded DSL,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html),
[Repo](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL),
[Examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL),
[Examples (Quack)](https://github.com/Dao-AILab/quack)

CuTe DSL is a domain-specific language embedded in Python for writing high-performance GPU kernels, built on top of CUTLASS.

## Mojo

Type: Programming language with Python interoperability,
Devices: CPU, NVIDIA GPU, AMD GPU,
[Homepage](https://docs.modular.com/mojo/manual/),
[Repo](https://github.com/modular/modular)

Mojo is a programming language developed by Modular that aims to combine the ease of use of Python with the performance of low-level languages like C++.
Mojo is not python but has python interoperability.

## Parrot C++

Type: C++ library,
Devices: NVIDIA GPU,
[Homepage](https://nvlabs.github.io/parrot/),
[Repo](https://github.com/NVlabs/parrot)

Parrot is a C++ library for fused array operations using CUDA/Thrust. It provides efficient GPU-accelerated operations with lazy evaluation semantics, allowing for chaining of operations without unnecessary intermediate materializations.

## PyCUDA

Type: Python wrapper for CUDA,
Devices: NVIDIA GPU,
[Homepage](https://documen.tician.de/pycuda/),
[Repo](https://github.com/inducer/pycuda)

## Numba

Type: Python JIT compiler,
Devices: CPU, NVIDIA GPU,
[Homepage](https://numba.pydata.org/),
[Repo](https://github.com/numba/numba)

## CuTile

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/cuda/cutile-python),
[Repo](https://github.com/NVIDIA/cutile-python)

## Mirage Persistent Kernel (MPK)

Type: Python embedded kernel DSL,
Devices: NVIDIA GPU, AMD GPU, Intel GPU (with XPU backend),
[Homepage](https://mirage-project.readthedocs.io/en/latest/index.html),
[Repo](https://github.com/mirage-project/mirage)

## CuBLAS

Type: C++ Linear Algebra library,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/cuda/cublas/),
[Examples](https://github.com/NVIDIA/CUDALibrarySamples)

## CuDNN Python

Type: Python wrapper for CuDNN,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/index.html)

## CuDNN C++

Type: C++ NN library,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/index.html)

## TileIR

Type: Intermediate representation,
Devices: NVIDIA GPU,
[Homepage](https://docs.nvidia.com/cuda/tile-ir/latest/index.html)

## Thrust

Type: C++ parallel algorithms library,
Devices: NVIDIA GPU,
[Homepage](https://nvidia.github.io/cccl/thrust/),
[Repo](https://github.com/nvidia/cccl)

## CUB

Type: Reusable C++ components for CUDA,
Devices: NVIDIA GPU,
[Homepage](https://nvidia.github.io/cccl/cub/index.html),
[Repo](https://github.com/nvidia/cccl)

## Bandicoot

Type: C++ library,
Devices: NVIDIA GPU,
[Homepage](https://coot.sourceforge.io/)

## TACO

Type: Tensor algebra compiler,
Devices: CPU, NVIDIA GPU,
[Homepage](http://tensor-compiler.org/)

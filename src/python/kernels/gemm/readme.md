# Gemm

This directory contains various implementations of the General Matrix Multiply (GEMM) operation.

```
M, N, K: int
InDType, OutDType: numerical data type
a: Array[M, K, InDType], b: Array[K, N, InDType] -> c: Array[M, N, OutDType]
```

## CUDA C++

- cuda_gemm: Naive implementation

## Triton

- triton_gemm1: Naive implementation
- triton_gemm2: Triton tutorial example

## TileLang

- tilelang_gemm: Basic TileLang implementation

## Tilus

- tilus_gemm: Basic Tilus implementation

## Helion

- helion_gemm: Basic Helion implementation

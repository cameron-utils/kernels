import torch
import tilelang
import tilelang.language as tl


@tilelang.jit
def build_gemm(
    M: int,
    N: int,
    K: int,
    BM: int = 64,
    BN: int = 64,
    BK: int = 64,
    comp_dtype: tl.dtype = tl.float32,
    acc_dtype: tl.dtype = tl.float32,
):

    @tl.prim_func
    def gemm_kernel(
        A: tl.Tensor((M, K), comp_dtype),  # type: ignore
        B: tl.Tensor((K, N), comp_dtype),  # type: ignore
        C: tl.Tensor((M, N), acc_dtype),  # type: ignore
    ):
        with tl.Kernel(
            tl.ceildiv(N, BN),
            tl.ceildiv(M, BM),
            threads=128,
        ) as (bx, by):
            A_s = tl.alloc_shared((BM, BK), comp_dtype)
            B_s = tl.alloc_shared((BK, BN), comp_dtype)
            C_f = tl.alloc_fragment((BM, BN), acc_dtype)
            tl.clear(C_f)

            for ko in tl.Pipelined(
                tl.ceildiv(K, BK),
                num_stages=3,
            ):
                tl.copy(
                    A[by * BM, ko * BK],
                    A_s,
                )
                tl.copy(
                    B[ko * BK, bx * BN],
                    B_s,
                )
                tl.gemm(A_s, B_s, C_f)

            tl.copy(C_f, C[by * BM, bx * BN])

    return gemm_kernel


def gemm(a, b, acc_dtype: tl.dtype = tl.float32):
    assert a.dim() == 2 and b.dim() == 2, "Input tensors must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.device == b.device, "Tensors must be on the same device"
    assert a.device.type == "cuda", "Tensors must be on CUDA device"
    assert a.dtype == b.dtype, "Tensors must have the same dtype"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    comp_dtype = str(a.dtype).split(".")[-1]

    gemm_kernel = build_gemm(
        M, N, K, comp_dtype=getattr(tl, comp_dtype), acc_dtype=getattr(tl, acc_dtype)
    )

    c = torch.empty((M, N), device=a.device, dtype=getattr(torch, acc_dtype))
    gemm_kernel(a, b, c)
    return c

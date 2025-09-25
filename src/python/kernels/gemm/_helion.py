import torch

import helion
import helion.language as hl


# @helion.kernel
# @helion.kernel(autotune_effort="quick")
@helion.kernel(autotune_effort="none")
def gemm(a: torch.Tensor, b: torch.Tensor, acc_dtype: str = "float32") -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matrix dimensions"
    assert a.device == b.device, "Tensors must be on the same device"
    assert a.dtype == b.dtype, "Tensors must have the same dtype"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"

    c = torch.zeros((M, N), device=a.device, dtype=getattr(torch, acc_dtype))

    for TM, TN in hl.tile((M, N)):
        acc = torch.zeros((TM, TN), device=a.device, dtype=getattr(torch, acc_dtype))

        for TK in hl.tile(K):
            acc = torch.addmm(acc, a[TM, TK], b[TK, TN])

        c[TM, TN] = acc

    return c

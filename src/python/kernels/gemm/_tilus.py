import torch
import tilus as tl


def get_gemm_builder(comp_dtype: str, acc_dtype: str):
    comp_dtype = getattr(tl, comp_dtype)
    acc_dtype = getattr(tl, acc_dtype)

    class Gemm(tl.Script):
        def __init__(self, BM=64, BN=64, BK=64):
            super().__init__()
            self.BM = BM
            self.BN = BN
            self.BK = BK

        def __call__(
            self,
            M: int,  # type: ignore
            N: int,  # type: ignore
            K: int,  # type: ignore
            a_ptr: ~comp_dtype,  # type: ignore
            b_ptr: ~comp_dtype,  # type: ignore
            c_ptr: ~acc_dtype,  # type: ignore
        ):
            self.attrs.blocks = (
                tl.utils.cdiv(M, self.BM),
                tl.utils.cdiv(N, self.BN),
            )
            self.attrs.warps = 4

            offset_m: tl.int32 = self.BM * self.blockIdx.x  # type: ignore
            offset_n: tl.int32 = self.BN * self.blockIdx.y  # type: ignore

            a_g = self.global_view(a_ptr, dtype=comp_dtype, shape=(M, K))
            b_g = self.global_view(b_ptr, dtype=comp_dtype, shape=(K, N))
            acc = self.register_tensor(
                dtype=acc_dtype, shape=(self.BM, self.BN), init=0.0
            )

            for offset_k in range(0, K, self.BK):
                a = self.load_global(
                    a_g,
                    offsets=(offset_m, offset_k),
                    shape=(self.BM, self.BK),
                )
                b = self.load_global(
                    b_g,
                    offsets=(offset_k, offset_n),
                    shape=(self.BK, self.BN),
                )
                self.dot(a, b, acc, out=acc, acc_dtype=acc_dtype)

            c_g = self.global_view(c_ptr, dtype=acc_dtype, shape=(M, N))
            self.store_global(c_g, acc, offsets=(offset_m, offset_n))

    return Gemm


build_gemm_fp16_fp16 = get_gemm_builder(comp_dtype="float16", acc_dtype="float16")
build_gemm_fp16_fp32 = get_gemm_builder(comp_dtype="float16", acc_dtype="float32")
build_gemm_bf16_fp32 = get_gemm_builder(comp_dtype="bfloat16", acc_dtype="float32")


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    acc_dtype: str = "float32",
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "Input tensors must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.device == b.device, "Tensors must be on the same device"
    assert a.device.type == "cuda", "Tensors must be on CUDA device"
    assert a.dtype == b.dtype, "Tensors must have the same dtype"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    comp_dtype = str(a.dtype).split(".")[-1]

    c = torch.empty((M, N), device=a.device, dtype=getattr(torch, acc_dtype))
    if comp_dtype == "float16" and acc_dtype == "float16":
        gemm_kernel = build_gemm_fp16_fp16()
    elif comp_dtype == "float16" and acc_dtype == "float32":
        gemm_kernel = build_gemm_fp16_fp32()
    elif comp_dtype == "bfloat16" and acc_dtype == "float32":
        gemm_kernel = build_gemm_bf16_fp32()
    else:
        raise ValueError(
            f"Unsupported combination of comp_dtype={comp_dtype} and acc_dtype={acc_dtype}"
        )

    gemm_kernel(M, N, K, a, b, c)
    return c

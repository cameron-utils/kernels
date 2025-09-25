#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    void gemm1_fp32_fp32(const float* A, const float* B, float* C,
        int M, int N, int K);

#ifdef __cplusplus
}
#endif

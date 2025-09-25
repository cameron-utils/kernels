#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launch_matmul(const float* A, const float* B, float* C,
                   int M, int N, int K);

#ifdef __cplusplus
}
#endif

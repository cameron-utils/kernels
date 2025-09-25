#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "gemm.h"

namespace nb = nanobind;

NB_MODULE(_cuda_gemm, m) {
    m.def("gemm1_fp32_fp32", [](nb::ndarray<const float, nb::shape<-1, -1>, nb::c_contig> A,
        nb::ndarray<const float, nb::shape<-1, -1>, nb::c_contig> B,
        nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig> C) {
            int M = A.shape(0);
            int K = A.shape(1);
            int N = B.shape(1);

            gemm1_fp32_fp32(A.data(), B.data(), C.data(), M, N, K);
        });
}

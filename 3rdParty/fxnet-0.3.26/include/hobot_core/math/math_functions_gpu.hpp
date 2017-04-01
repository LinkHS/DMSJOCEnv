/*
 * math_functions_gpu.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_TOOLS_MATH_FUNCTIONS_GPU_HPP_
#define HBOT_TOOLS_MATH_FUNCTIONS_GPU_HPP_

#include <float.h>
#include <math.h>  // for std::fabs and std::signbit
#include <stdint.h>
#include <cstring>

#include "hobot_core/base/base_common.hpp"
#include "hobot_core/base/base_device_alternate.hpp"
#include "hobot_core/math/math_lib_alternate.hpp"

namespace hbot {

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void hbot_gpu_scalar_max(const int N, const Dtype alpha,
    const Dtype* b, Dtype* y);
template <typename Dtype>
void hbot_gpu_scalar_min(const int N, const Dtype alpha,
    const Dtype* b, Dtype* y);


// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void hbot_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void hbot_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void hbot_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void hbot_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void hbot_gpu_copy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void hbot_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void hbot_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void hbot_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void hbot_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void hbot_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hbot_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hbot_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hbot_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void hbot_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hbot_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hbot_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void hbot_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// hbot_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void hbot_gpu_rng_uniform(const int n, unsigned int* r);

// hbot_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void hbot_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void hbot_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void hbot_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void hbot_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void hbot_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void hbot_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void hbot_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void hbot_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void hbot_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void hbot_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<HBOT_GET_BLOCKS(n), HBOT_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void hbot_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<HBOT_GET_BLOCKS(n), HBOT_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}  // namespace hbot


#endif /* HBOT_TOOLS_MATH_FUNCTIONS_GPU_HPP_ */

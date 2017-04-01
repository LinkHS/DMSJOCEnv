#ifndef HBOT_TOOLS_MATH_FUNCTIONS_H_
#define HBOT_TOOLS_MATH_FUNCTIONS_H_

#include <float.h>  // for std::fabs and std::signbit
#include <string.h>

#include <cmath>
#include "hobot_core/base/base_common.hpp"
#include "hobot_core/base/base_device_alternate.hpp"
#include "hobot_core/math/math_alan.hpp"
#include "hobot_core/math/math_functions_gpu.hpp"
#include "hobot_core/math/math_lib_alternate.hpp"

namespace hbot {

#define HBOT_PI 3.1415926535

inline int n_bit_int_upper_bound(int valid_bit_num) {
  return (static_cast<int>(1) << (valid_bit_num - 1)) - 1;
}

inline int n_bit_int_lower_bound(int valid_bit_num) {
  return (static_cast<int>(0xffffffff) << (valid_bit_num - 1));
}

// hbot gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void hbot_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

template <typename Dtype, typename DstType>
void hbot_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, DstType *C);

template <typename Dtype>
void hbot_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype* A, const Dtype* x,
                    const Dtype beta, Dtype* y);

template <typename Dtype>
void hbot_cpu_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void hbot_cpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

template <typename Dtype>
void hbot_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void hbot_set(const int N, const Dtype alpha, Dtype *X);

inline void hbot_memset(const size_t N, const int alpha, void *X) {
  memset(X, alpha, N);
}

template <typename Dtype>
void hbot_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void hbot_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype> void hbot_sqr(const int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void hbot_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void hbot_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void hbot_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void hbot_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void hbot_powx(const int n, const Dtype *a, const Dtype b, Dtype *y);

template <typename Dtype> void hbot_exp(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void hbot_log(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void hbot_abs(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void hbot_cpu_scalar_max(const int n, const Dtype scalar, const Dtype *x,
                          Dtype *y);

// mat should be stored in row-major order.
template <typename Dtype>
void hbot_cpu_mat_column_max(const int row, const int col, const Dtype *src,
                              Dtype *dst);

// mat should be stored in row-major order.
template <typename Dtype>
void hbot_cpu_mat_column_sum(const int row, const int col, const Dtype *src,
                              Dtype *dst);

template <typename Stype, typename Dtype>
void hbot_cpu_mat_column_sum(const int row, const int col, const Stype *src,
                              Dtype *dst);

double hbot_cpu_dot(const int n, const double *x, const double *y);

template <typename Dtype>
float hbot_cpu_dot(const int n, const Dtype *x, const Dtype *y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype> Dtype hbot_cpu_asum(const int n, const Dtype *x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename Dtype> inline int8_t hbot_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/hbot/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.

#define DEFINE_HBOT_CPU_UNARY_FUNC(name, operation)                           \
  template <typename Dtype>                                                    \
  void hbot_cpu_##name(const int n, const Dtype *x, Dtype *y) {               \
    assert(n > 0);                                                             \
    assert(x);                                                                 \
    assert(y);                                                                 \
    for (int i = 0; i < n; ++i) {                                              \
      operation;                                                               \
    }                                                                          \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_HBOT_CPU_UNARY_FUNC(sign, y[i] = hbot_sign<Dtype>(x[i]))

    // This returns a nonzero value if the input has its sign bit set.
    // The name sngbit is meant to avoid conflicts with std::signbit in the
    // macro.
    // The extra parens are needed because CUDA < 6.5 defines signbit as a
    // macro,
    // and we don't want that to expand here when CUDA headers are also
    // included.
DEFINE_HBOT_CPU_UNARY_FUNC(sgnbit,  \
    y[i] = static_cast<bool>((std::signbit)(x[i])))
DEFINE_HBOT_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))

template <typename Dtype>
void hbot_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

inline void hbot_cpu_scale(const int n, const float alpha, const int32_t *x, int32_t *y) {
  for (int i = 0; i < n; ++i) { y[i] = x[i]*alpha; }
}

template <typename Dtype>
void hbot_cpu_div(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

template <typename Dtype>
void hbot_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template <typename Dtype>
void hbot_rng_gaussian(const int n, const Dtype a, const Dtype sigma,
                        Dtype *r);

template <typename src_type, typename dst_type>
inline void hbot_cpu_axpb(const int n, float alpha, float beta,
                           const src_type *src, dst_type *dst) {
  hbot_cpu_axpb_kernel(n, alpha, beta, src, dst);
}

template void hbot_cpu_axpb<float, int32_t>(const int n, float alpha,
                                             float beta, const float *src,
                                             int32_t *dst);
template void hbot_cpu_axpb<double, int8_t>(const int n, float alpha,
                                             float beta, const double *src,
                                             int8_t *dst);
template void hbot_cpu_axpb<float, int8_t>(const int n, float alpha,
                                            float beta, const float *src,
                                            int8_t *dst);
template void hbot_cpu_axpb<int32_t, float>(const int n, float alpha,
                                             float beta, const int32_t *src,
                                             float *dst);
template void hbot_cpu_axpb<int8_t, float>(const int n, float alpha,
                                            float beta, const int8_t *src,
                                            float *dst);
template void hbot_cpu_axpb<float, float>(const int n, float alpha, float beta,
                                           const float *src, float *dst);
// template void hbot_cpu_axpb<int8_t, int8_t>(const int n, float alpha, float
// beta, const int8_t * src, int8_t * dst);
// template void hbot_cpu_axpb<int32_t, int32_t>(const int n, float alpha,
// float beta, const int32_t * src, int32_t * dst);
template void hbot_cpu_axpb<int32_t, double>(const int n, float alpha,
                                              float beta, const int32_t *src,
                                              double *dst);
// template void hbot_cpu_axpb<double, double>(const int n, float alpha, float
// beta, const double * src, double * dst);

inline void hbot_mat_mul(const int8_t *src1, const int8_t *src2, int32_t *dst,
                          int m, int n, int k, bool src1_trans = false,
                          bool src2_trans = false) {
  int8_t *src1_new = const_cast<int8_t *>(src1);
  int8_t *src2_new = const_cast<int8_t *>(src2);
  if (src1_trans == false && src2_trans == false)
    return hbot_cpu_matrix_mul_c8_i32(m, n, k, src1_new, src2_new, dst);
  else if (src1_trans == false && src2_trans == true)
    return hbot_cpu_matrix_mul_trans_c8_i32(m, n, k, src1_new, src2_new, dst);
  else
    NOT_IMPLEMENTED;
}

/**
 * return sum, and store min and max.
 */
template <typename Dtype, typename Ttype>
Ttype hbot_cpu_sum_min_max(const int n, const Dtype *x, Ttype *min = NULL,
                            Ttype *max = NULL);

template <typename Dtype>
inline void hbot_cpu_min_max(const int n, const Dtype *x, Dtype *min,
                              Dtype *max) {
  hbot_cpu_min_max_kernel(n, x, min, max);
}

template void hbot_cpu_min_max<float>(const int n, const float *x, float *min,
                                       float *max);
template void hbot_cpu_min_max<double>(const int n, const double *x,
                                        double *min, double *max);

static const float atan2_p1 =
    0.9997878412794807f * static_cast<float>(180 / HBOT_PI);
static const float atan2_p3 =
    -0.3258083974640975f * static_cast<float>(180 / HBOT_PI);
static const float atan2_p5 =
    0.1555786518463281f * static_cast<float>(180 / HBOT_PI);
static const float atan2_p7 =
    -0.04432655554792128f * static_cast<float>(180 / HBOT_PI);

inline float fastAtan2(float y, float x) {
  float ax = std::abs(x), ay = std::abs(y);
  float a, c, c2;
  if (ax >= ay) {
    c = ay / (ax + static_cast<float>(DBL_EPSILON));
    c2 = c * c;
    a = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
  } else {
    c = ax / (ay + static_cast<float>(DBL_EPSILON));
    c2 = c * c;
    a = 90.f -
        (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
  }
  if (x < 0)
    a = 180.f - a;
  if (y < 0)
    a = 360.f - a;
  if (a > 180.f)
    a -= 360.f;
  return a;
}

template <typename Dtype> inline int round(Dtype x) {
  return x > 0.0 ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5);
}

#define HBOT_2PI 6.283185307
#define HBOT_HALF_PI 1.57079632675

template <typename Dtype> inline Dtype sin(Dtype x) {
  while (x < -HBOT_PI)
    x += HBOT_2PI;
  while (x > HBOT_PI)
    x -= HBOT_2PI;

  Dtype x2 = x * x;
  Dtype x4 = x2 * x2;
  Dtype x6 = x4 * x2;

  const Dtype inv6 = Dtype(1) / 6;
  const Dtype inv120 = Dtype(1) / 120;
  const Dtype inv5040 = Dtype(1) / 5040;

  return x * (1 - inv6 * x2 + inv120 * x4 - inv5040 * x6);
}

template <typename Dtype> inline Dtype cos(Dtype x) {
  return sin(x + HBOT_HALF_PI);
}

template <typename Dtype>
Dtype hbot_rng_uniform(const Dtype a, const Dtype b);

template <typename Dtype>
Dtype hbot_rng_gaussian(const Dtype a, const Dtype sigma);



}  // namespace hbot

#endif  // HBOT_TOOLS_MATH_FUNCTIONS_H_

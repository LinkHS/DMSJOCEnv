/*
 * math_alan.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_TOOL_MATH_ALAN_HPP_
#define HBOT_TOOL_MATH_ALAN_HPP_



#include <stdint.h>
#include <algorithm>
#ifdef ARM
#include <arm_neon.h>

/**
 * src1: size of m*k
 * src2: size of k*n
 * dst:  size of m*n
 * buff: size of at least 8*k + 4*8
 */
void matrix_mul_c8_i32_kernel_neon(int m, int n, int k, int8_t* src1,
  int8_t * src2, int32_t * dst, int8_t* buff);

void matrix_mul_trans_c8_i32_kernel_neon(int m, int n, int k, int8_t* src1,
  int8_t * src2, int32_t * dst, int8_t* buff);


/**
 * @brief   vector scale: A[] = alpha * B[].
 *
 * @param   dst[out]    the result matrix A.
 *          src[in]     the input matrix B.
 *          alpha[in]   scale of B.
 *          elemCnt[in] number of elements to calc.
 *
 * @return  void.
 * @ By BoLi
 */
void neon_scale(float *dst,
    float *src,
    float alpha,
    int elemCnt);


/**
 * @brief   vector scale & accu: A[] = alpha * B[] + beta * A[].
 *
 * @param   dst[out]    the accumulating matrix A.
 *          src[in]     the input matrix B.
 *          alpha[in]   scale of B.
 *          beta[in]    scale of A.
 *          elemCnt[in] number of elements to calc.
 *
 * @return  void. by libo
 */
void neon_axpby(float *dst,
    float *src,
    float alpha,
    float beta,
    int elemCnt);



void libo_neon_matrixmul_4x8_c8_i32(int32_t * dst,
    int8_t * src1,
    int8_t * src2,
    int m,
    int k,
    int n);

void libo_neon_matrixmul_4x8_c8_i32_NT(int32_t * dst,
    int8_t * src1,
    int8_t * src2,
    int m,
    int k,
    int n);


template<typename Tdst, typename Tsrc>
void neon_axpb(Tdst *dst, const Tsrc* src, float alpha,
    float beta, int elemCnt);


template <typename Dtype>
void hbot_cpu_min_max_neon(const int n, const Dtype* x, Dtype* min, Dtype* max);

#endif


#define NAIVE_AXPB_TYPES(type_dst, type_src) \
inline void naive_axpb(type_dst *dst, const type_src * src,  \
    float alpha, float beta, int elemCnt) { \
  for (int i = 0; i < elemCnt; ++i) { \
    float result = static_cast<float>(src[i])*alpha + beta; \
    dst[i] = type_dst(result); \
  } \
}


#define NAIVE_AXPB_TYPES_BOUND(type_dst, type_src) \
inline void naive_axpb(type_dst *dst, const type_src * src, \
    float alpha, float beta, int elemCnt) { \
  for (int i = 0; i < elemCnt; ++i) { \
    float result = static_cast<float>(src[i])*alpha + beta; \
    dst[i] = type_dst(result+((result > 0) - (result < 0)) * 0.5); \
  } \
}



NAIVE_AXPB_TYPES(int32_t, float)
NAIVE_AXPB_TYPES(int8_t, float)
NAIVE_AXPB_TYPES(int8_t, double)
NAIVE_AXPB_TYPES(float, int32_t)
NAIVE_AXPB_TYPES(double, int32_t)
NAIVE_AXPB_TYPES(float, int8_t)
NAIVE_AXPB_TYPES(double, double)

NAIVE_AXPB_TYPES(float, float)
NAIVE_AXPB_TYPES(int32_t, int32_t)
NAIVE_AXPB_TYPES(int8_t, int8_t)





template <typename src_type, typename dst_type>
inline void hbot_cpu_axpb_kernel(const int n, float alpha, float beta,
     const src_type * src, dst_type * dst) {
#ifdef ARM
  neon_axpb(dst,  src, alpha, beta, n);
#else
  naive_axpb(dst,  src, alpha, beta, n);
#endif
}

template void hbot_cpu_axpb_kernel<float, int32_t>(const int n, float alpha,
    float beta, const float * src, int32_t * dst);
template void hbot_cpu_axpb_kernel<float, int8_t>(const int n, float alpha,
    float beta, const float * src, int8_t * dst);
template void hbot_cpu_axpb_kernel<int32_t, float>(const int n, float alpha,
    float beta, const int32_t * src, float * dst);
template void hbot_cpu_axpb_kernel<int8_t, float>(const int n, float alpha,
    float beta, const int8_t * src, float * dst);
template void hbot_cpu_axpb_kernel<float, float>(const int n, float alpha,
    float beta, const float * src, float * dst);
//  template void hbot_cpu_axpb_kernel<int8_t, int8_t>(const int n,
//     float alpha, float beta, const int8_t * src, int8_t * dst);
//  template void hbot_cpu_axpb_kernel<int32_t, int32_t>(const int n,
//     float alpha, float beta, const int32_t * src, int32_t * dst);
template void hbot_cpu_axpb_kernel<double, int8_t>(const int n, float alpha,
    float beta, const double * src, int8_t * dst);
template void hbot_cpu_axpb_kernel<int32_t, double>(const int n, float alpha,
    float beta, const int32_t * src, double * dst);
//  template void hbot_cpu_axpb_kernel<double, double>(const int n,
//     float alpha, float beta, const double * src, double * dst);


template <typename SrcT, typename DstT>
void matrix_mul_kernel_naive(int m, int n, int k, SrcT* src1,
    SrcT * src2, DstT* dst, DstT alpha = 1, DstT beta = 0) {
  for (int idx_i = 0; idx_i < m; ++idx_i) {
    for (int idx_j = 0; idx_j < n; ++idx_j) {
      DstT tmp = 0;
      for (int idx_k = 0; idx_k < k; ++idx_k) {
        tmp += (DstT)(src1[idx_i * k +idx_k]) *
            (DstT)(src2[idx_k * n + idx_j]);
      }
      dst[idx_i * n + idx_j] = dst[idx_i * n + idx_j]*beta + alpha * tmp;
    }
  }
}


template <typename SrcT, typename DstT>
void matrix_mul_trans_kernel_naive(int m, int n, int k, SrcT* src1,
    SrcT * src2, DstT* dst, DstT alpha = 1, DstT beta = 0) {
  for (int idx_i = 0; idx_i < m; ++idx_i) {
    for (int idx_j = 0; idx_j < n; ++idx_j) {
      DstT tmp = 0;
      for (int idx_k = 0; idx_k < k; ++idx_k) {
        tmp += (DstT)(src1[idx_i * k +idx_k]) *
            (DstT)(src2[idx_j * k + idx_k]);
      }
      dst[idx_i * n + idx_j] = dst[idx_i * n + idx_j] * beta + alpha*tmp;
    }
  }
}


inline void hbot_cpu_matrix_mul_c8_i32(int m, int n, int k, int8_t* src1,
  int8_t * src2, int32_t* dst) {
#ifdef ARM
  matrix_mul_c8_i32_kernel_neon(m, n, k, src1, src2, dst, NULL);
//  hobot_matrix_mul_c8_i32_kernel_neon(m, n, k, src1, src2, dst, 0);
//  libo_neon_matrixmul_4x8_c8_i32(dst, src1,  src2, m, k, n);
//  matrix_mul_kernel_naive(m, n, k, src1, src2, dst);
#else
  matrix_mul_kernel_naive(m, n, k, src1, src2, dst);
#endif
}

inline void hbot_cpu_matrix_mul_trans_c8_i32(int m, int n, int k, int8_t* src1,
    int8_t * src2, int32_t* dst) {
#ifdef ARM
  matrix_mul_trans_c8_i32_kernel_neon(m, n, k, src1, src2, dst, NULL);
//  libo_neon_matrixmul_4x8_c8_i32_NT(dst, src1,  src2, m, k, n);
//  matrix_mul_trans_kernel_naive(m, n, k, src1, src2, dst);
#else
  matrix_mul_trans_kernel_naive(m, n, k, src1, src2, dst);
#endif
}

template <typename Dtype>
void hbot_cpu_min_max_naive(const int n, const Dtype* x,
    Dtype* min_v, Dtype* max_v ) {
  for (int i = 0; i < n; ++i) {
    min_v[0] = (min_v[0] < x[i] ? min_v[0] : x[i]);
    max_v[0] = ((max_v[0]) > x[i] ? (max_v[0]) : x[i]);
  }
}

template void hbot_cpu_min_max_naive<float>(const int n, const float* x,
    float* min, float* max);
template void hbot_cpu_min_max_naive<double>(const int n, const double* x,
    double* min, double* max);


template <typename Dtype>
inline void hbot_cpu_min_max_kernel(const int n, const Dtype* x,
    Dtype* min, Dtype* max) {
#ifdef ARM
  hbot_cpu_min_max_neon(n, x, min, max);
//  hbot_cpu_min_max_naive(n, x, min, max);
#else
  hbot_cpu_min_max_naive(n, x, min, max);
#endif
}

template void hbot_cpu_min_max_kernel<float>(const int n, const float* x,
    float* min, float* max);
template void hbot_cpu_min_max_kernel<double>(const int n, const double* x,
    double* min, double* max);


#endif /*HBOT_TOOL_MATH_ALAN_HPP_ */

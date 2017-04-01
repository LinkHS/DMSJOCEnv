/*
 * math_functions_sse.hpp
 *
 *  Created on: 2015年11月26日
 *      Author: Alan_Huang
 */

#ifndef HBOT_TOOLS_MATH_FUNCTIONS_INTRIN_HPP_
#define HBOT_TOOLS_MATH_FUNCTIONS_INTRIN_HPP_

#ifdef __x86_64__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <algorithm>

#ifdef __AVX__
inline float hbot_cpu_fdot_avx(const int n, const float* x, const float* y) {
  float res = 0;
  int sse_iter = n/8 * 8;
  __m256 num1, num2, num3, num4;
  num4 = _mm256_setzero_ps();

  if (((size_t)x & 0x1F) == 0 && ((size_t)y & 0x1F) == 0) {
    for (int i = 0; i < sse_iter; i+= 8) {
      num1 = _mm256_load_ps(x+i);
      num2 = _mm256_load_ps(y+i);
      num3 = _mm256_mul_ps(num1, num2);
      num4 = _mm256_add_ps(num4, num3);
    }
  } else {
    for (int i = 0; i < sse_iter; i+= 8) {
      num1 = _mm256_loadu_ps(x+i);
      num2 = _mm256_loadu_ps(y+i);
      num3 = _mm256_mul_ps(num1, num2);
      num4 = _mm256_add_ps(num4, num3);
    }
  }

  __m128 hi = _mm256_extractf128_ps(num4, 1);
  __m128 lo = _mm256_extractf128_ps(num4, 0);
  lo = _mm_add_ps(hi, lo);
  hi = _mm_movehl_ps(hi, lo);
  lo = _mm_add_ps(hi, lo);
  hi = _mm_shuffle_ps(lo, lo, 1);
  lo = _mm_add_ss(hi, lo);
  res =  _mm_cvtss_f32(lo);
  for (int i = sse_iter; i < n; ++i) {
      res += x[i]*y[i];
  }
  return res;
}
#endif
#ifdef __SSE__
inline float hbot_cpu_fdot_sse(const int n, const float* x, const float* y) {
  float res = 0;
  int sse_iter = n/4 * 4;
  __m128 num1, num2, num3, num4;
  num4 = _mm_setzero_ps();
  num3 = _mm_setzero_ps();

  if (((size_t)x & 0xF) == 0 && ((size_t)y & 0xF) == 0) {
    for (int i = 0; i < sse_iter; i+=4) {
      num1 = _mm_load_ps(x+i);
      num2 = _mm_load_ps(y+i);
      num3 = _mm_mul_ps(num1, num2);
      num4 = _mm_add_ps(num4, num3);
    }
  } else {
    for (int i = 0; i < sse_iter; i+=4) {
      num1 = _mm_loadu_ps(x+i);
      num2 = _mm_loadu_ps(y+i);
      num3 = _mm_mul_ps(num1, num2);
      num4 = _mm_add_ps(num4, num3);
    }
  }
#ifdef __SSE3__
  num3 = _mm_hadd_ps(num4, num4);
  num4 = _mm_hadd_ps(num3, num3);
  _mm_store_ss(&res, num4);
#else
  float temp[4];
  _mm_storeu_ps(temp, num4);
  res = temp[0] + temp[1] + temp[2] + temp[3];
#endif  // end __SSE3__
  for (int i = sse_iter; i < n; ++i) {
      res += x[i]*y[i];
  }
  return res;
}
#endif
inline float hbot_cpu_fdot_naive(const int n, const float* x, const float* y) {
  float res = 0;
  for (int i = 0; i < n; ++i) {
    res += x[i]*y[i];
  }
  return res;
}

#ifdef __ARM_NEON__
inline float hbot_cpu_fdot_neon(const int n, const float* x, const float* y) {
  float res = 0;
  float acc_sum[4];
  int neon_iter = n/4 * 4;
  float32x4_t num3 = vdupq_n_f32(0.0);
  float32x4_t num1, num2;
  for (int i = 0; i < neon_iter; i +=4) {
    num1 = vld1q_f32(x+i);
    num2 = vld1q_f32(y+i);
    num3 = vmlaq_f32(num3, num1, num2);
  }
  vst1q_f32(acc_sum, num3);
  res += acc_sum[0] + acc_sum[1] + acc_sum[2]+ acc_sum[3];
  for (int i = neon_iter; i < n; ++i) {
    res += x[i]*y[i];
  }
  return res;
}
#endif



inline float hbot_cpu_fdot(const int n, const float* x, const float* y) {
#ifndef ARM
  #ifdef __AVX__
    return hbot_cpu_fdot_avx(n, x, y);
  #else  // if no __AVX__
    #ifdef __SSE__
      return hbot_cpu_fdot_sse(n, x, y);
    #else  // if no __SSE__
      return hbot_cpu_fdot_naive(n, x, y);
    #endif  // __SSE__
  #endif  // __AVX__

#else  // if ARM
  return hbot_cpu_fdot_neon(n, x, y);
#endif  // ARM
}

#ifdef __SSE__
inline void hbot_cpu_scalar_fmax_sse(const int n, const float scalar,
    const float* x, float* y) {
  float temp[4];
  int sse_iter = n/4 * 4;
  temp[0] = scalar, temp[1] = scalar, temp[2] = scalar, temp[3] = scalar;
  __m128 num1, num2;
  __m128 num_scalar = _mm_loadu_ps(temp);

  if (((size_t)x & 0xF) == 0 && ((size_t)y & 0xF) == 0) {
    for (int i = 0; i < sse_iter; i +=4) {
      num1 = _mm_load_ps(x+i);
      num2 = _mm_max_ps(num_scalar, num1);
      _mm_store_ps(y+i,  num2);
    }
  } else {
    for (int i = 0; i < sse_iter; i +=4) {
      num1 = _mm_loadu_ps(x+i);
      num2 = _mm_max_ps(num_scalar, num1);
      _mm_storeu_ps(y+i, num2);
    }
  }
  for (int i = sse_iter; i < n; ++i) {
    y[i] = std::max(scalar, x[i]);
  }
}
#endif
inline void hbot_cpu_scalar_fmax_naive(const int n, const float scalar,
    const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::max(x[i], scalar);
  }
}
#ifdef __ARM_NEON__
inline void hbot_cpu_scalar_fmax_neon(const int n, const float scalar,
    const float* x, float* y) {
  float temp[4];
  int neon_iter = n/4 * 4;
  temp[0] = scalar, temp[1] = scalar, temp[2] = scalar, temp[3] = scalar;
  float32x4_t num_scalar = vld1q_f32(temp);
  float32x4_t num1, num2;
  for (int i = 0; i < neon_iter; i +=4) {
    num1 = vld1q_f32(x+i);
    num2 = vmaxq_f32(num_scalar, num1);
    vst1q_f32(y+i, num2);
  }

  for (int i = neon_iter; i < n; ++i) {
    y[i] = std::max(scalar, x[i]);
  }
}
#endif

// y[i] =  max(scalar, x[i])
inline void hbot_cpu_scalar_fmax(const int n, const float scalar,
    const float* x, float* y) {
#ifndef ARM
  #ifdef __SSE__
  hbot_cpu_scalar_fmax_sse(n, scalar, x, y);
  #else
  hbot_cpu_scalar_fmax_naive(n, scalar, x, y);
  #endif
#else  // if ANDORID
  hbot_cpu_scalar_fmax_neon(n, scalar, x, y);
#endif
}


#ifdef __SSE__
inline void hbot_cpu_mat_column_fmax_sse(const int row, const int col,
    const float* src, float* dst) {
  int sse_col_iter = col/4 * 4;
  __m128 num1, num2;
  if (((size_t)src & 0xF) == 0 && ((size_t)dst & 0xF) == 0
      && col % 4 == 0) {
    for (int r = 0; r < row; ++r) {
      for (int c = 0; c < sse_col_iter; c +=4) {
        num1 = _mm_load_ps(src + r*col + c);
        num2 = _mm_load_ps(dst + c);
        num2 = _mm_max_ps(num2, num1);
        _mm_store_ps(dst + c,  num2);
      }
      for (int c = sse_col_iter; c < col ; ++c) {
        dst[c] = std::max(dst[c], src[r*col+c]);
      }
    }
  } else {
    for (int r = 0; r < row; ++r) {
      for (int c = 0; c < sse_col_iter; c +=4) {
        num1 = _mm_loadu_ps(src + r*col + c);
        num2 = _mm_loadu_ps(dst + c);
        num2 = _mm_max_ps(num2, num1);
        _mm_storeu_ps(dst + c, num2);
      }
      for (int c = sse_col_iter; c < col ; ++c) {
        dst[c] = std::max(dst[c], src[r*col +c]);
      }
    }
  }
}
#endif
inline void hbot_cpu_mat_column_fmax_naive(const int row, const int col,
    const float* src, float* dst) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; c++) {
      dst[c] = std::max(dst[c], src[r*col +c]);
    }
  }
}
#ifdef __ARM_NEON__
inline void hbot_cpu_mat_column_fmax_neon(const int row, const int col,
    const float* src, float* dst) {
  int sse_col_iter = col/4 * 4;
  float32x4_t num1, num2;
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < sse_col_iter; c +=4) {
      num1 = vld1q_f32(src + r*col + c);
      num2 = vld1q_f32(dst + c);
      num2 = vmaxq_f32(num2, num1);
      vst1q_f32(dst + c,  num2);
    }
    for (int c = sse_col_iter; c < col ; ++c) {
      dst[c] = std::max(dst[c], src[r*col +c]);
    }
  }
}
#endif
/*
 * dst[c] = std::max(dst[c], src[r*col +c]);
 */
inline void hbot_cpu_mat_column_fmax(const int row, const int col,
    const float* src, float* dst) {
#ifndef ARM
  #ifdef __SSE__
  hbot_cpu_mat_column_fmax_sse(row, col, src, dst);
  #else
  hbot_cpu_mat_column_fmax_naive(row, col, src, dst);
  #endif
#else  // if ANDORID
  hbot_cpu_mat_column_fmax_neon(row, col, src, dst);
#endif
}

#endif /* HBOT_TOOLS_MATH_FUNCTIONS_INTRIN_HPP_ */

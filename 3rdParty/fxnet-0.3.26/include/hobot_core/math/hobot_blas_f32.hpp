/*
 * hobot_blas_f32.hpp
 *    Header file of hobot_blas_f32
 *
 * Horizon Robotics Inc.'s implementation of Basic Linear Algebra Subprograms
 * for float32_t base type.  The implementation have been optimized for ARM NEON
 * architecture.
 * The implementation exists for the following reason:
 *    1. OpenBLAS only support -mfloat=hard while our code need to run on arm
 *       -mfloat=softfp platform usually. Therefore we create a version based
 *       on embeded assembly to walk around the softfp to hard fp ABI call
 *       issue.
 *    2. Sometimes we do not want to depend on the OpenBLAS implementation,
 *       therefore, we created our own implementation based on ARM NEON, which
 *       have better performance compare to naive implementation. But Openblas
 *       still have better performance currently.
 */

#ifndef HBOT_TOOL_HOBOT_BLAS_F32_HPP_
#define HBOT_TOOL_HOBOT_BLAS_F32_HPP_

#ifdef __cplusplus
extern "C" {
#endif

void hobot_sgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const float a,
                 const float *A, const int lda, const float *B, const int ldb,
                 const float b, float *C, const int ldc);

void hobot_sgemv(const int Order, const int Trans, const int M, const int N,
                 const float a, const float *x, const int lda, const float *y,
                 const int ldb, const float b, float *C, const int ldc);

// Scalar scaling
void hobot_sscal(const int N, const float a, float *x, const int incX);

void hobot_saxpy(const int N, const float a, const float *x, const int incX,
                 float *y, const int incY);

void hobot_saxpby(const int N, const float a, const float *x, const int incX,
                  const float b, float *y, const int incY);
float hobot_sasum(const int N, const float *x, const int incX);

float  hobot_sdot(const int n, const float  *x, const int incx, const float  *y,
                  const int incy);

void hobot_sger (const int order, const int M, const int N, const float alpha,
                 const float  *X, const int incX, const float  *Y, 
                 const int incY, float  *A, const int lda);


// The following double version of hobot BLAS implementation is only to make The
// code work as expect, please use OpenBLAS library if you need performance
void hobot_dscal(const int N, const double a, double *x, const int incX);

void hobot_dgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const double a,
                 const double *A, const int lda, const double *B, const int ldb,
                 const double b, double *C, const int ldc);

#ifdef __cplusplus
}
#endif
#endif /* HBOT_TOOL_HOBOT_BLAS_F32_HPP_ */

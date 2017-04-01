/*
 * hobot_blas_int8.hpp
 *    Header file of hobot_blas_int8
 *
 * Horizon Robotics Inc.'s implementation of Basic Linear Algebra Subprograms
 * for int8_t base type.  The implementation have been optimized for ARM NEON
 * architecture.
 *
 *
 */

#ifndef HBOT_TOOL_HOBOT_BLAS_INT8_HPP_
#define HBOT_TOOL_HOBOT_BLAS_INT8_HPP_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * BLAS interface for Int8 MatrixMultiplication with int32 result output.
 */
void hobot_i8_i32_gemm_inner(const int Order, const int TransA, const int TransB,
                       const int M, const int N, const int K, const int8_t a,
                       const int8_t *A, const int lda, const int8_t *B,
                       const int ldb, const int8_t b, int32_t *C,
                       const int ldc);
/**
 * BLAS interface for Int8 dot product with int32 result output.
 */
int32_t hobot_i8_i32_dot_inner(const int n, const int8_t *x, const int incx, const int8_t *y,
                  const int incy);

/**
 * BLAS interface for Int8 ger with int32 result output.
 */
void hobot_i8_i32_ger_inner (const int order, const int M, const int N, const int8_t alpha,
                 const int8_t  *X, const int incX, const int8_t *Y, 
                 const int incY, int32_t *A, const int lda);

/**
 * I8GEMM with shift support. 
 * Each input parameter has a shift property. When the value is positive, it
 * means that the corresponding float value of the param is shifted to right
 * by the given shift value, otherwise the float value is shifted to left.
 *
 * This function is required by MXNet for CNN quantization support (T360). 
 *
 * @param Order the majority order of the matrix. See OpenBlas interface 
 *              for details. currently only row major is supported.
 * @param TransA whether A is transposed.
 * @param TransB whether B is transposed.
 * @param M,N,K  the 2-D dim of the input matrix A and B
 * @param a,shift_a  the alpha value  and the corresponding shift bit count.
 * @param A, shift_A, lda  input matrix A and its corresponding shift bit count 
 *                         for each element.
 * @param B, shift_B, ldb  input matrix B and its corresponding shift bit count
 *                         for each element.
 * @param b, shift_b the beta value and its corresponding shift bit count.
 * @param C, shift_C, ldc  the input value of matrix C and its corresponding
 *                         shift bit count for each element. 
 * @param output_shift_given Whether the output C shift bit count is given.
 *                           If true, each gemm result element C will be shift
 *                           using the value stored in shift_C_output.
 *                           If false, the shift cnt is determined by the function
 *                           itself, and the shift cnt value will be put in the memory
 *                           pointed by shift_C_output.
 */
void hobot_i8_i8_gemm_shift_inner(const int Order, const int TransA, const int TransB,
                       const int M, const int N, const int K, const int8_t a,
                       const int8_t shift_a, const int8_t *A, const int8_t shift_A,
                       const int lda, const int8_t *B, const int8_t shift_B,
                       const int ldb, const int8_t b, const int8_t shift_b,
                       int8_t *C, const int8_t shift_C, bool output_shift_given,
                       const int8_t *shift_C_output, const int ldc);

#ifdef __cplusplus
}
#endif

#endif /* HBOT_TOOL_HOBOT_BLAS_INT8_HPP_ */

/*
 * base_device_alternate.hpp
 *      Author: Alan_Huang
 */

#ifndef HBOT_BASE_DEVICE_ALTERNATE_HPP_
#define HBOT_BASE_DEVICE_ALTERNATE_HPP_


#ifdef CPU_ONLY

#define NO_GPU std::cout << "Cannot use GPU in CPU-only Mode. " \
  << std::endl; std::abort()

#else
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <iostream>   // NOLINT

// CUDA macros
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { std::cout << " in " << __FILE__ << " " << \
      __LINE__ << " CUDA_CHECK:" << cudaGetErrorString(error) << std::endl; std::abort();} \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { std::cout << " in " << __FILE__ << \
      " " << __LINE__ << " CUBLAS_CHECK:" << hbot::cublasGetErrorString(status) \
      << std::endl; std::abort();}\
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    if (status != CURAND_STATUS_SUCCESS) { std::cout << " in " << __FILE__ << \
      " " << __LINE__ << " CURAND_CHECK:" << hbot::curandGetErrorString(status) \
      << std::endl;std::abort();}\
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


namespace hbot {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int HBOT_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int HBOT_GET_BLOCKS(const int N) {
  return (N + HBOT_CUDA_NUM_THREADS - 1) / HBOT_CUDA_NUM_THREADS;
}



}  // namespace hbot
#endif




#endif /* HBOT_BASE_DEVICE_ALTERNATE_HPP_ */

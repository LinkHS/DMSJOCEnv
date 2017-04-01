/*
 * base_common.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_BASE_COMMON_HPP_
#define HBOT_BASE_COMMON_HPP_

#ifdef DEBUG
#define DBG(CODE) CODE
#else
#define DBG(CODE)
#endif

#ifndef USE_CXX11
#define USE_CXX11 (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
                        __cplusplus >= 201103L || defined(_MSC_VER))
#endif

#if defined(USE_CXX11) && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 7
#define override
#endif
#endif

#define USE_POOL_MEM_THRED        32
#define RECYCLE_INTRALAYER_BLOB   0
#define RECYCLE_INTERLAYER_BLOB   0
#define RECYCLE_INPUT_BLOB        0
#if !defined(PAGE_SIZE)
#define PAGE_SIZE                 4096
#endif

//  #define MEM_POOL_PAGE_SIZE    4194304

#define MEM_POOL_PAGE_SIZE        PAGE_SIZE

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>  // NOLINT
#include "hobot_core/base/base_device_alternate.hpp"
//#include "hobot_core/base/logging.hpp"
#define assert_force(exp) if (!(exp)) std::cout << " in " << __FILE__ << " " \
  << __LINE__ << " check failed for "  << #exp << std::endl, std::abort()


// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED std::cout << " in " << __FILE__ << " " << __LINE__ \
  << "Not Implemented Yet"; std::abort()

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)



#ifdef HBOT_XINLINE
  #error  "HBOT_XINLINE must not be defined"
#endif

#ifdef _MSC_VER
#define HBOT_FORCE_INLINE __forceinline
//  #pragma warning(disable : 4068)
#else
#define HBOT_FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
  #define HBOT_XINLINE HBOT_FORCE_INLINE __device__ __host__
#else
  #define HBOT_XINLINE HBOT_FORCE_INLINE
#endif
#define HBOT_CINLINE HBOT_FORCE_INLINE

#ifdef __APPLE__
#include <sys/malloc.h>
inline void *memalign(size_t alignment, size_t size) {
  void * res;
  posix_memalign(&res, alignment, size);
  return res;
}
#else
#include <malloc.h>
#include <stdlib.h>
#endif

#if  defined(WINDOWS) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
#define memalign(alignment, size) _mm_malloc((size), (alignment))
#define memfree _mm_free
#else
#define memfree free
#endif

// UNUSED macro to get rid of unused warning since we need the code to be there.

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#define UNUSED_VAR  __attribute__ ((unused))
#endif


namespace hbot {

enum DataStorageOrder{
  STORATE_ORDER_NHWC = 0,
  STORATE_ORDER_NCHW = 1,
  STORATE_ORDER_UNKNOWN= 2
};

class HbotEngine {
 public:
  ~HbotEngine();
  static HbotEngine& Get();

  enum Brew { CPU, GPU };
  class RNG;
//  // Getters for boost rng, curand, and cublas handles
  inline  RNG& rng_stream() {
    return *random_generator_;
  }
#ifndef CPU_ONLY
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int _seed);
  static uint32_t cpu_rand_uint32();
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Parallel training info

 protected:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif

  RNG*   random_generator_;

  Brew mode_;

 private:
  // The private constructor to avoid duplicate instantiation.
  HbotEngine();

  DISABLE_COPY_AND_ASSIGN(HbotEngine);
};



}  // namespace hbot

#endif /* HBOT_BASE_COMMON_HPP_ */

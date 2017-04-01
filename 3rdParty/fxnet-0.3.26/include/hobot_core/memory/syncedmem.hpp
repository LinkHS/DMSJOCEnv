#ifndef HBOT_MEMORY_SYNCEDMEM_HPP_
#define HBOT_MEMORY_SYNCEDMEM_HPP_

#include <stdint.h>
#include <stdlib.h>
#include "hobot_core/base/base_common.hpp"
#include "hobot_core/math/math_functions.hpp"

namespace hbot {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

inline void MallocHost(void** ptr, size_t size , bool pined_malloc) {
#ifndef CPU_ONLY
  if (pined_malloc) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    return;
  }
#endif
  UNUSED(pined_malloc);
  *ptr = memalign(32, size);
  DBG(if (!(*ptr))
    std::cout << "host allocation of size " << size << " failed";)
  assert_force(*ptr);
}

inline void FreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#else
  UNUSED(use_cuda);
#endif
  memfree(ptr);
}


//  forward declair of ChunkAllocatedSyncedMemoryPool
class ChunkAllocatedSyncedMemoryPool;
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  friend class ChunkAllocatedSyncedMemoryPool;
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL),  size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t _size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(_size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {
  }
  ~SyncedMemory();

  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };

  inline const void* cpu_data() {
    to_cpu(); return (const void*)cpu_ptr_;
  }

  inline void set_cpu_data(void* data) {
    assert_force(data);
    if (own_cpu_data_) {
      FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    }
    cpu_ptr_ = data;
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = false;
  }

  inline const void* gpu_data() {
#ifndef CPU_ONLY
    to_gpu();
    return (const void*)gpu_ptr_;
#else
    NO_GPU;
    return NULL;
#endif
  }

  inline void set_gpu_data(void* data) {
#ifndef CPU_ONLY
    assert_force(data);
    if (own_gpu_data_) {
      int initial_device;
      cudaGetDevice(&initial_device);
      if (gpu_device_ != -1) {
        CUDA_CHECK(cudaSetDevice(gpu_device_));
      }
      CUDA_CHECK(cudaFree(gpu_ptr_));
      cudaSetDevice(initial_device);
    }
    gpu_ptr_ = data;
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = false;
#else
    UNUSED(data);
    NO_GPU;
#endif
  }

  inline void* mutable_cpu_data() {
    to_cpu();
    head_ = HEAD_AT_CPU;
    return cpu_ptr_;
  }

  inline void* mutable_gpu_data() {
#ifndef CPU_ONLY
    to_gpu();
    head_ = HEAD_AT_GPU;
    return gpu_ptr_;
#else
    NO_GPU;
    return NULL;
#endif
  }

  void share(const SyncedMemory* other_mem, size_t offset);

  inline SyncedHead head() const { return head_; }
  inline size_t size() const { return size_; }
  inline void set_size(size_t _size) {size_ = _size;}
  inline bool own_cpu_data() {return own_cpu_data_;}

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory



}  // namespace hbot

#endif  // HBOT_MEMORY_SYNCEDMEM_HPP_

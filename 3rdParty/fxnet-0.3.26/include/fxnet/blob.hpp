#ifndef FXNET_BLOB_HPP_
#define FXNET_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include "fxnet/common.hpp"
#include "fxnet/proto/fxnet.pb.h"
#include "hobot_core/memory/mem_management.hpp"
#include "hobot_core/ndarray/ndarray.hpp"
#include "hobot_core/memory/syncedmem.hpp"

namespace hbot {
namespace fxnet {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */

template <typename Dtype>
class Blob : public NDArray<Dtype>{
 public:

  Blob() : NDArray<Dtype>() {
    sum_per_instance_synced_flag_ = false;
    count_sum_per_instance_data_ = capacity_sum_per_instance_data_ =  0;
    shift_num_ = 0;
    current_shifted_num_ = 0;
  }
  ~Blob();
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int height, const int width,
      const int channels);
  explicit Blob(const vector<size_t>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  inline void Reshape(const int num, const int height, const int width,
      const int channels) {
    NDArray<Dtype>::Reshape(num, height, width, channels);
    this->count_sum_per_instance_data_ = this->shape_[0];
  }
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  inline void Reshape(const vector<size_t>& shape) {
    NDArray<Dtype>::Reshape(shape);
    this->count_sum_per_instance_data_ = this->shape_[0];
  }
  void Reshape(const BlobShape& shape);
  inline void ReshapeLike(const Blob& other) { Reshape(other.shape()); }

  /**
   * @brief Copy from a source Blob.
   *
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source,
      bool reshape = false);

  // @brief  recycle data
  void Recycle();

  // @allocate  data
  void Activate();

  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();


//  void Update();

  void FromProto(const BlobProto& proto, bool reshape = true,
      DataStorageOrder src_data_storage_order = STORATE_ORDER_NHWC);
  void ToProto(BlobProto* proto) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;

  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */

  bool ShapeEquals(const BlobProto& other,
      DataStorageOrder src_data_storage_order = STORATE_ORDER_NHWC);

  void sum_per_instance();
  inline const Dtype* sum_per_instance_cpu_data() const {
    assert_force(sum_per_instance_data_.mem());
    return (const Dtype*)sum_per_instance_data_.mem()->cpu_data();
  }

  inline bool sum_per_instance_synced_flag() {
    return sum_per_instance_synced_flag_;
  }

  inline float min_v() {
    return min_;
  }
  inline float max_v() {
    return max_;
  }

  // shift here is defined as left shift ( << )
  int shift_num_;
  int current_shifted_num_;

 protected:
  // by alan
  void FromProtoReshape(const BlobProto& proto, bool reshape,
      DataStorageOrder src_data_storage_order);
  void FromProtoDataOrderReArrange(DataStorageOrder src_data_storage_order);

  SharedSyncedMemory<USE_POOL_MEM_THRED>   sum_per_instance_data_;
  bool sum_per_instance_synced_flag_;
  size_t count_sum_per_instance_data_;
  size_t capacity_sum_per_instance_data_;
  Dtype min_, max_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob


template <typename Dtype>
std::ostream &operator<<(std::ostream &os,const Blob<Dtype> &inst);

}  // namespace fxnet

}  // namespace hbot

#endif  // FXNET_BLOB_HPP_

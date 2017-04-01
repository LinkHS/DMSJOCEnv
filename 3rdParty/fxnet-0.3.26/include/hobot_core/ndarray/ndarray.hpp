/*
 * ndarray.hpp
 *
 *  Created on: 2016年5月5日
 *      Author: Alan_Huang
 */

#ifndef HBOT_CORE_MEMORY_NDARRAY_HPP_
#define HBOT_CORE_MEMORY_NDARRAY_HPP_
#include <string>
#include <vector>
#include "hobot_core/base/logging.hpp"
#include "hobot_core/memory/mem_management.hpp"

namespace hbot {

template <int dim>
struct ShapeT;

template<int ndim>
inline std::ostream &operator<<(std::ostream &os, const ShapeT<ndim> &shape); // NOLINT(*)

template <int dim>
struct ShapeT {
  /*! \brief dimension of current shape */
  static const int Dimension = dim;
  static const int Subdim = dim - 1;
  size_t shape_[Dimension];
  inline ShapeT(void) {
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.  
    #pragma unroll
#endif
    for (int i = 0; i < Dimension; ++i) {
      this->shape_[i] = 0;
    }
  }
  /*! \brief constuctor */
  inline ShapeT(const ShapeT<Dimension> &s) {
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.  
    #pragma unroll
#endif
    for (int i = 0; i < Dimension; ++i) {
      this->shape_[i] = s[i];
    }
  }
  inline ShapeT(const std::vector<size_t> &s) {
    CHECK_EQ(s.size(), Dimension);
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.  
    #pragma unroll
#endif
    for (int i = 0; i < Dimension; ++i) {
      this->shape_[i] = s[i];
    }
  }
  inline size_t &operator[](size_t idx) {
    return shape_[idx];
  }
  inline const size_t &operator[](size_t idx) const {
    return shape_[idx];
  }
  inline bool operator==(const ShapeT<Dimension> &s) const {
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.  
    #pragma unroll
#endif
    for (int i = 0; i < Dimension; ++i) {
      if (s.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  inline bool operator!=(const ShapeT<Dimension> &s) const {
    return !(*this == s);
  }
  inline bool operator < (const ShapeT<Dimension> &s) const {
    for (int i = 0; i < Dimension; ++i) {
      if (s.shape_[i] < this->shape_[i]) return true;
      else if (s.shape_[i] == this->shape_[i]) continue;
      else return false;
    }
    return false;
  }
  inline size_t Count(void) const {
    size_t size = this->shape_[0];
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.  
    #pragma unroll
#endif
    for (int i = 1; i < Dimension; ++i) {
      size *= this->shape_[i];
    }
    return size;
  }

  template<int ndim>
  friend std::ostream &operator<<(std::ostream &os, const ShapeT<ndim> &shape); // NOLINT(*)
};

template<int ndim>
std::ostream &operator<<(std::ostream &os, const ShapeT<ndim> &shape) { // NOLINT(*)
  os << '(';
  for (int i = 0; i < ndim; ++i) {
    if (i != 0) os << ',';
    os << shape[i];
  }
  // python style tuple
  if (ndim == 1) os << ',';
  os << ')';
  return os;
}


inline ShapeT<1> Shape1(size_t s0) {
  ShapeT<1> s; s[0] = s0;
  return s;
}

inline ShapeT<2> Shape2(size_t s0, size_t s1) {
  ShapeT<2> s; s[0] = s0; s[1] = s1;
  return s;
}

inline ShapeT<3> Shape3(size_t s0, size_t s1, size_t s2) {
  ShapeT<3> s;
  s[0] = s0; s[1] = s1; s[2] = s2;
  return s;
}

inline ShapeT<4> Shape4(size_t s0, size_t s1,
    size_t s2, size_t s3) {
  ShapeT<4> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
  return s;
}

inline ShapeT<5> Shape5(size_t s0, size_t s1, size_t s2,
    size_t s3, size_t s4) {
  ShapeT<5> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; s[4] = s4;
  return s;
}

template <int dim, typename Dtype>
HBOT_XINLINE void Offset2Indices(Dtype offset,
    const Dtype* shape, Dtype* dst_indices) {
  for (int i = 0; i < dim; ++i) {
    int id = dim -1 -i;
    dst_indices[id] = offset % shape[id];
    offset /= shape[id];
  }
}

template <int dim, typename Dtype>
HBOT_XINLINE Dtype ShapeOffset(const Dtype* shape, const Dtype* indices) {
  Dtype _offset = indices[0];
  for ( int i = 1; i < dim; ++i){
    _offset *= shape[i];
    _offset += indices[i];
  }
  return _offset;
}

template <int dim>
inline size_t TensorShapeOffset(const ShapeT<dim>& shape,
    const ShapeT<dim>& indices) {
  CHECK_LT(indices[0], shape[0]);
  size_t _offset = indices[0];
  for ( int i = 1; i < dim; ++i){
    CHECK_LT(indices[i], shape[i]);
    _offset *= shape[i];
    _offset += indices[i];
  }
  return _offset;
}

template <int dim>
inline ShapeT<dim> OffsetToTensorIndices(const ShapeT<dim>& shape,
    size_t offset) {
  ShapeT<dim> indices;
  Offset2Indices<dim>(offset, shape.shape_, indices.shape_);
  CHECK_LT(indices[0], shape[0]);
  return indices;
}

const size_t kMaxNDArrayAxes = 32;

template <typename Dtype>
class NDArray;

template <typename Dtype>
std::ostream &operator<<(std::ostream &os, const NDArray<Dtype> &ndarray); // NOLINT(*)

template <typename Dtype>
class NDArray {
 public:
  NDArray() {
    count_ = capacity_ = 0;
  }
  ~NDArray();

  template <typename DataType>
  friend std::ostream &operator<<(std::ostream &os, const NDArray<DataType> &ndarray); // NOLINT(*)

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit NDArray(const int num, const int height,
      const int width, const int channels);
  explicit NDArray(const std::vector<size_t>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int height,
      const int width, const int channels);

  template<typename T>
  NDArray<T>* Cast() {
    count_ = count_ *sizeof(Dtype)/sizeof(T);
    capacity_ = capacity_ *sizeof(Dtype)/sizeof(T);
    shape_.clear();
    shape_.push_back(capacity_);
    return  reinterpret_cast<NDArray<T>*>(this);
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
  void Reshape(const std::vector<size_t>& shape);
  void ReshapeLike(const NDArray& other);
  inline bool ShapeEqual(const NDArray& other) {
    return other.shape_ == this->shape_;
  }
  inline std::string shape_string() const {
    std::ostringstream stream;
    for (size_t i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const std::vector<size_t>& shape() const { return shape_; }
  inline const std::vector<size_t>* shape_ptr() const { return &shape_ ; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline size_t shape(int index) const {
    return size_t(shape_[std::vector<int>::size_type(
        CanonicalAxisIndex(index))]);
  }
  inline size_t num_axes() const { return shape_.size(); }
  inline size_t count() const { return count_; }
  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline size_t count(size_t start_axis, size_t end_axis) const {
    CHECK(start_axis <= end_axis);
    CHECK(start_axis <= num_axes());
    CHECK(end_axis <= num_axes());
    size_t _count = 1;
    for (size_t i = start_axis; i < end_axis; ++i) {
      _count *= shape(i);
    }
    return _count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline size_t count(size_t start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline size_t CanonicalAxisIndex(int axis_index) const {
    CHECK(axis_index >= -static_cast<int>(num_axes())) << "axis " <<
        axis_index << " out of range for " << num_axes() <<
        "-D NDArray with shape " << shape_string();
    CHECK(axis_index < static_cast<int>(num_axes())) << "axis " <<
        axis_index << " out of range for " << num_axes() <<
        "-D NDArray with shape " << shape_string();
    if (axis_index < 0) {
      return size_t(axis_index + static_cast<int>(num_axes()));
    }
    return size_t(axis_index);
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline size_t num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(3) instead.
  inline size_t channels() const { return LegacyShape(3); }
  /// @brief Deprecated legacy shape accessor height: use shape(1) instead.
  inline size_t height() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor width: use shape(2) instead.
  inline size_t width() const { return LegacyShape(2); }
  inline size_t LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4) <<
        "Cannot use legacy accessors on NDArrays with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= static_cast<int>(num_axes()) ||
        index < - static_cast<int>(num_axes())) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  /**
   *   storage structure: (n, h, w, c); channel first.
   */
  inline size_t offset(const size_t n, const size_t h = 0, const size_t w = 0,
      const size_t c = 0) const {
    CHECK(n <= num());
    CHECK(c <= channels());
    CHECK(h <= height());
    CHECK(w <= width());
    return ((n * height() + h) * width() + w) * channels() + c;
  }

  inline size_t offset(const std::vector<size_t>& indices) const {
    size_t n_axes = num_axes();
    CHECK(indices.size() <= n_axes);
    size_t _offset = 0;

    for (size_t i = 0; i < n_axes; ++i) {
      _offset *= shape(i);
      if (indices.size() > i) {
        CHECK(indices[i] < shape(i));
        _offset += indices[i];
      }
    }
    return _offset;
  }
  /**
   * @brief Copy from a source NDArray.
   *
   * @param reshape if false, require this NDArray to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this NDArray to other's
   *        shape if necessary
   */
  void CopyFrom(const NDArray<Dtype>& source, bool gpu,
      bool reshape = false);

  template<typename SRC_TYPE>
  inline void CastCopyFrom(const NDArray<SRC_TYPE>& source, bool reshape = false) {
    if (source.count() != count_ || source.shape() != shape_) {
      if (reshape) { this->Reshape(source.shape()); }
      else { CHECK(0) << "Trying to copy blobs of different sizes."; }
    }
    const SRC_TYPE* src_data = source.cpu_data();
    Dtype* dst_data = this->mutable_cpu_data();
    for (size_t i = 0; i < this->count_; ++i) {
      dst_data[i] = src_data[i];
    }
  }

  inline Dtype data_at(const size_t n, const size_t h, const size_t w,
      const size_t c) const { return cpu_data()[offset(n, h, w, c)];  }

  inline Dtype data_at(const std::vector<size_t>& index) const {
    return cpu_data()[offset(index)];
  }

  // @brief  recycle data
  void Recycle();

  // @allocate  data
  void Activate();

  inline  SyncedMemory* data() {
    this->Activate();
    assert_force(data_.mem());
    return data_.mem();
  }

  void ShareData(const NDArray<Dtype>& other);
  const Dtype* cpu_data() const;
  const Dtype* gpu_data() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();

  inline void set_data(const SharedSyncedMemory<USE_POOL_MEM_THRED>& _data) {
    data_ = _data;
    capacity_ = _data.count();
  }

 protected:
  SharedSyncedMemory<USE_POOL_MEM_THRED> data_;
  std::vector<size_t> shape_;
  size_t count_;
  size_t capacity_;

};



}  // namespace hbot

#endif /* HBOT_CORE_MEMORY_NDARRAY_HPP_ */

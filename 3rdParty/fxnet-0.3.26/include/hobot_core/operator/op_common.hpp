/*
 * op_common.hpp
 *
 *  Created on: 2016年5月9日
 *      Author: Alan_Huang
 */

#ifndef HOBOT_CORE_OPERATOR_OP_COMMON_HPP_
#define HOBOT_CORE_OPERATOR_OP_COMMON_HPP_

#include <cstdlib>
#include <iostream>  // NOLINT
#include <string>
#include <vector>
#include "hobot_core/base/base_common.hpp"


namespace hbot {
namespace fxnet {

enum class OutShapeCalMode { FXNET, MXNET };
enum class ForwardMode { FLOAT, DOUBLE, INT8_TO_INT32, UNKNOWN};
enum class TensorDataType {UINT8,INT8, INT16, INT32, INT64, FLOAT, DOUBLE};
enum class MemSource {CPU, GPU};


#define HBOT_TYPE_SWITCH(type, DType, ...)          \
  switch (type) {                                   \
  case TensorDataType::UINT8:                       \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::INT8:                        \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::INT16:                       \
    {                                               \
      typedef int16_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::INT32:                       \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::INT64:                       \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::FLOAT:                       \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::DOUBLE:                     \
    {                                               \
      typedef double DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << static_cast<int>(type); \
  }


#define HBOT_REAL_TYPE_SWITCH(type, DType, ...)  \
  switch (type) {                                   \
  case TensorDataType::FLOAT:                       \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case TensorDataType::DOUBLE:                     \
    {                                               \
      typedef double DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                            \
    LOG(FATAL) << "Unknown type enum " << static_cast<int>(type); \
  }

typedef std::vector<size_t> TensorShape;

inline size_t TensorShapeOffset(const TensorShape& shape,
    const TensorShape& indices) {
  size_t n_axes = shape.size();
  assert_force(indices.size() <= n_axes);
  size_t _offset = 0;
  for (size_t i = 0; i < n_axes; ++i) {
    _offset *= shape[i];
    if (indices.size() > i) {
      assert_force(indices[i] < shape[i]);
      _offset += indices[i];
    }
  }
  return _offset;
}

#define DEF_TENSOR_SHAPE_OPERAROR(OP, REVERSE_OP) \
inline bool operator OP (const TensorShape&a, const TensorShape&b) { \
  if (a.size() != b.size()) {return false;}  \
  for (size_t i = 0; i < a.size(); ++i){  \
    if (a[i] REVERSE_OP b[i]) {return false;} \
  }           \
  return true;      \
}

DEF_TENSOR_SHAPE_OPERAROR(>=, <)
DEF_TENSOR_SHAPE_OPERAROR(>, <=)
DEF_TENSOR_SHAPE_OPERAROR(<=, >)
DEF_TENSOR_SHAPE_OPERAROR(<, >=)
DEF_TENSOR_SHAPE_OPERAROR(==, !=)

inline void FlattenTo4D(const TensorShape& src, TensorShape* dst) {
  dst->resize(4);
  for (int i = 0; i < 4; ++i) { (*dst)[i] = 1; }
  int last_dim = src.size() -1; int j = 3;
  for (int i = last_dim; i >= 0 && j >= 1; --i, --j) { (*dst)[j] = src[i]; }
  for (int i = 0; i < last_dim-2; ++i) { (*dst)[0] *= src[i]; }
}

class TensorDescriptor {
 public:
  TensorShape shape;
  TensorDataType data_type = TensorDataType::FLOAT;
  DataStorageOrder data_layout = STORATE_ORDER_NHWC;
  inline bool operator==(TensorDescriptor& other) {  //NOLINT
    return (shape == other.shape) && (data_type == other.data_type)
        && (data_layout == other.data_layout);
  }
  inline bool operator==(const TensorDescriptor& other) const {
    return (shape == other.shape) && (data_type == other.data_type)
        && (data_layout == other.data_layout);
  }
};

template <typename T>
struct Type2ID;
template<TensorDataType id>
struct ID2Type;

/*
template <typename T>
Type2ID<T> encode(const T&);
template <typename T>
Type2ID<T> encode(T&);
*/

template<typename Dtype>
TensorDataType GetTensorDataType() {
  return static_cast<TensorDataType>(sizeof(Type2ID<Dtype>)-1);
}


#define REGISTER_TensorDataType(type, n) \
    template <> \
    struct Type2ID<type> { char id[static_cast<int>(n)+1]; }; \
    template <> \
    struct ID2Type<n>    { typedef type type_t; };


REGISTER_TensorDataType(uint8_t,  TensorDataType::UINT8)
REGISTER_TensorDataType(int8_t,   TensorDataType::INT8)
REGISTER_TensorDataType(int16_t,  TensorDataType::INT16)
REGISTER_TensorDataType(int32_t,  TensorDataType::INT32)
REGISTER_TensorDataType(int64_t,  TensorDataType::INT64)
REGISTER_TensorDataType(float,    TensorDataType::FLOAT)
REGISTER_TensorDataType(double,   TensorDataType::DOUBLE)

inline size_t GetDataTypeSize(TensorDataType type) {
  size_t size = 0;
  switch (type) {
  case TensorDataType::INT8:
    size = sizeof(ID2Type<TensorDataType::INT8>::type_t);
    break;
  case TensorDataType::INT16:
    size = sizeof(ID2Type<TensorDataType::INT16>::type_t);
    break;
  case TensorDataType::INT32:
    size = sizeof(ID2Type<TensorDataType::INT32>::type_t);
    break;
  case TensorDataType::INT64:
    size = sizeof(ID2Type<TensorDataType::INT64>::type_t);
    break;
  case TensorDataType::FLOAT:
    size = sizeof(ID2Type<TensorDataType::FLOAT>::type_t);
    break;
  case TensorDataType::DOUBLE:
    size = sizeof(ID2Type<TensorDataType::DOUBLE>::type_t);
    break;
  default:
     std::cout << " in " << __FILE__ << " " << __LINE__ <<
       " Unknow TensorDataType";
     std::abort();
  }
  return size;
}

namespace op {

struct cpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = true;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 0;
};
/*! \brief device name CPU */
struct gpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = false;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1 << 1;
};

class DataDescriptor {
 public:
  std::vector<TensorDescriptor> input_desc;
  std::vector<TensorDescriptor> output_desc;
  std::vector<const void*> input_data;
  std::vector<void*> output_data;

  MemSource data_source = MemSource::CPU;

  bool operator==( DataDescriptor& other) {  //NOLINT
    return (input_desc == other.input_desc) &&
        data_source == other.data_source &&
        (output_desc == other.output_desc);
  }
  bool operator==(const DataDescriptor& other) const {
    return (input_desc == other.input_desc) &&
        data_source == other.data_source &&
        (output_desc == other.output_desc);
  }
};

class BaseOpExecutor {
 protected:
  template<typename Descriptor>
  inline bool CheckInputOuputData(const Descriptor& data_desc) {
    assert_force(data_desc.input_data.size() == data_desc.input_desc.size());
    assert_force(data_desc.output_data.size() == data_desc.output_desc.size());
    for (size_t i = 0; i < data_desc.input_data.size(); ++i) {
      const void* in_data = data_desc.input_data[i];
      void* out_data = data_desc.output_data[i];
      assert_force(in_data != NULL && in_data != nullptr);
      assert_force(out_data != NULL && out_data != nullptr);
    }
    return true;
  }
};

//template<int dim>
//friend std::ostream &operator<<(std::ostream &os, const Shape<dim> &shape); // NOLINT(*)

}  // namespace op
}  // namespace fxnet
}  // namespace hbot



#endif /* HOBOT_CORE_OPERATOR_OP_COMMON_HPP_ */

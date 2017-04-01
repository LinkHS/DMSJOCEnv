/*
 * operator.hpp
 *
 *  Created on: 2016年5月16日
 *      Author: Alan_Huang
 */

#ifndef FXNET_OPERATOR_HPP_
#define FXNET_OPERATOR_HPP_

#include "hobot_core/operator/op_common.hpp"

namespace hbot {
namespace fxnet {
namespace op {

enum class OpType{
  Undefined,
  Conv,
  Pooling,
  ReLU,
  Crop,
  HardNegMining
};

class OpDescriptor{
 public:
  explicit OpDescriptor():op_type(OpType::Undefined){}
  bool operator == (const OpDescriptor& other) const {
    return op_type == other.op_type;
  }

 protected:
  explicit OpDescriptor(OpType _op_type):op_type(_op_type){}
  OpType op_type;
};

class ParamDescriptor{
 public:
  explicit ParamDescriptor():op_type(OpType::Undefined){}
  bool operator == (const ParamDescriptor& other) const {
    return op_type == other.op_type;
  }

 protected:
  explicit ParamDescriptor(OpType _op_type):op_type(_op_type){}
  OpType op_type;
};

class BaseOp{
 public:
  virtual ~BaseOp(){}
  virtual void Setup(const DataDescriptor* data_desc,
      const OpDescriptor* op_desc = nullptr,
      const ParamDescriptor* param_desc = nullptr ) = 0;
  virtual void ForwardDataCPU(const DataDescriptor* data_desc,
      const ParamDescriptor* param_desc = nullptr) = 0;
#ifndef CPU_ONLY
  virtual void ForwardDataGPU(const DataDescriptor* data_desc,
      const ParamDescriptor* param_desc = nullptr) {
    NOT_IMPLEMENTED;
  }
#endif
  virtual void SuggestedOutputShape(const TensorDescriptor* input_desc,
      TensorDescriptor* output_desc,const OpDescriptor* op_desc = nullptr,
      const ParamDescriptor* param_desc = nullptr) = 0;
  virtual size_t InferWorkspaceSize(const ParamDescriptor* param_desc){
    return 0;
  }
};

BaseOp* CreateOperator (const char* type);

namespace conv {
class ConvOpDescriptor : public OpDescriptor{
 public:
  explicit ConvOpDescriptor():OpDescriptor(OpType::Conv) {}
  inline bool CheckOpType() const { return this->op_type == OpType::Conv; }
  TensorShape stride_shape;
  TensorShape pad_shape;
  bool forward_bias = true;

  bool operator==(const ConvOpDescriptor& other) const {
    return (stride_shape == other.stride_shape) &&
        (pad_shape == other.pad_shape)
        &&(forward_bias == other.forward_bias);
  }
};

class ConvParamDescriptor : public ParamDescriptor{
 public:
  explicit ConvParamDescriptor():ParamDescriptor(OpType::Conv) {}
  inline bool CheckOpType() const { return this->op_type == OpType::Conv; }
  TensorDescriptor kernel_desc;
  const void* kernel_data = nullptr;
  TensorDescriptor bias_desc;
  const void* bias_data = nullptr;

  void* buff_workspace = nullptr;
  int buff_workspace_size = 0;
  void* bias_multiplier = nullptr;
  int bias_multiplier_size = 0;
  int group = 1;
  bool weight_trans = false;
  bool operator==(const ConvParamDescriptor& other) const {
    return (kernel_desc == other.kernel_desc) && (bias_desc == other.bias_desc)
        && (buff_workspace_size == other.buff_workspace_size) &&
        (bias_multiplier_size == other.bias_multiplier_size)
        &&(group == other.group) && (weight_trans == other.weight_trans);
  }
};


class ConvOp : public BaseOp{
 public:
  ConvOp();
  virtual ~ConvOp();
  virtual void Setup(const DataDescriptor* data_desc, const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc ) override;
  virtual void ForwardDataCPU(const DataDescriptor* data_desc,
      const ParamDescriptor* param_desc) override;
  virtual void SuggestedOutputShape(const TensorDescriptor* input_desc,
      TensorDescriptor* output_desc,const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc) override;
  size_t InferWorkspaceSize(const ConvParamDescriptor& param_desc);
  virtual size_t InferWorkspaceSize(const ParamDescriptor* param_desc) override;
  void SetPadValue(int8_t v);
 protected:
  void*  exc;
};

}  // namespace conv


namespace pool {
enum class PoolMethod { MAX, AVE, UNKNOWN};

class PoolOpDescriptor : public OpDescriptor{
 public:
  explicit PoolOpDescriptor():OpDescriptor(OpType::Pooling) {}
  inline bool CheckOpType() const { return this->op_type == OpType::Pooling; }
  TensorShape kernel_shape;
  TensorShape pad_shape;
  TensorShape stride_shape;
  PoolMethod pool_method = PoolMethod::UNKNOWN;
  OutShapeCalMode out_shape_cal_mode = OutShapeCalMode::FXNET;

  bool operator==(const PoolOpDescriptor& other) const {
    return (kernel_shape == other.kernel_shape) &&
        (pad_shape == other.pad_shape) &&
        (stride_shape == other.stride_shape) &&
        (pool_method == other.pool_method) &&
        (out_shape_cal_mode == other.out_shape_cal_mode);
  }
};

class PoolingOp : public BaseOp{
 public:
  PoolingOp();
  virtual ~PoolingOp();
  virtual void Setup(const DataDescriptor* data_desc, const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc = nullptr) override;
  virtual void ForwardDataCPU(const DataDescriptor* data_desc,
      const ParamDescriptor* param_desc = nullptr) override;
  virtual void SuggestedOutputShape(const TensorDescriptor* input_desc,
      TensorDescriptor* output_desc,const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc = nullptr) override;
 protected:
  void* exc;
};

}  // namespace pool

namespace activation {

class ReLUOpDescriptor : public OpDescriptor{
 public:
  explicit ReLUOpDescriptor():OpDescriptor(OpType::ReLU) {}
  inline bool CheckOpType() const { return this->op_type == OpType::ReLU; }
  bool operator==(const ReLUOpDescriptor& other) const {
    return negative_slope == other.negative_slope;
  }
  float negative_slope = 0;
};

class ReLUOp : public BaseOp{
 public:
  ReLUOp();
  virtual ~ReLUOp();
  virtual void Setup(const DataDescriptor* data_desc, const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc = nullptr) override;
  virtual void ForwardDataCPU(const DataDescriptor* data_desc,
      const ParamDescriptor* param_desc = nullptr) override;
  virtual void SuggestedOutputShape(const TensorDescriptor* input_desc,
      TensorDescriptor* output_desc,const OpDescriptor* op_desc,
      const ParamDescriptor* param_desc = nullptr) override;
 protected:
  void* exc;
};

}  // namespace activation

namespace crop {
class CropOpDescriptor : public OpDescriptor{
 public:
  explicit CropOpDescriptor():OpDescriptor(OpType::Crop) {}
  inline bool CheckOpType() const { return this->op_type == OpType::Crop; }
  TensorShape crop_start;
  TensorShape crop_end;
  TensorShape dst_start;

  bool operator == (CropOpDescriptor& other) {  // NOLINT
    return (crop_start == other.crop_start) &&
        (crop_end == other.crop_end) &&(dst_start == other.dst_start);
  }
  bool operator == (const CropOpDescriptor& other) const {  // NOLINT
    return (crop_start == other.crop_start) &&
        (crop_end == other.crop_end) &&(dst_start == other.dst_start);
  }
};
}  // namespace crop


namespace sampling {

class HardNegMiningOpDescripter : public OpDescriptor {
 public:
  explicit HardNegMiningOpDescripter():OpDescriptor(OpType::HardNegMining) {}
  inline bool CheckOpType() const {
    return this->op_type == OpType::HardNegMining;
  }
  bool operator==(const HardNegMiningOpDescripter& other) const {
    return negative_ratio == other.negative_ratio &&
        hard_ratio == other.hard_ratio &&
        margin == other.margin &&
        ignore_largest_n == other.ignore_largest_n &&
        min_neg_num == other.min_neg_num &&
        value_if_masked == other.value_if_masked &&
        value_label_neg == other.value_label_neg &&
        value_label_positive == other.value_label_positive;
  }
  void CheckParam() const;

  float negative_ratio = 0.5;
  float hard_ratio = 0.5;
  int margin = 1;
  int ignore_largest_n = 0;
  int min_neg_num = 0;
  float value_if_masked = 0;
  float value_label_neg = 0;
  float value_label_positive = 1;
};

}  // namespace sampling

}  // namespace op
}  // namespace fxnet
}  // namespace hbot


#endif /* FXNET_OPERATOR_HPP_ */

#ifndef FXNET_QUANTIZATION_POOLING_LAYER_HPP_
#define FXNET_QUANTIZATION_POOLING_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layers/vision_layers.hpp"
#include "fxnet/layers/quantization_fixed_layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class QuantizationPoolingLayer : public PoolingLayer<Dtype> {
 public:
  explicit QuantizationPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param), PoolingLayer<Dtype>(param),
        fixer(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuantizationPooling"; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  LAYER_CREATOR(QuantizationPooling, Dtype);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  QuantizationFixedLayer<Dtype> fixer;
};

SET_LAYER_REGISTER_FLAG(QuantizationPooling);

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_QUANTIZATION_POOLING_LAYER_HPP_

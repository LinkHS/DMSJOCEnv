#ifndef FXNET_QUANTIZATION_UPSCALE_LAYER_HPP_
#define FXNET_QUANTIZATION_UPSCALE_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layers/upscale_layer.hpp"
#include "fxnet/layers/quantization_fixed_layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class QuantizationUpscaleLayer : public UpscaleLayer<Dtype> {
 public:
  explicit QuantizationUpscaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param), UpscaleLayer<Dtype>(param), fixer(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuantizationUpscale"; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  LAYER_CREATOR(QuantizationUpscale, Dtype);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  QuantizationFixedLayer<Dtype> fixer;
};

SET_LAYER_REGISTER_FLAG(QuantizationUpscale);

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_QUANTIZATION_UPSCALE_LAYER_HPP_

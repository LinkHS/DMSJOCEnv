#ifndef FXNET_FIXED_CONV_LAYER_HPP_
#define FXNET_FIXED_CONV_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

#include "fxnet/layers/vision_layers.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class FixedConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit FixedConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        ConvolutionLayer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FixedConvolution"; }
  LAYER_CREATOR(FixedConvolution, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  Dtype input_bound_scale_;
  Dtype weight_bound_scale_;


  Blob<Dtype> weight_backup_;

};
SET_LAYER_REGISTER_FLAG(FixedConvolution);
}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_FIXED_CONV_LAYER_HPP_

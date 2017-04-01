#ifndef FXNET_UPSCALE_LAYER_HPP_
#define FXNET_UPSCALE_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class UpscaleLayer : virtual public Layer<Dtype> {
 public:
  explicit UpscaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Upscale"; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  LAYER_CREATOR(Upscale, Dtype);
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

};

SET_LAYER_REGISTER_FLAG(Upscale);

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_UPSCALE_LAYER_HPP_

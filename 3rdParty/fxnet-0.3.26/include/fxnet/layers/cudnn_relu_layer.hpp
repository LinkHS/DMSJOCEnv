/*
 * cudnn_conv_layer.hpp
 *
 *      Author: Alan_Huang
 */



#ifndef FXNET_CUDNN_RELU_LAYER_HPP_
#define FXNET_CUDNN_RELU_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"
#include "fxnet/util/device_alternate.hpp"
#include "fxnet/layers/vision_layers.hpp"

namespace hbot {
namespace fxnet {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of ReLULayer.
 *        Fallback to ReLULayer for CPU mode.
 */
template <typename Dtype>
class CuDNNReLULayer : public ReLULayer<Dtype> {
 public:
  explicit CuDNNReLULayer(const LayerParameter& param)
      : Layer<Dtype>(param), ReLULayer<Dtype>(param),
        handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNReLULayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool handles_setup_;
  cudnnHandle_t             handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
};
#endif

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_CUDNN_RELU_LAYER_HPP_



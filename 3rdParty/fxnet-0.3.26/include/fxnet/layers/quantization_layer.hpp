/*
 * quantization_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_QUANTIZATION_LAYER_HPP_
#define FXNET_QUANTIZATION_LAYER_HPP_

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"
#include "fxnet/layers/neuron_layers.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class QuantizationLayer :  public NeuronLayer<Dtype> {
public:
  explicit QuantizationLayer(const LayerParameter& param)
      : Layer<Dtype>(param),NeuronLayer<Dtype>(param) {};
  virtual ~QuantizationLayer(){};
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Quantization"; }

  LAYER_CREATOR(Quantization, Dtype);
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool QuantizeInput(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  int valid_input_signed_int_bit_num_;
  int shift_num_queue_size_;
  Dtype saturate_rate_thred_;
};

SET_LAYER_REGISTER_FLAG(Quantization);
}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_QUANTIZATION_LAYER_HPP_ */

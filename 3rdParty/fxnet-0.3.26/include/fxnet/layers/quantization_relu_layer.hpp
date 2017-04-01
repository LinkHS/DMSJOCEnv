/*
 * quantization_relu_layer.hpp
 *
 *  Created on: 2016年4月6日
 *      Author: Alan_Huang
 */

#ifndef FXNET_QUANTIZATION_RELU_LAYER_HPP_
#define FXNET_QUANTIZATION_RELU_LAYER_HPP_

#include "fxnet/layers/neuron_layers.hpp"
#include "fxnet/layers/quantization_layer.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class QuantizationReLULayer : public QuantizationLayer<Dtype> {
public:

  explicit QuantizationReLULayer(const LayerParameter& param)
      : Layer<Dtype>(param),QuantizationLayer<Dtype>(param), relu_layer_(NULL) {
  }

  virtual ~QuantizationReLULayer(){
  	if(relu_layer_){
  		delete relu_layer_;
  	}
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuantizationReLU"; }
  LAYER_CREATOR(QuantizationReLU, Dtype);
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  ReLULayer<Dtype>* relu_layer_;

};

SET_LAYER_REGISTER_FLAG(QuantizationReLU);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_QUANTIZATION_RELU_LAYER_HPP_ */

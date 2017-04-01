/*
 * fpga_quantization_relu_layer.hpp
 *
 *  Created on: 2016年4月6日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_QUANTIZATION_RELU_LAYER_HPP_
#define FXNET_FPGA_QUANTIZATION_RELU_LAYER_HPP_

#include "fxnet/layers/neuron_layers.hpp"
#include "fxnet/fpga_layers/fpga_quantization_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAQuantizationReLULayer : public FPGAQuantizationLayer<Dtype>{
public:
  explicit FPGAQuantizationReLULayer(const LayerParameter& param)
      :Layer<int32_t>(param), FPGAQuantizationLayer<Dtype>(param),
       relu_layer_(NULL) {}

  virtual ~FPGAQuantizationReLULayer(){
  	if(relu_layer_){ delete relu_layer_; }
  }

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAQuantizationReLU"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("QuantizationReLU");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  FPGA_LAYER_CREATOR(FPGAQuantizationReLU, Dtype);

protected:
	ReLULayer<int32_t>* relu_layer_;

};

SET_LAYER_REGISTER_FLAG(FPGAQuantizationReLU);
}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_QUANTIZATION_RELU_LAYER_HPP_ */

/*
 * quantized_weight_conv_layer.hpp
 *
 *  Created on: 2016年3月31日
 *      Author: Alan_Huang
 */

#ifndef FXNET_QUANTIZED_WEIGHT_CONV_LAYER_HPP_
#define FXNET_QUANTIZED_WEIGHT_CONV_LAYER_HPP_

#include "fxnet/layers/vision_layers.hpp"


namespace hbot {
namespace fxnet {


template <typename Dtype>
class QuantizedWeightConvolutionLayer : public ConvolutionLayer<Dtype> {
public:

  explicit QuantizedWeightConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),ConvolutionLayer<Dtype>(param) {};

  virtual ~QuantizedWeightConvolutionLayer(){};
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual inline const char* type() const { return "QuantizedWeightConvolution"; }
  inline void set_int8_t_to_int32_t_forward(bool flag){
  	UNUSED(flag);
  	ConvolutionLayer<Dtype>::set_int8_t_to_int32_t_forward(false);
  	std::cout<<"int8_t_to_int32_t_forward is not supported in QuantizedWeightConvolutionLayer"<<std::endl;
  }
  LAYER_CREATOR(QuantizedWeightConvolution, Dtype);
protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool QuantizeWeight(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  bool fix_bias_term_;
  int valid_weight_signed_int_bit_num_;
  Dtype saturate_rate_thred_;
  int right_shift_num_queue_size_;

  bool is_quantized_;
  int output_shift_num_;

};

SET_LAYER_REGISTER_FLAG(QuantizedWeightConvolution);

}  //  namespace fxnet
}  //  namespace hbot

#endif /* FXNET_QUANTIZED_WEIGHT_CONV_LAYER_HPP_ */

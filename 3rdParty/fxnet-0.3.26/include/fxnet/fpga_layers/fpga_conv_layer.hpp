/*
 * conv_layer_fpga.hpp
 *
 *  Created on: 2016年4月5日
 *      Author: Alan_Huang
 */

#ifndef FXNET_CONV_LAYER_FPGA_HPP_
#define FXNET_CONV_LAYER_FPGA_HPP_

#include "fxnet/layers/vision_layers.hpp"
#include "fxnet/layers/quantized_weight_conv_layer.hpp"
#include "fxnet/fpga_layers/fpga_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAConvolutionLayer :  public FPGALayer<Dtype>,
  public BaseConvolutionLayer<int32_t> {
public:
  explicit FPGAConvolutionLayer(const LayerParameter& param)
      : Layer<int32_t>(param), FPGALayer<Dtype>(param),
        BaseConvolutionLayer<int32_t>(param){};
  virtual ~FPGAConvolutionLayer(){};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAConvolution"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("QuantizedWeightConvolution");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id );

  FPGA_LAYER_CREATOR(FPGAConvolution, Dtype);

protected:
  virtual void  SetParamBlobFrom(Layer<Dtype>& layer);
  virtual void  SetParamBlobTo(Layer<Dtype>& layer);
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top);

  void conv_forward_int32_t(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  int valid_weight_signed_int_bit_num_;
  int best_weight_shift_num_;
  int best_bias_shift_num_;
};

SET_LAYER_REGISTER_FLAG(FPGAConvolution);


} // namespace fxnet
}  //  namespace hbot


#endif /* FXNET_CONV_LAYER_FPGA_HPP_ */

/*
 * fpga_quantization_layer.hpp
 *
 *  Created on: 2016年4月6日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_QUANTIZATION_LAYER_HPP_
#define FXNET_FPGA_QUANTIZATION_LAYER_HPP_


#include "fxnet/fpga_layers/fpga_layer.hpp"
#include "fxnet/layers/neuron_layers.hpp"

namespace hbot {
namespace fxnet{

template<typename Dtype>
class FPGAQuantizationLayer : public FPGALayer<Dtype>, public NeuronLayer<int32_t>{
public:
	explicit FPGAQuantizationLayer(const LayerParameter& param)
	      : Layer<int32_t>(param),  FPGALayer<Dtype>(param),
	        NeuronLayer<int32_t>(param){};
	virtual ~FPGAQuantizationLayer(){};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAQuantization"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Quantization");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void FPGAForward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<int32_t>*>& top);

  virtual int GetShiftNumForOutput(){ return need_shift_num_;}

  virtual void GetFloatParam(Blob<Dtype>& dst, int id){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }
  FPGA_LAYER_CREATOR(FPGAQuantization, Dtype);

 protected:
  virtual void SetParamBlobFrom(Layer<Dtype>& layer);
  virtual void SetParamBlobTo(Layer<Dtype>& layer);
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top);

  int valid_input_signed_int_bit_num_;
  int best_input_shift_num_;

  int need_shift_num_;

};

SET_LAYER_REGISTER_FLAG(FPGAQuantization);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_QUANTIZATION_LAYER_HPP_ */

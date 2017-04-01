/*
 * conv_layer_fpga.hpp
 *
 *  Created on: 2016年4月5日
 *      Author: Alan_Huang
 */

#ifndef FXNET_UPSCALE_LAYER_FPGA_HPP_
#define FXNET_UPSCALE_LAYER_FPGA_HPP_

#include "fxnet/layers/upscale_layer.hpp"
#include "fxnet/fpga_layers/fpga_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAUpscaleLayer : public FPGALayer<Dtype>,  public UpscaleLayer<int32_t> {
public:
  explicit FPGAUpscaleLayer(const LayerParameter& param)
      : Layer<int32_t>(param),FPGALayer<Dtype>(param), UpscaleLayer<int32_t>(param){};
  virtual ~FPGAUpscaleLayer(){};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAUpscale"; }
  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Upscale");
    ret.push_back("QuantizationUpscale");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  } ;

  FPGA_LAYER_CREATOR(FPGAUpscale, Dtype);

 protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top);
};

SET_LAYER_REGISTER_FLAG(FPGAUpscale);

} // namespace fxnet
}  //  namespace hbot


#endif /* FXNET_UPSCALE_LAYER_FPGA_HPP_ */

/*
 *  Created on: 2016年4月5日
 *      Author: Alan_Huang
 */

#ifndef FXNET_RELU_LAYER_FPGA_HPP_
#define FXNET_RELU_LAYER_FPGA_HPP_

#include "fxnet/layers/vision_layers.hpp"
#include "fxnet/layers/vision_layers.hpp"
#include "fxnet/fpga_layers/fpga_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAReLULayer : public FPGALayer<Dtype>, public ReLULayer<int32_t> {
public:
  explicit FPGAReLULayer(const LayerParameter& param)
      : Layer<int32_t>(param),FPGALayer<Dtype>(param),
        ReLULayer<int32_t>(param){};
  virtual ~FPGAReLULayer(){};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAReLU"; }
  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("ReLU");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }

  FPGA_LAYER_CREATOR(FPGAReLU, Dtype);
 protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top);

};

SET_LAYER_REGISTER_FLAG(FPGAReLU);

} // namespace fxnet
}  //  namespace hbot


#endif /* FXNET_RELU_LAYER_FPGA_HPP_ */

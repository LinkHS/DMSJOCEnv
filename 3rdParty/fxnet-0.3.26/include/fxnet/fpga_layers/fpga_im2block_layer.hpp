/*
 * fpga_im2block_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_IM2BLOCK_LAYER_HPP_
#define FXNET_FPGA_IM2BLOCK_LAYER_HPP_

#include "fxnet/fpga_layers/fpga_layer.hpp"
#include "fxnet/layers/im2block_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAIm2BlockLayer :  public FPGALayer<Dtype>, public Im2BlockLayer<int32_t>{
public:
	explicit FPGAIm2BlockLayer(const LayerParameter& param):
		Layer<int32_t>(param), FPGALayer<Dtype>(param), Im2BlockLayer<int32_t>(param){}

	virtual ~FPGAIm2BlockLayer() {};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAIm2Block"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Im2Block");
    return ret;
  }
  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }

  FPGA_LAYER_CREATOR(FPGAIm2Block, Dtype);

protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top) {
    Blob<int32_t>& input_blob = *( bottom[0]);
    Blob<int32_t>& output_blob = *(top[0]);
    output_blob.shift_num_ = input_blob.shift_num_;
  }

};

SET_LAYER_REGISTER_FLAG(FPGAIm2Block);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_IM2BLOCK_LAYER_HPP_ */

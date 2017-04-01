/*
 * fpga_im2block_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_BLOCK2IM_LAYER_HPP_
#define FXNET_FPGA_BLOCK2IM_LAYER_HPP_

#include "fxnet/fpga_layers/fpga_layer.hpp"
#include "fxnet/layers/block2im_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGABlock2ImLayer :  public FPGALayer<Dtype>, public Block2ImLayer<int32_t>{
public:
	explicit FPGABlock2ImLayer(const LayerParameter& param):
		Layer<int32_t>(param), FPGALayer<Dtype>(param), Block2ImLayer<int32_t>(param){}

	virtual ~FPGABlock2ImLayer() {};

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGABlock2Im"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Block2Im");
    return ret;
  }
  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }

  FPGA_LAYER_CREATOR(FPGABlock2Im, Dtype);

protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top) {
    Blob<int32_t>& input_blob = *( bottom[0]);
    Blob<int32_t>& output_blob = *(top[0]);
    output_blob.shift_num_ = input_blob.shift_num_;
  }

};

SET_LAYER_REGISTER_FLAG(FPGABlock2Im);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_BLOCK2IM_LAYER_HPP_ */

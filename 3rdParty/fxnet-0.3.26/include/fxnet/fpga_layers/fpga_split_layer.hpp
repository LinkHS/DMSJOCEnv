/*
 * fpga_split_layer.hpp
 *
 *  Created on: 2016年4月7日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_SPLIT_LAYER_HPP_
#define FXNET_FPGA_SPLIT_LAYER_HPP_

#include <vector>

#include "fxnet/layers/split_layer.hpp"
#include "fxnet/fpga_layers/fpga_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGASplitLayer :  public FPGALayer<Dtype>, public SplitLayer<int32_t>{
public:
	explicit FPGASplitLayer(const LayerParameter& param):
		Layer<int32_t>(param), FPGALayer<Dtype>(param),SplitLayer<int32_t>(param){}

  virtual inline const char* type() const { return "FPGASplit"; }
  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Split");
    return ret;
  }

  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }

  FPGA_LAYER_CREATOR(FPGASplit, Dtype);

 protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
    const vector<Blob<int32_t>*>& top){
    Blob<int32_t>& input_blob = *( bottom[0]);
    for(vector<Blob<int32_t>*>::size_type i=0; i < top.size(); ++i){
      Blob<int32_t>& output_blob = *( top[i]);
      output_blob.shift_num_ = input_blob.shift_num_;
    }
  }
};

SET_LAYER_REGISTER_FLAG(FPGASplit);


}  // namespace fxnet
}  //  namespace fxnet


#endif /* FXNET_FPGA_SPLIT_LAYER_HPP_ */

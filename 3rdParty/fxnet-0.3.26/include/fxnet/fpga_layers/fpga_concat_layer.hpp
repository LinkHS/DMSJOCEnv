/*
 * fpga_eltwise_layer.hpp
 *
 *  Created on: 2016年4月6日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_CONCAT_LAYER_HPP_
#define FXNET_FPGA_CONCAT_LAYER_HPP_

#include "fxnet/fpga_layers/fpga_layer.hpp"
#include "fxnet/layers/concat_layer.hpp"

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGAConcatLayer :  public FPGALayer<Dtype>, public ConcatLayer<int32_t>{
public:
	explicit FPGAConcatLayer(const LayerParameter& param):
		Layer<int32_t>(param), FPGALayer<Dtype>(param),
		ConcatLayer<int32_t>(param){}

	virtual ~FPGAConcatLayer();

  virtual void LayerSetUp(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual inline const char* type() const { return "FPGAConcat"; }

  virtual inline const std::vector<std::string> corres_ori_type() const {
    std::vector<std::string> ret;
    ret.push_back("Concat");
    return ret;
  }
  virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top);

  virtual void GetFloatParam(Blob<Dtype>& dst, int id ){
    UNUSED(dst); UNUSED(id);
    LOG(INFO)<<"NO Param in " << type() << "Layer.";
  }

  vector<int>  GetShiftNumVecForInput(){return input_shift_num_;}

//  FPGA_LAYER_CREATOR(FPGAConcat, Dtype);

protected:
  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
        const vector<Blob<int32_t>*>& top);

  int max_shifed_num_;

  void  AlignInputBlob(const vector<Blob<int32_t>*>& bottom);

  vector<Blob<int32_t>* > bottom_aligned_;
  vector<int> input_shift_num_;
};

//SET_LAYER_REGISTER_FLAG(FPGAConcat);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_CONCAT_LAYER_HPP_ */

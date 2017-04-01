/*
 * fpga_layer.hpp
 *
 *  Created on: 2016年4月6日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_LAYER_HPP_
#define FXNET_FPGA_LAYER_HPP_

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"

#include <vector>

namespace hbot {
namespace fxnet {

template<typename Dtype>
class FPGANet;

template<typename Dtype>
class FPGALayer: virtual public Layer<int32_t>{
public:
  friend class FPGANet<Dtype>;

	explicit FPGALayer(const LayerParameter& param):Layer<int32_t>(param){}

	virtual ~FPGALayer(){};

  virtual inline const std::vector<std::string> corres_ori_type() const = 0;
  virtual inline const char* type() const { return "FPGALayer"; }

	virtual void GetFloatRes(const Blob<int32_t>& src,Blob<Dtype>& dst) ;

	virtual void GetFloatParam(Blob<Dtype>& dst, int id) = 0;

	virtual int GetShiftNumForOutput(){ return 0; }

	virtual void FPGAForward_cpu(const vector<Blob<int32_t>*>& bottom,
	  const vector<Blob<int32_t>*>& top) = 0;
	virtual void FPGAForward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<int32_t>*>& top){
		UNUSED(bottom);
		UNUSED(top);
		NOT_IMPLEMENTED;
	}

 protected:
	virtual bool  IsLayerMatch(Layer<Dtype>& layer) {
    bool found_corres_layer = false;
    std::vector<std::string> corres_ori_type = this->corres_ori_type();
    for (int i = 0; i < corres_ori_type.size(); ++i) {
      found_corres_layer |= (string(layer.type())  == corres_ori_type[i]);
    }
    return found_corres_layer;
	}
  virtual void  SetParamBlobTo(Layer<Dtype>& layer) {
    CHECK(this->IsLayerMatch(layer));
  }

  virtual void  SetParamBlobFrom(Layer<Dtype>& layer) {
    CHECK(this->IsLayerMatch(layer));
  }

  virtual void SetBlobShiftBackNum(const vector<Blob<int32_t>*>& bottom,
      const vector<Blob<int32_t>*>& top) = 0;

};


}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_FPGA_LAYER_HPP_ */

/*
 * conv_op_layer.hpp
 *
 *  Created on: 2016年5月11日
 *      Author: Alan_Huang
 */

#ifndef FXNET_CONV_OP_LAYER_HPP_
#define FXNET_CONV_OP_LAYER_HPP_

#include <vector>
#include "fxnet/layers/vision_layers.hpp"
#include "hobot_core/operator/operator.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class ConvolutionOpLayer : public ConvolutionLayer<Dtype> {
 public:
  virtual ~ConvolutionOpLayer() {
    delete exc;
  }
  explicit ConvolutionOpLayer(const LayerParameter& param)
      : Layer<Dtype>(param), ConvolutionLayer<Dtype>(param) {
    exc = op::CreateOperator("ConvOp");
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

//  op::conv::ConvOp exc;
  op::BaseOp * exc;
  op::conv::ConvOpDescriptor conv_desc;
  op::conv::ConvParamDescriptor param_desc;
  op::DataDescriptor data_desc;
};


}  //  namespace fxnet
}  //  namespace hbot

#endif /* FXNET_CONV_OP_LAYER_HPP_ */

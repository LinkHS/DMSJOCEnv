/*
 * relu_op_layer.hpp
 *
 *  Created on: 2016年5月11日
 *      Author: Alan_Huang
 */

#ifndef FXNET_RELU_OP_LAYER_HPP_
#define FXNET_RELU_OP_LAYER_HPP_

#include <vector>
#include "fxnet/layers/neuron_layers.hpp"
#include "hobot_core/operator/operator.hpp"
namespace hbot {
namespace fxnet {

template <typename Dtype>
class ReLUOpLayer : public ReLULayer<Dtype> {
 public:
  explicit ReLUOpLayer(const LayerParameter& param)
      : Layer<Dtype>(param), ReLULayer<Dtype>(param) {
    exc = op::CreateOperator("ReLUOp");
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    ReLULayer<Dtype>::LayerSetUp(bottom, top);
    relu_desc.negative_slope =
        ReLULayer<Dtype>::layer_param_.relu_param().negative_slope();
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    ReLULayer<Dtype>::Reshape(bottom, top);
    data_desc.input_desc.clear();
    data_desc.output_desc.clear();
    for (int i = 0; i < top.size(); ++i) {
      TensorDescriptor input_desc;
      input_desc.data_type = GetTensorDataType<Dtype>();
      input_desc.shape = bottom[i]->shape();
      data_desc.input_desc.push_back(input_desc);

      TensorDescriptor output_desc;
      output_desc.data_type = GetTensorDataType<Dtype>();
      output_desc.shape = top[i]->shape();
      data_desc.output_desc.push_back(output_desc);
    }
    exc->Setup(&data_desc, &relu_desc);
  }
  virtual ~ReLUOpLayer() {
    delete exc;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    data_desc.input_data.clear();
    data_desc.output_data.clear();
    for (int i = 0; i < top.size(); ++i) {
      data_desc.input_data.push_back(bottom[i]->cpu_data());
      data_desc.output_data.push_back(top[i]->mutable_cpu_data());
    }
    exc->ForwardDataCPU(&data_desc);
  }

//  op::activation::ReLUOp exc;
  op::BaseOp* exc;
  op::activation::ReLUOpDescriptor relu_desc;
  op::DataDescriptor data_desc;
};

}  //  namespace fxnet
}  //  namespace hbot


#endif /* FXNET_RELU_OP_LAYER_HPP_ */

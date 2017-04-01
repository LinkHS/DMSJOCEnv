/*
 * pooling_op_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_POOLING_OP_LAYER_HPP_
#define FXNET_POOLING_OP_LAYER_HPP_

#include <vector>
#include "fxnet/layers/vision_layers.hpp"
#include "hobot_core/operator/operator.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class PoolingOpLayer : public PoolingLayer<Dtype> {
 public:
  explicit PoolingOpLayer(const LayerParameter& param)
      : Layer<Dtype>(param), PoolingLayer<Dtype>(param) {
    exc = op::CreateOperator("PoolingOp");
  }
  virtual ~PoolingOpLayer() {
    delete exc;
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

//  op::pool::PoolingOp exc;
  op::BaseOp * exc;
  op::pool::PoolOpDescriptor pool_desc;
  op::DataDescriptor data_desc;
};

}  //  namespace fxnet
}  //  namespace hbot



#endif /* FXNET_POOLING_OP_LAYER_HPP_ */

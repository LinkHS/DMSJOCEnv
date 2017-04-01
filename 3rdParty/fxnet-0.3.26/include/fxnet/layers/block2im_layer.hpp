#ifndef FXNET_BLOCK2IM_LAYER_HPP_
#define FXNET_BLOCK2IM_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layers/im2block_layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

/*
 * @bref Partition image into blocks.
 *       (n, h, w, c) = (n/block_num_h/block_num_w, h*block_num_h, w*block_num_w, c)
 */
template <typename Dtype>
class Block2ImLayer : virtual public Layer<Dtype> {
 public:
  explicit Block2ImLayer(const LayerParameter& param)
      : Layer<Dtype>(param), block_num_h_(1), block_num_w_(1) {}
  virtual ~Block2ImLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Block2Im"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  LAYER_CREATOR(Block2Im, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int block_num_h_;
  int block_num_w_;
  Blob<int> bottom_shape_;
  Blob<int> top_shape_;
};

SET_LAYER_REGISTER_FLAG(Block2Im);


}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_BLOCK2IM_LAYER_HPP_

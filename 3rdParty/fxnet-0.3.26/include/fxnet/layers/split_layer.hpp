#ifndef FXNET_SPLIT_LAYER_HPP_
#define FXNET_SPLIT_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SplitLayer : virtual public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Split"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  LAYER_CREATOR(Split, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int count_;
};
SET_LAYER_REGISTER_FLAG(Split);
}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_SPLIT_LAYER_HPP_

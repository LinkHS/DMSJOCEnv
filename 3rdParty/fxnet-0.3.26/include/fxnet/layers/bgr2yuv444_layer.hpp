/*
 * bgr2yuv444_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef BGR2YUV444_LAYER_HPP_
#define BGR2YUV444_LAYER_HPP_


#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"

namespace hbot {
namespace fxnet {


/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BGR2YUV444Layer : public Layer<Dtype> {
 public:
  explicit BGR2YUV444Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BGR2YUV444"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  LAYER_CREATOR(BGR2YUV444, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  Blob<Dtype> yuv_weight_;
};
SET_LAYER_REGISTER_FLAG(BGR2YUV444);

}  //  namespace fxnet
}  //  namespace hbot



#endif /* BGR2YUV444_LAYER_HPP_ */

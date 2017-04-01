#ifndef FXNET_NEURON_LAYERS_HPP_
#define FXNET_NEURON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"


namespace hbot {
namespace fxnet {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
template <typename Dtype>
class NeuronLayer : virtual  public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}

  virtual inline void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  	top[0]->Reshape(bottom[0]->shape());
  }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};


/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit ReLULayer(const LayerParameter& param)
      : Layer<Dtype>(param),NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ReLU"; }
//  LAYER_CREATOR(ReLU, Dtype);
 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = \max(0, x)
   *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed outputs are @f$ y = \max(0, x) + \nu \min(0, x) @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

};

//SET_LAYER_REGISTER_FLAG(ReLU);

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_NEURON_LAYERS_HPP_

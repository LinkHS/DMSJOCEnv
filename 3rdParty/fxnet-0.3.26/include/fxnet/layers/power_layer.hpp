/*
 * power_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef POWER_LAYER_HPP_
#define POWER_LAYER_HPP_



#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layers/neuron_layers.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

/**
 * @brief Computes @f$ y = (\alpha x + \beta) ^ \gamma @f$,
 *        as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
 *        and power @f$ \gamma @f$.
 */
template <typename Dtype>
class PowerLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides PowerParameter power_param,
   *     with PowerLayer options:
   *   - scale (\b optional, default 1) the scale @f$ \alpha @f$
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   *   - power (\b optional, default 1) the power @f$ \gamma @f$
   */
  explicit PowerLayer(const LayerParameter& param)
      : Layer<Dtype>(param),NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Power"; }

  LAYER_CREATOR(Power, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief @f$ \gamma @f$ from layer_param_.power_param()
  Dtype power_;
  /// @brief @f$ \alpha @f$ from layer_param_.power_param()
  Dtype scale_;
  /// @brief @f$ \beta @f$ from layer_param_.power_param()
  Dtype shift_;
  /// @brief Result of @f$ \alpha \gamma @f$
  Dtype diff_scale_;
};

SET_LAYER_REGISTER_FLAG(Power);

}  // namespace fxnet
}  //  namespace hbot


#endif /* POWER_LAYER_HPP_ */

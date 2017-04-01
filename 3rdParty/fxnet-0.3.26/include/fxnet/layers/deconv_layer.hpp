/*
 * deconv_layer.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_DECONV_LAYER_HPP_
#define FXNET_DECONV_LAYER_HPP_

#include "fxnet/layers/vision_layers.hpp"

namespace hbot {
namespace fxnet {

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ConvolutionLayer.
 *
 *   ConvolutionLayer computes each output value by dotting an input window with
 *   a filter; DeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
 *   reversed. DeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit DeconvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Deconvolution"; }
  LAYER_CREATOR(Deconvolution, Dtype);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

SET_LAYER_REGISTER_FLAG(Deconvolution);
}  //  namespace fxnet
}  //  namespace hbot

#endif /* FXNET_DECONV_LAYER_HPP_ */

#ifndef FXNET_COMMON_LAYERS_HPP_
#define FXNET_COMMON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/layer_factory.hpp"
#include "fxnet/layer.hpp"

#include "fxnet/layers/neuron_layers.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {


/**
 * @brief Reshapes the input Blob into flat vectors.
 *
 * Note: because this layer does not change the input values -- merely the
 * dimensions -- it can simply copy the input. The copy happens "virtually"
 * (thus taking effectively 0 real time) by setting, in Forward, the data
 * pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
 * and in Backward, the diff pointer of the bottom Blob to that of the top Blob
 * (see Blob::ShareDiff).
 */
template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Flatten"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
  LAYER_CREATOR(Flatten, Dtype);
 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times CHW \times 1 \times 1) @f$
   *      the outputs -- i.e., the (virtually) copied, flattened inputs
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

};
SET_LAYER_REGISTER_FLAG(Flatten);

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  	int8_t_to_int32_t_forward_ = false;
  }
//  virtual ~InnerProductLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  inline void set_int8_t_to_int32_t_forward(bool flag){
  	int8_t_to_int32_t_forward_ = flag;
  }
  inline bool int8_t_to_int32_t_forward(){
  	return int8_t_to_int32_t_forward_;
  }
  LAYER_CREATOR(InnerProduct, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  void TransWeight();

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

  // For int8_to_int32_t
  Blob<int8_t> weight_int8_t_;
  Blob<int8_t> input_int8_t_;
  Blob<int32_t> output_int32_t_;
  Blob<Dtype> bias_bar_;
  Blob<Dtype> weight_temp_;
  Blob<Dtype> weight_trans_;

  bool int8_t_to_int32_t_forward_;
  float weight_alpha_;


 private:
  void weight_to_int8_t();

};

SET_LAYER_REGISTER_FLAG(InnerProduct);



/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SliceLayer : public Layer<Dtype> {
 public:
  explicit SliceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Slice"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  int count_;
  int num_slices_;
  int slice_size_;
  int slice_axis_;
  vector<int> slice_point_;
};



}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_COMMON_LAYERS_HPP_

#ifndef FXNET_IM2BLOCK_LAYER_HPP_
#define FXNET_IM2BLOCK_LAYER_HPP_

#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

/*
 * @bref Partition image into blocks.
 *       (n, h, w, c) = (n*block_num_h*block_num_w, h/block_num_h, w/block_num_w, c)
 */
template <typename Dtype>
class Im2BlockLayer : virtual public Layer<Dtype> {
 public:
  explicit Im2BlockLayer(const LayerParameter& param)
      : Layer<Dtype>(param), block_num_h_(1), block_num_w_(1) {}
  virtual ~Im2BlockLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Im2Block"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  LAYER_CREATOR(Im2Block, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int block_num_h_;
  int block_num_w_;
  Blob<int> bottom_shape_;
  Blob<int> top_shape_;
};

SET_LAYER_REGISTER_FLAG(Im2Block);


struct Im2BlockMappingNHWC {
  template<typename Dtype>
  HBOT_XINLINE static void Map(const Dtype* src_indices,
      const Dtype& block_num_h, const Dtype& block_num_w,
      const Dtype& block_h, const Dtype& block_w, Dtype* dst_indices) {
    dst_indices[0] = src_indices[0] * block_num_h * block_num_w +
        (src_indices[1]/block_h) * block_num_w +
        src_indices[2]/block_w;
    dst_indices[1] = src_indices[1] % block_h;
    dst_indices[2] = src_indices[2] % block_w;
    dst_indices[3] = src_indices[3];
  }
  static constexpr bool NEED_CHECK = false;
};

enum Im2BlockAssignType {
  ASSIGN_TOP,
  ASSIGN_BOTTOM
};

template <int Im2BlockAssignType, typename MappingType,
  typename ShapeDtype, typename Dtype>
HBOT_XINLINE void Im2BlockAssign(const ShapeDtype* bottom_shape,
    const ShapeDtype* top_shape, Dtype* bottom_data, Dtype* top_data) {

  ShapeDtype bottom_count = bottom_shape[0] * bottom_shape[1] *
      bottom_shape[2] * bottom_shape[3];
  ShapeDtype top_count = top_shape[0] * top_shape[1] * top_shape[2] * top_shape[3];
  ShapeDtype block_num_h = bottom_shape[1]/top_shape[1];
  ShapeDtype block_num_w = bottom_shape[2]/top_shape[2];

  ShapeDtype bottom_indices[4];
  ShapeDtype top_indices[4];
  for (ShapeDtype i = 0; i < bottom_count; ++i) {
    Offset2Indices<4>(i, bottom_shape, bottom_indices);
    MappingType::Map(bottom_indices, block_num_h,
        block_num_w, top_shape[1], top_shape[2], top_indices);
    int top_offset = ShapeOffset<4>(top_shape, top_indices);
    if (MappingType::NEED_CHECK) {
      CHECK_LT(top_offset, top_count) << top_offset << ", " << top_count
          << "top_indices: [" << top_indices[0] << ", " << top_indices[1]
          << ", " << top_indices[2] << ", " << top_indices[3] << "]."
          << " i: "<< i << " bottom_indices: [" << bottom_indices[0]
          << ", " << bottom_indices[1] << ", " << bottom_indices[2]
          << ", " << bottom_indices[3] << "],  bottom_shape: ["
          << bottom_shape[0] << ", " << bottom_shape[1] << ", "
          << bottom_shape[2] << ", " << bottom_shape[3] << "].";
    }

    if (Im2BlockAssignType == ASSIGN_TOP) {
      top_data[top_offset] = bottom_data[i];
    } else {
      bottom_data[i] = top_data[top_offset];
    }
  }
}


}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_IM2BLOCK_LAYER_HPP_

/*
 * img_pyramid.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_IMG_PYRAMID_HPP_
#define HBOT_IMG_PYRAMID_HPP_


#include <algorithm>
#include <cmath>
#include <vector>

#include "hobot_core/ndarray/ndarray.hpp"
#include "hobot_core/math/math_functions.hpp"
#include "./img_proc.hpp"

namespace hbot {

static inline void scale2logscale(const std::vector<float>& src_scale,
    std::vector<float>& dst_scale) {
  dst_scale.resize(src_scale.size());
  for (size_t i = 0; i < src_scale.size(); ++i) {
    dst_scale[i] = std::log10(src_scale[i])/std::log10(2);
  }
}

static inline void logscale2scale(const std::vector<float>& src_scale,
    std::vector<float>& dst_scale) {
  dst_scale.resize(src_scale.size());
  for (size_t i = 0; i < src_scale.size(); ++i) {
    dst_scale[i] = std::pow(2, src_scale[i]);
  }
}



template <typename Dtype>
class ImgPyramid {
 public:
  ImgPyramid(int channel = 3, int shape_stride = 16)
   :ori_img_h_(0), ori_img_w_(0), ori_img_c_(channel) {
    shape_stride_ = shape_stride;
  }
  ~ImgPyramid() {}

  inline int GetDataSize() { return scales_.size(); }
  inline NDArray<Dtype>* GetData(int idx) {
    return &(pyramid_data_[idx]);
  }

  inline void GetDataScaleAndPadh(int idx, float* scale, int* pad_h) {
    *scale = this->scales_[idx]; *pad_h = this->pad_h_[idx];
  }

  inline void BuildFrom(const std::vector<float>& scales, const Dtype* src_data,
      const int src_h, const int src_w, const int src_c) {
    ori_img_w_ = src_w; ori_img_h_ = src_h; ori_img_c_ = src_c;
    scales_ = scales;
    std::sort(scales_.begin(), scales_.end(), std::greater<float>());
    std::vector<int> out_h, out_w;
    this->FindScale(scales_, scales_, out_h, out_w, pad_h_);
    this->ResizePyramidData(out_h, out_w, pad_h_);

    int nearest_scale_id = FindNearestScaleId(1);
    this->ResizeTo(src_data, src_h, src_w, src_c, out_h[nearest_scale_id],
        out_w[nearest_scale_id], nearest_scale_id);
    for (int i = nearest_scale_id; i > 0; --i) {
      this->ResizeTo(pyramid_data_[i].cpu_data(), out_h[i], out_w[i], src_c,
          out_h[i-1], out_w[i-1], i-1);
    }
    for (int i = nearest_scale_id + 1; i < scales_.size(); ++i) {
      this->ResizeTo(pyramid_data_[i-1].cpu_data(), out_h[i-1], out_w[i-1],
          src_c, out_h[i], out_w[i], i);
    }
  }

 protected:
  inline int FindNearestScaleId(float scale) {
    int ret = -1; float abs_scale_diff = 1e9;
    for (size_t i = 0; i < scales_.size(); ++i) {
      float cur_scale_diff = std::abs(scale - scales_[i]);
      if (cur_scale_diff < abs_scale_diff) {
        ret = i; abs_scale_diff = cur_scale_diff;
      }
    }
    return ret;
  }

  inline void ResizeTo(const Dtype* src_data, const int src_h, const int src_w,
      const int src_c, const int dst_h, const int dst_w, const int scale_id) {
    NDArray<Dtype>& cur_array = pyramid_data_[scale_id];
    CHECK_EQ(dst_w, int(cur_array.width()));
    CHECK_EQ(src_c, int(cur_array.channels()));
    img_bilinear_resize(const_cast<Dtype*>(src_data), src_h,
        src_w, cur_array.mutable_cpu_data(), dst_h, dst_w, src_c);
    if (pad_h_[scale_id] > 0) {
      hbot::hbot_set(pad_h_[scale_id] * dst_w * src_c, Dtype(0),
          cur_array.mutable_cpu_data() + cur_array.offset(0, dst_h, 0, 0));
    }
  }

  inline void ResizePyramidData(const std::vector<int>& out_h,
      const std::vector<int>& out_w, const std::vector<int>& out_pad_h) {
    pyramid_data_.resize(scales_.size());
    for (size_t i = 0; i < pyramid_data_.size(); ++i) {
      NDArray<Dtype>& cur_array = pyramid_data_[i];
      cur_array.Reshape(1, 1, 1, (out_h[i]+ out_pad_h[i]) * out_w[i] * ori_img_c_ + 1);
      cur_array.Activate();
      cur_array.Reshape(1, out_h[i]+ out_pad_h[i], out_w[i], ori_img_c_);
    }
  }

  void FindScale(const std::vector<float>& in_scale,
      std::vector<float>& out_scale, std::vector<int>& out_h,
      std::vector<int>& out_w, std::vector<int>& out_pad_h);

  int ori_img_h_;
  int ori_img_w_;
  int ori_img_c_;
  int shape_stride_;
  std::vector<NDArray<Dtype> > pyramid_data_;
  std::vector<float> scales_;
  std::vector<int> pad_h_;
};


template <typename Dtype>
void ImgPyramid<Dtype>::FindScale(const std::vector<float>& in_scale,
    std::vector<float>& out_scale, std::vector<int>& out_h,
    std::vector<int>& out_w, std::vector<int>& out_pad_h) {
  out_scale.resize(in_scale.size());
  out_h.resize(in_scale.size());
  out_w.resize(in_scale.size());
  out_pad_h.resize(in_scale.size());
  for (size_t i = 0; i < in_scale.size(); ++i) {
    int dst_max_len =  int(ori_img_w_ * in_scale[i] +
            shape_stride_/2)/shape_stride_*shape_stride_;
    float cur_out_scale = float(dst_max_len)/ori_img_w_;
    out_scale[i] = cur_out_scale;
    out_h[i] = (ori_img_h_ * cur_out_scale + 0.5);
    out_w[i] = (ori_img_w_ * cur_out_scale + 0.5);
    out_pad_h[i] = (shape_stride_ - (out_h[i])%shape_stride_)%shape_stride_;
    int out_pad_w = (shape_stride_ - (out_w[i])%shape_stride_)%shape_stride_;
    CHECK_EQ(out_pad_w, 0) << out_pad_w;
  }
}

}  // namespace hbot




#endif /* HBOT_IMG_PYRAMID_HPP_ */

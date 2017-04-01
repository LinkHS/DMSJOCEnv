#ifndef FXNET_UTIL_IM2COL_HPP_
#define FXNET_UTIL_IM2COL_HPP_
#include "fxnet/common.hpp"


namespace hbot {
namespace fxnet {



template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col,DataStorageOrder storage_order = STORATE_ORDER_NHWC, int group = 1);



template <typename Dtype>
void col2img_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im,DataStorageOrder storage_order = STORATE_ORDER_NHWC, int group = 1);



template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const size_t* im_shape, const size_t* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col,DataStorageOrder storage_order = STORATE_ORDER_NHWC,int group = 1);


}  // namespace fxnet
}  // namespace hbot
#endif  // FXNET_UTIL_IM2COL_HPP_

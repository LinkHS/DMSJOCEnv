/*
 * cudnn.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_CUDNN_HPP_
#define FXNET_CUDNN_HPP_

//#define USE_CUDNN

#ifdef USE_CUDNN

#include <cudnn.h>
#include "hobot_core/base/base_common.hpp"
#include "fxnet/common.hpp"
#include "fxnet/proto/fxnet.pb.h"

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if(status != CUDNN_STATUS_SUCCESS){ std::cout<< " "\
      << cudnnGetErrorString(status)<<" in "<<__FILE__<<" "<<__LINE__<<std::endl; }\
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}


namespace hbot {

namespace cudnn {

//typedef hbot::PoolingParameter_PoolMethod PoolingParameter_PoolMethod;
//typedef hbot::PoolingParameter_PoolMethod_MAX PoolingParameter_PoolMethod_MAX;
//typedef hbot::PoolingParameter_PoolMethod_AVE PoolingParameter_PoolMethod_AVE;

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int h, int w, int c) {
	cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, tensorFormat,
				dataType<Dtype>::type, n, c, h, w ));
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int h, int w, int c) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type, CUDNN_TENSOR_NHWC,
      n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  UNUSED(bottom); UNUSED(filter);
  const int convDims = 2;
  int padA[convDims] = {pad_h,pad_w};
  int filterStrideA[convDims] = {stride_h,stride_w};
  int upscaleA[convDims] = {1,1};
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(*conv,
			convDims,
			padA,
			filterStrideA,
			upscaleA,
			CUDNN_CROSS_CORRELATION,
			dataType<Dtype>::type) );
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    std::cout << "Unknown pooling method."<<std::endl, std::abort();
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));

//  const int poolDims = 2;
//  int windowDimA[poolDims] = {h,w};
//  int paddingA[poolDims] = {pad_h,pad_w};
//  int strideA[poolDims] = {stride_h,stride_w};
//
//  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc, *mode, poolDims,
//  		windowDimA, paddingA, strideA));
}

}  // namespace cudnn

}  // namespace hbot
#endif
#endif /* FXNET_CUDNN_HPP_ */

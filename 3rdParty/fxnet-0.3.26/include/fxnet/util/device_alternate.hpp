#ifndef FXNET_UTIL_DEVICE_ALTERNATE_H_
#define FXNET_UTIL_DEVICE_ALTERNATE_H_



//#define CPU_ONLY
//#define USE_CUDNN

/**
 * @TODO CUDA , OpenCL, OpenGL support
 */

#include "hobot_core/base/base_device_alternate.hpp"

#ifdef CPU_ONLY

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { UNUSED(bottom); UNUSED(top);NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { UNUSED(bottom); UNUSED(top); NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { UNUSED(bottom); UNUSED(top); NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { UNUSED(bottom); UNUSED(top); NO_GPU; } \



#else

#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "fxnet/util/cudnn.hpp"
#endif


#endif


#endif  // FXNET_UTIL_DEVICE_ALTERNATE_H_

/*
 * layer_proto_helper.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_LAYER_PROTO_HELPER_HPP_
#define FXNET_LAYER_PROTO_HELPER_HPP_

//#define PROTOBUF_FULL

#ifdef PROTOBUF_FULL

#include <algorithm>
#include <string>
#include <vector>
#include "fxnet/proto/fxnet.pb.h"
#include "fxnet/util/io.hpp"
#include "hobot_core/base/logging.hpp"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "google/protobuf/message.h"

namespace hbot {
namespace fxnet {

inline std::string Input(NetParameter* net_param,
    const std::string input_name,
    const std::vector<int> input_shape = {1, 128, 128, 3}) {
  CHECK_EQ(input_shape.size(), 4);
  net_param->add_input(input_name);
  net_param->add_input_dim(input_shape[0]);
  net_param->add_input_dim(input_shape[1]);
  net_param->add_input_dim(input_shape[2]);
  net_param->add_input_dim(input_shape[3]);
  return input_name;
}

struct Im2BlockType{
  static constexpr const char* const type_name = "Im2Block";
};

struct Block2ImType{
  static constexpr const char* const type_name = "Block2Im";
};

template<typename Type>
inline std::string Im2BlockRelatedMaker(NetParameter* net_param,
    const std::string bottom, const std::string top,
    const int block_num_h, const int block_num_w) {
  CHECK_GE(block_num_h, 1);
  CHECK_GE(block_num_w, 1);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type(Type::type_name);
  layer->add_bottom(bottom);
  layer->add_top(top);
  Im2BlockParameter* param = layer->mutable_im2block_param();
  param->set_block_num_h(block_num_h);
  param->set_block_num_w(block_num_w);
  return top;
}

inline std::string Im2Block(NetParameter* net_param,
    const std::string bottom, const std::string top,
    const int block_num_h, const int block_num_w) {
  return Im2BlockRelatedMaker<Im2BlockType>(net_param,
      bottom, top, block_num_h, block_num_w);
}

inline std::string Block2Im(NetParameter* net_param,
    const std::string bottom, const std::string top,
    const int block_num_h, const int block_num_w) {
  return Im2BlockRelatedMaker<Block2ImType>(net_param,
      bottom, top, block_num_h, block_num_w);
}


inline std::string QuantizedWeightConvolution(NetParameter* net_param,
    const std::string bottom, const std::string top,
    const int kernel_num, const std::vector<int> kernel_hw,
    const std::vector<int> stride_hw = {1},
    const std::vector<int> pad_hw = {0}) {
  CHECK_NE(bottom , top);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("QuantizedWeightConvolution");
  layer->add_bottom(bottom);
  layer->add_top(top);
  ParamSpec* param_spec= layer->add_param();
  param_spec->set_lr_mult(1);
  param_spec->set_decay_mult(1);
  param_spec= layer->add_param();
  param_spec->set_lr_mult(2);
  param_spec->set_decay_mult(0);
  ConvolutionParameter* conv_param = layer->mutable_convolution_param();
  conv_param->set_num_output(kernel_num);

  // set kernel
  CHECK_GE(kernel_hw.size(), 1);
  if (kernel_hw.size() == 1) {
    conv_param->add_kernel_size(kernel_hw[0]);
  } else {
    conv_param->set_kernel_h(kernel_hw[0]);
    conv_param->set_kernel_w(kernel_hw[1]);
  }

  // set pad:
  CHECK_GE(pad_hw.size(), 1);
  if (pad_hw.size() == 1) {
    conv_param->add_pad(pad_hw[0]);
  } else {
    conv_param->set_pad_h(pad_hw[0]);
    conv_param->set_pad_w(pad_hw[1]);
  }

  // set stride
  CHECK_GE(stride_hw.size(), 1);
  if (stride_hw.size() == 1) {
    conv_param->add_stride(stride_hw[0]);
  } else {
    conv_param->set_stride_h(stride_hw[0]);
    conv_param->set_stride_w(stride_hw[1]);
  }

  conv_param->mutable_weight_filler()->set_type("xavier");
  conv_param->mutable_bias_filler()->set_type("constant");
  conv_param->mutable_bias_filler()->set_value(0);
  conv_param->set_valid_weight_signed_int_bit_num(8);

  return top;
}

inline std::string QuantizationReLU(NetParameter* net_param,
    const std::string bottom,
    const std::string top, const int bit_num = 8) {
  CHECK_NE(bottom , top);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("QuantizationReLU");
  layer->add_bottom(bottom);
  layer->add_top(top);
  QuantizationParameter* quant_param = layer->mutable_quantization_param();
  quant_param->set_valid_input_signed_int_bit_num(bit_num);
  ReLUParameter* relu_param = layer->mutable_relu_param();
  relu_param->set_negative_slope(0);
  return top;
}

inline std::string Quantization(NetParameter* net_param,
    const std::string bottom,
    const std::string top, const int bit_num = 8,
    const bool use_given_scale_factor = false,
    const int given_scaling_bit = 7) {
  CHECK_NE(bottom , top);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("Quantization");
  layer->add_bottom(bottom);
  layer->add_top(top);
  QuantizationParameter* quant_param = layer->mutable_quantization_param();
  quant_param->set_valid_input_signed_int_bit_num(bit_num);
  if (use_given_scale_factor) {
    quant_param->set_use_given_scaling_factor(true);
    quant_param->set_given_scaling_factor(given_scaling_bit);
  }
  return top;
}

inline std::string Eltwise(NetParameter* net_param,
    const std::string bottom1, const std::string bottom2,
    const std::string top) {
  CHECK_NE(bottom1 , top);
  CHECK_NE(bottom2 , top);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("Eltwise");
  layer->add_bottom(bottom1);
  layer->add_bottom(bottom2);
  layer->add_top(top);
  return top;
}

inline std::string QuantizationUpscale (NetParameter* net_param,
    const std::string bottom,
    const std::string top, const int bit_num = 8) {
  CHECK_NE(bottom , top);
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("QuantizationUpscale");
  layer->add_bottom(bottom);
  layer->add_top(top);
  QuantizationParameter* quant_param = layer->mutable_quantization_param();
  quant_param->set_valid_input_signed_int_bit_num(bit_num);
  return top;
}

inline std::string QuantizationPooling(NetParameter* net_param,
    const std::string bottom, const std::string top,
    const std::string pool_type,
    const std::vector<int> kernel_hw,
    const std::vector<int> stride_hw = {1},
    const std::vector<int> pad_hw = {0}) {
  CHECK_NE(bottom , top);
  CHECK(pool_type == "ave" || pool_type == "max");
  LayerParameter* layer = net_param->add_layer();
  layer->set_name(top);
  layer->set_type("QuantizationPooling");
  layer->add_bottom(bottom);
  layer->add_top(top);
  PoolingParameter* pool_param = layer->mutable_pooling_param();
  if (pool_type == "ave") {
    pool_param->set_pool(PoolingParameter::AVE);
  } else if (pool_type == "max") {
    pool_param->set_pool(PoolingParameter::MAX);
  }

  // set kernel
  CHECK_GE(kernel_hw.size(), 1);
  if (kernel_hw.size() == 1) {
    pool_param->set_kernel_size(kernel_hw[0]);
  } else {
    pool_param->set_kernel_h(kernel_hw[0]);
    pool_param->set_kernel_w(kernel_hw[1]);
  }

  // set pad:
  CHECK_GE(pad_hw.size(), 1);
  if (pad_hw.size() == 1) {
    pool_param->set_pad(pad_hw[0]);
  } else {
    pool_param->set_pad_h(pad_hw[0]);
    pool_param->set_pad_w(pad_hw[1]);
  }

  // set stride
  CHECK_GE(stride_hw.size(), 1);
  if (stride_hw.size() == 1) {
    pool_param->set_stride(stride_hw[0]);
  } else {
    pool_param->set_stride_h(stride_hw[0]);
    pool_param->set_stride_w(stride_hw[1]);
  }
  return top;
}

inline std::string EltwiseConv(NetParameter* net_param,
    const std::string bottom_conv,
    const std::string bottom_eltwise,
    const std::string top,
    const int kernel_num, const std::vector<int> kernel_hw,
    const std::vector<int> stride_hw = {1},
    const std::vector<int> pad_hw = {0},
    const bool relu_after_eltwise = true,
    const bool quantize_after_eltwise = true,
    const int quantize_bit_num = 8 ) {

  std::string conv_out = QuantizedWeightConvolution(net_param,
      bottom_conv, bottom_conv + "_" + top + "_conv", kernel_num,
      kernel_hw, stride_hw , pad_hw);

  std::string ret = conv_out;

  if (quantize_after_eltwise) {
    std::string eltwise = Eltwise(net_param, conv_out,
        bottom_eltwise, bottom_conv + "_" + top + "_eltwise");
    if (relu_after_eltwise) {
      ret = QuantizationReLU(net_param, eltwise, top, quantize_bit_num);
    } else {
      ret = Quantization(net_param, eltwise, top, quantize_bit_num);
    }
  } else {
    ret = Eltwise(net_param, conv_out, bottom_eltwise, top);
  }

  return ret;
}


inline std::string GeneralConv(NetParameter* net_param,
    const std::string bottom,
    const std::string top,
    const int kernel_num, const std::vector<int> kernel_hw,
    const std::vector<int> stride_hw = {1},
    const std::vector<int> pad_hw = {0},
    const bool need_relu = true,
    const bool need_quantization = true,
    const int quantize_bit_num = 8 ) {
  std::string ret = top;
  if (need_quantization) {
    std::string conv_out = QuantizedWeightConvolution(net_param,
        bottom, bottom + "_" + top + "_conv", kernel_num,
        kernel_hw, stride_hw , pad_hw);
    if (need_relu) {
      ret = QuantizationReLU(net_param, conv_out, top, quantize_bit_num);
    } else {
      ret = Quantization(net_param, conv_out, top, quantize_bit_num);
    }
  } else {
    ret = QuantizedWeightConvolution(net_param,
        bottom, top, kernel_num, kernel_hw, stride_hw , pad_hw);
  }
  return ret;
}

inline std::string SerializeAsString(NetParameter* net_param) {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(*net_param, &ret);
  return ret;
}

}  // namespace fxnet
}  // namespace hbot
#else
#pragma message("PROTOBUF_FULL is not activated. "                    \
                "Please enable PROTOBUF_FULL in Makefile.config" )
#endif

#endif /* FXNET_LAYER_PROTO_HELPER_HPP_ */

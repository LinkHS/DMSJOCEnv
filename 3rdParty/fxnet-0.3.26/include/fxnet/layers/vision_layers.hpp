#ifndef FXNET_VISION_LAYERS_HPP_
#define FXNET_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/layers/common_layers.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/layers/neuron_layers.hpp"
#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : virtual public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  inline bool is_1x1(){ return is_1x1_;}
  inline bool is_fc(){ return is_fc_; }
  inline const Blob<int>& pad(){return pad_;}
  inline const Blob<int>& stride(){return stride_;}

 protected:
  inline __attribute__((always_inline)) void im2col_block(Dtype* buff_data, const Dtype* data_im, int pacth_spatial_id_start, int patch_spatial_id_end,
  		const int& height, const int& width, const int& channels,const int& width_col, const int& stride_h,const int& stride_w,
  		const int& pad_h, const int& pad_w, const int& kernel_dim, const int& kernel_h, const int& kernel_w){
  	for(int out_spatial_id = pacth_spatial_id_start; out_spatial_id < patch_spatial_id_end; ++out_spatial_id){
  		int inner_block_id = out_spatial_id - pacth_spatial_id_start;
			int h_col = out_spatial_id/width_col;
			int w_col = out_spatial_id%width_col;
			const int w_im_off = w_col * stride_w - pad_w;
			int ignore_w = std::max(0-w_im_off,0);
			int end_w = std::min(width,w_im_off + kernel_w);
			int w_im = std::min(w_im_off + ignore_w, width);
			memset(buff_data + inner_block_id * kernel_dim,0,sizeof(Dtype) *kernel_dim );
			int h_im = h_col * stride_h - pad_h + 0;
			for(int h_offset = 0; h_offset < kernel_h; ++h_offset,++h_im){
				if(h_im >= 0 && h_im < height){
					memcpy(buff_data + inner_block_id * kernel_dim+ (ignore_w + h_offset *kernel_w)*channels,
							data_im + (h_im * width + w_im )*channels ,channels*(end_w - w_im)*sizeof(Dtype));
				}
			}
  	}
  }

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);

  void forward_cpu_gemm_1x1_or_fc_all(const Dtype* input, const Dtype* weights,
        Dtype* output, bool skip_im2col = false);

  void forward_cpu_gemm_int8_t_to_int32_t(const int8_t* input, const int8_t* weights,
      int32_t* output, bool skip_im2col = false,bool is_weight_trans = false);

  void forward_cpu_gemm_int8_t_to_int32_t_1x1_or_fc_all(const int8_t* input, const int8_t* weights,
        int32_t* output, bool skip_im2col = false,bool is_weight_trans = false);


  void forward_cpu_bias_all(Dtype* output, const Dtype* bias);

  void forward_cpu_bias(Dtype* output, const Dtype* bias);

//  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);

  void backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);

  inline void set_input_beta(Dtype value){
  	input_beta_ = value;
  }

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[1 + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<size_t> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<size_t> output_shape_;
  const vector<size_t>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

//  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

  bool is_fc_;

  bool zero_pad_;

  Blob<Dtype> output_buffer_; // used when group != 1
  Blob<Dtype> input_buffer_; // used when group != 1
  int max_input_count_; // used when group != 1


  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
//  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff, const int conv_in_channels,
//  		DataStorageOrder storage_order = STORATE_ORDER_NHWC, const int group = 1) {
//  	if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
//  		im2col_cpu(data, conv_in_channels,
//          conv_input_shape_.cpu_data()[0], conv_input_shape_.cpu_data()[1],
//          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
//          pad_.cpu_data()[0], pad_.cpu_data()[1],
//          stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff,storage_order,  group);
//  	}else{
//  		im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
//  		          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
//  		          pad_.cpu_data(), stride_.cpu_data(), col_buff,storage_order, group);
//  	}
//  }

  void group_output_to_output_Dtype(const Dtype* src, Dtype* output, const int output_spatial_dim,
  		 const int output_channel,const int group);
  void output_to_group_output_Dtype(const Dtype* src, Dtype* output, const int output_spatial_dim,
 		 const int output_channel,const int group);

  void group_output_to_output_int32_t(const int32_t* src, int32_t* output, const int output_spatial_dim,
  		 const int output_channel,const int group);
  void output_to_group_output_int32_t(const int32_t* src, int32_t* output, const int output_spatial_dim,
 		 const int output_channel,const int group);
  void group_output_to_output_int8_t(const int8_t* src, int8_t* output, const int output_spatial_dim,
    		 const int output_channel,const int group);
	void output_to_group_output_int8_t(const int8_t* src, int8_t* output, const int output_spatial_dim,
		 const int output_channel,const int group);

  void naive_conv_cpu(const Dtype* data_im, const int channels,
      const int height, const int width,const Dtype* kernel_data,
      const int kernel_n, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w, const int stride_h,
      const int stride_w,  Dtype* data_out, Dtype* buff_data,
      DataStorageOrder storage_order = STORATE_ORDER_NHWC, int group = 1);

  void naive_back_conv_cpu(const Dtype* data_out, const int channels,
      const int height, const int width,const Dtype* kernel_data,
      const int kernel_n, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w, const int stride_h,
      const int stride_w,  Dtype* data_im, Dtype* buff_data,
      DataStorageOrder storage_order = STORATE_ORDER_NHWC, int group = 1);


  void naive_conv_cpu_int8_t_to_int32_t(const int8_t* data_im, const int channels,
      const int height, const int width,const int8_t* kernel_data,
      const int kernel_n, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w, const int stride_h,
      const int stride_w,  int32_t* data_out, int8_t* buff_data,
      DataStorageOrder storage_order = STORATE_ORDER_NHWC, int group = 1,
      bool is_weight_trans = false);





  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  int buff_size_ ;




  // used for int8_t forward
  Blob<int8_t> input_buffer_int8_t_; // used when group != 1
  Blob<int8_t> col_buffer_int8_t_; // used when group != 1
  Blob<int32_t> output_buffer_int32_t_; // used when group != 1
  Dtype input_beta_;


};

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has FXNET (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),BaseConvolutionLayer<Dtype>(param) {
  	int8_t_to_int32_t_forward_ = false;
  	weight_alpha_ = 0;
  }
  
  virtual inline const char* type() const { return "Convolution"; }

  inline void set_int8_t_to_int32_t_forward(bool flag){
  	int8_t_to_int32_t_forward_ = flag;
  }
  inline bool int8_t_to_int32_t_forward(){
  	return int8_t_to_int32_t_forward_;
  }
//  LAYER_CREATOR(Convolution, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  void Forward_cpu_float(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_cpu_int8_t(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  void Forward_cpu_int8_t_v2(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  void TransWeight();
  /**
   * for  int8_t_to_int32_t_forward
   */

  Blob<int8_t> weight_int8_t_;
  Blob<int8_t> input_int8_t_;
  Blob<int32_t> output_int32_t_;
  Blob<Dtype> bias_bar_;
  Blob<Dtype> weight_temp_;
  Blob<Dtype> weight_trans_;



  bool int8_t_to_int32_t_forward_;
  Dtype weight_alpha_;

 private:
  void weight_to_int8_t();


};

//SET_LAYER_REGISTER_FLAG(Convolution);



/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class PoolingLayer : virtual public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
//  virtual inline int MaxTopBlobs() const {
//    return (this->layer_param_.pooling_param().pool() ==
//            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
//  }
  virtual inline int MaxTopBlobs() const {
    return 1;
  }
//  LAYER_CREATOR(Pooling, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // naive implementation of max pooling
  void Forward_MAX_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // naive implementation of ave pooling
  void Forward_AVE_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
//  Blob<Dtype> rand_idx_;
//  Blob<int> max_idx_;
};

//SET_LAYER_REGISTER_FLAG(Pooling);

/**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Im2col"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  LAYER_CREATOR(Im2col, Dtype);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

//  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  bool force_nd_im2col_;
};
SET_LAYER_REGISTER_FLAG(Im2col);

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_VISION_LAYERS_HPP_

/*
 *      Author: Alan_Huang
 */

#ifndef FXNET_PROCESS_HPP_
#define FXNET_PROCESS_HPP_

#include "fxnet/tools/io_process.hpp"
#include "fxnet/tools/fxnet_core.hpp"
#include <vector>
#include <iostream>
#include <limits.h>


#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

namespace hbot {
namespace fxnet {
typedef unsigned char uchar;

template <typename Dtype >
bool convert_detection_output(const Tensor<Dtype>& output_tensor, const void* additional_param,
		std::vector<Landmarks> &result, const int heatmap_a, const int heatmap_b,
		const Dtype threshold = 0, const int landmark_channel_start = 0);

template <typename Dtype >
bool convert_detection_output(const Tensor<Dtype>& output_tensor, const void* additional_param,
		std::vector<BBoxLandmarks> &result, const int heatmap_a, const int heatmap_b,
		const Dtype threshold = 0, const int landmark_channel_start = 0);


/**
 * @brief Load image to blob. Image should be gray or BGR color format.
 */
template <typename Dtype>
bool unscale_crop_image_to_tensor(Tensor<Dtype>& tensor,const int crop_w, const int crop_h,
		const int crop_start_x, const int crop_start_y, const int img_w, const int img_h, const int channels,
		const uchar* image, const int num_id, const Dtype input_mean = 0, const Dtype input_scale_factor = 1,
		const Dtype* mean_bgr = nullptr);

template<typename T>
inline bool get_ori_img_locations(std::vector<T >& inst_vector, std::vector<PatchInfo>& patches_info,
		int start_idx = 0){
	for(int i=start_idx; i < inst_vector.size(); ++i){
		T &cur_inst = inst_vector[i];
		PatchInfo & cur_patch_info = patches_info[ cur_inst.num_id];
		cur_inst.ToOriLocation(cur_patch_info);
	}
	return true;
}


void image_crop(const uchar* src, const int patch_start_x,const int patch_start_y,
		const int src_height,const int src_width, const int patch_w, const int patch_h,
		const int channel, uchar* dst, const int dst_height, const int dst_width);


/**
 * @brief Load image to blob. Use addition two cv::Mat as buffers to process image.
 */
template <typename Dtype>
bool crop_img_to_tensor(Tensor<Dtype>& tensor, const uchar* image, const int img_h, const int img_w,
		const int img_channel,void* temp_patch_blob_tensor, void* resized_patch_blob_tensor,const BBox& candidate_bbox,
		PatchInfo& patch_info,Dtype bbox_scale_height_factor, Dtype bbox_scale_width_factor,const int num_id,
		const Dtype input_mean  , const Dtype input_scale_factor,const int anticlockwise_rotate_angle = 0,
		const Dtype* mean_bgr = nullptr);

#ifdef USE_OPENCV

inline void adaptive_cvmat_crop(cv::Mat & src, cv::Mat & dst, const int patch_start_x, const int patch_start_y,
		const int patch_w, const int patch_h){
	int actual_patch_start_x = std::min(std::max(0,patch_start_x),src.cols);
	int actual_patch_start_y = std::min(std::max(0,patch_start_y),src.rows);
	int actual_patch_end_x = std::min(std::max(0,patch_start_x + patch_w),src.cols);
	int actual_patch_end_y = std::min(std::max(0,patch_start_y + patch_h),src.rows);
	cv::Rect crop_rect(actual_patch_start_x - patch_start_x, actual_patch_start_y - patch_start_y,
			actual_patch_end_x- actual_patch_start_x , actual_patch_end_y- actual_patch_start_y);

	src(cv::Rect(actual_patch_start_x, actual_patch_start_y, actual_patch_end_x- actual_patch_start_x ,
			actual_patch_end_y- actual_patch_start_y)).copyTo(dst(crop_rect));

}


template <typename Dtype>
bool TensorDataToCVMat(const Tensor<Dtype>& data, cv::Mat & mat, int sampleID, const Dtype img_mean, const Dtype img_scale);

/**
 * @brief Load image to blob. Use addition two cv::Mat as buffers to process image.
 */
template <typename Dtype>
bool opencv_crop_img_to_tensor(Tensor<Dtype>& tensor, cv::Mat & image, void* temp_patch_blob_tensor,
		void* resized_patch_blob_tensor,const BBox& candidate_bbox,PatchInfo& patch_info,Dtype bbox_scale_height_factor,
		Dtype bbox_scale_width_factor,const int num_id, const Dtype input_mean  , const Dtype input_scale_factor,
		const int anticlockwise_rotate_angle = 0, const Dtype* mean_bgr=nullptr);

#endif

}  // namesapce fxnet
}  // namespace hbot

#endif /* FXNET_PROCESS_HPP_ */

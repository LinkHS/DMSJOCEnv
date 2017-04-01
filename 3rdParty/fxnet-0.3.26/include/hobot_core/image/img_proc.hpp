/*
 * img_proc.hpp
 *      Author: Alan_Huang
 */

#ifndef HBOT_IMG_PROC_HPP_
#define HBOT_IMG_PROC_HPP_

#include "hobot_core/ndarray/ndarray.hpp"
#include "hobot_core/math/math_functions.hpp"
#include <stdlib.h>

namespace hbot {

//typedef unsigned char uint8_t;

template<typename Dtype>
inline void img_crop(const Dtype* src_data,const int patch_start_x, const int patch_start_y, const int src_height,
		const int src_width, const int patch_w, const int patch_h, const int channel, Dtype* dst_data,
		const int dst_height, const int dst_width){
	UNUSED(dst_height);
	int actrual_patch_start_x = std::min(std::max(0,patch_start_x),src_width);
	int actrual_patch_start_y = std::min(std::max(0,patch_start_y),src_height);
	int actrual_patch_end_x = std::min(std::max(0,patch_start_x + patch_w),src_width);
	int actrual_patch_end_y = std::min(std::max(0,patch_start_y + patch_h),src_height);

	int actrual_dst_y_start = actrual_patch_start_y - patch_start_y;
	int actrual_dst_x_start = actrual_patch_start_x - patch_start_x;

	for(int y=actrual_patch_start_y; y < actrual_patch_end_y; ++y ){
		int dim = (actrual_patch_end_x - actrual_patch_start_x)* channel;
		memcpy(dst_data + ((actrual_dst_y_start + y-actrual_patch_start_y )*dst_width + actrual_dst_x_start)*channel,
				src_data + (y * src_width + actrual_patch_start_x)*channel, sizeof(Dtype) * dim);
	}

}


/*
 * [src_p11, src_p21]    [wX = projected_x - loc_x_src_p11]
 * [src_p12, src_p22]    [wY = projected_y - loc_y_src_p11]
 */
template<int Dim>
inline void point_interpolate_2x2(const uint8_t* src_p11, const uint8_t* src_p21,
		const uint8_t* src_p12, const uint8_t* src_p22, uint8_t* dst_p, int wX, int wY ){
  int f24 = (wX * wY) >> 8; int f23 = wX - f24; int f14 = wY - f24;
  int f13 = ((256 - wX) * (256 - wY)) >> 8; // this one can be computed faster
  if( Dim == 1){
  	*dst_p =  ( *(src_p11) * f13 + *(src_p21) * f23 + *(src_p12) * f14 + *(src_p22) * f24)>>8;
  }else if(Dim == 3){
    *reinterpret_cast<int*>(dst_p) = (((( *reinterpret_cast<const int*>(src_p11)&0xFF00FF)*f13 +
    		(  *reinterpret_cast<const int*>(src_p21)&0xFF00FF)*f23+
    		( *reinterpret_cast<const int*>(src_p12)&0xFF00FF)*f14+
    		( *reinterpret_cast<const int*>(src_p22)&0xFF00FF)*f24)&0xFF00FF00)|
        (((*reinterpret_cast<const int*>(src_p11)&0x00FF00)*f13+
  			(*reinterpret_cast<const int*>(src_p21)&0x00FF00)*f23+
  			(*reinterpret_cast<const int*>(src_p12)&0x00FF00)*f14+
  			(*reinterpret_cast<const int*>(src_p22)&0x00FF00)*f24)&0x00FF0000))>>8;
  }else{
  	std::cout<<"channel == "<<Dim<<" is not supported in point_interpolate_4x4";
  	std::abort();
  }
}


template<int Dim>
inline void shrink_2_point( uint8_t* dst_pt,uint8_t* pt1, uint8_t* pt2){
	if(Dim == 3){
		*reinterpret_cast<int*>(dst_pt) = ( (( (( *reinterpret_cast<const int*>(pt1)&0xFF00FF)>>1) +
				(( *reinterpret_cast<const int*>(pt2)&0xFF00FF)>>1))&0xFF00FF)|(
				((( *reinterpret_cast<const int*>(pt1)&0x00FF00) >>1) +
				(( *reinterpret_cast<const int*>(pt2)&0x00FF00) >>1)
				)&0x00FF00 ));
	}else if(Dim == 1){
		*(dst_pt  ) = ((*pt1)>> 1) + ((*(pt2))>> 1);
  }else{
  	std::cout<<"channel == "<<Dim<<" is not supported in point_interpolate_4x4";
  	std::abort();
  }
}

template<int Dim>
inline void shrink_8_point(uint8_t* buff_ptr1, int channels, uint8_t*& line1_ptr){
	if(Dim == 3){
		shrink_2_point<Dim>(buff_ptr1, line1_ptr, line1_ptr+channels);
		line1_ptr += channels*2;
		buff_ptr1 += channels;
		shrink_2_point<Dim>(buff_ptr1, line1_ptr, line1_ptr+channels);
		line1_ptr += channels*2;
		buff_ptr1 += channels;
		shrink_2_point<Dim>(buff_ptr1, line1_ptr, line1_ptr+channels);
		line1_ptr += channels*2;
		buff_ptr1 += channels;
		shrink_2_point<Dim>(buff_ptr1, line1_ptr, line1_ptr+channels);
		line1_ptr += channels*2;
	}else if(Dim == 1){
		*(buff_ptr1 ++) = ((*line1_ptr)>> 1) + ((*(line1_ptr + channels))>> 1);
		line1_ptr += channels*2;
		*(buff_ptr1 ++) = ((*line1_ptr)>> 1) + ((*(line1_ptr + channels))>> 1);
		line1_ptr += channels*2;
		*(buff_ptr1 ++) = ((*line1_ptr)>> 1) + ((*(line1_ptr + channels))>> 1);
		line1_ptr += channels*2;
		*(buff_ptr1 ++) = ((*line1_ptr)>> 1) + ((*(line1_ptr + channels))>> 1);
		line1_ptr += channels*2;
  }else{
  	std::cout<<"channel == "<<Dim<<" is not supported in point_interpolate_4x4";
  	std::abort();
  }
}

template<int Dim>
inline void point_shrink_2x4(uint8_t*& dst_data,int channels, uint8_t* buff_ptr1, uint8_t* buff_ptr2){
	if(Dim == 3){
		shrink_2_point<Dim>(dst_data, buff_ptr1, buff_ptr2);
		buff_ptr1 += channels; buff_ptr2 += channels; dst_data += channels;
		shrink_2_point<Dim>(dst_data, buff_ptr1, buff_ptr2);
		buff_ptr1 += channels; buff_ptr2 += channels; dst_data += channels;
		shrink_2_point<Dim>(dst_data, buff_ptr1, buff_ptr2);
		buff_ptr1 += channels; buff_ptr2 += channels; dst_data += channels;
		shrink_2_point<Dim>(dst_data, buff_ptr1, buff_ptr2);
		dst_data += channels;
	}else if(Dim == 1){
		*(dst_data ++) = ((*buff_ptr1)>> 1) + ((*(buff_ptr2))>> 1);
		buff_ptr1 += channels; buff_ptr2 += channels;
		*(dst_data ++) = ((*buff_ptr1)>> 1) + ((*(buff_ptr2))>> 1);
		buff_ptr1 += channels; buff_ptr2 += channels;
		*(dst_data ++) = ((*buff_ptr1)>> 1) + ((*(buff_ptr2))>> 1);
		buff_ptr1 += channels; buff_ptr2 += channels;
		*(dst_data ++) = ((*buff_ptr1)>> 1) + ((*(buff_ptr2))>> 1);
		buff_ptr1 += channels; buff_ptr2 += channels;
  }else{
  	std::cout<<"channel == "<<Dim<<" is not supported in point_interpolate_4x4";
  	std::abort();
  }
}

template<int channels>
void img_shrink_uint8_t(const uint8_t* src_data, uint8_t* dst_data,
    const int src_height, const int src_width);

inline void img_shrink(const uint8_t* src_data,uint8_t* dst_data,
    int& src_height, int& src_width, int& dst_height, int& dst_width, int channels){
	if(channels ==3 ){
		img_shrink_uint8_t<3>(src_data,dst_data, src_height, src_width);
	}else if(channels == 1){
		img_shrink_uint8_t<1>(src_data,dst_data, src_height, src_width);
	}else{
		std::cout<<"image channels not supported: channels == "<<channels<<std::endl;
		std::abort();
	}
	dst_height = src_height/2 + src_height%2; dst_width = src_width/2 + src_width%2;
}


template<int channels>
void img_bilinear_resize_uint8_t(const uint8_t* src_data, const int src_height, const int src_width,
		uint8_t* dst_data, const int dst_height, const int dst_width);


inline void get_rotate_matrix(float * mat, float degree, float center_x, float center_y, float scale){
	float angle = degree * HBOT_PI / 180.f;
	float alpha = hbot::cos(angle) * scale;
	float beta = hbot::sin(angle) * scale;
//	printf("alpha: %f , beta: %f \n", alpha, beta);
	mat[0] = alpha; mat[1] = beta; mat[2] = (scale-alpha) * center_x - beta*center_y;
	mat[3] = -beta; mat[4] = alpha; mat[5] = beta*center_x + (scale-alpha)*center_y;
}

inline void get_inverse_rotate_matrix(float * dst_mat, float * src_mat = NULL ){
	if(src_mat == NULL){ src_mat = dst_mat; }
	float D = src_mat[0]*src_mat[4] - src_mat[1]*src_mat[3];
	D = (D != 0) ? 1./D : 0 ;
	float A11 = src_mat[4]*D, A22 = src_mat[0]*D;
	dst_mat[0] = A11; dst_mat[1] = src_mat[1] * -D; dst_mat[3] = src_mat[3] * -D; dst_mat[4] = A22;
	float b1 = -dst_mat[0]*src_mat[2] - dst_mat[1]*src_mat[5];
	float b2 = -dst_mat[3]*src_mat[2] - dst_mat[4]*src_mat[5];
	dst_mat[2] = b1; dst_mat[5] = b2;
}


template<int channel>
void img_bilinear_rotate_uint8_t(const uint8_t* src_data, const int src_height, const int src_width,
		uint8_t* dst_data, const int dst_height, const int dst_width, const int dst_start_x, const int dst_start_y,
		float * mat_rotate );

inline void img_bilinear_rotate(const uint8_t* src_data, const int src_height, const int src_width,
		uint8_t* dst_data, const int dst_height, const int dst_width, const int dst_start_x, const int dst_start_y,
		float * mat_rotate,const int channels = 3 ){
	if(channels ==3 ){
		img_bilinear_rotate_uint8_t<3>( src_data, src_height,   src_width, dst_data,  dst_height,
				dst_width,  dst_start_x,  dst_start_y, mat_rotate );
	}else if(channels == 1){
		img_bilinear_rotate_uint8_t<1>( src_data, src_height,   src_width, dst_data,  dst_height,
						dst_width,  dst_start_x,  dst_start_y, mat_rotate );
	}else{
		std::cout<<"image channels not supported: channels == "<<channels<<std::endl;
		std::abort();
	}
}


inline void img_bilinear_resize(uint8_t* src_data,  int src_height,  int src_width,
		uint8_t* dst_data, const int dst_height, const int dst_width, const int channels = 3,
		bool smooth = false) {

  int temp_dst_height = src_height/2 + src_height%2;
  int temp_dst_width = src_width/2 + src_width%2;
  hbot::NDArray<uint8_t> temp_space(1,temp_dst_height,temp_dst_width+1,channels);

	if(smooth){
	  uint8_t* buff = temp_space.mutable_cpu_data();
	  if (src_height > dst_height *2 && src_width > dst_width *2){
	    img_shrink(src_data, buff, src_height, src_width,
	        src_height, src_width,  channels);
	    src_data = buff;
	  }
		while(src_height > dst_height *2 && src_width > dst_width *2){
		  img_shrink(buff, buff, src_height, src_width,
		            src_height, src_width,  channels);
		}

	}
	if(channels ==3 ){
		img_bilinear_resize_uint8_t<3>( src_data,  src_height, src_width,
			 dst_data,  dst_height, dst_width);
	}else if(channels == 1){
		img_bilinear_resize_uint8_t<1>( src_data,  src_height, src_width,
			 dst_data,  dst_height, dst_width);
	}else{
		std::cout<<"image channels not supported: channels == "<<channels<<std::endl;
		std::abort();
	}
	temp_space.Recycle();
}

}  // namespace hbot

#endif /* HBOT_IMG_PROC_HPP_ */

/*
 *      Author: Alan_Huang
 */

#ifndef FXNET_IO_PROCESS_HPP_
#define FXNET_IO_PROCESS_HPP_

#include "fxnet/tools/fxnet_core.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

// #define USE_OPENCV 1
// #define SHOW_TIME 1
namespace hbot {

template <typename Dtype>
class ImgPyramid;

namespace fxnet{

struct Point2D{
	Point2D(){ x=y=0;}
	Point2D(int x_, int y_){x=x_;y=y_;}
	int x, y;
};

struct PatchInfo{
	inline PatchInfo(){
		left_top.x = left_top.y = scale_height = scale_width= 0;
	}
	inline PatchInfo(int x_, int y_){
		left_top.x = x_; left_top.y = y_; scale_height = scale_width= 0;
	}
	inline PatchInfo(int x_, int y_, float scale_height_, float scale_width_){
		left_top.x = x_; left_top.y = y_; scale_height = scale_height_; scale_width = scale_width_;
	}
	Point2D left_top;
	float scale_height, scale_width;

};

struct BBox{
	inline BBox(){ score = x1 = x2 = y1 = y2 = num_id = 0; }
	inline BBox( float score_, float x1_, float y1_, float x2_, float y2_, int num_id_){
		score = score_; x1 = x1_; y1 = y1_; x2 = x2_; y2 = y2_;
		num_id = num_id_;
	}
	inline friend std::ostream& operator << (std::ostream& out, BBox& bbox ){
		out<<"( x1: "<<bbox.x1<<" y1: "<<bbox.y1<<" x2: "<<bbox.x2<<" y2: "<<bbox.y2<<" score: "
				<<bbox.score<<" )";
		return out;
	}
	inline static bool greater(const BBox & a, const BBox & b){
			return a.score > b.score;
	}
	inline void ToOriLocation(PatchInfo & patches_info){
		x1 = patches_info.left_top.x + x1/patches_info.scale_width;
		x2 = patches_info.left_top.x + x2/patches_info.scale_width;
		y1 = patches_info.left_top.y + y1/patches_info.scale_height;
		y2 = patches_info.left_top.y + y2/patches_info.scale_height;
	}
	float score, x1, y1, x2, y2;
	int num_id;
};

struct BBoxLandmarks: public BBox{
	inline BBoxLandmarks():BBox(){}

	inline friend std::ostream& operator << (std::ostream& out, BBoxLandmarks& bbox ){
		out<<"Bbox( x1: "<<bbox.x1<<" y1: "<<bbox.y1<<" x2: "<<bbox.x2<<" y2: "<<bbox.y2<<" score: "
				<<bbox.score<<" ): Landmarks( ";
		for(int i=0; i < bbox.landmark_score.size(); ++i){
			out<< "["<<i<<"]:( score = "<<bbox.landmark_score[i]<<" ,x = "<<bbox.landmark_loc[i*2]<<
					" , y = "<<bbox.landmark_loc[i*2+1];
		}
		out<<" ). ";
		return out;
	}
	inline void ToOriLocation(PatchInfo & patches_info){
		BBox::ToOriLocation(patches_info);
		for(int i=0; i < landmark_score.size(); ++i){
			int x = landmark_loc[i*2]; int y = landmark_loc[i*2+1];
			landmark_loc[i*2] = patches_info.left_top.x + x/patches_info.scale_width;
			landmark_loc[i*2+1] = patches_info.left_top.y + y/patches_info.scale_height;
		}
	}

	std::vector<float> landmark_score;
	std::vector<float> landmark_loc;
};


struct Landmarks{
	inline Landmarks(){ num = 0; num_id = 0; pose = 0; }
	inline Landmarks(int num_){
		num = num_; score.resize(num,-1); x.resize(num,-10000); y.resize(num,-10000); num_id = 0;
    pose = 0;
	}
	inline void Push(float score_, float x_, float y_){
		score.push_back(score_); x.push_back(x_); y.push_back(y_);
	}
	inline void ToOriLocation(PatchInfo & patches_info){
		for(int i=0; i < num; ++i){
			x[i] = std::max(patches_info.left_top.x + x[i]/patches_info.scale_width,float(-1));
			y[i] = std::max(patches_info.left_top.y + y[i]/patches_info.scale_height,float(-1));
		}
	}

	inline bool IsValidFace(){
		for(int i=0; i < num; ++i){
			if(score[i] <= float(-0.9)  ){
				return false;
			}
		}
		return true;
	}
	inline void GetAnchoXY(std::vector<int>& anchor_id_list, float& out_x, float& out_y){
		int ancho_size = anchor_id_list.size();
		for(int i=0; i < ancho_size; ++i ){
			int anchor_id = anchor_id_list[i];
			out_x += x[anchor_id] ;
			out_y += y[anchor_id] ;
		}
		out_x /= ancho_size;
		out_y /= ancho_size;
	}

	int num;
	std::vector<float> score;
	std::vector<float> x;
	std::vector<float> y;
	int num_id;
  float pose;
	std::vector<int> anchor_points_idx_1,anchor_points_idx_2, anchor_center_idx;
};


struct FaceAttributes{
	inline FaceAttributes(){
		num_id = 0; attribute_score.clear();
	}
	std::vector<float> attribute_score;
	int num_id;
};


struct Scale {
  Scale(float _scale) { scale = _scale; }
  float scale;
};
//struct Scales : std::vector<float>{};


/* @brief  Non-maximum suppression. Overlap threshold for suppression
*          For a selected box Bi, all boxes Bj that are covered by
*          more than overlap are suppressed. Note that 'covered' is
*          is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over
*          union measure. This strategy keeps  small bboxes whose
*          confidence is high
*  @param  overlap_ratio
*  @param	 addscore	 if addscore == true, then the scores of all
*  									 the overlap bboxes will be added
*/

template<typename T>
void NMS(std::vector<T>& candidates, std::vector<T >& result,
		const float overlap_ratio, const int top_N, const bool addScore = false);

}  // namespace fxnet
}  // namespace hbot
#endif /* FXNET_IO_PROCESS_HPP_ */
